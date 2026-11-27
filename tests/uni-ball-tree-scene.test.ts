import { describe, expect, it } from 'vitest';
import { pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import { compute2dSinCosPosEmbed } from '../src/uni/rope';
import {
  buildUniBallTreeScene,
  UNI_BALL_TREE_ATLAS_BUDGET_BYTES,
  UNI_BALL_TREE_ATLAS_SIZE,
  UNI_BALL_TREE_PROD_DEPTH,
  UNI_BALL_TREE_REGION_COUNT,
} from '../src/uni/scene/ballTreeScene';
import {
  pickHitToUniCell,
  validateSceneGeometryFinite,
} from '../src/uni/scene/provenance';
import type {
  UniAtlasRegionBox,
  UniRefinementStepPreset,
  UniSpatialScene,
} from '../src/uni/scene/types';
import { UniSession } from '../src/uni/session';
import {
  analyzePrimeDepthFloorMismatch,
  getStagePolarityPair,
  getTreeBallMembership,
  semanticTreeLevelsForAges,
} from '../src/uni/tree';
import { UNI_MINI_CONFIG } from '../src/uni/types';

const sharedBundle = createDefaultUniRuntimeBundle();

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

function assertBallTreeAtlasContract(scene: UniSpatialScene): void {
  expect(scene.viewMode).toBe('ball_tree');
  expect(scene.layerIndex).toBeNull();
  expect(scene.rgbPreviews).toEqual([]);
  expect(scene.sheetCount).toBe(UNI_BALL_TREE_REGION_COUNT);
  expect(validateSceneGeometryFinite(scene).valid).toBe(true);

  expect(scene.valueAtlas).not.toBeNull();
  expect(scene.valueAtlas!.width).toBe(UNI_BALL_TREE_ATLAS_SIZE);
  expect(scene.valueAtlas!.height).toBe(UNI_BALL_TREE_ATLAS_SIZE);
  expect(scene.valueAtlas!.data.byteLength).toBe(UNI_BALL_TREE_ATLAS_BUDGET_BYTES);
  expect(scene.atlasKey).toMatch(/^ball_tree:128x128:/);

  const audit = scene.atlasAudit;
  expect(audit).not.toBeNull();
  expect(audit!.width).toBe(128);
  expect(audit!.height).toBe(128);
  expect(audit!.byteLength).toBe(131_072);
  expect(audit!.regions.length).toBe(5);
  expect(audit!.hasOverlap).toBe(false);
  expect(audit!.overlapPair).toBeNull();
  expect(audit!.outOfBoundsIds.length).toBe(0);
  expect(audit!.withinBudget).toBe(true);
  expect(audit!.usedTexels).toBe(128 * 8 + 128 * 8 + 16 * 16 + 16 * 64 + 24 * 8);
}

function rayAtBoxTopCenter(box: PickBox): Ray {
  const cx = (box.min[0] + box.max[0]) * 0.5;
  const cz = (box.min[2] + box.max[2]) * 0.5;
  const topY = box.max[1] + 5.0;
  return {
    origin: [cx, topY, cz],
    direction: [0, -1, 0],
  };
}

describe('UNI5 Slice 5A · View 4 (ball_tree) — Codebook Geometry & Ball-Tree Inspector', () => {
  it('builds cleanly standalone (session = null) with 5 RG32F rects, exact G-channel rules, 0 RGB previews, and finite geometry', () => {
    const scene = buildUniBallTreeScene(null);
    assertBallTreeAtlasContract(scene);

    // Verify all 5 named regions and exact dimensions
    const lo = readRegionTexels(scene, 'codebook_lo');
    const hi = readRegionTexels(scene, 'codebook_hi');
    const radii = readRegionTexels(scene, 'heap_radii');
    const membership = readRegionTexels(scene, 'ball_membership');
    const sinCos = readRegionTexels(scene, 'sincos_2d');

    expect(lo.region).toMatchObject({ cols: 128, rows: 8 });
    expect(hi.region).toMatchObject({ cols: 128, rows: 8 });
    expect(radii.region).toMatchObject({ cols: 16, rows: 16 });
    expect(membership.region).toMatchObject({ cols: 64, rows: 16 });
    expect(sinCos.region).toMatchObject({ cols: 8, rows: 24 });

    // CRITICAL SheetsPass G-channel rule:
    //   - codebook_lo, codebook_hi, sincos_2d: G = 0.0 on 100% of live cells
    //   - heap_radii[0..254]: G = 0.0, and unused padding slot heap_radii[255]: G = 1.0
    expect(Array.from(lo.gValues).every((g) => g === 0)).toBe(true);
    expect(Array.from(hi.gValues).every((g) => g === 0)).toBe(true);
    expect(Array.from(sinCos.gValues).every((g) => g === 0)).toBe(true);

    for (let h = 0; h < 255; h++) {
      expect(radii.gValues[h]).toBe(0);
      expect(radii.rValues[h]).toBeCloseTo(sharedBundle.tree.heapRadii[h]!, 6);
    }
    expect(radii.gValues[255]).toBe(1);
    expect(radii.rValues[255]).toBe(0);

    // Exact codebook values in codebook_lo (0..127) and codebook_hi (128..255)
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 128; c++) {
        expect(lo.rValues[r * 128 + c]).toBe(sharedBundle.vqvae.codebook[c * 8 + r]);
        expect(hi.rValues[r * 128 + c]).toBe(sharedBundle.vqvae.codebook[(128 + c) * 8 + r]);
      }
    }

    // Exact 2D SinCos values (4x6 marked target grid, canonicalContentSize=32, scale=0.0018, markerColumn=true)
    const expectedSinCos = compute2dSinCosPosEmbed(4, 6, 8, true, UNI_MINI_CONFIG.sinCos2d);
    expect(sinCos.rValues.length).toBe(expectedSinCos.embeddings.length);
    for (let i = 0; i < sinCos.rValues.length; i++) {
      expect(sinCos.rValues[i]).toBe(expectedSinCos.embeddings[i]);
    }

    // ball_membership row 0 (depth 0 = full 256-code root ball) has 0 masked slots;
    // row 8 (depth 8 = single target leaf) has 1 live group (G=0) and 63 excluded groups (G=1)
    let row0Live = 0;
    let row8Live = 0;
    let row8Masked = 0;
    for (let c = 0; c < 64; c++) {
      if (membership.gValues[0 * 64 + c] === 0) row0Live++;
      if (membership.gValues[8 * 64 + c] === 0) row8Live++;
      if (membership.gValues[8 * 64 + c] === 1) row8Masked++;
      // Row 9 (proximity) & Row 15 (depth-6 radii) are 100% live scalar cells (G = 0.0)
      expect(membership.gValues[9 * 64 + c]).toBe(0);
      expect(membership.gValues[15 * 64 + c]).toBe(0);
    }
    expect(row0Live).toBe(64);
    expect(row8Live).toBe(1);
    expect(row8Masked).toBe(63);

    // All 256 codebook nodes + 9 depth rings + 5*3 U26 bars emitted as solids
    expect(scene.solidCount).toBeGreaterThanOrEqual(256 + 9);
    expect(scene.lineCount).toBeGreaterThan(200);
    expect(scene.flowIndices.length).toBeGreaterThan(0);
  });

  it('works with a live UniSession across T2I denoising steps and without mutating session state', () => {
    // Also verify a fresh UniSession (before startBranch) does not throw
    const freshSession = new UniSession(sharedBundle);
    const freshScene = buildUniBallTreeScene(freshSession);
    assertBallTreeAtlasContract(freshScene);

    const session = new UniSession(sharedBundle);
    session.startBranch({
      taskMode: 'text_to_image',
      refinementSteps: 4,
      activationSteps: 8,
    });
    session.stepBranch();
    session.stepBranch();
    const activeBefore = session.getActiveRecordId();
    const countBefore = session.getSnapshot().memoryAudit.recordCount;

    // Default options resolve live code & stage cleanly from the active denoising record
    const defaultLiveScene = buildUniBallTreeScene(session);
    assertBallTreeAtlasContract(defaultLiveScene);
    expect(defaultLiveScene.executionViewId).toBe('text_to_image');

    const scene = buildUniBallTreeScene(session, {
      selectedCodeId: 173,
      refinementSteps: 4,
      stageIndex: 2,
    });

    assertBallTreeAtlasContract(scene);
    expect(scene.executionViewId).toBe('text_to_image');
    expect(scene.d5Comparison).not.toBeNull();
    expect(scene.viewReadout).not.toBeNull();
    expect(session.getActiveRecordId()).toBe(activeBefore);
    expect(session.getSnapshot().memoryAudit.recordCount).toBe(countBefore);
    expect(scene.subtitle).toContain('Code #173');
    expect(scene.subtitle).toContain('256 -> 64 -> 16 -> 4 -> 1');
    expect(scene.subtitle).toContain('|N_4|=16');
  });

  it('supports every refinementSteps preset {1, 2, 4, 8} and stageIndex {-1..N}, verifying N=4 funnel and N=1 binary MDM reduction', () => {
    const presets: readonly UniRefinementStepPreset[] = [1, 2, 4, 8];
    const expectedBallSequences: Record<UniRefinementStepPreset, readonly number[]> = {
      1: [256, 1],
      2: [256, 16, 1],
      4: [256, 64, 16, 4, 1],
      8: [256, 128, 64, 32, 16, 8, 4, 2, 1],
    };

    for (const N of presets) {
      const stages = Int32Array.from({ length: N + 1 }, (_, i) => i);
      const { ballSizes } = semanticTreeLevelsForAges(stages, 8, N);
      expect(Array.from(ballSizes)).toEqual(expectedBallSequences[N]);

      for (let stage = -1; stage <= N; stage++) {
        const scene = buildUniBallTreeScene(null, {
          selectedCodeId: 99,
          refinementSteps: N,
          stageIndex: stage,
        });
        assertBallTreeAtlasContract(scene);

        const polarity = getStagePolarityPair(stage, 8, N);
        const expectedMembership = getTreeBallMembership(sharedBundle.tree, 99, polarity.level);
        expect(expectedMembership.ballSize).toBe(polarity.ballSize);

        // Verify row 10 (active stage strip in ball_membership) matches exact live leaf-group count
        const membership = readRegionTexels(scene, 'ball_membership');
        let activeLiveGroups = 0;
        for (let c = 0; c < 64; c++) {
          if (membership.gValues[10 * 64 + c] === 0) activeLiveGroups++;
        }
        expect(activeLiveGroups).toBe(Math.max(1, Math.ceil(polarity.ballSize / 4)));

        // Verify U9 polarity badge carries both codeStage and paperAge
        expect(
          scene.provenanceBadges.some(
            (b) =>
              b.label.includes(`Stage=${polarity.codeStage}`) &&
              b.label.includes(`PaperAge=${polarity.paperAge}`) &&
              b.label.includes(`|N|=${polarity.ballSize}`),
          ),
        ).toBe(true);
      }
    }

    // Explicitly check N = 1 binary MDM badge & subtitle
    const mdmScene = buildUniBallTreeScene(null, {
      selectedCodeId: 12,
      refinementSteps: 1,
      stageIndex: 0,
    });
    expect(mdmScene.subtitle).toContain('256 -> 1');
    expect(mdmScene.provenanceBadges.some((b) => b.label.includes('Binary MDM'))).toBe(true);
  });

  it('attaches exact U26 production D=17 floor-rounding mismatch metadata vs miniature D=8 (0 mismatches), badges, and anchors', () => {
    // Default K = 4: trainLevel [13, 9, 5] -> (16, 256, 4096) vs inferLevel [12, 8, 4] -> (32, 512, 8192)
    const sceneK4 = buildUniBallTreeScene(null, {
      selectedCodeId: 42,
      refinementSteps: 4,
      stageIndex: 1,
      showProductionD17Comparison: true,
    });

    const u26 = sceneK4.u26Mismatch;
    expect(u26).toBeDefined();
    expect(u26).not.toBeNull();
    expect(u26!.treeDepth).toBe(UNI_BALL_TREE_PROD_DEPTH);
    expect(u26!.refinementSteps).toBe(4);
    expect(u26!.hasMismatch).toBe(true);
    expect(u26!.mismatchedStageCount).toBe(3);
    expect(u26!.totalInternalStages).toBe(3);

    const byTrainAge = [...u26!.rows].reverse();
    expect(byTrainAge.map((r) => r.trainLevel)).toEqual([13, 9, 5]);
    expect(byTrainAge.map((r) => r.inferLevel)).toEqual([12, 8, 4]);
    expect(byTrainAge.map((r) => r.trainBallSize)).toEqual([16, 256, 4096]);
    expect(byTrainAge.map((r) => r.inferBallSize)).toEqual([32, 512, 8192]);
    expect(byTrainAge.every((r) => r.ballSizeRatio === 2)).toBe(true);

    // Contrast against miniature D=8 (0 mismatches)
    const miniD8 = analyzePrimeDepthFloorMismatch(8, 4);
    expect(miniD8.hasMismatch).toBe(false);
    expect(miniD8.mismatchedStageCount).toBe(0);

    // Verify badges and U26 overlay anchor expose exact D=17 vs D=8 numbers
    expect(
      sceneK4.provenanceBadges.some(
        (b) =>
          b.label.includes('U26') &&
          b.label.includes('13, 9, 5') &&
          b.label.includes('12, 8, 4'),
      ),
    ).toBe(true);
    expect(
      sceneK4.provenanceBadges.some(
        (b) => b.label.includes('Mini D=8') && b.label.includes('0/3'),
      ),
    ).toBe(true);

    const u26Anchor = sceneK4.overlayAnchors.find((a) => a.id === 'anchor_u26_d17_mismatch');
    expect(u26Anchor).toBeDefined();
    expect(u26Anchor!.sublabel).toContain('13, 9, 5');
    expect(u26Anchor!.sublabel).toContain('16, 256, 4096');
    expect(u26Anchor!.sublabel).toContain('12, 8, 4');
    expect(u26Anchor!.sublabel).toContain('32, 512, 8192');
    expect(u26Anchor!.sublabel).toContain('Mini D=8: 0 mismatches');

    // Also verify K = 8 (7/7 mismatches) and K = 1 (0/0 internal stages)
    const sceneK8 = buildUniBallTreeScene(null, { refinementSteps: 8, stageIndex: 3 });
    expect(sceneK8.u26Mismatch!.mismatchedStageCount).toBe(7);
    expect(sceneK8.u26Mismatch!.totalInternalStages).toBe(7);

    const sceneK1 = buildUniBallTreeScene(null, { refinementSteps: 1, stageIndex: 0 });
    expect(sceneK1.u26Mismatch!.hasMismatch).toBe(false);
    expect(sceneK1.u26Mismatch!.mismatchedStageCount).toBe(0);
    expect(sceneK1.u26Mismatch!.totalInternalStages).toBe(0);
  });

  it('round-trips pickMacroBay -> pickHitToUniCell across all depth rings (0..8), codebook nodes, and all 5 atlas plates', () => {
    const scene = buildUniBallTreeScene(null, {
      selectedCodeId: 200,
      refinementSteps: 4,
      stageIndex: 3, // level 6, ballSize 4
    });

    // 1. Depth rings 0..8 decode exact treeDepth 0..8 via pickMacroBay
    for (let d = 0; d <= 8; d++) {
      const ringBox = scene.pickBoxes.find((pb) => pb.partId === `tree_depth_ring_${d}`);
      expect(ringBox).toBeDefined();
      const hit = pickMacroBay(rayAtBoxTopCenter(ringBox!), scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      expect(hit!.partId).toBe(`tree_depth_ring_${d}`);

      const decoded = pickHitToUniCell(hit, 'ball_tree');
      expect(decoded).not.toBeNull();
      expect(decoded!.viewMode).toBe('ball_tree');
      expect(decoded!.partId).toBe(`tree_depth_ring_${d}`);
      expect(decoded!.treeDepth).toBe(d);
      expect(decoded!.branch).toBe('g');
      expect(decoded!.layerIndex).toBeNull();
    }

    // 2. Codebook nodes decode tokenIndex = codeId and active treeDepth
    const selectedNodeBox = scene.pickBoxes.find((pb) => pb.partId === 'codebook_node_200');
    expect(selectedNodeBox).toBeDefined();
    // Cast directly at the isolated codebook node box (against itself and verify full box metadata)
    const nodeHit = pickMacroBay(rayAtBoxTopCenter(selectedNodeBox!), [selectedNodeBox!], []);
    expect(nodeHit).not.toBeNull();
    const decodedNode = pickHitToUniCell(nodeHit, 'ball_tree');
    expect(decodedNode).not.toBeNull();
    expect(decodedNode!.partId).toBe('codebook_node_200');
    expect(decodedNode!.tokenIndex).toBe(200);
    expect(decodedNode!.treeDepth).toBe(6);
    expect(decodedNode!.branch).toBe('g');

    // 3. All 5 RG32F atlas plates decode partId, (row, col), and active treeDepth
    const plateIds = [
      'codebook_lo',
      'codebook_hi',
      'heap_radii',
      'ball_membership',
      'sincos_2d',
    ] as const;

    for (const plateId of plateIds) {
      const plateBox = scene.pickBoxes.find((pb) => pb.partId === plateId);
      expect(plateBox).toBeDefined();
      expect(plateBox!.frame).toBeDefined();
      const hit = pickMacroBay(rayAtBoxTopCenter(plateBox!), scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      expect(hit!.partId).toBe(plateId);

      const decoded = pickHitToUniCell(hit, 'ball_tree');
      expect(decoded).not.toBeNull();
      expect(decoded!.partId).toBe(plateId);
      if (plateId === 'ball_membership') {
        // ball_membership omits static tapIndex/depth= so per-row depth inspection uses cell.row (0..8)
        expect(plateBox!.tapIndex).toBeUndefined();
        expect(decoded!.treeDepth).toBeNull();
      } else {
        expect(decoded!.treeDepth).toBe(6);
      }
      expect(decoded!.row).not.toBeNull();
      expect(decoded!.col).not.toBeNull();
      expect(decoded!.row!).toBeGreaterThanOrEqual(0);
      expect(decoded!.row!).toBeLessThan(plateBox!.gridRows!);
      expect(decoded!.col!).toBeGreaterThanOrEqual(0);
      expect(decoded!.col!).toBeLessThan(plateBox!.gridCols!);
    }

    // Verify per-row depth raycast on ball_membership decodes distinct row indices cleanly
    const memberBox = scene.pickBoxes.find((pb) => pb.partId === 'ball_membership')!;
    const mCx = (memberBox.min[0] + memberBox.max[0]) * 0.5;
    const mSpanZ = memberBox.max[2] - memberBox.min[2];
    const gridRows = memberBox.gridRows!;
    for (const targetDepthRow of [0, 3, 7]) {
      const z = memberBox.min[2] + ((targetDepthRow + 0.5) / gridRows) * mSpanZ;
      const rowHit = pickMacroBay(
        { origin: [mCx, memberBox.max[1] + 5.0, z], direction: [0, -1, 0] },
        scene.pickBoxes,
        [],
      );
      expect(rowHit?.partId).toBe('ball_membership');
      const rowDecoded = pickHitToUniCell(rowHit, 'ball_tree')!;
      expect(rowDecoded.treeDepth).toBeNull();
      expect(rowDecoded.row).toBe(targetDepthRow);
    }
  });
});
