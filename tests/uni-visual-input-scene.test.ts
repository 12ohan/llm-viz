import { describe, expect, it } from 'vitest';
import { pickMacroBay, type Ray } from '../src/render/pick';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SOFTMAX,
} from '../src/render/sheets';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import {
  pickHitToUniCell,
  UNI_SCENE_VISUAL_HONESTY_BADGE,
  validateSceneGeometryFinite,
} from '../src/uni/scene/provenance';
import {
  buildUniVisualInputScene,
  UNI_VISUAL_INPUT_ATLAS_BUDGET_BYTES,
  UNI_VISUAL_INPUT_ATLAS_SIZE,
  UNI_VISUAL_INPUT_REGION_COUNT,
} from '../src/uni/scene/visualInputScene';
import { UniSession } from '../src/uni/session';
import type { UniAtlasRegionBox, UniSpatialScene } from '../src/uni/scene/types';

const sharedBundle = createDefaultUniRuntimeBundle();

function createSessionWithBranch(
  options: Parameters<UniSession['startBranch']>[0],
): UniSession {
  const session = new UniSession(sharedBundle);
  session.startBranch(options);
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

function assertValidRgbPreviews(scene: UniSpatialScene): void {
  expect(scene.rgbPreviews.length).toBeGreaterThanOrEqual(1);
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

function assertCleanAtlasAudit(scene: UniSpatialScene): void {
  expect(scene.valueAtlas).not.toBeNull();
  expect(scene.valueAtlas!.width).toBe(UNI_VISUAL_INPUT_ATLAS_SIZE);
  expect(scene.valueAtlas!.height).toBe(UNI_VISUAL_INPUT_ATLAS_SIZE);
  expect(scene.valueAtlas!.data.byteLength).toBe(UNI_VISUAL_INPUT_ATLAS_BUDGET_BYTES);
  expect(scene.atlasKey).toMatch(/^visual_input:128x128:/);

  const audit = scene.atlasAudit;
  expect(audit).not.toBeNull();
  expect(audit!.width).toBe(128);
  expect(audit!.height).toBe(128);
  expect(audit!.byteLength).toBe(131_072);
  expect(audit!.regions.length).toBe(UNI_VISUAL_INPUT_REGION_COUNT);
  expect(audit!.hasOverlap).toBe(false);
  expect(audit!.overlapPair).toBeNull();
  expect(audit!.outOfBoundsIds.length).toBe(0);
  expect(audit!.withinBudget).toBe(true);
  expect(audit!.utilizationPct).toBeGreaterThan(5);
  expect(audit!.utilizationPct).toBeLessThan(100);
}

describe('UNI4 View 2 (visual_input) — Dual Visual Input Pipeline Scene & 128x128 RG32F Atlas', () => {
  it('builds square sunlit_cat_16x16 visual_understanding scene with 10 RG32F rects, dimmed discrete stream, U8 badge, and U25 compliance', () => {
    const session = createSessionWithBranch({
      taskMode: 'visual_understanding',
      scenePresetId: 'sunlit_cat_16x16',
    });

    const scene = buildUniVisualInputScene(session);

    expect(scene.viewMode).toBe('visual_input');
    expect(scene.layerIndex).toBeNull();
    expect(scene.executionViewId).toBe('visual_understanding');
    expect(scene.sheetCount).toBe(10);
    expect(scene.pickBoxes.length).toBe(10);
    expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    assertCleanAtlasAudit(scene);
    assertValidRgbPreviews(scene);

    // Square visual_understanding emits 'source' preview (16x16) and no forced letterbox preview
    expect(scene.rgbPreviews.length).toBe(1);
    expect(scene.rgbPreviews[0]!.kind).toBe('source');
    expect(scene.rgbPreviews[0]!.width).toBe(16);
    expect(scene.rgbPreviews[0]!.height).toBe(16);

    // Verify exact stage dimensions for 16x16
    expect(readRegionTexels(scene, 'u_patches').region).toMatchObject({ rows: 16, cols: 48 });
    expect(readRegionTexels(scene, 'u_vit_attn').region).toMatchObject({ rows: 20, cols: 20 });
    expect(readRegionTexels(scene, 'u_patch_strip').region).toMatchObject({ rows: 16, cols: 16 });
    expect(readRegionTexels(scene, 'u_projector_gather').region).toMatchObject({
      rows: 4,
      cols: 64,
    });
    expect(readRegionTexels(scene, 'u_projector_newline').region).toMatchObject({
      rows: 6,
      cols: 8,
    });
    expect(readRegionTexels(scene, 'g_vq_latents').region).toMatchObject({ rows: 16, cols: 8 });
    expect(readRegionTexels(scene, 'g_ibq_scores').region).toMatchObject({ rows: 16, cols: 64 });
    expect(readRegionTexels(scene, 'g_vq_code_grid').region).toMatchObject({ rows: 4, cols: 4 });
    expect(readRegionTexels(scene, 'g_packed_codec').region).toMatchObject({ rows: 4, cols: 8 });
    expect(readRegionTexels(scene, 'joint_multimodal_seq').region).toMatchObject({
      rows: 11,
      cols: 8,
    });

    // U25 compliance: ViT attention plate is SHEET_KIND_CAUSAL_SOFTMAX with positive TexRect & G = 0.0 unmasked
    const vitSheetIdx = scene.pickBoxes.findIndex((p) => p.tensorKey === 'u_vit_attn');
    expect(vitSheetIdx).toBeGreaterThanOrEqual(0);
    const vitBase = vitSheetIdx * SHEET_INSTANCE_FLOATS;
    expect(Math.round(scene.sheetData[vitBase + 8]!)).toBe(SHEET_KIND_CAUSAL_SOFTMAX);
    expect(scene.sheetData[vitBase + 15]!).toBeGreaterThan(0);
    expect(scene.sheetData[vitBase + 16]!).toBeGreaterThan(0);

    const vitAttn = readRegionTexels(scene, 'u_vit_attn');
    for (let r = 0; r < 20; r++) {
      let rowSum = 0;
      for (let c = 0; c < 20; c++) {
        expect(vitAttn.gValues[r * 20 + c]).toBe(0);
        rowSum += vitAttn.rValues[r * 20 + c]!;
      }
      expect(rowSum).toBeCloseTo(1.0, 4);
    }

    // Discrete stream sheets are dimmed in visual_understanding unless includeDiscreteStream is true
    const gStageIds = ['g_vq_latents', 'g_ibq_scores', 'g_vq_code_grid', 'g_packed_codec'];
    for (const gid of gStageIds) {
      const idx = scene.pickBoxes.findIndex((p) => p.tensorKey === gid);
      expect(Math.round(scene.sheetData[idx * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_DIMMED);
    }

    // In visual_understanding without includeDiscreteStream, assembled sequence is U-only (G = 0.0)
    const seqData = readRegionTexels(scene, 'joint_multimodal_seq');
    expect(Array.from(seqData.gValues).every((g) => g === 0)).toBe(true);

    // Explicit U8 provenance badge & anchor documenting encoder.image_newline unused
    expect(
      scene.provenanceBadges.some(
        (b) => b.label.includes('U8') && b.label.includes('image_newline'),
      ),
    ).toBe(true);
    expect(
      scene.overlayAnchors.some(
        (a) => a.id === 'anchor_u_projector_newline' && (a.badge ?? '').includes('U8'),
      ),
    ).toBe(true);
  });

  it('activates both U and G streams on square sunlit_cat_16x16 image_edit with mixed routing flags and letterboxed RGB preview', () => {
    const session = createSessionWithBranch({
      taskMode: 'image_edit',
      scenePresetId: 'sunlit_cat_16x16',
    });

    const scene = buildUniVisualInputScene(session);
    expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    assertCleanAtlasAudit(scene);
    assertValidRgbPreviews(scene);

    // image_edit emits both 'source' and 'letterboxed' RGB previews
    const kinds = scene.rgbPreviews.map((p) => p.kind);
    expect(kinds).toEqual(['source', 'letterboxed']);
    expect(scene.rgbPreviews[1]!.paddingLockMask).toBeDefined();
    expect(scene.rgbPreviews[1]!.paddingLockMask!.lockedCellCount).toBe(0);

    // Discrete stream plates are NOT dimmed in image_edit
    const gIdx = scene.pickBoxes.findIndex((p) => p.tensorKey === 'g_packed_codec');
    expect(Math.round(scene.sheetData[gIdx * SHEET_INSTANCE_FLOATS + 11]!)).not.toBe(
      SHEET_HL_DIMMED,
    );

    // Assembled sequence is [2 prefix U + 6 RADIO-H/nl U + 4 packed VQ G + 3 edit text U] = 15 x 8
    const seq = readRegionTexels(scene, 'joint_multimodal_seq');
    expect(seq.region).toMatchObject({ rows: 15, cols: 8 });
    let uRows = 0;
    let gRows = 0;
    let nonZeroGRValues = 0;
    for (let r = 0; r < 15; r++) {
      const gFlag = seq.gValues[r * 8]!;
      for (let c = 0; c < 8; c++) {
        expect(seq.gValues[r * 8 + c]).toBe(gFlag);
      }
      if (gFlag === 1) {
        gRows += 1;
        for (let c = 0; c < 8; c++) {
          if (Math.abs(seq.rValues[r * 8 + c]!) > 1e-7) nonZeroGRValues += 1;
        }
      } else {
        expect(gFlag).toBe(0);
        uRows += 1;
      }
    }
    expect(uRows).toBe(11);
    expect(gRows).toBe(4);
    // Crucial check: R embeddings on G-flagged rows are preserved (not zeroed by setTexel)
    expect(nonZeroGRValues).toBeGreaterThan(0);
  });

  it('emits 16x8 source + 16x16 letterboxed RGB preview with non-zero U18 paddingLockMask on harbor_panorama_16x8', () => {
    const session = createSessionWithBranch({
      taskMode: 'image_edit',
      scenePresetId: 'harbor_panorama_16x8',
    });

    const scene = buildUniVisualInputScene(session);
    expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    assertCleanAtlasAudit(scene);
    assertValidRgbPreviews(scene);

    const srcPreview = scene.rgbPreviews.find((p) => p.kind === 'source');
    const lbPreview = scene.rgbPreviews.find((p) => p.kind === 'letterboxed');
    expect(srcPreview).toBeDefined();
    expect(lbPreview).toBeDefined();
    expect(srcPreview!.width).toBe(16);
    expect(srcPreview!.height).toBe(8);
    expect(lbPreview!.width).toBe(16);
    expect(lbPreview!.height).toBe(16);
    expect(lbPreview!.paddingLockMask).toBeDefined();
    expect(lbPreview!.paddingLockMask!.lockedCellCount).toBe(8);
    expect(lbPreview!.paddingLockMask!.editableCellCount).toBe(8);

    // Continuous RADIO-H runs at native 16x8 aspect: 8 patches, 12x12 ViT attn, 2x64 gather, 3x8 newline
    expect(readRegionTexels(scene, 'u_patches').region).toMatchObject({ rows: 8, cols: 48 });
    expect(readRegionTexels(scene, 'u_vit_attn').region).toMatchObject({ rows: 12, cols: 12 });
    expect(readRegionTexels(scene, 'u_patch_strip').region).toMatchObject({ rows: 8, cols: 16 });
    expect(readRegionTexels(scene, 'u_projector_gather').region).toMatchObject({
      rows: 2,
      cols: 64,
    });
    expect(readRegionTexels(scene, 'u_projector_newline').region).toMatchObject({
      rows: 3,
      cols: 8,
    });

    // 4x4 quantized code grid encodes R = codeId/255 in [0, 1] and G = U18 paddingLockMask (8 locked)
    const codeGrid = readRegionTexels(scene, 'g_vq_code_grid');
    expect(codeGrid.region).toMatchObject({ rows: 4, cols: 4 });
    let lockedTexels = 0;
    for (let i = 0; i < 16; i++) {
      expect(codeGrid.rValues[i]!).toBeGreaterThanOrEqual(0);
      expect(codeGrid.rValues[i]!).toBeLessThanOrEqual(1);
      const expectedG = lbPreview!.paddingLockMask!.lockMask[i] === 1 ? 1.0 : 0.0;
      expect(codeGrid.gValues[i]).toBe(expectedG);
      if (expectedG === 1.0) lockedTexels += 1;
    }
    expect(lockedTexels).toBe(8);
  });

  it('handles text-only sessions cleanly via deterministic default preset fallback and explicit muted badge', () => {
    const session = createSessionWithBranch({
      taskMode: 'text_understanding',
    });

    const scene = buildUniVisualInputScene(session);
    expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    assertCleanAtlasAudit(scene);
    assertValidRgbPreviews(scene);

    expect(scene.rgbPreviews[0]!.width).toBe(16);
    expect(scene.rgbPreviews[0]!.height).toBe(16);
    expect(
      scene.provenanceBadges.some(
        (b) =>
          b.label === 'Text-Only Session · Previewing Default Visual Input Preset' &&
          b.tone === 'muted',
      ),
    ).toBe(true);

    // Enabling includeDiscreteStream explicitly illuminates G stream even on text-only session
    const withDiscrete = buildUniVisualInputScene(session, { includeDiscreteStream: true });
    expect(validateSceneGeometryFinite(withDiscrete).valid).toBe(true);
    const seqWithG = readRegionTexels(withDiscrete, 'joint_multimodal_seq');
    expect(Array.from(seqWithG.gValues).some((g) => g === 1.0)).toBe(true);
  });

  it('produces distinct live RG32F values on both RadioHProjector and PackedLinearImageCodec slices across D3 gain presets', () => {
    const session = createSessionWithBranch({
      taskMode: 'image_edit',
      scenePresetId: 'sunlit_cat_16x16',
    });

    const miniScene = buildUniVisualInputScene(session, { gainPresetId: 'mini_default' });
    const prodScene = buildUniVisualInputScene(session, { gainPresetId: 'prod_exact' });

    expect(validateSceneGeometryFinite(miniScene).valid).toBe(true);
    expect(validateSceneGeometryFinite(prodScene).valid).toBe(true);
    expect(miniScene.atlasKey).not.toBe(prodScene.atlasKey);

    const miniRadio = readRegionTexels(miniScene, 'u_projector_newline').rValues;
    const prodRadio = readRegionTexels(prodScene, 'u_projector_newline').rValues;
    let maxRadioDiff = 0;
    for (let i = 0; i < miniRadio.length; i++) {
      const diff = Math.abs(miniRadio[i]! - prodRadio[i]!);
      if (diff > maxRadioDiff) maxRadioDiff = diff;
    }
    expect(maxRadioDiff).toBeGreaterThan(1e-4);

    const miniCodec = readRegionTexels(miniScene, 'g_packed_codec').rValues;
    const prodCodec = readRegionTexels(prodScene, 'g_packed_codec').rValues;
    let maxCodecDiff = 0;
    for (let i = 0; i < miniCodec.length; i++) {
      const diff = Math.abs(miniCodec[i]! - prodCodec[i]!);
      if (diff > maxCodecDiff) maxCodecDiff = diff;
    }
    expect(maxCodecDiff).toBeGreaterThan(1e-4);
  });

  it('round-trips PickBox ray intersections cleanly through pickHitToUniCell for all 10 stage plates', () => {
    const session = createSessionWithBranch({
      taskMode: 'image_edit',
      scenePresetId: 'sunlit_cat_16x16',
    });

    const scene = buildUniVisualInputScene(session);
    expect(scene.pickBoxes.length).toBe(10);

    const expectedBranches: Record<string, 'u' | 'g' | 'joint'> = {
      u_patches: 'u',
      u_vit_attn: 'u',
      u_patch_strip: 'u',
      u_projector_gather: 'u',
      u_projector_newline: 'u',
      g_vq_latents: 'g',
      g_ibq_scores: 'g',
      g_vq_code_grid: 'g',
      g_packed_codec: 'g',
      joint_multimodal_seq: 'joint',
    };

    for (const box of scene.pickBoxes) {
      expect(box.frame).toBeDefined();
      const c = box.frame!.center;
      const ray: Ray = {
        origin: [c[0], c[1] + 5.0, c[2]],
        direction: [0, -1, 0],
      };

      const hit = pickMacroBay(ray, scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      expect(hit!.tensorKey).toBe(box.tensorKey);

      const cell = pickHitToUniCell(hit, 'visual_input');
      expect(cell).not.toBeNull();
      expect(cell!.viewMode).toBe('visual_input');
      expect(cell!.layerIndex).toBeNull();
      expect(cell!.tensorKey).toBe(box.tensorKey);
      expect(cell!.partId).toBe(box.partId);
      expect(cell!.branch).toBe(expectedBranches[box.tensorKey!]);
      expect(cell!.row).not.toBeNull();
      expect(cell!.col).not.toBeNull();
      expect(cell!.row!).toBeGreaterThanOrEqual(0);
      expect(cell!.row!).toBeLessThan(box.gridRows!);
      expect(cell!.col!).toBeGreaterThanOrEqual(0);
      expect(cell!.col!).toBeLessThan(box.gridCols!);
    }

    // Verify hoveredCell option highlights the targeted plate with SHEET_HL_ACTIVE
    const hoveredScene = buildUniVisualInputScene(session, {
      hoveredCell: {
        viewMode: 'visual_input',
        layerIndex: null,
        isSplitLayer: null,
        branch: 'g',
        partId: 'g_ibq_scores',
        tensorKey: 'g_ibq_scores',
        qHeadIndex: null,
        kvHeadIndex: null,
        cfgBatchRow: null,
        rotaryPlane: null,
        treeDepth: null,
        tokenIndex: 3,
        absPosition: null,
        row: 3,
        col: 12,
        semanticTag: null,
      },
    });
    expect(validateSceneGeometryFinite(hoveredScene).valid).toBe(true);
    const ibqIdx = hoveredScene.pickBoxes.findIndex((p) => p.tensorKey === 'g_ibq_scores');
    expect(Math.round(hoveredScene.sheetData[ibqIdx * SHEET_INSTANCE_FLOATS + 11]!)).toBe(
      SHEET_HL_ACTIVE,
    );
  });

  it('is strictly deterministic (bit-exact RG32F & atlasKey) and non-mutating across object_grounding and geometric_blocks_16x16', () => {
    const session = createSessionWithBranch({
      taskMode: 'object_grounding',
      scenePresetId: 'geometric_blocks_16x16',
    });
    const activeBefore = session.getActiveRecordId();
    const recordCountBefore = session.getAllRecords().length;

    const sceneA = buildUniVisualInputScene(session);
    const sceneB = buildUniVisualInputScene(session);

    expect(session.getActiveRecordId()).toBe(activeBefore);
    expect(session.getAllRecords().length).toBe(recordCountBefore);

    expect(validateSceneGeometryFinite(sceneA).valid).toBe(true);
    assertCleanAtlasAudit(sceneA);
    assertValidRgbPreviews(sceneA);
    expect(sceneA.atlasKey).toBe(sceneB.atlasKey);
    expect(sceneA.valueAtlas!.data).toEqual(sceneB.valueAtlas!.data);

    // Discrete stream is dimmed in object_grounding mode
    const vqIdx = sceneA.pickBoxes.findIndex((p) => p.tensorKey === 'g_vq_latents');
    expect(Math.round(sceneA.sheetData[vqIdx * SHEET_INSTANCE_FLOATS + 11]!)).toBe(
      SHEET_HL_DIMMED,
    );
  });
});
