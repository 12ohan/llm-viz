import { describe, expect, it } from 'vitest';
import { pickMacroBay, type Ray } from '../src/render/pick';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_STANDARD,
} from '../src/render/sheets';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import { buildDualStreamBlockDiffusionMask } from '../src/uni/masks';
import {
  buildUniOpdExplorerScene,
  UNI_OPD_ATLAS_BUDGET_BYTES,
  UNI_OPD_ATLAS_HEIGHT,
  UNI_OPD_ATLAS_WIDTH,
  UNI_OPD_READONLY_BADGE_LABEL,
  UNI_OPD_REGION_COUNT,
  UNI_OPD_SBD_BLOCK_SIZE,
  UNI_OPD_SBD_CLEAN_LEN,
} from '../src/uni/scene/opdExplorerScene';
import {
  pickHitToUniCell,
  validateSceneGeometryFinite,
} from '../src/uni/scene/provenance';
import type { UniAtlasRegionBox, UniSpatialScene } from '../src/uni/scene/types';
import { UniSession } from '../src/uni/session';

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

function assertCleanOpdAtlasAudit(scene: UniSpatialScene): void {
  expect(scene.valueAtlas).not.toBeNull();
  expect(scene.valueAtlas!.width).toBe(UNI_OPD_ATLAS_WIDTH);
  expect(scene.valueAtlas!.height).toBe(UNI_OPD_ATLAS_HEIGHT);
  expect(scene.valueAtlas!.data.byteLength).toBe(UNI_OPD_ATLAS_BUDGET_BYTES);
  expect(scene.atlasKey).toMatch(/^opd_readonly:128x64:/);

  const audit = scene.atlasAudit;
  expect(audit).not.toBeNull();
  expect(audit!.width).toBe(128);
  expect(audit!.height).toBe(64);
  expect(audit!.byteLength).toBe(65_536);
  expect(audit!.maxBudgetBytes).toBe(65_536);
  expect(audit!.withinBudget).toBe(true);
  expect(audit!.regions.length).toBe(UNI_OPD_REGION_COUNT);
  expect(audit!.hasOverlap).toBe(false);
  expect(audit!.overlapPair).toBeNull();
  expect(audit!.outOfBoundsIds.length).toBe(0);
  expect(audit!.utilizationPct).toBeGreaterThan(10);
  expect(audit!.utilizationPct).toBeLessThan(100);
}

describe('UNI5 Slice 5C · View 7 (opd_readonly) — Read-Only Cross-Paradigm OPD Explorer', () => {
  it('builds standalone scene (session = null) with strict read-only metadata, 128x64 RG32F atlas, 0 RGB previews, and U25 compliance', () => {
    const scene = buildUniOpdExplorerScene(null);

    expect(scene.viewMode).toBe('opd_readonly');
    expect(scene.layerIndex).toBeNull();
    expect(scene.rgbPreviews).toEqual([]);
    expect(scene.sheetCount).toBe(5);
    expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    assertCleanOpdAtlasAudit(scene);

    // Strictly read-only, zero-gradient metadata
    expect(scene.opdReadOnly).toBeDefined();
    expect(scene.opdReadOnly).not.toBeNull();
    expect(scene.opdReadOnly!.readOnly).toBe(true);
    expect(scene.opdReadOnly!.hasGradients).toBe(false);
    expect(scene.opdReadOnly!.badgeLabel).toBe(UNI_OPD_READONLY_BADGE_LABEL);
    expect(scene.opdReadOnly!.understandingOpd.readOnly).toBe(true);
    expect(scene.opdReadOnly!.understandingOpd.hasGradients).toBe(false);
    expect(scene.opdReadOnly!.flowEditingOpd.readOnly).toBe(true);
    expect(scene.opdReadOnly!.flowEditingOpd.hasGradients).toBe(false);
    expect(scene.opdReadOnly!.u28Diagnostics.readOnly).toBe(true);

    // Prominent readonly badge with tone === 'readonly'
    expect(
      scene.provenanceBadges.some(
        (b) => b.label === UNI_OPD_READONLY_BADGE_LABEL && b.tone === 'readonly',
      ),
    ).toBe(true);

    // Verify exact 5 region IDs & dimensions
    expect(scene.atlasRegions.map((r) => r.id)).toEqual([
      'sbd_dual_stream_mask',
      'opd_transitive_tail_kl',
      'opd_self_consistency_kl',
      'opd_matched_sigmas',
      'opd_flow_latents',
    ]);
    expect(readRegionTexels(scene, 'sbd_dual_stream_mask').region).toMatchObject({
      rows: 32,
      cols: 32,
      isMaskOrScore: true,
    });
    expect(readRegionTexels(scene, 'opd_transitive_tail_kl').region).toMatchObject({
      rows: 6,
      cols: 43,
      isMaskOrScore: false,
    });
    expect(readRegionTexels(scene, 'opd_self_consistency_kl').region).toMatchObject({
      rows: 6,
      cols: 43,
      isMaskOrScore: false,
    });
    expect(readRegionTexels(scene, 'opd_matched_sigmas').region).toMatchObject({
      rows: 2,
      cols: 40,
      isMaskOrScore: false,
    });
    expect(readRegionTexels(scene, 'opd_flow_latents').region).toMatchObject({
      rows: 3,
      cols: 16,
      isMaskOrScore: false,
    });

    // Hard U25 compliance: Rect 1 (sheet index 0) is SHEET_KIND_CAUSAL_SCORES with live non-empty TexRect
    const sbdBase = 0 * SHEET_INSTANCE_FLOATS;
    expect(Math.round(scene.sheetData[sbdBase + 8]!)).toBe(SHEET_KIND_CAUSAL_SCORES);
    expect(scene.sheetData[sbdBase + 15]!).toBeCloseTo(32 / 128, 6);
    expect(scene.sheetData[sbdBase + 16]!).toBeCloseTo(32 / 64, 6);

    // Other 4 plates use SHEET_KIND_STANDARD with live non-empty TexRects
    for (let s = 1; s < 5; s++) {
      const base = s * SHEET_INSTANCE_FLOATS;
      expect(Math.round(scene.sheetData[base + 8]!)).toBe(SHEET_KIND_STANDARD);
      expect(scene.sheetData[base + 15]!).toBeGreaterThan(0);
      expect(scene.sheetData[base + 16]!).toBeGreaterThan(0);
    }
  });

  it('enforces SheetsPass RG32F G-channel discipline: G=1.0 ONLY on blocked SBD cells, G=0.0 on all live SBD, tail_kl, matched_sigmas, and flow-latent cells', () => {
    const scene = buildUniOpdExplorerScene(null);
    const expectedVisible = buildDualStreamBlockDiffusionMask(
      UNI_OPD_SBD_CLEAN_LEN,
      UNI_OPD_SBD_BLOCK_SIZE,
    );

    // Rect 1: sbd_dual_stream_mask
    const sbd = readRegionTexels(scene, 'sbd_dual_stream_mask');
    let visibleCount = 0;
    let blockedCount = 0;
    for (let i = 0; i < expectedVisible.length; i++) {
      if (expectedVisible[i] === 1) {
        expect(sbd.gValues[i]).toBe(0.0);
        expect(sbd.rValues[i]).toBeGreaterThan(0);
        visibleCount++;
      } else {
        expect(sbd.gValues[i]).toBe(1.0);
        expect(sbd.rValues[i]).toBe(0.0);
        blockedCount++;
      }
    }
    expect(visibleCount).toBeGreaterThan(0);
    expect(blockedCount).toBeGreaterThan(0);
    expect(visibleCount + blockedCount).toBe(1024);

    // Rect 2: opd_transitive_tail_kl (all 258 cells active, G = 0.0, exact 129-bucket probabilities)
    const trans = readRegionTexels(scene, 'opd_transitive_tail_kl');
    const arKl = scene.opdReadOnly!.understandingOpd.arTailKl;
    expect(Array.from(trans.gValues).every((g) => g === 0.0)).toBe(true);
    for (let b = 0; b < 129; b++) {
      expect(trans.rValues[b]).toBeCloseTo(arKl.pTargetBuckets[b]!, 6);
      expect(trans.rValues[129 + b]).toBeCloseTo(arKl.pStudentBuckets[b]!, 6);
    }

    // Rect 3: opd_self_consistency_kl (all 258 cells active, G = 0.0, exact 129-bucket probabilities)
    const cons = readRegionTexels(scene, 'opd_self_consistency_kl');
    const consKl = scene.opdReadOnly!.understandingOpd.consistencyTailKl;
    expect(Array.from(cons.gValues).every((g) => g === 0.0)).toBe(true);
    for (let b = 0; b < 129; b++) {
      expect(cons.rValues[b]).toBeCloseTo(consKl.pTargetBuckets[b]!, 6);
      expect(cons.rValues[129 + b]).toBeCloseTo(consKl.pStudentBuckets[b]!, 6);
    }

    // Rect 4: opd_matched_sigmas (all 80 cells active, G = 0.0, row 0 matches sigmas)
    const sigmas = readRegionTexels(scene, 'opd_matched_sigmas');
    const matched = scene.opdReadOnly!.flowEditingOpd.matchedSigmas;
    expect(Array.from(sigmas.gValues).every((g) => g === 0.0)).toBe(true);
    for (let k = 0; k < 40; k++) {
      expect(sigmas.rValues[k]).toBeCloseTo(matched.sigmas[k]!, 6);
    }

    // Rect 5: opd_flow_latents (all 48 cells active, G = 0.0, exact x^0, delta, x^r)
    const latents = readRegionTexels(scene, 'opd_flow_latents');
    const flowOpd = scene.opdReadOnly!.flowEditingOpd;
    expect(Array.from(latents.gValues).every((g) => g === 0.0)).toBe(true);
    for (let i = 0; i < 16; i++) {
      expect(latents.rValues[i]).toBeCloseTo(flowOpd.x0[i]!, 6);
      expect(latents.rValues[16 + i]).toBeCloseTo(flowOpd.noiseDelta[i]!, 6);
      expect(latents.rValues[32 + i]).toBeCloseTo(flowOpd.xNoisyR[i]!, 6);
    }
  });

  it('surfaces "New Francisco" context-mismatch callout, all 3 U28 reference-trainer bugs, and hybrid editing CE readout', () => {
    const scene = buildUniOpdExplorerScene(null);
    const u28 = scene.opdReadOnly!.u28Diagnostics;

    // "New Francisco" context-mismatch callout anchor & pickBox
    const nfAnchor = scene.overlayAnchors.find((a) => a.id === 'anchor_opd_new_francisco');
    expect(nfAnchor).toBeDefined();
    expect(nfAnchor!.label).toContain('New Francisco');
    expect(nfAnchor!.sublabel).toContain('Francisco');
    expect(nfAnchor!.tone).toBe('warn');

    // All 3 U28 reference-trainer bugs surfaced in overlay anchor & provenance badges
    const u28Anchor = scene.overlayAnchors.find((a) => a.id === 'anchor_opd_u28_diagnostics');
    expect(u28Anchor).toBeDefined();
    expect(u28Anchor!.tone).toBe('warn');
    expect(u28Anchor!.sublabel).toContain('6.915');
    expect(u28Anchor!.sublabel).toContain('1,000');
    expect(u28Anchor!.sublabel).toContain('<|im_end|>');
    expect(u28Anchor!.sublabel).toContain('id=11');
    expect(u28Anchor!.sublabel).toContain('132,101 - 64');
    expect(u28Anchor!.sublabel).toContain('<LOC_*>');

    expect(u28.sftDoubleNorm.expectedInflationFactor).toBeCloseTo(Math.log(1000) / 0.999, 6);
    expect(u28.sftDoubleNorm.peakSpikeAtTMin).toBe(1000);
    expect(u28.accidentalImEndSupervision.eosTokenId).toBe(11);
    expect(u28.opdLocTailLeak.fullVocabSize - u28.opdLocTailLeak.topK).toBe(132_037);
    expect(u28.opdLocTailLeak.locExtensionTokens).toBe(1_029);

    // Right bay exact_ce + 0.2 * grouped_ce readout
    const requantAnchor = scene.overlayAnchors.find((a) => a.id === 'anchor_opd_g_requant_target');
    expect(requantAnchor).toBeDefined();
    expect(requantAnchor!.label).toContain('exact_ce + 0.2 * grouped_ce');
    expect(requantAnchor!.sublabel).toContain('exact_ce=');
    expect(requantAnchor!.sublabel).toContain('grouped_ce');
  });

  it('works with a live UniSession without mutating session state and attaches session readouts', () => {
    const session = new UniSession(sharedBundle);
    session.startBranch({ taskMode: 'image_edit', scenePresetId: 'sunlit_cat_16x16' });

    const activeBefore = session.getActiveRecordId();
    const branchBefore = session.getActiveBranchId();
    const recordCountBefore = session.getAllRecords().length;

    const scene = buildUniOpdExplorerScene(session);

    expect(session.getActiveRecordId()).toBe(activeBefore);
    expect(session.getActiveBranchId()).toBe(branchBefore);
    expect(session.getAllRecords().length).toBe(recordCountBefore);

    expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    assertCleanOpdAtlasAudit(scene);
    expect(scene.executionViewId).toBe('image_edit');
    expect(scene.d5Comparison).not.toBeNull();
    expect(scene.viewReadout).not.toBeNull();
    expect(scene.rgbPreviews).toEqual([]);
    expect(scene.opdReadOnly!.readOnly).toBe(true);
    expect(scene.opdReadOnly!.hasGradients).toBe(false);

    // Also works with a fresh UniSession before startBranch()
    const emptySession = new UniSession(sharedBundle);
    const emptyScene = buildUniOpdExplorerScene(emptySession);
    expect(validateSceneGeometryFinite(emptyScene).valid).toBe(true);
    assertCleanOpdAtlasAudit(emptyScene);
    expect(emptySession.getActiveRecordId()).toBeNull();
  });

  it('supports custom r, steps (including wrapped > 88 steps), seqLen, selectedBay, and strict determinism', () => {
    const sceneA = buildUniOpdExplorerScene(null, {
      r: 0.75,
      steps: 24,
      seqLen: 2048,
      selectedBay: 'understanding',
    });
    const sceneB = buildUniOpdExplorerScene(null, {
      r: 0.75,
      steps: 24,
      seqLen: 2048,
      selectedBay: 'understanding',
    });

    expect(validateSceneGeometryFinite(sceneA).valid).toBe(true);
    assertCleanOpdAtlasAudit(sceneA);
    expect(sceneA.atlasKey).toBe(sceneB.atlasKey);
    expect(sceneA.valueAtlas!.data).toEqual(sceneB.valueAtlas!.data);
    expect(sceneA.opdReadOnly!.flowEditingOpd.r).toBe(0.75);
    expect(sceneA.opdReadOnly!.flowEditingOpd.matchedSigmas.steps).toBe(24);
    expect(sceneA.opdReadOnly!.flowEditingOpd.matchedSigmas.seqLen).toBe(2048);

    const sig24 = readRegionTexels(sceneA, 'opd_matched_sigmas');
    expect(sig24.region).toMatchObject({ rows: 2, cols: 24 });
    expect(Array.from(sig24.gValues).every((g) => g === 0.0)).toBe(true);

    // selectedBay === 'understanding': left plates highlighted OPERAND_A, right plates DIMMED
    expect(Math.round(sceneA.sheetData[0 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_OPERAND_A);
    expect(Math.round(sceneA.sheetData[3 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_DIMMED);
    expect(Math.round(sceneA.sheetData[4 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_DIMMED);

    // selectedBay === 'editing': right plates highlighted OPERAND_B, left plates DIMMED
    const editingScene = buildUniOpdExplorerScene(null, { selectedBay: 'editing' });
    expect(validateSceneGeometryFinite(editingScene).valid).toBe(true);
    expect(Math.round(editingScene.sheetData[0 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_DIMMED);
    expect(Math.round(editingScene.sheetData[3 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_OPERAND_B);
    expect(Math.round(editingScene.sheetData[4 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(SHEET_HL_OPERAND_B);

    // Wrapped custom steps > 88 (e.g., steps = 100): fits inside 128x64, G=0.0 on 0..99, G=1.0 ONLY on 100..127 padding
    const wrappedScene = buildUniOpdExplorerScene(null, { steps: 100 });
    expect(validateSceneGeometryFinite(wrappedScene).valid).toBe(true);
    assertCleanOpdAtlasAudit(wrappedScene);
    const sig100 = readRegionTexels(wrappedScene, 'opd_matched_sigmas');
    expect(sig100.region).toMatchObject({ rows: 2, cols: 64 });
    for (let i = 0; i < 128; i++) {
      if (i < 100) {
        expect(sig100.gValues[i]).toBe(0.0);
        expect(sig100.rValues[i]).toBeGreaterThan(0);
      } else {
        expect(sig100.gValues[i]).toBe(1.0);
        expect(sig100.rValues[i]).toBe(0.0);
      }
    }
  });

  it('round-trips PickBox ray intersections through pickHitToUniCell on all 5 atlas plates and stage cards', () => {
    const scene = buildUniOpdExplorerScene(null);
    // 5 atlas plates + 9 stage/callout cards = 14 PickBoxes
    expect(scene.pickBoxes.length).toBe(14);

    const expectedBranches: Record<string, 'u' | 'g'> = {
      sbd_dual_stream_mask: 'u',
      opd_transitive_tail_kl: 'u',
      opd_self_consistency_kl: 'u',
      opd_matched_sigmas: 'g',
      opd_flow_latents: 'g',
      opd_u_ar_teacher_card: 'u',
      opd_u_student_ar_card: 'u',
      opd_u_stopgrad_card: 'u',
      opd_u_new_francisco_callout: 'u',
      opd_u28_diagnostics_card: 'u',
      opd_g_vq_rollout_card: 'g',
      opd_g_vae_encode_card: 'g',
      opd_g_teacher_ode_card: 'g',
      opd_g_requant_target_card: 'g',
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
      expect(hit!.partId).toBe(box.partId);

      const cell = pickHitToUniCell(hit, 'opd_readonly');
      expect(cell).not.toBeNull();
      expect(cell!.viewMode).toBe('opd_readonly');
      expect(cell!.layerIndex).toBeNull();
      expect(cell!.partId).toBe(box.partId);
      expect(cell!.tensorKey).toBe(box.tensorKey);
      expect(cell!.branch).toBe(expectedBranches[box.tensorKey!]);
      expect(cell!.row).not.toBeNull();
      expect(cell!.col).not.toBeNull();
      expect(cell!.row!).toBeGreaterThanOrEqual(0);
      expect(cell!.row!).toBeLessThan(box.gridRows!);
      expect(cell!.col!).toBeGreaterThanOrEqual(0);
      expect(cell!.col!).toBeLessThan(box.gridCols!);
    }

    // Verify hoveredCell option highlights targeted plate with SHEET_HL_ACTIVE and adds cell highlight geometry
    const hoveredScene = buildUniOpdExplorerScene(null, {
      hoveredCell: {
        viewMode: 'opd_readonly',
        layerIndex: null,
        isSplitLayer: null,
        branch: 'g',
        partId: 'opd_matched_sigmas',
        tensorKey: 'opd_matched_sigmas',
        qHeadIndex: null,
        kvHeadIndex: null,
        cfgBatchRow: null,
        rotaryPlane: null,
        treeDepth: null,
        tokenIndex: 0,
        absPosition: null,
        row: 0,
        col: 10,
        semanticTag: null,
      },
    });
    expect(validateSceneGeometryFinite(hoveredScene).valid).toBe(true);
    // opd_matched_sigmas is sheet index 4
    expect(Math.round(hoveredScene.sheetData[4 * SHEET_INSTANCE_FLOATS + 11]!)).toBe(
      SHEET_HL_ACTIVE,
    );
    expect(hoveredScene.flowIndices.length).toBeGreaterThan(scene.flowIndices.length);
  });
});
