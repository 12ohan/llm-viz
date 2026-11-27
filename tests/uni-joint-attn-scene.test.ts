import { describe, expect, it } from 'vitest';
import {
  packPickBoxFromFrame,
  pickHitToUniCell,
  resolveUniSessionContext,
  UniSheetBufferBuilder,
  validateSceneGeometryFinite,
} from '../src/uni/scene/provenance';
import {
  packUniJointAttnAtlas,
  readAtlasTexel,
  UNI_JOINT_ATTN_ATLAS_HEIGHT,
  UNI_JOINT_ATTN_ATLAS_WIDTH,
  UNI_JOINT_ATTN_MAX_REGIONS,
} from '../src/uni/scene/jointAttnAtlas';
import {
  buildUniJointAttnScene,
  UNI_JOINT_ATTN_LAYER_PRESETS,
} from '../src/uni/scene/jointAttnCutaway';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import { createUniSession } from '../src/uni/session';
import { UNI_ROUTE_G, UNI_ROUTE_U } from '../src/uni/types';
import { pickMacroBay, type PickHit } from '../src/render/pick';
import { makeSheetFrame3D, FRAME_XZ_AXES } from '../src/render/sheetFrame';
import {
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
} from '../src/render/sheets';

function createSharedBundle() {
  return createDefaultUniRuntimeBundle();
}

function getSheetTexRects(scene: ReturnType<typeof buildUniJointAttnScene>) {
  const rects: {
    index: number;
    sheetKind: number;
    u0: number;
    v0: number;
    uWidth: number;
    vHeight: number;
  }[] = [];
  for (let i = 0; i < scene.sheetCount; i++) {
    const base = i * SHEET_INSTANCE_FLOATS;
    rects.push({
      index: i,
      sheetKind: Math.round(scene.sheetData[base + 8]!),
      u0: scene.sheetData[base + 13]!,
      v0: scene.sheetData[base + 14]!,
      uWidth: scene.sheetData[base + 15]!,
      vHeight: scene.sheetData[base + 16]!,
    });
  }
  return rects;
}

describe('UNI4 View 3 (joint_attn): 256x256 RG32F Atlas & Cutaway Scene', () => {
  const bundle = createSharedBundle();

  it('packs 256x256 RG32F atlas (524,288 B, <= 35 rects, 0 overlaps, U25 & finite valid) across L0, L17, L18, L33', () => {
    const session = createUniSession(bundle);
    session.startBranch({
      taskMode: 'image_edit',
      scenePresetId: 'sunlit_cat_16x16',
      prefillMode: 'hf_faithful',
    });
    const denoiseRec = session.stepBranch()!;
    expect(denoiseRec.kind).toBe('image_denoise');

    for (const layerIndex of UNI_JOINT_ATTN_LAYER_PRESETS) {
      const scene = buildUniJointAttnScene(session, {
        layerIndex,
        batchRow: 2,
        selectedQHead: 5,
        selectedTokenRow: 1,
        recordId: denoiseRec.recordId,
      });

      expect(scene.viewMode).toBe('joint_attn');
      expect(scene.layerIndex).toBe(layerIndex);
      expect(scene.rgbPreviews).toHaveLength(0);

      expect(scene.valueAtlas).not.toBeNull();
      expect(scene.valueAtlas!.width).toBe(UNI_JOINT_ATTN_ATLAS_WIDTH);
      expect(scene.valueAtlas!.height).toBe(UNI_JOINT_ATTN_ATLAS_HEIGHT);
      expect(scene.valueAtlas!.data.byteLength).toBe(524_288);

      expect(scene.atlasAudit).not.toBeNull();
      expect(scene.atlasAudit!.byteLength).toBe(524_288);
      expect(scene.atlasAudit!.hasOverlap).toBe(false);
      expect(scene.atlasAudit!.overlapPair).toBeNull();
      expect(scene.atlasAudit!.outOfBoundsIds).toHaveLength(0);
      expect(scene.atlasAudit!.withinBudget).toBe(true);
      expect(scene.atlasAudit!.regions.length).toBeLessThanOrEqual(
        UNI_JOINT_ATTN_MAX_REGIONS,
      );
      expect(scene.atlasRegions.length).toBe(scene.atlasAudit!.regions.length);

      const validation = validateSceneGeometryFinite(scene);
      expect(validation.valid).toBe(true);
      expect(validation.errors).toHaveLength(0);

      // Every causal score/softmax sheet must bind a live RG32F slice
      const sheetRects = getSheetTexRects(scene);
      const causalSheets = sheetRects.filter(
        (s) =>
          s.sheetKind === SHEET_KIND_CAUSAL_SCORES ||
          s.sheetKind === SHEET_KIND_CAUSAL_SOFTMAX,
      );
      expect(causalSheets.length).toBeGreaterThanOrEqual(3);
      for (const cs of causalSheets) {
        expect(cs.uWidth).toBeGreaterThan(0);
        expect(cs.vHeight).toBeGreaterThan(0);
      }
    }
  });

  it('faithfully flips RG32F G-channel mask from cross-modal joint (G=0) at L17 to isolated U⊥G (G=1 across modalities) at L18 on mixed Image Editing prefill', () => {
    const session = createUniSession(bundle);
    const branch = session.startBranch({
      taskMode: 'image_edit',
      scenePresetId: 'sunlit_cat_16x16',
      prefillMode: 'hf_faithful',
    });
    const prefillRecId = branch.recordIds[0]!;
    const ctx = resolveUniSessionContext(session, prefillRecId);

    // Row 2 (`full`) in image_edit prefill has mixed routing: [RADIO-H (U) | 4 sourceVQ (G) | instr (U)]
    const seqLen = branch.prefixCache.seqLen;
    const row2Routing = new Uint8Array(seqLen);
    for (let t = 0; t < seqLen; t++) {
      row2Routing[t] = branch.prefixCache.slabs[t]!.routing[2]!;
    }
    const firstG = row2Routing.indexOf(UNI_ROUTE_G);
    const firstU = row2Routing.indexOf(UNI_ROUTE_U);
    let trailingU = -1;
    for (let t = firstG + 1; t < seqLen; t++) {
      if (row2Routing[t] === UNI_ROUTE_U) {
        trailingU = t;
        break;
      }
    }
    expect(firstU).toBeGreaterThanOrEqual(0);
    expect(firstG).toBeGreaterThan(firstU);
    expect(trailingU).toBeGreaterThan(firstG);

    const trace17 = ctx.resolveLayerTrace(17, 2);
    const trace18 = ctx.resolveLayerTrace(18, 2);

    const pack17 = packUniJointAttnAtlas({
      trace: trace17,
      layerWeights: bundle.core.layers[17]!,
      selectedQHead: 2,
      hiddenIn: ctx.record.perBranchResiduals[2]![17],
    });
    const pack18 = packUniJointAttnAtlas({
      trace: trace18,
      layerWeights: bundle.core.layers[18]!,
      selectedQHead: 2,
      hiddenIn: ctx.record.perBranchResiduals[2]![18],
    });

    const scoresRect17 = pack17.regionMap['scores_selected']!;
    const scoresRect18 = pack18.regionMap['scores_selected']!;
    const softmaxRect17 = pack17.regionMap['softmax_selected']!;
    const softmaxRect18 = pack18.regionMap['softmax_selected']!;

    // 1. G query attending to earlier U key (q = firstG, k = firstU < firstG):
    //    L17 joint: visible (G=0.0); L18 split: masked (G=1.0, R=0.0)
    const gToU_17 = readAtlasTexel(
      pack17.atlas,
      scoresRect17.x0 + firstU,
      scoresRect17.y0 + firstG,
    );
    const gToU_18 = readAtlasTexel(
      pack18.atlas,
      scoresRect18.x0 + firstU,
      scoresRect18.y0 + firstG,
    );
    expect(gToU_17.g).toBe(0.0);
    expect(Number.isFinite(gToU_17.r)).toBe(true);
    expect(gToU_18.g).toBe(1.0);
    expect(gToU_18.r).toBe(0.0);

    // Same transition on softmax sheet
    const gToUSoftmax17 = readAtlasTexel(
      pack17.atlas,
      softmaxRect17.x0 + firstU,
      softmaxRect17.y0 + firstG,
    );
    const gToUSoftmax18 = readAtlasTexel(
      pack18.atlas,
      softmaxRect18.x0 + firstU,
      softmaxRect18.y0 + firstG,
    );
    expect(gToUSoftmax17.g).toBe(0.0);
    expect(gToUSoftmax17.r).toBeGreaterThan(0);
    expect(gToUSoftmax18.g).toBe(1.0);
    expect(gToUSoftmax18.r).toBe(0.0);

    // 2. Trailing U query attending to earlier G key (q = trailingU, k = firstG < trailingU):
    //    L17 joint: visible (G=0.0); L18 split: masked (G=1.0)
    const uToG_17 = readAtlasTexel(
      pack17.atlas,
      scoresRect17.x0 + firstG,
      scoresRect17.y0 + trailingU,
    );
    const uToG_18 = readAtlasTexel(
      pack18.atlas,
      scoresRect18.x0 + firstG,
      scoresRect18.y0 + trailingU,
    );
    expect(uToG_17.g).toBe(0.0);
    expect(uToG_18.g).toBe(1.0);
    expect(uToG_18.r).toBe(0.0);

    // 3. Same-modality pairs stay visible (G=0.0) at both L17 and L18:
    //    G->G (firstG, firstG) and U->U (trailingU, firstU)
    expect(
      readAtlasTexel(pack18.atlas, scoresRect18.x0 + firstG, scoresRect18.y0 + firstG).g,
    ).toBe(0.0);
    expect(
      readAtlasTexel(pack18.atlas, scoresRect18.x0 + firstU, scoresRect18.y0 + trailingU).g,
    ).toBe(0.0);

    // 4. Future causal columns (k > q) remain masked (G=1.0) at both L17 and L18
    expect(
      readAtlasTexel(pack17.atlas, scoresRect17.x0 + trailingU, scoresRect17.y0 + firstU).g,
    ).toBe(1.0);
    expect(
      readAtlasTexel(pack18.atlas, scoresRect18.x0 + trailingU, scoresRect18.y0 + firstU).g,
    ).toBe(1.0);
  });

  it('flips cached U prefix columns from G=0.0 at L17 to G=1.0 masked at L18 on T2I Stage-2 denoising under hf_faithful', () => {
    const session = createUniSession(bundle);
    const branch = session.startBranch({
      taskMode: 'text_to_image',
      prefillMode: 'hf_faithful',
    });
    const denoiseRec = session.stepBranch(branch.branchId)!;
    const pastLen = branch.prefixCache.seqLen;
    expect(pastLen).toBeGreaterThan(0);

    const ctx = resolveUniSessionContext(session, denoiseRec.recordId);
    const trace17 = ctx.resolveLayerTrace(17, 1);
    const trace18 = ctx.resolveLayerTrace(18, 1);

    const pack17 = packUniJointAttnAtlas({
      trace: trace17,
      layerWeights: bundle.core.layers[17]!,
      selectedQHead: 3,
      hiddenIn: ctx.record.perBranchResiduals[1]![17],
    });
    const pack18 = packUniJointAttnAtlas({
      trace: trace18,
      layerWeights: bundle.core.layers[18]!,
      selectedQHead: 3,
      hiddenIn: ctx.record.perBranchResiduals[1]![18],
    });

    const s17 = pack17.regionMap['scores_selected']!;
    const s18 = pack18.regionMap['scores_selected']!;
    const p17 = pack17.regionMap['softmax_selected']!;
    const p18 = pack18.regionMap['softmax_selected']!;

    // At L17 (joint), all cached U prefix columns [0..pastLen-1] and current G columns [pastLen..K-1] are G=0.0
    for (let k = 0; k < pastLen; k++) {
      expect(readAtlasTexel(pack17.atlas, s17.x0 + k, s17.y0).g).toBe(0.0);
      expect(readAtlasTexel(pack17.atlas, p17.x0 + k, p17.y0).g).toBe(0.0);
      // At L18 (split), all cached U prefix columns flip to G=1.0 (masked)
      expect(readAtlasTexel(pack18.atlas, s18.x0 + k, s18.y0).g).toBe(1.0);
      expect(readAtlasTexel(pack18.atlas, p18.x0 + k, p18.y0).g).toBe(1.0);
      expect(readAtlasTexel(pack18.atlas, p18.x0 + k, p18.y0).r).toBe(0.0);
    }

    // Current G columns [pastLen..K-1] remain visible (G=0.0) at both L17 and L18
    for (let k = pastLen; k < pack18.keyTokens; k++) {
      expect(readAtlasTexel(pack17.atlas, s17.x0 + k, s17.y0).g).toBe(0.0);
      expect(readAtlasTexel(pack18.atlas, s18.x0 + k, s18.y0).g).toBe(0.0);
    }
  });

  it('maintains 0-ULP D5 Stage-2 equivalence while distinguishing hf_faithful vs elastic_pruned telemetry and badges', () => {
    const session = createUniSession(bundle);
    const hfBranch = session.startBranch({
      taskMode: 'text_to_image',
      prefillMode: 'hf_faithful',
      seed: 12345,
    });
    const hfPrefillId = hfBranch.recordIds[0]!;
    const prunedBranch = session.forkBranch(hfPrefillId, {
      prefillMode: 'elastic_pruned',
      seed: 12345,
    });

    const hfDenoise = session.stepBranch(hfBranch.branchId)!;
    const prunedDenoise = session.stepBranch(prunedBranch.branchId)!;

    // Inspect Stage-2 at L18
    const hfScene = buildUniJointAttnScene(session, {
      recordId: hfDenoise.recordId,
      layerIndex: 18,
      batchRow: 1,
      selectedQHead: 4,
    });
    const prunedScene = buildUniJointAttnScene(session, {
      recordId: prunedDenoise.recordId,
      layerIndex: 18,
      batchRow: 1,
      selectedQHead: 4,
    });

    expect(hfScene.prefillMode).toBe('hf_faithful');
    expect(prunedScene.prefillMode).toBe('elastic_pruned');
    expect(hfScene.d5Comparison?.stage2UlpDiff).toBe(0);
    expect(prunedScene.d5Comparison?.stage2UlpDiff).toBe(0);
    expect(hfScene.d5Comparison!.prodSavedPrefixKvBytes).toBeGreaterThan(0);

    // 0-ULP Stage-2 output equality in the RG32F valueAtlas across scores, softmax, and layer_out
    const hfLayerOutRect = hfScene.atlasRegions.find((r) => r.id === 'layer_out')!;
    const prLayerOutRect = prunedScene.atlasRegions.find((r) => r.id === 'layer_out')!;
    for (let r = 0; r < hfLayerOutRect.rows; r++) {
      for (let c = 0; c < hfLayerOutRect.cols; c++) {
        const a = readAtlasTexel(hfScene.valueAtlas!, hfLayerOutRect.x0 + c, hfLayerOutRect.y0 + r);
        const b = readAtlasTexel(prunedScene.valueAtlas!, prLayerOutRect.x0 + c, prLayerOutRect.y0 + r);
        expect(a.r).toBe(b.r);
        expect(a.g).toBe(b.g);
      }
    }

    // Badges & calipers clearly distinguish dead/dropped vs pruned prefix KV
    const hfBadgeText = hfScene.provenanceBadges.map((b) => b.label).join(' | ');
    const prunedBadgeText = prunedScene.provenanceBadges.map((b) => b.label).join(' | ');
    expect(hfBadgeText).toContain('hf_faithful');
    expect(hfBadgeText).toContain('dead/dropped');
    expect(prunedBadgeText).toContain('elastic_pruned');
    expect(prunedBadgeText).toContain('pruned');

    // Also confirm inspecting skipped L18 on an elastic_pruned T2I prefill record succeeds cleanly
    const skippedPrefillScene = buildUniJointAttnScene(session, {
      recordId: hfPrefillId,
      layerIndex: 18,
      prefillMode: 'elastic_pruned',
    });
    expect(validateSceneGeometryFinite(skippedPrefillScene).valid).toBe(true);
  });

  it('enforces U25 hard guard on unbound causal sheets and emits YaRN, Llama-4, and 2D SinCos provenance anchors', () => {
    const builder = new UniSheetBufferBuilder();
    expect(() =>
      builder.push({
        cx: 0,
        cy: 0,
        cz: 0,
        ex: 1,
        ey: 0.08,
        ez: 1,
        gridCols: 4,
        gridRows: 4,
        sheetKind: SHEET_KIND_CAUSAL_SCORES,
      }),
    ).toThrow(/U25 violation/);

    expect(() =>
      builder.push({
        cx: 0,
        cy: 0,
        cz: 0,
        ex: 1,
        ey: 0.08,
        ez: 1,
        gridCols: 4,
        gridRows: 4,
        sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
        texRect: [0, 0, 0, 0],
      }),
    ).toThrow(/U25 violation/);

    // Verify packed PickBox frame validation too
    const dummyFrame = makeSheetFrame3D([0, 0, 0], [2, 0.08, 2], FRAME_XZ_AXES.axisU, FRAME_XZ_AXES.axisN);
    const pb = packPickBoxFromFrame(dummyFrame, { bayId: 1, partId: 'test' });
    expect(pb.frame).toBeDefined();

    const session = createUniSession(bundle);
    session.startBranch({ taskMode: 'image_edit', scenePresetId: 'harbor_panorama_16x8' });
    session.stepBranch();
    const scene = buildUniJointAttnScene(session, {
      layerIndex: 18,
      batchRow: 2,
      selectedQHead: 6,
    });

    const badgeText = scene.provenanceBadges.map((b) => b.label).join(' | ');
    expect(badgeText).toContain('original');
    expect(badgeText).toContain('blended');
    expect(badgeText).toContain('interpolated');
    expect(badgeText).toContain('Llama-4');
    expect(badgeText).toContain('1.000');
    expect(badgeText).toContain('1.069');
    expect(badgeText).toContain('2D SinCos');
    expect(badgeText).toContain('canonicalContentSize=32');
    expect(badgeText).toContain('scale=0.0018');
    expect(badgeText).toContain('marker_column = !isSourceEdit');
    expect(badgeText).not.toContain('contentSize=16');
    expect(badgeText).not.toContain('scale=0.25');
    expect(badgeText).not.toContain('U11');

    const anchorIds = scene.overlayAnchors.map((a) => a.id);
    expect(anchorIds).toContain('anchor-yarn-rope');
    expect(anchorIds).toContain('anchor-llama4-scale');
    expect(anchorIds).toContain('anchor-2d-sincos-g');
    expect(anchorIds).toContain('anchor-l17-l18-boundary');
    const sinCosAnchor = scene.overlayAnchors.find((a) => a.id === 'anchor-2d-sincos-g')!;
    expect(sinCosAnchor.label).toContain('canonicalContentSize=32');
    expect(sinCosAnchor.label).toContain('scale=0.0018');
    expect(sinCosAnchor.sublabel).toContain('marker_column = !isSourceEdit');
    expect(sinCosAnchor.label).not.toContain('U11');
  });

  it('round-trips PickBox hits via pickHitToUniCell for layerIndex, isSplitLayer, qHeadIndex, kvHeadIndex, cfgBatchRow, and rotaryPlane', () => {
    const session = createUniSession(bundle);
    session.startBranch({ taskMode: 'image_edit', scenePresetId: 'sunlit_cat_16x16' });
    const denoiseRec = session.stepBranch()!;

    const scene = buildUniJointAttnScene(session, {
      layerIndex: 18,
      batchRow: 2,
      selectedQHead: 6,
      selectedTokenRow: 1,
      recordId: denoiseRec.recordId,
    });

    const qRotBox = scene.pickBoxes.find((pb) => pb.partId === 'q_rot')!;
    expect(qRotBox).toBeDefined();
    expect(qRotBox.headIndex).toBeUndefined();
    expect(qRotBox.kvHeadIndex).toBeUndefined();
    expect(qRotBox.tapIndex).toBeUndefined();

    // Ray-cast across multiple distinct columns of q_rot (64 cols = 8 Q heads × 8 headDim)
    // verifying pickHitToUniCell dynamically infers (qHeadIndex, kvHeadIndex, rotaryPlane) from hitCol
    const cz = (qRotBox.min[2] + qRotBox.max[2]) * 0.5;
    const widthX = qRotBox.max[0] - qRotBox.min[0];
    const probeCols = [
      { col: 2, expectedQHead: 0, expectedKvHead: 0, expectedPlane: 2 },
      { col: 29, expectedQHead: 3, expectedKvHead: 0, expectedPlane: 1 },
      { col: 51, expectedQHead: 6, expectedKvHead: 1, expectedPlane: 3 },
    ];
    for (const probe of probeCols) {
      const x = qRotBox.min[0] + ((probe.col + 0.5) / 64) * widthX;
      const hit: PickHit | null = pickMacroBay(
        { origin: [x, 5, cz], direction: [0, -1, 0] },
        scene.pickBoxes,
        [],
      );
      expect(hit).not.toBeNull();
      expect(hit!.partId).toBe('q_rot');
      expect(hit!.hitCol).toBe(probe.col);

      const picked = pickHitToUniCell(hit, 'joint_attn');
      expect(picked).not.toBeNull();
      expect(picked!.layerIndex).toBe(18);
      expect(picked!.isSplitLayer).toBe(true);
      expect(picked!.qHeadIndex).toBe(probe.expectedQHead);
      expect(picked!.kvHeadIndex).toBe(probe.expectedKvHead);
      expect(picked!.cfgBatchRow).toBe(2);
      expect(picked!.rotaryPlane).toBe(probe.expectedPlane);
    }

    // Also verify k_rot multi-head column inference (16 cols = 2 KV heads × 8 headDim)
    const kRotBox = scene.pickBoxes.find((pb) => pb.partId === 'k_rot')!;
    expect(kRotBox).toBeDefined();
    expect(kRotBox.kvHeadIndex).toBeUndefined();
    const kCz = (kRotBox.min[2] + kRotBox.max[2]) * 0.5;
    const kWidthX = kRotBox.max[0] - kRotBox.min[0];
    for (const probe of [
      { col: 1, expectedKvHead: 0, expectedPlane: 1 },
      { col: 11, expectedKvHead: 1, expectedPlane: 3 },
    ]) {
      const x = kRotBox.min[0] + ((probe.col + 0.5) / 16) * kWidthX;
      const hit = pickMacroBay({ origin: [x, 5, kCz], direction: [0, -1, 0] }, scene.pickBoxes, []);
      expect(hit?.partId).toBe('k_rot');
      const picked = pickHitToUniCell(hit, 'joint_attn')!;
      expect(picked.kvHeadIndex).toBe(probe.expectedKvHead);
      expect(picked.rotaryPlane).toBe(probe.expectedPlane);
      expect(picked.cfgBatchRow).toBe(2);
    }

    // Also test picking a layer L17 joint scores sheet and a CFG-tagged layer_out sheet
    const scene17 = buildUniJointAttnScene(session, {
      layerIndex: 17,
      batchRow: 1,
      selectedQHead: 2,
      recordId: denoiseRec.recordId,
    });
    const scores17Box = scene17.pickBoxes.find((pb) => pb.partId === 'scores_selected')!;
    const hitScores17 = pickMacroBay(
      {
        origin: [
          (scores17Box.min[0] + scores17Box.max[0]) * 0.5,
          5,
          (scores17Box.min[2] + scores17Box.max[2]) * 0.5,
        ],
        direction: [0, -1, 0],
      },
      scene17.pickBoxes,
      [],
    );
    const picked17 = pickHitToUniCell(hitScores17, 'joint_attn');
    expect(picked17).not.toBeNull();
    expect(picked17!.layerIndex).toBe(17);
    expect(picked17!.isSplitLayer).toBe(false);
    expect(picked17!.qHeadIndex).toBe(2);
    expect(picked17!.kvHeadIndex).toBe(0);
    expect(picked17!.cfgBatchRow).toBe(1);
  });
});
