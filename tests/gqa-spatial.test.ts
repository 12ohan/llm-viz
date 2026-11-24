import { describe, expect, it } from 'vitest';

import {
  aabbCorners,
  buildCameraSnapshot,
  makeViewportMetrics,
  projectWorldToInnerCss,
} from '../src/render/camera';
import { pickHitToGqaCell, pickMacroBay, type Ray } from '../src/render/pick';
import {
  SHEET_BASIS_FLOATS,
  SHEET_HL_ACTIVE,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
} from '../src/render/sheets';
import { sheetFrameWorldAabb } from '../src/render/sheetFrame';
import {
  buildGqaCutaway,
  fitSpatialGqaCamera,
  GQA_FIT_BOUNDS,
  GQA_SPATIAL_FIT_BOUNDS,
} from '../src/scene/cutaways';

describe('Stage A+1 · Spatial Increment 1 — GQA 3D Tensor Composition', () => {
  it('builds multi-plane oriented 3D sheets with valid orthonormal bases and smooth 3D ribbons', () => {
    const scene = buildGqaCutaway({
      bayId: 43,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    expect(scene.fitBounds).toEqual(GQA_SPATIAL_FIT_BOUNDS);
    expect(scene.ribbonCount).toBe(0);
    expect(scene.gqaStats?.voxelArrowCount).toBe(0);
    expect(scene.sheetCount).toBeGreaterThan(60);
    expect(scene.sheetData?.length).toBe((scene.sheetCount ?? 0) * SHEET_INSTANCE_FLOATS);
    expect(scene.sheetBasisData?.length).toBe((scene.sheetCount ?? 0) * SHEET_BASIS_FLOATS);

    const basis = scene.sheetBasisData!;
    let hasHorizY = false;
    let hasUprightZ = false;
    let hasStandingX = false;
    let hasRotatedXZ = false;

    for (let i = 0; i < (scene.sheetCount ?? 0); i++) {
      const o = i * SHEET_BASIS_FLOATS;
      const ux = basis[o], uy = basis[o + 1], uz = basis[o + 2];
      const nx = basis[o + 3], ny = basis[o + 4], nz = basis[o + 5];

      const uLen = Math.hypot(ux, uy, uz);
      const nLen = Math.hypot(nx, ny, nz);
      const dot = ux * nx + uy * ny + uz * nz;
      expect(uLen).toBeCloseTo(1, 5);
      expect(nLen).toBeCloseTo(1, 5);
      expect(Math.abs(dot)).toBeLessThan(1e-5);

      if (Math.abs(ny) > 0.9 && Math.abs(ux) > 0.9) hasHorizY = true;
      if (Math.abs(ny) > 0.9 && Math.abs(uz) > 0.9) hasRotatedXZ = true;
      if (Math.abs(nz) > 0.9) hasUprightZ = true;
      if (Math.abs(nx) > 0.9) hasStandingX = true;
    }

    // Meaningful 3D composition uses horizontal XZ, rotated XZ, upright XY fins, and standing YZ weight walls
    expect(hasHorizY).toBe(true);
    expect(hasRotatedXZ).toBe(true);
    expect(hasUprightZ).toBe(true);
    expect(hasStandingX).toBe(true);

    // Smooth 3D ribbons are non-empty and finite
    expect((scene.flowVertices?.length ?? 0)).toBeGreaterThan(100);
    expect((scene.flowIndices?.length ?? 0)).toBeGreaterThan(60);
    for (let i = 0; i < (scene.flowVertices?.length ?? 0); i++) {
      expect(Number.isFinite(scene.flowVertices![i])).toBe(true);
    }
  });

  it('keeps fitBounds invariant and contains every sheet (including lifted Q00..Q31) across T and precision', () => {
    const configs: Array<{
      tokens: number;
      precision: 'bf16' | 'fp8_selective';
      selectedPart: string | null;
    }> = [
      { tokens: 1, precision: 'bf16', selectedPart: null },
      { tokens: 8, precision: 'bf16', selectedPart: 'q-head-0' },
      { tokens: 8, precision: 'fp8_selective', selectedPart: 'q-head-18' },
      { tokens: 16, precision: 'bf16', selectedPart: 'q-head-31' },
      { tokens: 16, precision: 'fp8_selective', selectedPart: 'q-head-15' },
    ];

    for (const cfg of configs) {
      const scene = buildGqaCutaway({
        bayId: 6,
        tokens: cfg.tokens,
        precision: cfg.precision,
        selectedPart: cfg.selectedPart,
        hoveredPart: null,
        viewAngle: 'three_quarter',
      });

      expect(scene.fitBounds).toEqual(GQA_SPATIAL_FIT_BOUNDS);

      for (const box of scene.solidPickBoxes) {
        const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
        expect(aabb.min[0]).toBeGreaterThanOrEqual(GQA_SPATIAL_FIT_BOUNDS.min[0] - 1e-4);
        expect(aabb.min[1]).toBeGreaterThanOrEqual(GQA_SPATIAL_FIT_BOUNDS.min[1] - 1e-4);
        expect(aabb.min[2]).toBeGreaterThanOrEqual(GQA_SPATIAL_FIT_BOUNDS.min[2] - 1e-4);
        expect(aabb.max[0]).toBeLessThanOrEqual(GQA_SPATIAL_FIT_BOUNDS.max[0] + 1e-4);
        expect(aabb.max[1]).toBeLessThanOrEqual(GQA_SPATIAL_FIT_BOUNDS.max[1] + 1e-4);
        expect(aabb.max[2]).toBeLessThanOrEqual(GQA_SPATIAL_FIT_BOUNDS.max[2] + 1e-4);
      }
    }
  });

  it('projects all 8 fitBounds corners strictly inside desktop and narrow viewports with high canvas utilization', () => {
    const viewports = [
      { name: 'desktop-1440x900-stage', w: 1076, h: 900, top: 84, bottom: 118, isNarrow: false },
      { name: 'desktop-1280x800-stage', w: 920, h: 800, top: 84, bottom: 118, isNarrow: false },
      { name: 'narrow-mobile-stage', w: 390, h: 360, top: 12, bottom: 12, isNarrow: true },
    ];

    const corners = aabbCorners(GQA_SPATIAL_FIT_BOUNDS);

    for (const vp of viewports) {
      const metrics = makeViewportMetrics(0, 0, vp.w, vp.h, 1);
      const pose = fitSpatialGqaCamera(GQA_SPATIAL_FIT_BOUNDS, metrics, {
        topInsetPx: vp.top,
        bottomInsetPx: vp.bottom,
        isNarrow: vp.isNarrow,
      });
      const snap = buildCameraSnapshot(pose, GQA_SPATIAL_FIT_BOUNDS, metrics);

      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;

      for (const c of corners) {
        const pt = projectWorldToInnerCss(c, snap);
        expect(pt.visible).toBe(true);
        expect(pt.x).toBeGreaterThanOrEqual(0);
        expect(pt.x).toBeLessThanOrEqual(vp.w);
        expect(pt.y).toBeGreaterThanOrEqual(vp.top - 1);
        expect(pt.y).toBeLessThanOrEqual(vp.h - vp.bottom + 1);
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
      }

      // High horizontal utilization across desktop and narrow viewports
      const spanFracX = (maxX - minX) / vp.w;
      const spanFracY = (maxY - minY) / (vp.h - vp.top - vp.bottom);
      expect(Math.max(spanFracX, spanFracY)).toBeGreaterThan(0.74);
    }
  });

  it('keeps Shared K and Shared V explicitly separate along X with 2*T slices each', () => {
    const T = 8;
    const scene = buildGqaCutaway({
      bayId: 6,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    const kBoxes = scene.solidPickBoxes.filter((b) => b.tensorKey === 'kv_k');
    const vBoxes = scene.solidPickBoxes.filter((b) => b.tensorKey === 'kv_v');
    const scoreBoxes = scene.solidPickBoxes.filter((b) => b.tensorKey === 'scores');
    const softBoxes = scene.solidPickBoxes.filter((b) => b.tensorKey === 'softmax');

    expect(kBoxes.length).toBe(2 * T);
    expect(vBoxes.length).toBe(2 * T);
    expect(scoreBoxes.length).toBe(2);
    expect(softBoxes.length).toBe(2);

    const xK = kBoxes[0].frame!.center[0];
    const xScores = scoreBoxes[0].frame!.center[0];
    const xSoft = softBoxes[0].frame!.center[0];
    const xV = vBoxes[0].frame!.center[0];

    // Explicit mathematical ordering: K feeds Scores, Softmax + V feed V_out
    expect(xK).toBeLessThan(xScores);
    expect(xScores).toBeLessThan(xSoft);
    expect(xSoft).toBeLessThan(xV);
  });

  it('exposes Q18 -> KV1 via +Y spatial lift, highlights KV1 (not KV0), and supports front/back cell parity', () => {
    const scene = buildGqaCutaway({
      bayId: 43,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-18',
      hoveredPart: null,
      activeGroup0Head: 0,
      activeGroup1Head: 18,
      viewAngle: 'three_quarter',
    });

    const q18Box = scene.solidPickBoxes.find((b) => b.tensorKey === 'q' && b.headIndex === 18);
    const q19Box = scene.solidPickBoxes.find((b) => b.tensorKey === 'q' && b.headIndex === 19);
    expect(q18Box?.frame).toBeDefined();
    expect(q19Box?.frame).toBeDefined();

    // Q18 is lifted prominently above resting stack
    expect(q18Box!.frame!.center[1]).toBeGreaterThan(1.5);
    expect(q18Box!.frame!.center[1] - q19Box!.frame!.center[1]).toBeGreaterThan(0.9);

    // Front (+Z) and back (-Z) rays aimed at the exact same physical point on lifted Q18
    // hit the identical logical cell (row, col) and Float32 value
    const f = q18Box!.frame!;
    const samplePt: readonly [number, number, number] = [
      f.center[0] + 0.14 * f.extent[0],
      f.center[1] - 0.18 * f.extent[2],
      f.center[2],
    ];
    const frontRay: Ray = {
      origin: [samplePt[0], samplePt[1], samplePt[2] + 3],
      direction: [0, 0, -1],
    };
    const backRay: Ray = {
      origin: [samplePt[0], samplePt[1], samplePt[2] - 3],
      direction: [0, 0, 1],
    };

    const frontHit = pickMacroBay(frontRay, [q18Box!], []);
    const backHit = pickMacroBay(backRay, [q18Box!], []);
    expect(frontHit).not.toBeNull();
    expect(backHit).not.toBeNull();
    expect(frontHit!.hitRow).toBe(backHit!.hitRow);
    expect(frontHit!.hitCol).toBe(backHit!.hitCol);

    const cellFront = pickHitToGqaCell(frontHit!, 0, 18);
    const cellBack = pickHitToGqaCell(backHit!, 0, 18);
    expect(cellFront).toEqual(cellBack);
    expect(cellFront?.headIndex).toBe(18);

    // Verify Q18 sheet has ACTIVE highlight, KV1 slices have PATHWAY, and KV0 slices do NOT
    const sheetData = scene.sheetData!;
    const q18Idx = scene.solidPickBoxes.findIndex((b) => b.tensorKey === 'q' && b.headIndex === 18);
    const kv1Idx = scene.solidPickBoxes.findIndex((b) => b.tensorKey === 'kv_k' && b.kvHeadIndex === 1);
    const kv0Idx = scene.solidPickBoxes.findIndex((b) => b.tensorKey === 'kv_k' && b.kvHeadIndex === 0);
    expect(sheetData[q18Idx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_ACTIVE);
    expect(sheetData[kv1Idx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_PATHWAY);
    expect(sheetData[kv0Idx * SHEET_INSTANCE_FLOATS + 11]).not.toBe(SHEET_HL_PATHWAY);
    expect(sheetData[kv0Idx * SHEET_INSTANCE_FLOATS + 11]).not.toBe(SHEET_HL_ACTIVE);

    // Accurately head-labeled Scores & Softmax landmarks
    const scoresLabel = scene.landmarks.find((l) => l.id === 'g-scores')?.label ?? '';
    const softLabel = scene.landmarks.find((l) => l.id === 'g-softmax')?.label ?? '';
    expect(scoresLabel).toContain('Q18');
    expect(scoresLabel).toContain('K₁');
    expect(softLabel).toContain('Q18');
  });

  it('preserves Near-Plan View schematic bounds and identical formula readouts across mode switches', () => {
    const pinnedCell = {
      tensor: 'scores' as const,
      row: 4,
      col: 2,
      headIndex: 18,
      kvHeadIndex: 1 as const,
    };

    const spatial = buildGqaCutaway({
      bayId: 43,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-18',
      hoveredPart: null,
      activeGroup0Head: 0,
      activeGroup1Head: 18,
      activeCell: pinnedCell,
      viewAngle: 'three_quarter',
    });

    const nearPlan = buildGqaCutaway({
      bayId: 43,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-18',
      hoveredPart: null,
      activeGroup0Head: 0,
      activeGroup1Head: 18,
      activeCell: pinnedCell,
      viewAngle: 'near_plan',
    });

    expect(nearPlan.fitBounds).toEqual(GQA_FIT_BOUNDS);
    expect(nearPlan.sheetBasisData).toBeUndefined();
    expect(spatial.fitBounds).toEqual(GQA_SPATIAL_FIT_BOUNDS);
    expect(spatial.sheetBasisData).toBeDefined();

    // Identical mathematical inspector output & part catalog across Spatial <-> Near-Plan
    expect(spatial.activeFormula).toEqual(nearPlan.activeFormula);
    expect(Object.keys(spatial.partCatalog).sort()).toEqual(Object.keys(nearPlan.partCatalog).sort());
  });
});
