import { describe, expect, it } from 'vitest';

import { runMambaSim, type MambaCellRef } from '../src/math/mambaSim';
import {
  advanceMambaTemporalPlaybackTick,
  resolveEffectiveMambaCell,
  resolveMambaHoverDuringPlayback,
} from '../src/model/mambaPlayback';
import { createInitialState, createPerspectivePose } from '../src/model/state';
import {
  aabbCorners,
  buildCameraSnapshot,
  makeViewportMetrics,
  projectWorldToInnerCss,
} from '../src/render/camera';
import { pickMacroBay, type Ray } from '../src/render/pick';
import { pickHitToMambaCell } from '../src/scene/hybridPick';
import {
  SHEET_BASIS_FLOATS,
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
} from '../src/render/sheets';
import { sheetFrameWorldAabb } from '../src/render/sheetFrame';
import {
  buildMambaCutaway,
  fitSpatialMambaCamera,
  MAMBA_FIT_BOUNDS,
  MAMBA_SPATIAL_FIT_BOUNDS,
} from '../src/scene/cutaways';
import { SOLID_INSTANCE_FLOATS } from '../src/scene/spine';
import { computeEditorialCacheKey } from '../src/ui/controls';

function sheetHighlightAt(sheetData: Float32Array, idx: number): number {
  return sheetData[idx * SHEET_INSTANCE_FLOATS + 11];
}

function solidFp8FlagAt(solidData: Float32Array, idx: number): number {
  return solidData[idx * SOLID_INSTANCE_FLOATS + 7];
}

function solidPhaseAt(solidData: Float32Array, idx: number): number {
  return solidData[idx * SOLID_INSTANCE_FLOATS + 8];
}

describe('Spatial Mamba-2 3D Multi-Plane Cutaway & Temporal Recurrence Playback', () => {
  it('1. builds multi-plane oriented 3D sheets with valid right-handed orthonormal bases and smooth 3D ribbons', () => {
    const scene = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 3,
      viewAngle: 'three_quarter',
    });

    expect(scene.fitBounds).toEqual(MAMBA_SPATIAL_FIT_BOUNDS);
    expect(scene.ribbonCount).toBe(0);
    expect(scene.mambaStats?.voxelArrowCount).toBe(0);
    expect(scene.sheetCount).toBeGreaterThan(95);
    expect(scene.sheetData?.length).toBe((scene.sheetCount ?? 0) * SHEET_INSTANCE_FLOATS);
    expect(scene.sheetBasisData?.length).toBe((scene.sheetCount ?? 0) * SHEET_BASIS_FLOATS);
    expect(scene.solidCount).toBe(scene.sheetCount);

    const basis = scene.sheetBasisData!;
    let hasHorizXZ = false;
    let hasHorizChZ = false;
    let hasUprightXY = false;
    let hasStandingYZ = false;

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

      // Right-handed V = U x N has unit length
      const vx = uy * nz - uz * ny;
      const vy = uz * nx - ux * nz;
      const vz = ux * ny - uy * nx;
      expect(Math.hypot(vx, vy, vz)).toBeCloseTo(1, 5);

      if (Math.abs(ny) > 0.9 && Math.abs(ux) > 0.9) hasHorizXZ = true;
      if (Math.abs(ny) > 0.9 && Math.abs(uz) > 0.9) hasHorizChZ = true;
      if (Math.abs(nz) > 0.9) hasUprightXY = true;
      if (Math.abs(nx) > 0.9) hasStandingYZ = true;
    }

    // Meaningful 3D composition: horizontal decks, channel-aligned XZ decks, upright XY x-head fins, standing YZ weight walls
    expect(hasHorizXZ).toBe(true);
    expect(hasHorizChZ).toBe(true);
    expect(hasUprightXY).toBe(true);
    expect(hasStandingYZ).toBe(true);

    // Upright W_in and W_out walls stand above the ground deck, and H_prev/H_curr sit on the raised fixed pedestal
    const wInBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'w_in')!;
    const wOutBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'w_out')!;
    const hPrevBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'h_prev')!;
    const hCurrBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'h_curr')!;
    const activeXBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'x' && b.headIndex === 3)!;
    const restingXBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'x' && b.headIndex === 0)!;

    expect(wInBox.frame!.center[1]).toBeGreaterThan(1.0);
    expect(wOutBox.frame!.center[1]).toBeGreaterThan(1.0);
    expect(hPrevBox.frame!.center[1]).toBeGreaterThan(0.9);
    expect(hCurrBox.frame!.center[1]).toBeGreaterThan(0.9);
    expect(activeXBox.frame!.center[1] - restingXBox.frame!.center[1]).toBeGreaterThan(0.8);

    // Smooth 3D flow ribbons are finite and non-empty
    expect(scene.flowVertices?.length ?? 0).toBeGreaterThan(100);
    expect(scene.flowIndices?.length ?? 0).toBeGreaterThan(60);
    for (let i = 0; i < (scene.flowVertices?.length ?? 0); i++) {
      expect(Number.isFinite(scene.flowVertices![i])).toBe(true);
    }
  });

  it('2. keeps fitBounds, solidCount, Conv1D (4 taps), H_prev/H_curr, and 64 SSM bank tiles strictly FIXED while sequence decks grow T=1..16', () => {
    const t1 = buildMambaCutaway({
      bayId: 1,
      tokens: 1,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      viewAngle: 'three_quarter',
    });
    const t8 = buildMambaCutaway({
      bayId: 42,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: 'mamba-head-7',
      hoveredPart: null,
      activeHead: 7,
      viewAngle: 'three_quarter',
    });
    const t16 = buildMambaCutaway({
      bayId: 1,
      tokens: 16,
      precision: 'fp8_selective',
      selectedPart: 'mamba-head-15',
      hoveredPart: null,
      activeHead: 15,
      viewAngle: 'three_quarter',
    });

    // Invariant envelope and sheet/solid counts across T=1, 8, 16, Bay 01/42, BF16/FP8, and lifted head states
    for (const sc of [t1, t8, t16]) {
      expect(sc.fitBounds).toEqual(MAMBA_SPATIAL_FIT_BOUNDS);
      expect(sc.solidCount).toBe(t1.solidCount);
      expect(sc.sheetCount).toBe(t1.sheetCount);
      for (const box of sc.solidPickBoxes) {
        const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
        expect(aabb.min[0]).toBeGreaterThanOrEqual(MAMBA_SPATIAL_FIT_BOUNDS.min[0] - 1e-4);
        expect(aabb.min[1]).toBeGreaterThanOrEqual(MAMBA_SPATIAL_FIT_BOUNDS.min[1] - 1e-4);
        expect(aabb.min[2]).toBeGreaterThanOrEqual(MAMBA_SPATIAL_FIT_BOUNDS.min[2] - 1e-4);
        expect(aabb.max[0]).toBeLessThanOrEqual(MAMBA_SPATIAL_FIT_BOUNDS.max[0] + 1e-4);
        expect(aabb.max[1]).toBeLessThanOrEqual(MAMBA_SPATIAL_FIT_BOUNDS.max[1] + 1e-4);
        expect(aabb.max[2]).toBeLessThanOrEqual(MAMBA_SPATIAL_FIT_BOUNDS.max[2] + 1e-4);
      }
    }

    // FIXED persistent state: Conv1D (4 taps), H_prev, H_curr, and 64 SSM bank tiles have IDENTICAL world extents T=1 vs T=16
    const convT1 = t1.solidPickBoxes.filter((b) => b.partId === 'conv1d');
    const convT16 = t16.solidPickBoxes.filter((b) => b.partId === 'conv1d');
    expect(convT1).toHaveLength(4);
    expect(convT16).toHaveLength(4);
    for (let s = 0; s < 4; s++) {
      expect(convT1[s].frame!.extent).toEqual(convT16[s].frame!.extent);
    }

    const hPrevT1 = t1.solidPickBoxes.find((b) => b.tensorKey === 'h_prev')!;
    const hPrevT16 = t16.solidPickBoxes.find((b) => b.tensorKey === 'h_prev')!;
    const hCurrT1 = t1.solidPickBoxes.find((b) => b.tensorKey === 'h_curr')!;
    const hCurrT16 = t16.solidPickBoxes.find((b) => b.tensorKey === 'h_curr')!;
    expect(hPrevT1.frame!.extent).toEqual(hPrevT16.frame!.extent);
    expect(hCurrT1.frame!.extent).toEqual(hCurrT16.frame!.extent);

    const bankT1 = t1.solidPickBoxes.filter((b) => b.partId === 'ssm-state');
    const bankT16 = t16.solidPickBoxes.filter((b) => b.partId === 'ssm-state');
    expect(bankT1).toHaveLength(64);
    expect(bankT16).toHaveLength(64);
    expect(bankT1[0].frame!.extent).toEqual(bankT16[0].frame!.extent);

    // GROWING sequence activation decks strictly lengthen from T=1 -> T=16
    const growingKeys = ['input', 'rmsnorm', 'xbc_pre', 'z', 'dt', 'b', 'c', 'y', 'y_norm', 'residual'] as const;
    for (const key of growingKeys) {
      const box1 = t1.solidPickBoxes.find((b) => b.tensorKey === key)!;
      const box16 = t16.solidPickBoxes.find((b) => b.tensorKey === key)!;
      // Extent[2] is always the token-row axis (axisV) on these sequence decks
      expect(box16.frame!.extent[2]).toBeGreaterThan(box1.frame!.extent[2] * 2.0);
    }
  });

  it('3. projects all 8 MAMBA_SPATIAL_FIT_BOUNDS corners strictly inside desktop (1440x900) and narrow (1024x800) viewports', () => {
    const viewports = [
      { name: 'desktop-1440x900-stage', w: 1120, h: 900, top: 84, bottom: 118, isNarrow: false },
      { name: 'narrow-desktop-1024x800-stage', w: 720, h: 800, top: 84, bottom: 118, isNarrow: false },
      { name: 'mobile-stacked-stage', w: 390, h: 360, top: 12, bottom: 12, isNarrow: true },
    ];

    const corners = aabbCorners(MAMBA_SPATIAL_FIT_BOUNDS);
    for (const vp of viewports) {
      const metrics = makeViewportMetrics(0, 0, vp.w, vp.h, 1);
      const pose = fitSpatialMambaCamera(MAMBA_SPATIAL_FIT_BOUNDS, metrics, {
        topInsetPx: vp.top,
        bottomInsetPx: vp.bottom,
        isNarrow: vp.isNarrow,
      });
      const snap = buildCameraSnapshot(pose, MAMBA_SPATIAL_FIT_BOUNDS, metrics);

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

      const spanFracX = (maxX - minX) / vp.w;
      const spanFracY = (maxY - minY) / (vp.h - vp.top - vp.bottom);
      expect(Math.max(spanFracX, spanFracY)).toBeGreaterThan(0.72);
    }
  });

  it('4. enforces full-identity operand highlighting/dimming on H_t[H00] and front/back 3D picking parity on lifted x head fin', () => {
    const scene = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      activeCell: {
        tensor: 'h_curr',
        row: 1,
        col: 2,
        headIndex: 0,
        groupIndex: 0,
        tokenIndex: 7,
      },
      viewAngle: 'three_quarter',
    });

    const sheetData = scene.sheetData!;
    const idxOf = (pred: (b: (typeof scene.solidPickBoxes)[number]) => boolean) =>
      scene.solidPickBoxes.findIndex(pred);

    // Target H_curr active, upstream operands (H_prev, dt, x[H00], B) operand-highlighted
    expect(sheetHighlightAt(sheetData, idxOf((b) => b.tensorKey === 'h_curr'))).toBe(SHEET_HL_ACTIVE);
    expect(sheetHighlightAt(sheetData, idxOf((b) => b.tensorKey === 'h_prev'))).toBe(SHEET_HL_OPERAND_A);
    expect(sheetHighlightAt(sheetData, idxOf((b) => b.tensorKey === 'dt'))).toBe(SHEET_HL_OPERAND_A);
    expect(sheetHighlightAt(sheetData, idxOf((b) => b.tensorKey === 'x' && b.headIndex === 0))).toBe(SHEET_HL_OPERAND_B);
    expect(sheetHighlightAt(sheetData, idxOf((b) => b.tensorKey === 'b'))).toBe(SHEET_HL_OPERAND_B);

    // Unrelated heads H01..H15 and unrelated stages (W_in, Gate z, C, W_out, Residual) are dimmed
    for (let h = 1; h < 16; h++) {
      const idx = idxOf((b) => b.tensorKey === 'x' && b.headIndex === h);
      expect(sheetHighlightAt(sheetData, idx)).toBe(SHEET_HL_DIMMED);
    }
    for (const key of ['input', 'rmsnorm', 'w_in', 'z', 'c', 'y', 'y_norm', 'w_out', 'residual'] as const) {
      expect(sheetHighlightAt(sheetData, idxOf((b) => b.tensorKey === key))).toBe(SHEET_HL_DIMMED);
    }

    // 64-head bank: only H00 tiles highlight as pathway, other heads dim
    for (let i = 0; i < scene.solidPickBoxes.length; i++) {
      const box = scene.solidPickBoxes[i];
      if (box.partId !== 'ssm-state') continue;
      if (box.headIndex === 0) {
        expect(sheetHighlightAt(sheetData, i)).toBe(SHEET_HL_PATHWAY);
      } else {
        expect(sheetHighlightAt(sheetData, i)).toBe(SHEET_HL_DIMMED);
      }
    }

    // Front (+Z) and back (-Z) rays aimed at the exact same point on lifted x[H00] fin yield identical cell
    const x0Box = scene.solidPickBoxes.find((b) => b.tensorKey === 'x' && b.headIndex === 0)!;
    const f = x0Box.frame!;
    const pt: readonly [number, number, number] = [
      f.center[0] + 0.12 * f.extent[0],
      f.center[1] - 0.14 * f.extent[2],
      f.center[2],
    ];
    const frontRay: Ray = { origin: [pt[0], pt[1], pt[2] + 3], direction: [0, 0, -1] };
    const backRay: Ray = { origin: [pt[0], pt[1], pt[2] - 3], direction: [0, 0, 1] };
    const frontHit = pickMacroBay(frontRay, [x0Box], []);
    const backHit = pickMacroBay(backRay, [x0Box], []);
    expect(frontHit).not.toBeNull();
    expect(backHit).not.toBeNull();
    expect(frontHit!.hitRow).toBe(backHit!.hitRow);
    expect(frontHit!.hitCol).toBe(backHit!.hitCol);
    expect(pickHitToMambaCell(frontHit!, 0, 8)).toEqual(pickHitToMambaCell(backHit!, 0, 8));
  });

  it('5. temporal recurrence playback (stateTokenIndex = 0, 2, 7) updates Conv1D zero-padding & H_t atlas in place without changing geometry extents', () => {
    const sc0 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      stateTokenIndex: 0,
      viewAngle: 'three_quarter',
    });
    const sc2 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      stateTokenIndex: 2,
      viewAngle: 'three_quarter',
    });
    const sc7 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      stateTokenIndex: 7,
      viewAngle: 'three_quarter',
    });

    expect(sc0.mambaStats?.stateTokenIndex).toBe(0);
    expect(sc2.mambaStats?.stateTokenIndex).toBe(2);
    expect(sc7.mambaStats?.stateTokenIndex).toBe(7);

    // Geometry counts & fixed pedestal extents remain strictly invariant across t = 0, 2, 7
    expect(sc0.solidCount).toBe(sc7.solidCount);
    expect(sc2.solidCount).toBe(sc7.solidCount);

    // At t = 0, Conv1D taps 0, 1, 2 are zero-padded (dimmed in idle BF16) while tap 3 holds live t0!
    const convIdx = (sc: typeof sc0, slot: number) =>
      sc.solidPickBoxes.findIndex((b) => b.tensorKey === 'conv_buf' && b.tapIndex === slot);
    expect(sheetHighlightAt(sc0.sheetData!, convIdx(sc0, 0))).toBe(SHEET_HL_DIMMED);
    expect(sheetHighlightAt(sc0.sheetData!, convIdx(sc0, 1))).toBe(SHEET_HL_DIMMED);
    expect(sheetHighlightAt(sc0.sheetData!, convIdx(sc0, 2))).toBe(SHEET_HL_DIMMED);
    expect(sheetHighlightAt(sc0.sheetData!, convIdx(sc0, 3))).not.toBe(SHEET_HL_DIMMED);

    // At t = 2, only tap 0 is zero-padded; taps 1, 2, 3 hold t0, t1, t2
    expect(sheetHighlightAt(sc2.sheetData!, convIdx(sc2, 0))).toBe(SHEET_HL_DIMMED);
    expect(sheetHighlightAt(sc2.sheetData!, convIdx(sc2, 1))).not.toBe(SHEET_HL_DIMMED);

    // At t = 7, all 4 taps are populated (none zero-padded)
    for (let s = 0; s < 4; s++) {
      expect(sheetHighlightAt(sc7.sheetData!, convIdx(sc7, s))).not.toBe(SHEET_HL_DIMMED);
    }

    // In FP8 Selective at t = 1: leading 2 zero-padded taps dim while populated taps (slots 2 & 3) show BF16 gold
    const sc1Fp8 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      stateTokenIndex: 1,
      viewAngle: 'three_quarter',
    });
    expect(sheetHighlightAt(sc1Fp8.sheetData!, convIdx(sc1Fp8, 0))).toBe(SHEET_HL_DIMMED);
    expect(sheetHighlightAt(sc1Fp8.sheetData!, convIdx(sc1Fp8, 1))).toBe(SHEET_HL_DIMMED);
    expect(sheetHighlightAt(sc1Fp8.sheetData!, convIdx(sc1Fp8, 2))).toBe(SHEET_HL_BF16_GOLD);
    expect(sheetHighlightAt(sc1Fp8.sheetData!, convIdx(sc1Fp8, 3))).toBe(SHEET_HL_BF16_GOLD);

    // Atlas values for H_prev/H_curr & SSM bank phase shift cleanly in place from t=0 -> t=7
    let atlasDiff = 0;
    const d0 = sc0.valueAtlas!.data;
    const d7 = sc7.valueAtlas!.data;
    for (let i = 0; i < d0.length; i++) {
      atlasDiff = Math.max(atlasDiff, Math.abs(d0[i] - d7[i]));
    }
    expect(atlasDiff).toBeGreaterThan(0.05);

    const bankIdx0 = sc0.solidPickBoxes.findIndex((b) => b.partId === 'ssm-state');
    const bankIdx7 = sc7.solidPickBoxes.findIndex((b) => b.partId === 'ssm-state');
    expect(solidPhaseAt(sc0.solidData, bankIdx0)).not.toBeCloseTo(solidPhaseAt(sc7.solidData, bankIdx7), 3);

    // Landmark pills display explicit token tag during temporal playback
    expect(sc2.landmarks.find((l) => l.id === 'm-hcurr')?.label).toContain('t2');
    expect(sc2.landmarks.find((l) => l.id === 'm-conv1d')?.label).toContain('t2');
  });

  it('6. routes Step 6 Residual carry-in badge without expanding opFocusBounds[5], preserves Bay 01 vs Bay 42 precision & Near-Plan View parity', () => {
    const resCell = { tensor: 'residual' as const, row: 7, col: 2 };
    const bay01Spatial = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: resCell,
      viewAngle: 'three_quarter',
    });
    const bay01NearPlan = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: resCell,
      viewAngle: 'near_plan',
    });
    const bay42Spatial = buildMambaCutaway({
      bayId: 42,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    // Step 6 Residual carry-in badge is emitted with signed Input X value while opFocusBounds[5] stays focused on output stages (excluding Input X)
    const carryLm = bay01Spatial.landmarks.find((l) => l.id === 'm-res-carry');
    expect(carryLm).toBeDefined();
    expect(carryLm!.label).toContain('← Input X[2, t7]');

    const inputBox = bay01Spatial.solidPickBoxes.find((b) => b.tensorKey === 'input')!;
    const step5Bounds = bay01Spatial.opFocusBounds![5];
    expect(step5Bounds.min[0]).toBeGreaterThan(inputBox.max[0]);
    for (let s = 0; s < 6; s++) {
      const ob = bay01Spatial.opFocusBounds![s];
      expect(ob).toBeDefined();
      const spanX = ob.max[0] - ob.min[0];
      const fullSpanX = MAMBA_SPATIAL_FIT_BOUNDS.max[0] - MAMBA_SPATIAL_FIT_BOUNDS.min[0];
      expect(spanX).toBeLessThan(fullSpanX);
    }

    // Bay 01 FP8 Selective: W_in & W_out have fp8Flag=1, while Conv1D (4 taps) & SSM bank (64) stay fp8Flag=0
    const win01 = bay01Spatial.solidPickBoxes.findIndex((b) => b.partId === 'in-proj');
    const wout01 = bay01Spatial.solidPickBoxes.findIndex((b) => b.partId === 'out-proj');
    expect(solidFp8FlagAt(bay01Spatial.solidData, win01)).toBe(1);
    expect(solidFp8FlagAt(bay01Spatial.solidData, wout01)).toBe(1);
    for (let i = 0; i < bay01Spatial.solidPickBoxes.length; i++) {
      const pid = bay01Spatial.solidPickBoxes[i].partId;
      if (pid === 'conv1d' || pid === 'ssm-state') {
        expect(solidFp8FlagAt(bay01Spatial.solidData, i)).toBe(0);
      }
    }

    // Bay 42 Feeder FP8 Selective: W_in, W_out, and Conv1D preserve BF16 (fp8Flag=0, BF16 gold rim in idle view)
    const win42 = bay42Spatial.solidPickBoxes.findIndex((b) => b.partId === 'in-proj');
    const wout42 = bay42Spatial.solidPickBoxes.findIndex((b) => b.partId === 'out-proj');
    const conv42 = bay42Spatial.solidPickBoxes.findIndex((b) => b.partId === 'conv1d');
    expect(solidFp8FlagAt(bay42Spatial.solidData, win42)).toBe(0);
    expect(solidFp8FlagAt(bay42Spatial.solidData, wout42)).toBe(0);
    expect(sheetHighlightAt(bay42Spatial.sheetData!, win42)).toBe(SHEET_HL_BF16_GOLD);
    expect(sheetHighlightAt(bay42Spatial.sheetData!, wout42)).toBe(SHEET_HL_BF16_GOLD);
    expect(sheetHighlightAt(bay42Spatial.sheetData!, conv42)).toBe(SHEET_HL_BF16_GOLD);
    expect(bay42Spatial.landmarks.some((l) => l.label.includes('Bay 42 Feeder'))).toBe(true);

    // Near-Plan View parity: untouched bounds, undefined sheetBasisData, identical formula & part catalog
    expect(bay01NearPlan.fitBounds).toEqual(MAMBA_FIT_BOUNDS);
    expect(bay01NearPlan.sheetBasisData).toBeUndefined();
    expect(bay01Spatial.activeFormula).toEqual(bay01NearPlan.activeFormula);
    expect(Object.keys(bay01Spatial.partCatalog).sort()).toEqual(
      Object.keys(bay01NearPlan.partCatalog).sort(),
    );
  });

  it('7. enforces explicit hover vs. temporal playback policy so stale hover never freezes advancing snapshots', () => {
    const T = 8;
    const sim = runMambaSim(T);

    // 7A. Hovering fixed-size state (H_t / H_{t-1} / Conv1D) during playback keeps playback running
    // and synchronizes the hovered cell's token with the advancing temporal snapshot!
    const hoveredHtAt4: MambaCellRef = {
      tensor: 'h_curr',
      row: 0,
      col: 0,
      headIndex: 0,
      groupIndex: 0,
      tokenIndex: 4,
    };
    const hoverFixed = resolveMambaHoverDuringPlayback(hoveredHtAt4, true, 4, T);
    expect(hoverFixed.shouldPauseTemporalPlayback).toBe(false);
    expect(hoverFixed.hoveredCell?.tokenIndex).toBe(4);

    // Advance temporal tick from t=4 -> t=5 while cursor stays stationary on H_t
    const tick = advanceMambaTemporalPlaybackTick(4, T, null, hoverFixed.hoveredCell);
    expect(tick.nextToken).toBe(5);
    expect(tick.nextHoveredCell?.tensor).toBe('h_curr');
    expect(tick.nextHoveredCell?.tokenIndex).toBe(5);

    // Structural guard: even if a stale hovered H_t cell (tokenIndex=4) reaches resolveEffectiveMambaCell
    // while mambaTemporalPlaying is true at mambaStateToken=6, it remaps cleanly to t=6!
    const effCell = resolveEffectiveMambaCell({
      sim,
      tokens: T,
      activeHead: 0,
      mambaStateToken: 6,
      mambaTemporalPlaying: true,
      mambaHoveredCell: hoveredHtAt4,
      mambaSelectedCell: null,
      mambaStepperActive: false,
      mambaStepperStep: 0,
      mambaStepperSubProgress: 0,
    });
    expect(effCell?.tokenIndex).toBe(6);

    const sc4 = buildMambaCutaway({
      bayId: 1,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'ssm-state-curr',
      activeHead: 0,
      activeCell: hoveredHtAt4,
      stateTokenIndex: 4,
      viewAngle: 'three_quarter',
    });
    const sc6 = buildMambaCutaway({
      bayId: 1,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'ssm-state-curr',
      activeHead: 0,
      activeCell: effCell,
      stateTokenIndex: 6,
      viewAngle: 'three_quarter',
    });
    expect(sc4.mambaStats?.stateTokenIndex).toBe(4);
    expect(sc6.mambaStats?.stateTokenIndex).toBe(6);
    expect(sc6.activeFormula?.coordLabel).toContain('t6');
    expect(sc6.activeFormula?.equation).not.toBe(sc4.activeFormula?.equation);

    // 7B. Hovering token-independent weights (W_in / W_out) during playback keeps playback running
    const hoverWeight = resolveMambaHoverDuringPlayback(
      { tensor: 'w_in', row: 0, col: 10 },
      true,
      5,
      T,
    );
    expect(hoverWeight.shouldPauseTemporalPlayback).toBe(false);

    // 7C. Hovering a specific sequence-row tensor (Input X, x[H00], B, etc.) during playback cleanly
    // pauses playback so cursor inspection owns the displayed token without timer conflict
    const hoverSeq = resolveMambaHoverDuringPlayback(
      { tensor: 'input', row: 2, col: 3, tokenIndex: 2 },
      true,
      5,
      T,
    );
    expect(hoverSeq.shouldPauseTemporalPlayback).toBe(true);
    expect(hoverSeq.hoveredCell?.row).toBe(2);
    expect(hoverSeq.hoveredCell?.tokenIndex).toBe(2);
  });

  it('8. invalidates sidebar editorial cache key whenever effective inspected token or head changes within the same part', () => {
    const state = createInitialState(createPerspectivePose([0, 0, 0], 18));
    state.tourActive = false;
    state.mode = 'cutaway_mamba';
    state.tokens = 8;
    state.mambaStateToken = null;
    state.hoveredPart = 'residual-in';

    const scInputT7 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'residual-in',
      activeHead: 0,
      activeCell: { tensor: 'input', row: 7, col: 0, tokenIndex: 7 },
      stateTokenIndex: null,
      viewAngle: 'three_quarter',
    });
    const scInputT2 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'residual-in',
      activeHead: 0,
      activeCell: { tensor: 'input', row: 2, col: 3, tokenIndex: 2 },
      stateTokenIndex: null,
      viewAngle: 'three_quarter',
    });

    expect(scInputT7.mambaStats?.stateTokenIndex).toBe(7);
    expect(scInputT2.mambaStats?.stateTokenIndex).toBe(2);
    expect(computeEditorialCacheKey(state, scInputT7)).not.toBe(
      computeEditorialCacheKey(state, scInputT2),
    );

    // Same part ('dt-branch'), different head column (H00 -> H05) at same token
    state.hoveredPart = 'dt-branch';
    const scDtH0 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'dt-branch',
      activeHead: 0,
      activeCell: { tensor: 'dt', row: 5, col: 0, headIndex: 0, tokenIndex: 5 },
      stateTokenIndex: null,
      viewAngle: 'three_quarter',
    });
    const scDtH5 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'dt-branch',
      activeHead: 0,
      activeCell: { tensor: 'dt', row: 5, col: 5, headIndex: 5, tokenIndex: 5 },
      stateTokenIndex: null,
      viewAngle: 'three_quarter',
    });

    expect(scDtH0.mambaStats?.activeHead).toBe(0);
    expect(scDtH5.mambaStats?.activeHead).toBe(5);
    expect(computeEditorialCacheKey(state, scDtH0)).not.toBe(
      computeEditorialCacheKey(state, scDtH5),
    );
  });
});
