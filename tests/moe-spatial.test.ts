import { describe, expect, it } from 'vitest';

import { runMoeSim, type MoeCellFormula, type MoeCellRef } from '../src/math/moeSim';
import {
  advanceMoeTokenCycleTick,
  resolveEffectiveMoeCell,
  resolveMoeHoverDuringCycling,
  stepMoeFocusToken,
} from '../src/model/moePlayback';
import { createInitialState, DEFAULT_FOV_DEG } from '../src/model/state';
import {
  aabbCorners,
  buildCameraSnapshot,
  fitOpBoundsCamera,
  makeViewportMetrics,
  projectWorldToInnerCss,
} from '../src/render/camera';
import { pickHitToMoeCell, pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import {
  SHEET_BASIS_FLOATS,
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_INSTANCE_FLOATS,
} from '../src/render/sheets';
import { sheetFrameWorldAabb } from '../src/render/sheetFrame';
import {
  buildMoECutaway,
  fitSpatialMoeCamera,
  MOE_FIT_BOUNDS,
  MOE_SPATIAL_FIT_BOUNDS,
} from '../src/scene/cutaways';
import { SOLID_INSTANCE_FLOATS } from '../src/scene/spine';
import { computeEditorialCacheKey } from '../src/ui/controls';

function sheetHighlightAt(sheetData: Float32Array, idx: number): number {
  return sheetData[idx * SHEET_INSTANCE_FLOATS + 11];
}

function solidFp8FlagAt(solidData: Float32Array, idx: number): number {
  return solidData[idx * SOLID_INSTANCE_FLOATS + 7];
}

describe('Spatial MoE (128+1) 3D Multi-Plane Cutaway, 6-Branch Convergence & Top-6 Normalization', () => {
  it('1. builds valid right-handed 3D orthonormal bases, 6 risen active towers vs 122 idle floor tiles, and 6 weighted-output branches', () => {
    const scene = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    expect(scene.fitBounds).toEqual(MOE_SPATIAL_FIT_BOUNDS);
    expect(scene.ribbonCount).toBe(0);
    expect(scene.moeStats?.voxelArrowCount).toBe(0);
    expect(scene.sheetCount).toBeGreaterThan(150);
    expect(scene.sheetData?.length).toBe((scene.sheetCount ?? 0) * SHEET_INSTANCE_FLOATS);
    expect(scene.sheetBasisData?.length).toBe((scene.sheetCount ?? 0) * SHEET_BASIS_FLOATS);
    expect(scene.solidCount).toBe(scene.sheetCount);

    const basis = scene.sheetBasisData!;
    let hasHorizXZ = false;
    let hasColsZ = false;
    let hasStandingYZ = false;

    for (let i = 0; i < (scene.sheetCount ?? 0); i++) {
      const o = i * SHEET_BASIS_FLOATS;
      const ux = basis[o], uy = basis[o + 1], uz = basis[o + 2];
      const nx = basis[o + 3], ny = basis[o + 4], nz = basis[o + 5];

      expect(Math.hypot(ux, uy, uz)).toBeCloseTo(1, 5);
      expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1, 5);
      expect(Math.abs(ux * nx + uy * ny + uz * nz)).toBeLessThan(1e-5);

      const vx = uy * nz - uz * ny;
      const vy = uz * nx - ux * nz;
      const vz = ux * ny - uy * nx;
      expect(Math.hypot(vx, vy, vz)).toBeCloseTo(1, 5);

      if (Math.abs(ny) > 0.9 && Math.abs(ux) > 0.9) hasHorizXZ = true;
      if (Math.abs(ny) > 0.9 && Math.abs(uz) > 0.9) hasColsZ = true;
      if (Math.abs(nx) > 0.9) hasStandingYZ = true;
    }
    expect(hasHorizXZ).toBe(true);
    expect(hasColsZ).toBe(true);
    expect(hasStandingYZ).toBe(true);

    // 6 active experts × 2 paired plates = 12 'routed-bank' boxes; 122 'inactive-experts'; 6 'routed-branch' weighted outputs
    const activePlates = scene.solidPickBoxes.filter((b) => b.partId === 'routed-bank');
    const idleTiles = scene.solidPickBoxes.filter((b) => b.partId === 'inactive-experts');
    const branchStrips = scene.solidPickBoxes.filter((b) => b.partId === 'routed-branch');
    expect(activePlates).toHaveLength(12);
    expect(idleTiles).toHaveLength(122);
    expect(branchStrips).toHaveLength(6);

    // Active Top-6 expert paired plates physically rise well above the 122 resting idle floor tiles
    const minActiveY = Math.min(...activePlates.map((b) => b.frame!.center[1]));
    const maxIdleY = Math.max(...idleTiles.map((b) => b.frame!.center[1]));
    expect(minActiveY - maxIdleY).toBeGreaterThan(0.30);

    // Upright learned weight walls stand well above ground deck
    for (const key of ['w_router', 'routed_w_up', 'routed_w_down', 'shared_w_up', 'shared_w_down'] as const) {
      const box = scene.solidPickBoxes.find((b) => b.tensorKey === key)!;
      expect(box.frame!.center[1]).toBeGreaterThan(0.70);
      expect(Math.abs(box.frame!.axisN[0])).toBeGreaterThan(0.9);
    }

    // Smooth 3D flow ribbons finite and non-empty
    expect(scene.flowVertices?.length ?? 0).toBeGreaterThan(200);
    expect(scene.flowIndices?.length ?? 0).toBeGreaterThan(100);
    for (let i = 0; i < (scene.flowVertices?.length ?? 0); i++) {
      expect(Number.isFinite(scene.flowVertices![i])).toBe(true);
    }
  });

  it('2. upholds Safeguard #7: Shared Expert is strictly 2.0x intermediate width and 1.0x hidden width (never double both)', () => {
    const scene = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    const rUp = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_w_up')!.frame!;
    const sUp = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_up')!.frame!;
    const rAct = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_act')!.frame!;
    const sAct = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_act')!.frame!;
    const rDown = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_w_down')!.frame!;
    const sDown = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_down')!.frame!;

    // W_up [inter, hidden]: extent[0] = hidden (d=8), extent[2] = intermediate (dExp=4 vs dShared=8)
    expect(sUp.extent[0]).toBeCloseTo(rUp.extent[0], 5); // EXACT 1.0x hidden width!
    expect(sUp.extent[2]).toBeCloseTo(rUp.extent[2] * 2, 5); // EXACT 2.0x intermediate width!

    // ReLU² [inter, 1]: extent[2] = intermediate
    expect(sAct.extent[2]).toBeCloseTo(rAct.extent[2] * 2, 5);

    // W_down [hidden, inter]: extent[0] = intermediate, extent[2] = hidden
    expect(sDown.extent[0]).toBeCloseTo(rDown.extent[0] * 2, 5); // EXACT 2.0x intermediate width!
    expect(sDown.extent[2]).toBeCloseTo(rDown.extent[2], 5); // EXACT 1.0x hidden width!
  });

  it('3. keeps MOE_SPATIAL_FIT_BOUNDS invariant across T=1..16, Bay 02/52, BF16/FP8, and projects all 8 corners inside desktop & narrow viewports', () => {
    const configs: Array<{
      bayId: 2 | 52;
      tokens: number;
      precision: 'bf16' | 'fp8_selective';
      focusToken: number;
      activeExpert?: number;
    }> = [
      { bayId: 2, tokens: 1, precision: 'bf16', focusToken: 0 },
      { bayId: 2, tokens: 8, precision: 'fp8_selective', focusToken: 2, activeExpert: 0 },
      { bayId: 52, tokens: 16, precision: 'fp8_selective', focusToken: 15 },
    ];

    for (const cfg of configs) {
      const sc = buildMoECutaway({
        bayId: cfg.bayId,
        tokens: cfg.tokens,
        precision: cfg.precision,
        selectedPart: null,
        hoveredPart: null,
        focusToken: cfg.focusToken,
        activeExpert: cfg.activeExpert,
        viewAngle: 'three_quarter',
      });
      expect(sc.fitBounds).toEqual(MOE_SPATIAL_FIT_BOUNDS);
      for (const box of sc.solidPickBoxes) {
        const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
        expect(aabb.min[0]).toBeGreaterThanOrEqual(MOE_SPATIAL_FIT_BOUNDS.min[0] - 1e-4);
        expect(aabb.min[1]).toBeGreaterThanOrEqual(MOE_SPATIAL_FIT_BOUNDS.min[1] - 1e-4);
        expect(aabb.min[2]).toBeGreaterThanOrEqual(MOE_SPATIAL_FIT_BOUNDS.min[2] - 1e-4);
        expect(aabb.max[0]).toBeLessThanOrEqual(MOE_SPATIAL_FIT_BOUNDS.max[0] + 1e-4);
        expect(aabb.max[1]).toBeLessThanOrEqual(MOE_SPATIAL_FIT_BOUNDS.max[1] + 1e-4);
        expect(aabb.max[2]).toBeLessThanOrEqual(MOE_SPATIAL_FIT_BOUNDS.max[2] + 1e-4);
      }
    }

    const viewports = [
      { w: 1120, h: 900, top: 84, bottom: 118, isNarrow: false },
      { w: 720, h: 800, top: 84, bottom: 118, isNarrow: false },
      { w: 390, h: 360, top: 12, bottom: 12, isNarrow: true },
    ];
    const corners = aabbCorners(MOE_SPATIAL_FIT_BOUNDS);
    for (const vp of viewports) {
      const metrics = makeViewportMetrics(0, 0, vp.w, vp.h, 1);
      const pose = fitSpatialMoeCamera(MOE_SPATIAL_FIT_BOUNDS, metrics, {
        topInsetPx: vp.top,
        bottomInsetPx: vp.bottom,
        isNarrow: vp.isNarrow,
      });
      const snap = buildCameraSnapshot(pose, MOE_SPATIAL_FIT_BOUNDS, metrics);
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
      expect(Math.max((maxX - minX) / vp.w, (maxY - minY) / (vp.h - vp.top - vp.bottom))).toBeGreaterThan(0.72);
    }
  });

  it('4. visualizes full 6-term unbiased Top-6 denominator, all 6 weighted branches on routed_sum, and front/back 3D picking parity', () => {
    const T = 8;
    const focusTok = 7;
    const sim = runMoeSim(T);
    const winners = Array.from(sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6));
    const slot2Exp = winners[2];

    // 4A. Top-6 Normalization (Refinement 2): numerator expert = OPERAND_A, other 5 winners = OPERAND_B, 122 idle = DIMMED
    const topkScene = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeCell: {
        tensor: 'topk_weights',
        row: 0,
        col: 2,
        tokenIndex: focusTok,
        expertId: slot2Exp,
        rankSlot: 2,
      },
      viewAngle: 'three_quarter',
    });

    const sd = topkScene.sheetData!;
    const bankEntries = topkScene.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: sheetHighlightAt(sd, idx) }))
      .filter((e) => e.box.tensorKey === 'router_bank');
    const opA = bankEntries.filter((e) => e.hl === SHEET_HL_OPERAND_A);
    const opB = bankEntries.filter((e) => e.hl === SHEET_HL_OPERAND_B);
    const dimmed = bankEntries.filter((e) => e.hl === SHEET_HL_DIMMED);
    expect(opA).toHaveLength(2);
    expect(opA.every((e) => e.box.headIndex === slot2Exp)).toBe(true);
    expect(opB).toHaveLength(10);
    expect(dimmed).toHaveLength(122);

    // Top-6 output weights lights ACTIVE, Unbiased σ strip lights OPERAND_B + contextual note emitted
    const topkIdx = topkScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'topk_weights');
    expect(sheetHighlightAt(sd, topkIdx)).toBe(SHEET_HL_ACTIVE);
    const unbiasedIdx = topkScene.solidPickBoxes.findIndex((b) => b.partId === 'router-unbiased');
    expect(sheetHighlightAt(sd, unbiasedIdx)).toBe(SHEET_HL_OPERAND_B);
    expect(topkScene.landmarks.some((l) => l.id === 'moe-topk-norm-note' && l.label.includes('all 6 in denom'))).toBe(true);

    // 4B. Inspecting routed_sum (Refinement 1): ALL 6 weighted-output branch strips illuminate (1 OPERAND_A + 5 OPERAND_B)!
    const rSumScene = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeExpert: slot2Exp,
      activeCell: { tensor: 'routed_sum', row: focusTok, col: 3, tokenIndex: focusTok },
      viewAngle: 'three_quarter',
    });
    const branches = rSumScene.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: sheetHighlightAt(rSumScene.sheetData!, idx) }))
      .filter((e) => e.box.partId === 'routed-branch');
    expect(branches).toHaveLength(6);
    expect(branches.filter((e) => e.hl === SHEET_HL_OPERAND_A)).toHaveLength(1);
    expect(branches.filter((e) => e.hl === SHEET_HL_OPERAND_B)).toHaveLength(5);

    // 4C. Front (+X) and back (-X) 3D picking parity on upright W_router and Shared W_up walls
    const sharedUpBox = topkScene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_up')!;
    const f = sharedUpBox.frame!;
    const samplePt: readonly [number, number, number] = [
      f.center[0],
      f.center[1] + 0.14 * f.extent[2],
      f.center[2] - 0.16 * f.extent[0],
    ];
    const frontRay: Ray = { origin: [samplePt[0] + 3, samplePt[1], samplePt[2]], direction: [-1, 0, 0] };
    const backRay: Ray = { origin: [samplePt[0] - 3, samplePt[1], samplePt[2]], direction: [1, 0, 0] };
    const frontHit = pickMacroBay(frontRay, [sharedUpBox], []);
    const backHit = pickMacroBay(backRay, [sharedUpBox], []);
    expect(frontHit).not.toBeNull();
    expect(backHit).not.toBeNull();
    expect(frontHit!.hitRow).toBe(backHit!.hitRow);
    expect(frontHit!.hitCol).toBe(backHit!.hitCol);
    const topkSlice = sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6);
    expect(pickHitToMoeCell(frontHit!, focusTok, slot2Exp, topkSlice)).toEqual(
      pickHitToMoeCell(backHit!, focusTok, slot2Exp, topkSlice),
    );
  });

  it('5. verifies FP8 Selective precision metadata (FP32 router gold vs FP8 experts), 0 B cache, and explicit hover vs token cycling policy', () => {
    const T = 8;
    const sim = runMoeSim(T);
    const fp8Scene = buildMoECutaway({
      bayId: 52,
      tokens: T,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 7,
      viewAngle: 'three_quarter',
    });

    expect(fp8Scene.moeStats?.cacheBytes).toBe(0);
    expect(fp8Scene.moeStats?.weightsFp8).toBe(true);
    expect(fp8Scene.moeStats?.routerFp8).toBe(false);

    // Router walls/bridge stay FP32 (fp8Flag=0, gold rim when idle); MLP weights switch to FP8 (fp8Flag=1)
    for (const fp32Part of ['w-router', 'router-unbiased', 'router']) {
      const idxs = fp8Scene.solidPickBoxes
        .map((b, i) => ({ b, i }))
        .filter((e) => e.b.partId === fp32Part);
      expect(idxs.length).toBeGreaterThan(0);
      for (const { i } of idxs) {
        expect(solidFp8FlagAt(fp8Scene.solidData, i)).toBe(0);
        expect(sheetHighlightAt(fp8Scene.sheetData!, i)).toBe(SHEET_HL_BF16_GOLD);
      }
    }
    for (const fp8Part of ['routed-bank', 'routed-w-up', 'routed-w-down', 'shared-expert', 'shared-w-down']) {
      const idxs = fp8Scene.solidPickBoxes
        .map((b, i) => ({ b, i }))
        .filter((e) => e.b.partId === fp8Part);
      expect(idxs.length).toBeGreaterThan(0);
      for (const { i } of idxs) {
        expect(solidFp8FlagAt(fp8Scene.solidData, i)).toBe(1);
      }
    }

    // Explicit MoE token cycling vs hover policy:
    // (a) Snapshot hover (topk_weights) continues cycling and remaps slot winner to advancing token
    const topkHover: MoeCellRef = {
      tensor: 'topk_weights',
      row: 0,
      col: 0,
      rankSlot: 0,
      tokenIndex: 2,
      expertId: sim.topk_indices[2 * 6],
    };
    const rSnap = resolveMoeHoverDuringCycling(topkHover, true, 2, sim);
    expect(rSnap.shouldPauseTokenCycling).toBe(false);
    const tick = advanceMoeTokenCycleTick(2, sim, null, rSnap.hoveredCell);
    expect(tick.nextToken).toBe(3);
    expect(tick.nextHoveredCell?.tokenIndex).toBe(3);
    expect(tick.nextHoveredCell?.expertId).toBe(sim.topk_indices[3 * 6]);

    // Structural guard prevents stale hover token when cycling is active
    const eff = resolveEffectiveMoeCell({
      sim,
      moeFocusToken: 5,
      moeActiveExpert: null,
      moeTokenCycling: true,
      moeHoveredCell: topkHover,
      moeSelectedCell: null,
      moeStepperActive: false,
      moeStepperStep: 0,
      moeStepperSubProgress: 0,
    });
    expect(eff?.tokenIndex).toBe(5);
    expect(eff?.expertId).toBe(sim.topk_indices[5 * 6]);

    // (b) Weight hover keeps cycling active
    expect(
      resolveMoeHoverDuringCycling({ tensor: 'w_router', row: 10, col: 2 }, true, 3, sim)
        .shouldPauseTokenCycling,
    ).toBe(false);

    // (c) Sequence-row hover cleanly pauses token cycling
    expect(
      resolveMoeHoverDuringCycling({ tensor: 'input', row: 4, col: 1, tokenIndex: 4 }, true, 3, sim)
        .shouldPauseTokenCycling,
    ).toBe(true);
  });

  it('6. routes Step 6 Residual carry-in badge without expanding opFocusBounds[5] and preserves Near-Plan View parity', () => {
    const resCell: MoeCellRef = { tensor: 'residual', row: 7, col: 2, tokenIndex: 7 };
    const spatial = buildMoECutaway({
      bayId: 52,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 7,
      activeCell: resCell,
      viewAngle: 'three_quarter',
    });
    const nearPlan = buildMoECutaway({
      bayId: 52,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 7,
      activeCell: resCell,
      viewAngle: 'near_plan',
    });

    // Step 6 Residual carry-in badge emitted while opFocusBounds[5] stays focused on output merge (excluding Input X)
    // and projects cleanly inside the zoomed Step 6 Auto-Follow viewport
    const carryLm = spatial.landmarks.find((l) => l.id === 'moe-res-carry');
    expect(carryLm).toBeDefined();
    expect(carryLm!.label).toContain('← Input X[2, t7]');
    const inputBox = spatial.solidPickBoxes.find((b) => b.tensorKey === 'input')!;
    expect(spatial.opFocusBounds![5].min[0]).toBeGreaterThan(inputBox.max[0]);
    const step6Metrics = makeViewportMetrics(0, 0, 1120, 900, 1);
    const step6Pose = fitOpBoundsCamera(spatial.opFocusBounds![5], step6Metrics, {
      azimuthDeg: 20,
      elevationDeg: 23.5,
      fovDeg: DEFAULT_FOV_DEG,
      topInsetPx: 84,
      bottomInsetPx: 118,
      fillFraction: 0.80,
      isNarrow: false,
    });
    const step6Snap = buildCameraSnapshot(step6Pose, spatial.opFocusBounds![5], step6Metrics);
    const carryProj = projectWorldToInnerCss(carryLm!.world, step6Snap);
    expect(carryProj.visible).toBe(true);
    expect(carryProj.x).toBeGreaterThan(20);
    expect(carryProj.x).toBeLessThan(1100);
    expect(carryProj.y).toBeGreaterThan(84);
    expect(carryProj.y).toBeLessThan(900 - 118);

    for (let s = 0; s < 6; s++) {
      const ob = spatial.opFocusBounds![s];
      expect(ob).toBeDefined();
      expect(ob.max[0] - ob.min[0]).toBeLessThan(
        MOE_SPATIAL_FIT_BOUNDS.max[0] - MOE_SPATIAL_FIT_BOUNDS.min[0],
      );
    }

    // Preserved Near-Plan View parity
    expect(nearPlan.fitBounds).toEqual(MOE_FIT_BOUNDS);
    expect(nearPlan.sheetBasisData).toBeUndefined();
    expect(spatial.activeFormula).toEqual(nearPlan.activeFormula);
    expect(Object.keys(spatial.partCatalog).sort()).toEqual(
      Object.keys(nearPlan.partCatalog).sort(),
    );
  });

  it('7. keeps routing snapshots advancing under stationary picker-produced weight hover during token cycling', () => {
    const T = 8;
    const startTok = 1;
    const sim = runMoeSim(T);
    const startWinners = sim.topk_indices.subarray(startTok * 6, (startTok + 1) * 6);
    const baseScene = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: startTok,
      viewAngle: 'three_quarter',
    });

    const castBoxRay = (box: PickBox, offsetU = 0, offsetV = 0): Ray => {
      const f = box.frame!;
      const pt: [number, number, number] = [
        f.center[0] + offsetU * f.extent[0] * f.axisU[0] + offsetV * f.extent[2] * f.axisV[0],
        f.center[1] + offsetU * f.extent[0] * f.axisU[1] + offsetV * f.extent[2] * f.axisV[1],
        f.center[2] + offsetU * f.extent[0] * f.axisU[2] + offsetV * f.extent[2] * f.axisV[2],
      ];
      return {
        origin: [pt[0] + f.axisN[0] * 3, pt[1] + f.axisN[1] * 3, pt[2] + f.axisN[2] * 3],
        direction: [-f.axisN[0], -f.axisN[1], -f.axisN[2]],
      };
    };

    // Test ALL THREE token-independent weight walls with real ray-picked MoeCellRefs!
    for (const weightKey of ['w_router', 'shared_w_up', 'shared_w_down'] as const) {
      const weightBox = baseScene.solidPickBoxes.find((b) => b.tensorKey === weightKey)!;
      expect(weightBox).toBeDefined();
      expect(weightBox.tokenIndex).toBeUndefined();

      const hit = pickMacroBay(castBoxRay(weightBox, 0.1, -0.1), [weightBox], []);
      expect(hit).not.toBeNull();
      const pickedCell = pickHitToMoeCell(hit!, startTok, startWinners[0], startWinners);
      expect(pickedCell).not.toBeNull();
      expect(pickedCell!.tensor).toBe(weightKey);
      expect(pickedCell!.tokenIndex).toBeUndefined();

      // Hovering while cycling keeps cycling active and never pins startTok
      const hoverRes = resolveMoeHoverDuringCycling(pickedCell, true, startTok, sim);
      expect(hoverRes.shouldPauseTokenCycling).toBe(false);
      expect(hoverRes.hoveredCell?.tokenIndex).toBeUndefined();

      let curTok = startTok;
      let curHover = hoverRes.hoveredCell;
      for (const expectedTok of [2, 3, 4]) {
        const tick = advanceMoeTokenCycleTick(curTok, sim, null, curHover);
        expect(tick.nextToken).toBe(expectedTok);
        expect(tick.nextHoveredCell?.tensor).toBe(weightKey);
        expect(tick.nextHoveredCell?.tokenIndex).toBeUndefined();
        curTok = tick.nextToken;
        curHover = tick.nextHoveredCell;

        const effCell = resolveEffectiveMoeCell({
          sim,
          moeFocusToken: curTok,
          moeActiveExpert: null,
          moeTokenCycling: true,
          moeHoveredCell: curHover,
          moeSelectedCell: null,
          moeStepperActive: false,
          moeStepperStep: 0,
          moeStepperSubProgress: 0,
        });
        const rebuilt = buildMoECutaway({
          bayId: 2,
          tokens: T,
          precision: 'bf16',
          selectedPart: null,
          hoveredPart: weightBox.partId ?? null,
          focusToken: curTok,
          activeCell: effCell,
          viewAngle: 'three_quarter',
        });
        expect(rebuilt.moeStats?.focusToken).toBe(expectedTok);
        expect(rebuilt.moeStats?.activeExpertIds).toEqual(
          Array.from(sim.topk_indices.subarray(expectedTok * 6, (expectedTok + 1) * 6)),
        );
        expect((rebuilt.activeFormula as MoeCellFormula | null)?.upstreamSpans[0]?.tokenIndex).toBe(expectedTok);
      }

      // Defense-in-depth: even a manually stale tokenIndex on a weight cell cannot freeze buildMoECutaway
      const staleWeight: MoeCellRef = { ...pickedCell!, tokenIndex: startTok };
      const defensiveScene = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: 5,
        activeCell: staleWeight,
        viewAngle: 'three_quarter',
      });
      expect(defensiveScene.moeStats?.focusToken).toBe(5);
      expect((defensiveScene.activeFormula as MoeCellFormula | null)?.upstreamSpans[0]?.tokenIndex).toBe(5);
    }

    // Contrast: real picker-produced shared_act (snapshot) tracks advancing token; input (sequence) pauses cycling
    const sharedActBox = baseScene.solidPickBoxes.find((b) => b.tensorKey === 'shared_act')!;
    const sharedActCell = pickHitToMoeCell(
      pickMacroBay(castBoxRay(sharedActBox), [sharedActBox], [])!,
      startTok,
      startWinners[0],
      startWinners,
    )!;
    expect(sharedActCell.tokenIndex).toBe(startTok);
    expect(resolveMoeHoverDuringCycling(sharedActCell, true, startTok, sim).shouldPauseTokenCycling).toBe(false);
    expect(advanceMoeTokenCycleTick(startTok, sim, null, sharedActCell).nextHoveredCell?.tokenIndex).toBe(2);

    const inputBox = baseScene.solidPickBoxes.find((b) => b.tensorKey === 'input')!;
    const inputCell = pickHitToMoeCell(
      pickMacroBay(castBoxRay(inputBox), [inputBox], [])!,
      startTok,
      startWinners[0],
      startWinners,
    )!;
    expect(resolveMoeHoverDuringCycling(inputCell, true, startTok, sim).shouldPauseTokenCycling).toBe(true);
  });

  it('8. advances reduced-motion token step cleanly when a picker-produced cell is already pinned', () => {
    const T = 8;
    const startTok = 1;
    const sim = runMoeSim(T);
    const startWinners = sim.topk_indices.subarray(startTok * 6, (startTok + 1) * 6);
    const baseScene = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: startTok,
      viewAngle: 'three_quarter',
    });

    const pickFromBox = (box: PickBox, offsetU = 0, offsetV = 0): MoeCellRef => {
      const f = box.frame!;
      const pt: [number, number, number] = [
        f.center[0] + offsetU * f.extent[0] * f.axisU[0] + offsetV * f.extent[2] * f.axisV[0],
        f.center[1] + offsetU * f.extent[0] * f.axisU[1] + offsetV * f.extent[2] * f.axisV[1],
        f.center[2] + offsetU * f.extent[0] * f.axisU[2] + offsetV * f.extent[2] * f.axisV[2],
      ];
      const hit = pickMacroBay(
        {
          origin: [pt[0] + f.axisN[0] * 3, pt[1] + f.axisN[1] * 3, pt[2] + f.axisN[2] * 3],
          direction: [-f.axisN[0], -f.axisN[1], -f.axisN[2]],
        },
        [box],
        [],
      );
      expect(hit).not.toBeNull();
      const cell = pickHitToMoeCell(hit!, startTok, startWinners[0], startWinners);
      expect(cell).not.toBeNull();
      return cell!;
    };

    const pinnedCases: Array<{ label: string; box: PickBox; offsetU?: number; offsetV?: number }> = [
      { label: 'topk_weights', box: baseScene.solidPickBoxes.find((b) => b.tensorKey === 'topk_weights')!, offsetU: -0.4 },
      {
        label: 'router_bank',
        box: baseScene.solidPickBoxes.find(
          (b) => b.tensorKey === 'router_bank' && b.headIndex === startWinners[0],
        )!,
      },
      { label: 'routed_sum', box: baseScene.solidPickBoxes.find((b) => b.tensorKey === 'routed_sum')!, offsetV: -0.25 },
      { label: 'shared_w_up', box: baseScene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_up')! },
    ];

    const dummyPose = { target: [0, 0, 0] as const, azimuthDeg: 20, elevationDeg: 23.5, distance: 12, fovDeg: DEFAULT_FOV_DEG };
    const state = createInitialState(dummyPose, true);
    state.mode = 'cutaway_moe';
    state.tokens = T;

    for (const c of pinnedCases) {
      const pinned = pickFromBox(c.box, c.offsetU ?? 0, c.offsetV ?? 0);
      const pinnedTok = pinned.tokenIndex ?? startTok;
      const expectedNextTok = (pinnedTok + 1) % T;
      const beforeScene = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: c.box.partId ?? null,
        hoveredPart: null,
        focusToken: pinnedTok,
        activeCell: pinned,
        viewAngle: 'three_quarter',
      });
      state.tourActive = false;
      state.moeFocusToken = pinnedTok;
      state.selectedPart = c.box.partId ?? null;
      const keyBefore = computeEditorialCacheKey(state, beforeScene);

      // Simulate reduced-motion single-step advancement (`▶ Cycle t` with reducedMotion = true)
      const stepped = stepMoeFocusToken(pinnedTok, 1, sim, pinned);
      expect(stepped.nextToken, `${c.label} nextToken`).toBe(expectedNextTok);
      const effAfter = resolveEffectiveMoeCell({
        sim,
        moeFocusToken: stepped.nextToken,
        moeActiveExpert: null,
        moeTokenCycling: false, // reducedMotion keeps continuous cycling false!
        moeHoveredCell: null,
        moeSelectedCell: stepped.nextSelectedCell,
        moeStepperActive: false,
        moeStepperStep: 0,
        moeStepperSubProgress: 0,
      });
      const afterScene = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: c.box.partId ?? null,
        hoveredPart: null,
        focusToken: stepped.nextToken,
        activeCell: effAfter,
        viewAngle: 'three_quarter',
      });
      state.moeFocusToken = stepped.nextToken;
      const keyAfter = computeEditorialCacheKey(state, afterScene);

      expect(afterScene.moeStats?.focusToken, `${c.label} displayed focusToken`).toBe(expectedNextTok);
      expect(afterScene.moeStats?.activeExpertIds).toEqual(
        Array.from(sim.topk_indices.subarray(expectedNextTok * 6, (expectedNextTok + 1) * 6)),
      );
      expect(keyAfter).not.toBe(keyBefore);

      if (c.label === 'topk_weights') {
        const slot = pinned.rankSlot ?? pinned.col;
        expect(stepped.nextSelectedCell?.expertId).toBe(sim.topk_indices[expectedNextTok * 6 + slot]);
        expect(afterScene.activeFormula?.coordLabel).toContain(`t${expectedNextTok}`);
      } else if (c.label === 'routed_sum') {
        expect(stepped.nextSelectedCell?.row).toBe(expectedNextTok);
        expect(afterScene.activeFormula?.coordLabel).toContain(`t${expectedNextTok}`);
        expect(afterScene.moeStats?.focusExpert).toBe(sim.topk_indices[expectedNextTok * 6]);
      } else if (c.label === 'shared_w_up') {
        expect(stepped.nextSelectedCell?.tokenIndex).toBeUndefined();
        expect(afterScene.activeFormula?.equation).toContain(`t${expectedNextTok}`);
      }
    }
  });
});
