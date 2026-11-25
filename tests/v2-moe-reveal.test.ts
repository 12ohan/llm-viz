import { describe, expect, it } from 'vitest';

import { createDefaultHybridModel } from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeStep,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import {
  evaluateSingleRoutedExpert,
  formatExpertId,
  runMoeSim,
} from '../src/math/moeSim';
import { DEFAULT_FOV_DEG } from '../src/model/state';
import {
  aabbCorners,
  buildCameraSnapshot,
  fitOpBoundsCamera,
  makeViewportMetrics,
  projectWorldToHostCss,
  type CameraSnapshot,
} from '../src/render/camera';
import type { PickBox } from '../src/render/pick';
import { sheetFrameWorldAabb } from '../src/render/sheetFrame';
import {
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
} from '../src/render/sheets';
import {
  buildMoECutaway,
  fitSpatialMoeCamera,
  type CutawaySceneGeometry,
} from '../src/scene/cutaways';
import { fmtSigned } from '../src/scene/sheetInspector';
import type { LandmarkAnchor } from '../src/scene/spine';
import { computeSheetCellWorldPos, isCellHiddenByReveal } from '../src/ui/overlay';

interface Rect2D {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

function revealAt(scene: CutawaySceneGeometry, idx: number): readonly [number, number, number, number] {
  const rd = scene.sheetRevealData!;
  const o = idx * SHEET_REVEAL_FLOATS;
  return [rd[o], rd[o + 1], rd[o + 2], rd[o + 3]];
}

function boxScreenRect(box: PickBox, snap: CameraSnapshot): Rect2D {
  const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const c of aabbCorners(aabb)) {
    const p = projectWorldToHostCss(c, snap);
    if (p.x < x0) x0 = p.x;
    if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.y > y1) y1 = p.y;
  }
  return { x0, y0, x1, y1 };
}

function approxLabelRect(lm: LandmarkAnchor, snap: CameraSnapshot): Rect2D {
  const p = projectWorldToHostCss(lm.world, snap);
  const w = lm.label.length * 6.4 + 12;
  const h = lm.tier === 'detail' ? 17 : 18;
  const gap = 4;
  let lx = p.x - w * 0.5;
  let ly = p.y - h * 0.5;
  if (lm.align === 'top') {
    lx = p.x - w * 0.5;
    ly = p.y - h - gap;
  } else if (lm.align === 'bottom') {
    lx = p.x - w * 0.5;
    ly = p.y + gap;
  } else if (lm.align === 'left') {
    lx = p.x - w - gap;
    ly = p.y - h * 0.5;
  } else if (lm.align === 'right') {
    lx = p.x + gap;
    ly = p.y - h * 0.5;
  }
  return { x0: lx, y0: ly, x1: lx + w, y1: ly + h };
}

function rectsOverlap(a: Rect2D, b: Rect2D, pad = 0): boolean {
  return a.x0 < b.x1 + pad && a.x1 + pad > b.x0 && a.y0 < b.y1 + pad && a.y1 + pad > b.y0;
}

describe('V1 · Critical MoE Label Occlusion Fix & Exterior Clearance', () => {
  it('places moe-topk cleanly exterior to Unbiased σ and Top-6 strips with zero occlusion in Overview and Step 3 Auto-Follow', () => {
    const T = 8;
    const focusTok = 7;
    const sim = runMoeSim(T);
    const slot2Exp = sim.topk_indices[focusTok * 6 + 2];

    const scene = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 2,
      stepperActive: true,
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

    const unbiasedBox = scene.solidPickBoxes.find((b) => b.partId === 'router-unbiased')!;
    const topkBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'topk_weights')!;
    const unbiasedAabb = sheetFrameWorldAabb(unbiasedBox.frame!);
    const topkAabb = sheetFrameWorldAabb(topkBox.frame!);
    const maxStripZ = Math.max(unbiasedAabb.max[2], topkAabb.max[2]);

    const topkLm = scene.landmarks.find((l) => l.id === 'moe-topk')!;
    const noteLm = scene.landmarks.find((l) => l.id === 'moe-topk-norm-note')!;
    expect(topkLm).toBeDefined();
    expect(noteLm).toBeDefined();

    // World-space exterior guarantee: moe-topk sits cleanly in front (+Z) past maxStripZ + 0.55 with align='bottom'
    expect(topkLm.align).toBe('bottom');
    expect(topkLm.world[2]).toBeGreaterThan(maxStripZ + 0.55);
    // Contextual denominator note sits on the opposite rear (-Z) elevated side
    expect(noteLm.align).toBe('top');
    expect(noteLm.world[2]).toBeLessThan(Math.min(unbiasedAabb.min[2], topkAabb.min[2]));

    for (const [vpW, vpH] of [
      [1440, 900],
      [1120, 900],
      [720, 800],
    ] as const) {
      const metrics = makeViewportMetrics(0, 0, vpW, vpH, 1);
      const overviewPose = fitSpatialMoeCamera(scene.fitBounds, metrics, {
        topInsetPx: 84,
        bottomInsetPx: 118,
        isNarrow: false,
      });
      const step3Pose = fitOpBoundsCamera(scene.opFocusBounds![2], metrics, {
        azimuthDeg: 20,
        elevationDeg: 23.5,
        fovDeg: DEFAULT_FOV_DEG,
        topInsetPx: 84,
        bottomInsetPx: 118,
        fillFraction: 0.80,
        isNarrow: false,
      });

      for (const [tag, pose, bounds] of [
        [`overview-${vpW}`, overviewPose, scene.fitBounds],
        [`step3-${vpW}`, step3Pose, scene.opFocusBounds![2]],
      ] as const) {
        const snap = buildCameraSnapshot(pose, bounds, metrics);
        const uRect = boxScreenRect(unbiasedBox, snap);
        const tRect = boxScreenRect(topkBox, snap);
        const topkRect = approxLabelRect(topkLm, snap);
        const noteRect = approxLabelRect(noteLm, snap);
        const bankLm = scene.landmarks.find((l) => l.id === 'moe-bank')!;
        const bankRect = approxLabelRect(bankLm, snap);

        // Zero overlap between moe-topk / moe-topk-norm-note and BOTH 6-slot convergence strips
        expect(rectsOverlap(topkRect, uRect), `${tag} moe-topk vs Unbiased σ`).toBe(false);
        expect(rectsOverlap(topkRect, tRect), `${tag} moe-topk vs Top-6`).toBe(false);
        expect(rectsOverlap(noteRect, uRect), `${tag} note vs Unbiased σ`).toBe(false);
        expect(rectsOverlap(noteRect, tRect), `${tag} note vs Top-6`).toBe(false);
        expect(rectsOverlap(topkRect, noteRect), `${tag} moe-topk vs note`).toBe(false);
        expect(rectsOverlap(bankRect, noteRect), `${tag} moe-bank vs note`).toBe(false);

        // moe-topk stays strictly inside the unobstructed viewport band
        const pt = projectWorldToHostCss(topkLm.world, snap);
        expect(pt.visible, `${tag} moe-topk visible`).toBe(true);
        expect(pt.y, `${tag} moe-topk below top inset`).toBeGreaterThan(84);
        expect(pt.y, `${tag} moe-topk above bottom inset`).toBeLessThan(vpH - 118);

        // Verify clean exterior clearance for Router Bank, ReLU², 6× Active, Routed Sum, and Shared Expert labels
        for (const checkId of [
          'moe-bank',
          'moe-routed-act',
          'moe-branches',
          'moe-routed-sum',
          'moe-shared-up',
          'moe-shared-act',
          'moe-shared-down',
          'moe-shared-out',
        ]) {
          const lm = scene.landmarks.find((l) => l.id === checkId)!;
          expect(lm, `missing landmark ${checkId}`).toBeDefined();
          const lr = approxLabelRect(lm, snap);
          const ownBoxes = scene.solidPickBoxes.filter((b) => b.partId === lm.partId);
          for (const ob of ownBoxes) {
            expect(rectsOverlap(lr, boxScreenRect(ob, snap)), `${tag} ${checkId} vs own sheet`).toBe(false);
          }
        }
      }
    }
  });
});

describe('V2 · Progressive Accumulation Reveal (`sheetRevealData`)', () => {
  it('omits sheetRevealData when idle, hover-only, or reducedMotion is true', () => {
    const T = 8;
    const sim = runMoeSim(T);
    const slot2Exp = sim.topk_indices[7 * 6 + 2];

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // Idle (default stepperActive = false)
      const idle = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
      });
      expect(idle.sheetRevealData).toBeUndefined();

      // Hover cell without stepperActive
      const hovered = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: 'router',
        focusToken: 7,
        stepperStep: 2,
        stepperActive: false,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: 2,
          tokenIndex: 7,
          expertId: slot2Exp,
          rankSlot: 2,
        },
        viewAngle,
      });
      expect(hovered.sheetRevealData).toBeUndefined();

      // reducedMotion: true suppresses reveal masking even with stepperActive: true
      const reduced = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: 7,
        stepperStep: 2,
        stepperActive: true,
        reducedMotion: true,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: 2,
          tokenIndex: 7,
          expertId: slot2Exp,
          rankSlot: 2,
        },
        viewAngle,
      });
      expect(reduced.sheetRevealData).toBeUndefined();
    }
  });

  it('populates step-accurate progressive frontiers across Steps 1..6 (especially Step 3 Unbiased σ = NONE & Top-6 progressive)', () => {
    const T = 8;
    const focusTok = 7;
    const sim = runMoeSim(T);
    const winners = Array.from(sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6));

    const findIdx = (scene: CutawaySceneGeometry, pred: (b: PickBox) => boolean): number => {
      const idx = scene.solidPickBoxes.findIndex(pred);
      expect(idx).toBeGreaterThanOrEqual(0);
      return idx;
    };

    // --- Step 1 (stepperStep = 0, RMSNorm) ---
    const s1 = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 0,
      stepperActive: true,
      activeCell: { tensor: 'rmsnorm', row: focusTok, col: 3, tokenIndex: focusTok },
      viewAngle: 'three_quarter',
    });
    expect(s1.sheetRevealData?.length).toBe((s1.sheetCount ?? 0) * SHEET_REVEAL_FLOATS);
    expect(revealAt(s1, findIdx(s1, (b) => b.tensorKey === 'input'))[0]).toBe(SHEET_REVEAL_NONE);
    const rmsRev = revealAt(s1, findIdx(s1, (b) => b.tensorKey === 'rmsnorm'));
    expect(rmsRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(rmsRev[1]).toBe(3);
    expect(rmsRev[2]).toBe(focusTok);
    for (const downstream of [
      (b: PickBox) => b.tensorKey === 'router_bank',
      (b: PickBox) => b.partId === 'router-unbiased',
      (b: PickBox) => b.tensorKey === 'topk_weights',
      (b: PickBox) => b.tensorKey === 'routed_pre',
      (b: PickBox) => b.tensorKey === 'routed_act',
      (b: PickBox) => b.partId === 'routed-branch',
      (b: PickBox) => b.tensorKey === 'routed_sum',
      (b: PickBox) => b.tensorKey === 'shared_act',
      (b: PickBox) => b.tensorKey === 'shared_out',
      (b: PickBox) => b.tensorKey === 'moe_out',
      (b: PickBox) => b.tensorKey === 'residual',
    ]) {
      expect(revealAt(s1, findIdx(s1, downstream))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    }

    // --- Step 2 (stepperStep = 1, Router + Bias) ---
    const step2Exp = winners[2];
    const s2 = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeExpert: step2Exp,
      stepperStep: 1,
      stepperActive: true,
      activeCell: {
        tensor: 'router_bank',
        row: Math.floor(step2Exp / 16),
        col: step2Exp % 16,
        tokenIndex: focusTok,
        expertId: step2Exp,
      },
      viewAngle: 'three_quarter',
    });
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'rmsnorm'))[0]).toBe(SHEET_REVEAL_NONE);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'router_bank' && b.headIndex === winners[0]))[0]).toBe(SHEET_REVEAL_NONE);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'router_bank' && b.headIndex === winners[2]))[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'router_bank' && b.headIndex === winners[4]))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s2, findIdx(s2, (b) => b.partId === 'router-unbiased'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'topk_weights'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'routed_sum'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'moe_out'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s2, findIdx(s2, (b) => b.tensorKey === 'residual'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);

    // --- Step 3 (stepperStep = 2, Top-6 ×2.5) ---
    for (const activeSlot of [0, 3, 5]) {
      const s3 = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        stepperStep: 2,
        stepperActive: true,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: activeSlot,
          tokenIndex: focusTok,
          expertId: winners[activeSlot],
          rankSlot: activeSlot,
        },
        viewAngle: 'three_quarter',
      });
      // CRITICAL: Unbiased σ stays 100% revealed (SHEET_REVEAL_NONE) so all 6 denominator terms stay visible!
      expect(revealAt(s3, findIdx(s3, (b) => b.partId === 'router-unbiased'))[0]).toBe(SHEET_REVEAL_NONE);
      // Top-6 reveals slots 0..activeSlot progressively
      const topkRev = revealAt(s3, findIdx(s3, (b) => b.tensorKey === 'topk_weights'));
      expect(topkRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(topkRev[1]).toBe(activeSlot);
      expect(topkRev[2]).toBe(0);
      // Downstream workbench, 6 branches, Routed Sum, MoE Out, Residual are UNCOMPUTED
      expect(revealAt(s3, findIdx(s3, (b) => b.tensorKey === 'routed_pre'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(revealAt(s3, findIdx(s3, (b) => b.tensorKey === 'routed_act'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      const branchIdxs = s3.solidPickBoxes
        .map((b, i) => ({ b, i }))
        .filter((e) => e.b.partId === 'routed-branch')
        .map((e) => e.i);
      expect(branchIdxs).toHaveLength(6);
      for (const bi of branchIdxs) {
        expect(revealAt(s3, bi)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      }
      expect(revealAt(s3, findIdx(s3, (b) => b.tensorKey === 'routed_sum'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(revealAt(s3, findIdx(s3, (b) => b.tensorKey === 'moe_out'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(revealAt(s3, findIdx(s3, (b) => b.tensorKey === 'residual'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    }

    // --- Step 4 (stepperStep = 3, Routed ReLU²) ---
    const activeRank = 2;
    const rank2Exp = winners[activeRank];
    const s4 = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeExpert: rank2Exp,
      stepperStep: 3,
      stepperActive: true,
      activeCell: {
        tensor: 'routed_act',
        row: 2,
        col: 0,
        tokenIndex: focusTok,
        expertId: rank2Exp,
      },
      viewAngle: 'three_quarter',
    });
    expect(revealAt(s4, findIdx(s4, (b) => b.tensorKey === 'topk_weights'))[0]).toBe(SHEET_REVEAL_NONE);
    const preRev = revealAt(s4, findIdx(s4, (b) => b.tensorKey === 'routed_pre'));
    const actRev = revealAt(s4, findIdx(s4, (b) => b.tensorKey === 'routed_act'));
    expect(preRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(preRev[2]).toBe(2);
    expect(actRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(actRev[2]).toBe(2);

    const s4Branches = s4.solidPickBoxes
      .map((b, i) => ({ b, i }))
      .filter((e) => e.b.partId === 'routed-branch');
    expect(s4Branches).toHaveLength(6);
    for (let j = 0; j < 6; j++) {
      const mode = revealAt(s4, s4Branches[j].i)[0];
      if (j < activeRank) expect(mode).toBe(SHEET_REVEAL_NONE);
      else if (j === activeRank) expect(mode).toBe(SHEET_REVEAL_ROW_MAJOR);
      else expect(mode).toBe(SHEET_REVEAL_UNCOMPUTED);
    }
    expect(revealAt(s4, findIdx(s4, (b) => b.tensorKey === 'routed_sum'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s4, findIdx(s4, (b) => b.tensorKey === 'moe_out'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s4, findIdx(s4, (b) => b.tensorKey === 'residual'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);

    // --- Step 5 (stepperStep = 4, Shared 2×) ---
    const s5 = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 4,
      stepperActive: true,
      activeCell: { tensor: 'shared_act', row: 4, col: 0, tokenIndex: focusTok },
      viewAngle: 'three_quarter',
    });
    expect(revealAt(s5, findIdx(s5, (b) => b.tensorKey === 'routed_sum'))[0]).toBe(SHEET_REVEAL_NONE);
    const sActRev = revealAt(s5, findIdx(s5, (b) => b.tensorKey === 'shared_act'));
    const sOutRev = revealAt(s5, findIdx(s5, (b) => b.tensorKey === 'shared_out'));
    expect(sActRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(sActRev[2]).toBe(4);
    expect(sOutRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(sOutRev[2]).toBe(focusTok);
    expect(revealAt(s5, findIdx(s5, (b) => b.tensorKey === 'moe_out'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
    expect(revealAt(s5, findIdx(s5, (b) => b.tensorKey === 'residual'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);

    // --- Step 6 (stepperStep = 5, Merge + Res) ---
    const s6 = buildMoECutaway({
      bayId: 52,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 5,
      stepperActive: true,
      activeCell: { tensor: 'residual', row: focusTok, col: 5, tokenIndex: focusTok },
      viewAngle: 'three_quarter',
    });
    expect(revealAt(s6, findIdx(s6, (b) => b.tensorKey === 'shared_out'))[0]).toBe(SHEET_REVEAL_NONE);
    const moeOutRev = revealAt(s6, findIdx(s6, (b) => b.tensorKey === 'moe_out'));
    const resRev = revealAt(s6, findIdx(s6, (b) => b.tensorKey === 'residual'));
    expect(moeOutRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(moeOutRev[1]).toBe(5);
    expect(moeOutRev[2]).toBe(focusTok);
    expect(resRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(resRev[1]).toBe(5);
    expect(resRev[2]).toBe(focusTok);

    // Near-Plan parity check for Step 3 progressive reveal
    const s3Near = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 2,
      stepperActive: true,
      activeCell: {
        tensor: 'topk_weights',
        row: 0,
        col: 1,
        tokenIndex: focusTok,
        expertId: winners[1],
        rankSlot: 1,
      },
      viewAngle: 'near_plan',
    });
    expect(s3Near.sheetRevealData?.length).toBe((s3Near.sheetCount ?? 0) * SHEET_REVEAL_FLOATS);
    expect(revealAt(s3Near, findIdx(s3Near, (b) => b.tensorKey === 'topk_weights'))[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(revealAt(s3Near, findIdx(s3Near, (b) => b.tensorKey === 'topk_weights'))[1]).toBe(1);
    expect(revealAt(s3Near, findIdx(s3Near, (b) => b.tensorKey === 'routed_sum'))[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
  });
});

describe('V3 · Active-Op Part IDs, Floating Cell Popover & Live Zoom-Adaptive Cell Grids', () => {
  it('emits activeOpPartIds, anchored cellPopover, and live Float32 activeCellGrids matching sim arrays', () => {
    const T = 8;
    const focusTok = 6;
    const sim = runMoeSim(T);
    const winners = Array.from(sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6));
    const slot = 2;
    const slotExp = winners[slot];
    const focusedEval = evaluateSingleRoutedExpert(sim, slotExp, focusTok);

    // Idle state: no activeOpPartIds and no cellPopover, but core activeCellGrids remain populated
    const idle = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      viewAngle: 'three_quarter',
    });
    expect(idle.activeOpPartIds).toBeUndefined();
    expect(idle.cellPopover).toBeNull();
    expect(idle.activeCellGrids?.length).toBe(5);

    // Active Step 3 Top-6 inspection
    const scene = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeExpert: slotExp,
      stepperStep: 2,
      stepperActive: true,
      activeCell: {
        tensor: 'topk_weights',
        row: 0,
        col: slot,
        tokenIndex: focusTok,
        expertId: slotExp,
        rankSlot: slot,
      },
      viewAngle: 'three_quarter',
    });

    // 1. activeOpPartIds includes target + operands and excludes unrelated stages
    expect(scene.activeOpPartIds).toBeDefined();
    expect(scene.activeOpPartIds!.has('router')).toBe(true);
    expect(scene.activeOpPartIds!.has('router-unbiased')).toBe(true);
    expect(scene.activeOpPartIds!.has('routed-bank')).toBe(true);
    expect(scene.activeOpPartIds!.has('shared-act')).toBe(false);
    expect(scene.activeOpPartIds!.has('residual-out')).toBe(false);

    // 2. cellPopover matches active Top-6 weight & world position + sourceWorlds
    const pop = scene.cellPopover;
    expect(pop).not.toBeNull();
    const expectedWeight = sim.topk_weights[focusTok * 6 + slot];
    expect(pop!.valueText).toContain(expectedWeight.toFixed(3));
    expect(pop!.coordLabel).toContain(`t${focusTok}`);
    expect(pop!.coordLabel).toContain(formatExpertId(slotExp));
    expect(pop!.equation).toContain('2.5');
    expect(pop!.operandColors).toEqual(['indigo', 'teal', 'red']);
    expect(pop!.sourceWorlds?.length).toBeGreaterThanOrEqual(2);
    expect(pop!.sourceWorlds!.some((s) => s.role === 'indigo')).toBe(true);
    expect(pop!.sourceWorlds!.some((s) => s.role === 'teal')).toBe(true);

    // 3. activeCellGrids: Top-6 (6×1), Unbiased σ (6×1), ReLU² (1×4), Routed Sum (8×1), Shared Out (8×1)
    const grids = scene.activeCellGrids!;
    expect(grids).toHaveLength(5);

    const topkGrid = grids.find((g) => g.colDimLabel === 'Top-6 (×2.5)')!;
    const unbiasedGrid = grids.find((g) => g.colDimLabel === 'Unbiased σ')!;
    const reluGrid = grids.find((g) => g.rowDimLabel?.startsWith('ReLU²'))!;
    const routedSumGrid = grids.find((g) => g.colDimLabel === 'Routed Σ₆')!;
    const sharedOutGrid = grids.find((g) => g.colDimLabel === 'Shared 1×')!;

    expect(topkGrid).toBeDefined();
    expect(unbiasedGrid).toBeDefined();
    expect(reluGrid).toBeDefined();
    expect(routedSumGrid).toBeDefined();
    expect(sharedOutGrid).toBeDefined();

    // Verify popover world anchor coincides with Top-6 cell (slot, 0)
    const expectedCellWorld = computeSheetCellWorldPos(topkGrid, slot, 0, 0.5);
    expect(Math.hypot(
      pop!.world[0] - expectedCellWorld[0],
      pop!.world[1] - expectedCellWorld[1],
      pop!.world[2] - expectedCellWorld[2],
    )).toBeLessThan(0.05);

    // Exact live Float32 array parity & expert ID labels
    expect(topkGrid.cols).toBe(6);
    expect(topkGrid.rows).toBe(1);
    expect(topkGrid.colLabels).toEqual(winners.map(formatExpertId));
    expect(topkGrid.rowLabels).toEqual([`t${focusTok}`]);
    for (let j = 0; j < 6; j++) {
      expect(topkGrid.values[j]).toBe(sim.topk_weights[focusTok * 6 + j]);
      expect(unbiasedGrid.values[j]).toBe(sim.topk_raw_scores[focusTok * 6 + j]);
    }
    expect(unbiasedGrid.colLabels).toEqual(winners.map(formatExpertId));
    expect(unbiasedGrid.rowLabels).toEqual([`t${focusTok}`]);

    expect(reluGrid.cols).toBe(1);
    expect(reluGrid.rows).toBe(4);
    expect(reluGrid.colLabels).toEqual([`${formatExpertId(slotExp)} · t${focusTok}`]);
    for (let h = 0; h < 4; h++) {
      expect(reluGrid.values[h]).toBe(focusedEval.a[h]);
    }

    expect(routedSumGrid.cols).toBe(8);
    expect(routedSumGrid.rows).toBe(1);
    expect(routedSumGrid.rowLabels).toEqual([`t${focusTok}`]);
    expect(sharedOutGrid.cols).toBe(8);
    expect(sharedOutGrid.rows).toBe(1);
    expect(sharedOutGrid.rowLabels).toEqual([`t${focusTok}`]);
    for (let ch = 0; ch < 8; ch++) {
      expect(routedSumGrid.values[ch]).toBe(sim.y_routed[ch * T + focusTok]);
      expect(sharedOutGrid.values[ch]).toBe(sim.y_shared[ch * T + focusTok]);
    }
  });

  it('uses absolute token coordinate t${absFocusToken} in AR decode incremental mode and preserves 2× Shared proportions', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session); // P = 4, S = 1 -> absFocusToken = 4
    const step = getActiveDecodeStep(session);
    const eSim = step.getMoeTrace(2);
    expect(eSim.startTokenOffset).toBe(4);
    const rank0Exp = eSim.topk_indices[0];

    const scene = buildMoECutaway({
      bayId: 2,
      tokens: eSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 0,
      activeExpert: rank0Exp,
      stepperStep: 3,
      stepperActive: true,
      activeCell: {
        tensor: 'routed_act',
        row: 1,
        col: 0,
        tokenIndex: 0,
        expertId: rank0Exp,
      },
      viewAngle: 'three_quarter',
      simOverride: eSim,
    });

    // Popover & all 5 cell grids use exact absolute coordinate t4 (never relative t0!)
    expect(scene.cellPopover?.coordLabel).toContain('t4');
    const grids = scene.activeCellGrids!;
    expect(grids).toHaveLength(5);
    for (const g of grids) {
      const hasT4 =
        (g.rowLabels?.some((l) => l.includes('t4')) ?? false) ||
        (g.colLabels?.some((l) => l.includes('t4')) ?? false);
      expect(hasT4, `expected t4 in labels of grid ${g.colDimLabel ?? g.rowDimLabel}`).toBe(true);
    }

    // Safeguard #7 preserved under active V2/V3 mode: Shared Expert strictly 2× intermediate, 1× hidden
    const rUp = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_w_up')!.frame!;
    const sUp = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_up')!.frame!;
    const rDown = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_w_down')!.frame!;
    const sDown = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_down')!.frame!;
    expect(sUp.extent[0]).toBeCloseTo(rUp.extent[0], 5);
    expect(sUp.extent[2]).toBeCloseTo(rUp.extent[2] * 2, 5);
    expect(sDown.extent[0]).toBeCloseTo(rDown.extent[0] * 2, 5);
    expect(sDown.extent[2]).toBeCloseTo(rDown.extent[2], 5);
  });
});

describe('RC1 · Focused-Cell Partial-Sum Accumulation & Non-Spoiling Active Cell Grids', () => {
  it('accumulates Step 3 Top-6 6-term unbiased denominator without spoiling final ×2.5 weight and masks future grid cells in both views', () => {
    const T = 8;
    const focusTok = 5;
    const slot = 2;
    const sim = runMoeSim(T);
    const winners = Array.from(sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6));
    const slotExp = winners[slot];
    const exactW = sim.topk_weights[focusTok * 6 + slot];
    const exactWText = exactW.toFixed(3);

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const scene = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: slotExp,
        stepperStep: 2,
        stepperActive: true,
        termProgress: 0.25,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: slot,
          tokenIndex: focusTok,
          expertId: slotExp,
          rankSlot: slot,
        },
        viewAngle,
      });

      const accum = scene.focusedAccumulation;
      expect(accum, `${viewAngle} focusedAccumulation`).not.toBeNull();
      expect(accum!.active).toBe(true);
      expect(accum!.totalTerms).toBe(6);
      expect(accum!.currentTermIndex).toBe(1);
      expect(accum!.partialSum).not.toBeCloseTo(exactW, 3);

      // Popover displays running partial sum ONLY — never spoils completed final weight
      const pop = scene.cellPopover;
      expect(pop, `${viewAngle} cellPopover`).not.toBeNull();
      expect(pop!.isAccumulating).toBe(true);
      expect(pop!.badge).toBe(accum!.badgeText);
      expect(pop!.equation).toBe(accum!.popoverEquation);
      expect(pop!.valueText).toBe(accum!.valueBadgeText);
      expect(pop!.equation).not.toContain(exactWText);
      expect(pop!.valueText).not.toContain(exactWText);

      const grids = scene.activeCellGrids!;
      const topkGrid = grids.find((g) => g.colDimLabel === 'Top-6 (×2.5)')!;
      const reluGrid = grids.find((g) => g.rowDimLabel?.startsWith('ReLU²'))!;
      const routedSumGrid = grids.find((g) => g.colDimLabel === 'Routed Σ₆')!;
      const sharedOutGrid = grids.find((g) => g.colDimLabel === 'Shared 1×')!;

      // Top-6 grid: frontier at slot, activeValueOverride = running partialSum, future slots strictly masked
      expect(topkGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(topkGrid.revealColFrontier).toBe(slot);
      expect(topkGrid.revealRowFrontier).toBe(0);
      expect(topkGrid.activeCol).toBe(slot);
      expect(topkGrid.activeRow).toBe(0);
      expect(topkGrid.activeValueOverride).toBe(accum!.partialSum);
      for (let j = 0; j < 6; j++) {
        expect(isCellHiddenByReveal(j, 0, topkGrid), `${viewAngle} topk col ${j}`).toBe(j > slot);
      }

      // Unbiased σ grid (Spatial 3/4) stays 100% revealed (SHEET_REVEAL_NONE) during Step 3!
      if (viewAngle === 'three_quarter') {
        const unbiasedGrid = grids.find((g) => g.colDimLabel === 'Unbiased σ')!;
        expect(unbiasedGrid.revealMode).toBe(SHEET_REVEAL_NONE);
        expect(unbiasedGrid.activeValueOverride).toBeUndefined();
        for (let j = 0; j < 6; j++) {
          expect(isCellHiddenByReveal(j, 0, unbiasedGrid)).toBe(false);
        }

        // Safeguard #7 2× Shared Expert proportions remain exact during RC1 accumulation
        const rUp = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_w_up')!.frame!;
        const sUp = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_up')!.frame!;
        const rDown = scene.solidPickBoxes.find((b) => b.tensorKey === 'routed_w_down')!.frame!;
        const sDown = scene.solidPickBoxes.find((b) => b.tensorKey === 'shared_w_down')!.frame!;
        expect(sUp.extent[0]).toBeCloseTo(rUp.extent[0], 5);
        expect(sUp.extent[2]).toBeCloseTo(rUp.extent[2] * 2, 5);
        expect(sDown.extent[0]).toBeCloseTo(rDown.extent[0] * 2, 5);
        expect(sDown.extent[2]).toBeCloseTo(rDown.extent[2], 5);
      }

      // Downstream grids remain strictly UNCOMPUTED so zero future numbers can leak
      expect(reluGrid.revealMode).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(isCellHiddenByReveal(0, 0, reluGrid)).toBe(true);
      expect(routedSumGrid.revealMode).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(isCellHiddenByReveal(0, 0, routedSumGrid)).toBe(true);
      expect(sharedOutGrid.revealMode).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(isCellHiddenByReveal(0, 0, sharedOutGrid)).toBe(true);

      // At termProgress >= 0.999, accumulation clears while reveal frontiers remain intact
      const settled = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: slotExp,
        stepperStep: 2,
        stepperActive: true,
        termProgress: 0.999,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: slot,
          tokenIndex: focusTok,
          expertId: slotExp,
          rankSlot: slot,
        },
        viewAngle,
      });
      expect(settled.focusedAccumulation).toBeNull();
      expect(settled.cellPopover?.isAccumulating).toBeFalsy();
      expect(settled.cellPopover?.valueText).toContain(exactWText);
      const settledTopk = settled.activeCellGrids!.find((g) => g.colDimLabel === 'Top-6 (×2.5)')!;
      expect(settledTopk.activeValueOverride).toBeUndefined();
      expect(settledTopk.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(settledTopk.revealColFrontier).toBe(slot);
    }
  });

  it('accumulates Step 4 Routed Sum and Step 6 MoE Out 6-branch sums without spoiling and propagates token-slice reveal frontiers', () => {
    const T = 8;
    const focusTok = 6;
    const ch = 3;
    const sim = runMoeSim(T);
    const rank0Exp = sim.topk_indices[focusTok * 6];
    const exactRouted = sim.y_routed[ch * T + focusTok];
    const exactRoutedText = fmtSigned(exactRouted, 3);
    const exactMoe = sim.y_moe[ch * T + focusTok];
    const exactMoeText = fmtSigned(exactMoe, 3);

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // --- Step 4 (stepperStep = 3): Routed Sum 6-branch weighted aggregation ---
      const s4 = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: rank0Exp,
        stepperStep: 3,
        stepperActive: true,
        termProgress: 0.25,
        activeCell: {
          tensor: 'routed_sum',
          row: focusTok,
          col: ch,
          tokenIndex: focusTok,
          expertId: rank0Exp,
        },
        viewAngle,
      });

      const s4Acc = s4.focusedAccumulation;
      expect(s4Acc, `${viewAngle} s4 accum`).not.toBeNull();
      expect(s4Acc!.active).toBe(true);
      expect(s4Acc!.totalTerms).toBe(6);
      expect(s4Acc!.currentTermIndex).toBe(1);
      expect(s4.cellPopover?.isAccumulating).toBe(true);
      expect(s4.cellPopover?.badge).toBe(s4Acc!.badgeText);
      expect(s4.cellPopover?.equation).toBe(s4Acc!.popoverEquation);
      expect(s4.cellPopover?.valueText).toBe(s4Acc!.valueBadgeText);
      expect(s4.cellPopover?.equation).not.toContain(exactRoutedText);
      expect(s4.cellPopover?.valueText).not.toContain(exactRoutedText);

      // Token-row slice reveal on Routed Σ₆ maps full-sheet rowFrontier=focusTok cleanly onto row=0
      const s4Grids = s4.activeCellGrids!;
      const routedGrid = s4Grids.find((g) => g.colDimLabel === 'Routed Σ₆')!;
      const sharedOutGrid = s4Grids.find((g) => g.colDimLabel === 'Shared 1×')!;
      expect(routedGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(routedGrid.revealColFrontier).toBe(ch);
      expect(routedGrid.revealRowFrontier).toBe(0);
      expect(routedGrid.activeCol).toBe(ch);
      expect(routedGrid.activeRow).toBe(0);
      expect(routedGrid.activeValueOverride).toBe(s4Acc!.partialSum);
      for (let c = 0; c < sim.d; c++) {
        expect(isCellHiddenByReveal(c, 0, routedGrid), `${viewAngle} routed_sum col ${c}`).toBe(c > ch);
      }
      // Downstream Shared 1× remains UNCOMPUTED during Step 4
      expect(sharedOutGrid.revealMode).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(isCellHiddenByReveal(0, 0, sharedOutGrid)).toBe(true);

      // GPU sheetRevealData for routed_sum agrees with frontier (col=ch, row=focusTok)
      const rsIdx = s4.solidPickBoxes.findIndex((b) => b.tensorKey === 'routed_sum');
      expect(rsIdx).toBeGreaterThanOrEqual(0);
      const rsGpuRev = revealAt(s4, rsIdx);
      expect(rsGpuRev[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(rsGpuRev[1]).toBe(ch);
      expect(rsGpuRev[2]).toBe(focusTok);

      // --- Step 6 (stepperStep = 5): MoE Out 6-branch sum + shared merge ---
      const s6 = buildMoECutaway({
        bayId: 52,
        tokens: T,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: rank0Exp,
        stepperStep: 5,
        stepperActive: true,
        termProgress: 0.25,
        activeCell: {
          tensor: 'moe_out',
          row: focusTok,
          col: ch,
          tokenIndex: focusTok,
        },
        viewAngle,
      });

      const s6Acc = s6.focusedAccumulation;
      expect(s6Acc, `${viewAngle} s6 accum`).not.toBeNull();
      expect(s6Acc!.active).toBe(true);
      expect(s6Acc!.totalTerms).toBe(6);
      expect(s6Acc!.currentTermIndex).toBe(1);
      expect(s6.cellPopover?.isAccumulating).toBe(true);
      expect(s6.cellPopover?.badge).toBe(s6Acc!.badgeText);
      expect(s6.cellPopover?.equation).toBe(s6Acc!.popoverEquation);
      expect(s6.cellPopover?.valueText).toBe(s6Acc!.valueBadgeText);
      expect(s6.cellPopover?.equation).not.toContain(exactMoeText);
      expect(s6.cellPopover?.valueText).not.toContain(exactMoeText);

      const s6Grids = s6.activeCellGrids!;
      const moeOutGrid = s6Grids.find((g) => g.colDimLabel === 'MoE Out')!;
      const s6RoutedGrid = s6Grids.find((g) => g.colDimLabel === 'Routed Σ₆')!;
      const s6SharedGrid = s6Grids.find((g) => g.colDimLabel === 'Shared 1×')!;
      expect(moeOutGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(moeOutGrid.revealColFrontier).toBe(ch);
      expect(moeOutGrid.revealRowFrontier).toBe(0);
      expect(moeOutGrid.activeCol).toBe(ch);
      expect(moeOutGrid.activeRow).toBe(0);
      expect(moeOutGrid.activeValueOverride).toBe(s6Acc!.partialSum);
      for (let c = 0; c < sim.d; c++) {
        expect(isCellHiddenByReveal(c, 0, moeOutGrid), `${viewAngle} moe_out col ${c}`).toBe(c > ch);
      }
      // Upstream Routed Σ₆ and Shared 1× are fully computed (SHEET_REVEAL_NONE) in Step 6
      expect(s6RoutedGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(s6SharedGrid.revealMode).toBe(SHEET_REVEAL_NONE);

      // Final term (termProgress = 0.95) converges cleanly to exact Float32 y_moe
      const s6Final = buildMoECutaway({
        bayId: 52,
        tokens: T,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: rank0Exp,
        stepperStep: 5,
        stepperActive: true,
        termProgress: 0.95,
        activeCell: {
          tensor: 'moe_out',
          row: focusTok,
          col: ch,
          tokenIndex: focusTok,
        },
        viewAngle,
      });
      expect(s6Final.focusedAccumulation?.currentTermIndex).toBe(5);
      expect(Math.abs(s6Final.focusedAccumulation!.partialSum - exactMoe)).toBeLessThan(1e-5);
    }
  });

  it('keeps ordinary unrestricted inspection intact when stepperActive is false or reducedMotion is true', () => {
    const T = 8;
    const focusTok = 6;
    const slot = 1;
    const sim = runMoeSim(T);
    const slotExp = sim.topk_indices[focusTok * 6 + slot];
    const exactWText = sim.topk_weights[focusTok * 6 + slot].toFixed(3);

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const hoverUnrestricted = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: 'router',
        focusToken: focusTok,
        activeExpert: slotExp,
        stepperStep: 2,
        stepperActive: false,
        termProgress: 0.25,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: slot,
          tokenIndex: focusTok,
          expertId: slotExp,
          rankSlot: slot,
        },
        viewAngle,
      });

      expect(hoverUnrestricted.focusedAccumulation).toBeNull();
      expect(hoverUnrestricted.sheetRevealData).toBeUndefined();
      expect(hoverUnrestricted.cellPopover?.isAccumulating).toBeFalsy();
      expect(hoverUnrestricted.cellPopover?.valueText).toContain(exactWText);
      for (const g of hoverUnrestricted.activeCellGrids!) {
        expect(g.revealMode).toBeUndefined();
        expect(g.revealColFrontier).toBeUndefined();
        expect(g.revealRowFrontier).toBeUndefined();
        expect(g.activeValueOverride).toBeUndefined();
      }

      const reducedUnrestricted = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: slotExp,
        stepperStep: 2,
        stepperActive: true,
        reducedMotion: true,
        termProgress: 0.25,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: slot,
          tokenIndex: focusTok,
          expertId: slotExp,
          rankSlot: slot,
        },
        viewAngle,
      });

      expect(reducedUnrestricted.focusedAccumulation).toBeNull();
      expect(reducedUnrestricted.sheetRevealData).toBeUndefined();
      expect(reducedUnrestricted.cellPopover?.isAccumulating).toBeFalsy();
      expect(reducedUnrestricted.cellPopover?.valueText).toContain(exactWText);
      for (const g of reducedUnrestricted.activeCellGrids!) {
        expect(g.revealMode).toBeUndefined();
        expect(g.activeValueOverride).toBeUndefined();
      }
    }
  });
});
