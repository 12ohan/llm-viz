import { describe, expect, it } from 'vitest';

import { createDefaultHybridModel } from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeStep,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import {
  aabbCorners,
  buildCameraSnapshot,
  fitOpBoundsCamera,
  makeViewportMetrics,
  projectWorldToHostCss,
} from '../src/render/camera';
import {
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
} from '../src/render/sheets';
import { sheetFrameWorldAabb } from '../src/render/sheetFrame';
import {
  buildGqaCutaway,
  fitSpatialGqaCamera,
  GQA_SPATIAL_FIT_BOUNDS,
  type CutawaySceneGeometry,
} from '../src/scene/cutaways';

function revealTupleAt(
  scene: CutawaySceneGeometry,
  idx: number,
): readonly [number, number, number, number] {
  const rd = scene.sheetRevealData!;
  const off = idx * SHEET_REVEAL_FLOATS;
  return [rd[off], rd[off + 1], rd[off + 2], rd[off + 3]];
}

function sheetIndicesBy(
  scene: CutawaySceneGeometry,
  pred: (box: CutawaySceneGeometry['solidPickBoxes'][number]) => boolean,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < scene.solidPickBoxes.length; i++) {
    if (pred(scene.solidPickBoxes[i])) out.push(i);
  }
  return out;
}

describe('V1/V2/V3 GQA Cutaway: Label Hygiene, Progressive Accumulation Reveal, Popover & Cell Grids', () => {
  it('V1: anchors Q16..Q31 exterior to its stack and separates K0, Scores, and T=N cleanly in overview and Step 3 zoom', () => {
    const scene = buildGqaCutaway({
      bayId: 43,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    const gQ1 = scene.landmarks.find((l) => l.id === 'g-q1');
    const gQ0 = scene.landmarks.find((l) => l.id === 'g-q0');
    const gKv0K = scene.landmarks.find((l) => l.id === 'g-kv0-k');
    const gScores0 = scene.landmarks.find((l) => l.id === 'g-scores-0');
    const calScoresT = scene.calipers?.find((c) => c.id === 'cal-scores-t');

    expect(gQ1).toBeDefined();
    expect(gQ0).toBeDefined();
    expect(gKv0K).toBeDefined();
    expect(gScores0).toBeDefined();
    expect(calScoresT).toBeDefined();

    // Compute world and projected screen bounding envelope of Q16..Q31 cards
    const q16to31Boxes = scene.solidPickBoxes.filter(
      (b) => b.tensorKey === 'q' && b.headIndex !== undefined && b.headIndex >= 16,
    );
    expect(q16to31Boxes).toHaveLength(16);

    const minQ1X = Math.min(
      ...q16to31Boxes.map((b) => (b.frame ? sheetFrameWorldAabb(b.frame).min[0] : b.min[0])),
    );
    const maxQ1Z = Math.max(
      ...q16to31Boxes.map((b) => (b.frame ? sheetFrameWorldAabb(b.frame).max[2] : b.max[2])),
    );
    // g-q1 sits strictly exterior to the front-left perimeter of the Q16..Q31 stack with left alignment
    expect(gQ1!.align).toBe('left');
    expect(gQ1!.world[0]).toBeLessThan(minQ1X);
    expect(gQ1!.world[2]).toBeGreaterThan(maxQ1Z);

    const metrics = makeViewportMetrics(0, 0, 1076, 900, 1);
    const gKv1K = scene.landmarks.find((l) => l.id === 'g-kv1-k')!;
    for (const stageW of [1076, 680]) {
      const m = makeViewportMetrics(0, 0, stageW, stageW === 1076 ? 900 : 800, 1);
      const overviewPose = fitSpatialGqaCamera(GQA_SPATIAL_FIT_BOUNDS, m);
      const overviewSnap = buildCameraSnapshot(overviewPose, GQA_SPATIAL_FIT_BOUNDS, m);

      // Projected g-q1 anchor (align: 'left' extends leftward) sits strictly left of every Q16..Q31 card
      let q1StackMinScreenX = Infinity;
      for (const box of q16to31Boxes) {
        const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
        for (const c of aabbCorners(aabb)) {
          const pt = projectWorldToHostCss(c, overviewSnap);
          if (pt.visible && pt.x < q1StackMinScreenX) q1StackMinScreenX = pt.x;
        }
      }
      const gQ1Pt = projectWorldToHostCss(gQ1!.world, overviewSnap);
      const gKv1Pt = projectWorldToHostCss(gKv1K.world, overviewSnap);
      expect(gQ1Pt.visible).toBe(true);
      expect(gQ1Pt.x).toBeLessThan(q1StackMinScreenX);
      expect(gKv1Pt.x - gQ1Pt.x).toBeGreaterThan(75);
    }

    // Verify Step 3 Group 0 zoom separates K0 · FP8, Scores · Q00×K0, and T=8 caliper clearly from the active score cell
    const step3Scene = buildGqaCutaway({
      bayId: 43,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
      stepperActive: true,
      stepperStep: 2,
      activeCell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 7, col: 2 },
    });
    const opPose = fitOpBoundsCamera(step3Scene.opFocusBounds![2], metrics, {
      azimuthDeg: 20,
      elevationDeg: 23.5,
      topInsetPx: 84,
      bottomInsetPx: 118,
      fillFraction: 0.8,
    });
    const opSnap = buildCameraSnapshot(opPose, GQA_SPATIAL_FIT_BOUNDS, metrics);

    const kv0Pt = projectWorldToHostCss(
      step3Scene.landmarks.find((l) => l.id === 'g-kv0-k')!.world,
      opSnap,
    );
    const scores0Pt = projectWorldToHostCss(
      step3Scene.landmarks.find((l) => l.id === 'g-scores-0')!.world,
      opSnap,
    );
    const activeCellPt = projectWorldToHostCss(step3Scene.cellPopover!.world, opSnap);
    const calT = step3Scene.calipers!.find((c) => c.id === 'cal-scores-t')!;
    const calMidWorld: readonly [number, number, number] = [
      0.5 * (calT.aWorld[0] + calT.bWorld[0]),
      0.5 * (calT.aWorld[1] + calT.bWorld[1]),
      0.5 * (calT.aWorld[2] + calT.bWorld[2]),
    ];
    const calTPt = projectWorldToHostCss(calMidWorld, opSnap);

    // Generous screen separation between K0 label, Scores · Q00×K0 label, T=8 caliper, and active cell
    expect(Math.hypot(kv0Pt.x - scores0Pt.x, kv0Pt.y - scores0Pt.y)).toBeGreaterThan(80);
    expect(Math.hypot(scores0Pt.x - activeCellPt.x, scores0Pt.y - activeCellPt.y)).toBeGreaterThan(50);
    expect(Math.hypot(calTPt.x - activeCellPt.x, calTPt.y - activeCellPt.y)).toBeGreaterThan(30);
    expect(Math.hypot(calTPt.x - scores0Pt.x, calTPt.y - scores0Pt.y)).toBeGreaterThan(90);
  });

  it('V2: omits sheetRevealData by default and when reducedMotion is true', () => {
    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const idle = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
      });
      expect(idle.sheetRevealData).toBeUndefined();

      const reduced = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        stepperActive: true,
        stepperStep: 2,
        reducedMotion: true,
        activeCell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 7, col: 3 },
      });
      expect(reduced.sheetRevealData).toBeUndefined();

      const noStep = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        stepperActive: true,
        stepperStep: null,
      });
      expect(noStep.sheetRevealData).toBeUndefined();
    }
  });

  it('V2: populates exact progressive frontiers across Steps 1–6 in Full Seq (T=8) for both Spatial 3/4 and Near-Plan', () => {
    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // Step 1 (stepperStep = 0): RMSNorm reveals row-major; downstream Q/KV/Scores/Softmax/V_out/Concat/Residual uncomputed
      const step1 = buildGqaCutaway({
        bayId: 6,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        stepperActive: true,
        stepperStep: 0,
        activeCell: { tensor: 'rmsnorm', row: 7, col: 3 },
      });
      expect(step1.sheetRevealData?.length).toBe(step1.sheetCount! * SHEET_REVEAL_FLOATS);
      const rmsIdx = step1.solidPickBoxes.findIndex((b) => b.tensorKey === 'rmsnorm');
      const inputIdx = step1.solidPickBoxes.findIndex((b) => b.tensorKey === 'input');
      expect(revealTupleAt(step1, inputIdx)[0]).toBe(SHEET_REVEAL_NONE);
      expect(revealTupleAt(step1, rmsIdx).slice(0, 3)).toEqual([SHEET_REVEAL_ROW_MAJOR, 3, 7]);
      for (const downKey of ['q', 'kv_k', 'kv_v', 'scores', 'softmax', 'v_out', 'concat', 'residual']) {
        const indices = sheetIndicesBy(step1, (b) => b.tensorKey === downKey);
        expect(indices.length).toBeGreaterThan(0);
        for (const i of indices) {
          expect(revealTupleAt(step1, i)[0], `${viewAngle} step1 ${downKey}`).toBe(SHEET_REVEAL_UNCOMPUTED);
        }
      }

      // Step 3 (stepperStep = 2): Scores uses ROW_MAJOR with exact (col, row); Softmax/V_out/Concat/Residual UNCOMPUTED
      const step3 = buildGqaCutaway({
        bayId: 6,
        tokens: 8,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
        stepperActive: true,
        stepperStep: 2,
        activeCell: { tensor: 'scores', headIndex: 18, kvHeadIndex: 1, row: 7, col: 4 },
      });
      const scores1Idx = step3.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'scores' && b.kvHeadIndex === 1,
      );
      const soft1Idx = step3.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 1,
      );
      const soft0Idx = step3.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 0,
      );
      const q18Idx = step3.solidPickBoxes.findIndex((b) => b.tensorKey === 'q' && b.headIndex === 18);
      expect(revealTupleAt(step3, q18Idx)[0]).toBe(SHEET_REVEAL_NONE);
      const s3Tuple = revealTupleAt(step3, scores1Idx);
      expect(s3Tuple.slice(0, 3)).toEqual([SHEET_REVEAL_ROW_MAJOR, 4, 7]);
      expect(s3Tuple[3]).toBeGreaterThan(0);
      expect(revealTupleAt(step3, soft1Idx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(revealTupleAt(step3, soft0Idx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      for (const downKey of ['v_out', 'concat', 'residual']) {
        for (const i of sheetIndicesBy(step3, (b) => b.tensorKey === downKey)) {
          expect(revealTupleAt(step3, i)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
        }
      }

      // Step 4 (stepperStep = 3): Scores 100% revealed; Softmax uses ROW_MAJOR; V_out/Concat/Residual UNCOMPUTED
      const step4 = buildGqaCutaway({
        bayId: 6,
        tokens: 8,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
        stepperActive: true,
        stepperStep: 3,
        activeCell: { tensor: 'softmax', headIndex: 18, kvHeadIndex: 1, row: 7, col: 5 },
      });
      const s4ScoresIdx = step4.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'scores' && b.kvHeadIndex === 1,
      );
      const s4SoftIdx = step4.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 1,
      );
      expect(revealTupleAt(step4, s4ScoresIdx)[0]).toBe(SHEET_REVEAL_NONE);
      expect(revealTupleAt(step4, s4SoftIdx).slice(0, 3)).toEqual([SHEET_REVEAL_ROW_MAJOR, 5, 7]);
      for (const downKey of ['v_out', 'concat', 'residual']) {
        for (const i of sheetIndicesBy(step4, (b) => b.tensorKey === downKey)) {
          expect(revealTupleAt(step4, i)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
        }
      }

      // Step 5 (stepperStep = 4): Scores & Softmax revealed; V_out ROW_MAJOR; Concat & Residual UNCOMPUTED
      const step5 = buildGqaCutaway({
        bayId: 6,
        tokens: 8,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
        stepperActive: true,
        stepperStep: 4,
        activeCell: { tensor: 'v_out', headIndex: 18, kvHeadIndex: 1, row: 7, col: 10 },
      });
      const s5VOutIdx = step5.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'v_out' && b.kvHeadIndex === 1,
      );
      expect(revealTupleAt(step5, s5VOutIdx).slice(0, 3)).toEqual([SHEET_REVEAL_ROW_MAJOR, 10, 7]);
      expect(
        revealTupleAt(step5, step5.solidPickBoxes.findIndex((b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 1))[0],
      ).toBe(SHEET_REVEAL_NONE);
      expect(revealTupleAt(step5, step5.solidPickBoxes.findIndex((b) => b.tensorKey === 'concat'))[0]).toBe(
        SHEET_REVEAL_UNCOMPUTED,
      );
      expect(revealTupleAt(step5, step5.solidPickBoxes.findIndex((b) => b.tensorKey === 'residual'))[0]).toBe(
        SHEET_REVEAL_UNCOMPUTED,
      );

      // Step 6 (stepperStep = 5): Residual uses ROW_MAJOR frontier; Concat & W_O revealed
      const step6 = buildGqaCutaway({
        bayId: 6,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        stepperActive: true,
        stepperStep: 5,
        activeCell: { tensor: 'residual', row: 7, col: 6 },
      });
      expect(revealTupleAt(step6, step6.solidPickBoxes.findIndex((b) => b.tensorKey === 'concat'))[0]).toBe(
        SHEET_REVEAL_NONE,
      );
      expect(revealTupleAt(step6, step6.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_o'))[0]).toBe(
        SHEET_REVEAL_NONE,
      );
      expect(
        revealTupleAt(step6, step6.solidPickBoxes.findIndex((b) => b.tensorKey === 'residual')).slice(0, 3),
      ).toEqual([SHEET_REVEAL_ROW_MAJOR, 6, 7]);
    }
  });

  it('V2 AR Decode (P=4, S=1, [1×5]): keeps reused prefix KV 100% revealed in Step 2 and reveals rectangular Scores/Softmax in Steps 3 & 4', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session); // P = 4, S = 1, K = 5
    const gqaSim = getActiveDecodeStep(session).getGqaTrace(43);

    expect(gqaSim.prefixTokens).toBe(4);
    expect(gqaSim.tokens).toBe(1);
    expect(gqaSim.keyTokens).toBe(5);

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // Step 2 (Q/K/V): reused prefix KV slices k=0..3 MUST stay SHEET_REVEAL_NONE; new slice k=4 reveals progressively!
      const step2 = buildGqaCutaway({
        bayId: 43,
        tokens: gqaSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: gqaSim,
        stepperActive: true,
        stepperStep: 1,
        activeCell: { tensor: 'q', headIndex: 0, kvHeadIndex: 0, row: 0, col: 2 },
      });

      for (const tensorKey of ['kv_k', 'kv_v'] as const) {
        const slices = sheetIndicesBy(step2, (b) => b.tensorKey === tensorKey);
        expect(slices).toHaveLength(10); // 2 KV heads × 5 keys
        for (const idx of slices) {
          const tok = step2.solidPickBoxes[idx].tokenIndex!;
          const rev = revealTupleAt(step2, idx);
          if (tok < 4) {
            expect(rev[0], `${viewAngle} reused prefix ${tensorKey}[k=${tok}] must NEVER ghost`).toBe(
              SHEET_REVEAL_NONE,
            );
          } else {
            expect(rev.slice(0, 3), `${viewAngle} new ${tensorKey}[k=${tok}] reveals row-major`).toEqual([
              SHEET_REVEAL_ROW_MAJOR,
              2,
              0,
            ]);
          }
        }
      }
      // Downstream Scores & Softmax uncomputed during Step 2
      for (const idx of sheetIndicesBy(step2, (b) => b.tensorKey === 'scores' || b.tensorKey === 'softmax')) {
        expect(revealTupleAt(step2, idx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      }

      // Step 3 (QKᵀ/√d) on rectangular [1×5] Scores at col=2, row=0
      const step3 = buildGqaCutaway({
        bayId: 43,
        tokens: gqaSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: gqaSim,
        stepperActive: true,
        stepperStep: 2,
        activeCell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 0, col: 2 },
      });
      const scores0Idx = step3.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'scores' && b.kvHeadIndex === 0,
      );
      const soft0Idx = step3.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 0,
      );
      expect(revealTupleAt(step3, scores0Idx).slice(0, 3)).toEqual([SHEET_REVEAL_ROW_MAJOR, 2, 0]);
      expect(revealTupleAt(step3, soft0Idx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);

      // Step 4 (Softmax) on rectangular [1×5] Softmax at col=3, row=0
      const step4 = buildGqaCutaway({
        bayId: 43,
        tokens: gqaSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: gqaSim,
        stepperActive: true,
        stepperStep: 3,
        activeCell: { tensor: 'softmax', headIndex: 0, kvHeadIndex: 0, row: 0, col: 3 },
      });
      const s4ScoresIdx = step4.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'scores' && b.kvHeadIndex === 0,
      );
      const s4SoftIdx = step4.solidPickBoxes.findIndex(
        (b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 0,
      );
      expect(revealTupleAt(step4, s4ScoresIdx)[0]).toBe(SHEET_REVEAL_NONE);
      expect(revealTupleAt(step4, s4SoftIdx).slice(0, 3)).toEqual([SHEET_REVEAL_ROW_MAJOR, 3, 0]);
    }
  });

  it('V3: emits exact activeOpPartIds, cellPopover with provenance sourceWorlds, and live activeCellGrids matching sim Float32 arrays', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session); // P = 4, S = 1, K = 5
    const arSim = getActiveDecodeStep(session).getGqaTrace(43);

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // 1. Scores cell trace (Full Seq T=8, Group 1 Head Q18, q=6, k=2)
      const scoresScene = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
        activeCell: { tensor: 'scores', headIndex: 18, kvHeadIndex: 1, row: 6, col: 2 },
      });
      const sim = scoresScene.gqaSim!;

      // activeOpPartIds isolates target + operand A + operand B only
      expect(scoresScene.activeOpPartIds).toBeDefined();
      expect(scoresScene.activeOpPartIds!.has('attn-scores')).toBe(true);
      expect(scoresScene.activeOpPartIds!.has('q-proj')).toBe(true);
      expect(scoresScene.activeOpPartIds!.has('q-head-18')).toBe(true);
      expect(scoresScene.activeOpPartIds!.has('kv-head-1')).toBe(true);
      expect(scoresScene.activeOpPartIds!.has('kv-head-0')).toBe(false);
      expect(scoresScene.activeOpPartIds!.has('softmax')).toBe(false);
      expect(scoresScene.activeOpPartIds!.has('o-proj')).toBe(false);

      // cellPopover carries exact coordinate, value, operand colors, and finite sourceWorlds
      const pop = scoresScene.cellPopover!;
      expect(pop).not.toBeNull();
      expect(pop.tensorLabel).toBe('Scores · Q18');
      expect(pop.coordLabel).toBe('[q=t6, k=t2]');
      expect(pop.badge).toBe('Q18 → KV1');
      expect(pop.operandColors).toEqual(['indigo', 'teal', 'red']);
      expect(pop.valueText).toContain(scoresScene.activeFormula!.valueText);
      expect(pop.world.every(Number.isFinite)).toBe(true);
      expect(pop.sourceWorlds).toHaveLength(2);
      expect(pop.sourceWorlds!.map((s) => s.role)).toEqual(['indigo', 'teal']);

      // activeCellGrids: [Scores(Q18), Q18, K1[t2]] match sim.Scores, sim.Q, sim.K bit-for-bit!
      const grids = scoresScene.activeCellGrids!;
      expect(grids.length).toBeGreaterThanOrEqual(3);
      const scoreGrid = grids[0];
      expect(scoreGrid.cols).toBe(8);
      expect(scoreGrid.rows).toBe(8);
      expect(scoreGrid.format).toBe('signed');
      expect(scoreGrid.activeRow).toBe(6);
      expect(scoreGrid.activeCol).toBe(2);
      expect(scoreGrid.colDimLabel).toBe('K = 8');
      expect(scoreGrid.rowDimLabel).toBe('Q = 8');
      expect(scoreGrid.rowLabels?.[6]).toBe(`t6 ${sim.tokenLabels[6]}`);
      expect(scoreGrid.colLabels?.[2]).toBe(`t2 ${sim.keyTokenLabels[2]}`);

      const hOff = 18 * 8 * 8;
      for (let q = 0; q < 8; q++) {
        for (let k = 0; k < 8; k++) {
          const idx = q * 8 + k;
          expect(scoreGrid.values[idx]).toBe(sim.Scores[hOff + idx]);
          expect(scoreGrid.masked![idx]).toBe(k > q ? 1 : 0);
        }
      }

      const qGrid = grids[1];
      expect(qGrid.cols).toBe(4);
      expect(qGrid.rows).toBe(8);
      for (let t = 0; t < 8; t++) {
        for (let r = 0; r < 4; r++) {
          expect(qGrid.values[t * 4 + r]).toBe(sim.Q[(18 * 4 + r) * 8 + t]);
        }
      }

      const kSliceGrid = grids[2];
      expect(kSliceGrid.cols).toBe(4);
      expect(kSliceGrid.rows).toBe(1);
      for (let r = 0; r < 4; r++) {
        expect(kSliceGrid.values[r]).toBe(sim.K[(1 * 4 + r) * 8 + 2]);
      }

      // 2. Softmax cell trace matches sim.Softmax exactly with format: 'prob'
      const smScene = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
        activeCell: { tensor: 'softmax', headIndex: 18, kvHeadIndex: 1, row: 6, col: 2 },
      });
      const smGrid = smScene.activeCellGrids![0];
      expect(smGrid.format).toBe('prob');
      expect(smGrid.activeRow).toBe(6);
      expect(smGrid.activeCol).toBe(2);
      for (let q = 0; q < 8; q++) {
        for (let k = 0; k < 8; k++) {
          const idx = q * 8 + k;
          expect(smGrid.values[idx]).toBe(smScene.gqaSim!.Softmax[hOff + idx]);
          expect(smGrid.masked![idx]).toBe(k > q ? 1 : 0);
        }
      }

      // 3. V_out cell trace matches sim.V_out, sim.Softmax, and sim.V exactly
      const vOutScene = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
        activeCell: { tensor: 'v_out', headIndex: 18, kvHeadIndex: 1, row: 4, col: 9 },
      });
      const vOutGrid = vOutScene.activeCellGrids![0];
      expect(vOutGrid.cols).toBe(64);
      expect(vOutGrid.rows).toBe(8);
      expect(vOutGrid.values[4 * 64 + 9]).toBe(vOutScene.gqaSim!.V_out[(18 * 4 + 1) * 8 + 4]);
      // Active V1 slice at k=4 matches sim.V exactly
      const vSlice4Grid = vOutScene.activeCellGrids!.find((g) => g.rowDimLabel === 'V1[t4]');
      expect(vSlice4Grid).toBeDefined();
      for (let r = 0; r < 4; r++) {
        expect(vSlice4Grid!.values[r]).toBe(vOutScene.gqaSim!.V[(1 * 4 + r) * 8 + 4]);
      }

      // 4. AR Decode rectangular [1×5] Scores activeCellGrid has absolute token label t4 and 0 masked cells
      const arScoresScene = buildGqaCutaway({
        bayId: 43,
        tokens: arSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: arSim,
        activeCell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 0, col: 2 },
      });
      const arGrid = arScoresScene.activeCellGrids![0];
      expect(arGrid.cols).toBe(5);
      expect(arGrid.rows).toBe(1);
      expect(arGrid.colDimLabel).toBe('K = 5 (4+1)');
      expect(arGrid.rowDimLabel).toBe('Q = 1');
      expect(arGrid.rowLabels).toEqual([`t4 ${arSim.tokenLabels[0]}`]);
      expect(Array.from(arGrid.masked!)).toEqual([0, 0, 0, 0, 0]);
      for (let k = 0; k < 5; k++) {
        expect(arGrid.values[k]).toBe(arSim.Scores[k]);
      }
      expect(arScoresScene.cellPopover?.coordLabel).toBe('[q=t4, k=t2]');

      // 5. Idle cutaway has no activeOpPartIds and null cellPopover
      const idle = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
      });
      expect(idle.activeOpPartIds).toBeUndefined();
      expect(idle.cellPopover).toBeNull();
      expect(idle.activeCellGrids!.length).toBeGreaterThan(0);
    }
  });
});
