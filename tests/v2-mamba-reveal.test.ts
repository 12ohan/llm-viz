import { describe, expect, it } from 'vitest';

import { createDefaultHybridModel } from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeStep,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import {
  mambaStepperCellForState,
  runMambaSim,
  type MambaCellRef,
} from '../src/math/mambaSim';
import {
  buildCameraSnapshot,
  makeViewportMetrics,
  projectWorldToHostCss,
} from '../src/render/camera';
import {
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
} from '../src/render/sheets';
import {
  buildMambaCutaway,
  fitSpatialMambaCamera,
  MAMBA_SPATIAL_FIT_BOUNDS,
  type CutawaySceneGeometry,
} from '../src/scene/cutaways';

function revealTupleAt(scene: CutawaySceneGeometry, sheetIdx: number): [number, number, number, number] {
  const rd = scene.sheetRevealData!;
  const o = sheetIdx * SHEET_REVEAL_FLOATS;
  return [rd[o], rd[o + 1], rd[o + 2], rd[o + 3]];
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

describe('V1 · Mamba-2 Spatial Label Hygiene (Zero Front-Left Crowding)', () => {
  it('separates Conv1D, Δt, and x: H00–H15 cleanly with zero overlap across heads H00/H07/H15 and T=1/8/16', () => {
    const desktopMetrics = makeViewportMetrics(0, 0, 1120, 900, 1);
    const desktopPose = fitSpatialMambaCamera(MAMBA_SPATIAL_FIT_BOUNDS, desktopMetrics, {
      topInsetPx: 84,
      bottomInsetPx: 118,
      isNarrow: false,
    });
    const desktopSnap = buildCameraSnapshot(desktopPose, MAMBA_SPATIAL_FIT_BOUNDS, desktopMetrics);

    const narrowMetrics = makeViewportMetrics(0, 0, 720, 800, 1);
    const narrowPose = fitSpatialMambaCamera(MAMBA_SPATIAL_FIT_BOUNDS, narrowMetrics, {
      topInsetPx: 84,
      bottomInsetPx: 118,
      isNarrow: true,
    });
    const narrowSnap = buildCameraSnapshot(narrowPose, MAMBA_SPATIAL_FIT_BOUNDS, narrowMetrics);

    const projectLabelBox = (
      lm: CutawaySceneGeometry['landmarks'][number],
      snap: ReturnType<typeof buildCameraSnapshot>,
    ) => {
      const pt = projectWorldToHostCss(lm.world, snap);
      expect(pt.visible).toBe(true);
      const w = Math.ceil(lm.label.length * 6.3) + 12;
      const h = 18;
      let lx = pt.x - w * 0.5;
      let ly = pt.y - h * 0.5;
      if (lm.align === 'left') lx = pt.x - w - 4;
      else if (lm.align === 'right') lx = pt.x + 4;
      else if (lm.align === 'top') ly = pt.y - h - 4;
      else if (lm.align === 'bottom') ly = pt.y + 4;
      return { id: lm.id, x0: lx, y0: ly, x1: lx + w, y1: ly + h };
    };

    const rectsOverlap = (
      a: { x0: number; y0: number; x1: number; y1: number },
      b: { x0: number; y0: number; x1: number; y1: number },
      pad = 2,
    ) => a.x0 < b.x1 + pad && a.x1 + pad > b.x0 && a.y0 < b.y1 + pad && a.y1 + pad > b.y0;

    for (const activeHead of [0, 7, 15]) {
      for (const tokens of [1, 8, 16]) {
        const scene = buildMambaCutaway({
          bayId: 1,
          tokens,
          precision: 'fp8_selective',
          selectedPart: null,
          hoveredPart: null,
          activeHead,
          viewAngle: 'three_quarter',
        });

        const primaries = scene.landmarks.filter((lm) => lm.tier !== 'detail');
        const desktopBoxes = primaries.map((lm) => projectLabelBox(lm, desktopSnap));

        // Zero pairwise overlap across all 17 primary labels on desktop stage
        for (let i = 0; i < desktopBoxes.length; i++) {
          for (let j = i + 1; j < desktopBoxes.length; j++) {
            expect(
              rectsOverlap(desktopBoxes[i], desktopBoxes[j]),
              `Desktop H${activeHead} T=${tokens}: ${desktopBoxes[i].id} overlaps ${desktopBoxes[j].id}`,
            ).toBe(false);
          }
        }

        // Front-left trio (Conv1D, Δt, x: H00–H15) stays strictly non-overlapping even on narrow 720×800 stage
        const trioIds = ['m-conv1d', 'm-dt', 'm-x-heads'] as const;
        const narrowTrio = trioIds.map((id) => {
          const lm = primaries.find((l) => l.id === id)!;
          expect(lm, `Missing landmark ${id}`).toBeDefined();
          return projectLabelBox(lm, narrowSnap);
        });
        for (let i = 0; i < narrowTrio.length; i++) {
          for (let j = i + 1; j < narrowTrio.length; j++) {
            expect(
              rectsOverlap(narrowTrio[i], narrowTrio[j]),
              `Narrow H${activeHead} T=${tokens}: ${narrowTrio[i].id} overlaps ${narrowTrio[j].id}`,
            ).toBe(false);
          }
        }
      }
    }
  });
});

describe('V2 · Mamba-2 Progressive Accumulation Reveal (sheetRevealData)', () => {
  it('omits sheetRevealData when stepperActive is false/omitted or reducedMotion is true', () => {
    const activeCell: MambaCellRef = {
      tensor: 'h_curr',
      row: 2,
      col: 1,
      headIndex: 0,
      groupIndex: 0,
      tokenIndex: 7,
    };

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const idle = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeCell,
        stepperStep: 3,
        viewAngle,
      });
      expect(idle.sheetRevealData).toBeUndefined();

      const reduced = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeCell,
        stepperStep: 3,
        stepperActive: true,
        reducedMotion: true,
        viewAngle,
      });
      expect(reduced.sheetRevealData).toBeUndefined();
    }
  });

  it('enforces Step 4 H_t ROW_MAJOR frontier, H_{t-1} NONE, and downstream y/y_norm/Residual UNCOMPUTED in Full Seq and seeded AR Decode', () => {
    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // 1. Full Seq (T=8) Step 4 (stepperStep === 3) at H_t[H02, p=2, n=1]
      const fullStep4 = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: 2,
        activeCell: {
          tensor: 'h_curr',
          row: 2,
          col: 1,
          headIndex: 2,
          groupIndex: 1,
          tokenIndex: 7,
        },
        stepperStep: 3,
        stepperActive: true,
        stepperSubProgress: 0.56,
        reducedMotion: false,
        viewAngle,
      });

      expect(fullStep4.sheetRevealData).toBeDefined();
      expect(fullStep4.sheetRevealData!.length).toBe(
        (fullStep4.sheetCount ?? 0) * SHEET_REVEAL_FLOATS,
      );

      const hCurrIdx = sheetIndicesBy(fullStep4, (b) => b.tensorKey === 'h_curr')[0];
      const hPrevIdx = sheetIndicesBy(fullStep4, (b) => b.tensorKey === 'h_prev')[0];
      const yIdx = sheetIndicesBy(fullStep4, (b) => b.tensorKey === 'y')[0];
      const yNormIdx = sheetIndicesBy(fullStep4, (b) => b.tensorKey === 'y_norm')[0];
      const resIdx = sheetIndicesBy(fullStep4, (b) => b.tensorKey === 'residual')[0];

      const [hCurrMode, hCurrCol, hCurrRow] = revealTupleAt(fullStep4, hCurrIdx);
      expect(hCurrMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(hCurrCol).toBe(1);
      expect(hCurrRow).toBe(2);

      // Carried / prior H_{t-1} ALWAYS stays SHEET_REVEAL_NONE
      expect(revealTupleAt(fullStep4, hPrevIdx)[0]).toBe(SHEET_REVEAL_NONE);

      // Upstream x, B, C, Δt, Conv1D, W_in, RMSNorm, Input stay SHEET_REVEAL_NONE
      for (const upKey of ['input', 'rmsnorm', 'w_in', 'z', 'xbc_pre', 'conv_buf', 'dt', 'x', 'b', 'c'] as const) {
        const indices = sheetIndicesBy(fullStep4, (b) => b.tensorKey === upKey);
        expect(indices.length).toBeGreaterThan(0);
        for (const idx of indices) {
          expect(revealTupleAt(fullStep4, idx)[0], `Step 4 upstream ${upKey}`).toBe(SHEET_REVEAL_NONE);
        }
      }

      // Downstream y, y_norm, residual render UNCOMPUTED
      expect(revealTupleAt(fullStep4, yIdx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(revealTupleAt(fullStep4, yNormIdx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(revealTupleAt(fullStep4, resIdx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);

      // 2. Seeded AR Decode (P=4, S=1 -> abs t4): carried H_{t-1} stays SHEET_REVEAL_NONE across ALL steps (including Step 1 & Step 4)
      const model = createDefaultHybridModel();
      let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
      session = stepDecodeSessionGreedy(session);
      const decodeStep = getActiveDecodeStep(session);
      const mSim = decodeStep.getMambaTrace(1);
      expect(mSim.hasInitialState).toBe(true);

      for (const stepIdx of [0, 3]) {
        const cell = mambaStepperCellForState(mSim, 0, stepIdx, 0.35);
        const arScene = buildMambaCutaway({
          bayId: 1,
          tokens: mSim.tokens,
          precision: 'fp8_selective',
          selectedPart: null,
          hoveredPart: null,
          activeHead: 0,
          activeCell: cell,
          stepperStep: stepIdx,
          stepperActive: true,
          stepperSubProgress: 0.35,
          reducedMotion: false,
          viewAngle,
          simOverride: mSim,
        });
        expect(arScene.sheetRevealData).toBeDefined();
        const arHPrevIdx = sheetIndicesBy(arScene, (b) => b.tensorKey === 'h_prev')[0];
        const arHCurrIdx = sheetIndicesBy(arScene, (b) => b.tensorKey === 'h_curr')[0];
        const arYIdx = sheetIndicesBy(arScene, (b) => b.tensorKey === 'y')[0];
        const arResIdx = sheetIndicesBy(arScene, (b) => b.tensorKey === 'residual')[0];

        expect(revealTupleAt(arScene, arHPrevIdx)[0]).toBe(SHEET_REVEAL_NONE);
        if (stepIdx === 0) {
          expect(revealTupleAt(arScene, arHCurrIdx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
        } else {
          const [mode, col, row] = revealTupleAt(arScene, arHCurrIdx);
          expect(mode).toBe(SHEET_REVEAL_ROW_MAJOR);
          expect(col).toBe(cell.col);
          expect(row).toBe(cell.row);
        }
        expect(revealTupleAt(arScene, arYIdx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
        expect(revealTupleAt(arScene, arResIdx)[0]).toBe(SHEET_REVEAL_UNCOMPUTED);
      }
    }
  });

  it('transitions progressive reveal cleanly across all 6 stepper stages (Steps 0..5)', () => {
    const sim = runMambaSim(8);

    // Step 1 (0): RMSNorm ROW_MAJOR; downstream xBC_pre/z/Δt/Conv1D/x/B/C/H_t/y/y_norm/Residual UNCOMPUTED; H_{t-1} NONE
    const s0 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'rmsnorm', row: 7, col: 3 },
      stepperStep: 0,
      stepperActive: true,
      viewAngle: 'three_quarter',
      simOverride: sim,
    });
    expect(revealTupleAt(s0, sheetIndicesBy(s0, (b) => b.tensorKey === 'rmsnorm')[0]).slice(0, 3)).toEqual([
      SHEET_REVEAL_ROW_MAJOR,
      3,
      7,
    ]);
    expect(revealTupleAt(s0, sheetIndicesBy(s0, (b) => b.tensorKey === 'h_prev')[0])[0]).toBe(SHEET_REVEAL_NONE);
    for (const k of ['xbc_pre', 'z', 'dt', 'conv_buf', 'x', 'b', 'c', 'h_curr', 'y', 'y_norm', 'residual'] as const) {
      for (const idx of sheetIndicesBy(s0, (b) => b.tensorKey === k)) {
        expect(revealTupleAt(s0, idx)[0], `Step 0 downstream ${k}`).toBe(SHEET_REVEAL_UNCOMPUTED);
      }
    }

    // Step 2 (1): xBC_pre / z / Δt reveal progressively; x/B/C/H_t/y/y_norm/Residual UNCOMPUTED
    const s1 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'xbc_pre', row: 7, col: 20 },
      stepperStep: 1,
      stepperActive: true,
      viewAngle: 'three_quarter',
      simOverride: sim,
    });
    expect(revealTupleAt(s1, sheetIndicesBy(s1, (b) => b.tensorKey === 'rmsnorm')[0])[0]).toBe(SHEET_REVEAL_NONE);
    expect(revealTupleAt(s1, sheetIndicesBy(s1, (b) => b.tensorKey === 'xbc_pre')[0]).slice(0, 3)).toEqual([
      SHEET_REVEAL_ROW_MAJOR,
      20,
      7,
    ]);
    expect(revealTupleAt(s1, sheetIndicesBy(s1, (b) => b.tensorKey === 'dt')[0])[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    for (const k of ['x', 'b', 'c', 'h_curr', 'y', 'y_norm', 'residual'] as const) {
      for (const idx of sheetIndicesBy(s1, (b) => b.tensorKey === k)) {
        expect(revealTupleAt(s1, idx)[0], `Step 1 downstream ${k}`).toBe(SHEET_REVEAL_UNCOMPUTED);
      }
    }

    // Step 3 (2): Conv1D taps stay visible (NONE); x/B/C ROW_MAJOR; H_t/y/y_norm/Residual UNCOMPUTED
    const s2 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 1,
      activeCell: { tensor: 'x', row: 7, col: 2, headIndex: 1, groupIndex: 0 },
      stepperStep: 2,
      stepperActive: true,
      viewAngle: 'three_quarter',
      simOverride: sim,
    });
    for (const idx of sheetIndicesBy(s2, (b) => b.tensorKey === 'conv_buf')) {
      expect(revealTupleAt(s2, idx)[0]).toBe(SHEET_REVEAL_NONE);
    }
    expect(
      revealTupleAt(s2, sheetIndicesBy(s2, (b) => b.tensorKey === 'x' && b.headIndex === 1)[0]).slice(0, 3),
    ).toEqual([SHEET_REVEAL_ROW_MAJOR, 2, 7]);
    expect(revealTupleAt(s2, sheetIndicesBy(s2, (b) => b.tensorKey === 'b')[0])[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    expect(revealTupleAt(s2, sheetIndicesBy(s2, (b) => b.tensorKey === 'c')[0])[0]).toBe(SHEET_REVEAL_ROW_MAJOR);
    for (const k of ['h_curr', 'y', 'y_norm', 'residual'] as const) {
      expect(revealTupleAt(s2, sheetIndicesBy(s2, (b) => b.tensorKey === k)[0])[0]).toBe(
        SHEET_REVEAL_UNCOMPUTED,
      );
    }

    // Step 5 (4): H_t completed (NONE); y ROW_MAJOR; y_norm & Residual UNCOMPUTED
    const s4 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      activeCell: { tensor: 'y', row: 5, col: 2, headIndex: 0, groupIndex: 0, tokenIndex: 5 },
      stepperStep: 4,
      stepperActive: true,
      viewAngle: 'three_quarter',
      simOverride: sim,
    });
    expect(revealTupleAt(s4, sheetIndicesBy(s4, (b) => b.tensorKey === 'h_curr')[0])[0]).toBe(SHEET_REVEAL_NONE);
    expect(revealTupleAt(s4, sheetIndicesBy(s4, (b) => b.tensorKey === 'y')[0]).slice(0, 3)).toEqual([
      SHEET_REVEAL_ROW_MAJOR,
      2,
      5,
    ]);
    expect(revealTupleAt(s4, sheetIndicesBy(s4, (b) => b.tensorKey === 'y_norm')[0])[0]).toBe(
      SHEET_REVEAL_UNCOMPUTED,
    );
    expect(revealTupleAt(s4, sheetIndicesBy(s4, (b) => b.tensorKey === 'residual')[0])[0]).toBe(
      SHEET_REVEAL_UNCOMPUTED,
    );

    // Step 6 (5): y_norm ROW_MAJOR (residual UNCOMPUTED), then residual ROW_MAJOR (y_norm NONE)
    const s5Norm = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'y_norm', row: 7, col: 5 },
      stepperStep: 5,
      stepperActive: true,
      viewAngle: 'three_quarter',
      simOverride: sim,
    });
    expect(revealTupleAt(s5Norm, sheetIndicesBy(s5Norm, (b) => b.tensorKey === 'y_norm')[0]).slice(0, 3)).toEqual([
      SHEET_REVEAL_ROW_MAJOR,
      5,
      7,
    ]);
    expect(revealTupleAt(s5Norm, sheetIndicesBy(s5Norm, (b) => b.tensorKey === 'residual')[0])[0]).toBe(
      SHEET_REVEAL_UNCOMPUTED,
    );

    const s5Res = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'residual', row: 7, col: 4 },
      stepperStep: 5,
      stepperActive: true,
      viewAngle: 'three_quarter',
      simOverride: sim,
    });
    expect(revealTupleAt(s5Res, sheetIndicesBy(s5Res, (b) => b.tensorKey === 'y_norm')[0])[0]).toBe(
      SHEET_REVEAL_NONE,
    );
    expect(revealTupleAt(s5Res, sheetIndicesBy(s5Res, (b) => b.tensorKey === 'residual')[0]).slice(0, 3)).toEqual([
      SHEET_REVEAL_ROW_MAJOR,
      4,
      7,
    ]);
  });
});

describe('V3 · Mamba-2 activeOpPartIds, Floating cellPopover & Zoom-Adaptive activeCellGrids', () => {
  it('emits exact activeOpPartIds, anchored cellPopover with provenance sourceWorlds, and live H_t / H_{t-1} Float32 grids', () => {
    const sim = runMambaSim(8);
    const activeHead = 3;
    const tokIdx = 6;
    const p = 2;
    const n = 1;

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const scene = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead,
        activeCell: {
          tensor: 'h_curr',
          row: p,
          col: n,
          headIndex: activeHead,
          groupIndex: 1,
          tokenIndex: tokIdx,
        },
        viewAngle,
        simOverride: sim,
      });

      // 1. activeOpPartIds contains exact target + upstream operand parts only
      expect(scene.activeOpPartIds).toBeDefined();
      expect(scene.activeOpPartIds!.has('ssm-state-curr')).toBe(true);
      expect(scene.activeOpPartIds!.has('ssm-state-prev')).toBe(true);
      expect(scene.activeOpPartIds!.has('dt-branch')).toBe(true);
      expect(scene.activeOpPartIds!.has(`mamba-head-${activeHead}`)).toBe(true);
      expect(scene.activeOpPartIds!.has('b-proj')).toBe(true);
      expect(scene.activeOpPartIds!.has('in-proj')).toBe(false);
      expect(scene.activeOpPartIds!.has('out-proj')).toBe(false);
      expect(scene.activeOpPartIds!.has('residual-out')).toBe(false);
      expect(scene.activeOpPartIds!.has('mamba-head-0')).toBe(false);

      // 2. Floating cellPopover anchored at active cell with exact value & 4 operand provenance sources
      const pop = scene.cellPopover;
      expect(pop).toBeDefined();
      expect(pop).not.toBeNull();
      expect(pop!.tensorLabel).toContain('H_t');
      expect(pop!.coordLabel).toContain('H03');
      expect(pop!.valueText).toBe(`= ${scene.activeFormula!.valueText}`);
      expect(pop!.operandColors).toEqual(['indigo', 'teal', 'red']);
      expect(pop!.sourceWorlds).toHaveLength(4);
      for (const coord of pop!.world) {
        expect(Number.isFinite(coord)).toBe(true);
      }
      const hCurrBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'h_curr')!;
      expect(pop!.world[0]).toBeGreaterThanOrEqual(hCurrBox.min[0] - 0.1);
      expect(pop!.world[0]).toBeLessThanOrEqual(hCurrBox.max[0] + 0.1);
      expect(pop!.world[2]).toBeGreaterThanOrEqual(hCurrBox.min[2] - 0.1);
      expect(pop!.world[2]).toBeLessThanOrEqual(hCurrBox.max[2] + 0.1);

      // 3. activeCellGrids contains H_t (4×4) and H_{t-1} (4×4) matching sim.H_history bit-for-bit
      const grids = scene.activeCellGrids!;
      expect(grids.length).toBeGreaterThanOrEqual(2);
      const hCurrGrid = grids[0];
      const hPrevGrid = grids[1];
      expect(hCurrGrid.cols).toBe(4);
      expect(hCurrGrid.rows).toBe(4);
      expect(hCurrGrid.rowLabels).toEqual(['p0', 'p1', 'p2', 'p3']);
      expect(hCurrGrid.colLabels).toEqual(['n0', 'n1', 'n2', 'n3']);
      expect(hCurrGrid.activeRow).toBe(p);
      expect(hCurrGrid.activeCol).toBe(n);
      expect(hCurrGrid.rowDimLabel).toBe(`t${tokIdx} ${sim.tokenLabels[tokIdx]}`);

      const stateStride = sim.nHeads * sim.dHead * sim.dState;
      for (let rp = 0; rp < 4; rp++) {
        for (let cn = 0; cn < 4; cn++) {
          const localIdx = (activeHead * 4 + rp) * 4 + cn;
          expect(hCurrGrid.values[rp * 4 + cn]).toBe(sim.H_history[tokIdx * stateStride + localIdx]);
          expect(hPrevGrid.values[rp * 4 + cn]).toBe(
            sim.H_history[(tokIdx - 1) * stateStride + localIdx],
          );
        }
      }
    }
  });

  it('matches live seeded sim.initialSsmState and absolute token labels in AR Decode and emits active target grids for y / RMSNorm / Δt', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session); // P=4, S=1 -> abs token t4
    const decodeStep = getActiveDecodeStep(session);
    const mSim = decodeStep.getMambaTrace(1);
    const absTok = mSim.startTokenOffset; // 4
    const expectedTokLabel = `t${absTok} ${mSim.tokenLabels[0]}`;

    // 1. Seeded H_t / H_{t-1} grid values match sim.H_history & sim.initialSsmState bit-for-bit
    const hScene = buildMambaCutaway({
      bayId: 1,
      tokens: mSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      activeCell: { tensor: 'h_curr', row: 1, col: 3, headIndex: 0, tokenIndex: 0 },
      viewAngle: 'three_quarter',
      simOverride: mSim,
    });
    const hCurrGrid = hScene.activeCellGrids![0];
    const hPrevGrid = hScene.activeCellGrids![1];
    expect(hCurrGrid.rowDimLabel).toBe(expectedTokLabel);
    expect(hPrevGrid.colDimLabel).toContain('Carried');
    for (let rp = 0; rp < 4; rp++) {
      for (let cn = 0; cn < 4; cn++) {
        const localIdx = rp * 4 + cn;
        expect(hCurrGrid.values[localIdx]).toBe(mSim.H_history[localIdx]);
        expect(hPrevGrid.values[localIdx]).toBe(mSim.initialSsmState![localIdx]);
      }
    }

    // 2. Active target sheets (y, RMSNorm, Δt) appear first in activeCellGrids with live values & absolute token labels
    const yScene = buildMambaCutaway({
      bayId: 1,
      tokens: mSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'y', row: 0, col: 5 },
      viewAngle: 'three_quarter',
      simOverride: mSim,
    });
    const yGrid = yScene.activeCellGrids![0];
    expect(yGrid.cols).toBe(mSim.dInner);
    expect(yGrid.rows).toBe(1);
    expect(yGrid.activeCol).toBe(5);
    expect(yGrid.activeRow).toBe(0);
    expect(yGrid.rowLabels).toEqual([expectedTokLabel]);
    expect(yGrid.values[5]).toBe(mSim.y[5 * mSim.tokens + 0]);

    const rmsScene = buildMambaCutaway({
      bayId: 1,
      tokens: mSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'rmsnorm', row: 0, col: 3 },
      viewAngle: 'three_quarter',
      simOverride: mSim,
    });
    const rmsGrid = rmsScene.activeCellGrids![0];
    expect(rmsGrid.cols).toBe(mSim.d);
    expect(rmsGrid.rowLabels).toEqual([expectedTokLabel]);
    expect(rmsGrid.values[3]).toBe(mSim.U[3 * mSim.tokens + 0]);

    const dtScene = buildMambaCutaway({
      bayId: 1,
      tokens: mSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'dt', row: 0, col: 4, headIndex: 4 },
      viewAngle: 'three_quarter',
      simOverride: mSim,
    });
    const dtGrid = dtScene.activeCellGrids![0];
    expect(dtGrid.cols).toBe(mSim.nHeads);
    expect(dtGrid.rowLabels).toEqual([expectedTokLabel]);
    expect(dtGrid.values[4]).toBe(mSim.dt[4 * mSim.tokens + 0]);

    // 3. Idle scene emits null cellPopover and undefined activeOpPartIds while still providing H_t & H_{t-1} grids
    const idle = buildMambaCutaway({
      bayId: 1,
      tokens: mSim.tokens,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
      simOverride: mSim,
    });
    expect(idle.cellPopover).toBeNull();
    expect(idle.activeOpPartIds).toBeUndefined();
    expect(idle.activeCellGrids).toHaveLength(2);
  });
});
