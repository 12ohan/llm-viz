import { describe, expect, it } from 'vitest';

import { createDefaultHybridModel } from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeStep,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import {
  mambaStepperCellForState,
  readMambaPriorSsmValue,
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
import { fmtSigned } from '../src/scene/sheetInspector';
import { isCellHiddenByReveal } from '../src/ui/overlay';

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

describe('RC1 · Mamba-2 Focused-Cell Accumulation & Non-Spoiling Active Cell Grids', () => {
  it('wires Step 3 (Causal Conv1D 4-tap) partial sums without spoiling x/B/C/conv_buf and propagates reveal frontiers', () => {
    const sim = runMambaSim(8);

    // 1. Post-SiLU x target cell at tok=5, head=1, p=1
    const h = 1;
    const p = 1;
    const tok = 5;
    const ch = h * sim.dHead + p;
    const exactX = sim.x[ch * sim.tokens + tok];
    const exactXText = fmtSigned(exactX, 3);
    const xCell: MambaCellRef = { tensor: 'x', row: tok, col: p, headIndex: h, groupIndex: 0, tokenIndex: tok };

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const scene = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: h,
        activeCell: xCell,
        stepperStep: 2,
        stepperActive: true,
        stepperSubProgress: 0.6,
        termProgress: 0.25,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });

      const accum = scene.focusedAccumulation;
      expect(accum).not.toBeNull();
      expect(accum!.active).toBe(true);
      expect(accum!.totalTerms).toBe(4);
      expect(accum!.currentTermIndex).toBe(1);
      expect(accum!.partialEquation).not.toContain(exactXText);
      expect(accum!.popoverEquation).not.toContain(exactXText);
      expect(accum!.valueBadgeText).not.toContain(exactXText);

      // Popover carries running partial sum + .is-accumulating metadata with zero premature spoiling
      const pop = scene.cellPopover!;
      expect(pop).not.toBeNull();
      expect(pop.isAccumulating).toBe(true);
      expect(pop.badge).toBe(accum!.badgeText);
      expect(pop.equation).toBe(accum!.popoverEquation);
      expect(pop.valueText).toBe(accum!.valueBadgeText);
      expect(pop.equation).not.toContain(exactXText);
      expect(pop.valueText).not.toContain(exactXText);

      // activeCellGrids: [x (target), H_t (UNCOMPUTED), H_{t-1} (NONE)]
      const grids = scene.activeCellGrids!;
      expect(grids).toHaveLength(3);
      const [xGrid, hCurrGrid, hPrevGrid] = grids;

      expect(xGrid.activeCol).toBe(p);
      expect(xGrid.activeRow).toBe(tok);
      expect(xGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(xGrid.revealColFrontier).toBe(p);
      expect(xGrid.revealRowFrontier).toBe(tok);
      expect(xGrid.activeValueOverride).toBe(accum!.partialSum);
      expect(isCellHiddenByReveal(p, tok, xGrid)).toBe(false);
      expect(isCellHiddenByReveal(p + 1, tok, xGrid)).toBe(true);
      expect(isCellHiddenByReveal(0, tok + 1, xGrid)).toBe(true);

      // Downstream H_t grid MUST be fully masked (UNCOMPUTED), while H_{t-1} stays NONE
      expect(hCurrGrid.revealMode).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(hCurrGrid.activeValueOverride).toBeUndefined();
      expect(isCellHiddenByReveal(0, 0, hCurrGrid)).toBe(true);

      expect(hPrevGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(hPrevGrid.activeValueOverride).toBeUndefined();
      expect(isCellHiddenByReveal(0, 0, hPrevGrid)).toBe(false);

      // Final tap (termProgress = 0.9) converges to exact post-SiLU x
      const finalScene = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: h,
        activeCell: xCell,
        stepperStep: 2,
        stepperActive: true,
        termProgress: 0.9,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });
      expect(Math.abs(finalScene.focusedAccumulation!.partialSum - exactX)).toBeLessThan(1e-5);
      expect(finalScene.activeCellGrids![0].activeValueOverride).toBe(
        finalScene.focusedAccumulation!.partialSum,
      );
    }

    // 2. Also verify Conv1D rolling buffer (conv_buf), B, and C targets wire partial sums & reveal cleanly
    const extraTargets: readonly { readonly cell: MambaCellRef; readonly expectedReveal: number; readonly exactVal: number }[] = [
      {
        cell: { tensor: 'conv_buf', row: 2, col: ch, tapIndex: 2, tokenIndex: tok },
        expectedReveal: SHEET_REVEAL_NONE,
        exactVal: sim.xBC[ch * sim.tokens + tok],
      },
      {
        cell: { tensor: 'b', row: tok, col: 5, groupIndex: 1, tokenIndex: tok },
        expectedReveal: SHEET_REVEAL_ROW_MAJOR,
        exactVal: sim.B[5 * sim.tokens + tok],
      },
      {
        cell: { tensor: 'c', row: tok, col: 9, groupIndex: 2, tokenIndex: tok },
        expectedReveal: SHEET_REVEAL_ROW_MAJOR,
        exactVal: sim.C[9 * sim.tokens + tok],
      },
    ];
    for (const { cell, expectedReveal, exactVal } of extraTargets) {
      const s = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeCell: cell,
        stepperStep: 2,
        stepperActive: true,
        termProgress: 0.25,
        viewAngle: 'three_quarter',
        simOverride: sim,
      });
      expect(s.focusedAccumulation?.active).toBe(true);
      expect(s.cellPopover?.isAccumulating).toBe(true);
      expect(s.cellPopover?.valueText).not.toContain(fmtSigned(exactVal, 3));
      expect(s.activeCellGrids![0].revealMode).toBe(expectedReveal);
      expect(s.activeCellGrids![0].activeValueOverride).toBe(s.focusedAccumulation!.partialSum);
      expect(s.activeCellGrids![1].revealMode).toBe(SHEET_REVEAL_UNCOMPUTED);
      expect(s.activeCellGrids![2].revealMode).toBe(SHEET_REVEAL_NONE);
    }
  });

  it('wires Step 4 (SSM Update 2-term) in Full Seq and seeded AR Decode without spoiling H_t and keeps carried H_{t-1} unmasked', () => {
    // 1. Full Seq (T=8) Step 4
    const sim = runMambaSim(8);
    const stateStride = sim.nHeads * sim.dHead * sim.dState;
    const tokIdx = 6;
    // Find a cell with distinct decayed prior vs final H_t so non-spoiling assertion is airtight
    let bestH = 0;
    let bestP = 1;
    let bestN = 1;
    let bestDelta = -1;
    for (let hh = 0; hh < sim.nHeads; hh++) {
      for (let pp = 0; pp < sim.dHead; pp++) {
        for (let nn = 0; nn < sim.dState; nn++) {
          const idx = (hh * sim.dHead + pp) * sim.dState + nn;
          const prev = readMambaPriorSsmValue(sim, tokIdx, idx).value;
          const decayed = Math.fround(sim.dA[hh * sim.tokens + tokIdx] * prev);
          const cur = sim.H_history[tokIdx * stateStride + idx];
          const delta = Math.abs(cur - decayed);
          if (delta > bestDelta) {
            bestDelta = delta;
            bestH = hh;
            bestP = pp;
            bestN = nn;
          }
        }
      }
    }
    expect(bestDelta).toBeGreaterThan(0.05);

    const localIdx = (bestH * sim.dHead + bestP) * sim.dState + bestN;
    const exactH = sim.H_history[tokIdx * stateStride + localIdx];
    const exactHText = fmtSigned(exactH, 3);
    const expectedDecay = Math.fround(
      sim.dA[bestH * sim.tokens + tokIdx] * readMambaPriorSsmValue(sim, tokIdx, localIdx).value,
    );

    const fullCell: MambaCellRef = {
      tensor: 'h_curr',
      row: bestP,
      col: bestN,
      headIndex: bestH,
      tokenIndex: tokIdx,
    };

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const s0 = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: bestH,
        activeCell: fullCell,
        stepperStep: 3,
        stepperActive: true,
        termProgress: 0.25,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });

      const accum = s0.focusedAccumulation!;
      expect(accum).not.toBeNull();
      expect(accum.active).toBe(true);
      expect(accum.totalTerms).toBe(2);
      expect(accum.currentTermIndex).toBe(0);
      expect(Math.abs(accum.partialSum - expectedDecay)).toBeLessThan(1e-5);

      const pop = s0.cellPopover!;
      expect(pop.isAccumulating).toBe(true);
      expect(pop.badge).toBe('Term 1/2 · Running');
      expect(pop.equation).toBe(accum.popoverEquation);
      expect(pop.valueText).toBe(accum.valueBadgeText);
      expect(pop.equation).not.toContain(exactHText);
      expect(pop.valueText).not.toContain(exactHText);

      // activeCellGrids: [H_t, H_{t-1}, x]
      const [hCurrGrid, hPrevGrid, xGrid] = s0.activeCellGrids!;
      expect(hCurrGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(hCurrGrid.revealColFrontier).toBe(bestN);
      expect(hCurrGrid.revealRowFrontier).toBe(bestP);
      expect(hCurrGrid.activeValueOverride).toBe(accum.partialSum);
      expect(isCellHiddenByReveal(bestN, bestP, hCurrGrid)).toBe(false);
      // Any cell strictly past (bestN, bestP) in row-major order is hidden
      const futureCol = bestN + 1 < sim.dState ? bestN + 1 : 0;
      const futureRow = bestN + 1 < sim.dState ? bestP : bestP + 1;
      if (futureRow < sim.dHead) {
        expect(isCellHiddenByReveal(futureCol, futureRow, hCurrGrid)).toBe(true);
      }

      // H_{t-1} is NEVER masked and never overridden
      expect(hPrevGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(hPrevGrid.activeValueOverride).toBeUndefined();
      for (let rp = 0; rp < 4; rp++) {
        for (let cn = 0; cn < 4; cn++) {
          expect(isCellHiddenByReveal(cn, rp, hPrevGrid)).toBe(false);
        }
      }
      // Upstream x grid is already computed (NONE) with no activeValueOverride
      expect(xGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(xGrid.activeValueOverride).toBeUndefined();

      // Term 2/2 (termProgress = 0.75) completes H_t = dA·H_{t-1} + Δt·x·B
      const s1 = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: bestH,
        activeCell: fullCell,
        stepperStep: 3,
        stepperActive: true,
        termProgress: 0.75,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });
      expect(s1.focusedAccumulation!.currentTermIndex).toBe(1);
      expect(Math.abs(s1.focusedAccumulation!.partialSum - exactH)).toBeLessThan(1e-5);
      expect(s1.activeCellGrids![0].activeValueOverride).toBe(s1.focusedAccumulation!.partialSum);
    }

    // 2. Seeded AR Decode (P=4, S=1 -> abs t4): carried FP32 prior state H_{t-1}
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session);
    const decodeStep = getActiveDecodeStep(session);
    const arSim = decodeStep.getMambaTrace(1);
    expect(arSim.hasInitialState).toBe(true);

    let arH = 0;
    let arP = 0;
    let arN = 0;
    let arBestDelta = -1;
    for (let hh = 0; hh < arSim.nHeads; hh++) {
      for (let pp = 0; pp < arSim.dHead; pp++) {
        for (let nn = 0; nn < arSim.dState; nn++) {
          const idx = (hh * arSim.dHead + pp) * arSim.dState + nn;
          const prev = readMambaPriorSsmValue(arSim, 0, idx).value;
          const decayed = Math.fround(arSim.dA[hh] * prev);
          const cur = arSim.H[idx];
          const delta = Math.min(Math.abs(cur - prev), Math.abs(cur - decayed));
          if (delta > arBestDelta) {
            arBestDelta = delta;
            arH = hh;
            arP = pp;
            arN = nn;
          }
        }
      }
    }
    expect(arBestDelta).toBeGreaterThan(0.05);

    const arIdx = (arH * arSim.dHead + arP) * arSim.dState + arN;
    const arExactH = arSim.H[arIdx];
    const arExactText = fmtSigned(arExactH, 3);
    const arPriorVal = arSim.initialSsmState![arIdx];
    const arExpectedDecay = Math.fround(arSim.dA[arH] * arPriorVal);
    const arCell: MambaCellRef = { tensor: 'h_curr', row: arP, col: arN, headIndex: arH, tokenIndex: 0 };

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const arScene = buildMambaCutaway({
        bayId: 1,
        tokens: arSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        activeHead: arH,
        activeCell: arCell,
        stepperStep: 3,
        stepperActive: true,
        termProgress: 0.25,
        reducedMotion: false,
        viewAngle,
        simOverride: arSim,
      });

      const accum = arScene.focusedAccumulation!;
      expect(accum).not.toBeNull();
      expect(accum.currentTermIndex).toBe(0);
      expect(accum.partialSubEquation).toContain('carried FP32 state');
      expect(Math.abs(accum.partialSum - arExpectedDecay)).toBeLessThan(1e-5);
      expect(arScene.cellPopover!.isAccumulating).toBe(true);
      expect(arScene.cellPopover!.valueText).not.toContain(arExactText);
      expect(arScene.cellPopover!.equation).not.toContain(arExactText);

      const [arHCurrGrid, arHPrevGrid] = arScene.activeCellGrids!;
      expect(arHCurrGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(arHCurrGrid.revealColFrontier).toBe(arN);
      expect(arHCurrGrid.revealRowFrontier).toBe(arP);
      expect(arHCurrGrid.activeValueOverride).toBe(accum.partialSum);

      // Carried H_{t-1} grid stays strictly unmasked (SHEET_REVEAL_NONE) with exact seeded values
      expect(arHPrevGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(arHPrevGrid.activeValueOverride).toBeUndefined();
      expect(arHPrevGrid.values[arP * arSim.dState + arN]).toBe(arPriorVal);
    }
  });

  it('wires Step 5 (Readout + D) partial sums at termProgress=0.25 without spoiling y and masks future y cells', () => {
    const sim = runMambaSim(8);
    const tok = 4;
    const h = 3;
    const p = 2;
    const col = h * sim.dHead + p;
    const exactY = sim.y[col * sim.tokens + tok];
    const exactYText = fmtSigned(exactY, 3);
    const yCell: MambaCellRef = { tensor: 'y', row: tok, col, headIndex: h, tokenIndex: tok };

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const scene = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: h,
        activeCell: yCell,
        stepperStep: 4,
        stepperActive: true,
        termProgress: 0.25,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });

      const accum = scene.focusedAccumulation!;
      expect(accum).not.toBeNull();
      expect(accum.active).toBe(true);
      expect(accum.totalTerms).toBe(sim.dState + 1); // 4 state terms + D·x skip = 5
      expect(accum.currentTermIndex).toBe(1); // floor(0.25 * 5) = 1 (2/5)
      expect(accum.partialEquation).not.toContain(exactYText);

      const pop = scene.cellPopover!;
      expect(pop.isAccumulating).toBe(true);
      expect(pop.badge).toBe('Term 2/5 · Running');
      expect(pop.equation).toBe(accum.popoverEquation);
      expect(pop.valueText).toBe(accum.valueBadgeText);
      expect(pop.equation).not.toContain(exactYText);
      expect(pop.valueText).not.toContain(exactYText);

      // activeCellGrids: [y (ROW_MAJOR + override), H_t (NONE), H_{t-1} (NONE)]
      const [yGrid, hCurrGrid, hPrevGrid] = scene.activeCellGrids!;
      expect(yGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
      expect(yGrid.revealColFrontier).toBe(col);
      expect(yGrid.revealRowFrontier).toBe(tok);
      expect(yGrid.activeValueOverride).toBe(accum.partialSum);
      expect(isCellHiddenByReveal(col, tok, yGrid)).toBe(false);
      expect(isCellHiddenByReveal(col + 1, tok, yGrid)).toBe(true);
      expect(isCellHiddenByReveal(0, tok + 1, yGrid)).toBe(true);

      expect(hCurrGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(hCurrGrid.activeValueOverride).toBeUndefined();
      expect(hPrevGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      expect(hPrevGrid.activeValueOverride).toBeUndefined();

      // Final skip term (termProgress = 0.9) converges to exact y
      const finalScene = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: h,
        activeCell: yCell,
        stepperStep: 4,
        stepperActive: true,
        termProgress: 0.9,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });
      expect(finalScene.focusedAccumulation!.currentTermIndex).toBe(4);
      expect(Math.abs(finalScene.focusedAccumulation!.partialSum - exactY)).toBeLessThan(1e-5);
    }
  });

  it('preserves ordinary unrestricted inspection when stepperActive is false, reducedMotion is true, or termProgress >= 0.999', () => {
    const sim = runMambaSim(8);
    const activeCell: MambaCellRef = {
      tensor: 'h_curr',
      row: 2,
      col: 1,
      headIndex: 2,
      groupIndex: 1,
      tokenIndex: 6,
    };

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // 1. stepperActive: false -> fully unrestricted (no reveal masking, no focusedAccumulation, no override)
      const inactive = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: 2,
        activeCell,
        stepperStep: 3,
        stepperActive: false,
        termProgress: 0.25,
        reducedMotion: false,
        viewAngle,
        simOverride: sim,
      });
      expect(inactive.focusedAccumulation).toBeNull();
      expect(inactive.sheetRevealData).toBeUndefined();
      expect(inactive.cellPopover!.isAccumulating).toBeUndefined();
      expect(inactive.cellPopover!.valueText).toBe(`= ${inactive.activeFormula!.valueText}`);
      for (const grid of inactive.activeCellGrids!) {
        expect(grid.revealMode).toBeUndefined();
        expect(grid.revealColFrontier).toBeUndefined();
        expect(grid.revealRowFrontier).toBeUndefined();
        expect(grid.activeValueOverride).toBeUndefined();
      }

      // 2. reducedMotion: true -> fully unrestricted even while stepperActive is true
      const reduced = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        activeHead: 2,
        activeCell,
        stepperStep: 3,
        stepperActive: true,
        termProgress: 0.25,
        reducedMotion: true,
        viewAngle,
        simOverride: sim,
      });
      expect(reduced.focusedAccumulation).toBeNull();
      expect(reduced.sheetRevealData).toBeUndefined();
      expect(reduced.cellPopover!.isAccumulating).toBeUndefined();
      expect(reduced.cellPopover!.valueText).toBe(`= ${reduced.activeFormula!.valueText}`);
      for (const grid of reduced.activeCellGrids!) {
        expect(grid.revealMode).toBeUndefined();
        expect(grid.activeValueOverride).toBeUndefined();
      }

      // 3. termProgress >= 0.999 or omitted -> focusedAccumulation null & popover unrestricted, while stepper reveal still masks future cells
      for (const tp of [0.999, 1, undefined] as const) {
        const settled = buildMambaCutaway({
          bayId: 1,
          tokens: 8,
          precision: 'bf16',
          selectedPart: null,
          hoveredPart: null,
          activeHead: 2,
          activeCell,
          stepperStep: 3,
          stepperActive: true,
          termProgress: tp,
          reducedMotion: false,
          viewAngle,
          simOverride: sim,
        });
        expect(settled.focusedAccumulation).toBeNull();
        expect(settled.cellPopover!.isAccumulating).toBeUndefined();
        expect(settled.cellPopover!.valueText).toBe(`= ${settled.activeFormula!.valueText}`);
        const [hCurrGrid, hPrevGrid] = settled.activeCellGrids!;
        expect(hCurrGrid.activeValueOverride).toBeUndefined();
        expect(hCurrGrid.revealMode).toBe(SHEET_REVEAL_ROW_MAJOR);
        expect(hCurrGrid.revealColFrontier).toBe(1);
        expect(hCurrGrid.revealRowFrontier).toBe(2);
        expect(hPrevGrid.revealMode).toBe(SHEET_REVEAL_NONE);
      }
    }
  });
});
