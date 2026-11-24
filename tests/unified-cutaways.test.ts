import { describe, expect, it } from 'vitest';

import {
  createDefaultHybridModel,
  defaultTokenIdsForLength,
  runHybridExecution,
} from '../src/math/hybridFlow';
import { advanceMambaTemporalPlaybackTick } from '../src/model/mambaPlayback';
import { advanceMoeTokenCycleTick, stepMoeFocusToken } from '../src/model/moePlayback';
import { createInitialState, DEFAULT_FOV_DEG } from '../src/model/state';
import {
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
} from '../src/scene/cutaways';
import { computeEditorialCacheKey } from '../src/ui/controls';

function expectBitExact(a: Float32Array, b: Float32Array, label: string): void {
  expect(a.length, `${label} len`).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(a[i], `${label}[${i}]`).toBe(b[i]);
  }
}

describe('Gate B: Unified Cutaway Integration, Cross-Bay Continuity & Custom Token Labels', () => {
  it('1. preserves bit-exact cross-bay tensor continuity across all live cutaways in both Spatial 3/4 and Near-Plan views', () => {
    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      const mamba01 = buildMambaCutaway({
        bayId: 1,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: exec.getMambaTrace(1),
      });
      const moe02 = buildMoECutaway({
        bayId: 2,
        tokens: 8,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: exec.getMoeTrace(2),
      });
      const mamba42 = buildMambaCutaway({
        bayId: 42,
        tokens: 8,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: exec.getMambaTrace(42),
      });
      const gqa43 = buildGqaCutaway({
        bayId: 43,
        tokens: 8,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: exec.getGqaTrace(43),
      });
      const moe52 = buildMoECutaway({
        bayId: 52,
        tokens: 8,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: exec.getMoeTrace(52),
      });

      expectBitExact(mamba01.mambaSim!.X, exec.X_embed, `${viewAngle} Bay01 X === X_embed`);
      expectBitExact(mamba01.mambaSim!.Y, moe02.moeSim!.X, `${viewAngle} Bay01 Y === Bay02 X`);
      expectBitExact(mamba42.mambaSim!.Y, gqa43.gqaSim!.X, `${viewAngle} Bay42 Y === Bay43 X`);
      expectBitExact(moe52.moeSim!.Y, exec.residualStreams[52], `${viewAngle} Bay52 Y === stream[52]`);

      // Distinct bays inside unified execution carry genuinely distinct activations
      expect(mamba01.mambaSim!.Y[0]).not.toBe(mamba42.mambaSim!.Y[0]);
      expect(moe02.moeSim!.Y[0]).not.toBe(moe52.moeSim!.Y[0]);
    }
  });

  it('2. propagates custom token labels and GQA effective rmsEps=1e-5 through cutaway geometry, part catalog, and formulas', () => {
    const model = createDefaultHybridModel();
    const customIds = [2, 16, 2, 23];
    const customLabels = ['nem', 'eff', 'nem2', 'exp'];
    const exec = runHybridExecution(model, { tokenIds: customIds, tokenLabels: customLabels });

    const gqaScene = buildGqaCutaway({
      bayId: 6,
      tokens: 4,
      precision: 'bf16',
      selectedPart: 'rmsnorm',
      hoveredPart: null,
      activeCell: { tensor: 'rmsnorm', row: 3, col: 0 },
      viewAngle: 'three_quarter',
      simOverride: exec.getGqaTrace(6),
    });
    expect(gqaScene.gqaSim!.rmsEps).toBe(1e-5);
    expect(gqaScene.gqaSim!.tokenLabels).toEqual(customLabels);
    expect(gqaScene.partCatalog['rmsnorm'].title).toContain('1e−5');
    expect(gqaScene.activeFormula!.coordLabel).toContain('"exp"(3)');
    expect(gqaScene.activeFormula!.subEquation).toContain('1e−5');

    const moeScene = buildMoECutaway({
      bayId: 2,
      tokens: 4,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 1,
      activeCell: { tensor: 'input', row: 1, col: 2, tokenIndex: 1 },
      viewAngle: 'three_quarter',
      simOverride: exec.getMoeTrace(2),
    });
    expect(moeScene.activeFormula!.coordLabel).toContain('"eff"');
    expect(moeScene.landmarks.some((l) => l.id === 'moe-bank' && l.label.includes('"eff"'))).toBe(true);

    const mambaScene = buildMambaCutaway({
      bayId: 1,
      tokens: 4,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      stateTokenIndex: 2,
      activeCell: { tensor: 'input', row: 2, col: 1, tokenIndex: 2 },
      viewAngle: 'three_quarter',
      simOverride: exec.getMambaTrace(1),
    });
    expect(mambaScene.mambaSim!.tokenLabels).toEqual(customLabels);
    expect(mambaScene.activeFormula!.coordLabel).toContain('"nem2"');

    // Standalone buildGqaCutaway without simOverride retains 1e-6 metadata cleanly
    const standaloneGqa = buildGqaCutaway({
      bayId: 6,
      tokens: 4,
      precision: 'bf16',
      selectedPart: 'rmsnorm',
      hoveredPart: null,
      activeCell: { tensor: 'rmsnorm', row: 0, col: 0 },
      viewAngle: 'three_quarter',
    });
    expect(standaloneGqa.gqaSim!.rmsEps).toBe(1e-6);
    expect(standaloneGqa.partCatalog['rmsnorm'].title).toContain('1e−6');
    expect(standaloneGqa.activeFormula!.subEquation).toContain('1e−6');
  });

  it('3. keeps MoE cycling, Mamba temporal playback, and sidebar cache keys synchronized on unified bay traces', () => {
    const model = createDefaultHybridModel();
    const execA = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });
    const execB = runHybridExecution(model, { tokenIds: [1, 2, 3, 4, 5, 6, 7, 8] });

    const moe52 = execA.getMoeTrace(52);
    const tick = advanceMoeTokenCycleTick(1, moe52, {
      tensor: 'topk_weights',
      row: 0,
      col: 0,
      tokenIndex: 1,
      expertId: moe52.topk_indices[1 * 6],
      rankSlot: 0,
    }, null);
    expect(tick.nextToken).toBe(2);
    expect(tick.nextSelectedCell?.expertId).toBe(moe52.topk_indices[2 * 6]);

    const stepped = stepMoeFocusToken(2, 1, moe52, tick.nextSelectedCell);
    expect(stepped.nextToken).toBe(3);
    expect(stepped.nextSelectedCell?.expertId).toBe(moe52.topk_indices[3 * 6]);

    const mambaTick = advanceMambaTemporalPlaybackTick(2, 8, {
      tensor: 'h_curr',
      row: 0,
      col: 0,
      headIndex: 0,
      tokenIndex: 2,
    }, null);
    expect(mambaTick.nextToken).toBe(3);
    expect(mambaTick.nextSelectedCell?.tokenIndex).toBe(3);

    const dummyPose = { target: [0, 0, 0] as const, azimuthDeg: 20, elevationDeg: 23.5, distance: 12, fovDeg: DEFAULT_FOV_DEG };
    const state = createInitialState(dummyPose, false);
    state.mode = 'tower';
    state.section = 'embedding';

    const keyA = computeEditorialCacheKey(state, null, execA, 7);
    const keyB = computeEditorialCacheKey(state, null, execB, 7);
    const keyTok3 = computeEditorialCacheKey(state, null, execA, 3);
    expect(keyA).not.toBe(keyB);
    expect(keyA).not.toBe(keyTok3);
  });
});
