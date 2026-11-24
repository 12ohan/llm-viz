import { describe, expect, it } from 'vitest';

import { createDefaultHybridModel } from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeStep,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import { createInitialState, createPerspectivePose } from '../src/model/state';
import {
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
} from '../src/scene/cutaways';
import { computeEditorialCacheKey } from '../src/ui/controls';
import { SHEET_INSTANCE_FLOATS } from '../src/render/sheets';

describe('Gate B · Truthful Incremental AR Decode Cutaways & Provenance', () => {
  it('1. renders true rectangular [1 × (P+1)] GQA attention in both 3/4 Spatial and Near-Plan views with reused vs new KV provenance', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session); // P = 4, S = 1, K = 5
    const step = getActiveDecodeStep(session);
    const gqaSim = step.getGqaTrace(43);

    expect(gqaSim.tokens).toBe(1);
    expect(gqaSim.prefixTokens).toBe(4);
    expect(gqaSim.keyTokens).toBe(5);
    expect(step.getLayerTraceOrigin(43)).toBe('computed_this_step');

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // Inspect reused prefix KV cell k=1 (< P=4)
      const scReused = buildGqaCutaway({
        bayId: 43,
        tokens: gqaSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: gqaSim,
        activeCell: { tensor: 'kv_k', kvHeadIndex: 0, tokenIndex: 1, row: 0, col: 0 },
      });

      expect(scReused.gqaStats?.kvTokenSliceCount).toBe(10); // 2 KV heads * 5 keys (4 cached + 1 new)
      expect(scReused.activeFormula?.tensorLabel).toContain('Reused');
      expect(scReused.activeFormula?.headBadge).toContain('Reused Prefix Cache');
      expect(scReused.activeFormula?.equation).toContain('not recomputed');

      // Inspect newly appended KV cell k=4 (>= P=4)
      const scNew = buildGqaCutaway({
        bayId: 43,
        tokens: gqaSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: gqaSim,
        activeCell: { tensor: 'kv_k', kvHeadIndex: 0, tokenIndex: 4, row: 0, col: 0 },
      });
      expect(scNew.activeFormula?.tensorLabel).toContain('New');
      expect(scNew.activeFormula?.headBadge).toContain('Newly Computed');

      // Inspect rectangular Scores [1 × 5] sheet instance dimensions & formula
      const scScore = buildGqaCutaway({
        bayId: 43,
        tokens: gqaSim.tokens,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
        simOverride: gqaSim,
        activeCell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 0, col: 2 },
      });
      expect(scScore.activeFormula?.isMasked).toBe(false);
      expect(scScore.activeFormula?.headBadge).toContain('[1×5]');
      expect(scScore.activeFormula?.subEquation).toContain('Reused prefix K0');

      // Verify Scores PickBox carries true rectangular gridCols=5, gridRows=1
      const scorePick = scScore.solidPickBoxes.find(
        (p) => p.tensorKey === 'scores' && p.kvHeadIndex === 0,
      );
      expect(scorePick).toBeDefined();
      expect(scorePick!.gridCols).toBe(5);
      expect(scorePick!.gridRows).toBe(1);

      // Verify sheetData for Scores has gridCols=5, gridRows=1 and rectangular aspect ex > ez
      let foundRectScoreSheet = false;
      const sd = scScore.sheetData!;
      for (let i = 0; i < scScore.sheetCount!; i++) {
        const off = i * SHEET_INSTANCE_FLOATS;
        const ex = sd[off + 3];
        const ez = sd[off + 5];
        const cols = sd[off + 6];
        const rows = sd[off + 7];
        const kind = sd[off + 8];
        if (kind === 1 && cols === 5 && rows === 1) {
          expect(ex).toBeGreaterThan(ez);
          foundRectScoreSheet = true;
        }
      }
      expect(foundRectScoreSheet).toBe(true);
    }
  });

  it('2. renders truthful carried H_prev & Conv1D history in Mamba-2 and S=1 stateless routing in MoE during AR decode', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session); // P = 4, S = 1 -> abs position t4
    const step = getActiveDecodeStep(session);

    // Mamba-2 Bay 01 seeded incremental cutaway
    const mSim = step.getMambaTrace(1);
    expect(mSim.tokens).toBe(1);
    expect(mSim.startTokenOffset).toBe(4);
    expect(mSim.hasInitialState).toBe(true);

    const mScene = buildMambaCutaway({
      bayId: 1,
      tokens: mSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      viewAngle: 'three_quarter',
      simOverride: mSim,
      activeCell: { tensor: 'h_prev', row: 0, col: 0, headIndex: 0, tokenIndex: 0 },
    });
    expect(mScene.activeFormula?.tensorLabel).toContain('Carried State');
    expect(mScene.activeFormula?.equation).toContain('carried FP32 state');
    expect(mScene.activeFormula?.value).toBe(step.prevState.mambaStates.get(1)!.ssmState[0]);
    const hPrevLandmark = mScene.landmarks.find((l) => l.id === 'm-hprev');
    expect(hPrevLandmark?.label).toContain('Carried');

    // MoE Bay 02 S=1 stateless incremental cutaway
    const eSim = step.getMoeTrace(2);
    expect(eSim.tokens).toBe(1);
    expect(eSim.startTokenOffset).toBe(4);
    const eScene = buildMoECutaway({
      bayId: 2,
      tokens: eSim.tokens,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 0,
      viewAngle: 'three_quarter',
      simOverride: eSim,
      activeCell: { tensor: 'residual', row: 0, col: 0, tokenIndex: 0 },
    });
    expect(eScene.activeFormula?.coordLabel).toContain('t4');
    expect(eScene.moeStats?.activeExpertIds).toHaveLength(6);
    const moeBankLandmark = eScene.landmarks.find((l) => l.id === 'moe-bank');
    expect(moeBankLandmark?.label).toContain('t4');
    expect(eScene.partCatalog['router']?.specs[0]).toContain('t4');

    // Editorial cache key reacts immediately to AR decode stepping and candidate selection
    const state = createInitialState(createPerspectivePose([0, 0, 0], 10));
    state.executionMode = 'ar_decode';
    const k1 = computeEditorialCacheKey(state, null, null, null, session, step.predictedNextTokens[0].tokenId);
    const k2 = computeEditorialCacheKey(state, null, null, null, session, step.predictedNextTokens[1].tokenId);
    const session2 = stepDecodeSessionGreedy(session);
    const k3 = computeEditorialCacheKey(state, null, null, null, session2, getActiveDecodeStep(session2).greedyNextToken.tokenId);
    expect(k1).not.toBe(k2);
    expect(k1).not.toBe(k3);
  });
});
