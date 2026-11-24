import { describe, expect, it } from 'vitest';
import type { AABB } from '../src/render/camera';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_GHOST_RAIL,
} from '../src/render/sheets';
import {
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
  type CutawaySceneGeometry,
} from '../src/scene/cutaways';
import {
  createSheetFocusSet,
  matchesFocusEntry,
  resolveSheetFocusRole,
  resolveSheetHighlight,
} from '../src/scene/sheetInspector';

function highlightAt(scene: CutawaySceneGeometry, idx: number): number {
  return scene.sheetData![idx * SHEET_INSTANCE_FLOATS + 11];
}

function sheetKindAt(scene: CutawaySceneGeometry, idx: number): number {
  return scene.sheetData![idx * SHEET_INSTANCE_FLOATS + 8];
}

function aabbVolume(b: AABB): number {
  return (b.max[0] - b.min[0]) * (b.max[1] - b.min[1]) * (b.max[2] - b.min[2]);
}

function aabbAreaXZ(b: AABB): number {
  return (b.max[0] - b.min[0]) * (b.max[2] - b.min[2]);
}

function assertValidOpFocusBounds(scene: CutawaySceneGeometry, label: string): void {
  expect(scene.opFocusBounds, `${label} missing opFocusBounds`).toBeDefined();
  const full = scene.fitBounds;
  const fullAreaXZ = aabbAreaXZ(full);
  const fullVol = aabbVolume(full);

  for (let step = 0; step <= 5; step++) {
    const b = scene.opFocusBounds![step];
    expect(b, `${label} missing opFocusBounds[${step}]`).toBeDefined();
    for (let axis = 0; axis < 3; axis++) {
      expect(Number.isFinite(b.min[axis]), `${label}[${step}].min[${axis}] finite`).toBe(true);
      expect(Number.isFinite(b.max[axis]), `${label}[${step}].max[${axis}] finite`).toBe(true);
      expect(b.max[axis], `${label}[${step}] non-degenerate axis ${axis}`).toBeGreaterThan(b.min[axis]);
      expect(b.min[axis], `${label}[${step}].min[${axis}] inside fitBounds`).toBeGreaterThanOrEqual(full.min[axis] - 1e-5);
      expect(b.max[axis], `${label}[${step}].max[${axis}] inside fitBounds`).toBeLessThanOrEqual(full.max[axis] + 1e-5);
    }
    expect(aabbAreaXZ(b), `${label}[${step}] XZ area tighter than fitBounds`).toBeLessThan(fullAreaXZ);
    expect(aabbVolume(b), `${label}[${step}] volume tighter than fitBounds`).toBeLessThan(fullVol);
  }
}

describe('Phase 1 Shared Full-Identity Focus Engine', () => {
  it('enforces full tensor identity (never matches by tensor name alone when selectors mismatch)', () => {
    const focus = createSheetFocusSet([
      {
        tensor: 'kv_v',
        kvHeadIndex: 1,
        tokenStart: 0,
        tokenEnd: 4,
        role: 'operand_b',
      },
      {
        tensor: 'x',
        headIndex: 0,
        groupIndex: 0,
        role: 'operand_b',
      },
      {
        tensor: 'router_bank',
        expertId: 42,
        tokenStart: 3,
        tokenEnd: 3,
        role: 'target',
      },
      {
        tensor: 'conv_buf',
        tapStart: 2,
        tapEnd: 2,
        tokenStart: 5,
        tokenEnd: 5,
        role: 'target',
      },
    ]);

    // KV token range & group identity
    expect(resolveSheetHighlight(focus, { tensor: 'kv_v', kvHeadIndex: 1, tokenIndex: 3 }, 0)).toBe(SHEET_HL_OPERAND_B);
    expect(resolveSheetHighlight(focus, { tensor: 'kv_v', kvHeadIndex: 1, tokenIndex: 5 }, 0)).toBe(SHEET_HL_DIMMED);
    expect(resolveSheetHighlight(focus, { tensor: 'kv_v', kvHeadIndex: 0, tokenIndex: 3 }, 0)).toBe(SHEET_HL_DIMMED);

    // Mamba head identity (same group G0, different head H01 must NOT match H00!)
    expect(resolveSheetHighlight(focus, { tensor: 'x', headIndex: 0, groupIndex: 0 }, 0)).toBe(SHEET_HL_OPERAND_B);
    expect(resolveSheetHighlight(focus, { tensor: 'x', headIndex: 1, groupIndex: 0 }, 0)).toBe(SHEET_HL_DIMMED);

    // MoE expertId + tokenIndex identity
    expect(resolveSheetHighlight(focus, { tensor: 'router_bank', expertId: 42, tokenIndex: 3 }, 0)).toBe(SHEET_HL_ACTIVE);
    expect(resolveSheetHighlight(focus, { tensor: 'router_bank', expertId: 43, tokenIndex: 3 }, 0)).toBe(SHEET_HL_DIMMED);
    expect(resolveSheetHighlight(focus, { tensor: 'router_bank', expertId: 42, tokenIndex: 4 }, 0)).toBe(SHEET_HL_DIMMED);

    // Conv1D tapIndex + tokenIndex identity
    expect(resolveSheetHighlight(focus, { tensor: 'conv_buf', tapIndex: 2, tokenIndex: 5 }, 0)).toBe(SHEET_HL_ACTIVE);
    expect(resolveSheetHighlight(focus, { tensor: 'conv_buf', tapIndex: 1, tokenIndex: 5 }, 0)).toBe(SHEET_HL_DIMMED);
    expect(resolveSheetHighlight(focus, { tensor: 'conv_buf', tapIndex: 2, tokenIndex: 6 }, 0)).toBe(SHEET_HL_DIMMED);

    // Cell/span overlap check
    expect(
      matchesFocusEntry(
        { tensor: 'scores', headIndex: 18, rowStart: 4, rowEnd: 4, colStart: 0, colEnd: 4, role: 'operand_a' },
        { tensor: 'scores', headIndex: 18, rowStart: 4, rowEnd: 4, colStart: 2, colEnd: 2 },
      ),
    ).toBe(true);
    expect(
      matchesFocusEntry(
        { tensor: 'scores', headIndex: 18, rowStart: 4, rowEnd: 4, colStart: 0, colEnd: 4, role: 'operand_a' },
        { tensor: 'scores', headIndex: 18, rowStart: 5, rowEnd: 5, colStart: 2, colEnd: 2 },
      ),
    ).toBe(false);

    // Priority ordering: target > operand_a > operand_b
    const multiRole = createSheetFocusSet([
      { tensor: 'y', role: 'operand_b' },
      { tensor: 'y', role: 'operand_a' },
      { tensor: 'y', role: 'target' },
    ]);
    expect(resolveSheetFocusRole(multiRole, { tensor: 'y' })).toBe('target');
  });
});

describe('Phase 1 Mamba-2 Exact Operand Highlighting, Dimming & opFocusBounds', () => {
  it('highlights ONLY H_t (target), H_{t-1} (A), x[H00] (B), B (B), Δt (A) and dims all other heads, stages & bank tiles', () => {
    const scene = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'mamba-head-0', // even with head part selected, cell trace must NOT light all 15 stages!
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
    });

    const byTensor = (key: string) =>
      scene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(scene, idx) }))
        .filter((e) => e.box.tensorKey === key);

    // Exact target & upstream operands
    expect(byTensor('h_curr').map((e) => e.hl)).toEqual([SHEET_HL_ACTIVE]);
    expect(byTensor('h_prev').map((e) => e.hl)).toEqual([SHEET_HL_OPERAND_A]);
    expect(byTensor('dt').map((e) => e.hl)).toEqual([SHEET_HL_OPERAND_A]);
    expect(byTensor('b').map((e) => e.hl)).toEqual([SHEET_HL_OPERAND_B]);

    // ONLY x head H00 highlighted (operand B); H01..H15 MUST ALL dim!
    const xSheets = byTensor('x');
    expect(xSheets).toHaveLength(16);
    for (const entry of xSheets) {
      if (entry.box.headIndex === 0) {
        expect(entry.hl).toBe(SHEET_HL_OPERAND_B);
      } else {
        expect(entry.hl).toBe(SHEET_HL_DIMMED);
      }
    }

    // All unrelated stages MUST dim cleanly
    for (const unrelated of ['input', 'rmsnorm', 'w_in', 'z', 'xbc_pre', 'conv_buf', 'c', 'y', 'y_norm', 'w_out', 'residual']) {
      const sheets = byTensor(unrelated);
      expect(sheets.length).toBeGreaterThan(0);
      for (const entry of sheets) {
        expect(entry.hl, `Mamba ${unrelated} should dim during H_t trace`).toBe(SHEET_HL_DIMMED);
      }
    }

    // 64-head bank: ONLY H00's 4 tiles show pathway; all other 60 bank tiles dim cleanly
    const bankTiles = scene.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: highlightAt(scene, idx) }))
      .filter((e) => e.box.partId === 'ssm-state');
    expect(bankTiles).toHaveLength(64);
    const activeBank = bankTiles.filter((e) => e.hl === SHEET_HL_PATHWAY);
    const dimmedBank = bankTiles.filter((e) => e.hl === SHEET_HL_DIMMED);
    expect(activeBank).toHaveLength(4);
    expect(activeBank.every((e) => e.box.headIndex === 0)).toBe(true);
    expect(dimmedBank).toHaveLength(60);
  });

  it('matches exact Conv1D tapIndex on conv_buf inspection and preserves BF16 gold rims when idle in FP8 Selective', () => {
    const tapTrace = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'conv_buf', row: 2, col: 4, tapIndex: 2, tokenIndex: 7 },
    });

    const convTaps = tapTrace.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: highlightAt(tapTrace, idx) }))
      .filter((e) => e.box.tensorKey === 'conv_buf');
    expect(convTaps).toHaveLength(4);
    for (const tap of convTaps) {
      expect(tap.hl).toBe(tap.box.tapIndex === 2 ? SHEET_HL_ACTIVE : SHEET_HL_DIMMED);
    }

    // Idle FP8 Selective preserves BF16 gold rim on Conv1D taps and Bay 42 Feeder
    const idleBay01Fp8 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });
    const idleConvTaps = idleBay01Fp8.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: highlightAt(idleBay01Fp8, idx) }))
      .filter((e) => e.box.tensorKey === 'conv_buf');
    expect(idleConvTaps.every((t) => t.hl === SHEET_HL_BF16_GOLD)).toBe(true);

    // Valid, tight opFocusBounds[0..5]
    assertValidOpFocusBounds(idleBay01Fp8, 'Mamba-2');
  });
});

describe('Phase 1 MoE 128+1 Exact Operand Highlighting, Dimming & opFocusBounds', () => {
  it('isolates exact router_bank expertId, topk_weights winner, routed ReLU² chain, and 2× shared branch while dimming unrelated sheets & tiles', () => {
    const base = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
    });
    const sim = base.moeSim!;
    const focusTok = 7;
    const winners = Array.from(sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6));
    const slot2Expert = winners[2];

    // (1) Inspecting router_bank tile for slot2Expert (even with stepperStep=1 set)
    const routerScene = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeExpert: slot2Expert,
      stepperStep: 1,
      activeCell: {
        tensor: 'router_bank',
        row: Math.floor(slot2Expert / 16),
        col: slot2Expert % 16,
        tokenIndex: focusTok,
        expertId: slot2Expert,
      },
    });

    const routerBankEntries = routerScene.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: highlightAt(routerScene, idx) }))
      .filter((e) => e.box.tensorKey === 'router_bank');
    // 6 active * 2 plates + 122 idle = 134 boxes
    expect(routerBankEntries).toHaveLength(134);
    const activeRouterTiles = routerBankEntries.filter((e) => e.hl === SHEET_HL_ACTIVE);
    const dimmedRouterTiles = routerBankEntries.filter((e) => e.hl === SHEET_HL_DIMMED);
    expect(activeRouterTiles).toHaveLength(2); // the 2 micro-plates of slot2Expert ONLY
    expect(activeRouterTiles.every((e) => e.box.headIndex === slot2Expert)).toBe(true);
    expect(dimmedRouterTiles).toHaveLength(132); // other 5 active experts (10 plates) + 122 idle tiles!

    // Upstream operand A is rmsnorm; unrelated sheets dim cleanly
    const rmsEntry = routerScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'rmsnorm');
    expect(highlightAt(routerScene, rmsEntry)).toBe(SHEET_HL_OPERAND_A);
    for (const dimKey of ['input', 'w_router', 'topk_weights', 'routed_w_up', 'routed_act', 'routed_sum', 'shared_w_up', 'shared_act', 'shared_out', 'moe_out', 'residual']) {
      const idx = routerScene.solidPickBoxes.findIndex((b) => b.tensorKey === dimKey);
      expect(highlightAt(routerScene, idx), `MoE ${dimKey} should dim on router_bank trace`).toBe(SHEET_HL_DIMMED);
    }

    // (2) Inspecting topk_weights slot #2 -> highlights topk_weights (ACTIVE) & ONLY slot2Expert tile (OPERAND_A)
    const topkScene = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'router',
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 2,
      activeCell: {
        tensor: 'topk_weights',
        row: 0,
        col: 2,
        tokenIndex: focusTok,
        expertId: slot2Expert,
        rankSlot: 2,
      },
    });
    const topkIdx = topkScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'topk_weights');
    expect(highlightAt(topkScene, topkIdx)).toBe(SHEET_HL_ACTIVE);
    const topkBank = topkScene.solidPickBoxes
      .map((box, idx) => ({ box, idx, hl: highlightAt(topkScene, idx) }))
      .filter((e) => e.box.tensorKey === 'router_bank');
    const opABank = topkBank.filter((e) => e.hl === SHEET_HL_OPERAND_A);
    expect(opABank).toHaveLength(2);
    expect(opABank.every((e) => e.box.headIndex === slot2Expert)).toBe(true);
    expect(topkBank.filter((e) => e.hl === SHEET_HL_DIMMED)).toHaveLength(132);

    // (3) Inspecting routed_act (ReLU²) -> highlights ONLY routed_act (ACTIVE) & routed_pre (OPERAND_A)
    const routedActScene = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      activeExpert: slot2Expert,
      stepperStep: 3,
      activeCell: {
        tensor: 'routed_act',
        row: 1,
        col: 0,
        tokenIndex: focusTok,
        expertId: slot2Expert,
      },
    });
    expect(highlightAt(routedActScene, routedActScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'routed_act'))).toBe(SHEET_HL_ACTIVE);
    expect(highlightAt(routedActScene, routedActScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'routed_pre'))).toBe(SHEET_HL_OPERAND_A);
    expect(highlightAt(routedActScene, routedActScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'routed_w_up'))).toBe(SHEET_HL_DIMMED);
    expect(highlightAt(routedActScene, routedActScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'routed_w_down'))).toBe(SHEET_HL_DIMMED);
    expect(highlightAt(routedActScene, routedActScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'shared_act'))).toBe(SHEET_HL_DIMMED);

    // (4) Inspecting 2× shared_act -> highlights ONLY shared_act (ACTIVE) & shared_w_up (OPERAND_A)
    const sharedScene = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      focusToken: focusTok,
      stepperStep: 4,
      activeCell: {
        tensor: 'shared_act',
        row: 2,
        col: 0,
        tokenIndex: focusTok,
      },
    });
    expect(highlightAt(sharedScene, sharedScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'shared_act'))).toBe(SHEET_HL_ACTIVE);
    expect(highlightAt(sharedScene, sharedScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'shared_w_up'))).toBe(SHEET_HL_OPERAND_A);
    expect(highlightAt(sharedScene, sharedScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'shared_w_down'))).toBe(SHEET_HL_DIMMED);
    expect(highlightAt(sharedScene, sharedScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'routed_act'))).toBe(SHEET_HL_DIMMED);

    // Idle FP8 Selective preserves BF16/FP32 gold rim on W_router and Top-6 Router
    const idleFp8 = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });
    expect(highlightAt(idleFp8, idleFp8.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_router'))).toBe(SHEET_HL_BF16_GOLD);
    expect(highlightAt(idleFp8, idleFp8.solidPickBoxes.findIndex((b) => b.tensorKey === 'topk_weights'))).toBe(SHEET_HL_BF16_GOLD);

    // Valid, tight opFocusBounds[0..5]
    assertValidOpFocusBounds(idleFp8, 'MoE 128+1');
  });
});

describe('Phase 1 GQA Exact Operand Highlighting, Dimming & opFocusBounds (Both 3/4 Spatial & Near-Plan)', () => {
  for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
    it(`isolates Q18, Scores[Q18], Softmax[Q18], and V_out[Q18] with exact KV1 token slices in ${viewAngle} view`, () => {
      const T = 8;

      // (1) Inspecting cell in Q18: ONLY Q18 (ACTIVE), W_Q (A), RMSNorm (B); Q00..Q17, Q19..Q31 & all KV dim
      const qScene = buildGqaCutaway({
        bayId: 43,
        tokens: T,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        activeCell: { tensor: 'q', headIndex: 18, kvHeadIndex: 1, row: 7, col: 2 },
        viewAngle,
      });

      const qBoxes = qScene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(qScene, idx) }))
        .filter((e) => e.box.tensorKey === 'q');
      expect(qBoxes).toHaveLength(32);
      for (const e of qBoxes) {
        expect(e.hl).toBe(e.box.headIndex === 18 ? SHEET_HL_ACTIVE : SHEET_HL_DIMMED);
      }
      expect(highlightAt(qScene, qScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_q'))).toBe(SHEET_HL_OPERAND_A);
      expect(highlightAt(qScene, qScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'rmsnorm'))).toBe(SHEET_HL_OPERAND_B);
      expect(highlightAt(qScene, qScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_k'))).toBe(SHEET_HL_DIMMED);
      expect(highlightAt(qScene, qScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'scores' && b.kvHeadIndex === 1))).toBe(SHEET_HL_DIMMED);

      // (2) Inspecting Scores[Q18, q=6, k=2]: ONLY Scores[KV1] (ACTIVE), Q18 (A), KV1 K-slice t=2 (B)
      const scoresScene = buildGqaCutaway({
        bayId: 43,
        tokens: T,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        activeCell: { tensor: 'scores', headIndex: 18, kvHeadIndex: 1, row: 6, col: 2 },
        viewAngle,
      });

      expect(highlightAt(scoresScene, scoresScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'scores' && b.kvHeadIndex === 1))).toBe(SHEET_HL_ACTIVE);
      expect(highlightAt(scoresScene, scoresScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'scores' && b.kvHeadIndex === 0))).toBe(SHEET_HL_DIMMED);

      // ONLY Q18 = OPERAND_A; ALL other 31 Q heads (including Q16, Q17, Q19..Q31) = DIMMED
      for (let h = 0; h < 32; h++) {
        const idx = scoresScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'q' && b.headIndex === h);
        expect(highlightAt(scoresScene, idx)).toBe(h === 18 ? SHEET_HL_OPERAND_A : SHEET_HL_DIMMED);
      }

      // ONLY K1[t=2] = OPERAND_B; all other K1 slices, all V1 slices, and all KV0 slices = DIMMED
      const kvKScores = scoresScene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(scoresScene, idx) }))
        .filter((e) => e.box.tensorKey === 'kv_k');
      for (const e of kvKScores) {
        const expected = e.box.kvHeadIndex === 1 && e.box.tokenIndex === 2 ? SHEET_HL_OPERAND_B : SHEET_HL_DIMMED;
        expect(e.hl).toBe(expected);
      }
      const kvVScores = scoresScene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(scoresScene, idx) }))
        .filter((e) => e.box.tensorKey === 'kv_v');
      expect(kvVScores.every((e) => e.hl === SHEET_HL_DIMMED)).toBe(true);

      // (3) Inspecting Softmax[Q18, q=6, k=2]: ONLY Softmax[KV1] (ACTIVE) & Scores[KV1] (A)
      const smScene = buildGqaCutaway({
        bayId: 43,
        tokens: T,
        precision: 'bf16',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        activeCell: { tensor: 'softmax', headIndex: 18, kvHeadIndex: 1, row: 6, col: 2 },
        viewAngle,
      });
      expect(highlightAt(smScene, smScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 1))).toBe(SHEET_HL_ACTIVE);
      expect(highlightAt(smScene, smScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'scores' && b.kvHeadIndex === 1))).toBe(SHEET_HL_OPERAND_A);
      expect(highlightAt(smScene, smScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 0))).toBe(SHEET_HL_DIMMED);
      expect(highlightAt(smScene, smScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'scores' && b.kvHeadIndex === 0))).toBe(SHEET_HL_DIMMED);
      expect(highlightAt(smScene, smScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'q' && b.headIndex === 18))).toBe(SHEET_HL_DIMMED);

      // (4) Inspecting V_out[Q18, q=4, d=1] (col = (18-16)*4 + 1 = 9):
      //     ONLY V_out[KV1] (ACTIVE), Softmax[KV1] (A), and KV1 V-slices t=0..4 (B); future V1 t=5..7 & all KV0 dim!
      const vOutScene = buildGqaCutaway({
        bayId: 43,
        tokens: T,
        precision: 'fp8_selective',
        selectedPart: 'q-head-18',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        activeCell: { tensor: 'v_out', headIndex: 18, kvHeadIndex: 1, row: 4, col: 9 },
        viewAngle,
      });
      expect(highlightAt(vOutScene, vOutScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'v_out' && b.kvHeadIndex === 1))).toBe(SHEET_HL_ACTIVE);
      expect(highlightAt(vOutScene, vOutScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'v_out' && b.kvHeadIndex === 0))).toBe(SHEET_HL_DIMMED);
      expect(highlightAt(vOutScene, vOutScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 1))).toBe(SHEET_HL_OPERAND_A);
      expect(highlightAt(vOutScene, vOutScene.solidPickBoxes.findIndex((b) => b.tensorKey === 'softmax' && b.kvHeadIndex === 0))).toBe(SHEET_HL_DIMMED);

      const kvVOut = vOutScene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(vOutScene, idx) }))
        .filter((e) => e.box.tensorKey === 'kv_v');
      for (const e of kvVOut) {
        const expected =
          e.box.kvHeadIndex === 1 && e.box.tokenIndex !== undefined && e.box.tokenIndex <= 4
            ? SHEET_HL_OPERAND_B
            : SHEET_HL_DIMMED;
        expect(e.hl).toBe(expected);
      }
      const kvKOut = vOutScene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(vOutScene, idx) }))
        .filter((e) => e.box.tensorKey === 'kv_k');
      expect(kvKOut.every((e) => e.hl === SHEET_HL_DIMMED)).toBe(true);

      // Ghost rails for opposite KV0 group dim cleanly during KV1 cell trace
      const kv0GhostRails = vOutScene.solidPickBoxes
        .map((box, idx) => ({ box, idx, hl: highlightAt(vOutScene, idx), kind: sheetKindAt(vOutScene, idx) }))
        .filter((e) => e.box.partId === 'kv-head-0' && e.kind === SHEET_KIND_GHOST_RAIL);
      expect(kv0GhostRails).toHaveLength(2);
      expect(kv0GhostRails.every((e) => e.hl === SHEET_HL_DIMMED)).toBe(true);

      // Idle FP8 Selective preserves BF16 gold rims on compute sheets
      const idleFp8 = buildGqaCutaway({
        bayId: 43,
        tokens: T,
        precision: 'fp8_selective',
        selectedPart: null,
        hoveredPart: null,
        viewAngle,
      });
      expect(highlightAt(idleFp8, idleFp8.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_q'))).toBe(SHEET_HL_BF16_GOLD);
      expect(highlightAt(idleFp8, idleFp8.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_o'))).toBe(SHEET_HL_BF16_GOLD);

      // Valid, tight opFocusBounds[0..5]
      assertValidOpFocusBounds(vOutScene, `GQA ${viewAngle}`);

      // Group 0 vs Group 1 selection shifts opFocusBounds[2..4] to the active group's spatial lane
      const grp0Scene = buildGqaCutaway({
        bayId: 43,
        tokens: T,
        precision: 'bf16',
        selectedPart: 'q-head-0',
        hoveredPart: null,
        activeGroup0Head: 0,
        activeGroup1Head: 18,
        viewAngle,
      });
      assertValidOpFocusBounds(grp0Scene, `GQA ${viewAngle} G0`);
      for (const step of [2, 3, 4]) {
        const b0 = grp0Scene.opFocusBounds![step];
        const b1 = vOutScene.opFocusBounds![step];
        const c0 = viewAngle === 'three_quarter' ? (b0.min[2] + b0.max[2]) * 0.5 : (b0.min[0] + b0.max[0]) * 0.5;
        const c1 = viewAngle === 'three_quarter' ? (b1.min[2] + b1.max[2]) * 0.5 : (b1.min[0] + b1.max[0]) * 0.5;
        expect(c0).toBeLessThan(c1);
      }
    });
  }
});
