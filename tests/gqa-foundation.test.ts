import { describe, expect, it } from 'vitest';
import { pickMacroBay } from '../src/render/pick';
import { FLOW_RIBBON_VERTEX_FLOATS } from '../src/render/ribbons';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
} from '../src/render/sheets';
import {
  buildGqaCutaway,
  GQA_FIT_BOUNDS,
  kvHeadForQueryHead,
  parseQueryHeadPartId,
} from '../src/scene/gqaCutaway';

describe('Stage A+1 Increment 1: GQA Rendering Foundation', () => {
  it('makes all 32 query heads (Q0..Q31) individually pickable and maps 0..15 -> KV0, 16..31 -> KV1', () => {
    const gqa = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
    });

    const qBoxes = gqa.solidPickBoxes.filter((b) => b.partId === 'q-proj');
    expect(qBoxes).toHaveLength(32);

    const seenHeads = new Set<number>();
    for (let i = 0; i < 32; i++) {
      const box = qBoxes[i];
      expect(box.headIndex).toBe(i);
      const expectedKv: 0 | 1 = i < 16 ? 0 : 1;
      expect(kvHeadForQueryHead(i)).toBe(expectedKv);
      expect(box.kvHeadIndex).toBe(expectedKv);
      expect(parseQueryHeadPartId(`q-head-${i}`)).toBe(i);
      expect(gqa.partCatalog[`q-head-${i}`]).toBeDefined();

      // Cast a vertical downward ray directly onto the center of each query head's AABB
      const cx = (box.min[0] + box.max[0]) * 0.5;
      const cz = (box.min[2] + box.max[2]) * 0.5;
      const hit = pickMacroBay(
        { origin: [cx, 5.0, cz], direction: [0, -1, 0] },
        gqa.solidPickBoxes,
        [],
      );
      expect(hit).not.toBeNull();
      expect(hit!.partId).toBe('q-proj');
      expect(hit!.headIndex).toBe(i);
      expect(hit!.kvHeadIndex).toBe(expectedKv);
      seenHeads.add(hit!.headIndex!);
    }
    expect(seenHeads.size).toBe(32);

    // Selecting Q00 highlights Q00 as ACTIVE and KV Head 0 slices as PATHWAY (not KV Head 1)
    const selQ0 = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-0',
      hoveredPart: null,
    });
    expect(selQ0.gqaStats?.selectedQueryHead).toBe(0);
    const q0Idx = selQ0.solidPickBoxes.findIndex((b) => b.headIndex === 0);
    expect(selQ0.sheetData![q0Idx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_ACTIVE);

    const kv0SliceIdx = selQ0.solidPickBoxes.findIndex(
      (b, idx) => b.partId === 'kv-head-0' && selQ0.sheetData![idx * SHEET_INSTANCE_FLOATS + 8] === 0,
    );
    const kv1SliceIdx = selQ0.solidPickBoxes.findIndex(
      (b, idx) => b.partId === 'kv-head-1' && selQ0.sheetData![idx * SHEET_INSTANCE_FLOATS + 8] === 0,
    );
    expect(selQ0.sheetData![kv0SliceIdx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_PATHWAY);
    expect(selQ0.sheetData![kv1SliceIdx * SHEET_INSTANCE_FLOATS + 11]).not.toBe(SHEET_HL_PATHWAY);

    // Selecting Q18 highlights Q18 as ACTIVE and KV Head 1 slices as PATHWAY (not KV Head 0)
    const selQ18 = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-18',
      hoveredPart: null,
    });
    expect(selQ18.gqaStats?.selectedQueryHead).toBe(18);
    const q18Idx = selQ18.solidPickBoxes.findIndex((b) => b.headIndex === 18);
    expect(selQ18.sheetData![q18Idx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_ACTIVE);
    expect(selQ18.sheetData![kv1SliceIdx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_PATHWAY);
    expect(selQ18.sheetData![kv0SliceIdx * SHEET_INSTANCE_FLOATS + 11]).not.toBe(SHEET_HL_PATHWAY);
  });

  it('uses zero voxel-bead/cube-chain arrows and produces valid finite smooth flat ribbon geometry', () => {
    const gqa = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-5',
      hoveredPart: null,
    });

    expect(gqa.ribbonCount).toBe(0);
    expect(gqa.ribbonData.length).toBe(0);
    expect(gqa.gqaStats?.voxelArrowCount).toBe(0);
    expect(gqa.gqaStats?.usesSmoothRibbons).toBe(true);

    expect(gqa.flowVertices).toBeDefined();
    expect(gqa.flowIndices).toBeDefined();
    const verts = gqa.flowVertices!;
    const indices = gqa.flowIndices!;
    expect(verts.length).toBeGreaterThan(0);
    expect(verts.length % FLOW_RIBBON_VERTEX_FLOATS).toBe(0);
    expect(indices.length).toBeGreaterThan(0);
    expect(indices.length % 3).toBe(0);

    const vertCount = verts.length / FLOW_RIBBON_VERTEX_FLOATS;
    for (let i = 0; i < verts.length; i++) {
      expect(Number.isFinite(verts[i])).toBe(true);
    }
    for (let i = 0; i < indices.length; i++) {
      expect(indices[i]).toBeLessThan(vertCount);
    }
  });

  it('builds thin sheets cleanly across T = 1, 8, 16 with invariant bounds and growing T x T causal sheets', () => {
    const scoreSizes: number[] = [];

    for (const tokens of [1, 8, 16]) {
      for (const precision of ['bf16', 'fp8_selective'] as const) {
        const gqa = buildGqaCutaway({
          bayId: 6,
          tokens,
          precision,
          selectedPart: null,
          hoveredPart: null,
        });

        // Bounds remain strictly invariant across T and precision
        expect(gqa.fitBounds).toEqual(GQA_FIT_BOUNDS);
        expect(gqa.sheetCount).toBeGreaterThan(40);
        expect(gqa.sheetCount).toBe(gqa.solidCount);
        expect(gqa.sheetData!.length).toBe(gqa.sheetCount! * SHEET_INSTANCE_FLOATS);

        // Every sheet is finite and genuinely thin (ey <= 0.09)
        let causalScoreCount = 0;
        let causalSoftmaxCount = 0;
        let firstScoreExtent = 0;
        for (let i = 0; i < gqa.sheetCount!; i++) {
          const base = i * SHEET_INSTANCE_FLOATS;
          for (let f = 0; f < SHEET_INSTANCE_FLOATS; f++) {
            expect(Number.isFinite(gqa.sheetData![base + f])).toBe(true);
          }
          const ex = gqa.sheetData![base + 3];
          const ey = gqa.sheetData![base + 4];
          const ez = gqa.sheetData![base + 5];
          const kind = gqa.sheetData![base + 8];
          expect(ex).toBeGreaterThan(0);
          expect(ey).toBeGreaterThan(0);
          expect(ey).toBeLessThanOrEqual(0.09);
          expect(ez).toBeGreaterThan(0);

          if (kind === SHEET_KIND_CAUSAL_SCORES) {
            causalScoreCount++;
            // Explicitly square T x T
            expect(Math.abs(ex - ez)).toBeLessThan(1e-5);
            expect(gqa.sheetData![base + 6]).toBe(tokens);
            expect(gqa.sheetData![base + 7]).toBe(tokens);
            firstScoreExtent = ex;
          } else if (kind === SHEET_KIND_CAUSAL_SOFTMAX) {
            causalSoftmaxCount++;
            expect(Math.abs(ex - ez)).toBeLessThan(1e-5);
            expect(gqa.sheetData![base + 6]).toBe(tokens);
            expect(gqa.sheetData![base + 7]).toBe(tokens);
          }
        }

        expect(causalScoreCount).toBe(2);
        expect(causalSoftmaxCount).toBe(2);
        expect(gqa.gqaStats?.kvTokenSliceCount).toBe(2 * tokens);

        if (precision === 'bf16') {
          scoreSizes.push(firstScoreExtent);
        }
      }
    }

    // Causal T x T sheet strictly grows with T: T=1 < T=8 < T=16
    expect(scoreSizes[0]).toBeLessThan(scoreSizes[1]);
    expect(scoreSizes[1]).toBeLessThan(scoreSizes[2]);
  });
});
