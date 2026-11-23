import { describe, expect, it } from 'vitest';
import {
  bodyPrecision,
  conv1dPrecision,
  countByKind,
  kvCachePrecision,
  NEMOTRON_SPEC,
  ssmStateDtype,
} from '../src/model/spec';
import { buildMacroSpine, SOLID_INSTANCE_FLOATS } from '../src/scene/spine';

describe('Fixture 1: Pattern and roles', () => {
  it('verifies length 52, kind counts 23/23/6, attention IDs, feeder IDs, and sensitivity', () => {
    expect(NEMOTRON_SPEC.pattern).toHaveLength(52);
    expect(NEMOTRON_SPEC.bays).toHaveLength(52);

    const counts = countByKind(NEMOTRON_SPEC);
    expect(counts.M).toBe(23);
    expect(counts.E).toBe(23);
    expect(counts['*']).toBe(6);

    const attentionIds = NEMOTRON_SPEC.bays.filter((b) => b.kind === '*').map((b) => b.id);
    expect(attentionIds).toEqual([6, 13, 20, 27, 34, 43]);

    const feederIds = NEMOTRON_SPEC.bays.filter((b) => b.feeder).map((b) => b.id);
    expect(feederIds).toEqual([5, 12, 19, 26, 33, 42]);

    const sensitiveBays = NEMOTRON_SPEC.bays.filter((b) => b.sensitive);
    const eligibleBays = NEMOTRON_SPEC.bays.filter((b) => !b.sensitive);
    expect(sensitiveBays).toHaveLength(12);
    expect(eligibleBays).toHaveLength(40);

    // Body precision under BF16 vs FP8 Selective
    for (const bay of NEMOTRON_SPEC.bays) {
      expect(bodyPrecision(bay, 'bf16')).toBe('bf16');
      expect(bodyPrecision(bay, 'fp8_selective')).toBe(bay.sensitive ? 'bf16' : 'fp8');
    }

    // All 23 Conv1D overrides stay BF16 independently of bay body precision
    const mambaBays = NEMOTRON_SPEC.bays.filter((b) => b.kind === 'M');
    expect(mambaBays).toHaveLength(23);
    for (const bay of mambaBays) {
      expect(conv1dPrecision(bay, 'bf16')).toBe('bf16');
      expect(conv1dPrecision(bay, 'fp8_selective')).toBe('bf16');
      expect(ssmStateDtype(bay)).toBe('fp32');
    }

    // All 6 KV cache overrides switch to FP8 in Selective mode even beside BF16 attention bodies
    const attnBays = NEMOTRON_SPEC.bays.filter((b) => b.kind === '*');
    expect(attnBays).toHaveLength(6);
    for (const bay of attnBays) {
      expect(bodyPrecision(bay, 'fp8_selective')).toBe('bf16');
      expect(kvCachePrecision(bay, 'bf16')).toBe('bf16');
      expect(kvCachePrecision(bay, 'fp8_selective')).toBe('fp8');
    }
  });

  it('encodes component-level FP8 flags and M1.5 vertical tower geometry cleanly in macro spine', () => {
    const scene = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'fp8_selective',
      selectedBay: null,
      hoveredBay: null,
    });

    // Translucent 3D residual spine & directional block arrows are emitted
    expect(scene.ribbonCount).toBeGreaterThan(50);

    // Count Conv1D bands (ex = 0.56, ey = 0.96, ez = 0.94) and verify fp8Flag === 0 on all 23
    // Count KV slices (ex = 0.36, ey = 0.52, ez = 0.30) and verify all 6*4*8 = 192 have fp8Flag === 1
    let conv1dCount = 0;
    let kvSliceCount = 0;
    let fp8KvSliceCount = 0;

    for (let i = 0; i < scene.solidCount; i++) {
      const o = i * SOLID_INSTANCE_FLOATS;
      const cx = scene.solidData[o + 0];
      const cz = scene.solidData[o + 2];
      const ex = scene.solidData[o + 3];
      const ey = scene.solidData[o + 4];
      const ez = scene.solidData[o + 5];
      const fp8 = scene.solidData[o + 7];

      if (
        Math.abs(cx - 0.45) < 1e-4 &&
        Math.abs(cz - 0.08) < 1e-4 &&
        Math.abs(ex - 0.56) < 1e-4 &&
        Math.abs(ey - 0.96) < 1e-4
      ) {
        conv1dCount++;
        expect(fp8).toBe(0);
      }
      if (Math.abs(ex - 0.36) < 1e-4 && Math.abs(ey - 0.52) < 1e-4 && Math.abs(ez - 0.30) < 1e-4) {
        kvSliceCount++;
        if (fp8 === 1) fp8KvSliceCount++;
      }
    }

    expect(conv1dCount).toBe(23);
    // 6 attention bays * 4 rails * 8 tokens = 192 slices, all FP8 in selective mode
    expect(kvSliceCount).toBe(6 * 4 * 8);
    expect(fp8KvSliceCount).toBe(6 * 4 * 8);
  });
});

describe('Fixture 2: Report brackets', () => {
  it('expands motifs to match checkpoint string and partitions 01–52 once', () => {
    const expectedSpans: readonly [number, number][] = [
      [1, 35],
      [36, 41],
      [42, 44],
      [45, 52],
    ];
    const expectedLengths = [35, 6, 3, 8];

    expect(NEMOTRON_SPEC.brackets).toHaveLength(4);

    let concatenated = '';
    const seen = new Set<number>();
    let totalLen = 0;

    NEMOTRON_SPEC.brackets.forEach((bracket, idx) => {
      const [first, last, motif, repeats] = bracket;
      expect([first, last]).toEqual(expectedSpans[idx]);
      const spanLen = last - first + 1;
      expect(spanLen).toBe(expectedLengths[idx]);
      totalLen += spanLen;

      const expanded = motif.repeat(repeats);
      expect(expanded).toHaveLength(spanLen);
      concatenated += expanded;

      for (let id = first; id <= last; id++) {
        expect(seen.has(id)).toBe(false);
        seen.add(id);
        expect(NEMOTRON_SPEC.bays[id - 1].bracket).toBe(idx);
      }
    });

    expect(totalLen).toBe(52);
    expect(seen.size).toBe(52);
    expect(concatenated).toBe(NEMOTRON_SPEC.pattern);
  });
});
