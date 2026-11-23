import { describe, expect, it } from 'vitest';
import {
  activeRoutedExpertIndices,
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
} from '../src/scene/cutaways';
import { SOLID_INSTANCE_FLOATS } from '../src/scene/spine';

function solidFp8FlagsByPart(
  solidData: Float32Array,
  solidPickBoxes: readonly { partId?: string }[],
  partId: string,
): number[] {
  const flags: number[] = [];
  for (let i = 0; i < solidPickBoxes.length; i++) {
    if (solidPickBoxes[i].partId === partId) {
      flags.push(solidData[i * SOLID_INSTANCE_FLOATS + 7]);
    }
  }
  return flags;
}

function solidPhasesByPart(
  solidData: Float32Array,
  solidPickBoxes: readonly { partId?: string }[],
  partId: string,
): number[] {
  const phases: number[] = [];
  for (let i = 0; i < solidPickBoxes.length; i++) {
    if (solidPickBoxes[i].partId === partId) {
      phases.push(solidData[i * SOLID_INSTANCE_FLOATS + 8]);
    }
  }
  return phases;
}

describe('Milestone 2 Cutaway 1: Mamba-2 (Bay 01 Ordinary vs Bay 42 Feeder)', () => {
  it('keeps Conv1D BF16 and SSM state count strictly invariant across T=1..16 while updating phase in place', () => {
    const t1 = buildMambaCutaway({
      bayId: 1,
      tokens: 1,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });
    const t16 = buildMambaCutaway({
      bayId: 1,
      tokens: 16,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });

    // Strictly invariant geometry & SSM block count across T = 1 vs T = 16
    expect(t1.mambaStats?.ssmHeadCount).toBe(64);
    expect(t16.mambaStats?.ssmHeadCount).toBe(64);
    expect(t1.solidCount).toBe(t16.solidCount);

    const ssmBoxesT1 = t1.solidPickBoxes.filter((b) => b.partId === 'ssm-state');
    const ssmBoxesT16 = t16.solidPickBoxes.filter((b) => b.partId === 'ssm-state');
    expect(ssmBoxesT1).toHaveLength(64);
    expect(ssmBoxesT16).toHaveLength(64);

    // Phase shifts in place when T steps 1 -> 16
    const phasesT1 = solidPhasesByPart(t1.solidData, t1.solidPickBoxes, 'ssm-state');
    const phasesT16 = solidPhasesByPart(t16.solidData, t16.solidPickBoxes, 'ssm-state');
    expect(phasesT1[0]).not.toBeCloseTo(phasesT16[0], 3);

    // All 4 Conv1D tap strips stay BF16 (fp8Flag === 0) and all 64 SSM slabs stay FP32 (fp8Flag === 0)
    const convFlags = solidFp8FlagsByPart(t1.solidData, t1.solidPickBoxes, 'conv1d');
    const ssmFlags = solidFp8FlagsByPart(t1.solidData, t1.solidPickBoxes, 'ssm-state');
    expect(convFlags).toHaveLength(4);
    expect(convFlags.every((f) => f === 0)).toBe(true);
    expect(ssmFlags).toHaveLength(64);
    expect(ssmFlags.every((f) => f === 0)).toBe(true);
    expect(t1.mambaStats?.conv1dFp8).toBe(false);
    expect(t1.mambaStats?.ssmStateFp8).toBe(false);
  });

  it('switches Bay 01 In-Proj/Out-Proj to FP8 in FP8 Selective while preserving Bay 42 Feeder in BF16', () => {
    const bay01 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });
    const bay42 = buildMambaCutaway({
      bayId: 42,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });

    // Bay 01: In-Proj & Out-Proj are FP8 (fp8Flag === 1)
    expect(bay01.mambaStats?.isFeeder).toBe(false);
    expect(bay01.mambaStats?.inProjFp8).toBe(true);
    expect(bay01.mambaStats?.outProjFp8).toBe(true);
    expect(solidFp8FlagsByPart(bay01.solidData, bay01.solidPickBoxes, 'in-proj')).toEqual([1]);
    expect(solidFp8FlagsByPart(bay01.solidData, bay01.solidPickBoxes, 'out-proj')).toEqual([1]);

    // Bay 42 Feeder: entire compute block stays BF16 (fp8Flag === 0)
    expect(bay42.mambaStats?.isFeeder).toBe(true);
    expect(bay42.mambaStats?.inProjFp8).toBe(false);
    expect(bay42.mambaStats?.outProjFp8).toBe(false);
    expect(solidFp8FlagsByPart(bay42.solidData, bay42.solidPickBoxes, 'in-proj')).toEqual([0]);
    expect(solidFp8FlagsByPart(bay42.solidData, bay42.solidPickBoxes, 'out-proj')).toEqual([0]);
    expect(
      bay42.landmarks.some((lm) => lm.label.includes('Bay 42 Feeder') && lm.label.includes('BF16')),
    ).toBe(true);
  });
});

describe('Milestone 2 Cutaway 2: Granular MoE 128+1 (Bay 02)', () => {
  it('activates exactly 6 of 128 routed experts for every T in [1..16] with deterministic variation', () => {
    const signatures = new Set<string>();

    for (let t = 1; t <= 16; t++) {
      const picks = activeRoutedExpertIndices(t);
      expect(picks).toHaveLength(6);
      expect(new Set(picks).size).toBe(6);
      for (const id of picks) {
        expect(id).toBeGreaterThanOrEqual(0);
        expect(id).toBeLessThan(128);
      }
      // Deterministic repeat call
      expect(activeRoutedExpertIndices(t)).toEqual(picks);
      signatures.add(picks.join(','));
    }

    // Routing visibly shifts across tokens
    expect(signatures.size).toBeGreaterThanOrEqual(12);
  });

  it('encodes 128 routed slots (6 active × 2 non-gated ReLU² plates + 122 idle), 1 shared expert (3,712 = 2×1,856), and 0 B cache', () => {
    const moeBf16 = buildMoECutaway({
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
    });
    const moeFp8 = buildMoECutaway({
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });

    const stats = moeBf16.moeStats!;
    expect(stats.routedTotal).toBe(128);
    expect(stats.routedActiveCount).toBe(6);
    expect(stats.sharedExpertCount).toBe(1);
    expect(stats.routedWidth).toBe(1856);
    expect(stats.sharedWidth).toBe(3712);
    expect(stats.sharedWidth).toBe(2 * stats.routedWidth);
    expect(stats.matricesPerExpert).toBe(2);
    expect(stats.hasGateProj).toBe(false);
    expect(stats.cacheBytes).toBe(0);

    // Verify real emitted 3D geometry: 6 active routed experts × 2 plates = 12, + 122 idle blocks
    const activePlates = moeBf16.solidPickBoxes.filter((b) => b.partId === 'routed-bank');
    const idleBlocks = moeBf16.solidPickBoxes.filter((b) => b.partId === 'inactive-experts');
    expect(activePlates).toHaveLength(6 * 2);
    expect(idleBlocks).toHaveLength(122);

    // In FP8 Selective, MoE weights switch to FP8
    expect(moeFp8.moeStats?.weightsFp8).toBe(true);
    const activeFp8Flags = solidFp8FlagsByPart(moeFp8.solidData, moeFp8.solidPickBoxes, 'routed-bank');
    expect(activeFp8Flags.every((f) => f === 1)).toBe(true);
  });
});

describe('Milestone 2 Cutaway 3: GQA Attention (Bay 06 & Bay 43)', () => {
  it('encodes 32 Q heads (2×16), 2 KV heads (16:1), and live 2×T KV slices across T=1, 8, 16', () => {
    for (const tokens of [1, 8, 16]) {
      const gqa = buildGqaCutaway({
        bayId: 6,
        tokens,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
      });

      const stats = gqa.gqaStats!;
      expect(stats.qHeadCount).toBe(32);
      expect(stats.qGroupCount).toBe(2);
      expect(stats.qHeadsPerGroup).toBe(16);
      expect(stats.kvHeadCount).toBe(2);
      expect(stats.gqaRatio).toBe(16);
      expect(stats.kvTokenSliceCount).toBe(2 * tokens);

      // 32 individual Q head slabs in 3D geometry
      const qSlabs = gqa.solidPickBoxes.filter((b) => b.partId === 'q-proj');
      expect(qSlabs).toHaveLength(32);

      // Each KV head has 2 ghost guide rails + T K-slices + T V-slices = 2 + 2*T pick boxes
      const kv0Boxes = gqa.solidPickBoxes.filter((b) => b.partId === 'kv-head-0');
      const kv1Boxes = gqa.solidPickBoxes.filter((b) => b.partId === 'kv-head-1');
      expect(kv0Boxes).toHaveLength(2 + 2 * tokens);
      expect(kv1Boxes).toHaveLength(2 + 2 * tokens);
    }
  });

  it('keeps Q/K/V/O compute BF16 while switching all KV-cache slices to FP8 in FP8 Selective', () => {
    const gqaFp8 = buildGqaCutaway({
      bayId: 43,
      tokens: 12,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });

    expect(gqaFp8.gqaStats?.computeFp8).toBe(false);
    expect(gqaFp8.gqaStats?.kvCacheFp8).toBe(true);
    expect(gqaFp8.gqaStats?.kvTokenSliceCount).toBe(24);

    // All Q, K, V, O, Attn Scores, RMSNorm compute solids stay BF16 (fp8Flag === 0)
    for (const part of ['rmsnorm', 'q-proj', 'k-proj', 'v-proj', 'attn-scores', 'weighted-v', 'o-proj']) {
      const flags = solidFp8FlagsByPart(gqaFp8.solidData, gqaFp8.solidPickBoxes, part);
      expect(flags.length).toBeGreaterThan(0);
      expect(flags.every((f) => f === 0)).toBe(true);
    }

    // All live KV-cache slices on both KV Head 0 and KV Head 1 switch to FP8 (fp8Flag === 1)
    // (skipping the 2 neutral ghost rails per head whose fp8Flag === 0)
    for (const kvPart of ['kv-head-0', 'kv-head-1']) {
      const flags = solidFp8FlagsByPart(gqaFp8.solidData, gqaFp8.solidPickBoxes, kvPart);
      const fp8SliceCount = flags.filter((f) => f === 1).length;
      expect(fp8SliceCount).toBe(2 * 12); // 12 K slices + 12 V slices per KV head = 24 FP8 slabs
    }
  });
});
