import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  DIFFUSION_14B_COMPONENT_BREAKDOWN,
  DIFFUSION_ATLAS_BUDGETS,
  DIFFUSION_BACKBONE_SEED,
  DIFFUSION_CANONICAL_REFS,
  DIFFUSION_CONTEXT_TIERS,
  DIFFUSION_DISCREPANCY_LEDGER,
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_FAMILY_SPECS,
  DIFFUSION_LFS_INVENTORY,
  DIFFUSION_LORA_SEED,
  DIFFUSION_MINI_BOS_TOKEN_ID,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_END_THINK_TOKEN_ID,
  DIFFUSION_MINI_EOS_TOKEN_ID,
  DIFFUSION_MINI_MASK_TOKEN_ID,
  DIFFUSION_MINI_VOCAB,
  DIFFUSION_PRODUCTION_KV_TABLE,
  DIFFUSION_SPEC_CLARIFICATIONS,
  checkKvCapacityForStep,
  computeSpecCropProvenance,
  verifyDiffusionLfsInventory,
} from '../src/diffusion/spec';
import {
  createMulberry32,
  createXoshiro128ss,
  dualHashHex,
  fnv1a32,
  mixSeed32,
  nextGaussianBoxMuller,
  splitmix32,
} from '../src/diffusion/prng';
import {
  DIFFUSION_MASK_FIXTURES,
  buildArCausalMask,
  buildBlockBidirMask,
  buildCausalCommitVerifyMask,
  buildJointTrainingMask,
  buildVlmQuadraticSsMask,
  verifyMobcStrictOffset,
} from '../src/diffusion/masks';

function rowsFromMask(visible: Uint8Array, qLen: number, kvLen: number): number[][] {
  const rows: number[][] = [];
  for (let q = 0; q < qLen; q++) {
    const row: number[] = [];
    for (let k = 0; k < kvLen; k++) {
      row.push(visible[q * kvLen + k]!);
    }
    rows.push(row);
  }
  return rows;
}

describe('DF1 Slice 1A — Spec, LFS Inventory, Family Matrix & Clarifications', () => {
  it('re-exports locked miniature config, vocab, special tokens, dyadic YaRN inv_freq, and seeds', () => {
    expect(DIFFUSION_MINI_CONFIG.numLayers).toBe(40);
    expect(DIFFUSION_MINI_CONFIG.hiddenSize).toBe(8);
    expect(DIFFUSION_MINI_CONFIG.intermediateSize).toBe(24);
    expect(DIFFUSION_MINI_CONFIG.numAttentionHeads).toBe(32);
    expect(DIFFUSION_MINI_CONFIG.numKeyValueHeads).toBe(8);
    expect(DIFFUSION_MINI_CONFIG.headDim).toBe(8);
    expect(DIFFUSION_MINI_CONFIG.vocabSize).toBe(36);
    expect(DIFFUSION_MINI_CONFIG.loraRank).toBe(4);
    expect(DIFFUSION_MINI_CONFIG.loraAlpha).toBe(16);
    expect(DIFFUSION_MINI_CONFIG.loraScale).toBe(4.0);
    expect(DIFFUSION_MINI_CONFIG.kvCapacity).toBe(32);
    expect(DIFFUSION_MINI_VOCAB).toHaveLength(36);
    expect(DIFFUSION_MINI_VOCAB[DIFFUSION_MINI_MASK_TOKEN_ID]).toBe('[MASK]');
    expect(DIFFUSION_MINI_VOCAB[DIFFUSION_MINI_EOS_TOKEN_ID]).toBe('<|im_end|>');
    expect(DIFFUSION_MINI_VOCAB[DIFFUSION_MINI_BOS_TOKEN_ID]).toBe('<s>');
    expect(DIFFUSION_MINI_VOCAB[DIFFUSION_MINI_END_THINK_TOKEN_ID]).toBe('</think>');
    expect(DIFFUSION_DYADIC_YARN_INV_FREQ).toEqual([1.0, 0.3125, 0.0625, 0.03125]);
    expect(DIFFUSION_BACKBONE_SEED).toBe(0x4e4c4431);
    expect(DIFFUSION_LORA_SEED).toBe(0x4c4f5241);
  });

  it('pins canonical 14B Instruct, GitHub, paper, and explicit out-of-scope sister repo', () => {
    expect(DIFFUSION_CANONICAL_REFS.canonicalModel.repoId).toBe(
      'nvidia/Nemotron-Labs-Diffusion-14B',
    );
    expect(DIFFUSION_CANONICAL_REFS.canonicalModel.revision).toBe(
      'f8c3e2c078e193599b8882d965b1001c456ba738',
    );
    expect(DIFFUSION_CANONICAL_REFS.github.repo).toBe('NVlabs/Nemotron-Labs-Diffusion');
    expect(DIFFUSION_CANONICAL_REFS.github.commit).toBe(
      '2aabcf75ecf70264d4047ff090d842450853f2e4',
    );
    expect(DIFFUSION_CANONICAL_REFS.paper.arxivId).toBe('arXiv:2607.05722v1');
    expect(DIFFUSION_CANONICAL_REFS.outOfScopeSisterRepo.repo).toBe(
      'NVlabs/Nemotron-Labs-Diffusion-Uni',
    );
    expect(DIFFUSION_CANONICAL_REFS.outOfScopeSisterRepo.arxivId).toBe('arXiv:2610.36466');
    expect(DIFFUSION_CANONICAL_REFS.outOfScopeSisterRepo.inScope).toBe(false);
  });

  it('verifies all 13 LFS inventory entries with exact 0 B discrepancy', () => {
    expect(DIFFUSION_LFS_INVENTORY).toHaveLength(13);
    const verification = verifyDiffusionLfsInventory();
    expect(verification.valid).toBe(true);
    expect(verification.entryCount).toBe(13);
    expect(verification.mismatches).toEqual([]);

    for (const entry of DIFFUSION_LFS_INVENTORY) {
      expect(entry.headerBytes + entry.payloadBytes).toBe(entry.totalBytes);
      expect(entry.totalBytes).toBe(entry.hfLfsSizeBytes);
      expect(entry.diffBytes).toBe(0);
    }

    const vlmShards = DIFFUSION_LFS_INVENTORY.filter(
      (e) => e.repoId === 'nvidia/Nemotron-Labs-Diffusion-VLM-8B',
    );
    expect(vlmShards).toHaveLength(4);
    expect(vlmShards.reduce((s, e) => s + e.tensors, 0)).toBe(531);
    expect(vlmShards.reduce((s, e) => s + e.parameters, 0)).toBe(8_918_034_432);
    expect(vlmShards.reduce((s, e) => s + e.totalBytes, 0)).toBe(17_836_137_336);
  });

  it('matches exact 14B component breakdown (363 backbone + 80 LoRA)', () => {
    const { backbone, lora } = DIFFUSION_14B_COMPONENT_BREAKDOWN;
    const sumTensors = backbone.reduce((s, c) => s + c.tensorCount, 0);
    const sumParams = backbone.reduce((s, c) => s + c.parameters, 0);
    const sumBytes = backbone.reduce((s, c) => s + c.bf16Bytes, 0);

    expect(sumTensors).toBe(363);
    expect(sumParams).toBe(13_506_073_600);
    expect(sumBytes).toBe(27_012_147_200);
    expect(DIFFUSION_14B_COMPONENT_BREAKDOWN.backboneTensorCount).toBe(363);
    expect(DIFFUSION_14B_COMPONENT_BREAKDOWN.backboneParameters).toBe(13_506_073_600);
    expect(DIFFUSION_14B_COMPONENT_BREAKDOWN.backbonePayloadBytes).toBe(27_012_147_200);

    expect(lora.tensorCount).toBe(80);
    expect(lora.parameters).toBe(47_185_920);
    expect(lora.f32Bytes).toBe(188_743_680);
  });

  it('contains all 7 family variants, 3 KV rows, and 4 context tiers', () => {
    expect(DIFFUSION_FAMILY_SPECS.map((v) => v.id)).toEqual([
      '14B',
      '14B-Base',
      '8B',
      '8B-Base',
      '3B',
      '3B-Base',
      'VLM-8B',
    ]);
    const canonical = DIFFUSION_FAMILY_SPECS.find((v) => v.id === '14B')!;
    expect(canonical.isCanonical).toBe(true);
    expect(canonical.ropeTheta).toBe(1e9);
    expect(canonical.qProjOutDim).toBe(4096);
    expect(canonical.hiddenSize).toBe(5120);

    const base8b = DIFFUSION_FAMILY_SPECS.find((v) => v.id === '8B-Base')!;
    expect(base8b.yarnFactor).toBe(0.25);
    expect(base8b.maxPositionEmbeddings).toBe(4096);

    const vlm = DIFFUSION_FAMILY_SPECS.find((v) => v.id === 'VLM-8B')!;
    expect(vlm.vocabSize).toBe(131073);
    expect(vlm.vitLayers).toBe(24);
    expect(vlm.dlmLossWeight).toBe(0.5);
    expect(vlm.configEosTokenId).toEqual([11, 2]);

    expect(DIFFUSION_PRODUCTION_KV_TABLE).toHaveLength(3);
    const kv14b = DIFFUSION_PRODUCTION_KV_TABLE[0]!;
    expect(kv14b.variantId).toBe('14B');
    expect(kv14b.kvBytesPerToken).toBe(163_840);
    expect(kv14b.tempBlock32Bytes).toBe(5_242_880);
    expect(kv14b.kvAt4KBytes).toBe(671_088_640);
    expect(kv14b.kvAt16KBytes).toBe(2_684_354_560);
    expect(kv14b.kvAt20KBytes).toBe(3_355_443_200);
    expect(kv14b.kvAt64KBytes).toBe(10_737_418_240);
    expect(kv14b.kvAt256KBytes).toBe(42_949_672_960);

    expect(DIFFUSION_CONTEXT_TIERS.map((t) => t.tier)).toEqual([1, 2, 3, 4]);
    expect(DIFFUSION_CONTEXT_TIERS[0]!.badge).toBe('Verified Training');
    expect(DIFFUSION_CONTEXT_TIERS[1]!.badge).toBe('Harness / Benchmarked Cap');
    expect(DIFFUSION_CONTEXT_TIERS[2]!.badge).toBe('Configured YaRN Ceiling (Unverified)');
    expect(DIFFUSION_CONTEXT_TIERS[3]!.badge).toBe('Analytical Extrapolation');
  });

  it('contains all 24 D1..D24 discrepancy ledger entries and encodes Clarifications #1–#3', () => {
    expect(DIFFUSION_DISCREPANCY_LEDGER).toHaveLength(24);
    expect(DIFFUSION_DISCREPANCY_LEDGER.map((d) => d.id)).toEqual(
      Array.from({ length: 24 }, (_, i) => `D${i + 1}`),
    );

    // Clarification #1: EOS Crop Parity (D19)
    expect(DIFFUSION_DISCREPANCY_LEDGER[18]!.resolution).toContain('browserPostEosCropLen');
    expect(DIFFUSION_DISCREPANCY_LEDGER[18]!.resolution).toContain('upstreamPreEosCropLen');
    expect(computeSpecCropProvenance(12, 4, 1)).toEqual({
      cacheLenBeforeVerify: 12,
      acceptedCount: 4,
      firstEosOffset: 1,
      browserPostEosCropLen: 14,
      upstreamPreEosCropLen: 16,
    });
    expect(computeSpecCropProvenance(12, 4, null)).toEqual({
      cacheLenBeforeVerify: 12,
      acceptedCount: 4,
      firstEosOffset: null,
      browserPostEosCropLen: 16,
      upstreamPreEosCropLen: 16,
    });

    // Clarification #2: kvCapacity Tail-Block Boundary (never silently clamp B down to 1)
    expect(DIFFUSION_DISCREPANCY_LEDGER[15]!.resolution).toContain('capacity_exhausted');
    expect(DIFFUSION_SPEC_CLARIFICATIONS.kvCapacityTailBlockBoundary.rule).toContain('never');
    expect(checkKvCapacityForStep('dlm', 29, 32, 4)).toEqual({
      canProceed: false,
      lifecyclePhase: 'capacity_exhausted',
      remainingSlots: 3,
      requiredSlots: 4,
    });
    expect(checkKvCapacityForStep('linear_spec', 28, 32, 4)).toEqual({
      canProceed: true,
      lifecyclePhase: 'running',
      remainingSlots: 4,
      requiredSlots: 4,
    });
    expect(checkKvCapacityForStep('ar', 31, 32, 4)).toEqual({
      canProceed: true,
      lifecyclePhase: 'running',
      remainingSlots: 1,
      requiredSlots: 1,
    });
    expect(checkKvCapacityForStep('ar', 32, 32, 4)).toEqual({
      canProceed: false,
      lifecyclePhase: 'capacity_exhausted',
      remainingSlots: 0,
      requiredSlots: 1,
    });

    // Clarification #3: Atlas Headroom
    expect(DIFFUSION_ATLAS_BUDGETS.gqaRopeLora).toMatchObject({
      width: 256,
      height: 128,
      format: 'RG32F',
      bytesPerTexel: 8,
      totalBytes: 262_144,
      maxAuthorizedWidth: 256,
      maxAuthorizedHeight: 256,
      maxAuthorizedBytes: 524_288,
    });
    expect(DIFFUSION_ATLAS_BUDGETS.swiglu).toMatchObject({
      width: 64,
      height: 64,
      totalBytes: 32_768,
    });
    expect(DIFFUSION_ATLAS_BUDGETS.maskExplorer).toMatchObject({
      width: 64,
      height: 64,
      totalBytes: 32_768,
    });
  });
});

describe('DF1 Slice 1A — Deterministic PRNGs & Hashers', () => {
  it('xoshiro128** is deterministic, supports clone()/getState() replay, and bounds open-unit draws', () => {
    const g1 = createXoshiro128ss(DIFFUSION_BACKBONE_SEED);
    const g2 = createXoshiro128ss(DIFFUSION_BACKBONE_SEED);

    const seq1 = Array.from({ length: 16 }, () => g1.nextUint32());
    const seq2 = Array.from({ length: 16 }, () => g2.nextUint32());
    expect(seq1).toEqual(seq2);

    const savedState = g1.getState();
    const cloned = g1.clone();
    expect(cloned.getState()).toEqual(savedState);

    for (let i = 0; i < 64; i++) {
      expect(g1.nextFloat64()).toBe(cloned.nextFloat64());
    }

    for (let i = 0; i < 2000; i++) {
      const f32 = g1.nextFloat32();
      const f64 = g1.nextFloat64();
      const open = g1.nextUniformOpenUnit();
      expect(f32).toBeGreaterThanOrEqual(0);
      expect(f32).toBeLessThan(1);
      expect(f64).toBeGreaterThanOrEqual(0);
      expect(f64).toBeLessThan(1);
      expect(open).toBeGreaterThanOrEqual(1e-15);
      expect(open).toBeLessThanOrEqual(1 - 1e-15);
    }
  });

  it('splitmix32, mixSeed32, mulberry32, and Box-Muller are deterministic and finite', () => {
    expect(splitmix32(0)).not.toBe(0);
    expect(splitmix32(42)).toBe(splitmix32(42));
    expect(mixSeed32(0, 1)).not.toBe(mixSeed32(1, 0));
    expect(mixSeed32(0, 1)).not.toBe(mixSeed32(0, 1, 0));

    const m1 = createMulberry32(DIFFUSION_LORA_SEED);
    const m2 = createMulberry32(DIFFUSION_LORA_SEED);
    const gauss1 = Array.from({ length: 32 }, () => nextGaussianBoxMuller(m1));
    const gauss2 = Array.from({ length: 32 }, () => nextGaussianBoxMuller(m2));
    expect(gauss1).toEqual(gauss2);
    for (const v of gauss1) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('fnv1a32 and dualHashHex are sensitive to bytes, order, and length', () => {
    const encoder = new TextEncoder();
    expect(fnv1a32('NLD1')).toBe(fnv1a32(encoder.encode('NLD1')));
    expect(fnv1a32('')).toBe(0x811c9dc5);

    const hA = dualHashHex('nemotron-diffusion-14b');
    const hB = dualHashHex('nemotron-diffusion-14B');
    const hSwap = dualHashHex('nemotron-diffusion-1b4');
    const hLen = dualHashHex('nemotron-diffusion-14b\0');
    const hBytes = dualHashHex(encoder.encode('nemotron-diffusion-14b'));

    expect(hA).toMatch(/^[0-9a-f]{16}$/);
    expect(hA).toBe(hBytes);
    expect(new Set([hA, hB, hSwap, hLen]).size).toBe(4);
  });
});

describe('DF1 Slice 1A — Five-Mode Masks & Three-Coordinate Layout', () => {
  it('Fixture 1: Joint Training 12x12 (L=6, B=2) matches DF0 §6.3 and enforces strict M_OBC Δp >= 1', () => {
    const mask = buildJointTrainingMask(6, 2);
    expect(mask).toEqual(DIFFUSION_MASK_FIXTURES.jointTraining);
    expect(mask.qLen).toBe(12);
    expect(mask.kvLen).toBe(12);
    expect(mask.isReadOnlyReference).toBe(true);
    expect(mask.badge).toBe('Read-Only Analytical Training Mask (No Gradients)');

    expect(Array.from(mask.queryCoords.seqIdx)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(Array.from(mask.queryCoords.ropePositionIds)).toEqual([
      0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
    ]);
    expect(Array.from(mask.queryCoords.cachePositions)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]);

    const expectedRows = [
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // x_t0
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // x_t1
      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], // x_t2
      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], // x_t3
      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], // x_t4
      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], // x_t5
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], // x_00
      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], // x_01
      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], // x_02
      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], // x_03
      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], // x_04
      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], // x_05
    ];
    expect(rowsFromMask(mask.visible, 12, 12)).toEqual(expectedRows);

    const mobcCheck = verifyMobcStrictOffset(mask);
    expect(mobcCheck).toEqual({
      valid: true,
      minDeltaP: 1,
      zeroTwinViolations: 0,
    });

    // Verify D23 cachePositions divergence at SFT L=16384 (without allocating 2L x 2L matrix, verify coordinate rule at L=16384? Wait: buildJointTrainingMask allocates 4L^2 = 1GB at 16384! Test at L=32 or verify the formula directly, and check smaller L=16!)
    // At L=16, B=4: x_t has cachePositions 0..15 (<16 origMaxPos), x_0 has cachePositions 16..31 (>=16)!
    const d23Mask = buildJointTrainingMask(16, 4);
    expect(d23Mask.queryCoords.ropePositionIds[16]).toBe(0);
    expect(d23Mask.queryCoords.cachePositions[16]).toBe(16);
    expect(d23Mask.queryCoords.ropePositionIds[31]).toBe(15);
    expect(d23Mask.queryCoords.cachePositions[31]).toBe(31);
    expect(verifyMobcStrictOffset(d23Mask)).toEqual({
      valid: true,
      minDeltaP: 1,
      zeroTwinViolations: 0,
    });

    // Negative test: verifyMobcStrictOffset catches zero-twin leakage
    const corruptedVisible = new Uint8Array(mask.visible);
    corruptedVisible[0 * 12 + 6] = 1; // x_t0 (pos=0) illegally attends to x_00 (pos=0)
    const corruptedRegions = [...mask.regions!];
    corruptedRegions[0 * 12 + 6] = 'M_OBC';
    const corruptedCheck = verifyMobcStrictOffset({
      ...mask,
      visible: corruptedVisible,
      regions: corruptedRegions,
    });
    expect(corruptedCheck.valid).toBe(false);
    expect(corruptedCheck.minDeltaP).toBe(0);
    expect(corruptedCheck.zeroTwinViolations).toBe(1);
  });

  it('Fixtures 2A, 2B, 3, 4: AR Prefill (4x4), AR Decode (1x5), Block-Bidir (4x8), Causal Verify (4x8)', () => {
    const arPrefill = buildArCausalMask(0, 4);
    expect(arPrefill).toEqual(DIFFUSION_MASK_FIXTURES.arPrefill);
    expect(rowsFromMask(arPrefill.visible, 4, 4)).toEqual([
      [1, 0, 0, 0],
      [1, 1, 0, 0],
      [1, 1, 1, 0],
      [1, 1, 1, 1],
    ]);
    expect(Array.from(arPrefill.queryCoords.ropePositionIds)).toEqual([0, 1, 2, 3]);
    expect(Array.from(arPrefill.queryCoords.cachePositions)).toEqual([0, 1, 2, 3]);

    const arDecode = buildArCausalMask(4, 1);
    expect(arDecode).toEqual(DIFFUSION_MASK_FIXTURES.arDecode);
    expect(rowsFromMask(arDecode.visible, 1, 5)).toEqual([[1, 1, 1, 1, 1]]);
    expect(Array.from(arDecode.queryCoords.seqIdx)).toEqual([0]);
    expect(Array.from(arDecode.queryCoords.ropePositionIds)).toEqual([4]);
    expect(Array.from(arDecode.queryCoords.cachePositions)).toEqual([4]);
    expect(Array.from(arDecode.kvRopePositionIds)).toEqual([0, 1, 2, 3, 4]);

    const bidir = buildBlockBidirMask(4, 4);
    expect(bidir).toEqual(DIFFUSION_MASK_FIXTURES.blockBidir);
    expect(rowsFromMask(bidir.visible, 4, 8)).toEqual([
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
    ]);
    expect(Array.from(bidir.queryCoords.ropePositionIds)).toEqual([4, 5, 6, 7]);
    expect(Array.from(bidir.kvRopePositionIds)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);

    const verify = buildCausalCommitVerifyMask(4, 4);
    expect(verify).toEqual(DIFFUSION_MASK_FIXTURES.causalCommitVerify);
    expect(rowsFromMask(verify.visible, 4, 8)).toEqual([
      [1, 1, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 0],
      [1, 1, 1, 1, 1, 1, 1, 1],
    ]);
    expect(Array.from(verify.queryCoords.ropePositionIds)).toEqual([4, 5, 6, 7]);
    expect(Array.from(verify.queryCoords.cachePositions)).toEqual([4, 5, 6, 7]);
  });

  it('Fixture 5: VLM-8B Cached Quadratic SS 6x8 (P=2, k=2) matches DF0 §6.3 exactly', () => {
    const vlm = buildVlmQuadraticSsMask(2, 2);
    expect(vlm).toEqual(DIFFUSION_MASK_FIXTURES.vlmQuadraticSs);
    expect(vlm.qLen).toBe(6);
    expect(vlm.kvLen).toBe(8);
    expect(vlm.isReadOnlyReference).toBe(true);
    expect(vlm.badge).toBe('Quadratic SS · VLM/Paper Ref');

    expect(Array.from(vlm.queryCoords.seqIdx)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(vlm.queryCoords.ropePositionIds)).toEqual([2, 3, 4, 3, 4, 5]);
    expect(Array.from(vlm.queryCoords.cachePositions)).toEqual([2, 3, 4, 5, 6, 7]);
    expect(Array.from(vlm.kvRopePositionIds)).toEqual([2, 3, 4, 3, 4, 5, 0, 1]);
    expect(Array.from(vlm.kvCachePositions)).toEqual([2, 3, 4, 5, 6, 7, 0, 1]);

    expect(rowsFromMask(vlm.visible, 6, 8)).toEqual([
      [1, 0, 0, 0, 0, 0, 1, 1], // d0   (p=2)
      [1, 1, 1, 0, 0, 0, 1, 1], // m0.1 (p=3)
      [1, 1, 1, 0, 0, 0, 1, 1], // m0.2 (p=4)
      [1, 0, 0, 1, 0, 0, 1, 1], // d1   (p=3)
      [1, 0, 0, 1, 1, 1, 1, 1], // m1.1 (p=4)
      [1, 0, 0, 1, 1, 1, 1, 1], // m1.2 (p=5)
    ]);
  });

  it('enforces clean-room isolation (no Math.random, DOM/WebGL, or src/twotower imports)', () => {
    const ownedSourceFiles = [
      'src/diffusion/spec.ts',
      'src/diffusion/prng.ts',
      'src/diffusion/masks.ts',
    ];
    for (const relPath of ownedSourceFiles) {
      const raw = readFileSync(resolve(process.cwd(), relPath), 'utf8');
      const codeOnly = raw
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/.*$/gm, '');
      expect(codeOnly).not.toMatch(/Math\.random\s*\(/);
      expect(codeOnly).not.toMatch(/from\s+['"][^'"]*twotower/i);
      expect(codeOnly).not.toMatch(/\b(?:document|window)\s*\./);
      expect(codeOnly).not.toMatch(/\bWebGL2?RenderingContext\b|getContext\s*\(\s*['"]webgl/);
    }
  });
});
