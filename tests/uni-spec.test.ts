import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  createMulberry32,
  createXoshiro128ss,
  dualHashHex,
  fnv1a32,
  mixSeed32,
  nextGaussianBoxMuller,
  splitmix32,
} from '../src/uni/prng';
import {
  formatSiIecBytes,
  UNI_ATLAS_BUDGETS,
  UNI_AUXILIARY_LFS_ARTIFACTS,
  UNI_BACKBONE_SEED,
  UNI_CANONICAL_REFS,
  UNI_CHECKPOINT_SHARDS,
  UNI_COMPONENT_BREAKDOWN,
  UNI_CONTEXT_TIERS,
  UNI_DISCREPANCY_LEDGER,
  UNI_DYADIC_YARN_INV_FREQ,
  UNI_EXECUTION_VIEWS,
  UNI_GAIN_PRESETS,
  UNI_INDEX_TOTAL_SIZE_BYTES,
  UNI_INTERLEAVED_FULL_MULTIMODAL_PARAMS,
  UNI_MAX_ATLAS_BYTES,
  UNI_MINI_COMPONENT_FOOTPRINT,
  UNI_MINI_CONFIG,
  UNI_MINI_CORE_FLOAT_COUNT,
  UNI_MINI_CORE_TENSOR_COUNT,
  UNI_MINI_CORE_WEIGHT_BYTES,
  UNI_MINI_FULL_FLOAT_COUNT,
  UNI_MINI_FULL_TENSOR_COUNT,
  UNI_MINI_FULL_WEIGHT_BYTES,
  UNI_MINI_TOTAL_RESIDENT_BYTES,
  UNI_MINI_TREE_BUFFER_BYTES,
  UNI_MINI_VOCAB,
  UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16,
  UNI_PRODUCTION_SINGLE_LAYER_BRANCH_PARAMS,
  UNI_PROJECTION_REVERSAL_BADGE,
  UNI_SGLANG_PROVENANCE,
  UNI_SGLANG_VLM_RESIDENT_BYTES,
  UNI_TOTAL_CHECKPOINT_ELEMENTS,
  UNI_TOTAL_CHECKPOINT_TENSORS,
  UNI_TOTAL_FILE_BYTES,
  UNI_TOTAL_HEADER_FRAMING_BYTES,
  UNI_VISUAL_QUALITY_HONESTY_BADGE,
  UNI_WASTED_SPLIT_U_DECODER_PARAMS,
  UNI_WASTED_SPLIT_U_TOTAL_PARAMS,
  verifyUniLfsReconciliation,
} from '../src/uni/spec';

const ROOT_DIR = resolve(process.cwd());

describe('UNI1 Slice 1A — Canonical References, 0 B LFS Reconciliation & Component Breakdown', () => {
  it('pins canonical arXiv, HF checkpoint, GitHub commits, lineage, and 3-tier license policy', () => {
    expect(UNI_CANONICAL_REFS.primaryPaper.id).toBe('arXiv:2610.36466v1');
    expect(UNI_CANONICAL_REFS.primaryPaper.license).toBe('CC BY 4.0');
    expect(UNI_CANONICAL_REFS.checkpoint.repoId).toBe('nvidia/NLD-Uni');
    expect(UNI_CANONICAL_REFS.checkpoint.commit).toBe(
      '5db755a2eb0ab84654e3dcf80c7d0245bfe36a50',
    );
    expect(UNI_CANONICAL_REFS.checkpoint.releaseManifestTag).toBe(
      'generation-only-step-10000',
    );
    expect(UNI_CANONICAL_REFS.githubRepo.repoId).toBe(
      'NVlabs/Nemotron-Labs-Diffusion-Uni',
    );
    expect(UNI_CANONICAL_REFS.githubRepo.commit).toBe(
      '16916949402ca738dcc74a7a37e0c43facafe347',
    );

    const lineageIds = UNI_CANONICAL_REFS.lineage.map((l) => l.id);
    expect(lineageIds).toEqual(
      expect.arrayContaining([
        'nld',
        'lavida_o',
        'nld_image',
        'radio_h',
        'emu3_5_ibq',
        'ministral_3',
        'qwen_image',
      ]),
    );

    expect(
      UNI_CANONICAL_REFS.licenseBreakdown.implementationCode.adobeConfidentialFiles,
    ).toEqual([
      'sampling.py',
      'positions.py',
      'pmj.py',
      'schedules.py',
      'image_utils.py',
      'utils.py',
    ]);
    expect(
      UNI_CANONICAL_REFS.licenseBreakdown.implementationCode.nvidiaProprietaryFiles,
    ).toEqual(['radio_cls.py', 'radio_patch.py']);
  });

  it('reconciles all 8 safetensors shards to exact 0 B discrepancy against index.json and component table', () => {
    expect(UNI_CHECKPOINT_SHARDS).toHaveLength(8);
    expect(UNI_COMPONENT_BREAKDOWN).toHaveLength(14);

    for (let i = 0; i < UNI_CHECKPOINT_SHARDS.length; i++) {
      const shard = UNI_CHECKPOINT_SHARDS[i]!;
      expect(shard.shardIndex).toBe(i + 1);
      expect(shard.totalShards).toBe(8);
      expect(shard.headerFramingBytes).toBe(8 + shard.headerJsonBytes);
      expect(shard.headerFramingBytes + shard.payloadBytes).toBe(shard.totalFileBytes);
      expect(shard.diffBytes).toBe(0);
      expect(shard.sha256Prefix).toMatch(/^[0-9a-f]{12}$/);
    }

    // Verify per-component dtype byte math (BF16 = 2 B/el; vision_tower adds 3 I64 = +18 B over BF16)
    for (const comp of UNI_COMPONENT_BREAKDOWN) {
      if (comp.dtype === 'BF16') {
        expect(comp.payloadBytes).toBe(comp.elementCount * 2);
      } else {
        expect(comp.id).toBe('vision_tower');
        expect(comp.payloadBytes).toBe((comp.elementCount - 3) * 2 + 3 * 8);
      }
    }

    const deadNewline = UNI_COMPONENT_BREAKDOWN.find((c) => c.id === 'image_newline');
    expect(deadNewline?.elementCount).toBe(4_096);
    expect(deadNewline?.unusedInForward).toBe(true);

    const summary = verifyUniLfsReconciliation();
    expect(summary.shardCount).toBe(8);
    expect(summary.totalTensors).toBe(UNI_TOTAL_CHECKPOINT_TENSORS);
    expect(summary.totalTensors).toBe(1_557);
    expect(summary.totalElements).toBe(UNI_TOTAL_CHECKPOINT_ELEMENTS);
    expect(summary.totalElements).toBe(18_267_552_262);
    expect(summary.totalHeaderFramingBytes).toBe(UNI_TOTAL_HEADER_FRAMING_BYTES);
    expect(summary.totalHeaderFramingBytes).toBe(199_680);
    expect(summary.totalPayloadBytes).toBe(UNI_INDEX_TOTAL_SIZE_BYTES);
    expect(summary.totalPayloadBytes).toBe(36_535_104_542);
    expect(summary.totalFileBytes).toBe(UNI_TOTAL_FILE_BYTES);
    expect(summary.totalFileBytes).toBe(36_535_304_222);
    expect(summary.payloadDiffBytes).toBe(0);
    expect(summary.fileDiffBytes).toBe(0);
    expect(summary.allZeroDiff).toBe(true);
  });

  it('tracks auxiliary LFS artifacts as non-redistributed', () => {
    expect(UNI_AUXILIARY_LFS_ARTIFACTS).toHaveLength(2);
    const pcaTree = UNI_AUXILIARY_LFS_ARTIFACTS.find((a) => a.file === 'pca_tree.pt');
    const tokenizer = UNI_AUXILIARY_LFS_ARTIFACTS.find(
      (a) => a.file === 'tokenizer.json',
    );
    expect(pcaTree).toMatchObject({
      sizeBytes: 1_052_237,
      sha256Prefix: '4b7ee94fbb10',
      redistributed: false,
    });
    expect(tokenizer).toMatchObject({
      sizeBytes: 17_268_792,
      sha256Prefix: 'f9ad3482369b',
      redistributed: false,
    });
  });
});

describe('UNI1 Slice 1A — Execution Views, Active vs Eager HF Math & Context Honesty', () => {
  it('verifies exact component-level arithmetic for all 6 execution views', () => {
    expect(UNI_EXECUTION_VIEWS).toHaveLength(6);

    const byCompId = new Map(UNI_COMPONENT_BREAKDOWN.map((c) => [c.id, c.elementCount]));
    const getComp = (id: string): number => {
      const v = byCompId.get(id);
      if (v === undefined) throw new Error(`Missing component ${id}`);
      return v;
    };

    const embed = getComp('embed_tokens');
    const uJoint = getComp('u_joint_layers');
    const uSplit = getComp('u_split_layers');
    const uNormAndHead = getComp('u_norm_and_head');
    const gJoint = getComp('g_joint_layers');
    const gSplit = getComp('g_split_layers');
    const gIo = getComp('g_norm_embed_predictor');
    const codec = getComp('packed_image_codec');
    const visionActive = getComp('vision_tower') - 3; // exclude 3 dead I64 summary_idxs
    const projector = getComp('mm_projector');
    const vqEnc = getComp('vqvae_encoder');
    const vqCodebook = getComp('vqvae_codebook');
    const vqDec = getComp('vqvae_decoder');
    const encoderNorm = 4_096;

    // Single-branch layer arithmetic: 41,943,040 + 176,160,768 + 8,192 = 218,112,000
    expect(UNI_PRODUCTION_SINGLE_LAYER_BRANCH_PARAMS).toBe(
      41_943_040 + 176_160_768 + 8_192,
    );
    expect(uJoint).toBe(18 * UNI_PRODUCTION_SINGLE_LAYER_BRANCH_PARAMS);
    expect(uSplit).toBe(16 * UNI_PRODUCTION_SINGLE_LAYER_BRANCH_PARAMS);
    expect(UNI_WASTED_SPLIT_U_DECODER_PARAMS).toBe(uSplit);
    expect(UNI_WASTED_SPLIT_U_TOTAL_PARAMS).toBe(uSplit + encoderNorm);

    const viewMap = new Map(UNI_EXECUTION_VIEWS.map((v) => [v.id, v]));

    // 1. Text-Only Understanding
    const textUnd = viewMap.get('text_understanding')!;
    expect(textUnd.decoderBlockParams).toBe(uJoint + uSplit);
    expect(textUnd.activeCausalGraphParams).toBe(embed + uJoint + uSplit + uNormAndHead);
    expect(textUnd.activeCausalGraphParams).toBe(8_497_983_488);
    expect(textUnd.cfgBranches).toBe(1);
    expect(textUnd.kvBytesPerTokenBf16).toBe(139_264);
    expect(textUnd.residentSglangBytes).toBe(UNI_SGLANG_VLM_RESIDENT_BYTES);

    // 2. Visual Understanding & 3. Object Grounding
    for (const id of ['visual_understanding', 'object_grounding'] as const) {
      const view = viewMap.get(id)!;
      expect(view.decoderBlockParams).toBe(uJoint + uSplit);
      expect(view.activeCausalGraphParams).toBe(
        textUnd.activeCausalGraphParams + visionActive + projector,
      );
      expect(view.activeCausalGraphParams).toBe(9_187_382_784);
      expect(view.residentSglangBytes).toBe(view.activeCausalGraphParams * 2);
      expect(view.kvBytesPerTokenBf16).toBe(139_264);
    }

    // 4. Text-to-Image (T2I): exact active + wasted 16U + encoder.norm = eager HF
    const t2i = viewMap.get('text_to_image')!;
    expect(t2i.decoderBlockParams).toBe(uJoint + gJoint + gSplit);
    expect(t2i.activeCausalGraphParams).toBe(
      embed + uJoint + gJoint + gSplit + gIo + codec + vqCodebook + vqDec,
    );
    expect(t2i.activeCausalGraphParams).toBe(13_362_667_523);
    expect(t2i.eagerHfExecutedParams).toBe(16_852_463_619);
    expect(t2i.eagerHfExecutedParams - t2i.activeCausalGraphParams).toBe(
      UNI_WASTED_SPLIT_U_TOTAL_PARAMS,
    );
    expect(t2i.wastedSplitUParams).toBe(3_489_796_096);
    expect(t2i.cfgBranches).toBe(2);
    expect(t2i.kvBytesPerTokenBf16).toBe(278_528);
    expect(t2i.residentSglangBytes).toBeNull();

    // 5. Image Editing: exact active + wasted 16U + encoder.norm = eager HF
    const edit = viewMap.get('image_edit')!;
    expect(edit.decoderBlockParams).toBe(uJoint + gJoint + gSplit);
    expect(edit.activeCausalGraphParams).toBe(
      t2i.activeCausalGraphParams + visionActive + projector + vqEnc,
    );
    expect(edit.activeCausalGraphParams).toBe(14_236_666_371);
    expect(edit.eagerHfExecutedParams).toBe(17_726_462_467);
    expect(edit.eagerHfExecutedParams - edit.activeCausalGraphParams).toBe(
      UNI_WASTED_SPLIT_U_TOTAL_PARAMS,
    );
    expect(edit.wastedSplitUParams).toBe(3_489_796_096);
    expect(edit.cfgBranches).toBe(3);
    expect(edit.kvBytesPerTokenBf16).toBe(417_792);
    expect(edit.residentSglangBytes).toBeNull();

    // 6. Interleaved Multimodal
    const interleaved = viewMap.get('interleaved_mixed')!;
    expect(interleaved.decoderBlockParams).toBe(uJoint + uSplit + gJoint + gSplit);
    expect(interleaved.activeCausalGraphParams).toBe(
      embed + uJoint + uSplit + uNormAndHead + gJoint + gSplit + gIo + codec,
    );
    expect(interleaved.activeCausalGraphParams).toBe(17_122_811_904);
    expect(
      interleaved.activeCausalGraphParams +
        visionActive +
        projector +
        vqEnc +
        vqCodebook +
        vqDec,
    ).toBe(UNI_INTERLEAVED_FULL_MULTIMODAL_PARAMS);
    expect(UNI_INTERLEAVED_FULL_MULTIMODAL_PARAMS + 4_099).toBe(
      UNI_TOTAL_CHECKPOINT_ELEMENTS,
    );
    expect(interleaved.releasedEndToEndLoop).toBe(false);
    expect(interleaved.badge).toBe(
      'Kernel-Supported · No Released Multi-Turn Outer Loop',
    );

    // Universal invariants across all 6 views
    for (const view of UNI_EXECUTION_VIEWS) {
      expect(view.eagerHfExecutedParams - view.activeCausalGraphParams).toBe(
        view.wastedSplitUParams,
      );
      expect(view.kvBytesPerTokenBf16).toBe(
        UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16 * view.cfgBranches,
      );
      expect(view.residentHfBytes).toBeGreaterThanOrEqual(UNI_INDEX_TOTAL_SIZE_BYTES);
    }
  });

  it('exposes the 4-tier U3 context honesty table and SGLang provenance', () => {
    expect(UNI_CONTEXT_TIERS.map((t) => t.tokens)).toEqual([
      8_192,
      16_384,
      131_072,
      262_144,
    ]);
    expect(UNI_CONTEXT_TIERS.map((t) => t.tier)).toEqual([1, 2, 3, 4]);

    expect(UNI_SGLANG_PROVENANCE.loadedParams).toBe(9_187_382_784);
    expect(UNI_SGLANG_PROVENANCE.residentBytes).toBe(18_374_765_568);
    expect(UNI_SGLANG_PROVENANCE.knownUpstreamBugs).toHaveLength(2);
    expect(UNI_PROJECTION_REVERSAL_BADGE.label).toContain('8 -> 64 Q / 16 KV');
    expect(UNI_VISUAL_QUALITY_HONESTY_BADGE.label).toContain('16×16');
  });
});

describe('UNI1 Slice 1A — Complete U1–U30 Discrepancy & Browser-Guard Ledger', () => {
  it('contains exactly 30 sequential unique entries U1..U30 with all 6 browser guards flagged', () => {
    expect(UNI_DISCREPANCY_LEDGER).toHaveLength(30);

    const seenIds = new Set<string>();
    const guardIds: string[] = [];

    for (let i = 0; i < UNI_DISCREPANCY_LEDGER.length; i++) {
      const entry = UNI_DISCREPANCY_LEDGER[i]!;
      const expectedNum = i + 1;
      const expectedId = `U${expectedNum}`;

      expect(entry.numericId).toBe(expectedNum);
      expect(entry.id).toBe(expectedId);
      expect(seenIds.has(entry.id)).toBe(false);
      seenIds.add(entry.id);

      expect(entry.topic.trim().length).toBeGreaterThan(5);
      expect(entry.paperOrConfigClaim.trim().length).toBeGreaterThan(10);
      expect(entry.verifiedGroundTruth.trim().length).toBeGreaterThan(10);
      expect(entry.browserResolution.trim().length).toBeGreaterThan(10);
      expect(['VERIFIED', 'DERIVED']).toContain(entry.epistemicStatus);

      if (entry.isBrowserGuard) {
        guardIds.push(entry.id);
        expect(entry.category).toBe('browser_guard');
      } else {
        expect(entry.category).not.toBe('browser_guard');
      }
    }

    expect(guardIds).toEqual(['U20', 'U21', 'U22', 'U23', 'U25', 'U30']);
  });
});

describe('UNI1 Slice 1A — Miniature Footprint, Atlas Budgets, Re-exports & Byte Formatting', () => {
  it('reconciles exact miniature 766-tensor / 191,235-float / 768,008 B footprint and UNI1 core subtotal', () => {
    let totalTensors = 0;
    let totalFloats = 0;
    let totalBytes = 0;
    let weightBytes = 0;
    let treeBytes = 0;
    let uni1Tensors = 0;
    let uni1Floats = 0;
    let uni1Bytes = 0;

    for (const row of UNI_MINI_COMPONENT_FOOTPRINT) {
      totalTensors += row.tensorCount;
      totalFloats += row.floatCount;
      totalBytes += row.bytes;

      if (row.milestone === 'UNI3') {
        treeBytes += row.bytes;
      } else {
        expect(row.bytes).toBe(row.floatCount * 4);
        weightBytes += row.bytes;
      }

      if (row.milestone === 'UNI1') {
        uni1Tensors += row.tensorCount;
        uni1Floats += row.floatCount;
        uni1Bytes += row.bytes;
      }
    }

    expect(uni1Tensors).toBe(UNI_MINI_CORE_TENSOR_COUNT);
    expect(uni1Tensors).toBe(618);
    expect(uni1Floats).toBe(UNI_MINI_CORE_FLOAT_COUNT);
    expect(uni1Floats).toBe(139_024);
    expect(uni1Bytes).toBe(UNI_MINI_CORE_WEIGHT_BYTES);
    expect(uni1Bytes).toBe(556_096);

    expect(totalTensors).toBe(UNI_MINI_FULL_TENSOR_COUNT);
    expect(totalTensors).toBe(766);
    expect(totalFloats).toBe(UNI_MINI_FULL_FLOAT_COUNT);
    expect(totalFloats).toBe(191_235);
    expect(weightBytes).toBe(UNI_MINI_FULL_WEIGHT_BYTES);
    expect(weightBytes).toBe(764_940);
    expect(treeBytes).toBe(UNI_MINI_TREE_BUFFER_BYTES);
    expect(treeBytes).toBe(3_068);
    expect(totalBytes).toBe(UNI_MINI_TOTAL_RESIDENT_BYTES);
    expect(totalBytes).toBe(768_008);

    const miniNewline = UNI_MINI_COMPONENT_FOOTPRINT.find(
      (r) => r.id === 'mini_image_newline',
    );
    expect(miniNewline?.unusedInForward).toBe(true);
  });

  it('enforces <= 512 KiB scalar/mask-only RG32F atlas budgets across all 5 atlas views', () => {
    expect(UNI_ATLAS_BUDGETS).toHaveLength(5);
    for (const atlas of UNI_ATLAS_BUDGETS) {
      expect(atlas.bytes).toBe(atlas.width * atlas.height * 8);
      expect(atlas.bytes).toBeLessThanOrEqual(UNI_MAX_ATLAS_BYTES);
      expect(atlas.scalarMaskOnly).toBe(true);
      expect(atlas.rectCount).toBeGreaterThan(0);
    }
  });

  it('re-exports pinned constants from types.ts and formats dual SI/IEC byte strings cleanly', () => {
    expect(UNI_BACKBONE_SEED).toBe(0x4e4c5531);
    expect(UNI_DYADIC_YARN_INV_FREQ).toEqual([1.0, 0.28125, 0.03125, 0.015625]);
    expect(UNI_MINI_VOCAB).toHaveLength(64);
    expect(UNI_MINI_CONFIG.numLayers).toBe(34);
    expect(UNI_MINI_CONFIG.dualTowerLayers).toBe(18);
    expect(UNI_GAIN_PRESETS.mini_default.combinedGenerationScale).toBe(0.5);
    expect(UNI_GAIN_PRESETS.prod_exact.radioProjectorMultiply).toBe(0.01);

    expect(formatSiIecBytes(0)).toBe('0 B');
    expect(formatSiIecBytes(512)).toBe('512 B');
    expect(formatSiIecBytes(139_264)).toBe('139.26 KB (136 KiB)');
    expect(formatSiIecBytes(524_288)).toBe('524.29 KB (512 KiB)');
    expect(formatSiIecBytes(768_008)).toBe('768.01 KB (750.01 KiB)');
    expect(formatSiIecBytes(268_451_840)).toBe('268.45 MB (256.02 MiB)');
    expect(formatSiIecBytes(36_535_104_542)).toBe('36.535 GB (34.026 GiB)');
    expect(() => formatSiIecBytes(-1)).toThrow(RangeError);
    expect(() => formatSiIecBytes(Number.NaN)).toThrow(RangeError);
  });
});

describe('UNI1 Slice 1A — Deterministic PRNG & Hashing Primitives', () => {
  it('splitmix32 is deterministic uint32 with distinct outputs across adjacent seeds', () => {
    const a = splitmix32(0);
    const b = splitmix32(1);
    const c = splitmix32(UNI_BACKBONE_SEED);
    expect(a).toBe(splitmix32(0));
    expect(a).not.toBe(b);
    expect(b).not.toBe(c);
    for (const v of [a, b, c, splitmix32(-1)]) {
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(0xffffffff);
    }
  });

  it('createXoshiro128ss is reproducible, cloneable, and strictly open-unit in [1e-15, 1 - 1e-15]', () => {
    const g1 = createXoshiro128ss(UNI_BACKBONE_SEED);
    const g2 = createXoshiro128ss(UNI_BACKBONE_SEED);

    for (let i = 0; i < 64; i++) {
      expect(g1.nextUint32()).toBe(g2.nextUint32());
    }

    const cloned = g1.clone();
    expect(cloned.getState()).toEqual(g1.getState());

    for (let i = 0; i < 32; i++) {
      const f32a = g1.nextFloat32();
      const f32b = cloned.nextFloat32();
      expect(f32a).toBe(f32b);
      expect(f32a).toBeGreaterThanOrEqual(0);
      expect(f32a).toBeLessThan(1);

      const f64a = g1.nextFloat64();
      const f64b = cloned.nextFloat64();
      expect(f64a).toBe(f64b);
      expect(f64a).toBeGreaterThanOrEqual(0);
      expect(f64a).toBeLessThan(1);
    }

    // Mutating clone must not affect original
    const stateBefore = g1.getState();
    cloned.nextUint32();
    expect(g1.getState()).toEqual(stateBefore);

    // Open-unit samples remain strictly in [1e-15, 1 - 1e-15] and yield finite Gumbel noise (U22)
    for (let i = 0; i < 2_000; i++) {
      const u = g1.nextUniformOpenUnit();
      expect(u).toBeGreaterThanOrEqual(1e-15);
      expect(u).toBeLessThanOrEqual(1 - 1e-15);
      const gumbel = -Math.log(-Math.log(u));
      expect(Number.isFinite(gumbel)).toBe(true);
    }
  });

  it('fnv1a32 matches standard FNV-1a vectors and agrees between string and Uint8Array', () => {
    expect(fnv1a32('')).toBe(0x811c9dc5);
    expect(fnv1a32('a')).toBe(0xe40c292c);
    expect(fnv1a32('foobar')).toBe(0xbf9cf968);

    const text = 'nvidia/NLD-Uni@5db755a2';
    const bytes = new TextEncoder().encode(text);
    expect(fnv1a32(text)).toBe(fnv1a32(bytes));
  });

  it('mixSeed32 is deterministic, order-sensitive, and length-sensitive', () => {
    expect(mixSeed32(1, 2, 3)).toBe(mixSeed32(1, 2, 3));
    expect(mixSeed32(1, 2)).not.toBe(mixSeed32(2, 1));
    expect(mixSeed32(1, 2, 3)).not.toBe(mixSeed32(1, 3, 2));
    expect(mixSeed32(42)).not.toBe(mixSeed32(42, 0));
  });

  it('dualHashHex emits 16 lowercase hex chars sensitive to single-byte edits and order', () => {
    const hEmpty = dualHashHex('');
    const hA = dualHashHex('abc');
    const hB = dualHashHex('abd');
    const hRev = dualHashHex('cba');
    const hBytes = dualHashHex(new TextEncoder().encode('abc'));

    for (const h of [hEmpty, hA, hB, hRev]) {
      expect(h).toMatch(/^[0-9a-f]{16}$/);
    }
    expect(hA).toBe(hBytes);
    expect(new Set([hEmpty, hA, hB, hRev]).size).toBe(4);
  });

  it('createMulberry32 and nextGaussianBoxMuller produce deterministic finite distributions', () => {
    const m1 = createMulberry32(42);
    const m2 = createMulberry32(42);

    let sum = 0;
    let sumSq = 0;
    const n = 4_000;

    for (let i = 0; i < n; i++) {
      const v1 = m1();
      const v2 = m2();
      expect(v1).toBe(v2);
      expect(v1).toBeGreaterThanOrEqual(0);
      expect(v1).toBeLessThan(1);
    }

    const rng = createXoshiro128ss(12345);
    for (let i = 0; i < n; i++) {
      const z = nextGaussianBoxMuller(() => rng.nextUniformOpenUnit());
      expect(Number.isFinite(z)).toBe(true);
      sum += z;
      sumSq += z * z;
    }

    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.08);
    expect(variance).toBeGreaterThan(0.85);
    expect(variance).toBeLessThan(1.15);

    // Boundary safety when caller returns 0.0 or 1.0
    const boundaryZero = nextGaussianBoxMuller(() => 0.0);
    const boundaryOne = nextGaussianBoxMuller(() => 1.0);
    expect(Number.isFinite(boundaryZero)).toBe(true);
    expect(Number.isFinite(boundaryOne)).toBe(true);
  });
});

describe('UNI1 Slice 1A — Static Isolation & Determinism Guardrails', () => {
  const checkedFiles = [
    'src/uni/types.ts',
    'src/uni/prng.ts',
    'src/uni/spec.ts',
  ] as const;

  it('contains zero Math.random() calls and zero forbidden cross-app or controls.ts imports', () => {
    for (const relPath of checkedFiles) {
      const source = readFileSync(resolve(ROOT_DIR, relPath), 'utf8');

      // Strip comments before checking for executable Math.random() calls and imports
      const codeOnly = source
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/.*$/gm, '');

      expect(codeOnly).not.toMatch(/\bMath\.random\s*\(/);
      expect(codeOnly).not.toMatch(/from\s+['"][^'"]*twotower/);
      expect(codeOnly).not.toMatch(/from\s+['"][^'"]*diffusion/);
      expect(codeOnly).not.toMatch(/from\s+['"][^'"]*ui\/controls/);
    }
  });
});
