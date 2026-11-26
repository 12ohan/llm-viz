import { describe, expect, it } from 'vitest';

import {
  createDefaultDiffusionModel,
  createDiffusionSession,
  DIFFUSION_DISCREPANCY_LEDGER,
  DIFFUSION_FAMILY_SPECS,
  DIFFUSION_KV_SLAB_BYTES_PER_TOKEN,
  DIFFUSION_LFS_INVENTORY,
  type DiffusionExecutionMode,
  type DiffusionSessionMemoryReport,
} from '../src/diffusion';
import {
  buildDiffusionArchReadouts,
  buildDiffusionContextKvScalingRows,
  buildDiffusionMemoryBoundaryRows,
  buildDiffusionNfeTpfReadout,
  DIFFUSION_BACKBONE_MINI_BYTES,
  DIFFUSION_CONTEXT_SCALING_PRESETS,
  DIFFUSION_LORA_MINI_BYTES,
  filterDiffusionDiscrepancyLedger,
  findInvalidUiNumericStrings,
  formatDualSiIecBytes,
  formatParameterCount,
} from '../src/diffusion/archReadouts';

describe('DF4 Headless Architecture Readouts, Context KV Scaling & D1–D24 Ledger', () => {
  it('formats dual SI/IEC bytes accurately for production, mini, and edge-case inputs', () => {
    const prodBackbone = formatDualSiIecBytes(27_012_147_200);
    expect(prodBackbone.bytes).toBe(27_012_147_200);
    expect(prodBackbone.iecLabel).toBe('25.16 GiB');
    expect(prodBackbone.siLabel).toBe('27.01 GB');
    expect(prodBackbone.dualLabel).toBe('25.16 GiB (27.01 GB)');

    const prodLora = formatDualSiIecBytes(188_743_680);
    expect(prodLora.iecLabel).toBe('180.00 MiB');
    expect(prodLora.siLabel).toBe('188.74 MB');
    expect(prodLora.dualLabel).toBe('180.00 MiB (188.74 MB)');

    const miniBackbone = formatDualSiIecBytes(DIFFUSION_BACKBONE_MINI_BYTES);
    expect(DIFFUSION_BACKBONE_MINI_BYTES).toBe(916_256);
    expect(miniBackbone.iecLabel).toBe('894.78 KiB');
    expect(miniBackbone.siLabel).toBe('916.26 KB');

    const miniLora = formatDualSiIecBytes(DIFFUSION_LORA_MINI_BYTES);
    expect(DIFFUSION_LORA_MINI_BYTES).toBe(168_960);
    expect(miniLora.iecLabel).toBe('165.00 KiB');
    expect(miniLora.siLabel).toBe('168.96 KB');

    const small = formatDualSiIecBytes(512);
    expect(small.dualLabel).toBe('512 B');

    for (const bad of [0, -1, -1024, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const safe = formatDualSiIecBytes(bad);
      expect(safe.bytes).toBe(0);
      expect(safe.dualLabel).toBe('0 B');
      expect(findInvalidUiNumericStrings(safe)).toEqual([]);
    }
  });

  it('formats parameter counts cleanly and guards invalid numbers', () => {
    expect(formatParameterCount(13_506_073_600)).toBe('13.51B');
    expect(formatParameterCount(47_185_920)).toBe('47.19M');
    expect(formatParameterCount(229_064)).toBe('229.06K');
    expect(formatParameterCount(42_240)).toBe('42.24K');
    expect(formatParameterCount(512)).toBe('512');
    expect(formatParameterCount(0)).toBe('0');
    expect(formatParameterCount(-100)).toBe('0');
    expect(formatParameterCount(Number.NaN)).toBe('0');
    expect(formatParameterCount(Number.POSITIVE_INFINITY)).toBe('0');
  });

  it('verifies 7 family variants, 12+1 14B component breakdown totals, and 13 zero-diff LFS rows', () => {
    const model = createDefaultDiffusionModel();
    const session = createDiffusionSession(model);
    session.step();
    const record = session.getActiveRecord()!;

    const bundle = buildDiffusionArchReadouts({
      mode: record.configAtStep.mode,
      causalContext: record.configAtStep.causalContext,
      counters: record.counters,
      generatedTokens: record.generatedTokenIds.length,
      memoryReport: session.measureMemory(),
    });

    expect(bundle.familyVariants).toHaveLength(7);
    expect(bundle.familyVariants.map((v) => v.id)).toEqual(
      DIFFUSION_FAMILY_SPECS.map((v) => v.id),
    );
    expect(bundle.familyVariants.filter((v) => v.isCanonical)).toHaveLength(1);

    // 12 backbone + 1 LoRA rows
    expect(bundle.components14B).toHaveLength(13);
    const backboneRows = bundle.components14B.filter((r) => r.dtype === 'BF16');
    const loraRows = bundle.components14B.filter((r) => r.dtype === 'F32');
    expect(backboneRows).toHaveLength(12);
    expect(loraRows).toHaveLength(1);

    const sumBackboneTensors = backboneRows.reduce((s, r) => s + r.tensorCount, 0);
    const sumBackboneParams = backboneRows.reduce((s, r) => s + r.parameters, 0);
    const sumBackboneBytes = backboneRows.reduce((s, r) => s + r.payload.bytes, 0);
    expect(sumBackboneTensors).toBe(363);
    expect(sumBackboneParams).toBe(13_506_073_600);
    expect(sumBackboneBytes).toBe(27_012_147_200);

    expect(loraRows[0]!.tensorCount).toBe(80);
    expect(loraRows[0]!.parameters).toBe(47_185_920);
    expect(loraRows[0]!.payload.bytes).toBe(188_743_680);
    expect(loraRows[0]!.shareOfBackbonePct).toBe('+0.35%');

    expect(bundle.canonical14B.backboneParamsFormatted).toBe('13.51B');
    expect(bundle.canonical14B.backbonePayload.dualLabel).toBe('25.16 GiB (27.01 GB)');
    expect(bundle.canonical14B.loraParamsFormatted).toBe('47.19M');
    expect(bundle.canonical14B.loraPayload.dualLabel).toBe('180.00 MiB (188.74 MB)');
    expect(bundle.canonical14B.loraShareOfBackbonePct).toBe('+0.35%');
    expect(bundle.canonical14B.outOfScopeNote).toContain('Nemotron-Labs-Diffusion-Uni');
    expect(bundle.canonical14B.projectionReversalNote).toContain('5120→4096/1024');

    // 13 LFS rows all 0 B diff
    expect(bundle.lfsInventory).toHaveLength(DIFFUSION_LFS_INVENTORY.length);
    expect(bundle.lfsInventory).toHaveLength(13);
    expect(bundle.canonical14B.lfsFileCount).toBe(13);
    expect(bundle.canonical14B.lfsTotalDiffBytes).toBe(0);
    expect(bundle.canonical14B.lfsAllZeroDiff).toBe(true);
    for (const row of bundle.lfsInventory) {
      expect(row.diffBytes).toBe(0);
    }
  });

  it('computes 4-tier context KV scaling and exact Llama-4 transition at 16,384 vs 16,385', () => {
    const defaultRows = buildDiffusionContextKvScalingRows();
    expect(defaultRows.map((r) => r.tokens)).toEqual([
      4096,
      16384,
      20480,
      65536,
      131072,
      262144,
    ]);
    expect(DIFFUSION_CONTEXT_SCALING_PRESETS).toEqual([
      4096,
      16384,
      20480,
      65536,
      131072,
      262144,
    ]);

    const fullSweep = buildDiffusionContextKvScalingRows([
      4096,
      16384,
      16385,
      20480,
      65536,
      131072,
      262144,
      524288,
      1048576,
    ]);

    const byTokens = new Map(fullSweep.map((r) => [r.tokens, r]));

    expect(byTokens.get(4096)!.tier).toBe(1);
    expect(byTokens.get(4096)!.tierBadge).toBe('Verified Training');
    expect(byTokens.get(4096)!.crossesOrigMaxPos).toBe(false);
    expect(byTokens.get(4096)!.llama4ScaleLabel).toBe('1.0000×');

    const at16384 = byTokens.get(16384)!;
    expect(at16384.tier).toBe(1);
    expect(at16384.crossesOrigMaxPos).toBe(false);
    expect(at16384.llama4ScaleAtTail).toBe(1.0);
    expect(at16384.llama4ScaleLabel).toBe('1.0000×');
    expect(at16384.prodBf16Kv.bytes).toBe(16384 * 163_840);
    expect(at16384.prodWithTempBlock32.bytes).toBe(16384 * 163_840 + 5_242_880);
    expect(at16384.miniFp32Kv.bytes).toBe(16384 * DIFFUSION_KV_SLAB_BYTES_PER_TOKEN);

    const at16385 = byTokens.get(16385)!;
    expect(at16385.tier).toBe(3);
    expect(at16385.crossesOrigMaxPos).toBe(true);
    expect(at16385.llama4ScaleLabel).toBe('1.0693×');
    expect(at16385.llama4ScaleAtTail).toBeCloseTo(1 + 0.1 * Math.log(2), 10);

    expect(byTokens.get(20480)!.tier).toBe(2);
    expect(byTokens.get(20480)!.tierBadge).toBe('Harness / Benchmarked Cap');
    expect(byTokens.get(20480)!.llama4ScaleLabel).toBe('1.0693×');

    expect(byTokens.get(65536)!.tier).toBe(2);
    expect(byTokens.get(65536)!.crossesOrigMaxPos).toBe(true);

    expect(byTokens.get(131072)!.tier).toBe(3);
    expect(byTokens.get(131072)!.tierBadge).toBe('Configured YaRN Ceiling (Unverified)');
    expect(byTokens.get(131072)!.llama4ScaleLabel).toBe('1.2079×');

    expect(byTokens.get(262144)!.tier).toBe(3);
    expect(byTokens.get(262144)!.llama4ScaleLabel).toBe('1.2773×');

    expect(byTokens.get(524288)!.tier).toBe(4);
    expect(byTokens.get(524288)!.tierBadge).toBe('Analytical Extrapolation');
    expect(byTokens.get(524288)!.crossesOrigMaxPos).toBe(true);

    expect(byTokens.get(1048576)!.tier).toBe(4);
    expect(byTokens.get(1048576)!.tokenLabel).toBe('1M (1,048,576)');
  });

  it('builds 4-way memory boundary rows from live DiffusionSession.measureMemory()', () => {
    const model = createDefaultDiffusionModel();
    const session = createDiffusionSession(model, {
      config: { mode: 'linear_spec_lora', blockLength: 4, maxNewTokens: 8 },
    });
    session.step();
    session.step();

    const memoryReport = session.measureMemory();
    const rowsGqa = buildDiffusionMemoryBoundaryRows(memoryReport, {
      name: 'GQA+RoPE+LoRA',
      width: 256,
      height: 128,
      bytes: 262_144,
    });

    expect(rowsGqa.map((r) => r.id)).toEqual([
      'backbone_weights',
      'lora_weights',
      'session_retained',
      'active_atlas',
    ]);

    expect(rowsGqa[0]!.bytes).toBe(916_256);
    expect(rowsGqa[0]!.formatted.iecLabel).toBe('894.78 KiB');
    expect(rowsGqa[1]!.bytes).toBe(168_960);
    expect(rowsGqa[1]!.formatted.iecLabel).toBe('165.00 KiB');

    expect(rowsGqa[2]!.bytes).toBe(memoryReport.totalRetainedBytes);
    expect(rowsGqa[2]!.capBytes).toBe(12_582_912);
    expect(rowsGqa[2]!.withinCap).toBe(true);

    expect(rowsGqa[3]!.bytes).toBe(262_144);
    expect(rowsGqa[3]!.capBytes).toBe(524_288);
    expect(rowsGqa[3]!.utilizationPct).toBe(50);
    expect(rowsGqa[3]!.withinCap).toBe(true);

    const rowsSwiglu = buildDiffusionMemoryBoundaryRows(memoryReport, {
      name: 'SwiGLU',
      width: 64,
      height: 64,
    });
    expect(rowsSwiglu[3]!.bytes).toBe(32_768);
    expect(rowsSwiglu[3]!.capBytes).toBe(32_768);
    expect(rowsSwiglu[3]!.utilizationPct).toBe(100);
    expect(rowsSwiglu[3]!.withinCap).toBe(true);
  });

  it('covers all 24 D1–D24 discrepancy entries with exhaustive category partitioning and filters', () => {
    const all = filterDiffusionDiscrepancyLedger();
    expect(all.totalCount).toBe(24);
    expect(all.filteredCount).toBe(24);
    expect(all.rows.map((r) => r.id)).toEqual(
      DIFFUSION_DISCREPANCY_LEDGER.map((e) => e.id),
    );

    const categories = [
      'architecture_specs',
      'rope_masks',
      'execution_kv',
      'samplers_guards',
    ] as const;

    const seenIds = new Set<string>();
    let categorySum = 0;
    for (const category of categories) {
      const model = filterDiffusionDiscrepancyLedger({ category });
      expect(model.filteredCount).toBeGreaterThan(0);
      categorySum += model.filteredCount;
      for (const row of model.rows) {
        expect(row.category).toBe(category);
        expect(seenIds.has(row.id)).toBe(false);
        seenIds.add(row.id);
      }
    }
    expect(categorySum).toBe(24);
    expect(seenIds.size).toBe(24);

    // Free-text query matches
    expect(
      filterDiffusionDiscrepancyLedger({ query: 'D19' }).rows.some(
        (r) => r.id === 'D19',
      ),
    ).toBe(true);
    expect(
      filterDiffusionDiscrepancyLedger({ query: 'llama-4' }).rows.map((r) => r.id),
    ).toEqual(['D3', 'D23']);
    expect(
      filterDiffusionDiscrepancyLedger({ query: 'TRUNCATE' }).rows.map((r) => r.id),
    ).toEqual(['D3', 'D19']);
    expect(
      filterDiffusionDiscrepancyLedger({ query: 'truncate=true' }).rows.map((r) => r.id),
    ).toEqual(['D3']);
    expect(
      filterDiffusionDiscrepancyLedger({ query: 'vlm' }).rows.map((r) => r.id),
    ).toEqual(['D5', 'D10', 'D15', 'D24']);

    // Exact ID filter
    const idsOnly = filterDiffusionDiscrepancyLedger({ ids: ['d12', 'D19'] });
    expect(idsOnly.filteredCount).toBe(2);
    expect(idsOnly.rows.map((r) => r.id)).toEqual(['D12', 'D19']);
  });

  it('formats 3-counter NFE & TPF safely across all 4 modes including 0-denominator edge cases', () => {
    const modes: readonly DiffusionExecutionMode[] = [
      'ar',
      'dlm',
      'linear_spec',
      'linear_spec_lora',
    ];

    for (const mode of modes) {
      const zeroDenom = buildDiffusionNfeTpfReadout(
        mode,
        { actualForwardPasses: 0, decodeForwardPasses: 0, referenceMethodNfe: 0 },
        0,
        mode === 'dlm' ? false : true,
      );
      expect(zeroDenom.tpfActualLabel).toBe('—');
      expect(zeroDenom.tpfDecodeLabel).toBe('—');
      expect(zeroDenom.tpfRefLabel).toBe('—');
      expect(zeroDenom.actualRatioValue).toBeNull();
      expect(zeroDenom.decodeRatioValue).toBeNull();
      expect(zeroDenom.refRatioValue).toBeNull();
      expect(findInvalidUiNumericStrings(zeroDenom)).toEqual([]);
    }

    // Partial zero denominator (e.g. prefill-only decode=0)
    const prefillOnly = buildDiffusionNfeTpfReadout(
      'dlm',
      { actualForwardPasses: 1, decodeForwardPasses: 0, referenceMethodNfe: 0 },
      1,
      true,
    );
    expect(prefillOnly.tpfActualLabel).toBe('1.00 tok/pass');
    expect(prefillOnly.tpfDecodeLabel).toBe('—');
    expect(prefillOnly.tpfRefLabel).toBe('—');
    expect(prefillOnly.actualRatioValue).toBe(1);
    expect(prefillOnly.decodeRatioValue).toBeNull();
    expect(prefillOnly.refRatioValue).toBeNull();

    // Multi-pass speculative decode
    const specReadout = buildDiffusionNfeTpfReadout(
      'linear_spec_lora',
      { actualForwardPasses: 5, decodeForwardPasses: 4, referenceMethodNfe: 5 },
      7,
      true,
    );
    expect(specReadout.tpfActualLabel).toBe('1.40 tok/pass');
    expect(specReadout.tpfDecodeLabel).toBe('1.75 tok/pass');
    expect(specReadout.tpfRefLabel).toBe('1.40 tok/pass');
    expect(specReadout.actualRatioValue).toBeCloseTo(1.4, 10);
  });

  it('produces zero invalid UI numeric strings across live multi-mode session bundles and detects injected hazards', () => {
    const model = createDefaultDiffusionModel();
    const modes: readonly { readonly mode: DiffusionExecutionMode; readonly causalContext: boolean }[] = [
      { mode: 'ar', causalContext: true },
      { mode: 'dlm', causalContext: true },
      { mode: 'dlm', causalContext: false },
      { mode: 'linear_spec', causalContext: true },
      { mode: 'linear_spec_lora', causalContext: true },
    ];

    for (const cfg of modes) {
      const session = createDiffusionSession(model, {
        config: {
          mode: cfg.mode,
          causalContext: cfg.causalContext,
          blockLength: 4,
          maxNewTokens: 4,
        },
      });
      // Initial prefill + completion
      session.step();
      const prefillRecord = session.getActiveRecord()!;
      const prefillBundle = buildDiffusionArchReadouts({
        mode: cfg.mode,
        causalContext: cfg.causalContext,
        counters: prefillRecord.counters,
        generatedTokens: prefillRecord.generatedTokenIds.length,
        memoryReport: session.measureMemory(),
        activeAtlasName: 'GQA+RoPE+LoRA',
        activeAtlasWidth: 256,
        activeAtlasHeight: 128,
        activeAtlasBytes: 262_144,
      });
      expect(findInvalidUiNumericStrings(prefillBundle)).toEqual([]);

      session.runToCompletion();
      const tip = session.getActiveRecord()!;
      const tipBundle = buildDiffusionArchReadouts({
        mode: cfg.mode,
        causalContext: cfg.causalContext,
        counters: tip.counters,
        generatedTokens: tip.generatedTokenIds.length,
        memoryReport: session.measureMemory(),
        activeAtlasName: 'Mask Explorer',
        activeAtlasWidth: 64,
        activeAtlasHeight: 64,
        activeAtlasBytes: 32_768,
      });
      expect(findInvalidUiNumericStrings(tipBundle)).toEqual([]);
      expect(tipBundle.memoryTotalMiniBytes).toBe(
        DIFFUSION_BACKBONE_MINI_BYTES +
          DIFFUSION_LORA_MINI_BYTES +
          session.measureMemory().totalRetainedBytes +
          32_768,
      );
    }

    // Also validate findInvalidUiNumericStrings catches every hazardous pattern
    const badSample = {
      a: 'Infinity tok/pass',
      b: '-Infinity',
      c: 'NaN%',
      d: 'undefined',
      e: 'null',
      f: 'null tok/pass',
      g: Number.NaN,
      h: Number.POSITIVE_INFINITY,
    };
    const issues = findInvalidUiNumericStrings(badSample);
    expect(issues).toHaveLength(8);

    // Defensive check with malformed memory report
    const malformedReport: DiffusionSessionMemoryReport = {
      totalRetainedBytes: Number.NaN,
      maxRetainedBytes: Number.POSITIVE_INFINITY,
      withinBudget: true,
      committedKvSlabBytes: 0,
      activeCroppedSuffixBytes: 0,
      checkpointsAndLogitsBytes: 0,
      activeEagerTracesBytes: 0,
      uniqueBufferCount: 0,
      uniqueSlabCount: 0,
      branchCount: 0,
      recordCount: 0,
      activeRecordId: null,
    };
    const safeBundle = buildDiffusionArchReadouts({
      mode: 'ar',
      causalContext: true,
      counters: {
        actualForwardPasses: Number.NaN,
        decodeForwardPasses: Number.NEGATIVE_INFINITY,
        referenceMethodNfe: -5,
      },
      generatedTokens: Number.NaN,
      memoryReport: malformedReport,
    });
    expect(findInvalidUiNumericStrings(safeBundle)).toEqual([]);
  });
});
