import { describe, expect, it } from 'vitest';
import { formatMemoryReadout, memoryFor } from '../src/model/memory';
import { NEMOTRON_SPEC } from '../src/model/spec';

describe('Fixture 3: Fixed and growing bytes', () => {
  it('computes exact byte totals and slopes at T=8 and T=9 in BF16 and FP8 Selective', () => {
    const bf16T8 = memoryFor(NEMOTRON_SPEC, 8, 'bf16');
    const bf16T9 = memoryFor(NEMOTRON_SPEC, 9, 'bf16');
    const fp8T8 = memoryFor(NEMOTRON_SPEC, 8, 'fp8_selective');
    const fp8T9 = memoryFor(NEMOTRON_SPEC, 9, 'fp8_selective');

    // Fixed Mamba bytes identical in both modes
    expect(bf16T8.fixedMambaBytes).toBe(49_364_992);
    expect(bf16T9.fixedMambaBytes).toBe(49_364_992);
    expect(fp8T8.fixedMambaBytes).toBe(49_364_992);
    expect(fp8T9.fixedMambaBytes).toBe(49_364_992);

    // BF16 slopes and T=8,9 payloads
    expect(bf16T8.kvBytesPerToken).toBe(6_144);
    expect(bf16T8.referenceBytesPerToken).toBe(53_248);
    expect(bf16T8.kvBytes).toBe(49_152);
    expect(bf16T9.kvBytes).toBe(55_296);
    expect(bf16T8.referenceBytes).toBe(425_984);
    expect(bf16T9.referenceBytes).toBe(479_232);

    // FP8 Selective slopes and T=8,9 payloads
    expect(fp8T8.kvBytesPerToken).toBe(3_072);
    expect(fp8T8.referenceBytesPerToken).toBe(26_624);
    expect(fp8T8.kvBytes).toBe(24_576);
    expect(fp8T9.kvBytes).toBe(27_648);
    expect(fp8T8.referenceBytes).toBe(212_992);
    expect(fp8T9.referenceBytes).toBe(239_616);

    // Verify display readout formatting at T=8 and T=9
    const fmtBf16T8 = formatMemoryReadout(bf16T8);
    expect(fmtBf16T8.mambaText).toBe('Mamba state 47.1 MiB fixed');
    expect(fmtBf16T8.kvText).toBe('KV 48 KiB (+6 KiB/token)');
    expect(fmtBf16T8.referenceText).toBe('52-layer GQA reference 416 KiB (+52 KiB/token)');

    const fmtFp8T9 = formatMemoryReadout(fp8T9);
    expect(fmtFp8T9.mambaText).toBe('Mamba state 47.1 MiB fixed');
    expect(fmtFp8T9.kvText).toBe('KV 27 KiB (+3 KiB/token)');
    expect(fmtFp8T9.referenceText).toBe('52-layer GQA reference 234 KiB (+26 KiB/token)');
  });
});

describe('Fixture 4: Crossover with reference', () => {
  it('matches exact crossover F + actualKV(T) == referenceKV(T) at 1,048 BF16 and 2,096 FP8', () => {
    // BF16 crossover at 1,048 tokens
    const bf16Before = memoryFor(NEMOTRON_SPEC, 1_047, 'bf16');
    const bf16At = memoryFor(NEMOTRON_SPEC, 1_048, 'bf16');
    const bf16After = memoryFor(NEMOTRON_SPEC, 1_049, 'bf16');

    expect(bf16Before.hybridTotalBytes).toBeGreaterThan(bf16Before.referenceBytes);
    expect(bf16At.hybridTotalBytes).toBe(bf16At.referenceBytes);
    expect(bf16After.hybridTotalBytes).toBeLessThan(bf16After.referenceBytes);

    // FP8 Selective crossover at 2,096 tokens
    const fp8Before = memoryFor(NEMOTRON_SPEC, 2_095, 'fp8_selective');
    const fp8At = memoryFor(NEMOTRON_SPEC, 2_096, 'fp8_selective');
    const fp8After = memoryFor(NEMOTRON_SPEC, 2_097, 'fp8_selective');

    expect(fp8Before.hybridTotalBytes).toBeGreaterThan(fp8Before.referenceBytes);
    expect(fp8At.hybridTotalBytes).toBe(fp8At.referenceBytes);
    expect(fp8After.hybridTotalBytes).toBeLessThan(fp8After.referenceBytes);
  });

  it('finds first T with actualKV(T) >= F at 8,035 BF16 and 16,070 FP8', () => {
    const bf16Below = memoryFor(NEMOTRON_SPEC, 8_034, 'bf16');
    const bf16At = memoryFor(NEMOTRON_SPEC, 8_035, 'bf16');
    expect(bf16Below.kvBytes).toBeLessThan(bf16Below.fixedMambaBytes);
    expect(bf16At.kvBytes).toBeGreaterThanOrEqual(bf16At.fixedMambaBytes);

    const fp8Below = memoryFor(NEMOTRON_SPEC, 16_069, 'fp8_selective');
    const fp8At = memoryFor(NEMOTRON_SPEC, 16_070, 'fp8_selective');
    expect(fp8Below.kvBytes).toBeLessThan(fp8Below.fixedMambaBytes);
    expect(fp8At.kvBytes).toBeGreaterThanOrEqual(fp8At.fixedMambaBytes);
  });
});
