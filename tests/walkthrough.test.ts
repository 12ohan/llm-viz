import { describe, expect, it } from 'vitest';
import {
  formatBytesAdaptive,
  logPositionFromTokens,
  MAX_SCALING_TOKENS,
  memoryCrossovers,
  memoryFor,
  tokensFromLogPosition,
} from '../src/model/memory';
import { NEMOTRON_SPEC } from '../src/model/spec';
import { MAX_TOKENS, MIN_TOKENS, type TocSectionId, type ViewMode } from '../src/model/state';
import {
  getWalkthroughChapter,
  nextChapterIndex,
  prevChapterIndex,
  WALKTHROUGH_CHAPTERS,
  WALKTHROUGH_COUNT,
} from '../src/story/walkthrough';

const VALID_MODES = new Set<ViewMode>(['tower', 'cutaway_mamba', 'cutaway_moe', 'cutaway_gqa']);
const VALID_SECTIONS = new Set<TocSectionId>([
  'overview',
  'embedding',
  'mamba',
  'moe',
  'gqa',
  'kv_fp8',
  'output',
]);

describe('Milestone 3: Guided 7-Chapter Walkthrough', () => {
  it('defines exactly 7 sequential chapters targeting valid modes, sections, bays, tokens, and precisions', () => {
    expect(WALKTHROUGH_COUNT).toBe(7);
    expect(WALKTHROUGH_CHAPTERS).toHaveLength(7);

    WALKTHROUGH_CHAPTERS.forEach((ch, idx) => {
      expect(ch.index).toBe(idx);
      expect(ch.number).toBe(idx + 1);
      expect(VALID_MODES.has(ch.mode)).toBe(true);
      expect(VALID_SECTIONS.has(ch.section)).toBe(true);
      expect(ch.precision === 'bf16' || ch.precision === 'fp8_selective').toBe(true);
      expect(ch.tokens).toBeGreaterThanOrEqual(MIN_TOKENS);
      expect(ch.tokens).toBeLessThanOrEqual(MAX_TOKENS);
      expect(ch.paragraphs.length).toBeGreaterThanOrEqual(2);
      expect(ch.takeaways.length).toBeGreaterThanOrEqual(2);

      if (ch.selectedBay !== null) {
        expect(ch.selectedBay).toBeGreaterThanOrEqual(1);
        expect(ch.selectedBay).toBeLessThanOrEqual(52);
        expect(NEMOTRON_SPEC.bays[ch.selectedBay - 1]).toBeDefined();
      }
    });

    // Verify exact Milestone 3 chapter choreography
    expect(WALKTHROUGH_CHAPTERS[0]).toMatchObject({
      mode: 'tower',
      section: 'overview',
      precision: 'bf16',
      tokens: 8,
    });
    expect(WALKTHROUGH_CHAPTERS[1]).toMatchObject({
      mode: 'tower',
      section: 'embedding',
      precision: 'bf16',
      tokens: 8,
    });
    expect(WALKTHROUGH_CHAPTERS[2]).toMatchObject({
      mode: 'cutaway_mamba',
      section: 'mamba',
      precision: 'bf16',
      tokens: 8,
      mambaBayId: 1,
    });
    expect(WALKTHROUGH_CHAPTERS[3]).toMatchObject({
      mode: 'cutaway_moe',
      section: 'moe',
      precision: 'bf16',
      tokens: 8,
      selectedBay: 2,
    });
    expect(WALKTHROUGH_CHAPTERS[4]).toMatchObject({
      mode: 'cutaway_gqa',
      section: 'gqa',
      precision: 'bf16',
      tokens: 12,
      gqaBayId: 6,
    });
    expect(WALKTHROUGH_CHAPTERS[5]).toMatchObject({
      mode: 'tower',
      section: 'kv_fp8',
      precision: 'fp8_selective',
      tokens: 9,
      selectedBay: 42,
    });
    expect(WALKTHROUGH_CHAPTERS[6]).toMatchObject({
      mode: 'tower',
      section: 'output',
      precision: 'fp8_selective',
      highlightMemoryChart: true,
    });

    // Stepping helpers clamp cleanly
    expect(prevChapterIndex(0)).toBeNull();
    expect(nextChapterIndex(0)).toBe(1);
    expect(prevChapterIndex(6)).toBe(5);
    expect(nextChapterIndex(6)).toBeNull();
    expect(getWalkthroughChapter(-5).number).toBe(1);
    expect(getWalkthroughChapter(99).number).toBe(7);
  });
});

describe('Milestone 3: 1M-Token (1,048,576) Exact Memory Scaling & Crossovers', () => {
  it('computes exact 1M-token byte totals and GiB savings for BF16 and FP8 Selective', () => {
    expect(MAX_SCALING_TOKENS).toBe(1_048_576);

    const bf16 = memoryFor(NEMOTRON_SPEC, MAX_SCALING_TOKENS, 'bf16');
    const fp8 = memoryFor(NEMOTRON_SPEC, MAX_SCALING_TOKENS, 'fp8_selective');

    // Fixed Mamba-2 state across 23 bays: 23 * (2,097,152 + 49,152) = 49,364,992 B (~47.1 MiB)
    expect(bf16.fixedMambaBytes).toBe(49_364_992);
    expect(fp8.fixedMambaBytes).toBe(49_364_992);

    // BF16 @ 1,048,576 tokens:
    //   6 GQA KV = 6,144 * 1,048,576 = 6,442,450,944 B (6.00 GiB exact)
    //   Hybrid total = 49,364,992 + 6,442,450,944 = 6,491,815,936 B (6.05 GiB)
    //   52-layer GQA ref = 53,248 * 1,048,576 = 55,834,574,848 B (52.0 GiB exact)
    expect(bf16.kvBytes).toBe(6_442_450_944);
    expect(bf16.hybridTotalBytes).toBe(6_491_815_936);
    expect(bf16.referenceBytes).toBe(55_834_574_848);
    expect(formatBytesAdaptive(bf16.hybridTotalBytes)).toBe('6.05 GiB');
    expect(formatBytesAdaptive(bf16.referenceBytes)).toBe('52.0 GiB');

    // FP8 Selective @ 1,048,576 tokens:
    //   6 GQA KV = 3,072 * 1,048,576 = 3,221,225,472 B (3.00 GiB exact)
    //   Hybrid total = 49,364,992 + 3,221,225,472 = 3,270,590,464 B (3.05 GiB)
    //   52-layer GQA ref = 26,624 * 1,048,576 = 27,917,287,424 B (26.0 GiB exact)
    expect(fp8.kvBytes).toBe(3_221_225_472);
    expect(fp8.hybridTotalBytes).toBe(3_270_590_464);
    expect(fp8.referenceBytes).toBe(27_917_287_424);
    expect(formatBytesAdaptive(fp8.hybridTotalBytes)).toBe('3.05 GiB');
    expect(formatBytesAdaptive(fp8.referenceBytes)).toBe('26.0 GiB');

    // Crossovers & savings ratios
    const crossBf16 = memoryCrossovers(NEMOTRON_SPEC, 'bf16');
    const crossFp8 = memoryCrossovers(NEMOTRON_SPEC, 'fp8_selective');

    expect(crossBf16.beatsGqaTokens).toBe(1_048);
    expect(crossBf16.kvExceedsMambaTokens).toBe(8_035);
    expect(crossBf16.savingsRatioAt1M).toBeGreaterThan(8.5);
    expect(crossBf16.savingsRatioAt1M).toBeLessThan(8.7);

    expect(crossFp8.beatsGqaTokens).toBe(2_096);
    expect(crossFp8.kvExceedsMambaTokens).toBe(16_070);
    expect(crossFp8.savingsRatioAt1M).toBeGreaterThan(8.5);
    expect(crossFp8.savingsRatioAt1M).toBeLessThan(8.6);
  });

  it('round-trips log-scale slider probe positions cleanly across 1 -> 1M tokens', () => {
    expect(tokensFromLogPosition(0)).toBe(1);
    expect(tokensFromLogPosition(1)).toBe(MAX_SCALING_TOKENS);
    expect(logPositionFromTokens(1)).toBe(0);
    expect(logPositionFromTokens(MAX_SCALING_TOKENS)).toBe(1);

    for (const target of [1, 1024, 8192, 32_768, 131_072, 262_144, MAX_SCALING_TOKENS]) {
      const pos = logPositionFromTokens(target);
      expect(tokensFromLogPosition(pos)).toBe(target);
    }
  });
});
