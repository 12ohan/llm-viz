import { describe, expect, it } from 'vitest';
import { NEMOTRON_SPEC } from '../src/model/spec';
import {
  MAX_TOKENS,
  MIN_TOKENS,
  type ExecutionMode,
  type TocSectionId,
} from '../src/model/state';
import {
  beatsForChapter,
  findMatchingBeat,
  firstBeatIndexForChapter,
  getWalkthroughBeat,
  getWalkthroughChapter,
  nextBeatIndex,
  nextChapterIndex,
  prevBeatIndex,
  prevChapterIndex,
  WALKTHROUGH_BEAT_COUNT,
  WALKTHROUGH_BEATS,
  WALKTHROUGH_CHAPTERS,
  WALKTHROUGH_COUNT,
  type WalkthroughAct,
  type WalkthroughArAction,
  type WalkthroughBeatMode,
} from '../src/story/walkthrough';

const VALID_BEAT_MODES = new Set<WalkthroughBeatMode>([
  'tower',
  'cutaway_embedding',
  'cutaway_mamba',
  'cutaway_moe',
  'cutaway_gqa',
  'cutaway_output',
]);

const VALID_SECTIONS = new Set<TocSectionId>([
  'overview',
  'embedding',
  'mamba',
  'moe',
  'gqa',
  'kv_fp8',
  'output',
]);

const VALID_EXECUTION_MODES = new Set<ExecutionMode>(['full_sequence', 'ar_decode']);

const VALID_AR_ACTIONS = new Set<WalkthroughArAction>([
  'ensure_prefill',
  'step_once_with_wave',
  'inspect_cached_gqa',
  'invite_branch_and_chart',
]);

const EXPECTED_CHAPTER_BEAT_COUNTS = [2, 2, 4, 4, 4, 2, 4] as const;

describe('Increment V5: Reassessed 22-Beat, 3-Act Unified + AR Decode Walkthrough', () => {
  it('defines exactly 22 sequential beats covering all 7 chapters and all 3 Acts cleanly', () => {
    expect(WALKTHROUGH_BEAT_COUNT).toBe(22);
    expect(WALKTHROUGH_BEATS).toHaveLength(22);

    const seenIds = new Set<string>();
    const seenChapters = new Set<number>();
    const seenActs = new Set<WalkthroughAct>();

    WALKTHROUGH_BEATS.forEach((beat, idx) => {
      expect(beat.index).toBe(idx);
      expect(seenIds.has(beat.id)).toBe(false);
      seenIds.add(beat.id);

      expect(beat.chapterIndex).toBeGreaterThanOrEqual(0);
      expect(beat.chapterIndex).toBeLessThan(WALKTHROUGH_COUNT);
      seenChapters.add(beat.chapterIndex);

      const expectedCount = EXPECTED_CHAPTER_BEAT_COUNTS[beat.chapterIndex];
      expect(beat.chapterBeatCount).toBe(expectedCount);
      expect(beat.beatInChapter).toBeGreaterThanOrEqual(1);
      expect(beat.beatInChapter).toBeLessThanOrEqual(expectedCount);
      expect(beat.id).toBe(`${beat.chapterIndex + 1}.${beat.beatInChapter}`);

      expect([1, 2, 3]).toContain(beat.act);
      seenActs.add(beat.act);
      expect(beat.actLabel.length).toBeGreaterThan(0);

      expect(VALID_BEAT_MODES.has(beat.mode)).toBe(true);
      expect(VALID_SECTIONS.has(beat.section)).toBe(true);
      expect(VALID_EXECUTION_MODES.has(beat.executionMode)).toBe(true);
      expect(beat.precision === 'bf16' || beat.precision === 'fp8_selective').toBe(true);
      expect(beat.tokens).toBeGreaterThanOrEqual(MIN_TOKENS);
      expect(beat.tokens).toBeLessThanOrEqual(MAX_TOKENS);

      expect(beat.title.trim().length).toBeGreaterThan(0);
      expect(beat.badge.trim().length).toBeGreaterThan(0);
      expect(beat.paragraphs.length).toBeGreaterThanOrEqual(2);
      expect(beat.takeaways.length).toBeGreaterThanOrEqual(2);
      expect(beat.interactivePrompt && beat.interactivePrompt.trim().length > 0).toBe(true);

      if (beat.selectedBay !== null) {
        expect(beat.selectedBay).toBeGreaterThanOrEqual(1);
        expect(beat.selectedBay).toBeLessThanOrEqual(52);
        expect(NEMOTRON_SPEC.bays[beat.selectedBay - 1]).toBeDefined();
      }

      if (beat.stepperStep !== undefined) {
        expect(Number.isInteger(beat.stepperStep)).toBe(true);
        expect(beat.stepperStep).toBeGreaterThanOrEqual(0);
        expect(beat.stepperStep).toBeLessThanOrEqual(5);
      }

      if (beat.gqaHead !== undefined) {
        expect(beat.gqaHead).toBeGreaterThanOrEqual(0);
        expect(beat.gqaHead).toBeLessThan(NEMOTRON_SPEC.gqa.q);
      }

      if (beat.arAction !== undefined) {
        expect(VALID_AR_ACTIONS.has(beat.arAction)).toBe(true);
      }
    });

    // All 7 chapters and all 3 acts are represented
    expect(seenChapters.size).toBe(7);
    expect([...seenActs].sort()).toEqual([1, 2, 3]);

    // Acts and chapters progress monotonically
    for (let i = 1; i < WALKTHROUGH_BEATS.length; i++) {
      expect(WALKTHROUGH_BEATS[i].chapterIndex).toBeGreaterThanOrEqual(
        WALKTHROUGH_BEATS[i - 1].chapterIndex,
      );
      expect(WALKTHROUGH_BEATS[i].act).toBeGreaterThanOrEqual(WALKTHROUGH_BEATS[i - 1].act);
    }

    // Verify exact per-chapter beat counts & firstBeatIndexForChapter offsets
    let runningOffset = 0;
    EXPECTED_CHAPTER_BEAT_COUNTS.forEach((count, chapterIndex) => {
      const chBeats = beatsForChapter(chapterIndex);
      expect(chBeats).toHaveLength(count);
      expect(firstBeatIndexForChapter(chapterIndex)).toBe(runningOffset);
      chBeats.forEach((beat, j) => {
        expect(beat.index).toBe(runningOffset + j);
        expect(beat.beatInChapter).toBe(j + 1);
      });
      runningOffset += count;
    });
    expect(runningOffset).toBe(22);
  });

  it('enforces exact 3-Act choreography across Tower, Embedding, Mamba-2, MoE, GQA, FP8, and Output/AR Decode', () => {
    // Act I (Ch 1–5): full_sequence, BF16
    for (let i = 0; i <= 15; i++) {
      expect(WALKTHROUGH_BEATS[i].act).toBe(1);
      expect(WALKTHROUGH_BEATS[i].executionMode).toBe('full_sequence');
      expect(WALKTHROUGH_BEATS[i].precision).toBe('bf16');
    }

    // Ch 2 uses dedicated 3D cutaway_embedding and teaches untied W_embed + NoPE
    const ch2 = beatsForChapter(1);
    expect(ch2[0]).toMatchObject({ id: '2.1', mode: 'cutaway_embedding', tokens: 8 });
    expect(ch2[1]).toMatchObject({ id: '2.2', mode: 'cutaway_embedding', tokens: 8 });
    expect(ch2[0].paragraphs.join(' ')).toMatch(/W_embed/);
    expect(ch2[0].paragraphs.join(' ')).toMatch(/layers/);
    expect(ch2[1].paragraphs.join(' ')).toMatch(/NoPE/);
    expect(ch2[1].paragraphs.join(' ')).toMatch(/No Pos Embed/i);

    // Ch 3 uses Bay 01 Mamba-2 cutaway with exact stepper progression and 47.1 MiB / norm_before_gate=False
    const ch3 = beatsForChapter(2);
    expect(ch3.map((b) => b.mode)).toEqual([
      'cutaway_mamba',
      'cutaway_mamba',
      'cutaway_mamba',
      'cutaway_mamba',
    ]);
    expect(ch3.every((b) => b.mambaBayId === 1 && b.selectedBay === 1)).toBe(true);
    expect(ch3[2].paragraphs.join(' ')).toMatch(/47\.1 MiB/);
    expect(ch3[3].paragraphs.join(' ')).toMatch(/norm_before_gate\s*=\s*False/);

    // Ch 4 uses Bay 02 MoE cutaway and EXPLICITLY teaches the U vs router selection/scaling distinction
    const ch4 = beatsForChapter(3);
    expect(ch4.map((b) => b.mode)).toEqual([
      'cutaway_moe',
      'cutaway_moe',
      'cutaway_moe',
      'cutaway_moe',
    ]);
    expect(ch4.every((b) => b.moeBayId === 2 && b.selectedBay === 2)).toBe(true);

    const ch4Beat1Text = [...ch4[0].paragraphs, ...ch4[0].takeaways].join(' ');
    expect(ch4Beat1Text).toMatch(/U\s*=\s*RMSNorm\(X\)/);
    expect(ch4Beat1Text).toMatch(/expert MLPs.*consume U/i);
    expect(ch4Beat1Text).toMatch(/select.*6.*experts.*scale.*outputs/i);

    const ch4Beat2Text = [...ch4[1].paragraphs, ...ch4[1].takeaways].join(' ');
    expect(ch4Beat2Text).toMatch(/e_score_correction_bias/);
    expect(ch4Beat2Text).toMatch(/122/);

    const ch4Beat3Text = [...ch4[2].paragraphs, ...ch4[2].takeaways].join(' ');
    expect(ch4Beat3Text).toMatch(/bias.*dropped/i);
    expect(ch4Beat3Text).toMatch(/2\.5/);
    expect(ch4Beat3Text).toMatch(/6-term/i);

    const ch4Beat4Text = [...ch4[3].paragraphs, ...ch4[3].takeaways].join(' ');
    expect(ch4Beat4Text).toMatch(/ReLU²/);
    expect(ch4Beat4Text).toMatch(/gate_proj/);
    expect(ch4Beat4Text).toMatch(/Shared Expert/);
    expect(ch4Beat4Text).toMatch(/0 B/);

    // Ch 5 uses Bay 06 GQA at T=12 and showcases Q00 -> KV_0 vs Q16 -> KV_1
    const ch5 = beatsForChapter(4);
    expect(ch5.every((b) => b.mode === 'cutaway_gqa' && b.gqaBayId === 6 && b.tokens === 12)).toBe(
      true,
    );
    expect(ch5[0].gqaHead).toBe(0);
    expect(ch5[1].gqaHead).toBe(16);
    expect(ch5[1].title + ' ' + ch5[1].paragraphs.join(' ')).toMatch(/Q00.*KV₀/);
    expect(ch5[1].title + ' ' + ch5[1].paragraphs.join(' ')).toMatch(/Q16.*KV₁/);
    expect(ch5[2].paragraphs.join(' ')).toMatch(/−∞/);
    expect(ch5[3].paragraphs.join(' ')).toMatch(/8\.7×/);

    // Act II (Ch 6): Selective FP8 boundary on Tower Bays 42–43
    const ch6 = beatsForChapter(5);
    expect(ch6.every((b) => b.act === 2 && b.executionMode === 'full_sequence')).toBe(true);
    expect(ch6.every((b) => b.mode === 'tower' && b.precision === 'fp8_selective')).toBe(true);
    expect(ch6[0].selectedBay).toBe(42);
    expect(ch6[1].selectedBay).toBe(43);
    expect(ch6[0].paragraphs.join(' ')).toMatch(/12 bays/);
    expect(ch6[0].paragraphs.join(' ')).toMatch(/23 Conv1D/);
    expect(ch6[1].paragraphs.join(' ')).toMatch(/\+3 KiB\/tok/);
    expect(ch6[1].paragraphs.join(' ')).toMatch(/~99%/);

    // Act III (Ch 7): Dedicated cutaway_output -> live ar_decode spine wave -> [1×(P+1)] Bay 43 GQA -> branching + 1M chart
    const ch7 = beatsForChapter(6);
    expect(ch7.every((b) => b.act === 3 && b.precision === 'fp8_selective')).toBe(true);

    expect(ch7[0]).toMatchObject({
      id: '7.1',
      mode: 'cutaway_output',
      executionMode: 'full_sequence',
      arAction: 'ensure_prefill',
    });
    expect(ch7[0].paragraphs.join(' ')).toMatch(/Final RMSNorm/);
    expect(ch7[0].paragraphs.join(' ')).toMatch(/W_lm_head/);

    expect(ch7[1]).toMatchObject({
      id: '7.2',
      mode: 'tower',
      executionMode: 'ar_decode',
      arAction: 'step_once_with_wave',
    });
    expect(ch7[1].paragraphs.join(' ')).toMatch(/t₄/);
    expect(ch7[1].paragraphs.join(' ')).toMatch(/69 KiB/);
    expect(ch7[1].paragraphs.join(' ')).toMatch(/47\.1 MiB/);

    expect(ch7[2]).toMatchObject({
      id: '7.3',
      mode: 'cutaway_gqa',
      executionMode: 'ar_decode',
      gqaBayId: 43,
      selectedBay: 43,
      arAction: 'inspect_cached_gqa',
    });
    expect(ch7[2].paragraphs.join(' ')).toMatch(/\[1 × \(P\+1\)\]/);
    expect(ch7[2].paragraphs.join(' ')).toMatch(/zero prefix recompute/i);

    expect(ch7[3]).toMatchObject({
      id: '7.4',
      mode: 'tower',
      executionMode: 'ar_decode',
      arAction: 'invite_branch_and_chart',
      highlightMemoryChart: true,
    });
    expect(ch7[3].paragraphs.join(' ')).toMatch(/branch-1/);
    expect(ch7[3].paragraphs.join(' ')).toMatch(/3\.05 GiB/);
    expect(ch7[3].paragraphs.join(' ')).toMatch(/26\.0 GiB/);
    expect(ch7[3].paragraphs.join(' ')).toMatch(/8\.5×/);
  });

  it('provides clean clamping, stepping, chapter lookup, and mode/step matching helpers', () => {
    // Stepping & clamping
    expect(prevBeatIndex(0)).toBeNull();
    expect(prevBeatIndex(-3)).toBeNull();
    expect(nextBeatIndex(0)).toBe(1);
    expect(prevBeatIndex(21)).toBe(20);
    expect(nextBeatIndex(21)).toBeNull();
    expect(nextBeatIndex(99)).toBeNull();

    expect(getWalkthroughBeat(-10).id).toBe('1.1');
    expect(getWalkthroughBeat(0).id).toBe('1.1');
    expect(getWalkthroughBeat(21).id).toBe('7.4');
    expect(getWalkthroughBeat(999).id).toBe('7.4');
    expect(getWalkthroughBeat(Number.NaN).id).toBe('1.1');

    // Chapter boundary helpers
    expect(firstBeatIndexForChapter(-5)).toBe(0);
    expect(firstBeatIndexForChapter(0)).toBe(0);
    expect(firstBeatIndexForChapter(6)).toBe(18);
    expect(firstBeatIndexForChapter(99)).toBe(18);
    expect(beatsForChapter(-1)).toEqual([]);
    expect(beatsForChapter(7)).toEqual([]);

    // findMatchingBeat across modes, steps (including multi-step coverage), and execution modes
    expect(findMatchingBeat('tower')?.id).toBe('1.1');
    expect(findMatchingBeat('tower', undefined, 'ar_decode')?.id).toBe('7.2');
    expect(findMatchingBeat('cutaway_embedding', 0)?.id).toBe('2.1');
    expect(findMatchingBeat('cutaway_embedding', 1)?.id).toBe('2.2');

    expect(findMatchingBeat('cutaway_mamba', 0)?.id).toBe('3.1');
    expect(findMatchingBeat('cutaway_mamba', 1)?.id).toBe('3.1');
    expect(findMatchingBeat('cutaway_mamba', 2)?.id).toBe('3.2');
    expect(findMatchingBeat('cutaway_mamba', 3)?.id).toBe('3.3');
    expect(findMatchingBeat('cutaway_mamba', 4)?.id).toBe('3.4');
    expect(findMatchingBeat('cutaway_mamba', 5)?.id).toBe('3.4');

    expect(findMatchingBeat('cutaway_moe', 0)?.id).toBe('4.1');
    expect(findMatchingBeat('cutaway_moe', 1)?.id).toBe('4.2');
    expect(findMatchingBeat('cutaway_moe', 2)?.id).toBe('4.3');
    expect(findMatchingBeat('cutaway_moe', 3)?.id).toBe('4.4');
    expect(findMatchingBeat('cutaway_moe', 4)?.id).toBe('4.4');
    expect(findMatchingBeat('cutaway_moe', 5)?.id).toBe('4.4');

    expect(findMatchingBeat('cutaway_gqa', 0, 'full_sequence')?.id).toBe('5.1');
    expect(findMatchingBeat('cutaway_gqa', 2, 'full_sequence')?.id).toBe('5.3');
    expect(findMatchingBeat('cutaway_gqa', 3, 'full_sequence')?.id).toBe('5.3');
    expect(findMatchingBeat('cutaway_gqa', 4, 'full_sequence')?.id).toBe('5.4');
    expect(findMatchingBeat('cutaway_gqa', 5, 'full_sequence')?.id).toBe('5.4');
    expect(findMatchingBeat('cutaway_gqa', 2, 'ar_decode')?.id).toBe('7.3');
    expect(findMatchingBeat('cutaway_gqa', undefined, 'ar_decode')?.id).toBe('7.3');

    expect(findMatchingBeat('cutaway_output')?.id).toBe('7.1');
  });

  it('preserves the legacy 7-chapter WALKTHROUGH_CHAPTERS regression contract untouched', () => {
    expect(WALKTHROUGH_COUNT).toBe(7);
    expect(WALKTHROUGH_CHAPTERS).toHaveLength(7);
    expect(getWalkthroughChapter(0).id).toBe('hybrid-spine');
    expect(getWalkthroughChapter(6).id).toBe('long-context-output');
    expect(prevChapterIndex(0)).toBeNull();
    expect(nextChapterIndex(0)).toBe(1);
    expect(prevChapterIndex(6)).toBe(5);
    expect(nextChapterIndex(6)).toBeNull();
  });
});
