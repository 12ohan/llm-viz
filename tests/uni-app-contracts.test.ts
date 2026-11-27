/**
 * Milestone UNI6 — Slice 6C Headless Contract & Guardrail Test Suite
 * (`tests/uni-app-contracts.test.ts`)
 *
 * Exhaustively verifies in headless Node (no DOM/WebGL required):
 *   1. `pixelCanvas.ts`: `renderRgbDescriptorToRgba` for `16×16` & `16×8` CHW inputs,
 *      `U18` `paddingLockMask` hatching diff on `harbor_panorama_16x8`, and `<LOC_0..16>`
 *      `groundingBoxes` pixel-accurate border drawing.
 *   2. `a11yHelpers.ts`: `handleUniTablistKeydown`, `syncRovingTabindex`,
 *      `shouldDeferGlobalShortcut`, `createUniFocusTrap`, `isUniReducedMotionPreferred`.
 *   3. `spatialOverlay.ts`: `UNI_MAX_OVERLAY_CELL_NUM_NODES === 64`,
 *      `computeOverlayCellNumBudget` strict `<= 64` cap enforcement, and zero per-frame
 *      `createElement` / `createElementNS` inside `UniSpatialOverlay.update`.
 *   4. `timelineDock.ts`: `buildUniTimelineDockState` across all 4 dock tasks, `D5`
 *      prefill telemetry, and `taskChoiceToSessionTaskMode` / `sessionTaskModeToDockChoice`.
 *   5. `spatialApp.ts`: `createIdleRafScheduler` (idle rest `rafActive === false`, zero
 *      continuous spin, continuous reasons & disposal) and `createAtlasUploadTracker`
 *      (`null` / identical `atlasKey` dedup = `0` redundant uploads).
 *   6. Static isolation guardrails: scans ALL `.ts` files under `src/uni/` (stripping
 *      comments) and asserts `0` imports of `src/ui/controls.ts` (`ui/controls`),
 *      `src/twotower/`, or `src/diffusion/`, and `0` `Math.random(` calls!
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import {
  createUniFocusTrap,
  handleUniTablistKeydown,
  isUniReducedMotionPreferred,
  shouldDeferGlobalShortcut,
  syncRovingTabindex,
  type UniFocusableTabLike,
} from '../src/uni/app/a11yHelpers';
import { renderRgbDescriptorToRgba } from '../src/uni/app/pixelCanvas';
import {
  createAtlasUploadTracker,
  createIdleRafScheduler,
} from '../src/uni/app/spatialApp';
import {
  computeOverlayCellNumBudget,
  UNI_MAX_OVERLAY_CELL_NUM_NODES,
} from '../src/uni/app/spatialOverlay';
import {
  buildUniTimelineDockState,
  sessionTaskModeToDockChoice,
  taskChoiceToSessionTaskMode,
  UNI_DOCK_TASK_CHOICES,
} from '../src/uni/app/timelineDock';
import type { UniDockTaskChoice } from '../src/uni/app/types';
import { computeSourcePaddingVqMask } from '../src/uni/codec';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import type { UniRgbPreviewDescriptor } from '../src/uni/scene/types';
import { UniSession } from '../src/uni/session';
import { createUniScenePreset, padToSquareAndResize } from '../src/uni/vision';

const SHARED_BUNDLE = createDefaultUniRuntimeBundle();

function makeMockTab(): UniFocusableTabLike & {
  attrs: Record<string, string>;
  focused: number;
} {
  const attrs: Record<string, string> = {};
  const tab = {
    tabIndex: -1,
    attrs,
    focused: 0,
    setAttribute(k: string, v: string) {
      attrs[k] = v;
    },
    focus() {
      tab.focused += 1;
    },
  };
  return tab;
}

function stripTsComments(source: string): string {
  // Remove block comments and single-line comments while preserving newlines
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:\\])\/\/.*$/gm, '$1');
}

function collectTsFilesRecursive(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...collectTsFilesRecursive(full));
    } else if (entry.endsWith('.ts')) {
      out.push(full);
    }
  }
  return out;
}

// ============================================================================
// 1. `pixelCanvas.ts` CONTRACTS
// ============================================================================

describe('pixelCanvas.ts contracts (D4, U18 hatching & U27 grounding boxes)', () => {
  it('renders 16x16 and 16x8 CHW descriptors to exact nearest-neighbor RGBA dimensions & byte values', () => {
    const cat = createUniScenePreset('sunlit_cat_16x16');
    const desc16x16: UniRgbPreviewDescriptor = {
      id: 'cat_src',
      kind: 'source',
      label: 'Sunlit Cat',
      caption: '16x16 native RGB',
      width: cat.width,
      height: cat.height,
      rgbChw: cat.data,
      honestyBadge: 'Miniature 16×16 Mechanics Sim',
    };

    const buf16 = renderRgbDescriptorToRgba(desc16x16, { scale: 4 });
    expect(buf16.width).toBe(64);
    expect(buf16.height).toBe(64);
    expect(buf16.rgba.length).toBe(64 * 64 * 4);
    expect(buf16.hatchedPixelCount).toBe(0);
    expect(buf16.boxBorderPixelCount).toBe(0);

    // Check pixel (0, 0) repeated across 4x4 nearest-neighbor block matches CHW plane bytes
    const expR = Math.round(Math.max(0, Math.min(1, cat.data[0]!)) * 255);
    const expG = Math.round(Math.max(0, Math.min(1, cat.data[256]!)) * 255);
    const expB = Math.round(Math.max(0, Math.min(1, cat.data[512]!)) * 255);
    for (let dy = 0; dy < 4; dy++) {
      for (let dx = 0; dx < 4; dx++) {
        const off = (dy * 64 + dx) * 4;
        expect(buf16.rgba[off]).toBe(expR);
        expect(buf16.rgba[off + 1]).toBe(expG);
        expect(buf16.rgba[off + 2]).toBe(expB);
        expect(buf16.rgba[off + 3]).toBe(255);
      }
    }

    const pano = createUniScenePreset('harbor_panorama_16x8');
    const desc16x8: UniRgbPreviewDescriptor = {
      id: 'pano_src',
      kind: 'source',
      label: 'Harbor Panorama',
      caption: '16x8 native RGB',
      width: pano.width,
      height: pano.height,
      rgbChw: pano.data,
      honestyBadge: 'Miniature 16×16 Mechanics Sim',
    };

    const buf8 = renderRgbDescriptorToRgba(desc16x8, { scale: 6 });
    expect(buf8.width).toBe(96);
    expect(buf8.height).toBe(48);
    expect(buf8.rgba.length).toBe(96 * 48 * 4);
    expect(buf8.hatchedPixelCount).toBe(0);
  });

  it('draws U18 paddingLockMask hatching on harbor_panorama_16x8 locked cells and 0 when disabled', () => {
    const pano = createUniScenePreset('harbor_panorama_16x8');
    const letterboxed = padToSquareAndResize(pano, 16);
    const lockMask = computeSourcePaddingVqMask(letterboxed.info, 4);
    expect(lockMask.lockedCellCount).toBe(8);

    const desc: UniRgbPreviewDescriptor = {
      id: 'pano_letterbox',
      kind: 'letterboxed',
      label: 'Letterboxed Panorama',
      caption: '16x8 -> 16x16 with U18 lock',
      width: letterboxed.image.width,
      height: letterboxed.image.height,
      rgbChw: letterboxed.image.data,
      paddingLockMask: lockMask,
      honestyBadge: 'Miniature 16×16 Mechanics Sim',
    };

    const withLock = renderRgbDescriptorToRgba(desc, { scale: 8, showPaddingLock: true });
    const withoutLock = renderRgbDescriptorToRgba(desc, { scale: 8, showPaddingLock: false });

    expect(withLock.hatchedPixelCount).toBeGreaterThan(0);
    expect(withoutLock.hatchedPixelCount).toBe(0);

    // Top row (y in 0..31) is locked -> pixels must differ; middle rows (y in 32..95) are unlocked -> identical
    let lockedDiffPixels = 0;
    let unlockedDiffPixels = 0;
    for (let y = 0; y < withLock.height; y++) {
      for (let x = 0; x < withLock.width; x++) {
        const off = (y * withLock.width + x) * 4;
        const differs =
          withLock.rgba[off] !== withoutLock.rgba[off] ||
          withLock.rgba[off + 1] !== withoutLock.rgba[off + 1] ||
          withLock.rgba[off + 2] !== withoutLock.rgba[off + 2];
        if (differs) {
          if (y < 32 || y >= 96) lockedDiffPixels += 1;
          else unlockedDiffPixels += 1;
        }
      }
    }
    expect(lockedDiffPixels).toBeGreaterThan(0);
    expect(unlockedDiffPixels).toBe(0);
  });

  it('draws pixel-accurate <LOC_0..16> groundingBoxes borders when enabled and 0 when disabled', () => {
    const cat = createUniScenePreset('sunlit_cat_16x16');
    const desc: UniRgbPreviewDescriptor = {
      id: 'cat_grounding',
      kind: 'grounding',
      label: 'Grounded Cat',
      caption: '16x16 with <LOC> box',
      width: 16,
      height: 16,
      rgbChw: cat.data,
      groundingBoxes: [
        {
          tag: 'cat',
          locTokenIds: [45, 46, 55, 56],
          locTokenStrings: ['<LOC_2>', '<LOC_3>', '<LOC_12>', '<LOC_13>'],
          normalizedBins: [2, 3, 12, 13],
          pixelBox: [2, 3, 12, 13],
          rawTop1Ids: [45, 46, 55, 56],
          guardIntervenedCount: 0,
        },
      ],
      honestyBadge: 'Miniature 16×16 Mechanics Sim',
    };

    const scale = 8;
    const drawn = renderRgbDescriptorToRgba(desc, { scale, showGroundingBoxes: true });
    const hidden = renderRgbDescriptorToRgba(desc, { scale, showGroundingBoxes: false });

    expect(drawn.boxBorderPixelCount).toBeGreaterThan(0);
    expect(hidden.boxBorderPixelCount).toBe(0);

    // Top-left corner of box [2*8=16, 3*8=24] must be emerald-500 [16, 185, 129]
    const cornerOffset = (24 * drawn.width + 16) * 4;
    expect(drawn.rgba[cornerOffset]).toBe(16);
    expect(drawn.rgba[cornerOffset + 1]).toBe(185);
    expect(drawn.rgba[cornerOffset + 2]).toBe(129);
    expect(drawn.rgba[cornerOffset + 3]).toBe(255);

    // Interior center pixel [7*8=56, 8*8=64] must remain untouched
    const interiorOffset = (64 * drawn.width + 56) * 4;
    expect(drawn.rgba[interiorOffset]).toBe(hidden.rgba[interiorOffset]);
    expect(drawn.rgba[interiorOffset + 1]).toBe(hidden.rgba[interiorOffset + 1]);
    expect(drawn.rgba[interiorOffset + 2]).toBe(hidden.rgba[interiorOffset + 2]);
  });
});

// ============================================================================
// 2. `a11yHelpers.ts` CONTRACTS
// ============================================================================

describe('a11yHelpers.ts contracts (D4 localized WAI-ARIA helpers)', () => {
  it('syncRovingTabindex assigns tabindex=0 / aria-selected=true strictly to active tab', () => {
    const tabs = [makeMockTab(), makeMockTab(), makeMockTab()];
    syncRovingTabindex(tabs, 1);

    expect(tabs[0]!.tabIndex).toBe(-1);
    expect(tabs[0]!.attrs['aria-selected']).toBe('false');
    expect(tabs[1]!.tabIndex).toBe(0);
    expect(tabs[1]!.attrs['tabindex']).toBe('0');
    expect(tabs[1]!.attrs['aria-selected']).toBe('true');
    expect(tabs[2]!.tabIndex).toBe(-1);
    expect(tabs[2]!.attrs['aria-selected']).toBe('false');

    // Out-of-bounds clamping & empty array safety
    syncRovingTabindex(tabs, 99);
    expect(tabs[2]!.tabIndex).toBe(0);
    expect(() => syncRovingTabindex([], 0)).not.toThrow();
  });

  it('handleUniTablistKeydown handles ArrowRight, ArrowLeft, Home, End with wrap-around, preventDefault & focus', () => {
    const tabs = [makeMockTab(), makeMockTab(), makeMockTab()];
    const selected: number[] = [];

    const fire = (key: string, activeIdx: number) => {
      const preventDefault = vi.fn();
      const handled = handleUniTablistKeydown({ key, preventDefault }, tabs, activeIdx, (idx) => {
        selected.push(idx);
      });
      return { handled, preventDefault };
    };

    // ArrowRight wraps last -> 0
    const r1 = fire('ArrowRight', 2);
    expect(r1.handled).toBe(true);
    expect(r1.preventDefault).toHaveBeenCalledTimes(1);
    expect(selected[selected.length - 1]).toBe(0);
    expect(tabs[0]!.focused).toBe(1);
    expect(tabs[0]!.attrs['aria-selected']).toBe('true');

    // ArrowLeft wraps 0 -> last
    const r2 = fire('ArrowLeft', 0);
    expect(r2.handled).toBe(true);
    expect(r2.preventDefault).toHaveBeenCalledTimes(1);
    expect(selected[selected.length - 1]).toBe(2);
    expect(tabs[2]!.focused).toBe(1);
    expect(tabs[2]!.attrs['aria-selected']).toBe('true');

    // Home -> 0, End -> 2
    const r3 = fire('Home', 2);
    expect(r3.handled).toBe(true);
    expect(selected[selected.length - 1]).toBe(0);

    const r4 = fire('End', 0);
    expect(r4.handled).toBe(true);
    expect(selected[selected.length - 1]).toBe(2);

    // Unrelated key returns false without preventDefault
    const r5 = fire('Enter', 1);
    expect(r5.handled).toBe(false);
    expect(r5.preventDefault).not.toHaveBeenCalled();
  });

  it('shouldDeferGlobalShortcut defers for INPUT, TEXTAREA, SELECT, contentEditable, textbox & combobox', () => {
    expect(shouldDeferGlobalShortcut({ tagName: 'INPUT' })).toBe(true);
    expect(shouldDeferGlobalShortcut({ tagName: 'textarea' })).toBe(true);
    expect(shouldDeferGlobalShortcut({ tagName: 'SELECT' })).toBe(true);
    expect(shouldDeferGlobalShortcut({ tagName: 'DIV', isContentEditable: true })).toBe(true);
    expect(
      shouldDeferGlobalShortcut({
        tagName: 'DIV',
        getAttribute: (k) => (k === 'contenteditable' ? 'true' : null),
      }),
    ).toBe(true);
    expect(
      shouldDeferGlobalShortcut({
        tagName: 'DIV',
        getAttribute: (k) => (k === 'role' ? 'textbox' : null),
      }),
    ).toBe(true);
    expect(
      shouldDeferGlobalShortcut({
        tagName: 'DIV',
        getAttribute: (k) => (k === 'role' ? 'combobox' : null),
      }),
    ).toBe(true);

    expect(shouldDeferGlobalShortcut({ tagName: 'CANVAS' })).toBe(false);
    expect(shouldDeferGlobalShortcut({ tagName: 'BUTTON' })).toBe(false);
    expect(shouldDeferGlobalShortcut(null)).toBe(false);
  });

  it('createUniFocusTrap cycles Tab / Shift+Tab, fires Escape callback, and restores prior focus', () => {
    const prevFocusEl = {
      hidden: false,
      isConnected: true,
      focus: vi.fn(),
    } as unknown as HTMLElement;

    const btnA = {
      hidden: false,
      disabled: false,
      parentElement: null,
      focus: vi.fn(),
    } as unknown as HTMLElement;
    const btnB = {
      hidden: false,
      disabled: false,
      parentElement: null,
      focus: vi.fn(),
    } as unknown as HTMLElement;

    let currentActive: HTMLElement | null = prevFocusEl;
    const origDocument = globalThis.document;
    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: {
        get activeElement() {
          return currentActive;
        },
      },
    });

    try {
      const container = {
        focus: vi.fn(),
        querySelectorAll: () => [btnA, btnB],
      } as unknown as HTMLElement;

      const onEscape = vi.fn();
      const trap = createUniFocusTrap(container, onEscape);
      trap.activate();
      expect(btnA.focus).toHaveBeenCalledTimes(1);

      // Tab from last wraps to first
      currentActive = btnB;
      const tabEv = { key: 'Tab', shiftKey: false, preventDefault: vi.fn() } as unknown as KeyboardEvent;
      expect(trap.handleKeydown(tabEv)).toBe(true);
      expect(tabEv.preventDefault).toHaveBeenCalledTimes(1);
      expect(btnA.focus).toHaveBeenCalledTimes(2);

      // Shift+Tab from first wraps to last
      currentActive = btnA;
      const shiftTabEv = { key: 'Tab', shiftKey: true, preventDefault: vi.fn() } as unknown as KeyboardEvent;
      expect(trap.handleKeydown(shiftTabEv)).toBe(true);
      expect(shiftTabEv.preventDefault).toHaveBeenCalledTimes(1);
      expect(btnB.focus).toHaveBeenCalledTimes(1);

      // Escape triggers callback
      const escEv = { key: 'Escape', shiftKey: false, preventDefault: vi.fn() } as unknown as KeyboardEvent;
      expect(trap.handleKeydown(escEv)).toBe(true);
      expect(escEv.preventDefault).toHaveBeenCalledTimes(1);
      expect(onEscape).toHaveBeenCalledTimes(1);

      // Deactivate restores previous focus
      trap.deactivate();
      expect(prevFocusEl.focus).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(globalThis, 'document', {
        configurable: true,
        value: origDocument,
      });
    }
  });

  it('isUniReducedMotionPreferred respects matchMedia and handles null/throwing safely', () => {
    expect(
      isUniReducedMotionPreferred({
        matchMedia: () => ({ matches: true }) as MediaQueryList,
      }),
    ).toBe(true);
    expect(
      isUniReducedMotionPreferred({
        matchMedia: () => ({ matches: false }) as MediaQueryList,
      }),
    ).toBe(false);
    expect(isUniReducedMotionPreferred(null)).toBe(false);
    expect(
      isUniReducedMotionPreferred({
        matchMedia: () => {
          throw new Error('unsupported');
        },
      }),
    ).toBe(false);
  });
});

// ============================================================================
// 3. `spatialOverlay.ts` CONTRACTS
// ============================================================================

describe('spatialOverlay.ts contracts (<= 64 cell-num cap & zero per-frame createElement)', () => {
  it('enforces UNI_MAX_OVERLAY_CELL_NUM_NODES === 64 and strict <= 64 budget clamping', () => {
    expect(UNI_MAX_OVERLAY_CELL_NUM_NODES).toBe(64);
    expect(computeOverlayCellNumBudget(256)).toBe(64);
    expect(computeOverlayCellNumBudget(256, 128)).toBe(64);
    expect(computeOverlayCellNumBudget(64, 64)).toBe(64);
    expect(computeOverlayCellNumBudget(48, 32)).toBe(32);
    expect(computeOverlayCellNumBudget(12, 64)).toBe(12);
    expect(computeOverlayCellNumBudget(-5)).toBe(0);
    expect(computeOverlayCellNumBudget(Number.NaN)).toBe(0);
  });

  it('performs zero per-frame createElement / createElementNS calls inside UniSpatialOverlay.update()', () => {
    const src = readFileSync(
      resolve(__dirname, '../src/uni/app/spatialOverlay.ts'),
      'utf8',
    );
    const stripped = stripTsComments(src);
    const updateIdx = stripped.indexOf('update(input: UniSpatialOverlayUpdateInput)');
    const disposeIdx = stripped.indexOf('dispose(): void', updateIdx);
    expect(updateIdx).toBeGreaterThan(-1);
    expect(disposeIdx).toBeGreaterThan(updateIdx);
    const updateBody = stripped.slice(updateIdx, disposeIdx);
    expect(updateBody).not.toContain('createElement(');
    expect(updateBody).not.toContain('createElementNS(');
  });
});

// ============================================================================
// 4. `timelineDock.ts` CONTRACTS
// ============================================================================

describe('timelineDock.ts contracts (4 tasks, D5 telemetry & round-trip mapping)', () => {
  it('round-trips taskChoiceToSessionTaskMode and sessionTaskModeToDockChoice across all 4 dock tasks', () => {
    expect(UNI_DOCK_TASK_CHOICES).toHaveLength(4);
    for (const choice of UNI_DOCK_TASK_CHOICES) {
      const mode = taskChoiceToSessionTaskMode(choice.id);
      expect(mode).toBe(choice.taskMode);
      expect(sessionTaskModeToDockChoice(mode)).toBe(choice.id);
    }
    expect(sessionTaskModeToDockChoice('text_understanding')).toBe('understanding');
  });

  it('buildUniTimelineDockState summarizes all 4 tasks and reflects D5 prefill mode telemetry', () => {
    const session = new UniSession(SHARED_BUNDLE);

    const choices: readonly UniDockTaskChoice[] = [
      'understanding',
      'grounding',
      't2i',
      'image_edit',
    ];

    for (const choice of choices) {
      const taskMode = taskChoiceToSessionTaskMode(choice);
      const branch = session.startBranch({
        taskMode,
        prefillMode: 'elastic_pruned',
        ...(taskMode === 'image_edit' ? { scenePresetId: 'harbor_panorama_16x8' } : {}),
      });
      session.stepBranch(branch.branchId);

      const isGenTask = taskMode === 'text_to_image' || taskMode === 'image_edit';
      const state = buildUniTimelineDockState(session, { isPlaying: false, tourActive: false });
      expect(state.taskChoice).toBe(choice);
      expect(state.taskMode).toBe(taskMode);
      expect(state.executionViewId).toBe(taskMode);
      expect(state.prefillMode).toBe(isGenTask ? 'elastic_pruned' : 'hf_faithful');
      expect(state.totalRecordsInBranch).toBeGreaterThanOrEqual(2);
      expect(state.canStepBack).toBe(true);
      expect(state.canStepForward).toBe(false);
      expect(state.memoryAudit.withinBudget).toBe(true);
      if (isGenTask) {
        expect(state.layerSummary).toContain('0 wasted');
      } else {
        expect(state.layerSummary).toContain('34L U');
      }
    }

    // D5 HF-Faithful telemetry contrast on T2I
    const hfBranch = session.startBranch({
      taskMode: 'text_to_image',
      prefillMode: 'hf_faithful',
    });
    session.stepBranch(hfBranch.branchId);
    const hfState = buildUniTimelineDockState(session);
    expect(hfState.prefillMode).toBe('hf_faithful');
    expect(hfState.layerSummary).toContain('split-U waste');
    expect(hfState.layerSummary).toContain('34L U + 34L G');
  });
});

// ============================================================================
// 5. `spatialApp.ts` PURE STATE MACHINE CONTRACTS (IDLE-RAF & ATLAS DEDUP)
// ============================================================================

describe('spatialApp.ts pure state machines (idle-RAF & RG32F atlas upload dedup)', () => {
  it('createIdleRafScheduler wakes on requestFrame, ticks once, and settles back to rafActive=false with zero spin', () => {
    let nextHandle = 1;
    const queued = new Map<number, (nowMs: number) => void>();
    const cancelled: number[] = [];
    const frames: { nowMs: number; dtMs: number }[] = [];

    const scheduler = createIdleRafScheduler({
      requestFrameFn: (cb) => {
        const h = nextHandle++;
        queued.set(h, cb);
        return h;
      },
      cancelFrameFn: (h) => {
        cancelled.push(h);
        queued.delete(h);
      },
      onFrame: (nowMs, dtMs) => {
        frames.push({ nowMs, dtMs });
      },
    });

    // At rest initially
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);

    // One-shot wake
    scheduler.requestFrame();
    expect(scheduler.isActive()).toBe(true);
    expect(queued.size).toBe(1);

    // Execute tick -> must immediately settle back to idle (`rafActive === false`)
    const [h1, cb1] = Array.from(queued.entries())[0]!;
    queued.delete(h1);
    cb1(100);
    expect(frames).toHaveLength(1);
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);

    // Continuous reason keeps ticking until cleared
    scheduler.setContinuousReason('camera_fly', true);
    expect(scheduler.isActive()).toBe(true);
    expect(scheduler.getContinuousReasons()).toEqual(['camera_fly']);

    const [h2, cb2] = Array.from(queued.entries())[0]!;
    queued.delete(h2);
    cb2(116);
    expect(frames).toHaveLength(2);
    expect(scheduler.isActive()).toBe(true);
    expect(queued.size).toBe(1);

    // Clearing continuous reason immediately cancels queued RAF & returns to rest
    scheduler.setContinuousReason('camera_fly', false);
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);
    expect(cancelled.length).toBeGreaterThanOrEqual(1);

    // Dispose cancels any pending frame cleanly
    scheduler.requestFrame();
    expect(scheduler.isActive()).toBe(true);
    scheduler.dispose();
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);
  });

  it('createAtlasUploadTracker performs 0 redundant uploads on null or repeated atlasKey and increments only on change', () => {
    const tracker = createAtlasUploadTracker();

    expect(tracker.getLastAtlasKey()).toBeNull();
    expect(tracker.getUploadCount()).toBe(0);

    // Null atlas (e.g. View 1 Macro) causes 0 uploads
    expect(tracker.shouldUpload(null)).toBe(false);
    expect(tracker.getUploadCount()).toBe(0);

    // First real atlas key uploads once
    expect(tracker.shouldUpload('visual_input:128x128:abc')).toBe(true);
    expect(tracker.getUploadCount()).toBe(1);
    expect(tracker.getLastAtlasKey()).toBe('visual_input:128x128:abc');

    // Revisiting identical key -> 0 redundant uploads
    expect(tracker.shouldUpload('visual_input:128x128:abc')).toBe(false);
    expect(tracker.getUploadCount()).toBe(1);

    // Switching to Macro (null) and back to identical key -> still 0 redundant uploads
    expect(tracker.shouldUpload(null)).toBe(false);
    expect(tracker.shouldUpload('visual_input:128x128:abc')).toBe(false);
    expect(tracker.getUploadCount()).toBe(1);

    // New key increments count
    expect(tracker.shouldUpload('joint_attn:256x256:def')).toBe(true);
    expect(tracker.getUploadCount()).toBe(2);
    expect(tracker.getLastAtlasKey()).toBe('joint_attn:256x256:def');

    // Context-lost reset forces re-upload of same key on restore
    tracker.reset();
    expect(tracker.getLastAtlasKey()).toBeNull();
    expect(tracker.shouldUpload('joint_attn:256x256:def')).toBe(true);
    expect(tracker.getUploadCount()).toBe(3);
  });
});

// ============================================================================
// 6. STATIC ISOLATION GUARDRAILS ACROSS ALL `src/uni/**/*.ts`
// ============================================================================

describe('Static isolation guardrails across all src/uni/**/*.ts files', () => {
  const uniRoot = resolve(__dirname, '../src/uni');
  const uniFiles = collectTsFilesRecursive(uniRoot);

  it('discovers all src/uni/**/*.ts modules including app & story slices', () => {
    expect(uniFiles.length).toBeGreaterThanOrEqual(20);
  });

  it('contains ZERO imports of src/ui/controls.ts (D4), src/twotower/, or src/diffusion/, and ZERO Math.random(', () => {
    const forbiddenImportPatterns: readonly { readonly label: string; readonly re: RegExp }[] = [
      {
        label: 'ui/controls (Locked Decision D4)',
        re: /\b(?:import|export)\b[^;\n]*['"][^'"]*ui\/controls(?:\.ts)?['"]/,
      },
      {
        label: 'src/twotower/',
        re: /\b(?:import|export)\b[^;\n]*['"][^'"]*twotower\//,
      },
      {
        label: 'src/diffusion/',
        re: /\b(?:import|export)\b[^;\n]*['"][^'"]*diffusion\//,
      },
    ];

    const violations: string[] = [];

    for (const filePath of uniFiles) {
      const raw = readFileSync(filePath, 'utf8');
      const code = stripTsComments(raw);
      const rel = filePath.slice(uniRoot.length + 1);

      for (const rule of forbiddenImportPatterns) {
        if (rule.re.test(code)) {
          violations.push(`${rel}: forbidden import matching ${rule.label}`);
        }
      }
      if (/\bMath\.random\s*\(/.test(code)) {
        violations.push(`${rel}: forbidden Math.random(`);
      }
    }

    expect(violations).toEqual([]);
  });
});
