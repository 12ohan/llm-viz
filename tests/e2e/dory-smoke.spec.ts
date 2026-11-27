import { expect, test, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';
import type { DorySpatialDebugHandle, DorySpatialDebugSnapshot } from '../../src/dory/app/types';
import type { DorySpatialViewMode } from '../../src/dory/scene/types';
import {
  assertCanvasNonBlank,
  assertCleanUiStrings,
  attachRuntimeGuards,
  waitForAppSettled,
  waitFrames,
} from './helpers';

declare global {
  interface Window {
    __dorySpatialDebug?: DorySpatialDebugHandle;
    __doryTestSliderRef?: HTMLInputElement | null;
    __doryTestControlsSlotRef?: HTMLElement | null;
    __doryTestFirstSliderRef?: HTMLInputElement | null;
  }
}

const waitForDorySettled = (page: Page) => waitForAppSettled(page, 'dory');

async function getDoryDebug(page: Page): Promise<DorySpatialDebugSnapshot> {
  const snapshot = await page.evaluate(() => {
    const d = window.__dorySpatialDebug;
    return d ? d.getSnapshot() : null;
  });
  expect(snapshot, 'window.__dorySpatialDebug must be populated').not.toBeNull();
  return snapshot!;
}

const ALL_DORY_VIEWS: readonly DorySpatialViewMode[] = [
  'macro',
  'hypersphere',
  'gqa_rope',
  'swiglu',
  'kv_compare',
  'self_spec',
  'bookends',
] as const;

test.describe('NVIDIA Nemotron-Labs-Dory1.0-4B · Full Chromium WebGL2 E2E Suite', () => {
  test('1. Boot /dory/ Cleanly in Chromium WebGL2 with Zero GL Errors & Idle-RAF Settled', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    await expect(page.getByTestId('dory-app')).toBeVisible();
    await expect(page.getByTestId('dory-app-shell')).toBeVisible();
    await expect(page.getByTestId('dory-viewport-host')).toBeVisible();
    const canvas = page.getByTestId('dory-webgl-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.getByTestId('dory-webgl-fallback')).toBeHidden();

    await assertCanvasNonBlank(canvas, 'Root /dory/ Macro initial frame');

    const initDbg = await getDoryDebug(page);
    expect(initDbg.modelId).toBe('dory');
    expect(initDbg.viewMode).toBe('macro');
    expect(initDbg.kLoops).toBe(2);
    expect(initDbg.kvCacheMode).toBe('per_loop');
    expect(initDbg.executionMode).toBe('standard');
    expect(initDbg.executedDepth).toBe(112);
    expect(initDbg.kvBankCount).toBe(56);
    expect(initDbg.glErrorCount).toBe(0);
    expect(initDbg.rafActive).toBe(false);
    expect(initDbg.tourBeatCount).toBe(14);
    expect(initDbg.overlayCellNumNodes).toBeLessThanOrEqual(64);
    expect(initDbg.atlasByteLength).toBeLessThanOrEqual(524_288);
    expect(initDbg.readOnlyTbptlVerified).toBe(true);
    expect(initDbg.zeroFakeQuantToggles).toBe(true);
    expect(initDbg.memoryAudit.withinTotalBudgetWithWeights).toBe(true);
    expect(initDbg.memoryAudit.withinCheckpointPoolBudget).toBe(true);

    // Strict idle-RAF sleep across a 250ms rest window
    await page.waitForTimeout(250);
    const idleDbg = await getDoryDebug(page);
    expect(idleDbg.rafActive).toBe(false);
    expect(idleDbg.glErrorCount).toBe(0);

    assertCleanUiStrings(await page.locator('body').textContent(), 'Root /dory/ initial body');

    expect(guards.loadedJsAssets.length).toBeGreaterThan(0);
    expect(guards.loadedCssAssets.length).toBeGreaterThan(0);
    guards.assertClean();
  });

  test('2. All 7 Editorial Spatial Views + 2D Analytical SVG Insets + RG32F Atlas Deduplication', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    const canvas = page.getByTestId('dory-webgl-canvas');

    const expectedChartsByView: Readonly<Record<DorySpatialViewMode, readonly string[]>> = {
      macro: [],
      hypersphere: ['spanning_subspace', 'exploratory_pca'],
      gqa_rope: ['dual_rope_phasors'],
      swiglu: [],
      kv_compare: ['kv_scaling_256k'],
      self_spec: ['self_spec_probs'],
      bookends: ['tbptl_schematic'],
    };

    for (const view of ALL_DORY_VIEWS) {
      await page.getByTestId(`dory-view-tab-${view}`).click();
      await waitForDorySettled(page);

      const dbg = await getDoryDebug(page);
      expect(dbg.viewMode).toBe(view);
      expect(dbg.glErrorCount).toBe(0);
      expect(dbg.rafActive).toBe(false);
      expect(dbg.overlayCellNumNodes).toBeLessThanOrEqual(64);
      expect(dbg.atlasByteLength).toBeLessThanOrEqual(524_288);

      await assertCanvasNonBlank(canvas, `View ${view}`);
      assertCleanUiStrings(
        await page.locator('.dory-stage-title-card').textContent(),
        `Stage title card on ${view}`,
      );
      assertCleanUiStrings(
        await page.getByTestId('dory-inspector-body').textContent(),
        `Inspector body on ${view}`,
      );

      for (const chartKind of expectedChartsByView[view]) {
        const chartCard = page.getByTestId(`dory-chart-${chartKind}`);
        await expect(chartCard, `Chart ${chartKind} visible on ${view}`).toBeVisible();
        await expect(chartCard.locator('svg')).toBeVisible();
      }
    }

    // Verify content-addressed RG32F atlas upload deduplication on View 3 (`gqa_rope`)
    await page.getByTestId('dory-view-tab-gqa_rope').click();
    await waitForDorySettled(page);

    const beforeHover = await getDoryDebug(page);
    expect(beforeHover.atlasKey).not.toBeNull();
    expect(beforeHover.atlasUploadCount).toBeGreaterThan(0);

    const box = await canvas.boundingBox();
    expect(box, 'WebGL2 canvas must have a valid bounding box').not.toBeNull();
    const cx = box!.x + box!.width * 0.5;
    const cy = box!.y + box!.height * 0.5;

    // Hover and orbit-drag over the canvas — must NOT re-upload the RG32F atlas
    await page.mouse.move(cx - 40, cy - 20);
    await waitFrames(page, 2);
    await page.mouse.move(cx + 40, cy + 20);
    await waitFrames(page, 2);
    await page.mouse.down();
    await page.mouse.move(cx + 85, cy - 12, { steps: 5 });
    await page.mouse.up();
    await waitForDorySettled(page);

    const afterOrbit = await getDoryDebug(page);
    expect(afterOrbit.atlasKey).toBe(beforeHover.atlasKey);
    expect(afterOrbit.atlasUploadCount).toBe(beforeHover.atlasUploadCount);
    expect(afterOrbit.glErrorCount).toBe(0);

    // Re-clicking active tab or collapsing/expanding inspector must NOT re-upload the RG32F atlas
    await page.getByTestId('dory-view-tab-gqa_rope').click();
    await waitForDorySettled(page);
    expect((await getDoryDebug(page)).atlasUploadCount).toBe(beforeHover.atlasUploadCount);

    await page.getByTestId('dory-inspector-toggle-btn').click();
    await waitForDorySettled(page);
    await page.getByTestId('dory-inspector-toggle-btn').click();
    await waitForDorySettled(page);

    const afterToggle = await getDoryDebug(page);
    expect(afterToggle.atlasKey).toBe(beforeHover.atlasKey);
    expect(afterToggle.atlasUploadCount).toBe(beforeHover.atlasUploadCount);
    expect(afterToggle.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('3. Elastic Depth k Switching (1..4 & +6,8 Extrapolation) + per_loop/last_loop DR18 Immutable Forking', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    const canvas = page.getByTestId('dory-webgl-canvas');
    const initDbg = await getDoryDebug(page);
    const initialBranchId = initDbg.activeBranchId;
    expect(initialBranchId).not.toBeNull();
    expect(initDbg.branchCount).toBe(1);

    // Switch k=2 -> k=4: DR18 clean immutable branch fork (192L, 96 per_loop banks)
    await page.getByTestId('dory-dock-k-4').click();
    await waitForDorySettled(page);

    let dbg = await getDoryDebug(page);
    expect(dbg.kLoops).toBe(4);
    expect(dbg.executedDepth).toBe(192);
    expect(dbg.kvBankCount).toBe(96);
    expect(dbg.branchCount).toBe(2);
    expect(dbg.activeBranchId).not.toBe(initialBranchId);
    expect(dbg.glErrorCount).toBe(0);
    await expect(page.getByTestId('dory-dock-dr18-notice')).toBeVisible();
    await expect(page.getByTestId('dory-dock-dr18-notice')).toContainText('DR18');

    // Toggle Folded <-> Unrolled on View 1 Macro at k=4
    await page.getByTestId('dory-macro-unrolled-btn').click();
    await waitForDorySettled(page);
    expect((await getDoryDebug(page)).macroDisplayMode).toBe('unrolled');
    await assertCanvasNonBlank(canvas, 'View 1 Macro Unrolled k=4');
    await page.getByTestId('dory-macro-folded-btn').click();
    await waitForDorySettled(page);
    expect((await getDoryDebug(page)).macroDisplayMode).toBe('folded');

    // Enable base-model extrapolation +6,8 and verify k=8 stays bounded under D10 3.00 MiB LRU / 12 MiB cap
    await page.getByTestId('dory-dock-extrap-toggle').check();
    await expect(page.getByTestId('dory-dock-k-6')).toBeVisible();
    await expect(page.getByTestId('dory-dock-k-8')).toBeVisible();
    await page.getByTestId('dory-dock-k-8').click();
    await waitForDorySettled(page);

    dbg = await getDoryDebug(page);
    expect(dbg.kLoops).toBe(8);
    expect(dbg.executedDepth).toBe(352);
    expect(dbg.kvBankCount).toBe(176);
    expect(dbg.branchCount).toBe(3);
    expect(dbg.memoryAudit.withinTotalBudgetWithWeights).toBe(true);
    expect(dbg.memoryAudit.withinCheckpointPoolBudget).toBe(true);
    expect(dbg.glErrorCount).toBe(0);

    // Switch per_loop -> last_loop: DR18 clean fork freezes KV banks at 36
    const beforePolicyBranch = dbg.activeBranchId;
    await page.getByTestId('dory-dock-kv-last_loop').click();
    await waitForDorySettled(page);

    dbg = await getDoryDebug(page);
    expect(dbg.kvCacheMode).toBe('last_loop');
    expect(dbg.kvBankCount).toBe(36);
    expect(dbg.branchCount).toBe(4);
    expect(dbg.activeBranchId).not.toBe(beforePolicyBranch);
    await expect(page.getByTestId('dory-dock-kv-last_loop')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('dory-dock-kv-per_loop')).toHaveAttribute('aria-pressed', 'false');

    // Switch to View 5 (kv_compare) and exercise DR17 Theorems 1-3 partition modes
    await page.getByTestId('dory-view-tab-kv_compare').click();
    await waitForDorySettled(page);
    const kvDiag = page.getByTestId('dory-kv-compare-diagnostics');
    await expect(kvDiag).toBeVisible();
    await expect(kvDiag).toContainText('(4+5k)/9');
    await expect(kvDiag).toContainText('Thm 1');
    await expect(kvDiag).toContainText('Thm 3');

    for (const partId of ['single_shot', 'chunked', 'one_by_one'] as const) {
      await page.getByTestId(`dory-kv-part-${partId}`).click();
      await waitForDorySettled(page);
      expect((await getDoryDebug(page)).glErrorCount).toBe(0);
      await assertCanvasNonBlank(canvas, `View 5 kv_compare ${partId}`);
    }

    // Switch back to per_loop -> clean fork preserving previous branches
    await page.getByTestId('dory-dock-kv-per_loop').click();
    await waitForDorySettled(page);
    dbg = await getDoryDebug(page);
    expect(dbg.kvCacheMode).toBe('per_loop');
    expect(dbg.branchCount).toBe(5);
    await expect(page.getByTestId('dory-dock-kv-per_loop')).toHaveAttribute('aria-pressed', 'true');

    // Verify all 5 immutable branches are preserved in the branch selector
    const branchOptions = await page
      .getByTestId('dory-dock-branch-select')
      .locator('option')
      .count();
    expect(branchOptions).toBe(5);

    guards.assertClean();
  });

  test('4. Contract A Paper §6.2 Self-Speculative View (104 Banks) & View 7 Read-Only Bookends/Readouts/Ledger', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    const canvas = page.getByTestId('dory-webgl-canvas');

    // Visit View 6 (`self_spec`) on standard session -> read-only preview + Contract A badge
    await page.getByTestId('dory-view-tab-self_spec').click();
    await waitForDorySettled(page);

    const specBanner = page.getByTestId('dory-selfspec-banner');
    await expect(specBanner).toBeVisible();
    await expect(specBanner).toContainText('Reconstructed from Paper §6.2');
    await expect(specBanner).toContainText('Contract A');

    const specDiag = page.getByTestId('dory-selfspec-diagnostics');
    await expect(specDiag).toBeVisible();
    await expect(specDiag).toContainText('104');
    await expect(specDiag).toContainText('Shallow');
    await expect(specDiag).toContainText('Deep');
    await expect(page.getByTestId('dory-chart-self_spec_probs')).toBeVisible();
    await expect(page.getByTestId('dory-activate-selfspec-btn')).toBeVisible();
    await assertCanvasNonBlank(canvas, 'View 6 Self-Spec preview');

    // Activate live Contract A self-spec session (104 banks, k=1->4, per_loop)
    await page.getByTestId('dory-activate-selfspec-btn').click();
    await waitForDorySettled(page);

    let dbg = await getDoryDebug(page);
    expect(dbg.executionMode).toBe('self_spec');
    expect(dbg.kLoops).toBe(4);
    expect(dbg.kvCacheMode).toBe('per_loop');
    expect(dbg.kvBankCount).toBe(104);
    expect(dbg.glErrorCount).toBe(0);
    await expect(page.getByTestId('dory-dock-mode-self_spec')).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await expect(page.getByTestId('dory-activate-selfspec-btn')).toBeHidden();

    // Execute an additional live gamma=4 Contract A speculative cycle
    const prevRecordCount = dbg.recordCount;
    await page.getByTestId('dory-dock-execute-step').click();
    await waitForDorySettled(page);
    dbg = await getDoryDebug(page);
    expect(dbg.recordCount).toBeGreaterThanOrEqual(prevRecordCount);
    expect(dbg.glErrorCount).toBe(0);
    await assertCanvasNonBlank(canvas, 'View 6 Self-Spec live Contract A cycle');

    // Visit View 7 (`bookends`): verify strictly READ-ONLY · NO GRADIENTS banner & zero fake quant toggles
    await page.getByTestId('dory-view-tab-bookends').click();
    await waitForDorySettled(page);

    const roBanner = page.getByTestId('dory-readonly-banner');
    await expect(roBanner).toBeVisible();
    await expect(roBanner).toContainText('READ-ONLY · NO GRADIENTS');

    const bookDiag = page.getByTestId('dory-bookends-diagnostics');
    await expect(bookDiag).toBeVisible();
    await expect(bookDiag).toContainText('hasGradients=false');
    await expect(bookDiag).toContainText('zeroFakeQuant=true');
    await expect(page.getByTestId('dory-chart-tbptl_schematic')).toBeVisible();

    for (const fmt of ['mxfp8', 'nvfp4', 'q5_k_m', 'bf16'] as const) {
      await page.getByTestId(`dory-bookends-ckpt-${fmt}`).click();
      await waitForDorySettled(page);
      expect((await getDoryDebug(page)).selectedCheckpointFormat).toBe(fmt);
      expect((await getDoryDebug(page)).glErrorCount).toBe(0);
    }
    await assertCanvasNonBlank(canvas, 'View 7 Bookends & Read-Only TBPTL');

    // Inspect Readouts tab (`0 B` diff 4-checkpoint table)
    await page.getByTestId('dory-inspector-tab-readouts').click();
    await expect(page.getByTestId('dory-readouts-scope')).toBeVisible();
    const ckptSec = page.getByTestId('dory-readouts-checkpoints');
    await expect(ckptSec).toBeVisible();
    await expect(ckptSec).toContainText('BF16');
    await expect(ckptSec).toContainText('MXFP8');
    await expect(ckptSec).toContainText('NVFP4');
    await expect(ckptSec).toContainText('Q5_K_M');
    await expect(ckptSec).toContainText('0 B');
    assertCleanUiStrings(await ckptSec.textContent(), 'Readouts checkpoints table');

    // Inspect Ledger DR1-DR24 tab + live search
    await page.getByTestId('dory-inspector-tab-ledger').click();
    const ledgerSearch = page.getByTestId('dory-ledger-search');
    await expect(ledgerSearch).toBeVisible();
    await ledgerSearch.fill('DR23');
    await expect(page.getByTestId('dory-ledger-card-DR23')).toBeVisible();

    guards.assertClean();
  });

  test('5. Interactive Regressions: In-Place Slider DOM Identity & Hover Controls-Slot Preservation', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    // Helper verifying exact DOM node identity + focus retention across sequential `input` events
    async function assertSliderInPlace(
      testId: string,
      valA: string,
      valB: string,
    ): Promise<void> {
      const slider = page.getByTestId(testId);
      await expect(slider).toBeVisible();
      await slider.focus();

      await page.evaluate(
        ({ id, v }) => {
          const el = document.querySelector<HTMLInputElement>(`[data-testid="${id}"]`);
          window.__doryTestSliderRef = el;
          if (!el) throw new Error(`Missing slider ${id}`);
          el.value = v;
          el.dispatchEvent(new Event('input', { bubbles: true }));
        },
        { id: testId, v: valA },
      );
      await waitFrames(page, 2);

      await page.evaluate(
        ({ v }) => {
          const el = window.__doryTestSliderRef;
          if (!el) throw new Error('Missing saved slider reference');
          el.value = v;
          el.dispatchEvent(new Event('input', { bubbles: true }));
        },
        { v: valB },
      );
      await waitForDorySettled(page);

      const check = await page.evaluate((id) => {
        const ref = window.__doryTestSliderRef;
        const current = document.querySelector<HTMLInputElement>(`[data-testid="${id}"]`);
        return {
          hasRef: Boolean(ref),
          isConnected: Boolean(ref?.isConnected),
          isActiveElement: ref !== null && ref === document.activeElement,
          sameDomNode: ref !== null && ref === current,
          value: ref?.value ?? null,
        };
      }, testId);

      expect(check, `${testId} must stay in-place with focus`).toEqual({
        hasRef: true,
        isConnected: true,
        isActiveElement: true,
        sameDomNode: true,
        value: valB,
      });
    }

    // 1. View 2 (`hypersphere`): `dory-layer-slider` preserves DOM node & focus
    await page.getByTestId('dory-view-tab-hypersphere').click();
    await waitForDorySettled(page);
    await assertSliderInPlace('dory-layer-slider', '16', '22');

    // 2. View 3 (`gqa_rope`): `dory-gqa-layer-slider` preserves DOM node & focus
    await page.getByTestId('dory-view-tab-gqa_rope').click();
    await waitForDorySettled(page);
    await assertSliderInPlace('dory-gqa-layer-slider', '16', '22');

    // 3. View 4 (`swiglu`): `dory-swiglu-reveal-slider` preserves DOM node & focus
    await page.getByTestId('dory-view-tab-swiglu').click();
    await waitForDorySettled(page);
    await assertSliderInPlace('dory-swiglu-reveal-slider', '0.25', '0.75');

    // 4. Timeline Dock: `dory-dock-temp-slider` preserves DOM node & focus
    await assertSliderInPlace('dory-dock-temp-slider', '0.4', '0.8');

    // 5. Hovering across `[data-testid="dory-webgl-canvas"]` NEVER replaces controls slot or its slider
    await page.evaluate(() => {
      const controlsSlot = document.querySelector<HTMLElement>(
        '[data-testid="dory-inspector-controls-slot"]',
      );
      const firstSlider =
        controlsSlot?.querySelector<HTMLInputElement>('input[type="range"]') ?? null;
      window.__doryTestControlsSlotRef = controlsSlot;
      window.__doryTestFirstSliderRef = firstSlider;
    });

    const cellSlot = page.getByTestId('dory-inspector-cell-slot');
    await expect(cellSlot).toBeVisible();

    const canvas = page.getByTestId('dory-webgl-canvas');
    const canvasBox = await canvas.boundingBox();
    expect(canvasBox).not.toBeNull();

    const probeCoords = [
      [0.25, 0.35],
      [0.5, 0.45],
      [0.65, 0.55],
      [0.4, 0.65],
      [0.55, 0.35],
    ] as const;

    for (const [fx, fy] of probeCoords) {
      await page.mouse.move(
        canvasBox!.x + canvasBox!.width * fx,
        canvasBox!.y + canvasBox!.height * fy,
      );
      await waitFrames(page, 2);
    }
    await waitForDorySettled(page);

    const hoverPreserveCheck = await page.evaluate(() => {
      const curSlot = document.querySelector<HTMLElement>(
        '[data-testid="dory-inspector-controls-slot"]',
      );
      const curSlider = curSlot?.querySelector<HTMLInputElement>('input[type="range"]') ?? null;
      const curCellSlot = document.querySelector<HTMLElement>(
        '[data-testid="dory-inspector-cell-slot"]',
      );
      return {
        controlsSlotPreserved:
          window.__doryTestControlsSlotRef !== null && curSlot === window.__doryTestControlsSlotRef,
        firstSliderPreserved:
          window.__doryTestFirstSliderRef !== null &&
          curSlider === window.__doryTestFirstSliderRef &&
          Boolean(curSlider?.isConnected),
        cellSlotMounted: Boolean(curCellSlot?.isConnected),
      };
    });

    expect(hoverPreserveCheck).toEqual({
      controlsSlotPreserved: true,
      firstSliderPreserved: true,
      cellSlotMounted: true,
    });
    expect((await getDoryDebug(page)).glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('6. 4-Act / 14-Beat Guided Tour + Tour Control Lock + Escape Deferral + Free-Explore Isolation', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(`${ROOT_BASE_URL}/dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    const drawer = page.getByTestId('dory-inspector-drawer');
    await expect(drawer).toBeVisible();
    await expect(drawer).not.toHaveClass(/is-collapsed/);

    // Establish distinctive Free-Explore state before launching tour (k=4, stepped, View 3)
    await page.getByTestId('dory-dock-k-4').click();
    await waitForDorySettled(page);
    await page.getByTestId('dory-dock-execute-step').click();
    await waitForDorySettled(page);
    await page.getByTestId('dory-view-tab-gqa_rope').click();
    await waitForDorySettled(page);

    const preTour = await getDoryDebug(page);
    expect(preTour.viewMode).toBe('gqa_rope');
    expect(preTour.kLoops).toBe(4);
    expect(preTour.kvCacheMode).toBe('per_loop');
    expect(preTour.activeBranchId).not.toBeNull();
    expect(preTour.activeRecordId).not.toBeNull();
    expect(preTour.tourActive).toBe(false);

    // Launch isolated 4-Act / 14-Beat Guided Tour
    await page.getByTestId('dory-tour-launch-btn').click();
    await waitForDorySettled(page);

    const tourHud = page.getByTestId('dory-tour-hud');
    await expect(tourHud).toBeVisible();
    let dbg = await getDoryDebug(page);
    expect(dbg.tourActive).toBe(true);
    expect(dbg.tourBeatIndex).toBe(0);
    expect(dbg.tourBeatCount).toBe(14);
    expect(dbg.activeTourCoord).not.toBeNull();

    // Verify Next / Prev controls work cleanly
    await page.getByTestId('dory-tour-next-btn').click();
    await waitForDorySettled(page);
    expect((await getDoryDebug(page)).tourBeatIndex).toBe(1);
    await page.getByTestId('dory-tour-prev-btn').click();
    await waitForDorySettled(page);
    expect((await getDoryDebug(page)).tourBeatIndex).toBe(0);

    // Verify dock & inspector view controls are locked during tour
    await expect(page.getByTestId('dory-tour-lock-note')).toBeVisible();
    for (const dockId of [
      'dory-dock-k-1',
      'dory-dock-k-2',
      'dory-dock-k-4',
      'dory-dock-kv-per_loop',
      'dory-dock-kv-last_loop',
      'dory-dock-mode-standard',
      'dory-dock-mode-self_spec',
      'dory-dock-execute-step',
    ] as const) {
      await expect(page.getByTestId(dockId), `${dockId} disabled during tour`).toBeDisabled();
    }
    const unlockedInspectorControls = await page
      .getByTestId('dory-inspector-controls-slot')
      .locator('button:not([disabled]), input:not([disabled]), select:not([disabled])')
      .count();
    expect(unlockedInspectorControls).toBe(0);

    // Jump across all 4 Acts via Act buttons & representative beats spanning all 7 views
    const representativeBeats = [
      { idx: 0, actIdx: 0, expectedView: 'bookends' as const, label: 'Beat 1 (Act I · Bookends)' },
      { idx: 1, actIdx: 0, expectedView: 'macro' as const, label: 'Beat 2 (Act I · 72L Macro)' },
      { idx: 2, actIdx: 0, expectedView: 'hypersphere' as const, label: 'Beat 3 (Act I · NLERP)' },
      { idx: 3, actIdx: 1, expectedView: 'gqa_rope' as const, label: 'Beat 4 (Act II · SWA GQA)' },
      { idx: 5, actIdx: 1, expectedView: 'swiglu' as const, label: 'Beat 6 (Act II · SwiGLU)' },
      { idx: 6, actIdx: 2, expectedView: 'macro' as const, label: 'Beat 7 (Act III · Unrolled k=4)' },
      { idx: 8, actIdx: 2, expectedView: 'kv_compare' as const, label: 'Beat 9 (Act III · last_loop)' },
      { idx: 10, actIdx: 3, expectedView: 'self_spec' as const, label: 'Beat 11 (Act IV · Contract A)' },
      { idx: 13, actIdx: 3, expectedView: 'bookends' as const, label: 'Beat 14 (Act IV · TBPTL)' },
    ] as const;

    const canvas = page.getByTestId('dory-webgl-canvas');
    for (const beat of representativeBeats) {
      await page.evaluate((idx) => {
        window.__dorySpatialDebug?.goToBeat(idx);
      }, beat.idx);
      await waitForDorySettled(page);

      const beatDbg = await getDoryDebug(page);
      expect(beatDbg.tourActive).toBe(true);
      expect(beatDbg.tourBeatIndex).toBe(beat.idx);
      expect(beatDbg.tourActIndex).toBe(beat.actIdx);
      expect(beatDbg.viewMode).toBe(beat.expectedView);
      expect(beatDbg.activeTourCoord).not.toBeNull();
      expect(beatDbg.glErrorCount).toBe(0);
      await expect(page.getByTestId('dory-tour-coord')).toBeVisible();
      assertCleanUiStrings(await tourHud.textContent(), beat.label);
    }

    // Verify Act button click navigation (Act II -> Beat 3)
    await page.getByTestId('dory-tour-act-act2').click();
    await waitForDorySettled(page);
    expect((await getDoryDebug(page)).tourActIndex).toBe(1);
    await assertCanvasNonBlank(canvas, 'Guided Tour Act II WebGL2 frame');

    // Click a `.dory-tour-ref-chip` -> opens Ledger tab & filters search
    const firstRefChip = tourHud.locator('.dory-tour-ref-chip').first();
    await expect(firstRefChip).toBeVisible();
    await firstRefChip.click();

    await expect(page.getByTestId('dory-inspector-tab-ledger')).toHaveAttribute(
      'aria-selected',
      'true',
    );
    const ledgerSearch = page.getByTestId('dory-ledger-search');
    await expect(ledgerSearch).toBeVisible();
    expect((await ledgerSearch.inputValue()).length).toBeGreaterThan(0);

    // Pressing Escape while focused in Ledger search input MUST defer (tour stays active)
    await ledgerSearch.focus();
    await ledgerSearch.fill('DR7');
    await expect(ledgerSearch).toBeFocused();
    await expect(page.getByTestId('dory-ledger-card-DR7')).toBeVisible();

    await page.keyboard.press('Escape');
    await waitFrames(page, 2);
    expect((await getDoryDebug(page)).tourActive).toBe(true);
    await expect(tourHud).toBeVisible();

    // Blur search input and press Escape -> exits tour and restores Free-Explore state identically
    await ledgerSearch.evaluate((el) => {
      (el as HTMLElement).blur();
    });
    await page.keyboard.press('Escape');
    await waitForDorySettled(page);

    await expect(tourHud).toBeHidden();
    const restored = await getDoryDebug(page);
    expect(restored.tourActive).toBe(false);
    expect(restored.viewMode).toBe(preTour.viewMode);
    expect(restored.kLoops).toBe(preTour.kLoops);
    expect(restored.kvCacheMode).toBe(preTour.kvCacheMode);
    expect(restored.executionMode).toBe(preTour.executionMode);
    expect(restored.activeBranchId).toBe(preTour.activeBranchId);
    expect(restored.activeRecordId).toBe(preTour.activeRecordId);
    expect(restored.branchCount).toBe(preTour.branchCount);
    expect(restored.recordCount).toBe(preTour.recordCount);
    expect(restored.committedLen).toBe(preTour.committedLen);
    expect(restored.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('7. GitHub Pages Subpath /llm-viz/dory/ Parity (HTTP 200 Assets, WebGL2 Render & 5-Way Base-Path Switcher Links)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);
    const doryCanvas = page.getByTestId('dory-webgl-canvas');

    await page.goto(`${PAGES_BASE_URL}dory/`, { waitUntil: 'networkidle' });
    await waitForDorySettled(page);

    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(subpathCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

    await assertCanvasNonBlank(doryCanvas, 'Subpath /llm-viz/dory/ Macro');
    const initDbg = await getDoryDebug(page);
    expect(initDbg.glErrorCount).toBe(0);
    expect(initDbg.rafActive).toBe(false);

    // Verify all 5 base-path-aware links in top-bar switcher
    await expect(page.getByTestId('nav-to-nano')).toHaveAttribute('href', '/llm-viz/');
    await expect(page.getByTestId('nav-to-twotower')).toHaveAttribute('href', '/llm-viz/twotower/');
    await expect(page.getByTestId('nav-to-diffusion')).toHaveAttribute(
      'href',
      '/llm-viz/diffusion/',
    );
    await expect(page.getByTestId('nav-to-uni')).toHaveAttribute('href', '/llm-viz/uni/');
    await expect(page.getByTestId('nav-to-dory')).toHaveAttribute('href', '/llm-viz/dory/');
    await expect(page.getByTestId('nav-to-dory')).toHaveAttribute('aria-current', 'page');

    // Interactive view switch + Contract A Self-Spec activation under /llm-viz/ subpath
    await page.getByTestId('dory-view-tab-hypersphere').click();
    await waitForDorySettled(page);
    await assertCanvasNonBlank(doryCanvas, 'Subpath /llm-viz/dory/ Hypersphere');
    await expect(page.getByTestId('dory-chart-spanning_subspace')).toBeVisible();
    await expect(page.getByTestId('dory-chart-exploratory_pca')).toBeVisible();

    await page.getByTestId('dory-view-tab-self_spec').click();
    await waitForDorySettled(page);
    await page.getByTestId('dory-dock-mode-self_spec').click();
    await waitForDorySettled(page);
    const specDbg = await getDoryDebug(page);
    expect(specDbg.executionMode).toBe('self_spec');
    expect(specDbg.kvBankCount).toBe(104);
    expect(specDbg.glErrorCount).toBe(0);
    await assertCanvasNonBlank(doryCanvas, 'Subpath /llm-viz/dory/ Self-Spec');

    guards.assertClean();
  });
});
