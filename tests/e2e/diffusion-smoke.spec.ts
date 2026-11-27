import { expect, test, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';
import type { DiffusionSpatialDebugHandle } from '../../src/diffusion/app/spatialApp';
import {
  assertCanvasNonBlank,
  assertCleanUiStrings,
  attachRuntimeGuards,
  waitForAppSettled,
  waitFrames,
} from './helpers';

declare global {
  interface Window {
    __diffusionSpatialDebug?: DiffusionSpatialDebugHandle;
  }
}

const waitForDiffusionSettled = (page: Page) => waitForAppSettled(page, 'diffusion');

async function getDiffusionDebug(page: Page): Promise<DiffusionSpatialDebugHandle> {
  const snapshot = await page.evaluate(() => {
    const d = window.__diffusionSpatialDebug;
    if (!d) return null;
    return {
      viewMode: d.viewMode,
      maskTab: d.maskTab,
      layerIndex: d.layerIndex,
      glErrorCount: d.glErrorCount,
      lastGlError: d.lastGlError,
      frameCount: d.frameCount,
      atlasUploadCount: d.atlasUploadCount,
      atlasWidth: d.atlasWidth,
      atlasHeight: d.atlasHeight,
      atlasKey: d.atlasKey,
      recordCount: d.recordCount,
      kvSeqLen: d.kvSeqLen,
      mode: d.mode,
      loraActiveOnDraft: d.loraActiveOnDraft,
      rafActive: d.rafActive,
      tourActive: d.tourActive,
      tourBeatIndex: d.tourBeatIndex,
      tourTotalBeats: d.tourTotalBeats,
      inspectorTab: d.inspectorTab,
      inspectorOpen: d.inspectorOpen,
      branchCount: d.branchCount,
      activeBranchId: d.activeBranchId,
      activeRecordId: d.activeRecordId,
      retainedBytes: d.retainedBytes,
    };
  });
  expect(snapshot, 'window.__diffusionSpatialDebug must be populated').not.toBeNull();
  return snapshot!;
}

const ALL_SPATIAL_VIEWS = [
  'macro',
  'gqa_rope',
  'swiglu',
  'mask_explorer',
  'spec_verify',
  'embedding',
  'output',
] as const;

const ALL_MASK_TABS = [
  'joint_training',
  'ar_causal',
  'block_bidir',
  'causal_commit_verify',
  'vlm_quadratic_ss',
] as const;

test.describe('Nemotron-Labs-Diffusion-14B · Full Chromium WebGL2 E2E Suite', () => {
  test('1. Root /diffusion/ Boot, All 7 Spatial Views, 5 Mask Tabs & 2D RoPE Phasor Inset', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/diffusion/`, { waitUntil: 'networkidle' });
    await waitForDiffusionSettled(page);

    await expect(page.getByTestId('diffusion-app')).toBeVisible();
    const canvas = page.getByTestId('diffusion-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('.df-webgl-fallback')).toBeHidden();
    await assertCanvasNonBlank(canvas, 'Root /diffusion/ Macro initial frame');

    const initDbg = await getDiffusionDebug(page);
    expect(initDbg.glErrorCount).toBe(0);
    expect(initDbg.viewMode).toBe('macro');
    expect(initDbg.frameCount).toBeGreaterThan(0);

    const epistemicNote = page.getByTestId('diffusion-epistemic-note');
    await expect(epistemicNote).toBeVisible();
    await expect(epistemicNote).toContainText('40L');

    await expect(page.getByTestId('nav-to-nano')).toHaveAttribute('href', '/');
    await expect(page.getByTestId('nav-to-twotower')).toHaveAttribute('href', '/twotower/');

    // Sweep all 7 spatial views
    for (const view of ALL_SPATIAL_VIEWS) {
      await page.getByTestId(`diffusion-view-${view}`).click();
      await waitForDiffusionSettled(page);

      const dbg = await getDiffusionDebug(page);
      expect(dbg.viewMode).toBe(view);
      expect(dbg.glErrorCount).toBe(0);
      await assertCanvasNonBlank(canvas, `View ${view}`);

      if (view === 'gqa_rope') {
        const phasor = page.getByTestId('diffusion-rope-phasor');
        await expect(phasor).toBeVisible();

        // Scrub to p = 17 to cross p = 16 Llama-4 boundary (1.0000 -> 1.0693)
        const scrubber = page.getByTestId('diffusion-rope-pos-scrubber');
        await scrubber.evaluate((el) => {
          const input = el as HTMLInputElement;
          input.value = '17';
          input.dispatchEvent(new Event('input', { bubbles: true }));
        });
        await waitFrames(page, 2);
        await expect(phasor).toContainText('p = 17');
        await expect(phasor).toContainText('1.0693');

        // Switch across all 4 dyadic YaRN rotary planes (0..3)
        for (let planeIdx = 0; planeIdx < 4; planeIdx++) {
          const planeBtn = page.getByTestId(`diffusion-rope-plane-${planeIdx}`);
          await planeBtn.click();
          await waitFrames(page, 2);
          await expect(planeBtn).toHaveAttribute('aria-pressed', 'true');
        }
        await page.getByTestId('diffusion-rope-pos-sync').click();
        await waitFrames(page, 2);
      }

      if (view === 'mask_explorer') {
        for (const tab of ALL_MASK_TABS) {
          const tabBtn = page.getByTestId(`diffusion-mask-tab-${tab}`);
          await tabBtn.click();
          await waitForDiffusionSettled(page);
          await expect(tabBtn).toHaveAttribute('aria-selected', 'true');
          const maskDbg = await getDiffusionDebug(page);
          expect(maskDbg.maskTab).toBe(tab);
          expect(maskDbg.glErrorCount).toBe(0);
          await assertCanvasNonBlank(canvas, `Mask Explorer tab ${tab}`);
        }
      }
    }

    expect(guards.loadedJsAssets.length).toBeGreaterThan(0);
    expect(guards.loadedCssAssets.length).toBeGreaterThan(0);
    guards.assertClean();
  });

  test('2. All 4 Live Execution Modes, Graceful Re-Prefill Flow & Unified Timeline Branching', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/diffusion/`, { waitUntil: 'networkidle' });
    await waitForDiffusionSettled(page);

    const canvas = page.getByTestId('diffusion-canvas');
    const stepBtn = page.getByTestId('diffusion-step-btn');
    const timelineDock = page.getByTestId('diffusion-timeline-dock');
    await expect(timelineDock).toBeVisible();

    // 1. Step in AR
    await stepBtn.click();
    await waitForDiffusionSettled(page);
    let dbg = await getDiffusionDebug(page);
    expect(dbg.mode).toBe('ar');
    expect(dbg.recordCount).toBe(2);
    expect(dbg.kvSeqLen).toBeGreaterThanOrEqual(9);

    // 2. Switch to DLM Causal & step
    await page.getByTestId('diffusion-mode-dlm_causal').click();
    await waitForDiffusionSettled(page);
    await expect(page.getByTestId('diffusion-reprefill-banner')).toBeHidden();
    await stepBtn.click();
    await waitForDiffusionSettled(page);
    dbg = await getDiffusionDebug(page);
    expect(dbg.mode).toBe('dlm');
    expect(dbg.recordCount).toBe(3);

    // 3. Switch to DLM Bidir -> KvProvenanceMismatchError banner -> Confirm Re-Prefill -> Step
    await page.getByTestId('diffusion-mode-dlm_bidir').click();
    await waitFrames(page, 2);
    const reprefillBanner = page.getByTestId('diffusion-reprefill-banner');
    await expect(reprefillBanner).toBeVisible();
    await expect(reprefillBanner).toContainText('KV provenance guard');

    await page.getByTestId('diffusion-reprefill-confirm-btn').click();
    await waitForDiffusionSettled(page);
    await expect(reprefillBanner).toBeHidden();
    await stepBtn.click();
    await waitForDiffusionSettled(page);
    dbg = await getDiffusionDebug(page);
    expect(dbg.mode).toBe('dlm');
    expect(dbg.recordCount).toBe(5);

    // 4. Rewind via timeline click on first causal node (Prefill) -> switch to linear_spec & step
    const firstBranchFirstNode = timelineDock
      .locator('.df-timeline-branch')
      .first()
      .locator('.df-timeline-node')
      .first();
    await expect(firstBranchFirstNode).toContainText('Prefill');
    await firstBranchFirstNode.click();
    await waitForDiffusionSettled(page);

    await page.getByTestId('diffusion-mode-linear_spec').click();
    await waitForDiffusionSettled(page);
    await expect(reprefillBanner).toBeHidden();
    await stepBtn.click();
    await waitForDiffusionSettled(page);
    dbg = await getDiffusionDebug(page);
    expect(dbg.mode).toBe('linear_spec');
    expect(dbg.branchCount).toBeGreaterThanOrEqual(2);
    expect(dbg.loraActiveOnDraft).toBe(false);

    // 5. Rewind again -> switch to linear_spec_lora & step
    await firstBranchFirstNode.click();
    await waitForDiffusionSettled(page);
    await page.getByTestId('diffusion-mode-linear_spec_lora').click();
    await waitForDiffusionSettled(page);
    await stepBtn.click();
    await waitForDiffusionSettled(page);
    dbg = await getDiffusionDebug(page);
    expect(dbg.mode).toBe('linear_spec_lora');
    expect(dbg.branchCount).toBeGreaterThanOrEqual(2);
    expect(dbg.loraActiveOnDraft).toBe(true);

    // 6. Verify Spec & KV view renders non-blank
    await page.getByTestId('diffusion-view-spec_verify').click();
    await waitForDiffusionSettled(page);
    await assertCanvasNonBlank(canvas, 'Spec Verify after Linear Spec + LoRA cycle');
    expect((await getDiffusionDebug(page)).glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('3. Guided 4-Act / 26-Beat Tour, Exact Free-Explore Restoration, 3-Tab Inspector & WAI-ARIA + Idle-RAF', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/diffusion/`, { waitUntil: 'networkidle' });
    await waitForDiffusionSettled(page);

    // Perform distinct Free-Explore edits before entering Tour
    await page.getByTestId('diffusion-step-btn').click();
    await waitForDiffusionSettled(page);
    await page.getByTestId('diffusion-view-gqa_rope').click();
    await waitForDiffusionSettled(page);
    await page.getByTestId('diffusion-layer-select').selectOption('4');
    await waitForDiffusionSettled(page);

    const preTour = await getDiffusionDebug(page);
    expect(preTour.viewMode).toBe('gqa_rope');
    expect(preTour.layerIndex).toBe(4);
    expect(preTour.recordCount).toBe(2);

    // Enter Guided Tour
    const tourToggle = page.getByTestId('diffusion-tour-toggle');
    await tourToggle.click();
    await waitForDiffusionSettled(page);

    const tourHud = page.getByTestId('diffusion-tour-hud');
    await expect(tourHud).toBeVisible();
    await expect(tourToggle).toHaveAttribute('aria-pressed', 'true');
    await expect(tourHud).toContainText('Beat 1 / 26');

    const tourDbg = await getDiffusionDebug(page);
    expect(tourDbg.tourActive).toBe(true);
    expect(tourDbg.tourTotalBeats).toBe(26);

    // Visit Acts 1..4 and advance beats
    for (const actNum of [1, 2, 3, 4] as const) {
      await page.getByTestId(`diffusion-tour-act-${actNum}`).click();
      await waitForDiffusionSettled(page);
      assertCleanUiStrings(await tourHud.textContent(), `Tour Act ${actNum}`);

      if (actNum < 4) {
        await page.getByTestId('diffusion-tour-next').click();
        await waitForDiffusionSettled(page);
        assertCleanUiStrings(await tourHud.textContent(), `Tour Act ${actNum} + Next`);
      }
    }

    // Exit Tour -> verify EXACT Free-Explore restoration
    await page.getByTestId('diffusion-tour-exit').click();
    await waitForDiffusionSettled(page);

    await expect(tourHud).toBeHidden();
    await expect(tourToggle).toHaveAttribute('aria-pressed', 'false');
    const restored = await getDiffusionDebug(page);
    expect(restored.tourActive).toBe(false);
    expect(restored.activeRecordId).toBe(preTour.activeRecordId);
    expect(restored.activeBranchId).toBe(preTour.activeBranchId);
    expect(restored.branchCount).toBe(preTour.branchCount);
    expect(restored.recordCount).toBe(preTour.recordCount);
    expect(restored.kvSeqLen).toBe(preTour.kvSeqLen);
    expect(restored.mode).toBe(preTour.mode);
    expect(restored.viewMode).toBe(preTour.viewMode);
    expect(restored.layerIndex).toBe(preTour.layerIndex);

    // Open 3-tab Inspector
    await page.getByTestId('diffusion-inspector-toggle').click();
    const drawer = page.getByTestId('diffusion-inspector-drawer');
    const panel = page.getByTestId('diffusion-inspector-panel');
    await expect(drawer).toBeVisible();

    // Tab 1: Inspect
    await expect(page.getByTestId('diffusion-inspector-tab-inspect')).toHaveAttribute(
      'aria-selected',
      'true',
    );
    assertCleanUiStrings(await panel.textContent(), 'Inspector Inspect tab');

    // Tab 2: Architecture
    await page.getByTestId('diffusion-inspector-tab-architecture').click();
    await expect(panel).toContainText('25.16 GiB');
    await expect(panel).toContainText('180.00 MiB');
    await expect(panel).toContainText('16,384');
    await expect(panel).toContainText('12.00 MiB');
    assertCleanUiStrings(await panel.textContent(), 'Inspector Architecture tab');

    // Tab 3: Ledger (D1-D24) + Search D19
    const ledgerTab = page.getByTestId('diffusion-inspector-tab-ledger');
    await ledgerTab.click();
    await expect(panel).toContainText('24 / 24 discrepancies');
    await page.getByTestId('diffusion-ledger-search').fill('D19');
    await expect(page.getByTestId('diffusion-ledger-row-D19')).toBeVisible();
    await expect(panel).toContainText('1 / 24 discrepancies');
    assertCleanUiStrings(await panel.textContent(), 'Inspector Ledger tab');
    assertCleanUiStrings(await page.locator('body').textContent(), 'Full document body');

    // Keyboard roving tabindex (Home / ArrowRight / End) on Inspector tablist
    await ledgerTab.focus();
    await page.keyboard.press('Home');
    await expect(page.getByTestId('diffusion-inspector-tab-inspect')).toBeFocused();
    await expect(page.getByTestId('diffusion-inspector-tab-inspect')).toHaveAttribute('tabindex', '0');
    await page.keyboard.press('ArrowRight');
    await expect(page.getByTestId('diffusion-inspector-tab-architecture')).toBeFocused();
    await page.keyboard.press('End');
    await expect(ledgerTab).toBeFocused();

    // Keyboard roving tabindex (Home / ArrowRight / End) on Top view tablist
    const gqaTab = page.getByTestId('diffusion-view-gqa_rope');
    await gqaTab.focus();
    await page.keyboard.press('Home');
    await expect(page.getByTestId('diffusion-view-macro')).toBeFocused();
    await expect(page.getByTestId('diffusion-view-macro')).toHaveAttribute('tabindex', '0');
    await page.keyboard.press('ArrowRight');
    await expect(gqaTab).toBeFocused();
    await page.keyboard.press('End');
    await expect(page.getByTestId('diffusion-view-output')).toBeFocused();
    await waitForDiffusionSettled(page);

    // Escape closes Inspector
    await page.keyboard.press('Escape');
    await expect(drawer).toBeHidden();

    // Strict idle-RAF sleep & zero GL errors
    await waitForDiffusionSettled(page);
    await page.waitForTimeout(250);
    const idle1 = await getDiffusionDebug(page);
    await page.waitForTimeout(250);
    const idle2 = await getDiffusionDebug(page);
    expect(idle2.rafActive).toBe(false);
    expect(idle2.frameCount - idle1.frameCount).toBe(0);
    expect(idle2.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('4. Responsive Viewports (1440×900, 1024×800, 390×844)', async ({ page }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/diffusion/`, { waitUntil: 'networkidle' });

    const viewports = [
      { width: 1440, height: 900 },
      { width: 1024, height: 800 },
      { width: 390, height: 844 },
    ] as const;

    for (const vp of viewports) {
      await page.setViewportSize(vp);
      await waitForDiffusionSettled(page);

      const canvas = page.getByTestId('diffusion-canvas');
      await expect(canvas).toBeVisible();
      await assertCanvasNonBlank(canvas, `Viewport ${vp.width}x${vp.height}`);

      await expect(page.getByTestId('diffusion-step-btn')).toBeVisible();
      await expect(page.getByTestId('diffusion-mode-ar')).toBeVisible();
      await expect(page.getByTestId('model-switcher-trigger')).toBeVisible();

      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow, `Zero horizontal overflow at ${vp.width}x${vp.height}`).toBeLessThanOrEqual(
        1,
      );

      const dbg = await getDiffusionDebug(page);
      expect(dbg.glErrorCount).toBe(0);
    }

    guards.assertClean();
  });

  test('5. GitHub Pages Subpath /llm-viz/diffusion/ Parity (HTTP 200 Assets, RoPE GQA, Tour & Architecture)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${PAGES_BASE_URL}diffusion/`, { waitUntil: 'networkidle' });
    await waitForDiffusionSettled(page);

    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(subpathCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

    await expect(page.getByTestId('nav-to-nano')).toHaveAttribute('href', '/llm-viz/');
    await expect(page.getByTestId('nav-to-twotower')).toHaveAttribute('href', '/llm-viz/twotower/');
    await expect(page.getByTestId('nav-to-uni')).toHaveAttribute('href', '/llm-viz/uni/');

    const diffCanvas = page.getByTestId('diffusion-canvas');
    await expect(diffCanvas).toBeVisible();
    await assertCanvasNonBlank(diffCanvas, 'Subpath /llm-viz/diffusion/ Macro');

    await page.getByTestId('diffusion-view-gqa_rope').click();
    await waitForDiffusionSettled(page);
    await expect(page.getByTestId('diffusion-rope-phasor')).toBeVisible();
    await assertCanvasNonBlank(diffCanvas, 'Subpath /llm-viz/diffusion/ RoPE GQA');

    await page.getByTestId('diffusion-tour-toggle').click();
    await waitForDiffusionSettled(page);
    await expect(page.getByTestId('diffusion-tour-hud')).toBeVisible();
    await page.getByTestId('diffusion-tour-exit').click();
    await waitForDiffusionSettled(page);

    await page.getByTestId('diffusion-inspector-toggle').click();
    await page.getByTestId('diffusion-inspector-tab-architecture').click();
    await expect(page.getByTestId('diffusion-inspector-panel')).toContainText('25.16 GiB');
    expect((await getDiffusionDebug(page)).glErrorCount).toBe(0);

    guards.assertClean();
  });
});
