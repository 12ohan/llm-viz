import { expect, test } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';
import { MODEL_REGISTRY, resolveModelHref, type ModelEntryId } from '../../src/shared/modelRegistry';
import {
  assertCanvasNonBlank,
  assertSwitcherMenuUnclipped,
  attachRuntimeGuards,
  getAppCanvasLocator,
  getAppGlErrorCount,
  navigateViaModelSwitcher,
  waitForAppSettled,
} from './helpers';

const FULL_TOUR_SEQUENCE: readonly ModelEntryId[] = ['twotower', 'diffusion', 'uni', 'dory', 'nano'];

const REQUIRED_VIEWPORTS = [
  { width: 1440, height: 900 },
  { width: 1280, height: 800 },
  { width: 1024, height: 800 },
  { width: 390, height: 844 },
] as const;

function urlFor(baseUrl: string, id: ModelEntryId): string {
  const entry = MODEL_REGISTRY.find((e) => e.id === id)!;
  const cleanBase = baseUrl.replace(/\/?$/, '/');
  return entry.pathSuffix ? `${cleanBase}${entry.pathSuffix}` : cleanBase;
}

test.describe('Multi-Model Top-Bar Switcher · N-Way Navigation & Responsive Containment', () => {
  test('1. Root / N-way switcher navigation (click & keyboard ArrowDown/Enter) with non-blank WebGL2 & zero GL errors', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/`, { waitUntil: 'networkidle' });
    await waitForAppSettled(page, 'nano');
    await assertCanvasNonBlank(getAppCanvasLocator(page, 'nano'), 'Root Nano initial');

    // Click cycle across every registered visualizer: Nano -> TwoTower -> Diffusion -> Uni -> Dory -> Nano
    for (const targetId of FULL_TOUR_SEQUENCE) {
      const entry = MODEL_REGISTRY.find((e) => e.id === targetId)!;
      await navigateViaModelSwitcher(page, targetId, urlFor(ROOT_BASE_URL, targetId), {
        expectedHref: resolveModelHref('/', entry),
        mode: 'click',
      });
      await waitForAppSettled(page, targetId);
      await assertCanvasNonBlank(
        getAppCanvasLocator(page, targetId),
        `Root click hop -> ${targetId}`,
      );
      expect(
        await getAppGlErrorCount(page, targetId),
        `Root click hop -> ${targetId} GL errors`,
      ).toBe(0);
    }

    // Keyboard cycle (ArrowDown on trigger + Enter on focused target link): Nano -> TwoTower -> Diffusion -> Uni -> Dory -> Nano
    for (const targetId of FULL_TOUR_SEQUENCE) {
      const entry = MODEL_REGISTRY.find((e) => e.id === targetId)!;
      await navigateViaModelSwitcher(page, targetId, urlFor(ROOT_BASE_URL, targetId), {
        expectedHref: resolveModelHref('/', entry),
        mode: 'keyboard',
      });
      await waitForAppSettled(page, targetId);
      expect(
        await getAppGlErrorCount(page, targetId),
        `Root keyboard hop -> ${targetId} GL errors`,
      ).toBe(0);
    }

    guards.assertClean();
  });

  test('2. GitHub Pages /llm-viz/ subpath N-way switcher navigation (click & keyboard) with HTTP 200 assets', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(PAGES_BASE_URL, { waitUntil: 'networkidle' });
    await waitForAppSettled(page, 'nano');

    const initialJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const initialCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(initialJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(initialCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

    // Click cycle across all 5 apps under /llm-viz/
    for (const targetId of FULL_TOUR_SEQUENCE) {
      const entry = MODEL_REGISTRY.find((e) => e.id === targetId)!;
      await navigateViaModelSwitcher(page, targetId, urlFor(PAGES_BASE_URL, targetId), {
        expectedHref: resolveModelHref('/llm-viz/', entry),
        mode: 'click',
      });
      await waitForAppSettled(page, targetId);
      await assertCanvasNonBlank(
        getAppCanvasLocator(page, targetId),
        `Subpath /llm-viz/ click hop -> ${targetId}`,
      );
      expect(
        await getAppGlErrorCount(page, targetId),
        `Subpath click hop -> ${targetId} GL errors`,
      ).toBe(0);
    }

    // Keyboard cycle across all 5 apps under /llm-viz/
    for (const targetId of FULL_TOUR_SEQUENCE) {
      const entry = MODEL_REGISTRY.find((e) => e.id === targetId)!;
      await navigateViaModelSwitcher(page, targetId, urlFor(PAGES_BASE_URL, targetId), {
        expectedHref: resolveModelHref('/llm-viz/', entry),
        mode: 'keyboard',
      });
      await waitForAppSettled(page, targetId);
      expect(
        await getAppGlErrorCount(page, targetId),
        `Subpath keyboard hop -> ${targetId} GL errors`,
      ).toBe(0);
    }

    guards.assertClean();
  });

  test('3. Top-bar model switcher popover never clips across all 5 apps × 4 viewports (1440×900, 1280×800, 1024×800, 390×844)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    for (const entry of MODEL_REGISTRY) {
      await page.setViewportSize(REQUIRED_VIEWPORTS[0]);
      await page.goto(urlFor(ROOT_BASE_URL, entry.id), { waitUntil: 'networkidle' });
      await waitForAppSettled(page, entry.id);

      for (const vp of REQUIRED_VIEWPORTS) {
        await page.setViewportSize(vp);
        await waitForAppSettled(page, entry.id);
        await assertSwitcherMenuUnclipped(page, entry.id, `${entry.id} @ ${vp.width}×${vp.height}`);
      }
    }

    guards.assertClean();
  });
});
