import { inflateSync } from 'node:zlib';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';
import type { UniSpatialDebugHandle, UniSpatialDebugSnapshot } from '../../src/uni/app/types';
import type { UniSpatialViewMode } from '../../src/uni/scene/types';

declare global {
  interface Window {
    __uniSpatialDebug?: UniSpatialDebugHandle;
    __testSliderRef?: HTMLInputElement | null;
    __testControlsSlotRef?: HTMLElement | null;
    __testFirstSliderRef?: HTMLInputElement | null;
  }
}

interface RuntimeGuards {
  readonly pageErrors: string[];
  readonly consoleErrors: string[];
  readonly failedRequests: string[];
  readonly httpErrorResponses: string[];
  readonly loadedJsAssets: string[];
  readonly loadedCssAssets: string[];
  assertClean: () => void;
}

interface RectBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

function attachRuntimeGuards(page: Page): RuntimeGuards {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  const failedRequests: string[] = [];
  const httpErrorResponses: string[] = [];
  const loadedJsAssets: string[] = [];
  const loadedCssAssets: string[] = [];

  page.on('pageerror', (err) => {
    pageErrors.push(err.stack ?? err.message);
  });
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('requestfailed', (req) => {
    failedRequests.push(`${req.method()} ${req.url()} (${req.failure()?.errorText ?? 'failed'})`);
  });
  page.on('response', (res) => {
    const status = res.status();
    const url = res.url();
    if (status >= 400) {
      httpErrorResponses.push(`${status} ${url}`);
    } else if (status === 200) {
      const cleanUrl = url.split('?')[0];
      if (cleanUrl.endsWith('.js')) loadedJsAssets.push(url);
      if (cleanUrl.endsWith('.css')) loadedCssAssets.push(url);
    }
  });

  return {
    pageErrors,
    consoleErrors,
    failedRequests,
    httpErrorResponses,
    loadedJsAssets,
    loadedCssAssets,
    assertClean: () => {
      expect(pageErrors, `Uncaught page errors: ${pageErrors.join('\n')}`).toEqual([]);
      expect(consoleErrors, `Console errors: ${consoleErrors.join('\n')}`).toEqual([]);
      expect(failedRequests, `Failed network requests: ${failedRequests.join('\n')}`).toEqual([]);
      expect(httpErrorResponses, `HTTP >= 400 responses: ${httpErrorResponses.join('\n')}`).toEqual(
        [],
      );
    },
  };
}

async function waitFrames(page: Page, count = 4): Promise<void> {
  await page.evaluate(
    (n) =>
      new Promise<void>((resolve) => {
        let remaining = n;
        const tick = () => {
          remaining -= 1;
          if (remaining <= 0) resolve();
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
    count,
  );
}

async function waitForUniSettled(page: Page): Promise<void> {
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const dbg = window.__uniSpatialDebug;
          return dbg ? !dbg.rafActive : false;
        }),
      { timeout: 6000 },
    )
    .toBe(true);
  await waitFrames(page, 1);
}

async function waitForTwoTowerSettled(page: Page): Promise<void> {
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const tel = (
            window as unknown as { __TWOTOWER_TELEMETRY__?: { readonly rafActive?: boolean } }
          ).__TWOTOWER_TELEMETRY__;
          return tel?.rafActive;
        }),
      { timeout: 6000 },
    )
    .toBe(false);
  await waitFrames(page, 1);
}

async function waitForDiffusionSettled(page: Page): Promise<void> {
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const dbg = (
            window as unknown as {
              __diffusionSpatialDebug?: { readonly rafActive?: boolean; readonly frameCount?: number };
            }
          ).__diffusionSpatialDebug;
          return dbg ? !dbg.rafActive && (dbg.frameCount ?? 0) > 0 : false;
        }),
      { timeout: 6000 },
    )
    .toBe(true);
  await waitFrames(page, 1);
}

async function getUniDebug(page: Page): Promise<UniSpatialDebugSnapshot> {
  const snapshot = await page.evaluate(() => {
    const d = window.__uniSpatialDebug;
    return d ? d.getSnapshot() : null;
  });
  expect(snapshot, 'window.__uniSpatialDebug must be populated').not.toBeNull();
  return snapshot!;
}

/** Pure-Node PNG RGBA pixel variance inspector (zero external image deps). */
function analyzePngColorVariety(png: Buffer): {
  width: number;
  height: number;
  distinctBuckets: number;
  nonBackgroundRatio: number;
} {
  const PNG_SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (png.length < 33 || !png.subarray(0, 8).equals(PNG_SIG)) {
    throw new Error('Invalid PNG buffer');
  }

  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatChunks: Buffer[] = [];

  let offset = 8;
  while (offset + 12 <= png.length) {
    const len = png.readUInt32BE(offset);
    const type = png.subarray(offset + 4, offset + 8).toString('ascii');
    const dataStart = offset + 8;
    const dataEnd = dataStart + len;
    if (dataEnd > png.length) break;

    if (type === 'IHDR') {
      width = png.readUInt32BE(dataStart);
      height = png.readUInt32BE(dataStart + 4);
      bitDepth = png[dataStart + 8];
      colorType = png[dataStart + 9];
    } else if (type === 'IDAT') {
      idatChunks.push(png.subarray(dataStart, dataEnd));
    } else if (type === 'IEND') {
      break;
    }
    offset = dataEnd + 4;
  }

  if (bitDepth !== 8 || (colorType !== 2 && colorType !== 6)) {
    throw new Error(`Unsupported PNG format: bitDepth=${bitDepth} colorType=${colorType}`);
  }

  const bpp = colorType === 6 ? 4 : 3;
  const stride = width * bpp;
  const raw = inflateSync(Buffer.concat(idatChunks));
  const recon = Buffer.alloc(height * stride);

  const paeth = (a: number, b: number, c: number): number => {
    const p = a + b - c;
    const pa = Math.abs(p - a);
    const pb = Math.abs(p - b);
    const pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  };

  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const srcRow = y * (stride + 1) + 1;
    const dstRow = y * stride;
    for (let x = 0; x < stride; x++) {
      const val = raw[srcRow + x];
      const a = x >= bpp ? recon[dstRow + x - bpp] : 0;
      const b = y > 0 ? recon[dstRow - stride + x] : 0;
      const c = y > 0 && x >= bpp ? recon[dstRow - stride + x - bpp] : 0;
      let out = val;
      if (filter === 1) out = (val + a) & 0xff;
      else if (filter === 2) out = (val + b) & 0xff;
      else if (filter === 3) out = (val + ((a + b) >> 1)) & 0xff;
      else if (filter === 4) out = (val + paeth(a, b, c)) & 0xff;
      recon[dstRow + x] = out;
    }
  }

  const buckets = new Set<number>();
  let sampled = 0;
  let nonBg = 0;
  const bgR = recon[0];
  const bgG = recon[1];
  const bgB = recon[2];
  const stepY = height > 96 ? 4 : 2;
  const stepX = width > 96 ? 4 : 2;

  for (let y = 0; y < height; y += stepY) {
    const rowOff = y * stride;
    for (let x = 0; x < width; x += stepX) {
      const i = rowOff + x * bpp;
      const r = recon[i];
      const g = recon[i + 1];
      const b = recon[i + 2];
      sampled += 1;
      buckets.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
      if (Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB) > 20) {
        nonBg += 1;
      }
    }
  }

  return {
    width,
    height,
    distinctBuckets: buckets.size,
    nonBackgroundRatio: sampled > 0 ? nonBg / sampled : 0,
  };
}

async function assertCanvasNonBlank(canvas: Locator, label: string): Promise<void> {
  const png = await canvas.screenshot({ animations: 'allow' });
  const stats = analyzePngColorVariety(png);
  expect(stats.width, `${label}: canvas PNG width`).toBeGreaterThan(200);
  expect(stats.height, `${label}: canvas PNG height`).toBeGreaterThan(150);
  expect(
    stats.distinctBuckets,
    `${label}: expected non-uniform WebGL2 pixels (buckets=${stats.distinctBuckets})`,
  ).toBeGreaterThanOrEqual(8);
  expect(
    stats.nonBackgroundRatio,
    `${label}: expected visible 3D geometry above background (ratio=${stats.nonBackgroundRatio.toFixed(4)})`,
  ).toBeGreaterThan(0.01);
}

async function assertRgbCanvasNonBlank(canvas: Locator, label: string): Promise<void> {
  const png = await canvas.screenshot({ animations: 'allow' });
  const stats = analyzePngColorVariety(png);
  expect(stats.width, `${label}: RGB inset width`).toBeGreaterThanOrEqual(32);
  expect(stats.height, `${label}: RGB inset height`).toBeGreaterThanOrEqual(24);
  expect(
    stats.distinctBuckets,
    `${label}: expected non-uniform RGB inset pixels (buckets=${stats.distinctBuckets})`,
  ).toBeGreaterThanOrEqual(3);
  expect(
    stats.nonBackgroundRatio,
    `${label}: expected visible RGB inset variation (ratio=${stats.nonBackgroundRatio.toFixed(4)})`,
  ).toBeGreaterThan(0.01);
}

function assertCleanUiStrings(text: string | null, context: string): void {
  const safe = text ?? '';
  expect(safe, `${context}: must never render "Infinity"`).not.toMatch(/Infinity/i);
  expect(safe, `${context}: must never render "∞"`).not.toContain('∞');
  expect(safe, `${context}: must never render "NaN"`).not.toMatch(/\bNaN\b/);
  expect(safe, `${context}: must never render "undefined"`).not.toMatch(/\bundefined\b/);
}

function assertNoBoundingBoxOverlap(
  a: RectBox,
  b: RectBox,
  labelA: string,
  labelB: string,
  context: string,
): void {
  const separated =
    a.x + a.width <= b.x + 1 ||
    b.x + b.width <= a.x + 1 ||
    a.y + a.height <= b.y + 1 ||
    b.y + b.height <= a.y + 1;
  expect(
    separated,
    `${context}: "${labelA}" (${JSON.stringify(a)}) must not overlap "${labelB}" (${JSON.stringify(b)})`,
  ).toBe(true);
}

const ALL_UNI_VIEWS: readonly UniSpatialViewMode[] = [
  'macro',
  'visual_input',
  'joint_attn',
  'ball_tree',
  'refinement',
  'output_edit',
  'opd_readonly',
] as const;

test.describe('Nemotron-Labs-Diffusion-Uni · Full Chromium WebGL2 E2E Suite', () => {
  test('1. Boot /uni/ Cleanly in Chromium WebGL2 with Zero GL Errors & Idle-RAF Settled', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    await expect(page.getByTestId('uni-app')).toBeVisible();
    await expect(page.getByTestId('uni-app-shell')).toBeVisible();
    await expect(page.getByTestId('uni-viewport-host')).toBeVisible();
    const canvas = page.getByTestId('uni-webgl-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.getByTestId('uni-webgl-fallback')).toBeHidden();

    await assertCanvasNonBlank(canvas, 'Root /uni/ Macro initial frame');

    const initDbg = await getUniDebug(page);
    expect(initDbg.glErrorCount).toBe(0);
    expect(initDbg.viewMode).toBe('macro');
    expect(initDbg.rafActive).toBe(false);
    expect(initDbg.tourBeatCount).toBe(28);
    expect(initDbg.overlayCellNumNodes).toBeLessThanOrEqual(64);
    expect(initDbg.memoryAudit.withinBudget).toBe(true);

    // Strict idle-RAF sleep across a 250ms rest window
    await page.waitForTimeout(250);
    const idleDbg = await getUniDebug(page);
    expect(idleDbg.rafActive).toBe(false);
    expect(idleDbg.glErrorCount).toBe(0);

    assertCleanUiStrings(await page.locator('body').textContent(), 'Root /uni/ initial body');

    expect(guards.loadedJsAssets.length).toBeGreaterThan(0);
    expect(guards.loadedCssAssets.length).toBeGreaterThan(0);
    guards.assertClean();
  });

  test('2. All 7 Spatial Views + RG32F Atlas Upload Deduplication', async ({ page }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    const canvas = page.getByTestId('uni-webgl-canvas');

    for (const view of ALL_UNI_VIEWS) {
      await page.getByTestId(`uni-view-tab-${view}`).click();
      await waitForUniSettled(page);

      const dbg = await getUniDebug(page);
      expect(dbg.viewMode).toBe(view);
      expect(dbg.glErrorCount).toBe(0);
      expect(dbg.overlayCellNumNodes).toBeLessThanOrEqual(64);
      expect(dbg.rafActive).toBe(false);

      await assertCanvasNonBlank(canvas, `View ${view}`);
      assertCleanUiStrings(
        await page.locator('.uni-stage-title-card').textContent(),
        `Stage title card on ${view}`,
      );
      assertCleanUiStrings(
        await page.getByTestId('uni-inspector-body').textContent(),
        `Inspector body on ${view}`,
      );
    }

    // Verify RG32F atlas upload deduplication on View 3 (`joint_attn`)
    await page.getByTestId('uni-view-tab-joint_attn').click();
    await waitForUniSettled(page);

    const beforeHover = await getUniDebug(page);
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
    await page.mouse.move(cx + 90, cy - 10, { steps: 5 });
    await page.mouse.up();
    await waitForUniSettled(page);

    const afterOrbit = await getUniDebug(page);
    expect(afterOrbit.atlasKey).toBe(beforeHover.atlasKey);
    expect(afterOrbit.atlasUploadCount).toBe(beforeHover.atlasUploadCount);
    expect(afterOrbit.glErrorCount).toBe(0);

    // Re-triggering identical view/atlasKey without changing data must NOT increment atlasUploadCount
    await page.getByTestId('uni-view-tab-joint_attn').click();
    await waitForUniSettled(page);

    const afterReclick = await getUniDebug(page);
    expect(afterReclick.atlasKey).toBe(beforeHover.atlasKey);
    expect(afterReclick.atlasUploadCount).toBe(beforeHover.atlasUploadCount);

    // Visiting View 1 (`macro`, atlasKey === null) and returning to identical `joint_attn` atlasKey
    await page.getByTestId('uni-view-tab-macro').click();
    await waitForUniSettled(page);
    expect((await getUniDebug(page)).atlasUploadCount).toBe(beforeHover.atlasUploadCount);

    await page.getByTestId('uni-view-tab-joint_attn').click();
    await waitForUniSettled(page);
    const afterRoundTrip = await getUniDebug(page);
    expect(afterRoundTrip.atlasKey).toBe(beforeHover.atlasKey);
    expect(afterRoundTrip.atlasUploadCount).toBe(beforeHover.atlasUploadCount);
    expect(afterRoundTrip.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('3. Live D5 Prefill Toggle (hf_faithful vs elastic_pruned) in Views 1 & 3', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    const canvas = page.getByTestId('uni-webgl-canvas');
    const dockHf = page.getByTestId('uni-dock-prefill-hf_faithful');
    const dockPruned = page.getByTestId('uni-dock-prefill-elastic_pruned');
    const inspHf = page.getByTestId('uni-inspector-d5-hf');
    const inspPruned = page.getByTestId('uni-inspector-d5-pruned');

    // View 1 (Macro): dock toggle & inspector toggle
    expect((await getUniDebug(page)).prefillMode).toBe('elastic_pruned');

    await dockHf.click();
    await waitForUniSettled(page);
    let dbg = await getUniDebug(page);
    expect(dbg.viewMode).toBe('macro');
    expect(dbg.prefillMode).toBe('hf_faithful');
    expect(dbg.glErrorCount).toBe(0);
    await expect(dockHf).toHaveAttribute('aria-pressed', 'true');
    await expect(inspHf).toHaveClass(/is-active/);
    await assertCanvasNonBlank(canvas, 'View 1 Macro HF-Faithful');

    await dockPruned.click();
    await waitForUniSettled(page);
    dbg = await getUniDebug(page);
    expect(dbg.prefillMode).toBe('elastic_pruned');
    expect(dbg.glErrorCount).toBe(0);
    await expect(dockPruned).toHaveAttribute('aria-pressed', 'true');
    await expect(inspPruned).toHaveClass(/is-active/);

    await inspHf.click();
    await waitForUniSettled(page);
    dbg = await getUniDebug(page);
    expect(dbg.prefillMode).toBe('hf_faithful');
    expect(dbg.glErrorCount).toBe(0);

    await inspPruned.click();
    await waitForUniSettled(page);
    dbg = await getUniDebug(page);
    expect(dbg.prefillMode).toBe('elastic_pruned');
    expect(dbg.glErrorCount).toBe(0);

    // View 3 (Joint / Split Attention): dock toggle & inspector toggle
    await page.getByTestId('uni-view-tab-joint_attn').click();
    await waitForUniSettled(page);

    await dockHf.click();
    await waitForUniSettled(page);
    dbg = await getUniDebug(page);
    expect(dbg.viewMode).toBe('joint_attn');
    expect(dbg.prefillMode).toBe('hf_faithful');
    expect(dbg.glErrorCount).toBe(0);
    await expect(dockHf).toHaveAttribute('aria-pressed', 'true');
    await assertCanvasNonBlank(canvas, 'View 3 Joint Attn HF-Faithful');

    await inspPruned.click();
    await waitForUniSettled(page);
    dbg = await getUniDebug(page);
    expect(dbg.prefillMode).toBe('elastic_pruned');
    expect(dbg.glErrorCount).toBe(0);
    await expect(dockPruned).toHaveAttribute('aria-pressed', 'true');
    await assertCanvasNonBlank(canvas, 'View 3 Joint Attn Elastic-Pruned');

    await inspHf.click();
    await waitForUniSettled(page);
    dbg = await getUniDebug(page);
    expect(dbg.prefillMode).toBe('hf_faithful');
    expect(dbg.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('4. View 4 U26 D = 17 Mismatch Panel & View 7 READ-ONLY · NO GRADIENTS Banner', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    // View 4: Ball-Tree Geometry & U26 D = 17 Floor-Rounding Mismatch Panel
    await page.getByTestId('uni-view-tab-ball_tree').click();
    await waitForUniSettled(page);

    const u26Panel = page.getByTestId('uni-u26-mismatch-panel');
    await expect(u26Panel).toBeVisible();
    await expect(u26Panel).toContainText('U26');
    await expect(u26Panel).toContainText('D=17');
    await expect(page.getByTestId('uni-opd-readonly-banner')).toBeHidden();
    assertCleanUiStrings(await u26Panel.textContent(), 'View 4 U26 Mismatch Panel');

    // View 7: Read-Only OPD Explorer Banner & Diagnostics Panel
    await page.getByTestId('uni-view-tab-opd_readonly').click();
    await waitForUniSettled(page);

    const opdBanner = page.getByTestId('uni-opd-readonly-banner');
    await expect(opdBanner).toBeVisible();
    await expect(opdBanner).toContainText('READ-ONLY · NO GRADIENTS');

    const opdPanel = page.getByTestId('uni-opd-diagnostics-panel');
    await expect(opdPanel).toBeVisible();
    await expect(opdPanel).toContainText('Gradients');
    await expect(opdPanel).toContainText('false');
    assertCleanUiStrings(await opdPanel.textContent(), 'View 7 OPD Diagnostics Panel');

    const dbg = await getUniDebug(page);
    expect(dbg.viewMode).toBe('opd_readonly');
    expect(dbg.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('5. 2D RGB Canvas Pixel Insets on Views 2, 5 & 6 + U18 Lock & <LOC_*> Grounding Pills', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    const pixelStrip = page.getByTestId('uni-pixel-strip');
    const pixelCanvases = pixelStrip.locator('canvas[data-testid^="uni-pixel-canvas-"]');

    // Views 2, 5, and 6 under default T2I task: non-blank 2D RGB <canvas> insets
    for (const view of ['visual_input', 'refinement', 'output_edit'] as const) {
      await page.getByTestId(`uni-view-tab-${view}`).click();
      await waitForUniSettled(page);

      await expect(pixelStrip).toBeVisible();
      const count = await pixelCanvases.count();
      expect(count, `Expected >= 1 RGB inset canvas on view ${view}`).toBeGreaterThanOrEqual(1);
      for (let i = 0; i < count; i++) {
        await assertRgbCanvasNonBlank(pixelCanvases.nth(i), `View ${view} RGB inset #${i}`);
      }
      expect((await getUniDebug(page)).glErrorCount).toBe(0);
    }

    // Activate `image_edit` task -> verify U18 lock pill on Views 2 and 6
    await page.getByTestId('uni-dock-task-image_edit').click();
    await waitForUniSettled(page);
    expect((await getUniDebug(page)).taskMode).toBe('image_edit');

    const lockPills = pixelStrip.locator('[data-testid^="uni-pixel-lock-pill-"]');
    expect(await lockPills.count(), 'Expected U18 lock pill(s) on View 6 in image_edit').toBeGreaterThanOrEqual(1);
    await expect(lockPills.first()).toBeVisible();
    await expect(lockPills.first()).toContainText('U18 Lock');
    await assertRgbCanvasNonBlank(pixelCanvases.first(), 'View 6 image_edit RGB inset');

    await page.getByTestId('uni-view-tab-visual_input').click();
    await waitForUniSettled(page);
    expect(await lockPills.count(), 'Expected U18 lock pill(s) on View 2 in image_edit').toBeGreaterThanOrEqual(1);
    await expect(lockPills.first()).toBeVisible();
    await expect(lockPills.first()).toContainText('U18 Lock');
    await assertRgbCanvasNonBlank(pixelCanvases.first(), 'View 2 image_edit RGB inset');

    // Activate `object_grounding` task -> verify <LOC_*> grounding pill on View 6
    await page.getByTestId('uni-dock-task-grounding').click();
    await waitForUniSettled(page);
    expect((await getUniDebug(page)).taskMode).toBe('object_grounding');

    await page.getByTestId('uni-view-tab-output_edit').click();
    await waitForUniSettled(page);

    const locPills = pixelStrip.locator('[data-testid^="uni-pixel-loc-pill-"]');
    expect(
      await locPills.count(),
      'Expected <LOC_*> grounding pill(s) on View 6 in object_grounding',
    ).toBeGreaterThanOrEqual(1);
    await expect(locPills.first()).toBeVisible();
    await expect(locPills.first()).toContainText('<LOC_');
    await assertRgbCanvasNonBlank(pixelCanvases.first(), 'View 6 object_grounding RGB inset');

    expect((await getUniDebug(page)).glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('6. UNI6b Interactive Regressions: In-Place Slider DOM Identity & Hover Controls-Slot Preservation', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    // 1. View 3 (`joint_attn`): `uni-layer-slider` preserves exact DOM node & focus on 17 -> 18 input
    await page.getByTestId('uni-view-tab-joint_attn').click();
    await waitForUniSettled(page);

    const layerSlider = page.getByTestId('uni-layer-slider');
    await expect(layerSlider).toBeVisible();
    await layerSlider.focus();

    await page.evaluate(() => {
      const el = document.querySelector<HTMLInputElement>('[data-testid="uni-layer-slider"]');
      window.__testSliderRef = el;
      if (!el) throw new Error('Missing uni-layer-slider');
      el.value = '17';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFrames(page, 2);

    await page.evaluate(() => {
      const el = window.__testSliderRef;
      if (!el) throw new Error('Missing saved layer slider reference');
      el.value = '18';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForUniSettled(page);

    const layerSliderCheck = await page.evaluate(() => {
      const ref = window.__testSliderRef;
      const current = document.querySelector<HTMLInputElement>('[data-testid="uni-layer-slider"]');
      return {
        hasRef: Boolean(ref),
        isConnected: Boolean(ref?.isConnected),
        isActiveElement: ref !== null && ref === document.activeElement,
        sameDomNode: ref !== null && ref === current,
        value: ref?.value ?? null,
      };
    });
    expect(layerSliderCheck).toEqual({
      hasRef: true,
      isConnected: true,
      isActiveElement: true,
      sameDomNode: true,
      value: '18',
    });

    // 2. View 4 (`ball_tree`): `uni-code-slider` preserves exact DOM node & focus on input
    await page.getByTestId('uni-view-tab-ball_tree').click();
    await waitForUniSettled(page);

    const codeSlider = page.getByTestId('uni-code-slider');
    await expect(codeSlider).toBeVisible();
    await codeSlider.focus();

    await page.evaluate(() => {
      const el = document.querySelector<HTMLInputElement>('[data-testid="uni-code-slider"]');
      window.__testSliderRef = el;
      if (!el) throw new Error('Missing uni-code-slider');
      el.value = '99';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForUniSettled(page);

    const codeSliderCheck = await page.evaluate(() => {
      const ref = window.__testSliderRef;
      const current = document.querySelector<HTMLInputElement>('[data-testid="uni-code-slider"]');
      return {
        hasRef: Boolean(ref),
        isConnected: Boolean(ref?.isConnected),
        isActiveElement: ref !== null && ref === document.activeElement,
        sameDomNode: ref !== null && ref === current,
        value: ref?.value ?? null,
      };
    });
    expect(codeSliderCheck).toEqual({
      hasRef: true,
      isConnected: true,
      isActiveElement: true,
      sameDomNode: true,
      value: '99',
    });

    // 3. Hovering across `[data-testid="uni-webgl-canvas"]` NEVER replaces controls slot or its slider
    await page.evaluate(() => {
      const controlsSlot = document.querySelector<HTMLElement>(
        '[data-testid="uni-inspector-controls-slot"]',
      );
      const firstSlider = controlsSlot?.querySelector<HTMLInputElement>('input[type="range"]') ?? null;
      window.__testControlsSlotRef = controlsSlot;
      window.__testFirstSliderRef = firstSlider;
    });

    const cellSlot = page.getByTestId('uni-inspector-cell-slot');
    await expect(cellSlot).toBeVisible();

    const canvas = page.getByTestId('uni-webgl-canvas');
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
      await page.mouse.move(canvasBox!.x + canvasBox!.width * fx, canvasBox!.y + canvasBox!.height * fy);
      await waitFrames(page, 2);
    }
    await waitForUniSettled(page);

    await expect(cellSlot).toBeVisible();
    const hoverPreserveCheck = await page.evaluate(() => {
      const curSlot = document.querySelector<HTMLElement>(
        '[data-testid="uni-inspector-controls-slot"]',
      );
      const curSlider = curSlot?.querySelector<HTMLInputElement>('input[type="range"]') ?? null;
      const curCellSlot = document.querySelector<HTMLElement>(
        '[data-testid="uni-inspector-cell-slot"]',
      );
      return {
        controlsSlotPreserved:
          window.__testControlsSlotRef !== null && curSlot === window.__testControlsSlotRef,
        firstSliderPreserved:
          window.__testFirstSliderRef !== null &&
          curSlider === window.__testFirstSliderRef &&
          Boolean(curSlider?.isConnected),
        cellSlotMounted: Boolean(curCellSlot?.isConnected),
      };
    });
    expect(hoverPreserveCheck).toEqual({
      controlsSlotPreserved: true,
      firstSliderPreserved: true,
      cellSlotMounted: true,
    });

    expect((await getUniDebug(page)).glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('7. 28-Beat Guided Tour + 1280×800 Zero-Overlap Layout + Escape Deferral + Free-Explore Restoration', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(`${ROOT_BASE_URL}/uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    // Ensure inspector drawer is open at 1280x800
    const drawer = page.getByTestId('uni-inspector-drawer');
    await expect(drawer).toBeVisible();
    await expect(drawer).not.toHaveClass(/is-collapsed/);

    // Establish distinct Free-Explore state before launching tour
    await page.getByTestId('uni-view-tab-joint_attn').click();
    await waitForUniSettled(page);
    await page.getByTestId('uni-dock-step-back').click();
    await waitForUniSettled(page);

    const preTour = await getUniDebug(page);
    expect(preTour.viewMode).toBe('joint_attn');
    expect(preTour.activeRecordId).not.toBeNull();
    expect(preTour.tourActive).toBe(false);

    // Launch 28-beat Guided Tour
    await page.getByTestId('uni-tour-launch-btn').click();
    await waitForUniSettled(page);

    const tourHud = page.getByTestId('uni-tour-hud');
    await expect(tourHud).toBeVisible();
    let dbg = await getUniDebug(page);
    expect(dbg.tourActive).toBe(true);
    expect(dbg.tourBeatIndex).toBe(0);
    expect(dbg.tourBeatCount).toBe(28);

    // Verify Next / Prev controls work cleanly
    await page.getByTestId('uni-tour-next-btn').click();
    await waitForUniSettled(page);
    expect((await getUniDebug(page)).tourBeatIndex).toBe(1);
    await page.getByTestId('uni-tour-prev-btn').click();
    await waitForUniSettled(page);
    expect((await getUniDebug(page)).tourBeatIndex).toBe(0);

    // Representative beats across all 5 Acts
    const representativeBeats = [
      { idx: 0, expectedView: 'macro' as const, label: 'Beat 1 (Act 1 · Macro)' },
      { idx: 5, expectedView: 'visual_input' as const, label: 'Beat 6 (Act 2 · Visual Input)' },
      { idx: 11, expectedView: 'joint_attn' as const, label: 'Beat 12 (Act 3 · L17 Joint)' },
      { idx: 18, expectedView: 'ball_tree' as const, label: 'Beat 19 (Act 4 · U26 D=17)' },
      { idx: 19, expectedView: 'refinement' as const, label: 'Beat 20 (Act 4 · 4×6 VQ Canvas)' },
      { idx: 21, expectedView: 'output_edit' as const, label: 'Beat 22 (Act 4 · Dual-CFG Edit)' },
      { idx: 24, expectedView: 'output_edit' as const, label: 'Beat 25 (Act 5 · <LOC> Grounding)' },
      { idx: 25, expectedView: 'opd_readonly' as const, label: 'Beat 26 (Act 5 · OPD Read-Only)' },
    ] as const;

    const titleCard = page.locator('.uni-stage-title-card');
    const pixelStrip = page.getByTestId('uni-pixel-strip');
    const opdBanner = page.getByTestId('uni-opd-readonly-banner');

    for (const beat of representativeBeats) {
      if (beat.idx !== 0) {
        await page.evaluate((idx) => {
          window.__uniSpatialDebug?.goToBeat(idx);
        }, beat.idx);
        await waitForUniSettled(page);
      }

      const beatDbg = await getUniDebug(page);
      expect(beatDbg.tourActive).toBe(true);
      expect(beatDbg.tourBeatIndex).toBe(beat.idx);
      expect(beatDbg.viewMode).toBe(beat.expectedView);
      expect(beatDbg.glErrorCount).toBe(0);
      assertCleanUiStrings(await tourHud.textContent(), beat.label);

      // Strict zero bounding-box overlap check across visible HUD elements at 1280×800
      const hudBox = await tourHud.boundingBox();
      const titleBox = await titleCard.boundingBox();
      expect(hudBox, `${beat.label}: tourHud boundingBox`).not.toBeNull();
      expect(titleBox, `${beat.label}: stageTitleCard boundingBox`).not.toBeNull();

      const activeBoxes: { readonly label: string; readonly box: RectBox }[] = [
        { label: 'uni-tour-hud', box: hudBox! },
        { label: '.uni-stage-title-card', box: titleBox! },
      ];

      if (await pixelStrip.isVisible()) {
        const stripBox = await pixelStrip.boundingBox();
        if (stripBox && stripBox.width > 0 && stripBox.height > 0) {
          activeBoxes.push({ label: 'uni-pixel-strip', box: stripBox });
        }
      }

      if (await opdBanner.isVisible()) {
        const bannerBox = await opdBanner.boundingBox();
        if (bannerBox && bannerBox.width > 0 && bannerBox.height > 0) {
          activeBoxes.push({ label: 'uni-opd-readonly-banner', box: bannerBox });
        }
      }

      for (let i = 0; i < activeBoxes.length; i++) {
        for (let j = i + 1; j < activeBoxes.length; j++) {
          assertNoBoundingBoxOverlap(
            activeBoxes[i].box,
            activeBoxes[j].box,
            activeBoxes[i].label,
            activeBoxes[j].label,
            `${beat.label} @ 1280×800`,
          );
        }
      }

      // Check dock + inspector lock on Beat 19 (View 4, which has sliders, buttons, and checkbox)
      if (beat.idx === 18) {
        for (const dockTestId of [
          'uni-dock-step-back',
          'uni-dock-step-forward',
          'uni-dock-scrubber',
          'uni-dock-play-pause',
          'uni-dock-prefill-hf_faithful',
          'uni-dock-prefill-elastic_pruned',
          'uni-dock-execute-step',
        ] as const) {
          await expect(
            page.getByTestId(dockTestId),
            `${dockTestId} must be disabled during tour`,
          ).toBeDisabled();
        }

        await expect(page.getByTestId('uni-tour-lock-note')).toBeVisible();
        const unlockedControls = await page
          .getByTestId('uni-inspector-controls-slot')
          .locator('button:not([disabled]), input:not([disabled]), select:not([disabled])')
          .count();
        expect(
          unlockedControls,
          'All inspector view controls must be disabled during tour',
        ).toBe(0);
      }
    }

    // Click a `.uni-tour-ref-chip` -> switches inspector to Ledger tab and populates search
    const firstRefChip = tourHud.locator('.uni-tour-ref-chip').first();
    await expect(firstRefChip).toBeVisible();
    await firstRefChip.click();

    await expect(page.getByTestId('uni-inspector-tab-ledger')).toHaveAttribute(
      'aria-selected',
      'true',
    );
    const ledgerSearch = page.getByTestId('uni-ledger-search');
    await expect(ledgerSearch).toBeVisible();
    expect((await ledgerSearch.inputValue()).length).toBeGreaterThan(0);

    // Focus search input, type U26, press Escape -> must DEFER (tour stays active!)
    await ledgerSearch.focus();
    await ledgerSearch.fill('U26');
    await expect(ledgerSearch).toBeFocused();
    await expect(page.getByTestId('uni-ledger-card-U26')).toBeVisible();

    await page.keyboard.press('Escape');
    await waitFrames(page, 2);
    dbg = await getUniDebug(page);
    expect(dbg.tourActive, 'Escape inside ledger search input must NOT exit tour').toBe(true);
    await expect(tourHud).toBeVisible();

    // Blur search input and press Escape -> exits tour and restores Free-Explore state untouched
    await ledgerSearch.evaluate((el) => {
      (el as HTMLElement).blur();
    });
    await page.keyboard.press('Escape');
    await waitForUniSettled(page);

    await expect(tourHud).toBeHidden();
    const restored = await getUniDebug(page);
    expect(restored.tourActive).toBe(false);
    expect(restored.viewMode).toBe(preTour.viewMode);
    expect(restored.activeRecordId).toBe(preTour.activeRecordId);
    expect(restored.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('8. 4-Way Reciprocal Navigation on Root / and GitHub Pages Subpath /llm-viz/', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);
    const uniCanvas = page.getByTestId('uni-webgl-canvas');

    async function navViaSwitcher(targetTestId: string, expectedUrl: string): Promise<void> {
      const trigger = page.getByTestId('model-switcher-trigger');
      await expect(trigger).toBeVisible();
      await trigger.click();
      const link = page.getByTestId(targetTestId);
      await expect(link).toBeVisible();
      await Promise.all([page.waitForURL(expectedUrl), link.click()]);
    }

    // --- A. Root `/` 4-way reciprocal hops ---
    // Nano (`/`) -> Uni (`/uni/`)
    await page.goto(`${ROOT_BASE_URL}/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-uni', `${ROOT_BASE_URL}/uni/`);
    await waitForUniSettled(page);
    await assertCanvasNonBlank(uniCanvas, 'Root Nano -> Uni');
    expect((await getUniDebug(page)).glErrorCount).toBe(0);

    // Uni (`/uni/`) -> TwoTower (`/twotower/`) -> Uni (`/uni/`)
    await navViaSwitcher('nav-to-twotower', `${ROOT_BASE_URL}/twotower/`);
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('twotower-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-uni', `${ROOT_BASE_URL}/uni/`);
    await waitForUniSettled(page);
    await assertCanvasNonBlank(uniCanvas, 'Root TwoTower -> Uni');
    expect((await getUniDebug(page)).glErrorCount).toBe(0);

    // Uni (`/uni/`) -> Diffusion (`/diffusion/`) -> Uni (`/uni/`)
    await navViaSwitcher('nav-to-diffusion', `${ROOT_BASE_URL}/diffusion/`);
    await waitForDiffusionSettled(page);
    await expect(page.getByTestId('diffusion-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-uni', `${ROOT_BASE_URL}/uni/`);
    await waitForUniSettled(page);
    await assertCanvasNonBlank(uniCanvas, 'Root Diffusion -> Uni');
    expect((await getUniDebug(page)).glErrorCount).toBe(0);

    // Uni (`/uni/`) -> Nano (`/`)
    await navViaSwitcher('nav-to-nano', `${ROOT_BASE_URL}/`);
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    // --- B. GitHub Pages subpath `/llm-viz/` parity & 4-way links ---
    await page.goto(`${PAGES_BASE_URL}uni/`, { waitUntil: 'networkidle' });
    await waitForUniSettled(page);

    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(subpathCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

    await assertCanvasNonBlank(uniCanvas, 'Subpath /llm-viz/uni/ Macro');
    expect((await getUniDebug(page)).glErrorCount).toBe(0);

    await expect(page.getByTestId('nav-to-nano')).toHaveAttribute('href', '/llm-viz/');
    await expect(page.getByTestId('nav-to-twotower')).toHaveAttribute('href', '/llm-viz/twotower/');
    await expect(page.getByTestId('nav-to-diffusion')).toHaveAttribute(
      'href',
      '/llm-viz/diffusion/',
    );

    // Subpath reciprocal hops: Uni -> Nano -> Uni -> TwoTower -> Uni -> Diffusion -> Uni
    await navViaSwitcher('nav-to-nano', PAGES_BASE_URL);
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-uni', `${PAGES_BASE_URL}uni/`);
    await waitForUniSettled(page);
    await assertCanvasNonBlank(uniCanvas, 'Subpath Nano -> Uni');

    await navViaSwitcher('nav-to-twotower', `${PAGES_BASE_URL}twotower/`);
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('twotower-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-uni', `${PAGES_BASE_URL}uni/`);
    await waitForUniSettled(page);

    await navViaSwitcher('nav-to-diffusion', `${PAGES_BASE_URL}diffusion/`);
    await waitForDiffusionSettled(page);
    await expect(page.getByTestId('diffusion-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-uni', `${PAGES_BASE_URL}uni/`);
    await waitForUniSettled(page);
    await assertCanvasNonBlank(uniCanvas, 'Subpath Diffusion -> Uni');
    expect((await getUniDebug(page)).glErrorCount).toBe(0);

    guards.assertClean();
  });
});
