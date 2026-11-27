import { inflateSync } from 'node:zlib';
import { expect, type Locator, type Page } from '@playwright/test';
import { MODEL_REGISTRY, type ModelEntryId } from '../../src/shared/modelRegistry';

export interface RuntimeGuards {
  readonly pageErrors: string[];
  readonly consoleErrors: string[];
  readonly failedRequests: string[];
  readonly httpErrorResponses: string[];
  readonly loadedJsAssets: string[];
  readonly loadedCssAssets: string[];
  assertClean: () => void;
}

export function attachRuntimeGuards(page: Page): RuntimeGuards {
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

export async function waitFrames(page: Page, count = 4): Promise<void> {
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

/** Pure-Node PNG RGBA pixel variance inspector (zero external image deps). */
export function analyzePngColorVariety(
  png: Buffer,
  bgTolerance = 20,
): {
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
      if (Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB) > bgTolerance) {
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

export interface CanvasNonBlankOptions {
  readonly minWidth?: number;
  readonly minHeight?: number;
  readonly minBuckets?: number;
  readonly minNonBgRatio?: number;
  readonly bgTolerance?: number;
}

export async function assertCanvasNonBlank(
  canvas: Locator,
  label: string,
  opts: CanvasNonBlankOptions = {},
): Promise<void> {
  const {
    minWidth = 200,
    minHeight = 150,
    minBuckets = 8,
    minNonBgRatio = 0.01,
    bgTolerance = 20,
  } = opts;
  const png = await canvas.screenshot({ animations: 'allow' });
  const stats = analyzePngColorVariety(png, bgTolerance);
  expect(stats.width, `${label}: canvas PNG width`).toBeGreaterThanOrEqual(minWidth);
  expect(stats.height, `${label}: canvas PNG height`).toBeGreaterThanOrEqual(minHeight);
  expect(
    stats.distinctBuckets,
    `${label}: expected non-uniform rendered pixels (buckets=${stats.distinctBuckets})`,
  ).toBeGreaterThanOrEqual(minBuckets);
  expect(
    stats.nonBackgroundRatio,
    `${label}: expected visible rendered geometry above background (ratio=${stats.nonBackgroundRatio.toFixed(4)})`,
  ).toBeGreaterThan(minNonBgRatio);
}

export function assertNoInfinityText(text: string | null, context: string): void {
  const safe = text ?? '';
  expect(safe, `${context}: must never render "Infinity"`).not.toMatch(/Infinity/i);
  expect(safe, `${context}: must never render "∞"`).not.toContain('∞');
}

export function assertNoInfinityOrNaNText(text: string | null, context: string): void {
  assertNoInfinityText(text, context);
  const safe = text ?? '';
  expect(safe, `${context}: must never render "NaN"`).not.toMatch(/\bNaN\b/);
}

export function assertCleanUiStrings(text: string | null, context: string): void {
  assertNoInfinityOrNaNText(text, context);
  const safe = text ?? '';
  expect(safe, `${context}: must never render "undefined"`).not.toMatch(/\bundefined\b/);
}

export function getAppCanvasLocator(page: Page, appId: ModelEntryId): Locator {
  switch (appId) {
    case 'nano':
      return page.locator('#gl-canvas');
    case 'twotower':
      return page.getByTestId('twotower-canvas');
    case 'diffusion':
      return page.getByTestId('diffusion-canvas');
    case 'uni':
      return page.getByTestId('uni-webgl-canvas');
    case 'dory':
      return page.getByTestId('dory-webgl-canvas');
  }
}

export async function waitForAppSettled(page: Page, appId: ModelEntryId): Promise<void> {
  if (appId === 'nano') {
    await expect(page.locator('#gl-canvas')).toBeVisible();
    await waitFrames(page, 4);
    return;
  }
  if (appId === 'twotower') {
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
    return;
  }
  if (appId === 'diffusion') {
    await expect
      .poll(
        () =>
          page.evaluate(() => {
            const dbg = (
              window as unknown as {
                __diffusionSpatialDebug?: {
                  readonly rafActive?: boolean;
                  readonly frameCount?: number;
                };
              }
            ).__diffusionSpatialDebug;
            return dbg ? !dbg.rafActive && (dbg.frameCount ?? 0) > 0 : false;
          }),
        { timeout: 6000 },
      )
      .toBe(true);
    await waitFrames(page, 1);
    return;
  }
  if (appId === 'uni') {
    await expect
      .poll(
        () =>
          page.evaluate(() => {
            const dbg = (
              window as unknown as { __uniSpatialDebug?: { readonly rafActive?: boolean } }
            ).__uniSpatialDebug;
            return dbg ? !dbg.rafActive : false;
          }),
        { timeout: 6000 },
      )
      .toBe(true);
    await waitFrames(page, 1);
    return;
  }
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const dbg = (
            window as unknown as { __dorySpatialDebug?: { readonly rafActive?: boolean } }
          ).__dorySpatialDebug;
          return dbg ? !dbg.rafActive : false;
        }),
      { timeout: 6000 },
    )
    .toBe(true);
  await waitFrames(page, 1);
}

export async function getAppGlErrorCount(page: Page, appId: ModelEntryId): Promise<number> {
  if (appId === 'twotower') {
    return page.evaluate(
      () =>
        (window as unknown as { __TWOTOWER_TELEMETRY__?: { glErrorCount: number } })
          .__TWOTOWER_TELEMETRY__?.glErrorCount ?? -1,
    );
  }
  if (appId === 'diffusion') {
    return page.evaluate(
      () =>
        (window as unknown as { __diffusionSpatialDebug?: { glErrorCount: number } })
          .__diffusionSpatialDebug?.glErrorCount ?? -1,
    );
  }
  if (appId === 'uni') {
    return page.evaluate(
      () =>
        (
          window as unknown as {
            __uniSpatialDebug?: { getSnapshot: () => { glErrorCount: number } };
          }
        ).__uniSpatialDebug?.getSnapshot().glErrorCount ?? -1,
    );
  }
  if (appId === 'dory') {
    return page.evaluate(
      () =>
        (
          window as unknown as {
            __dorySpatialDebug?: { getSnapshot: () => { glErrorCount: number } };
          }
        ).__dorySpatialDebug?.getSnapshot().glErrorCount ?? -1,
    );
  }
  return 0;
}

export async function navigateViaModelSwitcher(
  page: Page,
  targetId: ModelEntryId,
  expectedUrl: string,
  opts: { readonly expectedHref?: string; readonly mode?: 'click' | 'keyboard' } = {},
): Promise<void> {
  const { expectedHref, mode = 'click' } = opts;
  const trigger = page.getByTestId('model-switcher-trigger');
  await expect(trigger).toBeVisible();
  if (mode === 'keyboard') {
    await trigger.focus();
    await page.keyboard.press('ArrowDown');
  } else {
    await trigger.click();
  }

  const menu = page.getByTestId('model-switcher-menu');
  await expect(menu).toBeVisible();

  const link = page.getByTestId(`nav-to-${targetId}`);
  await expect(link).toBeVisible();
  if (expectedHref !== undefined) {
    await expect(link).toHaveAttribute('href', expectedHref);
  }

  if (mode === 'keyboard') {
    await link.focus();
    await expect(link).toBeFocused();
    await Promise.all([page.waitForURL(expectedUrl), page.keyboard.press('Enter')]);
  } else {
    await Promise.all([page.waitForURL(expectedUrl), link.click()]);
  }
}

/**
 * Opens the shared top-bar model switcher at the current viewport and verifies:
 *  - Every registry entry is visible inside the popover with exactly one `aria-current="page"`.
 *  - The popover bounding box fits strictly inside the viewport (zero left/right/top/bottom clipping).
 *  - Zero document horizontal overflow while open (`scrollWidth - innerWidth <= 1`).
 *  - Pressing `Escape` closes the popover and restores focus to the trigger button.
 */
export async function assertSwitcherMenuUnclipped(
  page: Page,
  activeId: ModelEntryId,
  label: string,
): Promise<void> {
  const vp = page.viewportSize();
  expect(vp, `${label}: viewportSize must be defined`).not.toBeNull();

  const trigger = page.getByTestId('model-switcher-trigger');
  const menu = page.getByTestId('model-switcher-menu');
  await expect(trigger, `${label}: switcher trigger visible`).toBeVisible();
  await expect(trigger).toHaveAttribute('aria-expanded', 'false');
  await expect(menu).toBeHidden();

  await trigger.click();
  await expect(trigger).toHaveAttribute('aria-expanded', 'true');
  await expect(menu, `${label}: switcher menu visible`).toBeVisible();

  for (const entry of MODEL_REGISTRY) {
    const link = page.getByTestId(entry.testId);
    await expect(link, `${label}: link ${entry.testId} visible`).toBeVisible();
    if (entry.id === activeId) {
      await expect(link).toHaveAttribute('aria-current', 'page');
    } else {
      await expect(link).not.toHaveAttribute('aria-current', 'page');
    }
  }

  const box = await menu.boundingBox();
  expect(box, `${label}: menu boundingBox`).not.toBeNull();
  expect(box!.x, `${label}: popover left edge >= 0 (x=${box!.x})`).toBeGreaterThanOrEqual(0);
  expect(box!.y, `${label}: popover top edge >= 0 (y=${box!.y})`).toBeGreaterThanOrEqual(0);
  expect(
    box!.x + box!.width,
    `${label}: popover right edge <= ${vp!.width} (right=${box!.x + box!.width})`,
  ).toBeLessThanOrEqual(vp!.width + 0.5);
  expect(
    box!.y + box!.height,
    `${label}: popover bottom edge <= ${vp!.height} (bottom=${box!.y + box!.height})`,
  ).toBeLessThanOrEqual(vp!.height + 0.5);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(
    overflow,
    `${label}: zero horizontal overflow while switcher menu is open (overflow=${overflow}px)`,
  ).toBeLessThanOrEqual(1);

  await page.keyboard.press('Escape');
  await expect(menu).toBeHidden();
  await expect(trigger).toHaveAttribute('aria-expanded', 'false');
  await expect(trigger).toBeFocused();
}
