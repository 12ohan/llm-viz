import { inflateSync } from 'node:zlib';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';

interface WebGL2Telemetry {
  hasWebGL2: boolean;
  glErrors: number[];
  totalDrawCalls: number;
  linesPassProgramLinked: boolean;
  linesPassDraws: number;
  linesPassLastInstanceCount: number;
  linesPassLastTotalLineCount: number;
  linesPassLastGuideCount: number;
  rafCount: number;
  pendingRafCount: number;
}

declare global {
  interface Window {
    __webgl2Telemetry?: WebGL2Telemetry;
  }
}

interface PageErrorGuard {
  readonly pageErrors: string[];
  readonly consoleErrors: string[];
  assertClean: () => void;
}

function attachErrorGuards(page: Page): PageErrorGuard {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];

  page.on('pageerror', (err) => {
    pageErrors.push(err.stack ?? err.message);
  });
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  return {
    pageErrors,
    consoleErrors,
    assertClean: () => {
      expect(pageErrors, `Unexpected page errors: ${pageErrors.join('\n')}`).toEqual([]);
      expect(consoleErrors, `Unexpected console errors: ${consoleErrors.join('\n')}`).toEqual([]);
    },
  };
}

/**
 * Non-invasive WebGL2 & RAF telemetry probe injected before app boot.
 * Observes context creation, LinesPass guide-line uploads/draws (18-float stride, iStyle === 2),
 * WebGL errors (`gl.getError()`), and requestAnimationFrame counts without mutating app behavior.
 */
async function installWebGL2Telemetry(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const telemetry: WebGL2Telemetry = {
      hasWebGL2: false,
      glErrors: [],
      totalDrawCalls: 0,
      linesPassProgramLinked: false,
      linesPassDraws: 0,
      linesPassLastInstanceCount: 0,
      linesPassLastTotalLineCount: 0,
      linesPassLastGuideCount: 0,
      rafCount: 0,
      pendingRafCount: 0,
    };
    window.__webgl2Telemetry = telemetry;

    const pendingRafIds = new Set<number>();
    const origRaf = window.requestAnimationFrame.bind(window);
    const origCancelRaf = window.cancelAnimationFrame.bind(window);
    window.requestAnimationFrame = (cb: FrameRequestCallback): number => {
      let id = 0;
      id = origRaf((ts) => {
        pendingRafIds.delete(id);
        telemetry.pendingRafCount = pendingRafIds.size;
        telemetry.rafCount += 1;
        cb(ts);
      });
      pendingRafIds.add(id);
      telemetry.pendingRafCount = pendingRafIds.size;
      return id;
    };
    window.cancelAnimationFrame = (handle: number): void => {
      pendingRafIds.delete(handle);
      telemetry.pendingRafCount = pendingRafIds.size;
      origCancelRaf(handle);
    };

    const LINE_INSTANCE_FLOATS = 18;
    const LINE_STYLE_OFFSET = 7;
    const LINE_OWNER_OFFSET = 9;
    const LINE_STYLE_GUIDE = 2;

    const origGetContext = HTMLCanvasElement.prototype.getContext;
    const wrappedContexts = new WeakSet<WebGL2RenderingContext>();

    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      contextId: string,
      options?: unknown,
    ): RenderingContext | null {
      const ctx = origGetContext.call(
        this,
        contextId as never,
        options as never,
      ) as RenderingContext | null;
      if (contextId !== 'webgl2' || !ctx) {
        return ctx;
      }

      const gl = ctx as WebGL2RenderingContext;
      if (this.id === 'gl-canvas') {
        telemetry.hasWebGL2 = true;
      }
      if (wrappedContexts.has(gl)) {
        return ctx;
      }
      wrappedContexts.add(gl);

      const shaderSources = new WeakMap<WebGLShader, string>();
      const programShaders = new WeakMap<WebGLProgram, WebGLShader[]>();
      const linesPrograms = new WeakSet<WebGLProgram>();
      let currentProgram: WebGLProgram | null = null;
      let boundArrayBuffer: WebGLBuffer | null = null;
      const bufferLineStats = new WeakMap<WebGLBuffer, { total: number; guide: number }>();

      const origShaderSource = gl.shaderSource.bind(gl);
      gl.shaderSource = (shader: WebGLShader, source: string) => {
        shaderSources.set(shader, source);
        origShaderSource(shader, source);
      };

      const origAttachShader = gl.attachShader.bind(gl);
      gl.attachShader = (program: WebGLProgram, shader: WebGLShader) => {
        const list = programShaders.get(program) ?? [];
        list.push(shader);
        programShaders.set(program, list);
        origAttachShader(program, shader);
      };

      const origLinkProgram = gl.linkProgram.bind(gl);
      gl.linkProgram = (program: WebGLProgram) => {
        origLinkProgram(program);
        const shaders = programShaders.get(program) ?? [];
        const combinedSrc = shaders.map((sh) => shaderSources.get(sh) ?? '').join('\n');
        const isLinesPass =
          combinedSrc.includes('uGuideColor') && combinedSrc.includes('iCssWidth');
        if (isLinesPass && gl.getProgramParameter(program, gl.LINK_STATUS)) {
          linesPrograms.add(program);
          telemetry.linesPassProgramLinked = true;
        }
      };

      const origUseProgram = gl.useProgram.bind(gl);
      gl.useProgram = (program: WebGLProgram | null) => {
        currentProgram = program;
        origUseProgram(program);
      };

      const origBindBuffer = gl.bindBuffer.bind(gl);
      gl.bindBuffer = (target: number, buffer: WebGLBuffer | null) => {
        if (target === gl.ARRAY_BUFFER) {
          boundArrayBuffer = buffer;
        }
        origBindBuffer(target, buffer);
      };

      const origBufferData = gl.bufferData.bind(gl);
      gl.bufferData = ((...args: unknown[]) => {
        const [target, srcData] = args;
        if (
          target === gl.ARRAY_BUFFER &&
          boundArrayBuffer &&
          srcData instanceof Float32Array &&
          srcData.length >= LINE_INSTANCE_FLOATS &&
          srcData.length % LINE_INSTANCE_FLOATS === 0
        ) {
          const total = srcData.length / LINE_INSTANCE_FLOATS;
          let guide = 0;
          for (let i = 0; i < total; i++) {
            const base = i * LINE_INSTANCE_FLOATS;
            const style = Math.round(srcData[base + LINE_STYLE_OFFSET]);
            const owner = Math.round(srcData[base + LINE_OWNER_OFFSET]);
            const face0 = Math.round(srcData[base + 10]);
            const face1 = Math.round(srcData[base + 11]);
            if (style === LINE_STYLE_GUIDE && owner === -1 && face0 === -1 && face1 === -1) {
              guide += 1;
            }
          }
          bufferLineStats.set(boundArrayBuffer, { total, guide });
          if (guide > 0) {
            telemetry.linesPassLastTotalLineCount = total;
            telemetry.linesPassLastGuideCount = guide;
          }
        }
        return (origBufferData as (...a: unknown[]) => void)(...args);
      }) as WebGL2RenderingContext['bufferData'];

      const recordError = () => {
        const err = gl.getError();
        if (err !== gl.NO_ERROR) {
          telemetry.glErrors.push(err);
        }
      };

      const origDrawElementsInstanced = gl.drawElementsInstanced.bind(gl);
      gl.drawElementsInstanced = (
        mode: number,
        count: number,
        type: number,
        offset: number,
        instanceCount: number,
      ) => {
        origDrawElementsInstanced(mode, count, type, offset, instanceCount);
        telemetry.totalDrawCalls += 1;
        if (currentProgram && linesPrograms.has(currentProgram)) {
          telemetry.linesPassDraws += 1;
          telemetry.linesPassLastInstanceCount = instanceCount;
        }
        recordError();
      };

      const origDrawElements = gl.drawElements.bind(gl);
      gl.drawElements = (mode: number, count: number, type: number, offset: number) => {
        origDrawElements(mode, count, type, offset);
        telemetry.totalDrawCalls += 1;
        recordError();
      };

      const origDrawArrays = gl.drawArrays.bind(gl);
      gl.drawArrays = (mode: number, first: number, count: number) => {
        origDrawArrays(mode, first, count);
        telemetry.totalDrawCalls += 1;
        recordError();
      };

      const origDrawArraysInstanced = gl.drawArraysInstanced.bind(gl);
      gl.drawArraysInstanced = (
        mode: number,
        first: number,
        count: number,
        instanceCount: number,
      ) => {
        origDrawArraysInstanced(mode, first, count, instanceCount);
        telemetry.totalDrawCalls += 1;
        recordError();
      };

      return ctx;
    } as HTMLCanvasElement['getContext'];
  });
}

async function waitFrames(page: Page, count = 3): Promise<void> {
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

/**
 * Pure-Node PNG RGBA pixel variance inspector (zero external image deps).
 * Decodes standard 8-bit RGB/RGBA PNGs produced by Playwright's `locator.screenshot()`
 * and checks that rendered pixels are genuinely non-uniform (not blank / single clear color).
 */
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

  // Sample every 4th pixel for fast, deterministic color distribution analysis
  const buckets = new Set<number>();
  let sampled = 0;
  let nonBg = 0;
  // Studio white clear color is ~#F8F9FB (248, 249, 251)
  const bgR = recon[0];
  const bgG = recon[1];
  const bgB = recon[2];

  for (let y = 0; y < height; y += 4) {
    const rowOff = y * stride;
    for (let x = 0; x < width; x += 4) {
      const i = rowOff + x * bpp;
      const r = recon[i];
      const g = recon[i + 1];
      const b = recon[i + 2];
      sampled += 1;
      buckets.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
      if (Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB) > 24) {
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

async function assertCanvasRenderedNonBlank(canvas: Locator, label: string): Promise<void> {
  const png = await canvas.screenshot({ animations: 'allow' });
  const stats = analyzePngColorVariety(png);
  expect(stats.width, `${label}: canvas PNG width`).toBeGreaterThan(100);
  expect(stats.height, `${label}: canvas PNG height`).toBeGreaterThan(100);
  expect(
    stats.distinctBuckets,
    `${label}: expected non-uniform WebGL2 pixels (buckets=${stats.distinctBuckets})`,
  ).toBeGreaterThanOrEqual(8);
  expect(
    stats.nonBackgroundRatio,
    `${label}: expected visible 3D geometry above background (ratio=${stats.nonBackgroundRatio.toFixed(4)})`,
  ).toBeGreaterThan(0.015);
}

async function getTelemetry(page: Page): Promise<WebGL2Telemetry> {
  const data = await page.evaluate(() => window.__webgl2Telemetry);
  expect(data, 'WebGL2 telemetry hook should be initialized').toBeDefined();
  return data!;
}

async function resetLineCounters(page: Page): Promise<void> {
  await page.evaluate(() => {
    if (!window.__webgl2Telemetry) return;
    window.__webgl2Telemetry.linesPassDraws = 0;
    window.__webgl2Telemetry.linesPassLastInstanceCount = 0;
    window.__webgl2Telemetry.linesPassLastTotalLineCount = 0;
    window.__webgl2Telemetry.linesPassLastGuideCount = 0;
  });
}

async function setRangeInputValue(locator: Locator, value: number): Promise<void> {
  await locator.evaluate((el, val) => {
    const input = el as HTMLInputElement;
    input.value = String(val);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

interface OverlayLabelRect {
  text: string;
  className: string;
  left: number;
  top: number;
  right: number;
  bottom: number;
}

interface OverlayBoxRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

interface OverlaySettleSnapshot {
  rafCount: number;
  pendingRafCount: number;
  labels: OverlayLabelRect[];
  popover: OverlayBoxRect | null;
}

interface WaitForProjectedOverlaySettledOptions {
  timeoutMs?: number;
  quietMs?: number;
  minLabels?: number;
  includePopover?: boolean;
  predicate?: (labels: OverlayLabelRect[]) => boolean;
  description?: string;
}

async function getVisibleOverlayLabelRects(page: Page): Promise<OverlayLabelRect[]> {
  return page.evaluate(() => {
    const host = document.getElementById('overlay-labels');
    if (!host) return [];
    const out: OverlayLabelRect[] = [];
    for (const child of Array.from(host.children)) {
      const el = child as HTMLElement;
      if (el.classList.contains('ov-cell-popover')) continue;
      const style = window.getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
        continue;
      }
      const text = (el.textContent ?? '').trim();
      if (!text) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) continue;
      out.push({
        text,
        className: el.className,
        left: rect.left,
        top: rect.top,
        right: rect.right,
        bottom: rect.bottom,
      });
    }
    return out;
  });
}

async function captureOverlaySettleSnapshot(
  page: Page,
  includePopover: boolean,
): Promise<OverlaySettleSnapshot> {
  return page.evaluate((checkPopover) => {
    const telemetry = window.__webgl2Telemetry;
    const host = document.getElementById('overlay-labels');
    const labels: OverlayLabelRect[] = [];
    let popover: OverlayBoxRect | null = null;

    if (host) {
      for (const child of Array.from(host.children)) {
        const el = child as HTMLElement;
        const style = window.getComputedStyle(el);
        const hidden =
          style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0;
        if (el.classList.contains('ov-cell-popover')) {
          if (checkPopover && !hidden) {
            const rect = el.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
              popover = {
                left: rect.left,
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
              };
            }
          }
          continue;
        }
        if (hidden) continue;
        const text = (el.textContent ?? '').trim();
        if (!text) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;
        labels.push({
          text,
          className: el.className,
          left: rect.left,
          top: rect.top,
          right: rect.right,
          bottom: rect.bottom,
        });
      }
    }

    return {
      rafCount: telemetry?.rafCount ?? 0,
      pendingRafCount: telemetry?.pendingRafCount ?? 0,
      labels,
      popover,
    };
  }, includePopover);
}

function maxOverlayBoxDeltaPx(a: OverlayBoxRect, b: OverlayBoxRect): number {
  return Math.max(
    Math.abs(a.left - b.left),
    Math.abs(a.top - b.top),
    Math.abs(a.right - b.right),
    Math.abs(a.bottom - b.bottom),
  );
}

/**
 * Waits for Nano's smooth camera fly (`CAMERA_FLY_DURATION_MS = 650 ms`) and projected DOM overlays
 * (`#overlay-labels` + optional `.ov-cell-popover`) to settle deterministically across fast GPU and
 * variable-rate SwiftShader runners:
 *  1. Observable RAF quiescence: zero pending RAFs and `rafCount` delta === 0 across `quietMs`
 *     (using a pure timer window with no test RAFs inside the window).
 *  2. Projected overlay stability: identical visible `#overlay-labels` texts/classes with
 *     `max(|deltaPx|) < 0.5` across the quiet window, plus optional popover stability and predicate.
 */
async function waitForProjectedOverlaySettled(
  page: Page,
  opts: WaitForProjectedOverlaySettledOptions = {},
): Promise<OverlayLabelRect[]> {
  const {
    timeoutMs = 6000,
    quietMs = 120,
    minLabels = 1,
    includePopover = false,
    predicate,
    description = 'projected overlay',
  } = opts;

  const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

  // Let the post-interaction click/key/resize handler render at least one frame first.
  await waitFrames(page, 1);

  const deadline = Date.now() + timeoutMs;
  let lastDiag = 'no samples captured';

  while (Date.now() < deadline) {
    const before = await captureOverlaySettleSnapshot(page, includePopover);
    if (before.pendingRafCount > 0) {
      lastDiag = `pendingRafCount=${before.pendingRafCount}, labels=${before.labels.length}`;
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await sleep(Math.min(50, remaining));
      continue;
    }

    const quietWindow = Math.min(quietMs, Math.max(10, deadline - Date.now()));
    await sleep(quietWindow);

    const after = await captureOverlaySettleSnapshot(page, includePopover);
    const rafDelta = after.rafCount - before.rafCount;
    const rafIdle =
      before.pendingRafCount === 0 && after.pendingRafCount === 0 && rafDelta === 0;

    let maxLabelDelta = Infinity;
    if (
      before.labels.length === after.labels.length &&
      before.labels.every(
        (lbl, idx) =>
          lbl.text === after.labels[idx].text && lbl.className === after.labels[idx].className,
      )
    ) {
      maxLabelDelta = 0;
      for (let i = 0; i < after.labels.length; i++) {
        const delta = maxOverlayBoxDeltaPx(before.labels[i], after.labels[i]);
        if (delta > maxLabelDelta) maxLabelDelta = delta;
      }
    }

    const labelsStable = after.labels.length >= minLabels && maxLabelDelta < 0.5;

    let popoverStable = true;
    let popoverDelta = 0;
    if (includePopover) {
      if (!before.popover || !after.popover) {
        popoverStable = false;
        popoverDelta = Infinity;
      } else {
        popoverDelta = maxOverlayBoxDeltaPx(before.popover, after.popover);
        popoverStable = popoverDelta < 0.5;
      }
    }

    const predicateOk = predicate ? predicate(after.labels) : true;

    if (rafIdle && labelsStable && popoverStable && predicateOk) {
      return after.labels;
    }

    lastDiag =
      `rafDelta=${rafDelta}, pendingRaf=${after.pendingRafCount}, ` +
      `labels=${after.labels.length} (min=${minLabels}), maxLabelDeltaPx=${
        Number.isFinite(maxLabelDelta) ? maxLabelDelta.toFixed(2) : 'mismatch'
      }, ` +
      `popoverOk=${popoverStable} (deltaPx=${
        Number.isFinite(popoverDelta) ? popoverDelta.toFixed(2) : 'missing'
      }), ` +
      `predicateOk=${predicateOk}, visible=[${after.labels.map((l) => l.text).join(' | ')}]`;

    const remaining = deadline - Date.now();
    if (remaining <= 0) break;
    await sleep(Math.min(40, remaining));
  }

  throw new Error(
    `Timed out after ${timeoutMs}ms waiting for ${description} to settle (${lastDiag})`,
  );
}

test.describe('Nemotron 3 Nano v1.0 · WebGL2 & UI Hardening Smoke Suite', () => {
  test('1. WebGL2 init, non-blank render, and zero browser or GL errors', async ({ page }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    const canvas = page.locator('#gl-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#webgl-fallback')).toBeHidden();

    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(300);
    expect(box!.height).toBeGreaterThan(300);

    const tel = await getTelemetry(page);
    expect(tel.hasWebGL2).toBe(true);
    expect(tel.totalDrawCalls).toBeGreaterThan(0);
    expect(tel.glErrors).toEqual([]);

    await assertCanvasRenderedNonBlank(canvas, 'Initial Tower WebGL2 frame');
    guard.assertClean();
  });

  test('2. Tower & all 5 cutaways navigate cleanly with live landmarks and non-blank WebGL2 frames', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 3);

    // Exit tour so keyboard mode switching is verified in clean free-explore state
    await page.locator('#tour-toggle-btn').click();
    await page.locator('#gl-canvas').focus();
    await waitFrames(page, 2);

    const views: ReadonlyArray<{
      key: string;
      mode: string;
      landmarkPattern: RegExp;
    }> = [
      { key: '1', mode: 'tower', landmarkPattern: /Embed|Output|Bay|Mamba|GQA|Residual/i },
      { key: '2', mode: 'cutaway_mamba', landmarkPattern: /Conv1D|SSM|W_in|RMSNorm|H_t/i },
      { key: '3', mode: 'cutaway_moe', landmarkPattern: /Router|Shared|Expert|RMSNorm|Top-6/i },
      { key: '4', mode: 'cutaway_gqa', landmarkPattern: /Scores|KV|W_Q|Softmax|NoPE|RMSNorm/i },
      { key: '5', mode: 'cutaway_embedding', landmarkPattern: /NoPE|W_embed|X₀|Token|Pos/i },
      { key: '6', mode: 'cutaway_output', landmarkPattern: /#1\s+"|LM Head|Logits|RMSNorm|Top-5/i },
    ];

    const canvas = page.locator('#gl-canvas');
    for (const view of views) {
      await page.keyboard.press(view.key);
      const tab = page.locator(`.mode-btn[data-mode="${view.mode}"]`);
      await expect(tab).toHaveAttribute('aria-selected', 'true');
      await expect(tab).toHaveClass(/is-active/);
      await waitForProjectedOverlaySettled(page, { description: `Test 2 ${view.mode}` });

      await assertCanvasRenderedNonBlank(canvas, `View ${view.mode}`);

      const labels = await getVisibleOverlayLabelRects(page);
      expect(labels.length, `${view.mode}: expected visible overlay landmarks`).toBeGreaterThan(0);
      expect(
        labels.some((l) => view.landmarkPattern.test(l.text)),
        `${view.mode}: expected landmark matching ${view.landmarkPattern}, saw: ${labels.map((l) => l.text).join(' | ')}`,
      ).toBe(true);

      const tel = await getTelemetry(page);
      expect(tel.glErrors, `${view.mode}: WebGL errors`).toEqual([]);
    }

    guard.assertClean();
  });

  test('3. Rendered LinesPass guide-line geometry draws cleanly across Embedding, Mamba, MoE, and GQA cutaways', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await page.locator('#tour-toggle-btn').click();
    await page.locator('#gl-canvas').focus();
    await waitFrames(page, 2);

    const cutawayChecks: ReadonlyArray<{ key: string; name: string }> = [
      { key: '5', name: 'Embedding (NoPE X & stanchions)' },
      { key: '2', name: 'Mamba-2 (Conv1D/SSM frames)' },
      { key: '3', name: 'MoE (dashed routing frames)' },
      { key: '4', name: 'GQA (stanchions & rails)' },
    ];

    for (const item of cutawayChecks) {
      await resetLineCounters(page);
      await page.keyboard.press(item.key);
      await waitFrames(page, 4);

      const tel = await getTelemetry(page);
      expect(tel.linesPassProgramLinked, `${item.name}: LinesPass shader linked`).toBe(true);
      expect(tel.linesPassDraws, `${item.name}: LinesPass drawElementsInstanced calls`).toBeGreaterThan(0);
      expect(
        tel.linesPassLastInstanceCount,
        `${item.name}: LinesPass drawn instance count`,
      ).toBeGreaterThan(0);
      expect(
        tel.linesPassLastGuideCount,
        `${item.name}: uploaded LINE_STYLE_GUIDE instances (iStyle === 2)`,
      ).toBeGreaterThan(0);
      expect(tel.glErrors, `${item.name}: gl.getError() === 0`).toEqual([]);
    }

    guard.assertClean();
  });

  test('4. Guided Beats 7.1-7.3 continuity, AR branching, and tour vs free-explore session isolation', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 3);

    // Ensure Tour is active
    const tourBtn = page.locator('#tour-toggle-btn');
    await expect(tourBtn).toHaveClass(/is-active/);

    // Beat 7.1 (index 18): Output T=8 prediction matches prefix
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="18"]').click();
    await waitForProjectedOverlaySettled(page, {
      description: 'Test 4 Beat 7.1 Output Top-5',
      predicate: (labels) =>
        labels.some((l) => !l.className.includes('ov-axis-header') && l.text.startsWith('#1 ')),
    });
    await expect(page.locator('#editorial-panel .ed-title')).toContainText('7.1');
    await expect(page.locator('.mode-btn[data-mode="cutaway_output"]')).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await expect(page.locator('#token-value')).toHaveText('8 / 16');

    const beat71Labels = await getVisibleOverlayLabelRects(page);
    const rank1Label = beat71Labels.find(
      (l) => !l.className.includes('ov-axis-header') && l.text.startsWith('#1 '),
    );
    expect(rank1Label, 'Beat 7.1 should render #1 Top-5 prediction label').toBeDefined();
    const top1Match = rank1Label!.text.match(/^#1\s+"([^"]+)"\s+·\s+\d+\.\d+%$/);
    expect(top1Match, `Beat 7.1 #1 label format: ${rank1Label!.text}`).not.toBeNull();
    const expectedT8Word = top1Match![1];

    // Beat 7.2 (index 19): Slow instructional spine replay + AR step emitting Beat 7.1 #1 token
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="19"]').click();
    await waitFrames(page, 4);
    await expect(page.locator('#editorial-panel .ed-title')).toContainText('7.2');
    await expect(page.locator('.mode-btn[data-mode="tower"]')).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await expect(page.locator('#exec-mode-decode')).toHaveAttribute('aria-checked', 'true');
    await expect(page.locator('#token-value')).toHaveText('9 / 16 (AR)');
    await expect(page.locator('#spine-wave-hud')).toBeVisible();

    const seqChips = page.locator('[data-testid="ar-seq-strip"] .ar-seq-chip');
    await expect(seqChips).toHaveCount(9);
    await expect(seqChips.nth(8)).toContainText(`t8 ${expectedT8Word}`);

    // Beat 7.3 (index 20): GQA cached [1 × (P+1)] = [1 × 9] inspection in Bay 43
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="20"]').click();
    await waitFrames(page, 4);
    await expect(page.locator('#editorial-panel .ed-title')).toContainText('7.3');
    await expect(page.locator('.mode-btn[data-mode="cutaway_gqa"]')).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await expect(page.locator('#gqa-subbar [data-gqa-bay="43"]')).toHaveClass(/is-active/);
    await expect(page.locator('#editorial-panel')).toContainText('[1 × 9]');

    // Beat 7.4 (index 21): Step back to prefill, select alt #2 candidate, and fork branch-1
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="21"]').click();
    await waitFrames(page, 3);
    await page.locator('[data-testid="ar-step-prev"]').click();
    await expect(page.locator('[data-testid="ar-decode-panel"] .ar-decode-title')).toContainText(
      'Step 1/2',
    );
    const candRows = page.locator('[data-testid="ar-cand-list"] .ar-cand-row');
    await expect(candRows).toHaveCount(5);
    await candRows.nth(1).click();
    await page.locator('[data-testid="ar-insert-selected"]').click();
    await waitFrames(page, 3);

    const branchStrip = page.locator('[data-testid="ar-branch-strip"]');
    await expect(branchStrip).toBeVisible();
    await expect(branchStrip.locator('button[data-decode-branch="branch-0"]')).toBeVisible();
    await expect(branchStrip.locator('button[data-decode-branch="branch-1"]')).toBeVisible();

    // Exit Tour -> free-explore AR session must remain completely isolated from Tour's branch-1
    await tourBtn.click();
    await expect(tourBtn).not.toHaveClass(/is-active/);
    await page.locator('#exec-mode-decode').click();
    await waitFrames(page, 3);

    await expect(page.locator('[data-testid="ar-decode-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="ar-branch-strip"]')).toHaveCount(0);
    await expect(page.locator('[data-testid="ar-decode-panel"] .ar-branch-tag')).toHaveText(
      'branch-0',
    );
    await expect(page.locator('[data-testid="ar-decode-panel"] .ar-decode-title')).toContainText(
      'Step 1/1',
    );

    guard.assertClean();
  });

  test('5. Native Space/Enter control activation, roving tablist ARIA, and modal focus capture/restore', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 3);

    // 5a. Focused button Space during Tour activates the button and does NOT hijack into beat advance
    await expect(page.locator('#tour-toggle-btn')).toContainText('Beat 1.1');
    const fp8Btn = page.locator('#precision-fp8');
    await fp8Btn.focus();
    await page.keyboard.press('Space');
    await expect(fp8Btn).toHaveAttribute('aria-checked', 'true');
    await expect(page.locator('#tour-toggle-btn')).toContainText('Beat 1.1');

    // 5b. Focused button Enter in Tower with selected bay activates button and does NOT hijack into bay zoom
    await page.locator('#tour-toggle-btn').click();
    await page.locator('#bay-next').click();
    await expect(page.locator('#zoom-bay-btn')).toBeVisible();
    const bf16Btn = page.locator('#precision-bf16');
    await bf16Btn.focus();
    await page.keyboard.press('Enter');
    await expect(bf16Btn).toHaveAttribute('aria-checked', 'true');
    await expect(page.locator('.mode-btn[data-mode="tower"]')).toHaveAttribute(
      'aria-selected',
      'true',
    );

    // 5c. Focused button Space in Cutaway activates button and does NOT hijack into stepper Play
    await page.locator('.mode-btn[data-mode="cutaway_gqa"]').click();
    await waitFrames(page, 3);
    const playBtn = page.locator('#gqa-step-play');
    await expect(playBtn).toHaveAttribute('aria-pressed', 'false');
    await fp8Btn.focus();
    await page.keyboard.press('Space');
    await expect(fp8Btn).toHaveAttribute('aria-checked', 'true');
    await expect(playBtn).toHaveAttribute('aria-pressed', 'false');

    // 5d. #gqa-step-pills role="tab" roving tabindex + ArrowRight switching
    const stepTabs = page.locator('#gqa-step-pills [role="tab"]');
    await expect(stepTabs).toHaveCount(6);
    await expect(stepTabs.nth(0)).toHaveAttribute('tabindex', '0');
    await expect(stepTabs.nth(1)).toHaveAttribute('tabindex', '-1');

    await stepTabs.nth(0).click();
    await expect(stepTabs.nth(0)).toHaveAttribute('tabindex', '0');
    await expect(stepTabs.nth(0)).toHaveAttribute('aria-selected', 'true');
    await expect(stepTabs.nth(1)).toHaveAttribute('tabindex', '-1');
    await expect(stepTabs.nth(1)).toHaveAttribute('aria-selected', 'false');

    await stepTabs.nth(0).focus();
    await page.keyboard.press('ArrowRight');
    await expect(stepTabs.nth(1)).toBeFocused();
    await expect(stepTabs.nth(1)).toHaveAttribute('tabindex', '0');
    await expect(stepTabs.nth(1)).toHaveAttribute('aria-selected', 'true');
    await expect(stepTabs.nth(0)).toHaveAttribute('tabindex', '-1');
    await expect(stepTabs.nth(0)).toHaveAttribute('aria-selected', 'false');

    // 5e. '?' and 'M' modal focus move + Escape focus restoration
    const fitBtn = page.locator('#fit-btn');
    await fitBtn.focus();
    await expect(fitBtn).toBeFocused();

    await page.keyboard.press('?');
    const shortcutsDialog = page.locator('#shortcuts-popover');
    await expect(shortcutsDialog).toBeVisible();
    await expect(shortcutsDialog).toHaveAttribute('role', 'dialog');
    await expect(shortcutsDialog).toHaveAttribute('aria-modal', 'true');
    await expect(page.locator('#shortcuts-close')).toBeFocused();

    await page.keyboard.press('Escape');
    await expect(shortcutsDialog).toBeHidden();
    await expect(fitBtn).toBeFocused();

    await page.keyboard.press('m');
    const memDrawer = page.locator('#memory-chart-drawer');
    await expect(memDrawer).toBeVisible();
    await expect(memDrawer.locator('[role="dialog"]')).toHaveAttribute('aria-modal', 'true');
    await expect(memDrawer.locator('.mem-close-btn')).toBeFocused();

    await page.keyboard.press('Escape');
    await expect(memDrawer).toBeHidden();
    await expect(fitBtn).toBeFocused();

    guard.assertClean();
  });

  test('6. Reduced-motion emulation and live toggle disable continuous animations while preserving inspection', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 3);

    // Jump to Beat 7.2 (live instructional spine wave)
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="19"]').click();
    await waitFrames(page, 3);
    const waveHud = page.locator('#spine-wave-hud');
    await expect(waveHud).toBeVisible();
    await expect(waveHud).not.toHaveClass(/is-reduced-motion/);

    // Live toggle to reducedMotion: 'reduce' settles wave & disables continuous animation
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await waitFrames(page, 3);
    await expect(waveHud).toHaveClass(/is-reduced-motion/);

    // Switch to GQA cutaway, click step 3, verify idle RAF stops spinning
    await page.locator('#tour-toggle-btn').click();
    await page.locator('.mode-btn[data-mode="cutaway_gqa"]').click();
    await page.locator('#gqa-step-pills button[data-gqa-step="2"]').click();
    await waitFrames(page, 4);

    // Clicking Play under reducedMotion steps cleanly instead of running an endless loop
    const rafBefore = (await getTelemetry(page)).rafCount;
    await page.waitForTimeout(350);
    const rafAfter = (await getTelemetry(page)).rafCount;
    expect(
      rafAfter - rafBefore,
      `Expected no idle RAF spin under reducedMotion (delta=${rafAfter - rafBefore})`,
    ).toBeLessThanOrEqual(2);

    // Full manual inspection remains functional under reducedMotion
    const eqBefore = await page.locator('#gqa-formula-eq').textContent();
    await page.locator('#gqa-step-pills button[data-gqa-step="3"]').click();
    await waitFrames(page, 2);
    const eqAfter = await page.locator('#gqa-formula-eq').textContent();
    expect(eqAfter?.trim().length ?? 0).toBeGreaterThan(0);
    expect(eqAfter).not.toBe(eqBefore);

    // Toggle back to no-preference cleanly
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.locator('#gqa-step-play').click();
    await expect(page.locator('#gqa-step-play')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#gqa-step-play').click();
    await expect(page.locator('#gqa-step-play')).toHaveAttribute('aria-pressed', 'false');

    guard.assertClean();
  });

  test('7. Output Top-5 non-overlapping visibility, MoE running Σσ denominator & weight masking, and popover containment', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 3);

    // 7a. Beat 7.1 Output cutaway: ALL FIVE #1..#5 candidate pills visible, formatted, in-bounds, non-overlapping
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="18"]').click();
    await waitForProjectedOverlaySettled(page, {
      description: 'Test 7a Beat 7.1 Output Top-5',
      predicate: (labels) => {
        const pills = labels.filter((l) => !l.className.includes('ov-axis-header'));
        return [1, 2, 3, 4, 5].every((rank) => pills.some((l) => l.text.startsWith(`#${rank} `)));
      },
    });

    const stageBox = await page.locator('#stage-host').boundingBox();
    expect(stageBox).not.toBeNull();

    const outLabels = await getVisibleOverlayLabelRects(page);
    const outPillLabels = outLabels.filter((l) => !l.className.includes('ov-axis-header'));
    const top5 = [1, 2, 3, 4, 5].map((rank) => {
      const match = outPillLabels.find((l) => l.text.startsWith(`#${rank} `));
      expect(match, `Expected visible Output Top-5 label #${rank}`).toBeDefined();
      expect(
        match!.text,
        `Output Top-5 #${rank} should include token identity and % probability`,
      ).toMatch(new RegExp(`^#${rank}\\s+"[^"]+"\\s+·\\s+\\d+\\.\\d+%$`));
      expect(match!.left).toBeGreaterThanOrEqual(stageBox!.x - 1);
      expect(match!.right).toBeLessThanOrEqual(stageBox!.x + stageBox!.width + 1);
      expect(match!.top).toBeGreaterThanOrEqual(stageBox!.y - 1);
      expect(match!.bottom).toBeLessThanOrEqual(stageBox!.y + stageBox!.height + 1);
      return match!;
    });

    for (let i = 0; i < top5.length; i++) {
      for (let j = i + 1; j < top5.length; j++) {
        const a = top5[i];
        const b = top5[j];
        const overlaps =
          a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
        expect(overlaps, `Top-5 #${i + 1} overlaps #${j + 1}`).toBe(false);
      }
    }

    // 7b. MoE cutaway Step 3 (topk_weights) running Σσ denominator & sidebar masking (2/6 vs 6/6)
    await page.locator('#tour-toggle-btn').click();
    await page.locator('.mode-btn[data-mode="cutaway_moe"]').click();
    await waitFrames(page, 3);
    await page.locator('#gqa-step-pills button[data-gqa-step="2"]').click();
    await waitFrames(page, 2);

    // Scrub to 4% -> slot 0, term 2/6 running Σσ denominator
    await setRangeInputValue(page.locator('#gqa-step-scrub'), 4);
    await waitForProjectedOverlaySettled(page, {
      includePopover: true,
      description: 'Test 7b MoE Step 3 scrub=4%',
    });

    const eqRunning = (await page.locator('#gqa-formula-eq').textContent()) ?? '';
    expect(eqRunning).toContain('Σσ =');
    expect(eqRunning).toContain('(2/6)');
    expect(eqRunning).not.toContain('w ≈');

    const popover = page.locator('.ov-cell-popover');
    await expect(popover).toBeVisible();
    const popRunningText = (await popover.textContent()) ?? '';
    expect(popRunningText).toContain('Σσ =');
    expect(popRunningText).toContain('(2/6)');
    expect(popRunningText).not.toContain('w ≈');

    const popBox = await popover.boundingBox();
    expect(popBox).not.toBeNull();
    expect(popBox!.x).toBeGreaterThanOrEqual(stageBox!.x - 1);
    expect(popBox!.y).toBeGreaterThanOrEqual(stageBox!.y - 1);
    expect(popBox!.x + popBox!.width).toBeLessThanOrEqual(stageBox!.x + stageBox!.width + 1);
    expect(popBox!.y + popBox!.height).toBeLessThanOrEqual(stageBox!.y + stageBox!.height + 1);

    // Sidebar Top-6 Active Experts masks active ('…') and uncomputed ('—') weights
    const sidebarWeightsRunning = await page
      .locator('[data-testid="moe-inspector-controls"] .moe-exp-btn .moe-exp-w')
      .allTextContents();
    expect(sidebarWeightsRunning).toHaveLength(6);
    expect(sidebarWeightsRunning[0]).toBe('…');
    expect(sidebarWeightsRunning.slice(1)).toEqual(['—', '—', '—', '—', '—']);

    // Scrub to 99% -> 6/6 final normalized weight reveal across dock & all 6 sidebar experts
    await setRangeInputValue(page.locator('#gqa-step-scrub'), 99);
    await waitFrames(page, 4);

    const eqFinal = (await page.locator('#gqa-formula-eq').textContent()) ?? '';
    expect(eqFinal).toContain('× 2.5');
    const sidebarWeightsFinal = await page
      .locator('[data-testid="moe-inspector-controls"] .moe-exp-btn .moe-exp-w')
      .allTextContents();
    expect(sidebarWeightsFinal).toHaveLength(6);
    for (const w of sidebarWeightsFinal) {
      expect(w).toMatch(/^\d+\.\d{2}$/);
    }

    // 7c. Verify popover stays inside stage viewport bounds on GQA zoomed Scores accumulation too
    await page.locator('.mode-btn[data-mode="cutaway_gqa"]').click();
    await page.locator('#gqa-step-pills button[data-gqa-step="2"]').click();
    await setRangeInputValue(page.locator('#gqa-step-scrub'), 30);
    await waitForProjectedOverlaySettled(page, {
      includePopover: true,
      description: 'Test 7c GQA Step 3 scrub=30%',
    });

    await expect(popover).toBeVisible();
    const gqaPopBox = await popover.boundingBox();
    expect(gqaPopBox).not.toBeNull();
    expect(gqaPopBox!.x).toBeGreaterThanOrEqual(stageBox!.x - 1);
    expect(gqaPopBox!.y).toBeGreaterThanOrEqual(stageBox!.y - 1);
    expect(gqaPopBox!.x + gqaPopBox!.width).toBeLessThanOrEqual(stageBox!.x + stageBox!.width + 1);
    expect(gqaPopBox!.y + gqaPopBox!.height).toBeLessThanOrEqual(
      stageBox!.y + stageBox!.height + 1,
    );

    guard.assertClean();
  });

  test('8. Responsive layouts across 1440x900, 1024x800 (HUD & toolbar), and mobile 390x844 (Tower & shortcuts)', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    // 8a. 1440x900 baseline
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(ROOT_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 3);
    await assertCanvasRenderedNonBlank(page.locator('#gl-canvas'), '1440x900 desktop');

    // 8b. 1024x800 compact desktop: .spine-wave-hud unclipped & toolbar usable
    await page.setViewportSize({ width: 1024, height: 800 });
    await page.locator('[data-testid="tour-progress-rail"] button[data-tour-beat="19"]').click();
    await waitFrames(page, 4);

    const stage1024 = await page.locator('#stage-host').boundingBox();
    const hud1024 = await page.locator('#spine-wave-hud').boundingBox();
    expect(stage1024).not.toBeNull();
    expect(hud1024).not.toBeNull();
    expect(hud1024!.x).toBeGreaterThanOrEqual(stage1024!.x - 1);
    expect(hud1024!.x + hud1024!.width).toBeLessThanOrEqual(stage1024!.x + stage1024!.width + 1);
    expect(hud1024!.y).toBeGreaterThanOrEqual(stage1024!.y - 1);
    expect(hud1024!.y + hud1024!.height).toBeLessThanOrEqual(stage1024!.y + stage1024!.height + 1);

    // Toolbar remains clickable and usable at 1024px
    await page.locator('.mode-btn[data-mode="cutaway_mamba"]').click();
    await expect(page.locator('.mode-btn[data-mode="cutaway_mamba"]')).toHaveAttribute(
      'aria-selected',
      'true',
    );

    // 8c. Mobile 390x844: clean #stage-host, Tower landmarks near top (no 84px dead margin), unclipped shortcuts popover
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.mode-btn[data-mode="tower"]').click();
    await page.locator('#fit-btn').click();
    await waitForProjectedOverlaySettled(page, {
      description: 'Test 8c mobile 390x844 Tower fit',
    });

    const stageMobile = await page.locator('#stage-host').boundingBox();
    expect(stageMobile).not.toBeNull();
    expect(stageMobile!.width).toBeGreaterThanOrEqual(350);
    expect(stageMobile!.height).toBeGreaterThanOrEqual(280);
    await assertCanvasRenderedNonBlank(page.locator('#gl-canvas'), '390x844 mobile Tower');

    // Toolbar stacks statically above #stage-host on mobile (no floating overlap on top of canvas)
    const toolbarMobile = await page.locator('#floating-toolbar').boundingBox();
    expect(toolbarMobile).not.toBeNull();
    expect(toolbarMobile!.y + toolbarMobile!.height).toBeLessThanOrEqual(stageMobile!.y + 2);

    const mobileLabels = await getVisibleOverlayLabelRects(page);
    expect(mobileLabels.length).toBeGreaterThan(0);
    const minTopOffset = Math.min(...mobileLabels.map((l) => l.top - stageMobile!.y));
    expect(
      minTopOffset,
      `Expected mobile Tower landmarks near top of stage without 84px blank gap (got ${minTopOffset.toFixed(1)}px)`,
    ).toBeLessThan(78);

    // Open shortcuts popover on mobile -> must stay completely inside 390x844 viewport without clipping
    await page.locator('#shortcuts-btn').click();
    const shortcuts = page.locator('#shortcuts-popover');
    await expect(shortcuts).toBeVisible();
    const shortBox = await shortcuts.boundingBox();
    expect(shortBox).not.toBeNull();
    expect(shortBox!.x).toBeGreaterThanOrEqual(0);
    expect(shortBox!.y).toBeGreaterThanOrEqual(0);
    expect(shortBox!.x + shortBox!.width).toBeLessThanOrEqual(390);
    expect(shortBox!.y + shortBox!.height).toBeLessThanOrEqual(844);

    guard.assertClean();
  });

  test('9. Production build under GitHub Pages subpath /llm-viz/ serves all assets HTTP 200 and renders WebGL2 cutaways', async ({
    page,
  }) => {
    const guard = attachErrorGuards(page);
    await installWebGL2Telemetry(page);

    const failedRequests: string[] = [];
    const errorResponses: string[] = [];
    const loadedJsAssets: string[] = [];
    const loadedCssAssets: string[] = [];

    page.on('requestfailed', (req) => {
      failedRequests.push(`${req.method()} ${req.url()} (${req.failure()?.errorText ?? 'failed'})`);
    });
    page.on('response', (res) => {
      const url = res.url();
      const status = res.status();
      if (status >= 400) {
        errorResponses.push(`${status} ${url}`);
      }
      if (status === 200 && url.includes('/llm-viz/assets/')) {
        if (url.endsWith('.js')) loadedJsAssets.push(url);
        if (url.endsWith('.css')) loadedCssAssets.push(url);
      }
    });

    await page.goto(PAGES_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    expect(failedRequests, `Failed network requests under /llm-viz/`).toEqual([]);
    expect(errorResponses, `HTTP >= 400 responses under /llm-viz/`).toEqual([]);
    expect(loadedJsAssets.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(loadedCssAssets.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

    const canvas = page.locator('#gl-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#webgl-fallback')).toBeHidden();
    await assertCanvasRenderedNonBlank(canvas, '/llm-viz/ Tower');

    // Verify cutaways load and render cleanly under /llm-viz/ subpath
    await page.locator('#tour-toggle-btn').click();
    await page.locator('#gl-canvas').focus();
    for (const [key, mode] of [
      ['5', 'cutaway_embedding'],
      ['2', 'cutaway_mamba'],
      ['3', 'cutaway_moe'],
      ['4', 'cutaway_gqa'],
      ['6', 'cutaway_output'],
    ] as const) {
      await page.keyboard.press(key);
      await expect(page.locator(`.mode-btn[data-mode="${mode}"]`)).toHaveAttribute(
        'aria-selected',
        'true',
      );
      await waitForProjectedOverlaySettled(page, { description: `Test 9 /llm-viz/ ${mode}` });
      await assertCanvasRenderedNonBlank(canvas, `/llm-viz/ ${mode}`);
    }

    const tel = await getTelemetry(page);
    expect(tel.hasWebGL2).toBe(true);
    expect(tel.glErrors).toEqual([]);
    expect(errorResponses).toEqual([]);
    guard.assertClean();
  });
});
