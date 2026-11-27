import { inflateSync } from 'node:zlib';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';
import type {
  TwoTowerAppHandle,
  TwoTowerSpatialDebugHandle,
  TwoTowerTelemetrySnapshot,
} from '../../src/twotower/app/spatialApp';

declare global {
  interface Window {
    __TWOTOWER_TELEMETRY__?: TwoTowerTelemetrySnapshot;
    __TWOTOWER_APP__?: TwoTowerAppHandle;
    __TWOTOWER_SPATIAL_DEBUG__?: TwoTowerSpatialDebugHandle;
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

async function waitForTwoTowerSettled(page: Page): Promise<void> {
  await expect
    .poll(() => page.evaluate(() => window.__TWOTOWER_TELEMETRY__?.rafActive), {
      timeout: 5000,
    })
    .toBe(false);
  await waitFrames(page, 1);
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

  for (let y = 0; y < height; y += 4) {
    const rowOff = y * stride;
    for (let x = 0; x < width; x += 4) {
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
  expect(stats.height, `${label}: canvas PNG height`).toBeGreaterThan(200);
  expect(
    stats.distinctBuckets,
    `${label}: expected non-uniform WebGL2 pixels (buckets=${stats.distinctBuckets})`,
  ).toBeGreaterThanOrEqual(8);
  expect(
    stats.nonBackgroundRatio,
    `${label}: expected visible 3D geometry above studio-white background (ratio=${stats.nonBackgroundRatio.toFixed(4)})`,
  ).toBeGreaterThan(0.012);
}

async function getTelemetry(page: Page): Promise<TwoTowerTelemetrySnapshot> {
  const tel = await page.evaluate(() => window.__TWOTOWER_TELEMETRY__);
  expect(tel, 'window.__TWOTOWER_TELEMETRY__ must be populated').toBeDefined();
  return tel!;
}

function assertNoInfinityText(text: string | null, context: string): void {
  const safe = text ?? '';
  expect(safe, `${context}: must never render "Infinity"`).not.toMatch(/Infinity/i);
  expect(safe, `${context}: must never render "∞"`).not.toContain('∞');
}

function assertNoInfinityOrNaNText(text: string | null, context: string): void {
  assertNoInfinityText(text, context);
  const safe = text ?? '';
  expect(safe, `${context}: must never render "NaN"`).not.toMatch(/\bNaN\b/);
}

test.describe('Nemotron-Labs-TwoTower Stage TT3 · Interactive WebGL2 Spatial Smoke Suite', () => {
  test('1. Root /twotower/ WebGL2 Dual-Tower Macro boot, epistemic contract, linker channels, and 10-axis bay picking', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 5);

    // Compact model switcher trigger visible + base-path-safe root href `/` on Nano link
    const switcherTrigger = page.getByTestId('model-switcher-trigger');
    await expect(switcherTrigger).toBeVisible();
    await expect(page.getByTestId('nav-to-nano')).toHaveAttribute('href', '/');

    // WebGL2 canvas mounted, fallback hidden, non-blank rendered pixels
    const canvas = page.getByTestId('twotower-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#webgl-fallback')).toBeHidden();
    await assertCanvasNonBlank(canvas, 'Dual-Tower Macro initial frame');

    const tel = await getTelemetry(page);
    expect(tel.viewMode).toBe('macro');
    expect(tel.maxTextureSize).toBeGreaterThanOrEqual(2048);
    expect(tel.atlasUploads).toBeGreaterThanOrEqual(0);
    expect(tel.drawCallsLastFrame).toBeGreaterThan(0);
    expect(tel.solidCount).toBeGreaterThan(100);
    expect(tel.lineCount).toBeGreaterThan(0);
    expect(tel.flowTriangleCount).toBeGreaterThan(0);
    expect(tel.glErrorCount).toBe(0);
    expect(tel.rafIntervalSampleCount).toBeGreaterThanOrEqual(0);

    // Epistemic miniature disclosure contract, explicit JS cb + RAF Δ telemetry & attribution
    const pill = page.getByTestId('epistemic-pill');
    await expect(pill).toContainText('d=8');
    await expect(pill).toContainText('V=34');
    await expect(pill).toContainText('2×52L');
    await expect(pill).toContainText('Synthetic Mini');
    const footer = page.getByTestId('inspector-telemetry-footer');
    await expect(footer).toContainText('35b64985');
    const timingReadout = page.getByTestId('telemetry-timing-readout');
    await expect(timingReadout).toContainText('JS cb:');
    await expect(timingReadout).toContainText('RAF Δ:');
    await expect(timingReadout).toHaveAttribute('title', /performance\.now\(\)/);
    await expect(timingReadout).toHaveAttribute('title', /NOT full asynchronous GPU execution/);
    await expect(page.getByTestId('twotower-attribution')).toContainText('bbycroft/llm-viz');
    await expect(page.getByTestId('twotower-attribution')).toContainText('arXiv:2606.26493v2');

    // Toggle all linker channel filters cleanly
    for (const ch of ['gqa_kv', 'mamba_conv', 'mamba_ssm', 'timestep', 'all'] as const) {
      await page.getByTestId(`linker-channel-${ch}`).click();
      await waitFrames(page, 2);
      await expect(page.getByTestId(`linker-channel-${ch}`)).toHaveClass(/is-active/);
    }
    await assertCanvasNonBlank(canvas, 'Macro after linker channel toggles');

    // Click Context Bay 06 directly on the 3D canvas and verify all 10 identity axes
    const bayPt = await page.evaluate(() =>
      window.__TWOTOWER_APP__?.getCanvasClickPointForBay(6, 'context'),
    );
    expect(bayPt).not.toBeNull();
    expect(bayPt!.visible).toBe(true);
    await page.mouse.click(bayPt!.x, bayPt!.y);
    await waitFrames(page, 3);

    const idCard = page.getByTestId('selection-identity-card');
    await expect(idCard).toContainText('context');
    await expect(idCard).toContainText('twotower-context-mini:');
    await expect(idCard).toContainText('Bay 06 (L5)');
    await expect(idCard).toContainText('*');
    await expect(idCard).toContainText('10. traceOrigin');

    const telAfter = await getTelemetry(page);
    expect(telAfter.glErrorCount).toBe(0);
    expect(guards.loadedJsAssets.length).toBeGreaterThan(0);
    expect(guards.loadedCssAssets.length).toBeGreaterThan(0);
    guards.assertClean();
  });

  test('2. Full Diffusion Lifecycle & All 7 Spatial Views (incl. 128×128 GQA, Mamba Conv1D replacement, MoE partial-denom guard, Output zero-Infinity)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Init Block -> Context remains 8 / 24 and Commit Block is disabled
    await page.getByTestId('sp-btn-init').click();
    await expect(page.getByTestId('sp-phase-badge')).toHaveText('block_initialized');
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 8 / 24');
    await expect(page.getByTestId('sp-btn-commit')).toBeDisabled();

    // Refine -> Commit Conf until completed_uncommitted (Context MUST remain 8 / 24!)
    let refinementPairs = 0;
    for (let iter = 0; iter < 8; iter++) {
      const phase = (await page.getByTestId('sp-phase-badge').textContent())?.trim();
      if (phase === 'completed_uncommitted') break;

      await expect(page.getByTestId('sp-btn-refine')).toBeEnabled();
      await page.getByTestId('sp-btn-refine').click();
      await expect(page.getByTestId('sp-phase-badge')).toHaveText('refinement_denoised');

      await expect(page.getByTestId('sp-btn-conf')).toBeEnabled();
      await page.getByTestId('sp-btn-conf').click();
      refinementPairs += 1;
      await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 8 / 24');
    }

    expect(refinementPairs).toBeGreaterThanOrEqual(1);
    await expect(page.getByTestId('sp-phase-badge')).toHaveText('completed_uncommitted');
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 8 / 24');
    await expect(page.getByTestId('sp-btn-commit')).toBeEnabled();

    const canvas = page.getByTestId('twotower-canvas');

    // View 1: Macro at completed_uncommitted
    await waitFrames(page, 3);
    await assertCanvasNonBlank(canvas, 'View 1: Macro (completed_uncommitted)');

    // View 2: Paired GQA (128×128 atlas + bidirectional_block + 4 lengths)
    await page.getByTestId('view-tab-paired_gqa').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'View 2: Paired GQA');
    const gqaTel = await getTelemetry(page);
    expect(gqaTel.viewMode).toBe('paired_gqa');
    expect(gqaTel.currentAtlasDims).toEqual([128, 128]);
    expect(gqaTel.atlasUploads).toBeGreaterThanOrEqual(1);
    expect(gqaTel.glErrorCount).toBe(0);
    const gqaDiag = page.getByTestId('view-diagnostic-card');
    await expect(gqaDiag).toContainText('bidirectional_block');
    await expect(gqaDiag).toContainText('128×128');
    await expect(gqaDiag).toContainText('allocatedKvCapacity');

    // View 3: Paired Mamba (256×128 atlas + full-block position switch + Conv1D 3->0 replacement)
    await page.getByTestId('view-tab-paired_mamba').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'View 3: Paired Mamba');
    const mambaTel = await getTelemetry(page);
    expect(mambaTel.viewMode).toBe('paired_mamba');
    expect(mambaTel.currentAtlasDims).toEqual([256, 128]);
    expect(mambaTel.glErrorCount).toBe(0);
    const mambaDiag = page.getByTestId('view-diagnostic-card');
    await expect(mambaDiag).toContainText('Inherited: 3');
    await expect(mambaDiag).toContainText('Block: 1');
    await page.getByTestId('mamba-pos-select').selectOption('3');
    await waitFrames(page, 3);
    await expect(mambaDiag).toContainText('Inherited: 0');
    await expect(mambaDiag).toContainText('Block: 4');
    await expect(mambaDiag).toContainText('Inherited Retention');
    await expect(mambaDiag).toContainText('Block Write');

    // View 4: Timestep Conditioning (ACTIVE t + asymmetric pre-norm vs post-norm)
    await page.getByTestId('view-tab-timestep').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'View 4: Timestep Conditioning');
    const timeDiag = page.getByTestId('view-diagnostic-card');
    await expect(timeDiag).toContainText('ACTIVE');
    await expect(timeDiag).toContainText('modulate_then_norm');
    await page.getByTestId('timestep-bay-select').selectOption('2');
    await waitFrames(page, 3);
    await expect(timeDiag).toContainText('norm_then_modulate');
    await expect(timeDiag).toContainText('unmodulated');

    // View 5: MoE (FP64 6th-vs-7th margin + STRICT partial-denominator non-exposure at Step 2 termProgress=0.4!)
    await page.getByTestId('view-tab-moe').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'View 5: Granular MoE');
    const moeDiag = page.getByTestId('view-diagnostic-card');
    await expect(moeDiag).toContainText('FP64 Router');
    await expect(moeDiag).toContainText('0 B Stateless');
    await expect(moeDiag).toContainText('6th-vs-7th Margin');

    // Capture finalized slot-0 weight before entering Step 2 partial accumulation
    const revealedWeightsText = await page.getByTestId('moe-sidebar-weights').textContent();
    expect(revealedWeightsText).toBeTruthy();

    // Activate Step 2 with partial termProgress = 0.4 -> final weights MUST be masked!
    await page.getByTestId('stepper-step-2').click();
    await page.evaluate(() => window.__TWOTOWER_APP__?.setStepperTermProgress(0.4));
    await waitFrames(page, 3);

    const partialState = await page.evaluate(() => window.__TWOTOWER_APP__?.getState());
    expect(partialState?.diagnostics?.partialDenominatorActive).toBe(true);
    const partialSidebarText = await page.getByTestId('moe-sidebar-weights').textContent();
    expect(partialSidebarText).toContain('…');
    expect(partialSidebarText).toContain('—');
    const partialFormulaText = await page.getByTestId('formula-card').textContent();
    expect(partialFormulaText).toContain('Σ');

    // View 6: Embedding (NoPE, V=34, independent W_embed_den)
    await page.getByTestId('view-tab-embedding').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'View 6: Embedding');
    await expect(page.getByTestId('view-diagnostic-card')).toContainText('W_embed_den');
    await expect(page.getByTestId('view-diagnostic-card')).toContainText('V=34');

    // View 7: Output (Unshifted Block Logits + Presentation-Safe Sampler Table: ZERO Infinity, Locked prior step!)
    await page.getByTestId('view-tab-output').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'View 7: Output LM Head');
    const outDiag = page.getByTestId('view-diagnostic-card');
    await expect(outDiag).toContainText('Unshifted Block Logits');
    await expect(outDiag).toContainText('Strictly UNMODULATED');
    const samplerTable = page.getByTestId('spatial-sampler-table');
    await expect(samplerTable).toBeVisible();
    const samplerText = await samplerTable.textContent();
    assertNoInfinityText(samplerText, 'Spatial Output sampler table');
    expect(samplerText).toContain('[MASK]=0.0000');
    if (refinementPairs >= 2) {
      expect(samplerText).toContain('Locked (prior step)');
    }

    // Explicit Commit Block -> Context advances 8 -> 12 / 24!
    await page.getByTestId('sp-btn-commit').click();
    await waitFrames(page, 3);
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 12 / 24');

    const finalTel = await getTelemetry(page);
    expect(finalTel.glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('3. AR, Mock-AR (Timestep BYPASSED), Rewind, Branch Fork, and Synced Developer Workbench Drawer', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Mock-AR (8 -> 9) and confirm Timestep view shows BYPASSED (t = null)
    await page.getByTestId('spatial-mode-select').selectOption('mock_ar');
    await page.getByTestId('sp-btn-mock-ar').click();
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 9 / 24');

    await page.getByTestId('view-tab-timestep').click();
    await waitFrames(page, 4);
    await expect(page.getByTestId('view-diagnostic-card')).toContainText('BYPASSED (t = null)');

    // AR (9 -> 10)
    await page.getByTestId('spatial-mode-select').selectOption('ar');
    await page.getByTestId('sp-btn-ar').click();
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 10 / 24');

    // Rewind (<- Back: 10 -> 9) and confirm Historical Record guard badge
    await page.getByTestId('sp-btn-back').click();
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 9 / 24');
    await expect(page.getByTestId('historical-config-card')).toContainText('Historical Record #1');

    // Fork branch at rewound cursor
    await page.getByTestId('sp-btn-fork').click();
    const spBranch = page.getByTestId('sp-branch-select');
    await expect(spBranch.locator('option')).toHaveCount(2);
    await expect(spBranch).toHaveValue('branch-1');

    // Open collapsible Developer Workbench drawer & verify modal semantics + bidirectional sync!
    await page.getByTestId('btn-toggle-workbench').click();
    const drawer = page.getByTestId('workbench-drawer');
    await expect(drawer).toBeVisible();
    await expect(drawer).toHaveAttribute('aria-modal', 'true');
    await expect(drawer.getByTestId('capacity-meter')).toHaveText('Context 9 / 24 (15 free)');
    await expect(drawer.getByTestId('branch-select')).toHaveValue('branch-1');

    // Background shortcuts (e.g. 't' for Tour) must be suppressed while modal Workbench drawer is open,
    // and Shift+Tab from the close button wraps inside the modal focus trap!
    await page.keyboard.press('t');
    expect((await getTelemetry(page)).tourActive).toBe(false);
    await page.keyboard.press('Shift+Tab');
    const focusInsideDrawer = await page.evaluate(() => {
      const d = document.querySelector('[data-testid="workbench-drawer"]');
      return Boolean(d && document.activeElement && d.contains(document.activeElement));
    });
    expect(focusInsideDrawer).toBe(true);

    // Step AR INSIDE the workbench drawer -> spatial app strip updates immediately!
    await drawer.getByTestId('btn-step-ar').click();
    await expect(drawer.getByTestId('capacity-meter')).toHaveText('Context 10 / 24 (14 free)');
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 10 / 24');

    // Escape closes drawer and restores focus to Workbench toggle button
    await page.keyboard.press('Escape');
    await expect(drawer).toBeHidden();
    await expect(page.getByTestId('btn-toggle-workbench')).toBeFocused();

    const tel = await getTelemetry(page);
    expect(tel.glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('4. Reduced-Motion, Idle-RAF Gating, Native Space/Enter Keyboard, and Clean Narrow 1024×800 Layout', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.setViewportSize({ width: 1024, height: 800 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Reduced motion detected & continuous animations disabled
    const tel1 = await getTelemetry(page);
    expect(tel1.reducedMotion).toBe(true);
    expect(tel1.glErrorCount).toBe(0);
    await expect(page.getByTestId('btn-replay-wave')).toBeDisabled();

    // Idle-RAF gating: no continuous spinning while idle
    const framesBefore = (await getTelemetry(page)).rafFramesRendered;
    await page.waitForTimeout(350);
    const framesAfter = (await getTelemetry(page)).rafFramesRendered;
    expect(
      framesAfter - framesBefore,
      `Expected idle-RAF gating (delta=${framesAfter - framesBefore})`,
    ).toBeLessThanOrEqual(2);

    // Narrow 1024×800 layout hygiene: no horizontal overflow and generous non-blank WebGL2 canvas
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow, 'Zero horizontal overflow at 1024×800').toBeLessThanOrEqual(1);
    const canvas = page.getByTestId('twotower-canvas');
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThanOrEqual(480);
    expect(box!.height).toBeGreaterThanOrEqual(450);
    await assertCanvasNonBlank(canvas, 'Narrow 1024×800 WebGL2 render');

    // Roving tabindex + ArrowRight on view tablist
    const macroTab = page.getByTestId('view-tab-macro');
    const gqaTab = page.getByTestId('view-tab-paired_gqa');
    await expect(macroTab).toHaveAttribute('tabindex', '0');
    await expect(gqaTab).toHaveAttribute('tabindex', '-1');
    await macroTab.focus();
    await page.keyboard.press('ArrowRight');
    await expect(gqaTab).toBeFocused();
    await expect(gqaTab).toHaveAttribute('aria-selected', 'true');
    await expect(gqaTab).toHaveAttribute('tabindex', '0');
    await expect(macroTab).toHaveAttribute('tabindex', '-1');

    // Native Space & Enter activation on focused buttons without hijacking
    const initBtn = page.getByTestId('sp-btn-init');
    await initBtn.focus();
    await page.keyboard.press('Space');
    await expect(page.getByTestId('sp-phase-badge')).toHaveText('block_initialized');

    const refineBtn = page.getByTestId('sp-btn-refine');
    await refineBtn.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('sp-phase-badge')).toHaveText('refinement_denoised');

    // Mobile 430×932 layout hygiene + inspector drawer focus entry & Escape focus restoration
    await page.setViewportSize({ width: 430, height: 932 });
    await waitFrames(page, 3);
    const mobileOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(mobileOverflow, 'Zero horizontal overflow at 430×932').toBeLessThanOrEqual(1);
    const mobileInspectorBtn = page.getByTestId('btn-toggle-inspector');
    await expect(mobileInspectorBtn).toBeVisible();
    await expect(mobileInspectorBtn).toHaveAttribute('aria-controls', 'sp-inspector-drawer');
    await mobileInspectorBtn.click();
    await expect(page.getByTestId('btn-close-mobile-inspector')).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(mobileInspectorBtn).toBeFocused();

    guards.assertClean();
  });

  test('5. GitHub Pages Subpath /llm-viz/twotower/ WebGL2 Build, HTTP 200 Assets & Interactive Cutaway', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${PAGES_BASE_URL}twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 5);

    // Subpath-prefixed JS and CSS assets loaded HTTP 200 with zero 404s, and Nano link points to /llm-viz/
    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(subpathCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);
    await expect(page.getByTestId('nav-to-nano')).toHaveAttribute('href', '/llm-viz/');

    const canvas = page.getByTestId('twotower-canvas');
    await expect(canvas).toBeVisible();
    await expect(page.locator('#webgl-fallback')).toBeHidden();
    await assertCanvasNonBlank(canvas, 'Subpath /llm-viz/twotower/ Macro WebGL2 frame');

    // Interactive diffusion step + Paired GQA cutaway switch under subpath build
    await page.getByTestId('sp-btn-init').click();
    await page.getByTestId('sp-btn-refine').click();
    await page.getByTestId('sp-btn-conf').click();
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 8 / 24');

    await page.getByTestId('view-tab-paired_gqa').click();
    await waitFrames(page, 5);
    await assertCanvasNonBlank(canvas, 'Subpath /llm-viz/twotower/ Paired GQA WebGL2 frame');

    const tel = await getTelemetry(page);
    expect(tel.viewMode).toBe('paired_gqa');
    expect(tel.currentAtlasDims).toEqual([128, 128]);
    expect(tel.glErrorCount).toBe(0);

    guards.assertClean();
  });
});

test.describe('Nemotron-Labs-TwoTower Stage TT4 · Walkthrough, Timeline, Sampler, Architecture & Perf Suite', () => {
  test('6. Guided 4-Act Walkthrough (28 Beats) & Strict User-Session Restoration', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Perform a distinct user exploration sequence BEFORE starting the tour:
    // Mock-AR step (P=8 -> 9), fork branch-1, switch back to mask_diffusion, Init Block -> Refine, select Bay 06
    await page.getByTestId('spatial-mode-select').selectOption('mock_ar');
    await page.getByTestId('sp-btn-mock-ar').click();
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 9 / 24');

    await page.getByTestId('sp-btn-fork').click();
    await expect(page.getByTestId('sp-branch-select')).toHaveValue('branch-1');

    await page.getByTestId('spatial-mode-select').selectOption('mask_diffusion');
    await page.getByTestId('sp-btn-init').click();
    await page.getByTestId('sp-btn-refine').click();
    await expect(page.getByTestId('sp-phase-badge')).toHaveText('refinement_denoised');

    await page.evaluate(() => window.__TWOTOWER_APP__?.selectBay(6, 'context'));
    await waitFrames(page, 3);

    // Record EXACT user exploration state prior to launching Guided Tour
    const preTourTel = await getTelemetry(page);
    const preTourState = await page.evaluate(() => window.__TWOTOWER_APP__?.getState());
    expect(preTourTel.recordIndex).toBe(3);
    expect(preTourTel.branchId).toBe('branch-1');
    expect(preTourTel.committedPrefixLen).toBe(9);
    expect(preTourTel.lifecyclePhase).toBe('refinement_denoised');
    expect(preTourTel.viewMode).toBe('macro');
    expect(preTourState?.selectedBayId).toBe(6);
    expect(preTourTel.activeSidebarTab).toBe('inspect');

    // Launch Guided Tour via topbar toggle button
    const tourToggle = page.getByTestId('tt-tour-toggle');
    await tourToggle.click();
    await waitForTwoTowerSettled(page);

    const tourCard = page.getByTestId('tt-tour-card');
    await expect(tourCard).toBeVisible();
    await expect(tourToggle).toHaveAttribute('aria-pressed', 'true');
    await expect(tourToggle).toHaveText('Exit Tour');
    // Tour isolation: Workbench drawer toggle must be disabled & 'w' ignored while tour is active
    await expect(page.getByTestId('btn-toggle-workbench')).toBeDisabled();
    await page.keyboard.press('w');
    await expect(page.getByTestId('workbench-drawer')).toBeHidden();

    // Verify Act I–IV chapter buttons, Beat 1 / 28, live metrics, provenance, and epistemic tag
    for (const [idx, label] of [
      [0, 'Act I'],
      [1, 'Act II'],
      [2, 'Act III'],
      [3, 'Act IV'],
    ] as const) {
      await expect(page.getByTestId(`tour-act-${idx}`)).toHaveText(label);
    }
    await expect(page.getByTestId('tour-act-0')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('tour-beat-counter')).toContainText('Beat 1 / 28');
    await expect(page.getByTestId('tour-btn-prev')).toBeDisabled();
    await expect(page.getByTestId('sp-btn-init')).toBeDisabled();

    const checkTourBeatClean = async (label: string) => {
      await expect(page.getByTestId('tour-beat-title')).not.toBeEmpty();
      await expect(page.getByTestId('tour-live-metrics')).not.toBeEmpty();
      const provText = await page.getByTestId('tour-provenance').textContent();
      expect(provText, `${label}: tour provenance`).toContain('Rec #');
      expect(provText, `${label}: tour provenance`).toContain('Cache ');
      await expect(page.getByTestId('tour-epistemic-tag')).not.toBeEmpty();
      const fullCardText = await tourCard.textContent();
      assertNoInfinityOrNaNText(fullCardText, `${label}: tour HUD card`);
    };

    await checkTourBeatClean('Beat 1 (Act I)');

    // Step forward with Next button
    await page.getByTestId('tour-btn-next').click();
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('tour-beat-counter')).toContainText('Beat 2 / 28');
    await expect(page.getByTestId('tour-btn-prev')).toBeEnabled();
    await checkTourBeatClean('Beat 2 (Act I)');

    const canvas = page.getByTestId('twotower-canvas');

    // Jump across Act II, Act III, and Act IV (wait until fly animation settles: rafActive === false!)
    await page.getByTestId('tour-act-1').click();
    await waitForTwoTowerSettled(page);
    expect((await getTelemetry(page)).tourActIndex).toBe(1);
    await expect(page.getByTestId('tour-act-1')).toHaveAttribute('aria-pressed', 'true');
    await checkTourBeatClean('Act II entry');

    await page.getByTestId('tour-act-2').click();
    await waitForTwoTowerSettled(page);
    expect((await getTelemetry(page)).tourActIndex).toBe(2);
    await expect(page.getByTestId('tour-act-2')).toHaveAttribute('aria-pressed', 'true');
    await checkTourBeatClean('Act III entry');

    // Advance one beat into Act III Sampler view (Beat 3.2 opens Sampler panel)
    await page.getByTestId('tour-btn-next').click();
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('sidebar-panel-sampler')).toBeVisible();
    assertNoInfinityOrNaNText(
      await page.getByTestId('sidebar-panel-sampler').textContent(),
      'Tour Act III Sampler panel',
    );
    await checkTourBeatClean('Act III beat 3.2');
    await assertCanvasNonBlank(canvas, 'Tour Act III Output WebGL2 frame');

    await page.getByTestId('tour-act-3').click();
    await waitForTwoTowerSettled(page);
    expect((await getTelemetry(page)).tourActIndex).toBe(3);
    await expect(page.getByTestId('tour-act-3')).toHaveAttribute('aria-pressed', 'true');
    await checkTourBeatClean('Act IV entry');
    await assertCanvasNonBlank(canvas, 'Tour Act IV WebGL2 frame');

    // Exit Tour -> verify EXACT pre-tour exploration session, cursor, branch, bay & view restored untouched!
    await page.getByTestId('tour-btn-exit').click();
    await waitForTwoTowerSettled(page);

    await expect(tourCard).toBeHidden();
    await expect(tourToggle).toHaveAttribute('aria-pressed', 'false');
    await expect(tourToggle).toHaveText('Guided Tour');

    const restoredTel = await getTelemetry(page);
    const restoredState = await page.evaluate(() => window.__TWOTOWER_APP__?.getState());
    expect(restoredTel.tourActive).toBe(false);
    expect(restoredTel.tourBeatIndex).toBe(-1);
    expect(restoredTel.recordIndex).toBe(preTourTel.recordIndex);
    expect(restoredTel.branchId).toBe(preTourTel.branchId);
    expect(restoredTel.committedPrefixLen).toBe(preTourTel.committedPrefixLen);
    expect(restoredTel.lifecyclePhase).toBe(preTourTel.lifecyclePhase);
    expect(restoredTel.viewMode).toBe(preTourTel.viewMode);
    expect(restoredTel.activeSidebarTab).toBe(preTourTel.activeSidebarTab);
    expect(restoredState?.selectedBayId).toBe(preTourState?.selectedBayId);
    await expect(page.getByTestId('sp-phase-badge')).toHaveText('refinement_denoised');
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 9 / 24');
    expect(restoredTel.glErrorCount).toBe(0);

    guards.assertClean();
  });

  test('7. Persistent Nested Timeline, Completed-Uncommitted vs Committed Boundary Warning & Branching', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Prefill node renders & active-branch delete protection starts disabled
    await expect(page.getByTestId('tt-timeline-dock')).toBeVisible();
    await expect(page.getByTestId('timeline-node-0')).toContainText('Prefill P=8');
    await expect(page.getByTestId('timeline-node-0')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByTestId('sp-btn-delete-branch')).toBeDisabled();

    // Block Init -> Refinement Eval -> Confidence Commit -> Completed (Uncommitted)
    await page.getByTestId('sp-btn-init').click();
    const blockGroup = page.getByTestId('timeline-block-0');
    await expect(blockGroup).toBeVisible();
    await expect(page.getByTestId('timeline-node-1')).toHaveAttribute(
      'data-record-kind',
      'diffusion_block_init',
    );
    await expect(page.getByTestId('timeline-node-1')).toContainText('Init');

    for (let iter = 0; iter < 8; iter++) {
      const phase = (await page.getByTestId('sp-phase-badge').textContent())?.trim();
      if (phase === 'completed_uncommitted') break;
      await page.getByTestId('sp-btn-refine').click();
      await page.getByTestId('sp-btn-conf').click();
    }

    await expect(page.getByTestId('sp-phase-badge')).toHaveText('completed_uncommitted');
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 8 / 24');
    await expect(page.getByTestId('timeline-node-2')).toHaveAttribute(
      'data-record-kind',
      'diffusion_refinement',
    );
    await expect(page.getByTestId('timeline-node-2')).toContainText('Eval r0');
    await expect(page.getByTestId('timeline-node-3')).toHaveAttribute(
      'data-record-kind',
      'diffusion_confidence_commit',
    );
    await expect(page.getByTestId('timeline-node-3')).toContainText('Commit r0');

    // Completed-Uncommitted provenance: clearly distinct from Committed-to-Context!
    await expect(blockGroup).toHaveClass(/is-uncommitted/);
    await expect(blockGroup).not.toHaveClass(/is-committed/);
    await expect(blockGroup).toContainText('Uncommitted');
    await expect(page.getByTestId('historical-denoiser-banner')).not.toHaveClass(/is-mismatch/);

    // Explicit Commit -> Context (P=8 -> 12)
    await page.getByTestId('sp-btn-commit').click();
    await waitFrames(page, 3);
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 12 / 24');
    await expect(blockGroup).toHaveClass(/is-committed/);
    await expect(blockGroup).toContainText('→ Context');

    // Wave 1 Historical Denoiser Boundary Warning MUST appear explicitly after P=8 -> P=12 commit!
    const histDenoiserBanner = page.getByTestId('historical-denoiser-banner');
    await expect(histDenoiserBanner).toBeVisible();
    await expect(histDenoiserBanner).toHaveClass(/is-mismatch/);
    await expect(histDenoiserBanner).toContainText('HISTORICAL DENOISER · BOUNDARY MISMATCH');
    await expect(histDenoiserBanner).toContainText('Source P=8');
    await expect(histDenoiserBanner).toContainText('Cur P=12');
    await expect(histDenoiserBanner).toContainText('does NOT belong to current Context boundary P=12');

    const canvasHistBadge = page.getByTestId('canvas-historical-badge');
    await expect(canvasHistBadge).toBeVisible();
    await expect(canvasHistBadge).toContainText('HISTORICAL DENOISER · BOUNDARY MISMATCH');
    await expect(canvasHistBadge).toContainText('P=8 (cur P=12)');
    await expect(page.getByTestId('timestep-conditioning-mode')).toHaveText(
      'Historical Prior Pass',
    );

    // Click-to-scrub on earlier timeline node (Eval r0 at cursorIndex=2)
    await page.getByTestId('timeline-node-2').click();
    await waitFrames(page, 3);
    const rewoundTel = await getTelemetry(page);
    expect(rewoundTel.recordIndex).toBe(2);
    expect(rewoundTel.committedPrefixLen).toBe(8);
    await expect(page.getByTestId('sp-capacity-badge')).toHaveText('Context 8 / 24');
    await expect(page.getByTestId('timeline-node-2')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByTestId('historical-config-card')).toContainText('Historical Record #2');
    await expect(page.getByTestId('sp-btn-fwd')).toBeEnabled();

    // Fork + Branch Switch + Strict Active-Branch Delete Protection
    await page.getByTestId('sp-btn-fork').click();
    const branchSelect = page.getByTestId('sp-branch-select');
    await expect(branchSelect.locator('option')).toHaveCount(2);
    await expect(branchSelect).toHaveValue('branch-1');
    await expect(page.getByTestId('sp-btn-delete-branch')).toBeEnabled();

    // Switch back to branch-0 and verify inactive branch-1 is deleted while active branch-0 is protected
    await branchSelect.selectOption('branch-0');
    expect((await getTelemetry(page)).branchId).toBe('branch-0');
    await expect(page.getByTestId('sp-btn-delete-branch')).toHaveAttribute(
      'title',
      /Delete inactive branch branch-1/,
    );
    await page.getByTestId('sp-btn-delete-branch').click();

    await expect(branchSelect.locator('option')).toHaveCount(1);
    await expect(branchSelect).toHaveValue('branch-0');
    expect((await getTelemetry(page)).branchId).toBe('branch-0');
    // Active branch cannot be deleted!
    await expect(page.getByTestId('sp-btn-delete-branch')).toBeDisabled();

    expect((await getTelemetry(page)).glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('8. Advanced Sampler Inspector, Presets (0.22/0.80/0.90), Independent B & K, Zero-Infinity & Non-Retroactive Scope', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Open Sampler sidebar tab
    await page.getByTestId('sidebar-tab-sampler').click();
    const samplerPanel = page.getByTestId('sidebar-panel-sampler');
    await expect(samplerPanel).toBeVisible();

    // Explicit calibration caveat banner
    const caveat = page.getByTestId('sampler-calibration-banner');
    await expect(caveat).toBeVisible();
    await expect(caveat).toContainText(
      'Unscaled confidence is a heuristic, not calibrated semantic correctness.',
    );
    await expect(caveat).toContainText('NOT a calibrated probability of semantic correctness');

    // Verify all 3 threshold presets (0.22 Mini Default / 0.80 Paper / 0.90 Ref Code)
    const presetMini = page.getByTestId('sampler-preset-mini');
    const presetPaper = page.getByTestId('sampler-preset-paper');
    const presetCode = page.getByTestId('sampler-preset-code');
    const thresholdInput = page.getByTestId('sampler-threshold-input');

    await expect(presetMini).toHaveClass(/is-active/);
    await expect(thresholdInput).toHaveValue('0.22');

    await presetCode.click();
    await expect(presetCode).toHaveClass(/is-active/);
    await expect(thresholdInput).toHaveValue('0.90');

    await presetPaper.click();
    await expect(presetPaper).toHaveClass(/is-active/);
    await expect(thresholdInput).toHaveValue('0.80');

    // Run 2 refinement+confidence steps at γ=0.80 so step 1 commits via budget floor and step 2 has a Locked (prior step) slot
    await page.getByTestId('sp-btn-init').click();
    await page.getByTestId('sp-btn-refine').click();
    await page.getByTestId('sp-btn-conf').click();

    await page.getByTestId('sp-btn-refine').click();
    await page.getByTestId('sp-btn-conf').click();
    await waitFrames(page, 3);

    // Verify summary counts (threshold-qualified vs budget-floor extra) & raw vs constrained MDLM rows
    const summaryCard = page.getByTestId('sampler-summary-card');
    await expect(summaryCard).toContainText('Executed t = nMasked/B');
    await expect(summaryCard).toContainText('Commitment Floor');
    await expect(summaryCard).toContainText('Qualified by Confidence');
    await expect(summaryCard).toContainText('Extra for Budget Floor');
    await expect(summaryCard).toContainText('Prior Locked');

    const slotsList = page.getByTestId('sampler-slots-list');
    await expect(slotsList).toBeVisible();
    for (let pos = 0; pos < 4; pos++) {
      const slot = page.getByTestId(`sampler-slot-${pos}`);
      await expect(slot).toBeVisible();
      await expect(slot).toContainText('Raw LM Top-3:');
      await expect(slot).toContainText('MDLM Top-3 ([MASK]=0%):');
    }

    const samplerFullText = await samplerPanel.textContent();
    expect(samplerFullText).toContain('Locked (prior step)');
    assertNoInfinityOrNaNText(samplerFullText, 'Advanced Sampler panel');

    // Independent B and K controls (changing B must not overwrite K, changing K must not overwrite B)
    const btnB8 = page.getByTestId('sampler-block-size-8');
    const stepsInput = page.getByTestId('sampler-steps-input');
    await expect(page.getByTestId('scope-badge-block-size')).toContainText('Next block');
    await expect(page.getByTestId('scope-badge-steps')).toContainText('Next block');
    await expect(page.getByTestId('scope-badge-gamma')).toContainText('Next refinement');

    await btnB8.click();
    await expect(btnB8).toHaveClass(/is-active/);
    await expect(stepsInput).toHaveValue('4');

    await stepsInput.fill('6');
    await stepsInput.dispatchEvent('change');
    await expect(stepsInput).toHaveValue('6');
    await expect(btnB8).toHaveClass(/is-active/);

    // Rewind to inspect history & modify pendingConfig -> Future-vs-Historical scope banner must appear without mutating historical record!
    await page.getByTestId('sp-btn-back').click();
    await presetCode.click();
    await waitFrames(page, 2);

    const diffBanner = page.getByTestId('config-scope-diff-banner');
    await expect(diffBanner).toBeVisible();
    await expect(diffBanner).toContainText('Historical Record #');
    await expect(diffBanner).toContainText('Pending edits');
    await expect(page.getByTestId('sampler-config-card')).toContainText('Historical: 0.80');
    await expect(page.getByTestId('sampler-config-card')).toContainText('Future pending: 0.90');

    // Confirm Inspect tab still displays frozen historical configAtStep (B=4, K=4, γ=0.80)
    await page.getByTestId('sidebar-tab-inspect').click();
    const histCard = page.getByTestId('historical-config-card');
    await expect(histCard).toContainText('B=4');
    await expect(histCard).toContainText('K=4');
    await expect(histCard).toContainText('γ=0.80');
    await expect(histCard).toContainText('Pending ≠ Hist');

    expect((await getTelemetry(page)).glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('9. Production Architecture & Context-Length Scaling Tab (63.19B / 126.37 GB, Caches, Ledger & SVG Chart)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    await page.getByTestId('sidebar-tab-architecture').click();
    const archPanel = page.getByTestId('sidebar-panel-architecture');
    await expect(archPanel).toBeVisible();

    // 1. 63.19B / Dual GB/GiB Dual-Tower Parameter Inventory + FP32 Bias Extra + Unit Convention
    const invCard = page.getByTestId('arch-inventory-card');
    await expect(invCard).toContainText('63.19B');
    await expect(invCard).toContainText('63.16 GB / 58.82 GiB');
    await expect(invCard).toContainText('63.22 GB / 58.87 GiB');
    await expect(invCard).toContainText('60.04 MB / 57.26 MiB');
    await expect(invCard).toContainText('126.37 GB / 117.69 GiB');
    await expect(invCard).toContainText('0 B audit diff');
    await expect(invCard).toContainText('24 Shards');
    const fp32Cell = page.getByTestId('arch-fp32-bias-cell');
    await expect(fp32Cell).toContainText('46 tensors');
    await expect(fp32Cell).toContainText('5,888 params');
    await expect(fp32Cell).toContainText('23,552 B (+11,776 B vs BF16)');
    await expect(page.getByTestId('arch-unit-convention-note')).toContainText('Decimal SI');
    await expect(page.getByTestId('arch-unit-convention-note')).toContainText('Binary IEC');

    // 2. "Two Resident Weight Sets ≠ Two Persistent Caches" + 4-Way Memory Separation & Peak Exclusion
    const wvcCard = page.getByTestId('arch-weights-vs-caches');
    await expect(wvcCard).toContainText('Two Resident Weight Sets ≠ Two Persistent Caches');
    await expect(wvcCard).toContainText('1 Persistent Cache');
    await expect(page.getByTestId('arch-bars-schematic-note')).toContainText(
      'Schematic (not linear to scale)',
    );
    await expect(page.getByTestId('arch-mem-persistent-cache')).toContainText(
      'Persistent Context Cache',
    );
    await expect(page.getByTestId('arch-mem-optional-mirrors')).toContainText('mamba_s2');
    await expect(page.getByTestId('arch-mem-optional-mirrors')).toContainText(
      'Cross-GPU Read-Only Mirror',
    );
    await expect(page.getByTestId('arch-mem-transient-kv')).toContainText(
      'Transient Denoiser Block KV',
    );
    const peakExcl = page.getByTestId('arch-peak-exclusion-banner');
    await expect(peakExcl).toContainText(
      'Accounted Weight & Cache Buffers ≠ Peak Inference Memory',
    );
    await expect(peakExcl).toContainText('Intermediate Layer Activations');
    await expect(peakExcl).toContainText('[B × 131,072]');
    await expect(peakExcl).toContainText('Allocator Fragmentation');

    // 3. Inline 3-Regime SVG Context-Length Scaling Chart (128K operational + 256K max_pos limit + 8,035 crossover + binary MiB/GiB)
    const scalingCard = page.getByTestId('arch-scaling-chart');
    await expect(scalingCard.locator('svg.sp-scaling-svg')).toBeVisible();
    await expect(scalingCard.getByTestId('scaling-operational-128k-boundary')).toBeAttached();
    await expect(scalingCard.getByTestId('scaling-max-pos-boundary')).toBeAttached();
    await expect(scalingCard.getByTestId('scaling-crossover-marker')).toBeAttached();
    await expect(scalingCard).toContainText('128K operational');
    await expect(scalingCard).toContainText('256K max_pos limit');
    await expect(scalingCard).toContainText('1 MiB');
    await expect(scalingCard).toContainText('100 GiB');
    await expect(scalingCard).toContainText('binary MiB/GiB');
    await expect(scalingCard).toContainText('Operational (≤128K)');
    await expect(scalingCard).toContainText('Configured Limit (≤256K)');
    await expect(scalingCard).toContainText('Extrapolated (>256K)');
    await expect(scalingCard).not.toContainText('Supported (≤256K)');
    await expect(scalingCard).toContainText('8,035 crossover');
    await expect(scalingCard).toContainText('Crossover: 8,035 tok');
    await expect(page.getByTestId('scaling-regime-disclaimer')).toContainText(
      'NOT empirically validated production runtime support',
    );

    // Zero mid-word clipping on scaling table headers & regime cells (e.g. PERSISTENT / Extrapolated (>256K))
    const scalingCellMetrics = await page.evaluate(() => {
      const cells = Array.from(
        document.querySelectorAll<HTMLElement>(
          '[data-testid="arch-scaling-table"] th, [data-testid="arch-scaling-table"] td',
        ),
      );
      return cells.map((cell) => ({
        text: (cell.textContent ?? '').trim(),
        whiteSpace: window.getComputedStyle(cell).whiteSpace,
        height: cell.getBoundingClientRect().height,
      }));
    });
    expect(scalingCellMetrics.length).toBeGreaterThan(10);
    for (const m of scalingCellMetrics) {
      expect(m.whiteSpace, `Scaling cell "${m.text}" must stay nowrap`).toBe('nowrap');
      expect(m.height, `Scaling cell "${m.text}" must render on 1 line without mid-word wrap`).toBeLessThan(
        26,
      );
    }

    // 4. Precision vs Miniature matrix
    await expect(page.getByTestId('arch-precision-table')).toContainText('BF16');
    await expect(page.getByTestId('arch-precision-table')).toContainText('FP64');

    // 5. Paper vs Checkpoint Discrepancy Ledger (collapsible <details>)
    const ledger = page.getByTestId('arch-discrepancy-ledger');
    await expect(ledger).toContainText('Paper vs Checkpoint Discrepancy Ledger');
    const items = ledger.locator('details.sp-discrepancy-item');
    expect(await items.count()).toBeGreaterThanOrEqual(3);
    await items.first().locator('summary').click();
    await expect(items.first()).toContainText('Resolution:');

    assertNoInfinityOrNaNText(await archPanel.textContent(), 'Architecture sidebar tab');
    expect((await getTelemetry(page)).glErrorCount).toBe(0);
    guards.assertClean();
  });

  test('10. Keyboard Navigation (T/Arrows/Esc/W/Tab), Reduced Motion Instant Transitions, 0 Geometry Rebuilds on Orbit & Idle RAF Sleep', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);

    // Reduced motion active & debug handle populated
    const initialTel = await getTelemetry(page);
    expect(initialTel.reducedMotion).toBe(true);
    const hasDebug = await page.evaluate(() => Boolean(window.__TWOTOWER_SPATIAL_DEBUG__));
    expect(hasDebug).toBe(true);

    // Keyboard shortcuts: T starts tour, ArrowRight/ArrowLeft step beats, Escape exits tour
    await page.getByTestId('twotower-canvas').focus();
    await page.keyboard.press('t');
    await waitFrames(page, 2);
    await expect(page.getByTestId('tt-tour-card')).toBeVisible();
    await expect(page.getByTestId('tour-beat-counter')).toContainText('Beat 1 / 28');
    expect((await getTelemetry(page)).tourActive).toBe(true);
    expect((await getTelemetry(page)).rafActive).toBe(false);

    await page.keyboard.press('ArrowRight');
    await waitFrames(page, 2);
    await expect(page.getByTestId('tour-beat-counter')).toContainText('Beat 2 / 28');
    expect((await getTelemetry(page)).tourBeatIndex).toBe(1);

    await page.keyboard.press('ArrowLeft');
    await waitFrames(page, 2);
    await expect(page.getByTestId('tour-beat-counter')).toContainText('Beat 1 / 28');
    expect((await getTelemetry(page)).tourBeatIndex).toBe(0);

    await page.keyboard.press('Escape');
    await waitFrames(page, 2);
    await expect(page.getByTestId('tt-tour-card')).toBeHidden();
    expect((await getTelemetry(page)).tourActive).toBe(false);
    await expect(page.getByTestId('tt-tour-toggle')).toBeFocused();

    // W toggles diagnostic workbench drawer open & closed
    await page.keyboard.press('w');
    await expect(page.getByTestId('workbench-drawer')).toBeVisible();
    await expect(page.getByTestId('btn-toggle-workbench')).toHaveAttribute('aria-expanded', 'true');

    await page.keyboard.press('w');
    await expect(page.getByTestId('workbench-drawer')).toBeHidden();
    await expect(page.getByTestId('btn-toggle-workbench')).toBeFocused();

    // Native Tab focusability across topbar buttons
    await page.getByTestId('tt-tour-toggle').focus();
    await page.keyboard.press('Tab');
    await expect(page.getByTestId('btn-reset-camera')).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(page.getByTestId('btn-toggle-workbench')).toBeFocused();

    // Instant non-animated view transition under reducedMotion: 'reduce'
    await page.evaluate(() => window.__TWOTOWER_SPATIAL_DEBUG__?.resetPerfCounters());
    await page.getByTestId('view-tab-paired_gqa').click();
    await waitFrames(page, 2);
    const afterSwitch = await getTelemetry(page);
    expect(afterSwitch.viewMode).toBe('paired_gqa');
    expect(afterSwitch.rafActive).toBe(false);
    expect(afterSwitch.rafFramesRendered).toBeLessThanOrEqual(2);

    // Return to Macro and verify 0 geometryRebuilds across camera orbit deltas + idle RAF sleep
    await page.getByTestId('view-tab-macro').click();
    await waitFrames(page, 3);
    await page.evaluate(() => window.__TWOTOWER_SPATIAL_DEBUG__?.resetPerfCounters());

    const orbitPerf = await page.evaluate(() => {
      const dbg = window.__TWOTOWER_SPATIAL_DEBUG__!;
      for (let i = 0; i < 6; i++) {
        dbg.orbitDelta(22, -6);
      }
      return dbg.getSnapshot();
    });
    expect(orbitPerf.geometryRebuilds).toBe(0);
    expect(orbitPerf.rafFramesRendered).toBeGreaterThanOrEqual(6);
    expect(orbitPerf.glErrorCount).toBe(0);

    // Idle RAF sleep: rafActive === false, rafIdlePauses > 0, zero spurious frames while idle
    await page.waitForTimeout(250);
    const idle1 = await getTelemetry(page);
    expect(idle1.rafActive).toBe(false);
    expect(idle1.rafIdlePauses).toBeGreaterThan(0);

    await page.waitForTimeout(250);
    const idle2 = await getTelemetry(page);
    expect(idle2.rafActive).toBe(false);
    expect(idle2.rafFramesRendered - idle1.rafFramesRendered).toBe(0);

    await assertCanvasNonBlank(page.getByTestId('twotower-canvas'), 'Post-orbit Macro WebGL2 frame');
    guards.assertClean();
  });

  test('11. GitHub Pages Subpath /llm-viz/twotower/ TT4 Parity (Tour, Timeline, Sampler & Architecture)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${PAGES_BASE_URL}twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 5);

    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length).toBeGreaterThan(0);
    expect(subpathCss.length).toBeGreaterThan(0);

    // Step diffusion once and verify Timeline + Sampler + Architecture tabs on subpath build
    await page.getByTestId('sp-btn-init').click();
    await page.getByTestId('sp-btn-refine').click();
    await page.getByTestId('sp-btn-conf').click();
    await expect(page.getByTestId('timeline-block-0')).toBeVisible();

    await page.getByTestId('sidebar-tab-sampler').click();
    await expect(page.getByTestId('sampler-calibration-banner')).toBeVisible();
    await expect(page.getByTestId('sampler-slots-list')).toBeVisible();
    assertNoInfinityOrNaNText(
      await page.getByTestId('sidebar-panel-sampler').textContent(),
      'Subpath Sampler panel',
    );

    await page.getByTestId('sidebar-tab-architecture').click();
    await expect(page.getByTestId('arch-inventory-card')).toContainText('63.19B');
    await expect(page.getByTestId('scaling-max-pos-boundary')).toBeAttached();
    await expect(page.getByTestId('scaling-crossover-marker')).toBeAttached();

    // Launch Guided Tour, jump to Act IV, exit and confirm user record restored on subpath
    const preTel = await getTelemetry(page);
    await page.getByTestId('tt-tour-toggle').click();
    await expect(page.getByTestId('tt-tour-card')).toBeVisible();
    await page.getByTestId('tour-act-3').click();
    await waitFrames(page, 3);
    assertNoInfinityOrNaNText(
      await page.getByTestId('tt-tour-card').textContent(),
      'Subpath Tour Act IV',
    );
    await page.getByTestId('tour-btn-exit').click();
    await waitFrames(page, 3);

    const postTel = await getTelemetry(page);
    expect(postTel.tourActive).toBe(false);
    expect(postTel.recordIndex).toBe(preTel.recordIndex);
    expect(postTel.activeSidebarTab).toBe('architecture');
    expect(postTel.glErrorCount).toBe(0);

    await assertCanvasNonBlank(
      page.getByTestId('twotower-canvas'),
      'Subpath /llm-viz/twotower/ post-tour WebGL2 frame',
    );
    guards.assertClean();
  });

  test('12. Reciprocal Cross-App Navigation (Nano <-> TwoTower via Click & Enter at Root / and Subpath /llm-viz/)', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    async function navViaSwitcher(
      targetTestId: string,
      expectedHref: string,
      expectedUrl: string,
      mode: 'click' | 'keyboard',
    ): Promise<void> {
      const trigger = page.getByTestId('model-switcher-trigger');
      await expect(trigger).toBeVisible();
      await trigger.click();
      const link = page.getByTestId(targetTestId);
      await expect(link).toBeVisible();
      await expect(link).toHaveAttribute('href', expectedHref);
      if (mode === 'keyboard') {
        await link.focus();
        await expect(link).toBeFocused();
        await Promise.all([page.waitForURL(expectedUrl), page.keyboard.press('Enter')]);
      } else {
        await Promise.all([page.waitForURL(expectedUrl), link.click()]);
      }
    }

    // 1. Root `/` -> `/twotower/` -> `/` via pointer click
    await page.goto(`${ROOT_BASE_URL}/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    await navViaSwitcher('nav-to-twotower', '/twotower/', `${ROOT_BASE_URL}/twotower/`, 'click');
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('twotower-canvas')).toBeVisible();
    await assertCanvasNonBlank(
      page.getByTestId('twotower-canvas'),
      'Root Nano -> TwoTower click navigation',
    );
    expect((await getTelemetry(page)).glErrorCount).toBe(0);

    await navViaSwitcher('nav-to-nano', '/', `${ROOT_BASE_URL}/`, 'click');
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    // 2. Root `/` -> `/twotower/` -> `/` via native keyboard Enter
    await navViaSwitcher('nav-to-twotower', '/twotower/', `${ROOT_BASE_URL}/twotower/`, 'keyboard');
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('twotower-canvas')).toBeVisible();
    expect((await getTelemetry(page)).glErrorCount).toBe(0);

    await navViaSwitcher('nav-to-nano', '/', `${ROOT_BASE_URL}/`, 'keyboard');
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    // 3. GitHub Pages Subpath `/llm-viz/` -> `/llm-viz/twotower/` -> `/llm-viz/` via Enter & click
    await page.goto(PAGES_BASE_URL, { waitUntil: 'networkidle' });
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    await navViaSwitcher(
      'nav-to-twotower',
      '/llm-viz/twotower/',
      `${PAGES_BASE_URL}twotower/`,
      'keyboard',
    );
    await waitForTwoTowerSettled(page);
    await expect(page.getByTestId('twotower-canvas')).toBeVisible();
    await assertCanvasNonBlank(
      page.getByTestId('twotower-canvas'),
      'Subpath /llm-viz/ -> /llm-viz/twotower/ Enter navigation',
    );
    expect((await getTelemetry(page)).glErrorCount).toBe(0);

    await navViaSwitcher('nav-to-nano', '/llm-viz/', PAGES_BASE_URL, 'click');
    await waitFrames(page, 4);
    await expect(page.getByTestId('gl-canvas')).toBeVisible();

    guards.assertClean();
  });
});
