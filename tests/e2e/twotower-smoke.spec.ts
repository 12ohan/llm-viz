import { inflateSync } from 'node:zlib';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';
import type { TwoTowerAppHandle, TwoTowerTelemetrySnapshot } from '../../src/twotower/app/spatialApp';

declare global {
  interface Window {
    __TWOTOWER_TELEMETRY__?: TwoTowerTelemetrySnapshot;
    __TWOTOWER_APP__?: TwoTowerAppHandle;
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

test.describe('Nemotron-Labs-TwoTower Stage TT3 · Interactive WebGL2 Spatial Smoke Suite', () => {
  test('1. Root /twotower/ WebGL2 Dual-Tower Macro boot, epistemic contract, linker channels, and 10-axis bay picking', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 5);

    // Must NOT expose any cross-app navigation link back to Nano (deferred to TT5)
    await expect(page.locator('a[href="../"]')).toHaveCount(0);

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

    // Epistemic miniature disclosure contract
    const pill = page.getByTestId('epistemic-pill');
    await expect(pill).toContainText('d=8');
    await expect(pill).toContainText('V=34');
    await expect(pill).toContainText('2×52L');
    await expect(pill).toContainText('Synthetic Mini');
    await expect(page.getByTestId('inspector-telemetry-footer')).toContainText('35b64985');

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

    // Open collapsible Developer Workbench drawer & verify bidirectional sync!
    await page.getByTestId('btn-toggle-workbench').click();
    const drawer = page.getByTestId('workbench-drawer');
    await expect(drawer).toBeVisible();
    await expect(drawer.getByTestId('capacity-meter')).toHaveText('Context 9 / 24 (15 free)');
    await expect(drawer.getByTestId('branch-select')).toHaveValue('branch-1');

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

    guards.assertClean();
  });

  test('5. GitHub Pages Subpath /llm-viz/twotower/ WebGL2 Build, HTTP 200 Assets & Interactive Cutaway', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${PAGES_BASE_URL}twotower/`, { waitUntil: 'networkidle' });
    await waitFrames(page, 5);

    // Subpath-prefixed JS and CSS assets loaded HTTP 200 with zero 404s
    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(subpathCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

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
