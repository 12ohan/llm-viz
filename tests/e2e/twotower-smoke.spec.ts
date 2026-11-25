import { expect, test, type Page } from '@playwright/test';
import { PAGES_BASE_URL, ROOT_BASE_URL } from '../../playwright.config';

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

function assertNoInfinityText(text: string | null, context: string): void {
  const safe = text ?? '';
  expect(safe, `${context}: must never render "Infinity"`).not.toMatch(/Infinity/i);
  expect(safe, `${context}: must never render "∞"`).not.toContain('∞');
}

test.describe('Nemotron-Labs-TwoTower Stage TT2 · Diagnostic UI Smoke Suite', () => {
  test('1. Root /twotower/ boot, epistemic disclosure, defaults, split prefill provenance, and persistent_cache', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });

    await expect(page.getByTestId('twotower-app')).toBeVisible();
    await expect(page.locator('h1')).toHaveText(
      'Nemotron-Labs-TwoTower — Diagnostic Session Workbench (Stage TT2)',
    );

    // Stage TT2 must NOT expose a cross-app navigation link back to Nano
    await expect(page.locator('a[href="../"]')).toHaveCount(0);

    // Epistemic disclosure & threshold calibration contract
    const epistemic = page.getByTestId('epistemic-banner');
    await expect(epistemic).toBeVisible();
    await expect(epistemic).toContainText('d = 8');
    await expect(epistemic).toContainText('V = 34');
    await expect(epistemic).toContainText('L = 52');
    await expect(epistemic).toContainText('kvCapacity = 24');
    await expect(epistemic).toContainText('35b64985');
    await expect(epistemic).toContainText('γ = 0.22');
    await expect(epistemic).toContainText('0.80');
    await expect(epistemic).toContainText('0.90');

    // Default session controls and status badges
    await expect(page.getByTestId('lifecycle-phase')).toHaveText('ready');
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 8 / 24 (16 free)');
    await expect(page.getByTestId('mode-select')).toHaveValue('mask_diffusion');
    await expect(page.getByTestId('block-size-select')).toHaveValue('4');
    await expect(page.getByTestId('steps-input')).toHaveValue('4');
    await expect(page.getByTestId('threshold-input')).toHaveValue('0.22');
    await expect(page.getByTestId('preset-mini')).toHaveClass(/is-active/);
    await expect(page.getByTestId('preset-paper')).toBeVisible();
    await expect(page.getByTestId('preset-code')).toBeVisible();

    // Context prefill split-segment provenance + persistent_cache + eager context_computed trace
    const inspector = page.getByTestId('inspector-panel');
    await expect(inspector).toContainText('split_prefix_for_mamba_s2');
    await expect(inspector).toContainText('split_final_token');
    await expect(inspector).toContainText('persistent_cache');
    await expect(inspector).toContainText('context_computed');
    await expect(inspector).toContainText('70,656 B');

    // Assets loaded cleanly HTTP 200
    expect(guards.loadedJsAssets.length, 'Expected root TwoTower JS assets HTTP 200').toBeGreaterThan(
      0,
    );
    expect(
      guards.loadedCssAssets.length,
      'Expected root TwoTower CSS assets HTTP 200',
    ).toBeGreaterThan(0);

    guards.assertClean();
  });

  test('2. Full Diffusion Lifecycle: Init -> Denoise -> Confidence Commit -> completed_uncommitted -> Explicit Block Commit', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });

    // Initialize [MASK]^4 block
    await page.getByTestId('btn-init-block').click();
    await expect(page.getByTestId('lifecycle-phase')).toHaveText('block_initialized');
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 8 / 24 (16 free)');
    await expect(page.getByTestId('btn-commit-block')).toBeDisabled();

    // Step explicitly through Denoiser Pass -> Commit Confidence until completed_uncommitted
    let refinementPairs = 0;
    for (let iter = 0; iter < 8; iter++) {
      const phase = (await page.getByTestId('lifecycle-phase').textContent())?.trim();
      if (phase === 'completed_uncommitted') break;

      await expect(page.getByTestId('btn-run-refinement')).toBeEnabled();
      await page.getByTestId('btn-run-refinement').click();
      await expect(page.getByTestId('lifecycle-phase')).toHaveText('refinement_denoised');
      await expect(page.getByTestId('inspector-panel')).toContainText('denoiser_scratch');
      await expect(page.getByTestId('capacity-meter')).toHaveText('Context 8 / 24 (16 free)');

      await expect(page.getByTestId('btn-commit-confidence')).toBeEnabled();
      await page.getByTestId('btn-commit-confidence').click();
      refinementPairs += 1;

      // Context prefix MUST remain 8 / 24 until explicit block commit
      await expect(page.getByTestId('capacity-meter')).toHaveText('Context 8 / 24 (16 free)');

      const samplerText = await page.getByTestId('sampler-panel').textContent();
      assertNoInfinityText(samplerText, `Sampler panel after refinement pair ${refinementPairs}`);
      expect(samplerText).toContain('[MASK]=0.0000');
      if (refinementPairs >= 2) {
        expect(samplerText).toContain('Locked (prior step)');
      }
    }

    // Must land in explicit completed_uncommitted state with Context STILL 8 / 24
    expect(refinementPairs).toBeGreaterThanOrEqual(1);
    await expect(page.getByTestId('lifecycle-phase')).toHaveText('completed_uncommitted');
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 8 / 24 (16 free)');
    await expect(page.getByTestId('btn-commit-block')).toBeEnabled();
    assertNoInfinityText(
      await page.getByTestId('sampler-panel').textContent(),
      'Sampler panel at completed_uncommitted',
    );

    // Explicit Block Commit -> Context advances 8 -> 12 / 24 with split-segment commit provenance
    await page.getByTestId('btn-commit-block').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 12 / 24 (12 free)');

    const inspector = page.getByTestId('inspector-panel');
    await expect(inspector).toContainText('diffusion_block_commit');
    await expect(inspector).toContainText('split_prefix_for_mamba_s2');
    await expect(inspector).toContainText('range [8..10]');
    await expect(inspector).toContainText('split_final_token');
    await expect(inspector).toContainText('range [11..11]');

    // Deduplicated conditioning base count stays exactly 1 for the completed block
    await expect(page.getByTestId('mem-unique-block-conditioning-count')).toHaveText('1');

    guards.assertClean();
  });

  test('3. AR, Mock-AR (unmodulated denoiser_scratch), Mode Switch, Rewind, and Branch Forking', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${ROOT_BASE_URL}/twotower/`, { waitUntil: 'networkidle' });

    // Execute Step Mock-AR on default 8-token prompt (8 -> 9) and verify unmodulated denoiser_scratch
    await page.getByTestId('mode-select').selectOption('mock_ar');
    await page.getByTestId('btn-step-mock-ar').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 9 / 24 (15 free)');
    await expect(page.getByTestId('lifecycle-phase')).toHaveText('ready');

    const inspector = page.getByTestId('inspector-panel');
    await expect(inspector).toContainText('denoiser_scratch');
    await expect(inspector).toContainText('tModel=null');
    await expect(inspector).toContainText('bypassed (unmodulated Mock-AR)');

    // Switch mode to AR and execute Step AR (9 -> 10)
    await page.getByTestId('mode-select').selectOption('ar');
    await page.getByTestId('btn-step-ar').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 10 / 24 (14 free)');
    await expect(inspector).toContainText('single_segment');

    // Rewind backward via <- Back (10 -> 9)
    await page.getByTestId('btn-nav-back').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 9 / 24 (15 free)');
    await expect(page.getByTestId('btn-nav-forward')).toBeEnabled();

    // Fork a new branch at rewound cursor (#1)
    await page.getByTestId('btn-fork-branch').click();
    const branchSelect = page.getByTestId('branch-select');
    await expect(branchSelect.locator('option')).toHaveCount(2);
    await expect(branchSelect).toHaveValue('branch-1');
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 9 / 24 (15 free)');

    // Switch back to original branch-0 and verify its full 3-record trajectory (#0..#2, 10/24) is intact
    await branchSelect.selectOption('branch-0');
    await expect(branchSelect).toHaveValue('branch-0');
    await expect(page.getByTestId('history-list').locator('button')).toHaveCount(3);
    await page.getByTestId('btn-nav-forward').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 10 / 24 (14 free)');

    // Also verify Short 4 tokens preset runs Step AR (4 -> 5) -> Step Mock-AR (5 -> 6) cleanly
    await page.getByTestId('prompt-select').selectOption('short_4');
    await page.getByTestId('btn-restart').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 4 / 24 (20 free)');
    await page.getByTestId('btn-step-ar').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 5 / 24 (19 free)');
    await page.getByTestId('mode-select').selectOption('mock_ar');
    await page.getByTestId('btn-step-mock-ar').click();
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 6 / 24 (18 free)');
    await expect(inspector).toContainText('denoiser_scratch');
    await expect(inspector).toContainText('tModel=null');

    guards.assertClean();
  });

  test('4. GitHub Pages Subpath /llm-viz/twotower/ assets HTTP 200 and interactive diffusion smoke', async ({
    page,
  }) => {
    const guards = attachRuntimeGuards(page);

    await page.goto(`${PAGES_BASE_URL}twotower/`, { waitUntil: 'networkidle' });

    // Confirm subpath-prefixed JS and CSS assets loaded HTTP 200 with zero 404s
    const subpathJs = guards.loadedJsAssets.filter((u) => u.includes('/llm-viz/assets/'));
    const subpathCss = guards.loadedCssAssets.filter((u) => u.includes('/llm-viz/assets/'));
    expect(subpathJs.length, 'Expected /llm-viz/assets/*.js HTTP 200').toBeGreaterThan(0);
    expect(subpathCss.length, 'Expected /llm-viz/assets/*.css HTTP 200').toBeGreaterThan(0);

    await expect(page.getByTestId('lifecycle-phase')).toHaveText('ready');
    await expect(page.getByTestId('capacity-meter')).toHaveText('Context 8 / 24 (16 free)');

    // Interactive block init + denoiser pass + confidence commit under /llm-viz/twotower/
    await page.getByTestId('btn-init-block').click();
    await expect(page.getByTestId('lifecycle-phase')).toHaveText('block_initialized');

    await page.getByTestId('btn-run-refinement').click();
    await expect(page.getByTestId('lifecycle-phase')).toHaveText('refinement_denoised');

    await page.getByTestId('btn-commit-confidence').click();
    const samplerPanel = page.getByTestId('sampler-panel');
    await expect(samplerPanel.locator('table tbody tr')).toHaveCount(4);
    assertNoInfinityText(
      await samplerPanel.textContent(),
      'Subpath /llm-viz/twotower/ sampler panel (step 1)',
    );

    // If block still has masked slots, run one additional refinement + commit to verify locked-slot display
    if (await page.getByTestId('btn-run-refinement').isEnabled()) {
      await page.getByTestId('btn-run-refinement').click();
      await page.getByTestId('btn-commit-confidence').click();
      const samplerText2 = await samplerPanel.textContent();
      assertNoInfinityText(samplerText2, 'Subpath /llm-viz/twotower/ sampler panel (step 2)');
      expect(samplerText2).toContain('Locked (prior step)');
    }

    guards.assertClean();
  });
});
