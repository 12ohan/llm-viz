import { defineConfig, devices } from '@playwright/test';

/**
 * Headless Chromium software WebGL2 launch flags suitable for both macOS local runs
 * and GPU-less Ubuntu GitHub Actions runners (ANGLE + SwiftShader Vulkan/GL backend).
 */
export const HEADLESS_WEBGL2_CHROMIUM_ARGS: string[] = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-webgl',
  '--ignore-gpu-blocklist',
  // Required by Chromium 137+ headless on GPU-less CI runners to allow software SwiftShader WebGL2
  '--enable-unsafe-swiftshader',
];

export const ROOT_PREVIEW_PORT = 4173;
export const PAGES_PREVIEW_PORT = 4174;
export const ROOT_BASE_URL = `http://127.0.0.1:${ROOT_PREVIEW_PORT}`;
export const PAGES_BASE_URL = `http://127.0.0.1:${PAGES_PREVIEW_PORT}/llm-viz/`;

const SKIP_BUILD = process.env.PLAYWRIGHT_SKIP_BUILD === '1';

export default defineConfig({
  testDir: './tests/e2e',
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : 2,
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
  ],
  outputDir: 'test-results/playwright',
  use: {
    baseURL: ROOT_BASE_URL,
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [
    {
      name: 'chromium-webgl2',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
        headless: true,
        launchOptions: {
          args: HEADLESS_WEBGL2_CHROMIUM_ARGS,
        },
      },
    },
  ],
  webServer: [
    {
      // Built production bundle served at root `/` (dedicated port 4173, never conflicts with dev 3000/5173)
      command: SKIP_BUILD
        ? `npx vite preview --outDir dist --base / --port ${ROOT_PREVIEW_PORT} --strictPort --host 127.0.0.1`
        : `npx vite build --outDir dist && npx vite preview --outDir dist --base / --port ${ROOT_PREVIEW_PORT} --strictPort --host 127.0.0.1`,
      url: `${ROOT_BASE_URL}/`,
      reuseExistingServer: false,
      timeout: 90_000,
      env: {
        ...process.env,
        VITE_BASE_PATH: '/',
      },
    },
    {
      // Built production bundle for GitHub Pages subpath `/llm-viz/` in gitignored test-results/
      command: SKIP_BUILD
        ? `npx vite preview --outDir test-results/dist-pages --base /llm-viz/ --port ${PAGES_PREVIEW_PORT} --strictPort --host 127.0.0.1`
        : `npx vite build --outDir test-results/dist-pages && npx vite preview --outDir test-results/dist-pages --base /llm-viz/ --port ${PAGES_PREVIEW_PORT} --strictPort --host 127.0.0.1`,
      url: PAGES_BASE_URL,
      reuseExistingServer: false,
      timeout: 90_000,
      env: {
        ...process.env,
        VITE_BASE_PATH: '/llm-viz/',
      },
    },
  ],
});
