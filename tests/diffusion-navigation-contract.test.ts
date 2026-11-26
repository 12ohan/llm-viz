import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = resolve(__dirname, '..');

const FORBIDDEN_DIFFUSION_MARKERS = [
  'DIFFUSION_MINI_CONFIG',
  'DiffusionSession',
  'buildDiffusionSpatialScene',
  'DIFFUSION_DISCREPANCY_LEDGER',
  'linear_spec_lora',
  'DIFFUSION_DYADIC_YARN_INV_FREQ',
] as const;

const FORBIDDEN_TWOTOWER_MARKERS = [
  'NemotronHTwoTowerForCausalLM',
  'twotower-denoiser-v1',
  'twotower-context-v1',
  'TwoTowerSession',
  'diffusion_block_commit',
  'TWOTOWER_SPEC',
] as const;

function stripJsComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '');
}

describe('Diffusion / TwoTower / Nano 3-Way Reciprocal Navigation & Release Metadata Contract', () => {
  it('exposes base-path-safe Nano -> Diffusion and TwoTower links in index.html with zero forbidden Diffusion markers', () => {
    const html = readFileSync(resolve(ROOT, 'index.html'), 'utf8');
    expect(html).toContain('data-testid="nav-to-twotower"');
    expect(html).toContain('href="%BASE_URL%twotower/"');
    expect(html).toContain('data-testid="nav-to-diffusion"');
    expect(html).toContain('href="%BASE_URL%diffusion/"');
    expect(html).toContain('class="top-nav-link"');

    const twotowerIdx = html.indexOf('data-testid="nav-to-twotower"');
    const diffusionIdx = html.indexOf('data-testid="nav-to-diffusion"');
    expect(twotowerIdx).toBeGreaterThan(-1);
    expect(diffusionIdx).toBeGreaterThan(twotowerIdx);

    for (const marker of FORBIDDEN_DIFFUSION_MARKERS) {
      expect(html).not.toContain(marker);
    }

    const nanoCss = readFileSync(resolve(ROOT, 'src/ui/theme.css'), 'utf8');
    expect(nanoCss).toContain('.top-nav-link');
    expect(nanoCss).toContain('.top-nav-link:focus-visible');
    for (const marker of FORBIDDEN_DIFFUSION_MARKERS) {
      expect(nanoCss).not.toContain(marker);
    }
  });

  it('uses clean release metadata in diffusion/index.html without dev stage tags or TwoTower imports', () => {
    const diffHtml = readFileSync(resolve(ROOT, 'diffusion/index.html'), 'utf8');
    expect(diffHtml).toContain(
      '<title>Nemotron-Labs-Diffusion-14B — Interactive 3D Architecture Visualizer</title>',
    );
    expect(diffHtml).toContain('property="og:title"');
    expect(diffHtml).toContain('property="og:description"');
    expect(diffHtml).toContain('id="diffusion-app"');
    expect(diffHtml).toContain('data-testid="diffusion-app"');
    expect(diffHtml).toContain('/src/diffusion/app/main.ts');
    expect(diffHtml).not.toMatch(/\bDF[1-5]\b/i);
    expect(diffHtml).not.toMatch(/Stage\s+DF\d/i);
    expect(diffHtml).not.toContain('src/twotower/');
    for (const marker of FORBIDDEN_TWOTOWER_MARKERS) {
      expect(diffHtml).not.toContain(marker);
    }
  });

  it('wires base-path-safe TwoTower -> Diffusion link after navToNanoLink without src/diffusion/ imports', () => {
    const src = readFileSync(resolve(ROOT, 'src/twotower/app/spatialApp.ts'), 'utf8');
    expect(src).toContain("navToNanoLink.dataset.testid = 'nav-to-nano'");
    expect(src).toContain("navToDiffusionLink.dataset.testid = 'nav-to-diffusion'");
    expect(src).toContain('`${baseUrl}diffusion/`');
    const nanoIdx = src.indexOf('navToNanoLink,');
    const diffIdx = src.indexOf('navToDiffusionLink,');
    expect(nanoIdx).toBeGreaterThan(-1);
    expect(diffIdx).toBeGreaterThan(nanoIdx);

    expect(src).not.toMatch(/from\s+['"][^'"]*diffusion\//);
    for (const marker of FORBIDDEN_DIFFUSION_MARKERS) {
      expect(src).not.toContain(marker);
    }
  });

  it('wires base-path-safe Diffusion -> Nano and TwoTower links in src/diffusion/app/spatialApp.ts without TwoTower imports or dev stage tags in UI strings', () => {
    const src = readFileSync(resolve(ROOT, 'src/diffusion/app/spatialApp.ts'), 'utf8');
    expect(src).toContain("import.meta.env.BASE_URL || '../'");
    expect(src).toContain("navToNanoLink.dataset.testid = 'nav-to-nano'");
    expect(src).toContain('navToNanoLink.href = baseUrl;');
    expect(src).toContain("navToTwoTowerLink.dataset.testid = 'nav-to-twotower'");
    expect(src).toContain('`${baseUrl}twotower/`');

    const nanoIdx = src.indexOf('navToNanoLink,');
    const ttIdx = src.indexOf('navToTwoTowerLink,');
    expect(nanoIdx).toBeGreaterThan(-1);
    expect(ttIdx).toBeGreaterThan(nanoIdx);

    expect(src).not.toContain('src/twotower/');
    expect(src).not.toMatch(/from\s+['"][^'"]*twotower\//);
    for (const marker of FORBIDDEN_TWOTOWER_MARKERS) {
      expect(src).not.toContain(marker);
    }

    const codeWithoutComments = stripJsComments(src);
    expect(codeWithoutComments).not.toMatch(/\bDF[1-5]\b/);
    expect(codeWithoutComments).not.toMatch(/Stage\s+DF\d/i);
  });
});
