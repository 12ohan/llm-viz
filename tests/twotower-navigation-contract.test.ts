import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = resolve(__dirname, '..');

const FORBIDDEN_NANO_MARKERS = [
  'NemotronHTwoTowerForCausalLM',
  'twotower-denoiser-v1',
  'twotower-context-v1',
  'TwoTowerSession',
  'diffusion_block_commit',
  'TWOTOWER_SPEC',
] as const;

describe('TwoTower / Nano Reciprocal Navigation & Release Metadata Contract', () => {
  it('exposes base-path-safe Nano -> TwoTower link in index.html with zero forbidden Nano markers', () => {
    const html = readFileSync(resolve(ROOT, 'index.html'), 'utf8');
    expect(html).toContain('data-testid="nav-to-twotower"');
    expect(html).toContain('href="%BASE_URL%twotower/"');
    expect(html).toContain('class="top-nav-link"');
    for (const marker of FORBIDDEN_NANO_MARKERS) {
      expect(html).not.toContain(marker);
    }

    const nanoCss = readFileSync(resolve(ROOT, 'src/ui/theme.css'), 'utf8');
    expect(nanoCss).toContain('.top-nav-link');
    expect(nanoCss).toContain('.top-nav-link:focus-visible');
    for (const marker of FORBIDDEN_NANO_MARKERS) {
      expect(nanoCss).not.toContain(marker);
    }
  });

  it('uses clean release metadata in twotower/index.html and diagnosticUi.ts without dev stage tags', () => {
    const ttHtml = readFileSync(resolve(ROOT, 'twotower/index.html'), 'utf8');
    expect(ttHtml).toContain(
      '<title>Nemotron-Labs-TwoTower — Interactive 3D Architecture Visualizer</title>',
    );
    expect(ttHtml).toContain('property="og:title"');
    expect(ttHtml).toContain('property="og:description"');
    expect(ttHtml).not.toMatch(/Stage\s+TT\d/i);

    const diagSrc = readFileSync(resolve(ROOT, 'src/twotower/app/diagnosticUi.ts'), 'utf8');
    expect(diagSrc).toContain("workbench.setAttribute('role', 'region')");
    expect(diagSrc).not.toContain("workbench.setAttribute('role', 'application')");
    expect(diagSrc).not.toMatch(/Stage\s+TT\d/i);
  });

  it('wires base-path-safe TwoTower -> Nano link after btnToggleWorkbench in spatialApp.ts', () => {
    const src = readFileSync(resolve(ROOT, 'src/twotower/app/spatialApp.ts'), 'utf8');
    expect(src).toContain("navToNanoLink.dataset.testid = 'nav-to-nano'");
    expect(src).toContain("import.meta.env.BASE_URL || '../'");
    const wbIdx = src.indexOf('btnToggleWorkbench,');
    const nanoIdx = src.indexOf('navToNanoLink,');
    expect(wbIdx).toBeGreaterThan(-1);
    expect(nanoIdx).toBeGreaterThan(wbIdx);
  });
});
