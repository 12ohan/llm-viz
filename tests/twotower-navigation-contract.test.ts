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
  it('mounts compact model switcher in index.html / src/main.ts with zero forbidden Nano markers', () => {
    const html = readFileSync(resolve(ROOT, 'index.html'), 'utf8');
    expect(html).toContain('id="model-switcher-slot"');
    for (const marker of FORBIDDEN_NANO_MARKERS) {
      expect(html).not.toContain(marker);
    }

    const mainSrc = readFileSync(resolve(ROOT, 'src/main.ts'), 'utf8');
    expect(mainSrc).toContain("mountModelSwitcher(switcherSlot");
    expect(mainSrc).toContain("activeId: 'nano'");

    const nanoCss = readFileSync(resolve(ROOT, 'src/ui/theme.css'), 'utf8');
    expect(nanoCss).toContain('.top-bar');
    expect(nanoCss).toContain('overflow: visible;');
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

  it('mounts base-path-safe compact model switcher after btnToggleWorkbench in spatialApp.ts', () => {
    const src = readFileSync(resolve(ROOT, 'src/twotower/app/spatialApp.ts'), 'utf8');
    expect(src).toContain("import.meta.env.BASE_URL || '../'");
    expect(src).toContain("mountModelSwitcher(topActions");
    expect(src).toContain("activeId: 'twotower'");
    const wbIdx = src.indexOf('btnToggleWorkbench,');
    const switcherIdx = src.indexOf('mountModelSwitcher(topActions');
    expect(wbIdx).toBeGreaterThan(-1);
    expect(switcherIdx).toBeGreaterThan(wbIdx);
  });
});
