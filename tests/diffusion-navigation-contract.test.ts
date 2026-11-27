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

describe('Diffusion / TwoTower / Nano Multi-App Navigation & Release Metadata Contract', () => {
  it('mounts compact model switcher slot in index.html with zero forbidden Diffusion markers', () => {
    const html = readFileSync(resolve(ROOT, 'index.html'), 'utf8');
    expect(html).toContain('id="model-switcher-slot"');

    for (const marker of FORBIDDEN_DIFFUSION_MARKERS) {
      expect(html).not.toContain(marker);
    }

    const nanoCss = readFileSync(resolve(ROOT, 'src/ui/theme.css'), 'utf8');
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

  it('mounts compact model switcher in TwoTower without src/diffusion/ imports or markers', () => {
    const src = readFileSync(resolve(ROOT, 'src/twotower/app/spatialApp.ts'), 'utf8');
    expect(src).toContain("mountModelSwitcher(topActions");
    expect(src).toContain("activeId: 'twotower'");

    expect(src).not.toMatch(/from\s+['"][^'"]*diffusion\//);
    for (const marker of FORBIDDEN_DIFFUSION_MARKERS) {
      expect(src).not.toContain(marker);
    }
  });

  it('mounts compact model switcher in src/diffusion/app/spatialApp.ts without TwoTower imports or dev stage tags in UI strings', () => {
    const src = readFileSync(resolve(ROOT, 'src/diffusion/app/spatialApp.ts'), 'utf8');
    expect(src).toContain("import.meta.env.BASE_URL || '../'");
    expect(src).toContain("mountModelSwitcher(topActions");
    expect(src).toContain("activeId: 'diffusion'");

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
