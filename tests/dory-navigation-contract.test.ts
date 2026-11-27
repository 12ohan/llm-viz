import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { buildDorySwitcherEntries } from '../src/dory/app/spatialApp';
import { DORY_APP_THEME } from '../src/dory/app/types';
import { MODEL_REGISTRY, resolveModelHref } from '../src/shared/modelRegistry';
import { mountModelSwitcher } from '../src/shared/modelSwitcher';

const ROOT = resolve(__dirname, '..');

const FORBIDDEN_DORY_MARKERS = [
  'DORY_MINI_CONFIG',
  'DorySession',
  'buildDorySpatialScene',
  'DORY_DISCREPANCY_LEDGER',
  'runDorySelfSpecCycle',
  'dory-nlerp-residual',
  '__dorySpatialDebug',
  'output_shallow',
] as const;

const FORBIDDEN_SIBLING_MARKERS = [
  'NemotronHTwoTowerForCausalLM',
  'TwoTowerSession',
  'TWOTOWER_SPEC',
  'DIFFUSION_MINI_CONFIG',
  'DiffusionSession',
  'buildDiffusionSpatialScene',
  'UNI_MINI_CONFIG',
  'UniSession',
  'buildUniSpatialScene',
  'PackedLinearImageCodec',
] as const;

type Listener = (ev: unknown) => void;

class MockElement {
  readonly tagName: string;
  id = '';
  className = '';
  textContent = '';
  title = '';
  type = '';
  href = '';
  hidden = false;
  disabled = false;
  isConnected = true;
  parentNode: MockElement | null = null;
  readonly children: MockElement[] = [];
  readonly dataset: Record<string, string> = {};
  readonly style: Record<string, string> = {};
  readonly attrs = new Map<string, string>();
  readonly listeners = new Map<string, Listener[]>();
  rect = { left: 0, right: 90, top: 0, bottom: 26, width: 90, height: 26 };

  getBoundingClientRect() {
    return this.rect;
  }

  readonly classList = {
    set: new Set<string>(),
    toggle: (cls: string, force?: boolean) => {
      if (force === undefined) {
        if (this.classList.set.has(cls)) this.classList.set.delete(cls);
        else this.classList.set.add(cls);
      } else if (force) {
        this.classList.set.add(cls);
      } else {
        this.classList.set.delete(cls);
      }
    },
  };

  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }

  appendChild(child: MockElement): MockElement {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  append(...nodes: MockElement[]): void {
    for (const n of nodes) this.appendChild(n);
  }

  removeChild(child: MockElement): MockElement {
    const idx = this.children.indexOf(child);
    if (idx >= 0) this.children.splice(idx, 1);
    child.parentNode = null;
    return child;
  }

  setAttribute(k: string, v: string): void {
    this.attrs.set(k, v);
  }

  getAttribute(k: string): string | null {
    return this.attrs.get(k) ?? null;
  }

  addEventListener(type: string, fn: Listener): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(fn);
    this.listeners.set(type, arr);
  }

  removeEventListener(type: string, fn: Listener): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    const idx = arr.indexOf(fn);
    if (idx >= 0) arr.splice(idx, 1);
  }

  dispatch(type: string, ev: Record<string, unknown>): void {
    for (const fn of [...(this.listeners.get(type) ?? [])]) {
      fn(ev);
    }
  }

  contains(other: unknown): boolean {
    if (other === this) return true;
    for (const c of this.children) {
      if (c.contains(other)) return true;
    }
    return false;
  }

  focus(): void {
    (globalThis as unknown as { document: { activeElement: unknown } }).document.activeElement = this;
  }
}

describe('Dory1.0-4B 5-Way MPA Navigation, Switcher & Release Metadata Contract', () => {
  it('registers dory as the 5th canonical entry in MODEL_REGISTRY for root / and /llm-viz/', () => {
    expect(MODEL_REGISTRY.map((e) => e.id)).toEqual([
      'nano',
      'twotower',
      'diffusion',
      'uni',
      'dory',
    ]);

    expect(MODEL_REGISTRY.map((e) => resolveModelHref('/', e))).toEqual([
      '/',
      '/twotower/',
      '/diffusion/',
      '/uni/',
      '/dory/',
    ]);
    expect(MODEL_REGISTRY.map((e) => resolveModelHref('/llm-viz/', e))).toEqual([
      '/llm-viz/',
      '/llm-viz/twotower/',
      '/llm-viz/diffusion/',
      '/llm-viz/uni/',
      '/llm-viz/dory/',
    ]);
    expect(MODEL_REGISTRY.map((e) => resolveModelHref('/llm-viz', e))).toEqual([
      '/llm-viz/',
      '/llm-viz/twotower/',
      '/llm-viz/diffusion/',
      '/llm-viz/uni/',
      '/llm-viz/dory/',
    ]);

    const doryMeta = MODEL_REGISTRY.find((e) => e.id === 'dory');
    expect(doryMeta).toEqual({
      id: 'dory',
      shortLabel: 'Dory1.0-4B',
      title: 'Nemotron-Labs-Dory1.0-4B',
      subtitle: '72L Looped nGPT · SWA/Full + Self-Spec',
      pathSuffix: 'dory/',
      testId: 'nav-to-dory',
    });

    // Once registered in MODEL_REGISTRY, buildDorySwitcherEntries() returns the registry directly with 0 duplicates
    expect(buildDorySwitcherEntries()).toBe(MODEL_REGISTRY);
  });

  it('uses clean release metadata in dory/index.html and wires all 5 MPA inputs in vite.config.ts', () => {
    const doryHtml = readFileSync(resolve(ROOT, 'dory/index.html'), 'utf8');
    expect(doryHtml).toContain(
      '<title>Nemotron-Labs-Dory1.0-4B — Interactive 3D Architecture Visualizer</title>',
    );
    expect(doryHtml).toContain('property="og:title"');
    expect(doryHtml).toContain('property="og:description"');
    expect(doryHtml).toContain('id="dory-app"');
    expect(doryHtml).toContain('data-testid="dory-app"');
    expect(doryHtml).toContain('/src/dory/app/main.ts');
    expect(doryHtml).not.toMatch(/\bDORY[0-7]\b/);
    expect(doryHtml).not.toMatch(/Stage\s+DORY\d/i);
    for (const marker of FORBIDDEN_SIBLING_MARKERS) {
      expect(doryHtml).not.toContain(marker);
    }

    const viteCfg = readFileSync(resolve(ROOT, 'vite.config.ts'), 'utf8');
    expect(viteCfg).toContain("resolve(__dirname, 'index.html')");
    expect(viteCfg).toContain("resolve(__dirname, 'twotower/index.html')");
    expect(viteCfg).toContain("resolve(__dirname, 'diffusion/index.html')");
    expect(viteCfg).toContain("resolve(__dirname, 'uni/index.html')");
    expect(viteCfg).toContain("resolve(__dirname, 'dory/index.html')");
  });

  it('keeps shared switcher and all 4 prior entry shells free of forbidden Dory markers and imports', () => {
    const isolatedFiles = [
      'src/shared/modelRegistry.ts',
      'src/shared/modelSwitcher.ts',
      'src/shared/modelSwitcher.css',
      'index.html',
      'src/main.ts',
      'src/ui/theme.css',
      'twotower/index.html',
      'diffusion/index.html',
      'uni/index.html',
      'src/twotower/app/spatialApp.ts',
      'src/diffusion/app/spatialApp.ts',
      'src/uni/app/spatialApp.ts',
    ];
    for (const rel of isolatedFiles) {
      const src = readFileSync(resolve(ROOT, rel), 'utf8');
      for (const marker of FORBIDDEN_DORY_MARKERS) {
        expect(src, `${rel} must not contain forbidden Dory marker "${marker}"`).not.toContain(
          marker,
        );
      }
      expect(src, `${rel} must not import src/dory/`).not.toMatch(/from\s+['"][^'"]*dory\//);
    }
  });

  it('mounts light-theme compact model switcher in src/dory/app/spatialApp.ts with zero sibling markers', () => {
    expect(DORY_APP_THEME).toBe('light');
    const src = readFileSync(resolve(ROOT, 'src/dory/app/spatialApp.ts'), 'utf8');
    expect(src).toContain('mountModelSwitcher(topActions');
    expect(src).toContain("activeId: 'dory'");
    expect(src).toContain('theme: DORY_APP_THEME');
    expect(src).not.toMatch(/from\s+['"][^'"]*(?:twotower|diffusion|uni|ui\/controls|ui\/overlay)/);
    for (const marker of FORBIDDEN_SIBLING_MARKERS) {
      expect(src, `dory spatialApp.ts must not contain sibling marker "${marker}"`).not.toContain(
        marker,
      );
    }
  });

  it('registers dory as a non-hybrid entry in scripts/verify_bundle_isolation.mjs', () => {
    const isoSrc = readFileSync(resolve(ROOT, 'scripts/verify_bundle_isolation.mjs'), 'utf8');
    expect(isoSrc).toContain("key: 'dory'");
    expect(isoSrc).toContain("htmlRelPath: 'dory/index.html'");
    expect(isoSrc).toContain('FORBIDDEN_DORY_MARKERS');
    const doryEntrySlice = isoSrc.slice(isoSrc.indexOf("key: 'dory'"), isoSrc.indexOf("key: 'dory'") + 400);
    expect(doryEntrySlice).toContain('usesNanoHybridMath: false');
  });

  it('renders all 5 WAI-ARIA links and keyboard navigation when mounted with activeId="dory"', () => {
    const docListeners = new Map<string, Listener[]>();
    const prevDoc = (globalThis as unknown as { document?: unknown }).document;
    const mockDoc = {
      activeElement: null as unknown,
      createElement: (tag: string) => new MockElement(tag),
      addEventListener: (type: string, fn: Listener) => {
        const arr = docListeners.get(type) ?? [];
        arr.push(fn);
        docListeners.set(type, arr);
      },
      removeEventListener: (type: string, fn: Listener) => {
        const arr = docListeners.get(type);
        if (!arr) return;
        const idx = arr.indexOf(fn);
        if (idx >= 0) arr.splice(idx, 1);
      },
    };
    (globalThis as unknown as { document: unknown }).document = mockDoc;

    try {
      const host = new MockElement('div');
      const handle = mountModelSwitcher(host as unknown as HTMLElement, {
        activeId: 'dory',
        baseUrl: '/llm-viz/',
        theme: 'light',
      });

      expect(handle.links).toHaveLength(5);
      expect(handle.links.map((a) => a.href)).toEqual([
        '/llm-viz/',
        '/llm-viz/twotower/',
        '/llm-viz/diffusion/',
        '/llm-viz/uni/',
        '/llm-viz/dory/',
      ]);
      expect(handle.links[4]!.getAttribute('aria-current')).toBe('page');
      expect(handle.links[0]!.getAttribute('aria-current')).toBeNull();

      // Open via ArrowDown -> focuses first non-current entry (Nano, links[0])
      (handle.trigger as unknown as MockElement).dispatch('keydown', {
        key: 'ArrowDown',
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(handle.isOpen()).toBe(true);
      expect(mockDoc.activeElement).toBe(handle.links[0]);

      // End -> focuses last entry (Dory, links[4])
      (handle.menu as unknown as MockElement).dispatch('keydown', {
        key: 'End',
        target: handle.links[0],
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(mockDoc.activeElement).toBe(handle.links[4]);

      // Escape -> closes and restores focus to trigger
      (handle.menu as unknown as MockElement).dispatch('keydown', {
        key: 'Escape',
        target: handle.links[4],
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(handle.isOpen()).toBe(false);
      expect(mockDoc.activeElement).toBe(handle.trigger);

      handle.destroy();
      expect(host.children).toHaveLength(0);
    } finally {
      (globalThis as unknown as { document?: unknown }).document = prevDoc;
    }
  });
});
