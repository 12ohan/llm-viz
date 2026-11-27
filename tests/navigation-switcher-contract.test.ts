import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MODEL_REGISTRY, resolveModelHref } from '../src/shared/modelRegistry';
import { mountModelSwitcher } from '../src/shared/modelSwitcher';

const ROOT = resolve(__dirname, '..');

const ALL_FORBIDDEN_APP_MARKERS = [
  '3D WebGL2 Canvas Unavailable',
  'Required stage DOM elements are missing',
  'Mamba-2 fixed floor',
  'mem-drawer-card',
  'memory-probe-stats',
  'NemotronHTwoTowerForCausalLM',
  'twotower-denoiser-v1',
  'twotower-context-v1',
  'TwoTowerSession',
  'diffusion_block_commit',
  'TWOTOWER_SPEC',
  'DIFFUSION_MINI_CONFIG',
  'DiffusionSession',
  'buildDiffusionSpatialScene',
  'DIFFUSION_DISCREPANCY_LEDGER',
  'linear_spec_lora',
  'DIFFUSION_DYADIC_YARN_INV_FREQ',
  'UNI_MINI_CONFIG',
  'UniSession',
  'buildUniSpatialScene',
  'UNI_DISCREPANCY_LEDGER',
  'PackedLinearImageCodec',
  'UNI_DYADIC_YARN_INV_FREQ',
  'buildUniOpdExplorerScene',
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
  focusedCount = 0;

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
    this.focusedCount += 1;
    (globalThis as unknown as { document: { activeElement: unknown } }).document.activeElement = this;
  }
}

describe('Shared Multi-Model Top-Bar Switcher Contract', () => {
  it('resolves canonical model hrefs cleanly for both root / and GitHub Pages /llm-viz/', () => {
    expect(MODEL_REGISTRY.map((e) => e.id)).toEqual(['nano', 'twotower', 'diffusion', 'uni']);

    expect(MODEL_REGISTRY.map((e) => resolveModelHref('/', e))).toEqual([
      '/',
      '/twotower/',
      '/diffusion/',
      '/uni/',
    ]);
    expect(MODEL_REGISTRY.map((e) => resolveModelHref('/llm-viz/', e))).toEqual([
      '/llm-viz/',
      '/llm-viz/twotower/',
      '/llm-viz/diffusion/',
      '/llm-viz/uni/',
    ]);
    expect(MODEL_REGISTRY.map((e) => resolveModelHref('/llm-viz', e))).toEqual([
      '/llm-viz/',
      '/llm-viz/twotower/',
      '/llm-viz/diffusion/',
      '/llm-viz/uni/',
    ]);
  });

  it('contains zero forbidden app markers and zero cross-app imports in shared switcher sources', () => {
    const files = [
      'src/shared/modelRegistry.ts',
      'src/shared/modelSwitcher.ts',
      'src/shared/modelSwitcher.css',
    ];
    for (const rel of files) {
      const src = readFileSync(resolve(ROOT, rel), 'utf8');
      for (const marker of ALL_FORBIDDEN_APP_MARKERS) {
        expect(src, `${rel} must not contain forbidden marker "${marker}"`).not.toContain(marker);
      }
      expect(src).not.toMatch(/from\s+['"][^'"]*(?:twotower|diffusion|uni|ui\/controls|ui\/overlay)/);
    }
  });

  it('supports WAI-ARIA open/close, ArrowDown/ArrowUp/Home/End/Escape focus management, outside click, and teardown', () => {
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
        activeId: 'diffusion',
        baseUrl: '/llm-viz/',
        theme: 'dark',
      });

      expect(handle.isOpen()).toBe(false);
      expect(handle.menu.hidden).toBe(true);
      expect(handle.trigger.getAttribute('aria-expanded')).toBe('false');
      expect(handle.links).toHaveLength(MODEL_REGISTRY.length);
      expect(handle.links.map((a) => a.href)).toEqual([
        '/llm-viz/',
        '/llm-viz/twotower/',
        '/llm-viz/diffusion/',
        '/llm-viz/uni/',
      ]);
      expect(handle.links[2]!.getAttribute('aria-current')).toBe('page');
      expect(handle.links[0]!.getAttribute('aria-current')).toBeNull();

      // ArrowDown on trigger opens menu and focuses first non-current model (Nano)
      let prevented = false;
      (handle.trigger as unknown as MockElement).dispatch('keydown', {
        key: 'ArrowDown',
        preventDefault: () => {
          prevented = true;
        },
        stopPropagation: () => {},
      });
      expect(prevented).toBe(true);
      expect(handle.isOpen()).toBe(true);
      expect(handle.menu.hidden).toBe(false);
      expect(handle.trigger.getAttribute('aria-expanded')).toBe('true');
      expect(mockDoc.activeElement).toBe(handle.links[0]);

      // ArrowDown / End / Home move cleanly across menu items
      (handle.menu as unknown as MockElement).dispatch('keydown', {
        key: 'ArrowDown',
        target: handle.links[0],
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(mockDoc.activeElement).toBe(handle.links[1]);

      (handle.menu as unknown as MockElement).dispatch('keydown', {
        key: 'End',
        target: handle.links[1],
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(mockDoc.activeElement).toBe(handle.links[3]);

      (handle.menu as unknown as MockElement).dispatch('keydown', {
        key: 'Home',
        target: handle.links[3],
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(mockDoc.activeElement).toBe(handle.links[0]);

      // Escape closes menu and restores focus to trigger
      (handle.menu as unknown as MockElement).dispatch('keydown', {
        key: 'Escape',
        target: handle.links[0],
        preventDefault: () => {},
        stopPropagation: () => {},
      });
      expect(handle.isOpen()).toBe(false);
      expect(handle.menu.hidden).toBe(true);
      expect(mockDoc.activeElement).toBe(handle.trigger);

      // Click opens, outside pointerdown closes
      (handle.trigger as unknown as MockElement).dispatch('click', {
        stopPropagation: () => {},
      });
      expect(handle.isOpen()).toBe(true);
      const outside = new MockElement('div');
      for (const fn of [...(docListeners.get('pointerdown') ?? [])]) {
        fn({ target: outside });
      }
      expect(handle.isOpen()).toBe(false);

      // Destroy removes DOM node and document listeners
      handle.destroy();
      expect(host.children).toHaveLength(0);
      expect(docListeners.get('pointerdown') ?? []).toHaveLength(0);
      expect(docListeners.get('keydown') ?? []).toHaveLength(0);
    } finally {
      (globalThis as unknown as { document?: unknown }).document = prevDoc;
    }
  });

  it('clamps popover horizontal coordinates within narrow and wide viewports without clipping', () => {
    const prevDoc = (globalThis as unknown as { document?: unknown }).document;
    const prevWin = (globalThis as unknown as { window?: unknown }).window;
    const winListeners = new Map<string, Listener[]>();
    const mockWin = {
      innerWidth: 390,
      addEventListener: (type: string, fn: Listener) => {
        const arr = winListeners.get(type) ?? [];
        arr.push(fn);
        winListeners.set(type, arr);
      },
      removeEventListener: (type: string, fn: Listener) => {
        const arr = winListeners.get(type);
        if (!arr) return;
        const idx = arr.indexOf(fn);
        if (idx >= 0) arr.splice(idx, 1);
      },
    };
    const mockDoc = {
      activeElement: null as unknown,
      createElement: (tag: string) => new MockElement(tag),
      addEventListener: () => {},
      removeEventListener: () => {},
    };
    (globalThis as unknown as { document: unknown }).document = mockDoc;
    (globalThis as unknown as { window: unknown }).window = mockWin;

    try {
      const host = new MockElement('div');
      const handle = mountModelSwitcher(host as unknown as HTMLElement, {
        activeId: 'nano',
        baseUrl: '/',
        theme: 'dark',
      });
      const triggerMock = handle.trigger as unknown as MockElement;
      const menuMock = handle.menu as unknown as MockElement;

      // Mobile 390px centered trigger [150, 235]: right-align would clip left (-45), left-align would clip right (430)
      triggerMock.rect = { left: 150, right: 235, top: 8, bottom: 34, width: 85, height: 26 };
      handle.open('none');
      const leftPxMobile = Number.parseFloat(menuMock.style.left);
      const absLeftMobile = triggerMock.rect.left + leftPxMobile;
      expect(absLeftMobile).toBeGreaterThanOrEqual(8);
      expect(absLeftMobile + 280).toBeLessThanOrEqual(390 - 8);

      // Desktop 1280px right-side trigger [1160, 1250]: aligns flush to right edge of trigger
      mockWin.innerWidth = 1280;
      triggerMock.rect = { left: 1160, right: 1250, top: 8, bottom: 34, width: 90, height: 26 };
      for (const fn of [...(winListeners.get('resize') ?? [])]) fn({});
      const leftPxDesktop = Number.parseFloat(menuMock.style.left);
      expect(triggerMock.rect.left + leftPxDesktop + 280).toBe(1250);

      handle.destroy();
      expect(winListeners.get('resize') ?? []).toHaveLength(0);
    } finally {
      (globalThis as unknown as { document?: unknown }).document = prevDoc;
      (globalThis as unknown as { window?: unknown }).window = prevWin;
    }
  });
});
