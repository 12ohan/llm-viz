import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  computeSpineWaveGeometryKey,
  computeValueAtlasContentKey,
  shouldAnimateCutawayPulse,
  shouldContinueRafLoop,
  shouldDeferGlobalShortcut,
  type ValueAtlasKeyInput,
} from '../src/main';
import {
  advanceSpineWavePlayback,
  startSpineWavePlayback,
} from '../src/model/rcPlayback';
import { NEMOTRON_SPEC } from '../src/model/spec';
import { createInitialState, createPerspectivePose } from '../src/model/state';
import { makeViewportMetrics } from '../src/render/camera';
import { RenderDevice } from '../src/render/device';
import { getWalkthroughBeat, getWalkthroughChapter } from '../src/story/walkthrough';
import {
  handleModalTabTrap,
  handleTablistKeydown,
  MacroControls,
  restoreFocusIfConnected,
  type ControlsCallbacks,
  type ControlsElements,
} from '../src/ui/controls';
import { MemoryScalingDrawer } from '../src/ui/memoryChart';

// ============================================================================
// Minimal Interactive DOM Shim (supports focus, closest, querySelectorAll, keydown)
// ============================================================================

class ShimClassList {
  private readonly set = new Set<string>();

  setFromString(value: string): void {
    this.set.clear();
    for (const tok of value.split(/\s+/)) {
      if (tok) this.set.add(tok);
    }
  }

  toString(): string {
    return [...this.set].join(' ');
  }

  add(...tokens: string[]): void {
    for (const t of tokens) if (t) this.set.add(t);
  }

  remove(...tokens: string[]): void {
    for (const t of tokens) this.set.delete(t);
  }

  toggle(token: string, force?: boolean): boolean {
    if (force === undefined) {
      if (this.set.has(token)) {
        this.set.delete(token);
        return false;
      }
      this.set.add(token);
      return true;
    }
    if (force) this.set.add(token);
    else this.set.delete(token);
    return force;
  }

  contains(token: string): boolean {
    return this.set.has(token);
  }
}

interface ShimEvent {
  type: string;
  key?: string;
  shiftKey?: boolean;
  target: unknown;
  defaultPrevented: boolean;
  propagationStopped: boolean;
  preventDefault: () => void;
  stopPropagation: () => void;
}

type ShimListener = (ev: ShimEvent) => void;

class ShimElement {
  readonly tagName: string;
  id = '';
  type = '';
  title = '';
  value = '';
  min = '';
  max = '';
  step = '';
  htmlFor = '';
  disabled = false;
  hidden = false;
  isConnected = true;
  tabIndex = -1;
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  readonly attributes = new Map<string, string>();
  readonly classList = new ShimClassList();
  readonly children: ShimElement[] = [];
  parentNode: ShimElement | null = null;
  private readonly listeners = new Map<string, ShimListener[]>();
  private _textContent = '';

  constructor(tagName: string) {
    this.tagName = tagName.toUpperCase();
  }

  get parentElement(): ShimElement | null {
    return this.parentNode;
  }

  get className(): string {
    return this.classList.toString();
  }

  set className(value: string) {
    this.classList.setFromString(value);
  }

  get textContent(): string {
    if (this.children.length === 0) return this._textContent;
    return this._textContent + this.children.map((c) => c.textContent).join('');
  }

  set textContent(value: string) {
    this._textContent = value;
    this.children.length = 0;
  }

  set innerHTML(value: string) {
    this._textContent = value;
    this.children.length = 0;
  }

  get innerHTML(): string {
    return this.textContent;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name === 'id') this.id = value;
    if (name === 'tabindex') this.tabIndex = Number.parseInt(value, 10);
    if (name.startsWith('data-')) {
      const camel = name
        .slice(5)
        .replace(/-([a-z])/g, (_, ch: string) => ch.toUpperCase());
      this.dataset[camel] = value;
    }
  }

  getAttribute(name: string): string | null {
    if (name === 'id' && this.id) return this.id;
    return this.attributes.get(name) ?? null;
  }

  appendChild<T extends ShimElement>(child: T): T {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  append(...nodes: ShimElement[]): void {
    for (const n of nodes) this.appendChild(n);
  }

  contains(other: unknown): boolean {
    if (other === this) return true;
    return this.children.some((c) => c.contains(other));
  }

  focus(): void {
    if (mockDocRef) {
      mockDocRef.activeElement = this;
    }
  }

  addEventListener(type: string, fn: ShimListener): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(fn);
    this.listeners.set(type, arr);
  }

  removeEventListener(type: string, fn: ShimListener): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    const idx = arr.indexOf(fn);
    if (idx >= 0) arr.splice(idx, 1);
  }

  dispatchEvent(ev: ShimEvent): void {
    let cur: ShimElement | null = this;
    while (cur) {
      for (const fn of cur.listeners.get(ev.type) ?? []) {
        fn(ev);
      }
      if (ev.propagationStopped) break;
      cur = cur.parentNode;
    }
    if (!ev.propagationStopped && mockDocRef) {
      for (const fn of mockDocRef.listeners.get(ev.type) ?? []) {
        fn(ev);
      }
    }
  }

  dispatchKeydown(key: string, shiftKey = false): ShimEvent {
    const ev: ShimEvent = {
      type: 'keydown',
      key,
      shiftKey,
      target: this,
      defaultPrevented: false,
      propagationStopped: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      stopPropagation() {
        this.propagationStopped = true;
      },
    };
    this.dispatchEvent(ev);
    return ev;
  }

  matchesSingleSelector(sel: string): boolean {
    const s = sel.trim();
    if (s === 'button' || s === 'button:not([disabled])') {
      return this.tagName === 'BUTTON' && (!s.includes(':not([disabled])') || !this.disabled);
    }
    if (s === 'input:not([disabled])') {
      return this.tagName === 'INPUT' && !this.disabled;
    }
    if (s === 'select:not([disabled])') {
      return this.tagName === 'SELECT' && !this.disabled;
    }
    if (s === 'textarea:not([disabled])') {
      return this.tagName === 'TEXTAREA' && !this.disabled;
    }
    if (s === 'a[href]') {
      return this.tagName === 'A' && this.getAttribute('href') !== null;
    }
    if (s === 'summary') {
      return this.tagName === 'SUMMARY';
    }
    if (s === '[role="button"]') {
      return this.getAttribute('role') === 'button';
    }
    if (s === '[role="tab"]') {
      return this.getAttribute('role') === 'tab';
    }
    if (s === '[tabindex]:not([tabindex="-1"]):not(#gl-canvas)') {
      const ti = this.getAttribute('tabindex');
      return ti !== null && ti !== '-1' && this.id !== 'gl-canvas';
    }
    if (s === '[tabindex]:not([tabindex="-1"])') {
      const ti = this.getAttribute('tabindex');
      return ti !== null && ti !== '-1';
    }
    if (s.startsWith('.')) {
      return this.classList.contains(s.slice(1));
    }
    if (s.startsWith('#')) {
      return this.id === s.slice(1);
    }
    return false;
  }

  matches(selectorList: string): boolean {
    return selectorList.split(',').some((sel) => this.matchesSingleSelector(sel));
  }

  closest(selectorList: string): ShimElement | null {
    let cur: ShimElement | null = this;
    while (cur) {
      if (cur.matches(selectorList)) return cur;
      cur = cur.parentNode;
    }
    return null;
  }

  findAll(predicate: (el: ShimElement) => boolean): ShimElement[] {
    const out: ShimElement[] = [];
    const visit = (node: ShimElement) => {
      if (predicate(node)) out.push(node);
      for (const child of node.children) visit(child);
    };
    visit(this);
    return out;
  }

  querySelector(selector: string): ShimElement | null {
    return this.findAll((el) => el !== this && el.matches(selector))[0] ?? null;
  }

  querySelectorAll(selector: string): ShimElement[] {
    return this.findAll((el) => el !== this && el.matches(selector));
  }
}

interface MockDocument {
  activeElement: ShimElement | null;
  listeners: Map<string, ShimListener[]>;
  createElement: (tag: string) => ShimElement;
  createElementNS: (_ns: string, tag: string) => ShimElement;
  addEventListener: (type: string, fn: ShimListener) => void;
  removeEventListener: (type: string, fn: ShimListener) => void;
}

const origDocument = globalThis.document;
let mockDocRef: MockDocument | null = null;

function installShimDocument(): MockDocument {
  const listeners = new Map<string, ShimListener[]>();
  const doc: MockDocument = {
    activeElement: null,
    listeners,
    createElement: (tag: string) => new ShimElement(tag),
    createElementNS: (_ns: string, tag: string) => new ShimElement(tag),
    addEventListener: (type: string, fn: ShimListener) => {
      const arr = listeners.get(type) ?? [];
      arr.push(fn);
      listeners.set(type, arr);
    },
    removeEventListener: (type: string, fn: ShimListener) => {
      const arr = listeners.get(type);
      if (!arr) return;
      const idx = arr.indexOf(fn);
      if (idx >= 0) arr.splice(idx, 1);
    },
  };
  mockDocRef = doc;
  (globalThis as unknown as { document: unknown }).document = doc;
  return doc;
}

function restoreDocument(): void {
  mockDocRef = null;
  if (origDocument === undefined) {
    delete (globalThis as { document?: unknown }).document;
  } else {
    (globalThis as unknown as { document: unknown }).document = origDocument;
  }
}

function makeShimButton(attrs?: {
  id?: string;
  className?: string;
  role?: string;
  tabindex?: string;
  dataset?: Record<string, string>;
}): ShimElement {
  const btn = new ShimElement('button');
  btn.type = 'button';
  if (attrs?.id) btn.id = attrs.id;
  if (attrs?.className) btn.className = attrs.className;
  if (attrs?.role) btn.setAttribute('role', attrs.role);
  if (attrs?.tabindex !== undefined) btn.setAttribute('tabindex', attrs.tabindex);
  if (attrs?.dataset) {
    for (const [k, v] of Object.entries(attrs.dataset)) {
      btn.dataset[k] = v;
    }
  }
  return btn;
}

function createControlsHarness() {
  const modeContainer = new ShimElement('div');
  modeContainer.setAttribute('role', 'tablist');
  const modeValues = [
    'tower',
    'cutaway_embedding',
    'cutaway_mamba',
    'cutaway_moe',
    'cutaway_gqa',
    'cutaway_output',
  ] as const;
  const modeButtons = modeValues.map((m, i) => {
    const b = makeShimButton({
      className: 'mode-btn',
      role: 'tab',
      tabindex: i === 0 ? '0' : '-1',
      dataset: { mode: m, view: m },
    });
    modeContainer.appendChild(b);
    return b;
  });

  const stepPillsContainer = new ShimElement('div');
  stepPillsContainer.id = 'gqa-step-pills';
  stepPillsContainer.setAttribute('role', 'tablist');
  const gqaStepPills = Array.from({ length: 6 }, (_, i) => {
    const b = makeShimButton({
      className: 'gqa-step-pill',
      role: 'tab',
      tabindex: i === 0 ? '0' : '-1',
      dataset: { gqaStep: String(i) },
    });
    stepPillsContainer.appendChild(b);
    return b;
  });

  const shortcutsPopover = new ShimElement('div');
  shortcutsPopover.id = 'shortcuts-popover';
  shortcutsPopover.setAttribute('role', 'dialog');
  shortcutsPopover.setAttribute('aria-modal', 'true');
  shortcutsPopover.hidden = true;
  const shortcutsCloseBtn = makeShimButton({ id: 'shortcuts-close' });
  shortcutsPopover.appendChild(shortcutsCloseBtn);

  const editorialPanel = new ShimElement('div');
  const gqaStepScrub = new ShimElement('input');
  gqaStepScrub.value = '0';
  const shortcutsBtn = makeShimButton({ id: 'shortcuts-btn' });

  const els = {
    sidebarTop: new ShimElement('section'),
    sidebarTopBody: new ShimElement('div'),
    sidebarTocToggle: makeShimButton({ id: 'sidebar-toc-toggle' }),
    sidebarCurrentSection: new ShimElement('span'),
    modeButtons,
    tocButtons: [],
    presetButtons: [],
    mambaBayButtons: [],
    mambaViewButtons: [],
    moeBayButtons: [],
    moeViewButtons: [],
    gqaBayButtons: [],
    gqaViewButtons: [],
    miniNodes: [],
    tourToggleBtn: makeShimButton({ id: 'tour-toggle-btn' }),
    floatingToolbar: new ShimElement('div'),
    towerSubbar: new ShimElement('div'),
    mambaSubbar: new ShimElement('div'),
    moeSubbar: new ShimElement('div'),
    gqaSubbar: new ShimElement('div'),
    floatingBadge: new ShimElement('div'),
    orbitHint: new ShimElement('div'),
    gqaMathDock: new ShimElement('section'),
    gqaFormulaTensor: new ShimElement('span'),
    gqaFormulaCoord: new ShimElement('span'),
    gqaFormulaHead: new ShimElement('span'),
    gqaFormulaValue: new ShimElement('span'),
    gqaFormulaNoteBtn: makeShimButton({ id: 'gqa-formula-note-btn' }),
    gqaFormulaNote: new ShimElement('div'),
    gqaCellClear: makeShimButton({ id: 'gqa-cell-clear' }),
    gqaFormulaEq: new ShimElement('div'),
    gqaFormulaSub: new ShimElement('div'),
    gqaStepPrev: makeShimButton(),
    gqaStepPlay: makeShimButton(),
    gqaStepNext: makeShimButton(),
    gqaStepPills,
    gqaStepPillsList: stepPillsContainer,
    gqaStepAutoFollow: makeShimButton(),
    gqaStepScrub,
    editorialPanel,
    execModeFull: makeShimButton(),
    execModeDecode: makeShimButton(),
    tokenSlider: new ShimElement('input'),
    tokenDec: makeShimButton(),
    tokenInc: makeShimButton(),
    tokenValue: new ShimElement('output'),
    precisionBf16: makeShimButton(),
    precisionFp8: makeShimButton(),
    memoryReadout: new ShimElement('div'),
    memoryReadoutLines: new ShimElement('div'),
    memoryChartBtn: makeShimButton(),
    bayStepRow: new ShimElement('div'),
    bayPrev: makeShimButton(),
    bayNext: makeShimButton(),
    bayClear: makeShimButton(),
    zoomBayBtn: makeShimButton(),
    fitBtn: makeShimButton(),
    resetBtn: makeShimButton(),
    shortcutsBtn,
    shortcutsPopover,
    shortcutsCloseBtn,
    spineWaveHud: new ShimElement('div'),
  } as unknown as ControlsElements;

  const cb: ControlsCallbacks = {
    onSelectMode: vi.fn(),
    onSelectSection: vi.fn(),
    onSelectMambaBay: vi.fn(),
    onSelectMambaView: vi.fn(),
    onSelectMambaStateToken: vi.fn(),
    onStepMambaStateToken: vi.fn(),
    onToggleMambaTemporalPlay: vi.fn(),
    onSelectMoeBay: vi.fn(),
    onSelectMoeView: vi.fn(),
    onSelectMoeFocusToken: vi.fn(),
    onStepMoeFocusToken: vi.fn(),
    onToggleMoeTokenCycle: vi.fn(),
    onSelectMoeExpert: vi.fn(),
    onSelectGqaBay: vi.fn(),
    onSelectGqaView: vi.fn(),
    onSelectPart: vi.fn(),
    onInspectInternals: vi.fn(),
    onTokensChange: vi.fn(),
    onPrecisionChange: vi.fn(),
    onFitAll: vi.fn(),
    onZoomSelectedBay: vi.fn(),
    onReset: vi.fn(),
    onStepBay: vi.fn(),
    onClearSelection: vi.fn(),
    onTourToggle: vi.fn(),
    onTourPrev: vi.fn(),
    onTourNext: vi.fn(),
    onTourRestart: vi.fn(),
    onTourExit: vi.fn(),
    onToggleMemoryChart: vi.fn(),
    onToggleShortcuts: vi.fn(),
    onCloseShortcuts: vi.fn(),
    onGqaStepPrev: vi.fn(),
    onGqaStepNext: vi.fn(),
    onGqaStepTogglePlay: vi.fn(),
    onGqaStepSelect: vi.fn(),
    onGqaStepScrub: vi.fn(),
    onGqaClearCell: vi.fn(),
    onToggleAutoFollow: vi.fn(),
    onSelectExecutionMode: vi.fn(),
    onDecodeStepToken: vi.fn(),
    onDecodeStepGreedy: vi.fn(),
    onDecodeNavStep: vi.fn(),
    onDecodeSwitchBranch: vi.fn(),
    onDecodeReset: vi.fn(),
  };

  return {
    els,
    cb,
    raw: {
      modeContainer,
      modeButtons,
      stepPillsContainer,
      gqaStepPills,
      gqaStepScrub,
      editorialPanel,
      shortcutsBtn,
      shortcutsPopover,
      shortcutsCloseBtn,
    },
  };
}

// ============================================================================
// 1. RB-3: Preserve Native Keyboard Activation
// ============================================================================

describe('RB-3: shouldDeferGlobalShortcut preserves native activation on interactive controls', () => {
  it('defers Space and Enter on <button>, <a href>, [role="button"], [role="tab"], <summary>, and custom [tabindex="0"]', () => {
    const btn = makeShimButton();
    const innerSpan = new ShimElement('span');
    btn.appendChild(innerSpan);

    const link = new ShimElement('a');
    link.setAttribute('href', 'https://arxiv.org/abs/2512.20848');

    const roleBtn = new ShimElement('div');
    roleBtn.setAttribute('role', 'button');

    const roleTab = makeShimButton({ role: 'tab', tabindex: '0' });

    const summary = new ShimElement('summary');

    const focusableDiv = new ShimElement('div');
    focusableDiv.setAttribute('tabindex', '0');

    for (const el of [btn, innerSpan, link, roleBtn, roleTab, summary, focusableDiv]) {
      expect(shouldDeferGlobalShortcut(' ', el as unknown as EventTarget)).toBe(true);
      expect(shouldDeferGlobalShortcut('Enter', el as unknown as EventTarget)).toBe(true);
    }
  });

  it('keeps Space and Enter active on #gl-canvas, <body>, plain <div>, and [tabindex="-1"]', () => {
    const canvas = new ShimElement('canvas');
    canvas.id = 'gl-canvas';
    canvas.setAttribute('tabindex', '0');

    const body = new ShimElement('body');
    const div = new ShimElement('div');
    const negTab = new ShimElement('div');
    negTab.setAttribute('tabindex', '-1');
    const anchorNoHref = new ShimElement('a');

    for (const el of [canvas, body, div, negTab, anchorNoHref, null]) {
      expect(shouldDeferGlobalShortcut(' ', el as unknown as EventTarget | null)).toBe(false);
      expect(shouldDeferGlobalShortcut('Enter', el as unknown as EventTarget | null)).toBe(false);
    }
  });

  it('keeps Escape, ?, and single-key shortcuts global even when focused inside a button or tab', () => {
    const btn = makeShimButton({ role: 'tab', tabindex: '0' });
    for (const key of ['Escape', '?', '1', '2', '3', '4', '5', '6', 'p', 'm', 'r', 'ArrowRight']) {
      expect(shouldDeferGlobalShortcut(key, btn as unknown as EventTarget)).toBe(false);
    }
  });

  it('supports fallback parent walk for objects without closest()', () => {
    const parentBtn = {
      tagName: 'BUTTON',
      id: '',
      parentNode: null,
      getAttribute: () => null,
    };
    const child = {
      tagName: 'SPAN',
      id: '',
      parentNode: parentBtn,
      getAttribute: () => null,
    };
    expect(shouldDeferGlobalShortcut(' ', child as unknown as EventTarget)).toBe(true);
    expect(shouldDeferGlobalShortcut('Enter', child as unknown as EventTarget)).toBe(true);
    expect(shouldDeferGlobalShortcut('Escape', child as unknown as EventTarget)).toBe(false);
  });
});

// ============================================================================
// 2. RB-5 & PR-20: Synchronized Shortcuts Popover & Pre-Hydration HTML
// ============================================================================

describe('RB-5 & PR-20: index.html pre-hydration copy, shortcuts dialog, and tab roles', () => {
  const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8');

  it('updates Tour button pre-hydration markup to Tour 1/22 / 22-beat guided walkthrough', () => {
    expect(html).toContain('Tour 1/22');
    expect(html).toContain('Toggle 22-beat guided walkthrough');
    expect(html).not.toContain('Tour 1/7');
    expect(html).not.toContain('7-chapter');
  });

  it('synchronizes #shortcuts-popover with main.ts shortcuts and replaces chapter with beat', () => {
    const popStart = html.indexOf('id="shortcuts-popover"');
    const popEnd = html.indexOf('</dl>', popStart);
    expect(popStart).toBeGreaterThan(-1);
    const popoverHtml = html.slice(popStart, popEnd);

    expect(popoverHtml).toContain('aria-modal="true"');
    expect(popoverHtml).toContain('<kbd>1</kbd>–<kbd>4</kbd>');
    expect(popoverHtml).toContain('<kbd>5</kbd>');
    expect(popoverHtml).toContain('<kbd>6</kbd>');
    expect(popoverHtml).toContain('<kbd>Space</kbd>');
    expect(popoverHtml).toContain('<kbd>Shift</kbd>+<kbd>Space</kbd>');
    expect(popoverHtml).toContain('<kbd>←</kbd> <kbd>→</kbd>');
    expect(popoverHtml).toContain('<kbd>↑</kbd> <kbd>↓</kbd>');
    expect(popoverHtml).toContain('<kbd>Enter</kbd>');
    expect(popoverHtml).toContain('<kbd>P</kbd>');
    expect(popoverHtml).toContain('<kbd>M</kbd>');
    expect(popoverHtml).toContain('<kbd>R</kbd>');
    expect(popoverHtml).toContain('<kbd>?</kbd>');
    expect(popoverHtml).toContain('<kbd>Esc</kbd>');
    expect(popoverHtml.toLowerCase()).not.toContain('chapter');
    expect(popoverHtml.toLowerCase()).toContain('beat');
  });

  it('marks #gqa-step-pills with role="tab" and roving tabindex="0"/"-1"', () => {
    const start = html.indexOf('id="gqa-step-pills"');
    const end = html.indexOf('</div>', start);
    const pillsHtml = html.slice(start, end);
    const tabMatches = pillsHtml.match(/role="tab"/g) ?? [];
    expect(tabMatches).toHaveLength(6);
    expect(pillsHtml.match(/tabindex="0"/g) ?? []).toHaveLength(1);
    expect(pillsHtml.match(/tabindex="-1"/g) ?? []).toHaveLength(5);
  });
});

// ============================================================================
// 3. PR-17 & PR-3 & PR-13: WAI-ARIA Tabs, Modal Focus Trap, Concise Tour Sub, Progress-Only Sync
// ============================================================================

describe('PR-17, PR-3, PR-13: WAI-ARIA Tabs, modal focus trap/restore, concise tour subtitle, progress sync', () => {
  afterEach(() => {
    restoreDocument();
  });

  it('maintains roving tabindex and ArrowLeft/ArrowRight/Home/End keyboard switching on #gqa-step-pills', () => {
    const doc = installShimDocument();
    const { els, cb, raw } = createControlsHarness();
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);
    const state = createInitialState(createPerspectivePose([0, 0, 0], 18), false);

    // GQA cutaway with stepper active at step 2
    state.mode = 'cutaway_gqa';
    state.gqaStepperActive = true;
    state.gqaStepperStep = 2;
    controls.sync(state, null, false, null, null);

    expect(raw.gqaStepPills.map((b) => b.getAttribute('tabindex'))).toEqual([
      '-1',
      '-1',
      '0',
      '-1',
      '-1',
      '-1',
    ]);
    expect(raw.gqaStepPills[2].getAttribute('aria-selected')).toBe('true');

    // ArrowRight moves to step 3, focuses pill 3, stops propagation so global arrow handler won't fire
    const rightEv = raw.gqaStepPills[2].dispatchKeydown('ArrowRight');
    expect(rightEv.defaultPrevented).toBe(true);
    expect(rightEv.propagationStopped).toBe(true);
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(3);
    expect(doc.activeElement).toBe(raw.gqaStepPills[3]);
    expect(raw.gqaStepPills[3].getAttribute('tabindex')).toBe('0');
    expect(raw.gqaStepPills[2].getAttribute('tabindex')).toBe('-1');

    // End jumps to last visible tab (5)
    const endEv = raw.gqaStepPills[3].dispatchKeydown('End');
    expect(endEv.defaultPrevented).toBe(true);
    expect(endEv.propagationStopped).toBe(true);
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(5);
    expect(doc.activeElement).toBe(raw.gqaStepPills[5]);

    // ArrowRight wraps cleanly from 5 -> 0
    raw.gqaStepPills[5].dispatchKeydown('ArrowRight');
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(0);
    expect(doc.activeElement).toBe(raw.gqaStepPills[0]);

    // ArrowLeft wraps cleanly from 0 -> 5
    raw.gqaStepPills[0].dispatchKeydown('ArrowLeft');
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(5);
    expect(doc.activeElement).toBe(raw.gqaStepPills[5]);

    // Home jumps back to 0
    raw.gqaStepPills[5].dispatchKeydown('Home');
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(0);
    expect(doc.activeElement).toBe(raw.gqaStepPills[0]);

    // Embedding cutaway exposes 4 tabs (0..3) and hides 4..5 — End & wrap stay inside visible tabs!
    state.mode = 'cutaway_embedding';
    controls.sync(state, null, false, null, null);
    expect(raw.gqaStepPills[4].hidden).toBe(true);
    expect(raw.gqaStepPills[5].hidden).toBe(true);
    raw.gqaStepPills[0].dispatchKeydown('End');
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(3);
    expect(doc.activeElement).toBe(raw.gqaStepPills[3]);
    raw.gqaStepPills[3].dispatchKeydown('ArrowRight');
    expect(cb.onGqaStepSelect).toHaveBeenLastCalledWith(0);

    controls.dispose();
  });

  it('traps Tab and Shift+Tab cleanly inside modals via handleModalTabTrap and restores focus on close', () => {
    const doc = installShimDocument();
    const triggerBtn = makeShimButton({ id: 'trigger' });
    triggerBtn.focus();
    expect(doc.activeElement).toBe(triggerBtn);

    const dialog = new ShimElement('div');
    const firstBtn = makeShimButton({ id: 'first' });
    const midInput = new ShimElement('input');
    midInput.id = 'mid';
    const lastLink = new ShimElement('a');
    lastLink.setAttribute('href', 'https://example.com');
    dialog.append(firstBtn, midInput, lastLink);

    // Tab on last element wraps to first
    lastLink.focus();
    const tabWrapEv = lastLink.dispatchKeydown('Tab', false);
    expect(
      handleModalTabTrap(dialog as unknown as HTMLElement, tabWrapEv as unknown as KeyboardEvent),
    ).toBe(true);
    expect(tabWrapEv.defaultPrevented).toBe(true);
    expect(doc.activeElement).toBe(firstBtn);

    // Shift+Tab on first element wraps to last
    const shiftWrapEv = firstBtn.dispatchKeydown('Tab', true);
    expect(
      handleModalTabTrap(dialog as unknown as HTMLElement, shiftWrapEv as unknown as KeyboardEvent),
    ).toBe(true);
    expect(shiftWrapEv.defaultPrevented).toBe(true);
    expect(doc.activeElement).toBe(lastLink);

    // Tab on middle element lets browser advance naturally (no preventDefault)
    midInput.focus();
    const midEv = midInput.dispatchKeydown('Tab', false);
    expect(
      handleModalTabTrap(dialog as unknown as HTMLElement, midEv as unknown as KeyboardEvent),
    ).toBe(false);
    expect(midEv.defaultPrevented).toBe(false);

    // Focus restoration restores connected element, skips disconnected element
    expect(restoreFocusIfConnected(triggerBtn as unknown as HTMLElement)).toBe(true);
    expect(doc.activeElement).toBe(triggerBtn);

    triggerBtn.isConnected = false;
    firstBtn.focus();
    expect(restoreFocusIfConnected(triggerBtn as unknown as HTMLElement)).toBe(false);
    expect(doc.activeElement).toBe(firstBtn);
  });

  it('moves focus into MemoryScalingDrawer and Shortcuts Popover on open, traps Tab, and restores focus on close', () => {
    const doc = installShimDocument();
    const host = new ShimElement('div');
    host.hidden = true;
    const trigger = makeShimButton({ id: 'mem-trigger' });
    trigger.focus();

    const onClose = vi.fn();
    const drawer = new MemoryScalingDrawer(NEMOTRON_SPEC, host as unknown as HTMLElement, {
      onClose,
      onPrecisionChange: vi.fn(),
      onProbeTokensChange: vi.fn(),
    });

    // Open -> focuses close button inside modal
    drawer.sync(true, 'bf16', 8192);
    expect(doc.activeElement).not.toBe(trigger);
    expect(doc.activeElement?.getAttribute('aria-label')).toBe('Close Memory Scaling chart');

    // Tab trap on last focusable element inside drawer wraps to first
    const card = host.findAll((el) => el.getAttribute('role') === 'dialog')[0]!;
    const focusables = card.querySelectorAll(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    expect(focusables.length).toBeGreaterThan(2);
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    last.focus();
    const ev = last.dispatchKeydown('Tab', false);
    expect(ev.defaultPrevented).toBe(true);
    expect(doc.activeElement).toBe(first);

    // Close -> restores focus back to trigger
    drawer.sync(false, 'bf16', 8192);
    expect(doc.activeElement).toBe(trigger);
    drawer.dispose();

    // Shortcuts popover via MacroControls
    const { els, cb, raw } = createControlsHarness();
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);
    const state = createInitialState(createPerspectivePose([0, 0, 0], 18), false);

    raw.shortcutsBtn.focus();
    state.shortcutsOpen = true;
    controls.sync(state, null, false, null, null);
    expect(doc.activeElement).toBe(raw.shortcutsCloseBtn);

    // Single-button shortcuts modal traps Tab on close button
    const tabEv = raw.shortcutsCloseBtn.dispatchKeydown('Tab', false);
    expect(tabEv.defaultPrevented).toBe(true);
    expect(doc.activeElement).toBe(raw.shortcutsCloseBtn);

    state.shortcutsOpen = false;
    controls.sync(state, null, false, null, null);
    expect(doc.activeElement).toBe(raw.shortcutsBtn);
    controls.dispose();
  });

  it('renders concise chapter.shortTitle in Tour .ed-sub (PR-3) and updates scrubber via syncStepperProgressOnly (PR-13)', () => {
    installShimDocument();
    const { els, cb, raw } = createControlsHarness();
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);
    const state = createInitialState(createPerspectivePose([0, 0, 0], 18), false);

    // Beat 5 (cutaway with stepper bar)
    state.tourActive = true;
    state.tourBeat = 5;
    const beat = getWalkthroughBeat(5);
    const chapter = getWalkthroughChapter(beat.chapterIndex);
    state.mode = beat.mode;
    controls.sync(state, null, false, null, null);

    const sub = raw.editorialPanel.findAll((el) => el.dataset.testid === 'tour-chapter-subtitle')[0]!;
    expect(sub.textContent).toBe(chapter.shortTitle);
    expect(sub.textContent).not.toContain(beat.actLabel);

    const childSnapshotBefore = [...raw.editorialPanel.children];
    controls.syncStepperProgressOnly(0.42);
    expect(raw.gqaStepScrub.value).toBe('42');
    // Editorial DOM nodes were NOT rebuilt!
    expect(raw.editorialPanel.children).toEqual(childSnapshotBefore);
    const beatFill = raw.editorialPanel.querySelector('.tour-beat-fill');
    if (beatFill) {
      expect(beatFill.style.width).toBe('42%');
    }

    // Also verify handleTablistKeydown ignores unrelated keys cleanly
    const noopEv = raw.gqaStepPills[0].dispatchKeydown('Enter');
    expect(
      handleTablistKeydown(
        raw.gqaStepPills as unknown as HTMLButtonElement[],
        noopEv as unknown as KeyboardEvent,
        vi.fn(),
      ),
    ).toBe(false);

    controls.dispose();
  });
});

// ============================================================================
// 4. PR-10: RenderDevice FBO Allocation Failure Cleanup & Non-Retry Latch
// ============================================================================

describe('PR-10: RenderDevice FBO failure cleanup & non-retry latch', () => {
  function createMockGl(opts: {
    failDepthRb?: boolean;
    fboStatus?: number;
  }) {
    let fboSeq = 0;
    let rbSeq = 0;
    const createdFbos: object[] = [];
    const deletedFbos: object[] = [];
    const createdRbs: object[] = [];
    const deletedRbs: object[] = [];

    const gl = {
      RENDERBUFFER: 0x8d41,
      FRAMEBUFFER: 0x8d40,
      RGBA8: 0x8058,
      DEPTH_COMPONENT24: 0x81a6,
      SAMPLES: 0x80a9,
      COLOR_ATTACHMENT0: 0x8ce0,
      DEPTH_ATTACHMENT: 0x8d00,
      FRAMEBUFFER_COMPLETE: 0x8cd5,
      FRAMEBUFFER_INCOMPLETE_ATTACHMENT: 0x8cd6,
      getInternalformatParameter: vi.fn(() => new Int32Array([0])),
      createFramebuffer: vi.fn(() => {
        const handle = { id: ++fboSeq, kind: 'fbo' };
        createdFbos.push(handle);
        return handle;
      }),
      createRenderbuffer: vi.fn(() => {
        rbSeq++;
        if (opts.failDepthRb && rbSeq % 2 === 0) {
          return null;
        }
        const handle = { id: rbSeq, kind: 'rb' };
        createdRbs.push(handle);
        return handle;
      }),
      deleteFramebuffer: vi.fn((h: object) => {
        deletedFbos.push(h);
      }),
      deleteRenderbuffer: vi.fn((h: object) => {
        deletedRbs.push(h);
      }),
      bindFramebuffer: vi.fn(),
      bindRenderbuffer: vi.fn(),
      renderbufferStorage: vi.fn(),
      renderbufferStorageMultisample: vi.fn(),
      framebufferRenderbuffer: vi.fn(),
      checkFramebufferStatus: vi.fn(
        () => opts.fboStatus ?? 0x8cd5, // FRAMEBUFFER_COMPLETE
      ),
    };

    const canvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => gl),
    } as unknown as HTMLCanvasElement;

    return { gl, canvas, createdFbos, deletedFbos, createdRbs, deletedRbs };
  }

  it('deletes all locally allocated handles when createRenderbuffer returns null and latches against same-size retries', () => {
    const { gl, canvas, createdFbos, deletedFbos, createdRbs, deletedRbs } = createMockGl({
      failDepthRb: true,
    });
    const device = new RenderDevice(canvas);
    const metrics = makeViewportMetrics(0, 0, 800, 600, 1);

    device.ensureSize(metrics);
    expect(device.isFboAllocationFailed).toBe(true);
    // All partially created local handles were deleted (zero GPU handle leaks)
    expect(createdFbos).toHaveLength(1);
    expect(deletedFbos).toEqual(createdFbos);
    expect(createdRbs).toHaveLength(1);
    expect(deletedRbs).toEqual(createdRbs);

    // Subsequent frames at identical dimensions MUST NOT retry FBO allocation
    const callsBefore = gl.createFramebuffer.mock.calls.length;
    device.ensureSize(metrics);
    device.ensureSize(metrics);
    expect(gl.createFramebuffer.mock.calls.length).toBe(callsBefore);
  });

  it('cleans up handles on incomplete FBO status, suppresses same-size retries, and resets latch on resize or context restore', () => {
    const { gl, canvas, createdFbos, deletedFbos, createdRbs, deletedRbs } = createMockGl({
      fboStatus: 0x8cd6, // FRAMEBUFFER_INCOMPLETE_ATTACHMENT
    });
    const device = new RenderDevice(canvas);
    const m1 = makeViewportMetrics(0, 0, 800, 600, 1);

    device.ensureSize(m1);
    expect(device.isFboAllocationFailed).toBe(true);
    expect(deletedFbos).toEqual(createdFbos);
    expect(deletedRbs).toEqual(createdRbs);

    // No retry at identical dimensions
    expect(gl.createFramebuffer).toHaveBeenCalledTimes(1);
    device.ensureSize(m1);
    expect(gl.createFramebuffer).toHaveBeenCalledTimes(1);

    // Dimension change clears latch and retries once
    gl.checkFramebufferStatus.mockReturnValue(gl.FRAMEBUFFER_COMPLETE);
    const m2 = makeViewportMetrics(0, 0, 1024, 768, 1);
    device.ensureSize(m2);
    expect(gl.createFramebuffer).toHaveBeenCalledTimes(2);
    expect(device.isFboAllocationFailed).toBe(false);

    // Context restore (markLost(false)) also clears latch cleanly
    gl.checkFramebufferStatus.mockReturnValue(gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    const m3 = makeViewportMetrics(0, 0, 640, 480, 1);
    device.ensureSize(m3);
    expect(device.isFboAllocationFailed).toBe(true);
    device.markLost(true);
    device.markLost(false);
    expect(device.isFboAllocationFailed).toBe(false);
    gl.checkFramebufferStatus.mockReturnValue(gl.FRAMEBUFFER_COMPLETE);
    device.ensureSize(m3);
    expect(device.isFboAllocationFailed).toBe(false);
  });
});

// ============================================================================
// 5. PR-11 / PR-12 / PR-14: Idle RAF Gate, Atlas Key Invariance & Spine Wave Throttling
// ============================================================================

describe('PR-11 / PR-12 / PR-14: Idle RAF gate, atlasContentKey invariance, and spine-wave rebuild throttling', () => {
  it('PR-11: idle RAF gate is false for paused/pinned/hovered steady state and true only during active motion/playback', () => {
    const baseState = createInitialState(createPerspectivePose([0, 0, 0], 18), false);

    // Tower idle -> false
    expect(
      shouldContinueRafLoop({
        state: baseState,
        flyAnimActive: false,
        spineWaveActive: false,
        spineWavePaused: false,
      }),
    ).toBe(false);

    // Static pinned / hovered cell in GQA with paused stepper -> false (no infinite 60fps spin!)
    const pausedGqa = {
      ...baseState,
      mode: 'cutaway_gqa' as const,
      gqaStepperActive: true,
      gqaStepperPlaying: false,
      selectedPart: 'q-head-0',
      hoveredPart: 'scores',
    };
    expect(shouldAnimateCutawayPulse(pausedGqa)).toBe(false);
    expect(
      shouldContinueRafLoop({
        state: pausedGqa,
        flyAnimActive: false,
        spineWaveActive: false,
        spineWavePaused: false,
      }),
    ).toBe(false);

    // Paused Mamba / MoE steppers -> false
    expect(
      shouldAnimateCutawayPulse({
        ...baseState,
        mode: 'cutaway_mamba',
        mambaStepperActive: true,
        mambaStepperPlaying: false,
        mambaTemporalPlaying: false,
      }),
    ).toBe(false);
    expect(
      shouldAnimateCutawayPulse({
        ...baseState,
        mode: 'cutaway_moe',
        moeStepperActive: true,
        moeStepperPlaying: false,
        moeTokenCycling: false,
      }),
    ).toBe(false);

    // Active GQA stepper Play -> true (unless reducedMotion or contextLost)
    const playingGqa = { ...pausedGqa, gqaStepperPlaying: true };
    expect(shouldAnimateCutawayPulse(playingGqa)).toBe(true);
    expect(
      shouldContinueRafLoop({
        state: playingGqa,
        flyAnimActive: false,
        spineWaveActive: false,
        spineWavePaused: false,
      }),
    ).toBe(true);
    expect(shouldAnimateCutawayPulse({ ...playingGqa, reducedMotion: true })).toBe(false);
    expect(
      shouldContinueRafLoop({
        state: playingGqa,
        flyAnimActive: true,
        spineWaveActive: false,
        spineWavePaused: false,
        contextLost: true,
      }),
    ).toBe(false);

    // Active Mamba temporal play & MoE token cycling -> true
    expect(
      shouldAnimateCutawayPulse({
        ...baseState,
        mode: 'cutaway_mamba',
        mambaTemporalPlaying: true,
      }),
    ).toBe(true);
    expect(
      shouldAnimateCutawayPulse({
        ...baseState,
        mode: 'cutaway_moe',
        moeTokenCycling: true,
      }),
    ).toBe(true);

    // Camera fly & unpaused tower spine wave -> true; paused spine wave -> false
    expect(
      shouldContinueRafLoop({
        state: baseState,
        flyAnimActive: true,
        spineWaveActive: false,
        spineWavePaused: false,
      }),
    ).toBe(true);
    expect(
      shouldContinueRafLoop({
        state: baseState,
        flyAnimActive: false,
        spineWaveActive: true,
        spineWavePaused: false,
      }),
    ).toBe(true);
    expect(
      shouldContinueRafLoop({
        state: baseState,
        flyAnimActive: false,
        spineWaveActive: true,
        spineWavePaused: true,
      }),
    ).toBe(false);
  });

  it('PR-12: computeValueAtlasContentKey is invariant across cell-hover changes within the same head/token/expert and sensitive to texel-affecting inputs', () => {
    const baseInput: ValueAtlasKeyInput = {
      mode: 'cutaway_gqa',
      bayId: 6,
      executionMode: 'full_sequence',
      executionId: 'exec-t8-hash',
      tokenCount: 8,
      precision: 'bf16',
      gqaGroup0Head: 0,
      gqaGroup1Head: 16,
      mambaActiveHead: 0,
      mambaStateToken: 7,
      moeFocusToken: 7,
      moeFocusExpert: 4,
      outputFocusToken: 7,
      stepperActive: false,
      stepperStep: 0,
      stepperTermBucket: -1,
      reducedMotion: false,
      atlasWidth: 128,
      atlasHeight: 64,
      atlasByteLength: 128 * 64 * 8,
    };

    // Hovering across different cells within the same GQA head keeps all atlas inputs identical -> identical key
    const kHoverCellA = computeValueAtlasContentKey(baseInput);
    const kHoverCellB = computeValueAtlasContentKey({ ...baseInput });
    expect(kHoverCellA).toBe(kHoverCellB);

    // Changing any texel-affecting input invalidates the key cleanly
    expect(computeValueAtlasContentKey({ ...baseInput, gqaGroup0Head: 5 })).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, bayId: 43 })).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, tokenCount: 9 })).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, precision: 'fp8_selective' })).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, executionId: 'exec-ar-step-2' })).not.toBe(kHoverCellA);
    expect(
      computeValueAtlasContentKey({
        ...baseInput,
        stepperActive: true,
        stepperStep: 2,
        stepperTermBucket: 1,
      }),
    ).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, reducedMotion: true })).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, atlasWidth: 256 })).not.toBe(kHoverCellA);
    expect(computeValueAtlasContentKey({ ...baseInput, mode: 'cutaway_mamba' })).not.toBe(kHoverCellA);
  });

  it('PR-14: computeSpineWaveGeometryKey throttles 330-frame 5.5s instructional replay to ~55 discrete stage/bay rebuilds', () => {
    let wave = startSpineWavePlayback('instructional', false);
    let lastKey = computeSpineWaveGeometryKey(wave.progress, wave.mode);
    const transitions: string[] = [lastKey];

    const dtMs = 5500 / 330; // ~16.67ms per frame at 60fps
    let frames = 0;
    while (wave.progress !== null && frames < 400) {
      frames++;
      const stepped = advanceSpineWavePlayback(wave, dtMs, false);
      wave = stepped.next;
      const key = computeSpineWaveGeometryKey(wave.progress, wave.mode);
      if (key !== lastKey) {
        lastKey = key;
        transitions.push(key);
      }
    }

    expect(frames).toBeGreaterThanOrEqual(329);
    expect(frames).toBeLessThanOrEqual(332);
    // Embedding (1) + 52 bays (52) + Final RMSNorm (1) + Output (1) + idle (1) = 56 keys
    expect(transitions.length).toBeGreaterThanOrEqual(54);
    expect(transitions.length).toBeLessThanOrEqual(58);
    expect(transitions[0]).toBe('instructional:embedding:none');
    expect(transitions).toContain('instructional:bay:1');
    expect(transitions).toContain('instructional:bay:52');
    expect(transitions).toContain('instructional:final_norm:none');
    expect(transitions).toContain('instructional:output:none');
    expect(transitions[transitions.length - 1]).toBe('idle');
  });
});
