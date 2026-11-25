import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createDefaultHybridModel,
  defaultTokenIdsForLength,
  runHybridExecution,
} from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeBranch,
  getActiveDecodeStep,
  navigateDecodeSession,
  stepDecodeSession,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import { NEMOTRON_SPEC } from '../src/model/spec';
import { createInitialState, createPerspectivePose, type ViewMode } from '../src/model/state';
import {
  computeGqaFocusedAccumulation,
  computeMambaFocusedAccumulation,
  computeMoeFocusedAccumulation,
} from '../src/scene/accumulation';
import {
  buildEmbeddingCutaway,
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
  buildOutputCutaway,
  type CutawaySceneGeometry,
  type EmbeddingCellFormula,
  type OutputCellFormula,
} from '../src/scene/cutaways';
import { describeSpineWaveStage, SPINE_REPLAY_DISCLAIMER } from '../src/scene/spine';
import {
  TOUR_PREFILL_TOKEN_IDS,
  WALKTHROUGH_BEAT_COUNT,
  WALKTHROUGH_BEATS,
} from '../src/story/walkthrough';
import {
  buildCrossBayNeighbors,
  computeEditorialCacheKey,
  MacroControls,
  type ControlsCallbacks,
  type ControlsElements,
  type CrossBayNeighbor,
} from '../src/ui/controls';

// ============================================================================
// Minimal Deterministic DOM Shim (matches tests/overlay-callout.test.ts pattern)
// ============================================================================

class MockClassList {
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

type EventListenerFn = (ev: { target?: unknown; stopPropagation: () => void }) => void;

class MockElement {
  readonly tagName: string;
  id = '';
  type = '';
  title = '';
  value = '';
  disabled = false;
  hidden = false;
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  readonly attributes = new Map<string, string>();
  readonly classList = new MockClassList();
  readonly children: MockElement[] = [];
  parentNode: MockElement | null = null;
  private readonly listeners = new Map<string, EventListenerFn[]>();
  private _textContent = '';

  constructor(tagName: string) {
    this.tagName = tagName.toUpperCase();
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
    if (name.startsWith('data-')) {
      const camel = name
        .slice(5)
        .replace(/-([a-z])/g, (_, ch: string) => ch.toUpperCase());
      this.dataset[camel] = value;
    }
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  appendChild<T extends MockElement>(child: T): T {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  append(...nodes: MockElement[]): void {
    for (const n of nodes) this.appendChild(n);
  }

  contains(other: unknown): boolean {
    if (other === this) return true;
    return this.children.some((c) => c.contains(other));
  }

  addEventListener(type: string, fn: EventListenerFn): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(fn);
    this.listeners.set(type, arr);
  }

  removeEventListener(type: string, fn: EventListenerFn): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    const idx = arr.indexOf(fn);
    if (idx >= 0) arr.splice(idx, 1);
  }

  click(): void {
    let stopped = false;
    const ev = {
      target: this,
      stopPropagation: () => {
        stopped = true;
      },
    };
    let cur: MockElement | null = this;
    while (cur) {
      for (const fn of cur.listeners.get('click') ?? []) {
        fn(ev);
      }
      if (stopped) break;
      cur = cur.parentNode;
    }
  }

  findAll(predicate: (el: MockElement) => boolean): MockElement[] {
    const out: MockElement[] = [];
    const visit = (node: MockElement) => {
      if (predicate(node)) out.push(node);
      for (const child of node.children) visit(child);
    };
    visit(this);
    return out;
  }

  findOne(predicate: (el: MockElement) => boolean): MockElement | null {
    return this.findAll(predicate)[0] ?? null;
  }

  querySelector(selector: string): MockElement | null {
    if (selector.startsWith('.')) {
      const cls = selector.slice(1);
      return this.findOne((el) => el.classList.contains(cls));
    }
    if (selector.startsWith('#')) {
      const id = selector.slice(1);
      return this.findOne((el) => el.id === id);
    }
    return null;
  }
}

const origDocument = globalThis.document;

function installMockDocument(): void {
  const docListeners = new Map<string, EventListenerFn[]>();
  (globalThis as unknown as { document: unknown }).document = {
    createElement: (tag: string) => new MockElement(tag),
    addEventListener: (type: string, fn: EventListenerFn) => {
      const arr = docListeners.get(type) ?? [];
      arr.push(fn);
      docListeners.set(type, arr);
    },
    removeEventListener: (type: string, fn: EventListenerFn) => {
      const arr = docListeners.get(type);
      if (!arr) return;
      const idx = arr.indexOf(fn);
      if (idx >= 0) arr.splice(idx, 1);
    },
  };
}

function makeButton(attrs?: {
  id?: string;
  className?: string;
  dataset?: Record<string, string>;
}): MockElement {
  const el = new MockElement('button');
  if (attrs?.id) el.id = attrs.id;
  if (attrs?.className) el.className = attrs.className;
  if (attrs?.dataset) Object.assign(el.dataset, attrs.dataset);
  return el;
}

function createMockControlsElements(): {
  els: ControlsElements;
  raw: {
    modeButtons: MockElement[];
    tourToggleBtn: MockElement;
    editorialPanel: MockElement;
    gqaMathDock: MockElement;
    gqaFormulaTensor: MockElement;
    gqaFormulaCoord: MockElement;
    gqaFormulaHead: MockElement;
    gqaFormulaValue: MockElement;
    gqaFormulaEq: MockElement;
    gqaFormulaSub: MockElement;
    gqaCellClear: MockElement;
    gqaStepPills: MockElement[];
    floatingBadge: MockElement;
    orbitHint: MockElement;
    spineWaveHud: MockElement;
  };
} {
  const modeValues: ViewMode[] = [
    'tower',
    'cutaway_embedding',
    'cutaway_mamba',
    'cutaway_moe',
    'cutaway_gqa',
    'cutaway_output',
  ];
  const modeButtons = modeValues.map((m) =>
    makeButton({
      id: m === 'cutaway_embedding' ? 'tab-embed' : m === 'cutaway_output' ? 'tab-output' : undefined,
      className: 'mode-btn',
      dataset: { mode: m, view: m },
    }),
  );
  const gqaStepPills = Array.from({ length: 6 }, (_, i) =>
    makeButton({ className: 'gqa-step-pill', dataset: { gqaStep: String(i) } }),
  );

  const tourToggleBtn = makeButton({ id: 'tour-toggle-btn' });
  const editorialPanel = new MockElement('div');
  const gqaMathDock = new MockElement('section');
  const gqaFormulaTensor = new MockElement('span');
  const gqaFormulaCoord = new MockElement('span');
  const gqaFormulaHead = new MockElement('span');
  const gqaFormulaValue = new MockElement('span');
  const gqaFormulaEq = new MockElement('div');
  const gqaFormulaSub = new MockElement('div');
  const gqaCellClear = makeButton({ id: 'gqa-cell-clear' });
  const floatingBadge = new MockElement('div');
  const orbitHint = new MockElement('div');
  const spineWaveHud = new MockElement('div');
  spineWaveHud.id = 'spine-wave-hud';
  spineWaveHud.className = 'spine-wave-hud';
  spineWaveHud.dataset.testid = 'spine-wave-hud';
  spineWaveHud.hidden = true;

  const els = {
    sidebarTop: new MockElement('section'),
    sidebarTopBody: new MockElement('div'),
    sidebarTocToggle: makeButton({ id: 'sidebar-toc-toggle' }),
    sidebarCurrentSection: new MockElement('span'),
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
    tourToggleBtn,
    floatingToolbar: new MockElement('div'),
    towerSubbar: new MockElement('div'),
    mambaSubbar: new MockElement('div'),
    moeSubbar: new MockElement('div'),
    gqaSubbar: new MockElement('div'),
    floatingBadge,
    orbitHint,
    gqaMathDock,
    gqaFormulaTensor,
    gqaFormulaCoord,
    gqaFormulaHead,
    gqaFormulaValue,
    gqaFormulaNoteBtn: makeButton({ id: 'gqa-formula-note-btn' }),
    gqaFormulaNote: new MockElement('div'),
    gqaCellClear,
    gqaFormulaEq,
    gqaFormulaSub,
    gqaStepPrev: makeButton(),
    gqaStepPlay: makeButton(),
    gqaStepNext: makeButton(),
    gqaStepPills,
    gqaStepAutoFollow: makeButton(),
    gqaStepScrub: new MockElement('input'),
    editorialPanel,
    execModeFull: makeButton(),
    execModeDecode: makeButton(),
    tokenSlider: new MockElement('input'),
    tokenDec: makeButton(),
    tokenInc: makeButton(),
    tokenValue: new MockElement('output'),
    precisionBf16: makeButton(),
    precisionFp8: makeButton(),
    memoryReadout: new MockElement('div'),
    memoryReadoutLines: new MockElement('div'),
    memoryChartBtn: makeButton(),
    bayStepRow: new MockElement('div'),
    bayPrev: makeButton(),
    bayNext: makeButton(),
    bayClear: makeButton(),
    zoomBayBtn: makeButton(),
    fitBtn: makeButton(),
    resetBtn: makeButton(),
    shortcutsBtn: makeButton(),
    shortcutsPopover: new MockElement('div'),
    shortcutsCloseBtn: makeButton(),
    spineWaveHud,
  } as unknown as ControlsElements;

  return {
    els,
    raw: {
      modeButtons,
      tourToggleBtn,
      editorialPanel,
      gqaMathDock,
      gqaFormulaTensor,
      gqaFormulaCoord,
      gqaFormulaHead,
      gqaFormulaValue,
      gqaFormulaEq,
      gqaFormulaSub,
      gqaCellClear,
      gqaStepPills,
      floatingBadge,
      orbitHint,
      spineWaveHud,
    },
  };
}

function createSpyCallbacks(overrides: Partial<ControlsCallbacks> = {}): ControlsCallbacks {
  return {
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
    ...overrides,
  };
}

afterEach(() => {
  (globalThis as unknown as { document: unknown }).document = origDocument;
});

// ============================================================================
// 1. Editorial Cache Key Reactivity (V4 + V5)
// ============================================================================

describe('V4/V5 computeEditorialCacheKey reactivity', () => {
  const defaultPose = createPerspectivePose([0, 0, 0], 18);

  it('reacts immediately to tourBeat and tourExploring transitions', () => {
    const state = createInitialState(defaultPose, false);
    state.tourActive = true;
    state.tourBeat = 0;
    state.tourExploring = false;

    const kBeat0 = computeEditorialCacheKey(state, null);

    state.tourBeat = 4; // Beat 3.1 (Mamba-2)
    const kBeat4 = computeEditorialCacheKey(state, null);
    expect(kBeat4).not.toBe(kBeat0);

    state.tourExploring = true;
    const kBeat4Exploring = computeEditorialCacheKey(state, null);
    expect(kBeat4Exploring).not.toBe(kBeat4);

    state.tourExploring = false;
    expect(computeEditorialCacheKey(state, null)).toBe(kBeat4);
  });

  it('reacts to cutaway_embedding, cutaway_output, embeddingFocusToken, outputFocusToken, and spineWaveProgress', () => {
    const state = createInitialState(defaultPose, false);
    state.tourActive = false;

    state.mode = 'tower';
    const kTowerIdle = computeEditorialCacheKey(state, null);

    // spineWaveProgress null -> active changes key once, stays stable across intermediate progress ticks
    state.spineWaveProgress = 0.25;
    const kTowerWaveA = computeEditorialCacheKey(state, null);
    state.spineWaveProgress = 0.75;
    const kTowerWaveB = computeEditorialCacheKey(state, null);
    expect(kTowerWaveA).not.toBe(kTowerIdle);
    expect(kTowerWaveA).toBe(kTowerWaveB);
    state.spineWaveProgress = null;

    // cutaway_embedding + embeddingFocusToken
    state.mode = 'cutaway_embedding';
    state.embeddingFocusToken = 7;
    const kEmbed7 = computeEditorialCacheKey(state, null);
    expect(kEmbed7).not.toBe(kTowerIdle);

    state.embeddingFocusToken = 2;
    const kEmbed2 = computeEditorialCacheKey(state, null);
    expect(kEmbed2).not.toBe(kEmbed7);

    // cutaway_output + outputFocusToken
    state.mode = 'cutaway_output';
    state.outputFocusToken = 7;
    const kOut7 = computeEditorialCacheKey(state, null);
    expect(kOut7).not.toBe(kEmbed2);

    state.outputFocusToken = 1;
    const kOut1 = computeEditorialCacheKey(state, null);
    expect(kOut1).not.toBe(kOut7);
  });
});

// ============================================================================
// 2. Cross-Bay Continuity Breadcrumb Helper (`buildCrossBayNeighbors`)
// ============================================================================

describe('V4 buildCrossBayNeighbors helper', () => {
  it('resolves exact upstream/current/downstream continuity for Embedding, Output, and key bays', () => {
    // Embedding (start of network)
    const embed = buildCrossBayNeighbors('cutaway_embedding', 0);
    expect(embed).not.toBeNull();
    expect(embed!.prev).toBeNull();
    expect(embed!.current).toMatchObject({ kind: 'embedding', bayId: 0, mode: 'cutaway_embedding' });
    expect(embed!.next).toMatchObject({ kind: 'M', bayId: 1, mode: 'cutaway_mamba' });

    // Bay 01 Mamba-2 (Embed -> Bay 01 -> Bay 02 MoE)
    const bay01 = buildCrossBayNeighbors('cutaway_mamba', 1);
    expect(bay01!.prev).toMatchObject({ kind: 'embedding', bayId: 0, mode: 'cutaway_embedding' });
    expect(bay01!.current).toMatchObject({ kind: 'M', bayId: 1, label: 'Bay 01 Mamba-2' });
    expect(bay01!.next).toMatchObject({ kind: 'E', bayId: 2, mode: 'cutaway_moe' });

    // Bay 02 MoE (Bay 01 M -> Bay 02 MoE -> Bay 03 M)
    const bay02 = buildCrossBayNeighbors('cutaway_moe', 2);
    expect(bay02!.prev).toMatchObject({ kind: 'M', bayId: 1, shortLabel: 'Bay 01 M' });
    expect(bay02!.current).toMatchObject({ kind: 'E', bayId: 2, label: 'Bay 02 MoE' });
    expect(bay02!.next).toMatchObject({ kind: 'M', bayId: 3, shortLabel: 'Bay 03 M' });

    // Bay 06 GQA (Bay 05 M feeder -> Bay 06 GQA -> Bay 07 MoE)
    const bay06 = buildCrossBayNeighbors('cutaway_gqa', 6);
    expect(bay06!.prev).toMatchObject({ kind: 'M', bayId: 5, mode: 'cutaway_mamba' });
    expect(bay06!.current).toMatchObject({ kind: '*', bayId: 6, label: 'Bay 06 GQA' });
    expect(bay06!.next).toMatchObject({ kind: 'E', bayId: 7, mode: 'cutaway_moe' });

    // Bay 42 Mamba Feeder -> Bay 43 GQA
    const bay42 = buildCrossBayNeighbors('cutaway_mamba', 42);
    expect(bay42!.prev).toMatchObject({ kind: 'E', bayId: 41 });
    expect(bay42!.current).toMatchObject({ kind: 'M', bayId: 42 });
    expect(bay42!.next).toMatchObject({ kind: '*', bayId: 43, mode: 'cutaway_gqa' });

    // Bay 52 Final MoE -> Output LM Head
    const bay52 = buildCrossBayNeighbors('cutaway_moe', 52);
    expect(bay52!.prev).toMatchObject({ kind: 'M', bayId: 51 });
    expect(bay52!.current).toMatchObject({ kind: 'E', bayId: 52 });
    expect(bay52!.next).toMatchObject({ kind: 'output', bayId: 53, mode: 'cutaway_output' });

    // Output LM Head (end of network)
    const output = buildCrossBayNeighbors('cutaway_output', 53);
    expect(output!.prev).toMatchObject({ kind: 'E', bayId: 52, mode: 'cutaway_moe' });
    expect(output!.current).toMatchObject({ kind: 'output', bayId: 53, mode: 'cutaway_output' });
    expect(output!.next).toBeNull();

    // Tower without a bayId returns null
    expect(buildCrossBayNeighbors('tower')).toBeNull();
  });
});

// ============================================================================
// 3. MacroControls End-to-End DOM & Callback Integration (V4 + V5)
// ============================================================================

describe('V4/V5 MacroControls UI integration', () => {
  const defaultPose = createPerspectivePose([0, 0, 0], 18);

  it('wires all 6 mode tabs including #tab-embed and #tab-output to onSelectMode and syncs active state', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const cb = createSpyCallbacks();
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);

    const tabEmbed = raw.modeButtons.find((b) => b.id === 'tab-embed')!;
    const tabOutput = raw.modeButtons.find((b) => b.id === 'tab-output')!;
    tabEmbed.click();
    tabOutput.click();

    expect(cb.onSelectMode).toHaveBeenCalledWith('cutaway_embedding');
    expect(cb.onSelectMode).toHaveBeenCalledWith('cutaway_output');

    const state = createInitialState(defaultPose, false);
    state.tourActive = false;
    state.mode = 'cutaway_embedding';
    controls.sync(state);
    expect(tabEmbed.classList.contains('is-active')).toBe(true);
    expect(tabEmbed.getAttribute('aria-selected')).toBe('true');
    expect(tabOutput.classList.contains('is-active')).toBe(false);

    state.mode = 'cutaway_output';
    controls.sync(state);
    expect(tabOutput.classList.contains('is-active')).toBe(true);
    expect(tabOutput.getAttribute('aria-selected')).toBe('true');
    expect(tabEmbed.classList.contains('is-active')).toBe(false);

    controls.dispose();
  });

  it('renders the V5 22-beat Guided Walkthrough card with progress rail, exploring pill, beat bar, embedded AR decode, and Space cue', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const onTourBeatPrev = vi.fn();
    const onTourBeatNext = vi.fn();
    const onTourJumpBeat = vi.fn();
    const onTourResumeBeat = vi.fn();
    const onTourRefitBeat = vi.fn();
    const onTourToggleBeatPlay = vi.fn();
    const onReplaySpineWave = vi.fn();
    const cb = createSpyCallbacks({
      onTourBeatPrev,
      onTourBeatNext,
      onTourJumpBeat,
      onTourResumeBeat,
      onTourRefitBeat,
      onTourToggleBeatPlay,
      onReplaySpineWave,
    });
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);

    // Beat 3.2 (index 5, cutaway_mamba with stepperStep=2, stepperAutoPlay=true) in exploring state
    const state = createInitialState(defaultPose, false);
    state.tourActive = true;
    state.tourBeat = 5;
    state.tourExploring = true;
    state.mode = 'cutaway_mamba';
    state.mambaBayId = 1;
    state.mambaStepperActive = true;
    state.mambaStepperPlaying = true;
    controls.sync(state);

    expect(raw.tourToggleBtn.textContent).toContain('Beat 3.2');
    expect(raw.tourToggleBtn.textContent).toContain(`6/${WALKTHROUGH_BEAT_COUNT}`);

    const counter = raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-chapter-counter');
    expect(counter?.textContent).toContain('Act I');
    expect(counter?.textContent).toContain('Beat 3.2');
    expect(counter?.textContent).toContain('6 / 22');

    // Progress rail has 22 clickable pips
    const rail = raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-progress-rail');
    expect(rail).not.toBeNull();
    const pips = rail!.findAll((el) => el.tagName === 'BUTTON' && el.dataset.tourBeat !== undefined);
    expect(pips).toHaveLength(WALKTHROUGH_BEAT_COUNT);
    pips[12].click();
    expect(onTourJumpBeat).toHaveBeenCalledWith(12);

    // Prev/Next beat buttons
    raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-beat-prev')!.click();
    raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-beat-next')!.click();
    expect(onTourBeatPrev).toHaveBeenCalledTimes(1);
    expect(onTourBeatNext).toHaveBeenCalledTimes(1);

    // Exploring freely pill + Resume Beat button keeps narrative visible
    const expPill = raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-exploring-pill');
    expect(expPill).not.toBeNull();
    expect(expPill!.textContent).toContain('Exploring freely');
    expect(expPill!.textContent).toContain('Resume Beat 3.2');
    raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-resume-beat')!.click();
    expect(onTourResumeBeat).toHaveBeenCalledTimes(1);

    // Inline beat bar: Refit, Pause (since mambaStepperPlaying=true), and 4 chapter dots
    const beatBar = raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-beat-bar');
    expect(beatBar).not.toBeNull();
    const refitBtn = beatBar!.findOne((el) => el.dataset.testid === 'tour-beat-refit')!;
    const playBtn = beatBar!.findOne((el) => el.dataset.testid === 'tour-beat-play')!;
    expect(playBtn.textContent).toContain('Pause');
    refitBtn.click();
    playBtn.click();
    expect(onTourRefitBeat).toHaveBeenCalledTimes(1);
    expect(onTourToggleBeatPlay).toHaveBeenCalledTimes(1);
    const chDots = beatBar!.findAll((el) => el.classList.contains('tour-beat-dot'));
    expect(chDots).toHaveLength(4);

    // Space hint renders beat.interactivePrompt
    const spaceHint = raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-space-hint');
    expect(spaceHint?.textContent).toBe(WALKTHROUGH_BEATS[5].interactivePrompt);

    // Act III Beat 7.2 (index 19, executionMode: 'ar_decode', mode: 'tower') embeds AR Decode + Replay Wave!
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    session = stepDecodeSessionGreedy(session);
    state.tourBeat = 19;
    state.tourExploring = false;
    state.mode = 'tower';
    state.executionMode = 'ar_decode';
    controls.sync(state, null, false, null, session);

    expect(raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-exploring-pill')).toBeNull();
    expect(raw.editorialPanel.findOne((el) => el.dataset.testid === 'ar-decode-panel')).not.toBeNull();
    const replayBtn = raw.editorialPanel.findOne((el) => el.dataset.testid === 'ar-replay-wave');
    expect(replayBtn).not.toBeNull();
    expect(replayBtn!.textContent).toContain('Fast Wave (0.9s)');
    replayBtn!.click();
    expect(onReplaySpineWave).toHaveBeenCalledTimes(1);

    controls.dispose();
  });

  it('renders dedicated Embedding & Output editorial panels, cross-bay breadcrumbs with live RMS, and syncs Math Dock formulas', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const onSelectEmbeddingToken = vi.fn();
    const onSelectOutputToken = vi.fn();
    const onNavigateAdjacentBay = vi.fn<(n: CrossBayNeighbor) => void>();
    const cb = createSpyCallbacks({
      onSelectEmbeddingToken,
      onSelectOutputToken,
      onNavigateAdjacentBay,
    });
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);

    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });

    // 1. Embedding Cutaway
    const embedSceneBase = buildEmbeddingCutaway({
      model,
      embedding: { X: exec.X_embed },
      tokenIds: exec.tokenIds,
      tokenLabels: exec.tokenLabels,
      focusToken: 3,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'x_embed', row: 2, col: 3, tokenIndex: 3 },
    });
    const embedScene: CutawaySceneGeometry = {
      ...embedSceneBase,
      embeddingFormula: embedSceneBase.activeFormula as EmbeddingCellFormula,
    };

    const state = createInitialState(defaultPose, false);
    state.tourActive = false;
    state.mode = 'cutaway_embedding';
    state.embeddingFocusToken = 3;
    state.embeddingActiveCell = { tensor: 'x_embed', row: 2, col: 3, tokenIndex: 3 };
    controls.sync(state, embedScene, false, exec, null);

    expect(raw.editorialPanel.textContent).toContain('Embedding Cutaway (NoPE)');
    expect(raw.editorialPanel.textContent).toContain('131,072 × 2,688 · Untied');
    expect(raw.editorialPanel.findOne((el) => el.dataset.testid === 'embedding-rms-readout')?.textContent).toContain('RMS(X₀[:, t3])');

    // Clickable token chip calls onSelectEmbeddingToken
    const tokChip = raw.editorialPanel.findOne((el) => el.dataset.embeddingToken === '5')!;
    tokChip.click();
    expect(onSelectEmbeddingToken).toHaveBeenCalledWith(5);

    // CTA button navigates to Bay 01 Mamba-2
    raw.editorialPanel.findOne((el) => el.dataset.testid === 'embed-cta-bay01')!.click();
    expect(cb.onInspectInternals).toHaveBeenCalledWith('M', 1);

    // Breadcrumb shows Embed X₀ -> Bay 01 M with live RMS and navigates on click
    const embedBc = raw.editorialPanel.findOne((el) => el.dataset.testid === 'cutaway-breadcrumb')!;
    expect(embedBc.textContent).toContain('Embed X₀');
    expect(embedBc.textContent).toContain('Bay 01 M');
    const nextBcBtn = embedBc.findOne((el) => el.dataset.breadcrumbDir === 'next')!;
    nextBcBtn.click();
    expect(onNavigateAdjacentBay).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'M', bayId: 1, mode: 'cutaway_mamba' }),
    );

    // Math Dock synced with embeddingFormula and hides unused pills 4 & 5
    expect(raw.gqaMathDock.hidden).toBe(false);
    expect(raw.gqaFormulaTensor.textContent).toBe(embedScene.embeddingFormula!.tensorLabel);
    expect(raw.gqaFormulaEq.textContent).toBe(embedScene.embeddingFormula!.equation);
    expect(raw.gqaStepPills[0].hidden).toBe(false);
    expect(raw.gqaStepPills[3].hidden).toBe(false);
    expect(raw.gqaStepPills[4].hidden).toBe(true);
    expect(raw.gqaStepPills[5].hidden).toBe(true);

    // 2. Output Cutaway
    const outSceneBase = buildOutputCutaway({
      model,
      residualIn: exec.residualStreams[52],
      finalNorm: exec.finalNorm,
      lmHead: exec.lmHead,
      tokenLabels: exec.tokenLabels,
      focusToken: 7,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'probs', row: 0, col: 0, tokenIndex: 7, vocabId: 0 },
    });
    const outScene: CutawaySceneGeometry = {
      ...outSceneBase,
      outputFormula: outSceneBase.activeFormula as OutputCellFormula,
    };

    state.mode = 'cutaway_output';
    state.outputFocusToken = 7;
    state.outputActiveCell = { tensor: 'probs', row: 0, col: 0, tokenIndex: 7, vocabId: 0 };
    controls.sync(state, outScene, false, exec, null);

    expect(raw.editorialPanel.textContent).toContain('Output & Untied LM Head');
    expect(raw.editorialPanel.textContent).toContain('2,688 → 131,072 · ε=1e−5');
    expect(raw.editorialPanel.findOne((el) => el.dataset.testid === 'output-top5-bars')).not.toBeNull();
    expect(raw.editorialPanel.findOne((el) => el.dataset.testid === 'output-synthetic-disclaimer')?.textContent).toContain('Synthetic');

    const outTokChip = raw.editorialPanel.findOne((el) => el.dataset.outputToken === '2')!;
    outTokChip.click();
    expect(onSelectOutputToken).toHaveBeenCalledWith(2);

    raw.editorialPanel.findOne((el) => el.dataset.testid === 'output-cta-bay52')!.click();
    expect(cb.onInspectInternals).toHaveBeenCalledWith('E', 52);

    // Output breadcrumb shows Bay 52 MoE -> Output LM Head
    const outBc = raw.editorialPanel.findOne((el) => el.dataset.testid === 'cutaway-breadcrumb')!;
    expect(outBc.textContent).toContain('Bay 52 MoE');
    expect(outBc.textContent).toContain('Output LM Head');
    outBc.findOne((el) => el.dataset.breadcrumbDir === 'prev')!.click();
    expect(onNavigateAdjacentBay).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'E', bayId: 52, mode: 'cutaway_moe' }),
    );

    // Math Dock synced with outputFormula and restores all 6 pills
    expect(raw.gqaFormulaTensor.textContent).toBe(outScene.outputFormula!.tensorLabel);
    expect(raw.gqaFormulaEq.textContent).toBe(outScene.outputFormula!.equation);
    expect(raw.gqaStepPills.every((p) => p.hidden === false)).toBe(true);

    controls.dispose();
  });

  it('falls back cleanly when optional V4/V5 callbacks are omitted (Wave 1/2 backward compatibility)', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const cb = createSpyCallbacks(); // NO optional V4/V5 callbacks supplied!
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);

    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });
    const state = createInitialState(defaultPose, false);
    state.tourActive = true;
    state.tourBeat = 2; // Beat 2.1
    controls.sync(state, null, false, exec, null);

    // Prev/Next fall back to onTourPrev/onTourNext
    raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-beat-prev')!.click();
    raw.editorialPanel.findOne((el) => el.dataset.testid === 'tour-beat-next')!.click();
    expect(cb.onTourPrev).toHaveBeenCalledTimes(1);
    expect(cb.onTourNext).toHaveBeenCalledTimes(1);

    // Breadcrumb fallback navigates via onSelectMode / onInspectInternals
    state.tourActive = false;
    state.mode = 'cutaway_mamba';
    state.mambaBayId = 1;
    controls.sync(state, null, false, exec, null);

    const bc = raw.editorialPanel.findOne((el) => el.dataset.testid === 'cutaway-breadcrumb')!;
    bc.findOne((el) => el.dataset.breadcrumbDir === 'prev')!.click();
    expect(cb.onSelectMode).toHaveBeenCalledWith('cutaway_embedding');
    bc.findOne((el) => el.dataset.breadcrumbDir === 'next')!.click();
    expect(cb.onInspectInternals).toHaveBeenCalledWith('E', 2);

    // AR Decode Replay Wave button on Tower does not throw when onReplaySpineWave is omitted
    const session = stepDecodeSessionGreedy(createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] }));
    state.mode = 'tower';
    state.executionMode = 'ar_decode';
    controls.sync(state, null, false, exec, session);
    const replayBtn = raw.editorialPanel.findOne((el) => el.dataset.testid === 'ar-replay-wave')!;
    expect(() => replayBtn.click()).not.toThrow();

    // Active decode step baySummaries also populate breadcrumbs cleanly
    const activeStep = getActiveDecodeStep(session);
    expect(activeStep.baySummaries).toHaveLength(52);
    state.mode = 'cutaway_moe';
    state.moeBayId = 2;
    controls.sync(state, null, false, exec, session);
    const moeBc = raw.editorialPanel.findOne((el) => el.dataset.testid === 'cutaway-breadcrumb')!;
    expect(moeBc.textContent).toContain('Bay 01 M');
    expect(moeBc.textContent).toContain('Bay 02 MoE');
    expect(moeBc.textContent).toContain('Bay 03 M');

    controls.dispose();
  });
});

// ============================================================================
// 4. RC1 / RC2 / RC3 Refinements (Math Dock, Live Act III Copy, Spine HUD)
// ============================================================================

describe('RC1/RC2/RC3 MacroControls refinements', () => {
  const defaultPose = createPerspectivePose([0, 0, 0], 18);

  it('tracks spineWaveMode and active AR decode step digest in computeEditorialCacheKey', () => {
    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [...TOUR_PREFILL_TOKEN_IDS] });
    session = stepDecodeSessionGreedy(session);

    const state = createInitialState(defaultPose, false);
    state.tourActive = true;
    state.tourBeat = 19; // Beat 7.2
    state.mode = 'tower';
    state.executionMode = 'ar_decode';
    state.spineWaveMode = 'fast';
    state.spineWaveProgress = 0.25;

    const kFastWave = computeEditorialCacheKey(state, null, null, null, session);

    // Intermediate progress ticks within the same mode keep editorial cache key stable for 60fps RAF
    state.spineWaveProgress = 0.72;
    expect(computeEditorialCacheKey(state, null, null, null, session)).toBe(kFastWave);

    // Switching fast -> instructional immediately invalidates the cache key
    state.spineWaveMode = 'instructional';
    const kSlowWave = computeEditorialCacheKey(state, null, null, null, session);
    expect(kSlowWave).not.toBe(kFastWave);

    // Stepping greedy further changes step index & state digest
    const stepped = stepDecodeSessionGreedy(session);
    const kStepped = computeEditorialCacheKey(state, null, null, null, stepped);
    expect(kStepped).not.toBe(kSlowWave);

    // Navigating back & forking branch-1 immediately invalidates cache key
    const atPrefill = navigateDecodeSession(stepped, 0);
    const altCand = getActiveDecodeStep(atPrefill).predictedNextTokens[1];
    const branched = stepDecodeSession(atPrefill, altCand.tokenId);
    const kBranched = computeEditorialCacheKey(state, null, null, null, branched);
    expect(kBranched).not.toBe(kSlowWave);
    expect(kBranched).not.toBe(kStepped);
  });

  it('RC1: renders non-spoiling partial Math Dock equations, badges, and .is-accumulating across GQA, Mamba-2, and MoE', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const controls = new MacroControls(NEMOTRON_SPEC, els, createSpyCallbacks());

    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });
    const state = createInitialState(defaultPose, false);
    state.tourActive = false;

    // 1. GQA Step 2 (scores, term 2/4 running)
    const gqaSim = exec.getGqaTrace(6);
    const gqaCell = {
      tensor: 'scores' as const,
      row: 5,
      col: 3,
      headIndex: 0,
      kvHeadIndex: 0 as const,
    };
    const gqaAccum = computeGqaFocusedAccumulation(gqaSim, gqaCell, 2, 0.35);
    expect(gqaAccum).not.toBeNull();
    expect(gqaAccum!.active).toBe(true);

    const gqaBase = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      viewAngle: 'three_quarter',
      selectedPart: null,
      hoveredPart: null,
      simOverride: gqaSim,
      activeCell: gqaCell,
      stepperActive: true,
      stepperStep: 2,
    });
    const gqaWithAccum: CutawaySceneGeometry = {
      ...gqaBase,
      focusedAccumulation: gqaAccum,
    };

    state.mode = 'cutaway_gqa';
    state.gqaBayId = 6;
    state.gqaSelectedCell = gqaCell;
    controls.sync(state, gqaWithAccum, false, exec, null);

    expect(raw.gqaFormulaEq.textContent).toBe(gqaAccum!.partialEquation);
    expect(raw.gqaFormulaEq.textContent).toContain('…');
    expect(raw.gqaFormulaSub.textContent).toBe(gqaAccum!.partialSubEquation);
    expect(raw.gqaFormulaValue.textContent).toBe(gqaAccum!.valueBadgeText);
    expect(raw.gqaFormulaValue.textContent).toContain('(2/4)');
    expect(raw.gqaFormulaValue.classList.contains('is-accumulating')).toBe(true);
    expect(raw.gqaFormulaHead.textContent).toBe(gqaAccum!.badgeText);
    expect(raw.gqaFormulaHead.textContent).toBe('Term 2/4 · Running');
    expect(raw.gqaFormulaHead.hidden).toBe(false);
    // Ensure completed total is NOT leaked while running (when partial differs from final)
    if (gqaBase.activeFormula && gqaAccum!.partialSum !== gqaBase.activeFormula.value) {
      expect(raw.gqaFormulaEq.textContent).not.toBe(gqaBase.activeFormula.equation);
      expect(raw.gqaFormulaValue.textContent).not.toBe(`= ${gqaBase.activeFormula.valueText}`);
    }

    // Clearing focusedAccumulation restores complete unrestricted formula & removes .is-accumulating
    controls.sync(state, { ...gqaBase, focusedAccumulation: null }, false, exec, null);
    expect(raw.gqaFormulaValue.classList.contains('is-accumulating')).toBe(false);
    expect(raw.gqaFormulaEq.textContent).toBe(gqaBase.activeFormula!.equation);
    expect(raw.gqaFormulaSub.textContent).toBe(gqaBase.activeFormula!.subEquation);
    expect(raw.gqaFormulaValue.textContent).toBe(`= ${gqaBase.activeFormula!.valueText}`);
    expect(raw.gqaFormulaHead.textContent).toBe(gqaBase.activeFormula!.headBadge);

    // 2. Mamba-2 Step 3 (h_curr, term 1/2 running)
    const mambaSim = exec.getMambaTrace(1);
    const mambaCell = {
      tensor: 'h_curr' as const,
      row: 1,
      col: 2,
      headIndex: 0,
      tokenIndex: 7,
    };
    const mambaAccum = computeMambaFocusedAccumulation(mambaSim, mambaCell, 3, 0.2);
    expect(mambaAccum).not.toBeNull();
    const mambaBase = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      viewAngle: 'three_quarter',
      selectedPart: null,
      hoveredPart: null,
      simOverride: mambaSim,
      activeCell: mambaCell,
      stepperActive: true,
      stepperStep: 3,
      stepperSubProgress: 0.4,
    });
    state.mode = 'cutaway_mamba';
    state.mambaBayId = 1;
    state.mambaSelectedCell = mambaCell;
    controls.sync(state, { ...mambaBase, focusedAccumulation: mambaAccum }, false, exec, null);

    expect(raw.gqaFormulaEq.textContent).toBe(mambaAccum!.partialEquation);
    expect(raw.gqaFormulaSub.textContent).toBe(mambaAccum!.partialSubEquation);
    expect(raw.gqaFormulaValue.textContent).toBe(mambaAccum!.valueBadgeText);
    expect(raw.gqaFormulaValue.classList.contains('is-accumulating')).toBe(true);
    expect(raw.gqaFormulaHead.textContent).toBe('Term 1/2 · Running');

    // 3. MoE Step 2 (topk_weights, term 2/6 running)
    const moeSim = exec.getMoeTrace(2);
    const moeCell = { tensor: 'topk_weights' as const, row: 0, col: 0, rankSlot: 0, tokenIndex: 7 };
    const moeAccum = computeMoeFocusedAccumulation(moeSim, moeCell, 2, 0.25, 7, moeSim.topk_indices[7 * moeSim.topK]);
    expect(moeAccum).not.toBeNull();
    const moeBase = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      viewAngle: 'three_quarter',
      selectedPart: null,
      hoveredPart: null,
      simOverride: moeSim,
      activeCell: moeCell,
      stepperActive: true,
      stepperStep: 2,
      stepperSubProgress: 0.5,
    });
    state.mode = 'cutaway_moe';
    state.moeBayId = 2;
    state.moeSelectedCell = moeCell;
    controls.sync(state, { ...moeBase, focusedAccumulation: moeAccum }, false, exec, null);

    expect(raw.gqaFormulaEq.textContent).toBe(moeAccum!.partialEquation);
    expect(raw.gqaFormulaSub.textContent).toBe(moeAccum!.partialSubEquation);
    expect(raw.gqaFormulaValue.textContent).toBe(moeAccum!.valueBadgeText);
    expect(raw.gqaFormulaValue.classList.contains('is-accumulating')).toBe(true);
    expect(raw.gqaFormulaHead.textContent).toBe('Term 2/6 · Running');

    // Idle cutaway removes .is-accumulating cleanly
    state.moeSelectedCell = null;
    controls.sync(state, { ...moeBase, activeFormula: null, focusedAccumulation: null }, false, exec, null);
    expect(raw.gqaFormulaValue.classList.contains('is-accumulating')).toBe(false);
    expect(raw.gqaFormulaValue.hidden).toBe(true);

    controls.dispose();
  });

  it('RC2: syncs Act III walkthrough badge, paragraphs, and takeaways dynamically across stepping and branch-1 forks', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const controls = new MacroControls(NEMOTRON_SPEC, els, createSpyCallbacks());

    const model = createDefaultHybridModel();
    let session = createDecodeSession(model, { promptTokenIds: [...TOUR_PREFILL_TOKEN_IDS] });
    session = stepDecodeSessionGreedy(session); // branch-0, step 1, t8 (8P+1S = 9)

    const step1 = getActiveDecodeStep(session);
    const t8Label = step1.sequenceTokenLabels[8];

    const state = createInitialState(defaultPose, false);
    state.tourActive = true;
    state.executionMode = 'ar_decode';
    state.precision = 'fp8_selective';
    state.tokens = step1.totalTokens;

    // Beat 7.2 on branch-0 at t8 (8P+1S = 9)
    state.tourBeat = 19;
    state.mode = 'tower';
    controls.sync(state, null, false, null, session);

    const badge72 = raw.editorialPanel.findOne((el) => el.classList.contains('ed-badge'))!;
    expect(badge72.textContent).toContain('branch-0');
    expect(badge72.textContent).toContain(`t8 "${t8Label}"`);
    expect(badge72.textContent).toContain('8P+1S');
    expect(raw.editorialPanel.textContent).toContain('8 reused prefix + 1 new');
    expect(raw.editorialPanel.textContent).toContain(step1.greedyNextToken.label);

    // Beat 7.3 on branch-0 -> [1 × 9]
    state.tourBeat = 20;
    state.mode = 'cutaway_gqa';
    state.gqaBayId = 43;
    controls.sync(state, null, false, null, session);

    const badge73 = raw.editorialPanel.findOne((el) => el.classList.contains('ed-badge'))!;
    expect(badge73.textContent).toBe('Bay 43 · branch-0 · [1 × 9]');
    expect(raw.editorialPanel.textContent).toContain('rectangular [1 × 9] query row');

    // Step forward once more on branch-0 -> t9 (9P+1S = 10, [1 × 10])
    session = stepDecodeSessionGreedy(session);
    const step2 = getActiveDecodeStep(session);
    state.tokens = step2.totalTokens;
    controls.sync(state, null, false, null, session);

    const badge73Step2 = raw.editorialPanel.findOne((el) => el.classList.contains('ed-badge'))!;
    expect(badge73Step2.textContent).toBe('Bay 43 · branch-0 · [1 × 10]');
    expect(raw.editorialPanel.textContent).toContain('9 reused prefix + 1 new');
    expect(raw.editorialPanel.textContent).toContain(`t9 "${step2.sequenceTokenLabels[9]}"`);

    // Navigate back to prefill (step 0) and fork branch-1 with #2 candidate while exploring!
    const atPrefill = navigateDecodeSession(session, 0);
    const altCand = getActiveDecodeStep(atPrefill).predictedNextTokens[1];
    const branched = stepDecodeSession(atPrefill, altCand.tokenId);
    const branchedStep = getActiveDecodeStep(branched);
    const branchedBranch = getActiveDecodeBranch(branched);
    expect(branchedBranch.branchId).toBe('branch-1');

    state.tourBeat = 21; // Beat 7.4
    state.tourExploring = true;
    state.mode = 'tower';
    state.tokens = branchedStep.totalTokens;
    controls.sync(state, null, false, null, branched);

    const badge74 = raw.editorialPanel.findOne((el) => el.classList.contains('ed-badge'))!;
    expect(badge74.textContent).toContain('branch-1');
    expect(badge74.textContent).toContain(`t8 "${altCand.label}"`);
    expect(raw.editorialPanel.textContent).toContain('branch-1');
    expect(raw.editorialPanel.textContent).toContain(`t8 "${altCand.label}"`);
    expect(raw.editorialPanel.textContent).toContain(branchedStep.greedyNextToken.label);

    controls.dispose();
  });

  it('RC3: renders Fast/Slow replay controls and updates live in-canvas .spine-wave-hud with disclaimer, pause/close, and reduced-motion summary', () => {
    installMockDocument();
    const { els, raw } = createMockControlsElements();
    const onReplaySpineWaveMode = vi.fn();
    const onToggleSpineWavePause = vi.fn();
    const onDismissSpineWaveHud = vi.fn();
    const controls = new MacroControls(
      NEMOTRON_SPEC,
      els,
      createSpyCallbacks({
        onReplaySpineWaveMode,
        onToggleSpineWavePause,
        onDismissSpineWaveHud,
      }),
    );

    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });
    let session = createDecodeSession(model, { promptTokenIds: [...TOUR_PREFILL_TOKEN_IDS] });
    session = stepDecodeSessionGreedy(session);

    const state = createInitialState(defaultPose, false);
    state.tourActive = false;
    state.mode = 'tower';
    state.executionMode = 'ar_decode';
    state.precision = 'fp8_selective';
    state.tokens = 9;
    state.spineWaveMode = 'fast';
    state.spineWaveProgress = null;

    controls.sync(state, null, false, exec, session);

    // 1. AR Wave controls bar in Tower mode
    const waveBar = raw.editorialPanel.findOne((el) => el.dataset.testid === 'ar-wave-controls');
    expect(waveBar).not.toBeNull();
    const fastBtn = waveBar!.findOne((el) => el.dataset.testid === 'ar-replay-wave')!;
    const slowBtn = waveBar!.findOne((el) => el.dataset.testid === 'ar-replay-wave-slow')!;
    expect(fastBtn.textContent).toContain('Fast Wave (0.9s)');
    expect(slowBtn.textContent).toContain('Slow Replay (5.5s)');
    expect(fastBtn.classList.contains('is-active')).toBe(true);
    expect(slowBtn.classList.contains('is-active')).toBe(false);
    expect(waveBar!.textContent).toContain('Educational replay (not measured runtime)');

    fastBtn.click();
    slowBtn.click();
    expect(onReplaySpineWaveMode).toHaveBeenNthCalledWith(1, 'fast');
    expect(onReplaySpineWaveMode).toHaveBeenNthCalledWith(2, 'instructional');

    // HUD stays hidden while spineWaveProgress is null
    expect(raw.spineWaveHud.hidden).toBe(true);

    // 2. Active instructional wave renders HUD with stage dot, title, detail, RMS, progress, disclaimer, Pause & Close
    state.spineWaveMode = 'instructional';
    state.spineWaveProgress = 0.14; // Bay 06 GQA in 0.05..0.91 span
    state.spineWavePaused = false;
    controls.sync(state, null, false, exec, session);

    // Slow button now highlighted in sidebar
    const slowBtnActive = raw.editorialPanel.findOne(
      (el) => el.dataset.testid === 'ar-replay-wave-slow',
    )!;
    expect(slowBtnActive.classList.contains('is-active')).toBe(true);

    const activeStep = getActiveDecodeStep(session);
    const expectedStage = describeSpineWaveStage(
      0.14,
      NEMOTRON_SPEC,
      'fp8_selective',
      activeStep.baySummaries,
      0,
    );
    expect(raw.spineWaveHud.hidden).toBe(false);
    expect(raw.spineWaveHud.dataset.stageKind).toBe(expectedStage.stageKind);
    const dotEl = raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-dot')!;
    expect(dotEl.classList.contains(`is-${expectedStage.stageKind}`)).toBe(true);
    expect(
      raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-title')?.textContent,
    ).toBe(expectedStage.title);
    expect(
      raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-detail')?.textContent,
    ).toBe(expectedStage.actionDetail);
    expect(
      raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-rms')?.textContent,
    ).toBe(expectedStage.rmsText);
    expect(
      raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-fill')?.style.width,
    ).toBe(`${expectedStage.progressPct}%`);
    expect(
      raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-disclaimer')
        ?.textContent,
    ).toBe(SPINE_REPLAY_DISCLAIMER);

    const pauseBtn = raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-pause')!;
    const closeBtn = raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-close')!;
    expect(pauseBtn.hidden).toBe(false);
    expect(pauseBtn.textContent).toContain('Pause');
    pauseBtn.click();
    closeBtn.click();
    expect(onToggleSpineWavePause).toHaveBeenCalledTimes(1);
    expect(onDismissSpineWaveHud).toHaveBeenCalledTimes(1);

    // 3. Direct RAF update via public syncSpineWaveHud updates HUD smoothly without rebuilding editorial DOM
    const editorialSnapshot = raw.editorialPanel.children[0];
    state.spineWavePaused = true;
    state.spineWaveProgress = 0.98; // Output stage (no RMS pill)
    controls.syncSpineWaveHud(state, exec, session);
    expect(raw.editorialPanel.children[0]).toBe(editorialSnapshot);
    expect(raw.spineWaveHud.dataset.stageKind).toBe('output');
    expect(dotEl.classList.contains('is-output')).toBe(true);
    expect(raw.spineWaveHud.findOne((el) => el.dataset.testid === 'spine-wave-hud-rms')?.hidden).toBe(
      true,
    );
    expect(pauseBtn.textContent).toContain('Resume');

    // 4. Fast mode hides Pause button; Reduced motion hides Pause and adds explicit static summary note
    state.spineWaveMode = 'fast';
    state.spineWavePaused = false;
    state.spineWaveProgress = 0.02; // Embedding stage
    controls.syncSpineWaveHud(state, exec, session);
    expect(pauseBtn.hidden).toBe(true);
    expect(raw.spineWaveHud.dataset.stageKind).toBe('embedding');

    state.spineWaveMode = 'instructional';
    state.reducedMotion = true;
    state.spineWaveProgress = 0.93; // Final RMSNorm stage
    controls.syncSpineWaveHud(state, exec, session);
    expect(pauseBtn.hidden).toBe(true);
    expect(dotEl.classList.contains('is-final_norm')).toBe(true);
    const reducedDisclaimer = raw.spineWaveHud.findOne(
      (el) => el.dataset.testid === 'spine-wave-hud-disclaimer',
    )!.textContent;
    expect(reducedDisclaimer).toContain(SPINE_REPLAY_DISCLAIMER);
    expect(reducedDisclaimer).toContain('Static replay summary (reduced motion)');

    // 5. Leaving Tower hides HUD
    state.mode = 'cutaway_gqa';
    controls.syncSpineWaveHud(state, exec, session);
    expect(raw.spineWaveHud.hidden).toBe(true);

    controls.dispose();
  });
});
