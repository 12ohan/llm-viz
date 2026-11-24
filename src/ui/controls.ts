import { formatMemoryReadout, memoryFor } from '../model/memory';
import {
  bayKindLabel,
  bodyPrecision,
  conv1dPrecision,
  formatBayId,
  formatBracketCaption,
  kvCachePrecision,
  PARAMETER_CAPTION,
  ssmStateDtype,
  type BayDescriptor,
  type NemotronSpec,
  type PrecisionMode,
} from '../model/spec';

type BayKind = BayDescriptor['kind'];
import {
  MAX_TOKENS,
  MIN_TOKENS,
  type GqaViewAngle,
  type SceneState,
  type TocSectionId,
  type ViewMode,
} from '../model/state';
import { GQA_STEPPER_STEPS } from '../math/gqaSim';
import { MAMBA_STEPPER_STEPS } from '../math/mambaSim';
import { formatExpertId, MOE_STEPPER_STEPS } from '../math/moeSim';
import {
  formatMambaHeadId,
  formatQueryHeadId,
  mambaGroupForHead,
  parseMambaHeadPartId,
  parseQueryHeadPartId,
  type CutawayPartInfo,
  type CutawaySceneGeometry,
} from '../scene/cutaways';
import { getWalkthroughChapter, WALKTHROUGH_COUNT } from '../story/walkthrough';

export interface ControlsCallbacks {
  onSelectMode: (mode: ViewMode) => void;
  onSelectSection: (section: TocSectionId) => void;
  onSelectMambaBay: (bayId: 1 | 42) => void;
  onSelectMambaView: (view: GqaViewAngle) => void;
  onSelectMoeBay: (bayId: 2 | 52) => void;
  onSelectMoeView: (view: GqaViewAngle) => void;
  onSelectMoeFocusToken: (tokenIdx: number) => void;
  onSelectMoeExpert: (expertId: number) => void;
  onSelectGqaBay: (bayId: 6 | 43) => void;
  onSelectGqaView: (view: GqaViewAngle) => void;
  onSelectPart: (partId: string | null) => void;
  onInspectInternals: (kind: BayKind, bayId?: number) => void;
  onTokensChange: (tokens: number) => void;
  onPrecisionChange: (mode: PrecisionMode) => void;
  onFitAll: () => void;
  onZoomSelectedBay: () => void;
  onReset: () => void;
  onStepBay: (delta: -1 | 1) => void;
  onClearSelection: () => void;
  onTourToggle: () => void;
  onTourPrev: () => void;
  onTourNext: () => void;
  onTourRestart: () => void;
  onTourExit: () => void;
  onToggleMemoryChart: () => void;
  onToggleShortcuts: () => void;
  onCloseShortcuts: () => void;
  onGqaStepPrev: () => void;
  onGqaStepNext: () => void;
  onGqaStepTogglePlay: () => void;
  onGqaStepSelect: (step: number) => void;
  onGqaStepScrub: (progress01: number) => void;
  onGqaClearCell: () => void;
}

interface SectionCopy {
  readonly title: string;
  readonly badge: string;
  readonly body: string;
  readonly specs: readonly string[];
  readonly inspectKind?: BayKind;
}

const WEIGHTS_SUMMARY_LINE = 'Weights: 31.6B (~63 GB BF16, ~94% MoE) · Active: 3.2B/tok';
const SHARED_EXPERT_NOTE =
  'Shared expert width 3,712 (= 2 × 1,856 in NVIDIA paper, fused in checkpoint)';

const SECTION_COPY: Record<TocSectionId, SectionCopy> = {
  overview: {
    title: '52-Layer Hybrid Architecture',
    badge: '23 Mamba-2 · 23 MoE · 6 GQA',
    body:
      'Nemotron 3 Nano replaces most quadratic attention blocks with linear-time Mamba-2 state-space layers and sparse Mixture-of-Experts banks. Only six bays (06, 13, 20, 27, 34, 43) accumulate K/V token history, while a continuous 2,688-channel emerald residual stream carries hidden state top to bottom.',
    specs: [
      PARAMETER_CAPTION,
      'Figure 2 layout: 01–35 [M·E·M·E·M·*·E] ×5 · 36–41 [M·E] ×3 · 42–44 [M·*·E] ×1 · 45–52 [M·E] ×4',
      'Pre-RMSNorm · NoPE (no rotary position embedding) · Untied 131,072-token output head',
    ],
  },
  embedding: {
    title: 'Token Embedding Matrix',
    badge: '131,072 × 2,688',
    body:
      'Each input token ID looks up one 2,688-dimensional row from the embedding table and injects it directly into the emerald residual stream on the right. Because embeddings are untied from the final LM head, input and output projections maintain independent parameters.',
    specs: [
      'Vocab size: 131,072 · Hidden width: 2,688 · Tie word embeddings: false',
      PARAMETER_CAPTION,
    ],
  },
  mamba: {
    title: 'Mamba-2 Recurrent Layers',
    badge: '23 bays · Fixed-size state',
    body:
      'Each Mamba-2 layer updates a four-slot causal Conv1D buffer and a square cyan-speckled state-space register in place. Scrubbing the token count shifts the register cell pattern without growing memory — all 23 Mamba-2 layers together stay fixed at 47.1 MiB.',
    specs: [
      'SSM state: 64 heads × 64 channels × 128 state values (FP32 storage)',
      'Conv1D state: 6,144 channels × 4 slots (always BF16 even in FP8 Selective)',
      'Output projection: 4,096 → 2,688 after gated RMSNorm',
    ],
    inspectKind: 'M',
  },
  moe: {
    title: 'Mixture of Experts (128 + 1)',
    badge: '23 bays · 6 routed + 1 shared',
    body:
      'A 128-score sigmoid router activates exactly six routed experts per token in parallel (amber-lit tiles, width 1,856, non-gated ReLU²), leaving 122 experts idle (95.3% skipped). Beside the bank sits one always-active parallel shared expert.',
    specs: [
      'Routed experts: 6 active of 128 · width 2,688 → 1,856 → 2,688 (non-gated ReLU²)',
      SHARED_EXPERT_NOTE,
      '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B token cache',
    ],
    inspectKind: 'E',
  },
  gqa: {
    title: 'Grouped-Query Attention',
    badge: '6 bays · 06, 13, 20, 27, 34, 43',
    body:
      'Only six layers use full attention, standing out along the tower with periwinkle Q/K/V projection plates and two paired emerald/cyan KV-cache tracks extending forward. Thirty-two query heads share just two KV heads (16 query heads per KV pair) with NoPE.',
    specs: [
      'Q projection: 32 heads × 128 = 4,096 · K & V projections: 2 heads × 128 = 256 each',
      'Logical KV payload: +6 KiB/token in BF16 · +3 KiB/token in FP8 Selective',
      '52-layer pure GQA reference would cost +52 KiB/token (BF16) or +26 KiB/token (FP8)',
    ],
    inspectKind: '*',
  },
  kv_fp8: {
    title: 'KV Cache & FP8 Selective Mode',
    badge: 'NVIDIA §4.2 · 12 BF16 / 40 FP8',
    body:
      'Toggle FP8 Selective below to see NVIDIA’s sensitivity boundary pop: the 40 quantized layers fade to frosted slate, while the six GQA bays (06, 13, 20, 27, 34, 43) and their six immediately preceding Mamba-2 feeders (05, 12, 19, 26, 33, 42) stay vivid BF16 with warm gold borders.',
    specs: [
      'All 23 Mamba Conv1D strips remain BF16; all 23 SSM registers remain FP32 (47.1 MiB fixed)',
      'All 6 GQA KV caches switch to FP8 even beside BF16 attention bodies (+3 KiB/token)',
      'Crossover vs 52-layer GQA reference: 1,048 tokens (BF16) · 2,096 tokens (FP8)',
    ],
    inspectKind: '*',
  },
  output: {
    title: 'Final RMSNorm & Untied LM Head',
    badge: '2,688 → 131,072 logits',
    body:
      'After Bay 52 (MoE), the emerald residual stream passes through a final amber RMSNorm wafer and an untied periwinkle projection plate that maps the 2,688-channel vector into 131,072 vocabulary logits.',
    specs: [
      'Final RMSNorm (2,688) → Untied LM Head (2,688 × 131,072) → Output Logits',
      PARAMETER_CAPTION,
    ],
  },
};

export interface ControlsElements {
  modeButtons: HTMLButtonElement[];
  tocButtons: HTMLButtonElement[];
  presetButtons: HTMLButtonElement[];
  mambaBayButtons: HTMLButtonElement[];
  mambaViewButtons: HTMLButtonElement[];
  moeBayButtons: HTMLButtonElement[];
  moeViewButtons: HTMLButtonElement[];
  gqaBayButtons: HTMLButtonElement[];
  gqaViewButtons: HTMLButtonElement[];
  miniNodes: SVGGElement[];
  tourToggleBtn: HTMLButtonElement;
  towerSubbar: HTMLDivElement;
  mambaSubbar: HTMLDivElement;
  moeSubbar: HTMLDivElement;
  gqaSubbar: HTMLDivElement;
  floatingBadge: HTMLDivElement;
  orbitHint: HTMLDivElement;
  gqaMathDock: HTMLElement;
  gqaFormulaTensor: HTMLSpanElement;
  gqaFormulaCoord: HTMLSpanElement;
  gqaFormulaHead: HTMLSpanElement;
  gqaFormulaValue: HTMLSpanElement;
  gqaFormulaNote: HTMLDivElement;
  gqaCellClear: HTMLButtonElement;
  gqaFormulaEq: HTMLDivElement;
  gqaFormulaSub: HTMLDivElement;
  gqaStepPrev: HTMLButtonElement;
  gqaStepPlay: HTMLButtonElement;
  gqaStepNext: HTMLButtonElement;
  gqaStepPills: HTMLButtonElement[];
  gqaStepScrub: HTMLInputElement;
  editorialPanel: HTMLDivElement;
  tokenSlider: HTMLInputElement;
  tokenDec: HTMLButtonElement;
  tokenInc: HTMLButtonElement;
  tokenValue: HTMLOutputElement;
  precisionBf16: HTMLButtonElement;
  precisionFp8: HTMLButtonElement;
  memoryReadout: HTMLDivElement;
  memoryReadoutLines: HTMLDivElement;
  memoryChartBtn: HTMLButtonElement;
  bayStepRow: HTMLDivElement;
  bayPrev: HTMLButtonElement;
  bayNext: HTMLButtonElement;
  bayClear: HTMLButtonElement;
  zoomBayBtn: HTMLButtonElement;
  fitBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  shortcutsBtn: HTMLButtonElement;
  shortcutsPopover: HTMLDivElement;
  shortcutsCloseBtn: HTMLButtonElement;
}

export function queryControlsElements(): ControlsElements {
  const get = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing required DOM element #${id}`);
    return el as T;
  };
  return {
    modeButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.mode-btn[data-mode]')),
    tocButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.toc-btn[data-section]')),
    presetButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.pill-btn[data-preset]')),
    mambaBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-mamba-bay]')),
    mambaViewButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-mamba-view]')),
    moeBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-moe-bay]')),
    moeViewButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-moe-view]')),
    gqaBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-bay]')),
    gqaViewButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-view]')),
    miniNodes: Array.from(document.querySelectorAll<SVGGElement>('.mini-node[data-mini-section]')),
    tourToggleBtn: get<HTMLButtonElement>('tour-toggle-btn'),
    towerSubbar: get<HTMLDivElement>('tower-subbar'),
    mambaSubbar: get<HTMLDivElement>('mamba-subbar'),
    moeSubbar: get<HTMLDivElement>('moe-subbar'),
    gqaSubbar: get<HTMLDivElement>('gqa-subbar'),
    floatingBadge: get<HTMLDivElement>('floating-badge'),
    orbitHint: get<HTMLDivElement>('orbit-hint'),
    gqaMathDock: get<HTMLElement>('gqa-math-dock'),
    gqaFormulaTensor: get<HTMLSpanElement>('gqa-formula-tensor'),
    gqaFormulaCoord: get<HTMLSpanElement>('gqa-formula-coord'),
    gqaFormulaHead: get<HTMLSpanElement>('gqa-formula-head'),
    gqaFormulaValue: get<HTMLSpanElement>('gqa-formula-value'),
    gqaFormulaNote: get<HTMLDivElement>('gqa-formula-note'),
    gqaCellClear: get<HTMLButtonElement>('gqa-cell-clear'),
    gqaFormulaEq: get<HTMLDivElement>('gqa-formula-eq'),
    gqaFormulaSub: get<HTMLDivElement>('gqa-formula-sub'),
    gqaStepPrev: get<HTMLButtonElement>('gqa-step-prev'),
    gqaStepPlay: get<HTMLButtonElement>('gqa-step-play'),
    gqaStepNext: get<HTMLButtonElement>('gqa-step-next'),
    gqaStepPills: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-step]')),
    gqaStepScrub: get<HTMLInputElement>('gqa-step-scrub'),
    editorialPanel: get<HTMLDivElement>('editorial-panel'),
    tokenSlider: get<HTMLInputElement>('token-slider'),
    tokenDec: get<HTMLButtonElement>('token-dec'),
    tokenInc: get<HTMLButtonElement>('token-inc'),
    tokenValue: get<HTMLOutputElement>('token-value'),
    precisionBf16: get<HTMLButtonElement>('precision-bf16'),
    precisionFp8: get<HTMLButtonElement>('precision-fp8'),
    memoryReadout: get<HTMLDivElement>('memory-readout'),
    memoryReadoutLines: get<HTMLDivElement>('memory-readout-lines'),
    memoryChartBtn: get<HTMLButtonElement>('memory-chart-btn'),
    bayStepRow: get<HTMLDivElement>('bay-step-row'),
    bayPrev: get<HTMLButtonElement>('bay-prev'),
    bayNext: get<HTMLButtonElement>('bay-next'),
    bayClear: get<HTMLButtonElement>('bay-clear'),
    zoomBayBtn: get<HTMLButtonElement>('zoom-bay-btn'),
    fitBtn: get<HTMLButtonElement>('fit-btn'),
    resetBtn: get<HTMLButtonElement>('reset-btn'),
    shortcutsBtn: get<HTMLButtonElement>('shortcuts-btn'),
    shortcutsPopover: get<HTMLDivElement>('shortcuts-popover'),
    shortcutsCloseBtn: get<HTMLButtonElement>('shortcuts-close'),
  };
}

export class MacroControls {
  private readonly spec: NemotronSpec;
  private readonly els: ControlsElements;
  private readonly cb: ControlsCallbacks;
  private readonly cleanupFns: (() => void)[] = [];

  private lastReadoutKey = '';
  private lastEditorialKey = '';

  constructor(spec: NemotronSpec, els: ControlsElements, cb: ControlsCallbacks) {
    this.spec = spec;
    this.els = els;
    this.cb = cb;
    this.bindEvents();
  }

  private on<K extends keyof HTMLElementEventMap>(
    el: Element,
    type: K,
    handler: (ev: HTMLElementEventMap[K]) => void,
  ): void {
    el.addEventListener(type, handler as EventListener);
    this.cleanupFns.push(() => el.removeEventListener(type, handler as EventListener));
  }

  private bindEvents(): void {
    for (const btn of this.els.modeButtons) {
      this.on(btn, 'click', () => {
        const mode = btn.dataset.mode as ViewMode | undefined;
        if (mode) this.cb.onSelectMode(mode);
      });
    }

    for (const btn of this.els.tocButtons) {
      this.on(btn, 'click', () => {
        const sec = btn.dataset.section as TocSectionId | undefined;
        if (sec) this.cb.onSelectSection(sec);
      });
    }

    for (const btn of this.els.presetButtons) {
      this.on(btn, 'click', () => {
        const preset = btn.dataset.preset as TocSectionId | undefined;
        if (preset) this.cb.onSelectSection(preset);
      });
    }

    for (const btn of this.els.mambaBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.mambaBay ?? '1', 10);
        if (id === 1 || id === 42) this.cb.onSelectMambaBay(id);
      });
    }

    for (const btn of this.els.mambaViewButtons) {
      this.on(btn, 'click', () => {
        const view = btn.dataset.mambaView as GqaViewAngle | undefined;
        if (view === 'three_quarter' || view === 'near_plan') {
          this.cb.onSelectMambaView(view);
        }
      });
    }

    for (const btn of this.els.moeBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.moeBay ?? '2', 10);
        if (id === 2 || id === 52) this.cb.onSelectMoeBay(id);
      });
    }

    for (const btn of this.els.moeViewButtons) {
      this.on(btn, 'click', () => {
        const view = btn.dataset.moeView as GqaViewAngle | undefined;
        if (view === 'three_quarter' || view === 'near_plan') {
          this.cb.onSelectMoeView(view);
        }
      });
    }

    for (const btn of this.els.gqaBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.gqaBay ?? '6', 10);
        if (id === 6 || id === 43) this.cb.onSelectGqaBay(id);
      });
    }

    for (const btn of this.els.gqaViewButtons) {
      this.on(btn, 'click', () => {
        const view = btn.dataset.gqaView as GqaViewAngle | undefined;
        if (view === 'three_quarter' || view === 'near_plan') {
          this.cb.onSelectGqaView(view);
        }
      });
    }

    for (const node of this.els.miniNodes) {
      this.on(node, 'click', () => {
        const sec = node.dataset.miniSection as TocSectionId | undefined;
        if (sec) this.cb.onSelectSection(sec);
      });
    }

    this.on(this.els.tokenSlider, 'input', () => {
      const val = Number.parseInt(this.els.tokenSlider.value, 10);
      if (Number.isFinite(val)) {
        this.cb.onTokensChange(Math.max(MIN_TOKENS, Math.min(MAX_TOKENS, val)));
      }
    });

    this.on(this.els.tokenDec, 'click', () => {
      const cur = Number.parseInt(this.els.tokenSlider.value, 10) || 8;
      this.cb.onTokensChange(Math.max(MIN_TOKENS, cur - 1));
    });

    this.on(this.els.tokenInc, 'click', () => {
      const cur = Number.parseInt(this.els.tokenSlider.value, 10) || 8;
      this.cb.onTokensChange(Math.min(MAX_TOKENS, cur + 1));
    });

    this.on(this.els.precisionBf16, 'click', () => this.cb.onPrecisionChange('bf16'));
    this.on(this.els.precisionFp8, 'click', () => this.cb.onPrecisionChange('fp8_selective'));

    this.on(this.els.fitBtn, 'click', () => this.cb.onFitAll());
    this.on(this.els.zoomBayBtn, 'click', () => this.cb.onZoomSelectedBay());
    this.on(this.els.resetBtn, 'click', () => this.cb.onReset());

    this.on(this.els.bayPrev, 'click', () => this.cb.onStepBay(-1));
    this.on(this.els.bayNext, 'click', () => this.cb.onStepBay(1));
    this.on(this.els.bayClear, 'click', () => this.cb.onClearSelection());

    this.on(this.els.tourToggleBtn, 'click', () => this.cb.onTourToggle());
    this.on(this.els.memoryChartBtn, 'click', () => this.cb.onToggleMemoryChart());
    this.on(this.els.shortcutsBtn, 'click', () => this.cb.onToggleShortcuts());
    this.on(this.els.shortcutsCloseBtn, 'click', () => this.cb.onCloseShortcuts());

    this.on(this.els.gqaStepPrev, 'click', () => this.cb.onGqaStepPrev());
    this.on(this.els.gqaStepPlay, 'click', () => this.cb.onGqaStepTogglePlay());
    this.on(this.els.gqaStepNext, 'click', () => this.cb.onGqaStepNext());
    this.on(this.els.gqaCellClear, 'click', () => this.cb.onGqaClearCell());
    for (const btn of this.els.gqaStepPills) {
      this.on(btn, 'click', () => {
        const step = Number.parseInt(btn.dataset.gqaStep ?? '0', 10);
        if (Number.isFinite(step)) this.cb.onGqaStepSelect(step);
      });
    }
    this.on(this.els.gqaStepScrub, 'input', () => {
      const val = Number.parseInt(this.els.gqaStepScrub.value, 10);
      if (Number.isFinite(val)) {
        this.cb.onGqaStepScrub(Math.max(0, Math.min(100, val)) / 100);
      }
    });
  }

  sync(state: SceneState, activeCutaway: CutawaySceneGeometry | null = null): void {
    // Active View Mode Switcher
    for (const btn of this.els.modeButtons) {
      const isAct = btn.dataset.mode === state.mode;
      btn.classList.toggle('is-active', isAct);
      btn.setAttribute('aria-selected', String(isAct));
    }

    // Contextual sub-bars
    this.els.towerSubbar.hidden = state.mode !== 'tower';
    this.els.mambaSubbar.hidden = state.mode !== 'cutaway_mamba';
    this.els.moeSubbar.hidden = state.mode !== 'cutaway_moe';
    this.els.gqaSubbar.hidden = state.mode !== 'cutaway_gqa';

    // Mamba, MoE & GQA representative bay & camera view toggles
    for (const btn of this.els.mambaBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.mambaBay) === state.mambaBayId);
    }
    for (const btn of this.els.mambaViewButtons) {
      btn.classList.toggle('is-active', btn.dataset.mambaView === state.mambaViewAngle);
    }
    for (const btn of this.els.moeBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.moeBay) === state.moeBayId);
    }
    for (const btn of this.els.moeViewButtons) {
      btn.classList.toggle('is-active', btn.dataset.moeView === state.moeViewAngle);
    }
    for (const btn of this.els.gqaBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.gqaBay) === state.gqaBayId);
    }
    for (const btn of this.els.gqaViewButtons) {
      btn.classList.toggle('is-active', btn.dataset.gqaView === state.gqaViewAngle);
    }

    // Floating corner badge & orbit hint (hidden when shared live Math Dock occupies bottom)
    const usesLiveMathDock =
      state.mode === 'cutaway_gqa' ||
      state.mode === 'cutaway_mamba' ||
      state.mode === 'cutaway_moe';
    this.els.floatingBadge.hidden = usesLiveMathDock;
    this.els.orbitHint.hidden = usesLiveMathDock;
    this.els.gqaMathDock.hidden = !usesLiveMathDock;

    if (state.mode === 'tower') {
      this.els.floatingBadge.textContent = 'Nemotron 3 Nano 30B-A3B · 3.2B active';
    } else if (state.mode === 'cutaway_mamba') {
      this.els.floatingBadge.textContent =
        state.mambaBayId === 42
          ? 'Cutaway: Mamba-2 · Bay 42 Feeder (BF16 §4.2)'
          : 'Cutaway: Mamba-2 · Bay 01 (47.1 MiB fixed state)';
      this.syncMathDock(state, activeCutaway);
    } else if (state.mode === 'cutaway_moe') {
      this.els.floatingBadge.textContent =
        state.moeBayId === 52
          ? 'Cutaway: MoE 128+1 · Bay 52 Final MoE (6 Active + Shared 3,712)'
          : 'Cutaway: MoE 128+1 · Bay 02 (6 Active + Shared 3,712)';
      this.syncMathDock(state, activeCutaway);
    } else {
      this.els.floatingBadge.textContent = `Cutaway: GQA Attention · Bay ${formatBayId(state.gqaBayId)} (32Q / 2KV, NoPE)`;
      this.syncMathDock(state, activeCutaway);
    }

    // Sync active TOC & mini-diagram highlight
    const effectiveSection: TocSectionId =
      state.mode === 'cutaway_mamba'
        ? 'mamba'
        : state.mode === 'cutaway_moe'
          ? 'moe'
          : state.mode === 'cutaway_gqa'
            ? 'gqa'
            : state.section;

    for (const btn of this.els.tocButtons) {
      btn.classList.toggle('is-active', btn.dataset.section === effectiveSection);
    }
    for (const btn of this.els.presetButtons) {
      btn.classList.toggle('is-active', btn.dataset.preset === state.section);
    }
    for (const node of this.els.miniNodes) {
      node.classList.toggle('is-active', node.dataset.miniSection === effectiveSection);
    }

    // Scrubber & precision
    const tokenStr = String(state.tokens);
    if (this.els.tokenSlider.value !== tokenStr) {
      this.els.tokenSlider.value = tokenStr;
    }
    this.els.tokenValue.textContent = `${state.tokens} / ${MAX_TOKENS}`;
    this.els.tokenDec.disabled = state.tokens <= MIN_TOKENS;
    this.els.tokenInc.disabled = state.tokens >= MAX_TOKENS;

    const isBf16 = state.precision === 'bf16';
    this.els.precisionBf16.classList.toggle('is-active', isBf16);
    this.els.precisionBf16.setAttribute('aria-checked', String(isBf16));
    this.els.precisionFp8.classList.toggle('is-active', !isBf16);
    this.els.precisionFp8.setAttribute('aria-checked', String(!isBf16));

    // Zoom bay button & bay stepping row (only shown in 52-Layer Tower mode)
    const isTower = state.mode === 'tower';
    const hasPinnedBay = isTower && state.selectedBay !== null;
    const hasPinnedPart = !isTower && state.selectedPart !== null;

    this.els.zoomBayBtn.hidden = !isTower;
    this.els.zoomBayBtn.disabled = !hasPinnedBay;
    this.els.zoomBayBtn.textContent = hasPinnedBay
      ? `Zoom Bay ${formatBayId(state.selectedBay!)}`
      : 'Zoom Bay';

    this.els.bayPrev.hidden = !isTower;
    this.els.bayNext.hidden = !isTower;
    this.els.bayClear.hidden = !(hasPinnedBay || hasPinnedPart);
    this.els.bayClear.textContent = hasPinnedPart ? 'Clear slab' : 'Clear';
    this.els.bayStepRow.hidden = !isTower && !hasPinnedPart;

    // Guided Tour header toggle button
    const activeChapter = getWalkthroughChapter(state.tourChapter);
    this.els.tourToggleBtn.classList.toggle('is-active', state.tourActive);
    this.els.tourToggleBtn.textContent = state.tourActive
      ? `Tour ${activeChapter.number}/${WALKTHROUGH_COUNT}`
      : 'Guided Tour';

    // Memory Scaling chart trigger highlight (Chapter 7 + active open state)
    const highlightChartBtn =
      state.tourActive && Boolean(activeChapter.highlightMemoryChart) && !state.memoryChartOpen;
    this.els.memoryChartBtn.classList.toggle('is-open', state.memoryChartOpen);
    this.els.memoryChartBtn.classList.toggle('is-highlighted', highlightChartBtn);
    this.els.memoryChartBtn.setAttribute('aria-expanded', String(state.memoryChartOpen));

    // Shortcuts popover
    this.els.shortcutsBtn.classList.toggle('is-active', state.shortcutsOpen);
    this.els.shortcutsBtn.setAttribute('aria-expanded', String(state.shortcutsOpen));
    this.els.shortcutsPopover.hidden = !state.shortcutsOpen;

    // Memory readout (with Part A #4 static weights vs dynamic sequence cache clarification!)
    const readoutKey = `${state.tokens}:${state.precision}`;
    if (readoutKey !== this.lastReadoutKey) {
      this.lastReadoutKey = readoutKey;
      const mem = memoryFor(this.spec, state.tokens, state.precision);
      const fmt = formatMemoryReadout(mem);

      this.els.memoryReadoutLines.innerHTML = '';
      const rows = [
        { text: fmt.mambaText, cls: 'readout-line readout-fixed' },
        { text: fmt.kvText, cls: 'readout-line readout-kv' },
        { text: fmt.referenceText, cls: 'readout-line readout-ref' },
        { text: WEIGHTS_SUMMARY_LINE, cls: 'readout-line readout-weights' },
      ];
      for (const r of rows) {
        const div = document.createElement('div');
        div.className = r.cls;
        div.textContent = r.text;
        this.els.memoryReadoutLines.appendChild(div);
      }
    }

    // Middle editorial panel
    const activeBayId = isTower ? state.hoveredBay ?? state.selectedBay : null;
    const activePart = !isTower ? state.hoveredPart ?? state.selectedPart : null;
    const editorialKey = `${state.tourActive ? `tour:${state.tourChapter}` : 'free'}:${state.mode}:${state.mambaBayId}:${state.mambaActiveHead}:${state.moeBayId}:${state.moeFocusToken ?? 'last'}:${state.moeActiveExpert ?? 'top0'}:${state.gqaBayId}:${state.gqaActiveGroup0Head}:${state.gqaActiveGroup1Head}:${state.section}:${activeBayId ?? 'none'}:${state.selectedBay ?? 'none'}:${activePart ?? 'none'}:${state.selectedPart ?? 'none'}:${state.precision}:${state.tokens}`;

    if (editorialKey !== this.lastEditorialKey) {
      this.lastEditorialKey = editorialKey;
      if (state.tourActive) {
        this.renderTourEditorial(state);
      } else if (isTower) {
        this.renderTowerEditorial(
          state.section,
          activeBayId,
          state.selectedBay !== null,
          state.precision,
          state.tokens,
        );
      } else {
        this.renderCutawayEditorial(state, activePart, activeCutaway);
      }
    }
  }

  private syncMathDock(state: SceneState, cutaway: CutawaySceneGeometry | null): void {
    const isMamba = state.mode === 'cutaway_mamba';
    const isMoe = state.mode === 'cutaway_moe';
    const formula = cutaway?.activeFormula ?? null;
    const isPinnedCell = isMamba
      ? state.mambaSelectedCell !== null
      : isMoe
        ? state.moeSelectedCell !== null
        : state.gqaSelectedCell !== null;
    const steps = isMamba
      ? MAMBA_STEPPER_STEPS
      : isMoe
        ? MOE_STEPPER_STEPS
        : GQA_STEPPER_STEPS;

    // Update honest miniature simulator disclaimer note
    this.els.gqaFormulaNote.textContent = isMamba
      ? 'Deterministic miniature Mamba-2 sim (d=8, 16H/8G, dₕ=4, dₛ=4; Float32 storage, rounded outputs) · Educational token labels · BF16/FP8 = production metadata'
      : isMoe
        ? 'Deterministic miniature MoE sim (d=8, 128E top-6, d_exp=4, d_shared=8, ×2.5; Float32 storage, rounded outputs) · Educational token labels · BF16/FP8/FP32 = production metadata'
        : 'Deterministic miniature GQA sim (d=8, 32Q/2KV, dₕ=4; Float32 storage, rounded outputs) · Educational token labels · BF16/FP8 = production metadata';

    if (formula) {
      this.els.gqaFormulaTensor.textContent = formula.tensorLabel;
      this.els.gqaFormulaCoord.textContent = formula.coordLabel;
      if (formula.headBadge) {
        this.els.gqaFormulaHead.textContent = formula.headBadge;
        this.els.gqaFormulaHead.hidden = false;
      } else {
        this.els.gqaFormulaHead.textContent = '';
        this.els.gqaFormulaHead.hidden = true;
      }
      this.els.gqaFormulaValue.textContent = formula.isMasked
        ? formula.valueText
        : `= ${formula.valueText}`;
      this.els.gqaFormulaValue.classList.toggle('is-masked', formula.isMasked);
      this.els.gqaFormulaValue.hidden = false;
      this.els.gqaCellClear.hidden = !isPinnedCell;
      this.els.gqaFormulaEq.textContent = formula.equation;
      this.els.gqaFormulaSub.textContent = formula.subEquation;
    } else {
      // Idle state: keep head badge and numeric value badge completely empty & hidden (Part 0 #2)
      this.els.gqaFormulaHead.textContent = '';
      this.els.gqaFormulaHead.hidden = true;
      this.els.gqaFormulaValue.textContent = '';
      this.els.gqaFormulaValue.classList.remove('is-masked');
      this.els.gqaFormulaValue.hidden = true;
      this.els.gqaCellClear.hidden = true;

      if (isMamba) {
        const activeH = cutaway?.mambaStats?.activeHead ?? state.mambaActiveHead;
        const activeG = mambaGroupForHead(activeH);
        this.els.gqaFormulaTensor.textContent = `Miniature Mamba-2 · Bay ${formatBayId(state.mambaBayId)}`;
        this.els.gqaFormulaCoord.textContent = `Active Head: ${formatMambaHeadId(activeH)} (G${activeG})`;
        this.els.gqaFormulaEq.textContent =
          'Hover or click any matrix cell for rounded numerical explanations, or press Play to step through Mamba-2 recurrence.';
        const tokPreview = cutaway?.mambaSim?.tokenLabels
          ? cutaway.mambaSim.tokenLabels.map((tok, i) => `${i}:"${tok}"`).join(' ')
          : '';
        this.els.gqaFormulaSub.textContent = tokPreview
          ? `Educational token labels (T=${state.tokens}): ${tokPreview}`
          : '';
      } else if (isMoe) {
        const fTok = cutaway?.moeStats?.focusToken ?? Math.max(0, state.tokens - 1);
        const fExp = cutaway?.moeStats?.focusExpert ?? cutaway?.moeStats?.activeExpertIds[0] ?? 0;
        const tokLabel = cutaway?.moeSim?.tokenLabels[fTok] ?? `t${fTok}`;
        this.els.gqaFormulaTensor.textContent = `Miniature MoE 128+1 · Bay ${formatBayId(state.moeBayId)}`;
        this.els.gqaFormulaCoord.textContent = `Focus: t${fTok} "${tokLabel}" · ${formatExpertId(fExp)}`;
        this.els.gqaFormulaEq.textContent =
          'Hover or click any router tile, Top-6 slot, routed ReLU² cell, or 2× shared cell, or press Play to step through parallel MoE.';
        const topList = cutaway?.moeStats?.activeExpertIds?.map(formatExpertId).join(', ') ?? '';
        this.els.gqaFormulaSub.textContent = topList
          ? `Top-6 active for t${fTok} "${tokLabel}": ${topList} (122 idle = 0 contrib) · 0 B token cache`
          : '';
      } else {
        const g0 = formatQueryHeadId(cutaway?.gqaStats?.activeGroup0Head ?? state.gqaActiveGroup0Head);
        const g1 = formatQueryHeadId(cutaway?.gqaStats?.activeGroup1Head ?? state.gqaActiveGroup1Head);
        this.els.gqaFormulaTensor.textContent = `Miniature GQA · Bay ${formatBayId(state.gqaBayId)}`;
        this.els.gqaFormulaCoord.textContent = `Displayed: ${g0}→KV₀ · ${g1}→KV₁`;
        this.els.gqaFormulaEq.textContent =
          'Hover or click any matrix cell for rounded numerical explanations, or press Play to step through causal attention.';
        const tokPreview = cutaway?.gqaSim?.tokenLabels
          ? cutaway.gqaSim.tokenLabels.map((tok, i) => `${i}:"${tok}"`).join(' ')
          : '';
        this.els.gqaFormulaSub.textContent = tokPreview
          ? `Educational token labels (T=${state.tokens}): ${tokPreview}`
          : '';
      }
    }

    const stepperActive = isMamba
      ? state.mambaStepperActive
      : isMoe
        ? state.moeStepperActive
        : state.gqaStepperActive;
    const stepperPlaying = isMamba
      ? state.mambaStepperPlaying
      : isMoe
        ? state.moeStepperPlaying
        : state.gqaStepperPlaying;
    const stepperStep = isMamba
      ? state.mambaStepperStep
      : isMoe
        ? state.moeStepperStep
        : state.gqaStepperStep;
    const stepperSub = isMamba
      ? state.mambaStepperSubProgress
      : isMoe
        ? state.moeStepperSubProgress
        : state.gqaStepperSubProgress;

    const playing = stepperActive && stepperPlaying;
    this.els.gqaStepPlay.textContent = playing ? '❚❚ Pause' : '▶ Play';
    this.els.gqaStepPlay.classList.toggle('is-playing', playing);
    this.els.gqaStepPlay.setAttribute('aria-pressed', String(playing));

    for (let i = 0; i < this.els.gqaStepPills.length; i++) {
      const btn = this.els.gqaStepPills[i];
      if (steps[i]) {
        btn.textContent = steps[i].shortLabel;
        btn.title = steps[i].summary;
      }
      const isAct = stepperActive && i === stepperStep;
      btn.classList.toggle('is-active', isAct);
      btn.setAttribute('aria-selected', String(isAct));
    }

    const pct = String(Math.round(stepperSub * 100));
    if (this.els.gqaStepScrub.value !== pct) {
      this.els.gqaStepScrub.value = pct;
    }
  }

  private renderTourEditorial(state: SceneState): void {
    const ch = getWalkthroughChapter(state.tourChapter);
    const panel = this.els.editorialPanel;
    panel.classList.add('is-tour-mode');
    panel.innerHTML = '';

    // Top bbycroft/llm-viz style < Chapter N / 7 > bar + subtle progress track
    const navBar = document.createElement('div');
    navBar.className = 'tour-nav-bar';

    const prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.className = 'tour-arrow-btn';
    prevBtn.setAttribute('aria-label', 'Previous chapter');
    prevBtn.textContent = '‹';
    prevBtn.disabled = ch.index <= 0;
    prevBtn.addEventListener('click', () => this.cb.onTourPrev());

    const counter = document.createElement('span');
    counter.className = 'tour-counter';
    counter.setAttribute('data-testid', 'tour-chapter-counter');
    counter.textContent = `Chapter ${ch.number} / ${WALKTHROUGH_COUNT}`;

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'tour-arrow-btn';
    nextBtn.setAttribute('aria-label', 'Next chapter');
    nextBtn.textContent = '›';
    nextBtn.disabled = ch.index >= WALKTHROUGH_COUNT - 1;
    nextBtn.addEventListener('click', () => this.cb.onTourNext());

    navBar.append(prevBtn, counter, nextBtn);

    const progressTrack = document.createElement('div');
    progressTrack.className = 'tour-progress-track';
    progressTrack.setAttribute('aria-hidden', 'true');
    const progressFill = document.createElement('div');
    progressFill.className = 'tour-progress-fill';
    progressFill.style.width = `${(ch.number / WALKTHROUGH_COUNT) * 100}%`;
    progressTrack.appendChild(progressFill);

    const scrollBody = document.createElement('div');
    scrollBody.className = 'tour-scroll-body';

    const header = document.createElement('div');
    header.className = 'ed-header tour-ed-header';

    const title = document.createElement('h2');
    title.className = 'ed-title';
    title.textContent = `${ch.number}. ${ch.title}`;

    const badge = document.createElement('span');
    badge.className = 'ed-badge';
    if (ch.precision === 'fp8_selective') badge.classList.add('ed-badge-gold');
    badge.textContent = ch.badge;

    header.append(title, badge);
    scrollBody.appendChild(header);

    for (const para of ch.paragraphs) {
      const p = document.createElement('p');
      p.className = 'ed-body';
      p.textContent = para;
      scrollBody.appendChild(p);
    }

    const list = document.createElement('ul');
    list.className = 'ed-specs';
    for (const item of ch.takeaways) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }
    scrollBody.appendChild(list);

    // Bottom Continue -> / Exit Tour bar (always pinned visible!)
    const actions = document.createElement('div');
    actions.className = 'ed-actions tour-actions';

    const isLast = ch.index >= WALKTHROUGH_COUNT - 1;
    if (!isLast) {
      const contBtn = document.createElement('button');
      contBtn.type = 'button';
      contBtn.id = 'tour-continue-btn';
      contBtn.className = 'ed-cta-btn tour-continue-btn';
      contBtn.textContent = 'Continue →';
      contBtn.addEventListener('click', () => this.cb.onTourNext());
      actions.appendChild(contBtn);
    } else {
      const chartBtn = document.createElement('button');
      chartBtn.type = 'button';
      chartBtn.id = 'tour-chart-btn';
      chartBtn.className = 'ed-cta-btn tour-continue-btn';
      chartBtn.textContent = 'Memory Scaling 1 → 1M ↗';
      chartBtn.addEventListener('click', () => this.cb.onToggleMemoryChart());
      actions.appendChild(chartBtn);

      const restartBtn = document.createElement('button');
      restartBtn.type = 'button';
      restartBtn.className = 'ed-cta-btn ed-cta-secondary';
      restartBtn.textContent = 'Restart';
      restartBtn.addEventListener('click', () => this.cb.onTourRestart());
      actions.appendChild(restartBtn);
    }

    const exitBtn = document.createElement('button');
    exitBtn.type = 'button';
    exitBtn.id = 'tour-exit-btn';
    exitBtn.className = 'ed-cta-btn ed-cta-secondary';
    exitBtn.textContent = isLast ? 'Free Explore' : 'Exit Tour';
    exitBtn.addEventListener('click', () => this.cb.onTourExit());
    actions.appendChild(exitBtn);

    panel.append(navBar, progressTrack, scrollBody, actions);
  }

  private renderTowerEditorial(
    section: TocSectionId,
    bayId: number | null,
    isPinned: boolean,
    precision: PrecisionMode,
    tokens: number,
  ): void {
    const panel = this.els.editorialPanel;
    panel.classList.remove('is-tour-mode');
    panel.innerHTML = '';

    if (bayId !== null) {
      const bay = this.spec.bays[bayId - 1];
      if (bay) {
        const bracket = this.spec.brackets[bay.bracket];
        const computePrec = bodyPrecision(bay, precision);
        const computeLine =
          precision === 'bf16'
            ? 'Compute: BF16 weights & activations'
            : computePrec === 'bf16'
              ? 'Compute: BF16 preserved (sensitive bay in FP8 Selective)'
              : 'Compute: FP8 weights & activations (quantized)';

        let storageLine: string;
        let narrative: string;
        const extraSpecs: string[] = [];

        if (bay.kind === 'M') {
          const ssm = ssmStateDtype(bay)!.toUpperCase();
          const conv = conv1dPrecision(bay, precision)!.toUpperCase();
          storageLine = `Memory: SSM ${ssm} (64×64×128) · Conv1D ${conv} (6,144×4) — fixed`;
          narrative = bay.feeder
            ? `Bay ${formatBayId(bay.id)} is a Mamba-2 feeder immediately preceding GQA attention bay ${formatBayId(bay.id + 1)}. NVIDIA §4.2 retains BF16 compute here to protect attention input fidelity, while its cyan SSM register updates in place.`
            : `Bay ${formatBayId(bay.id)} runs a linear-time Mamba-2 state-space update. Its four-slot Conv1D strip stays BF16 in all modes, and its cyan-speckled FP32 state matrix updates in place without growing with sequence length.`;
        } else if (bay.kind === '*') {
          const kvPrec = kvCachePrecision(bay, precision)!;
          const perLayerTotalKiB = ((kvPrec === 'fp8' ? 512 : 1024) * tokens) / 1024;
          const perLayerSlope = kvPrec === 'fp8' ? '+0.5 KiB/token' : '+1 KiB/token';
          storageLine = `KV Cache: ${kvPrec.toUpperCase()} (${perLayerTotalKiB} KiB at T=${tokens}, ${perLayerSlope})`;
          narrative = `Bay ${formatBayId(bay.id)} is one of six Grouped-Query Attention layers (32Q / 2KV, NoPE). In FP8 Selective mode, the attention body stays BF16 while its KV cache switches to FP8.`;
        } else {
          storageLine = 'Memory: 0 B token history (stateless MoE layer)';
          narrative = `Bay ${formatBayId(bay.id)} routes each token through a 128-score sigmoid gate to activate six routed experts (width 1,856, non-gated ReLU²) in parallel alongside one always-active shared expert.`;
          extraSpecs.push(SHARED_EXPERT_NOTE);
        }

        const header = document.createElement('div');
        header.className = 'ed-header';

        const title = document.createElement('h2');
        title.className = 'ed-title';
        title.textContent = `Bay ${formatBayId(bay.id)} / 52 · ${bay.kind} ${bayKindLabel(bay.kind)}`;

        const badge = document.createElement('span');
        badge.className = 'ed-badge ed-badge-active';
        badge.textContent = isPinned ? 'Pinned' : 'Hover';

        header.append(title, badge);

        const sub = document.createElement('div');
        sub.className = 'ed-sub';
        sub.textContent = formatBracketCaption(bracket);

        const p = document.createElement('p');
        p.className = 'ed-body';
        p.textContent = narrative;

        const list = document.createElement('ul');
        list.className = 'ed-specs';
        for (const item of [computeLine, storageLine, ...extraSpecs, PARAMETER_CAPTION]) {
          const li = document.createElement('li');
          li.textContent = item;
          list.appendChild(li);
        }

        const actions = document.createElement('div');
        actions.className = 'ed-actions';
        const inspectBtn = document.createElement('button');
        inspectBtn.type = 'button';
        inspectBtn.className = 'ed-cta-btn';
        inspectBtn.dataset.testid = 'inspect-internals-btn';
        inspectBtn.textContent = 'Inspect Internals →';
        inspectBtn.addEventListener('click', () => {
          this.cb.onInspectInternals(bay.kind, bay.id);
        });
        actions.appendChild(inspectBtn);

        panel.append(header, sub, p, list, actions);
        return;
      }
    }

    const copy = SECTION_COPY[section];
    const header = document.createElement('div');
    header.className = 'ed-header';

    const title = document.createElement('h2');
    title.className = 'ed-title';
    title.textContent = copy.title;

    const badge = document.createElement('span');
    badge.className = 'ed-badge';
    badge.textContent = copy.badge;

    header.append(title, badge);

    const p = document.createElement('p');
    p.className = 'ed-body';
    p.textContent = copy.body;

    const list = document.createElement('ul');
    list.className = 'ed-specs';
    for (const item of copy.specs) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }

    panel.append(header, p, list);

    if (copy.inspectKind) {
      const actions = document.createElement('div');
      actions.className = 'ed-actions';
      const inspectBtn = document.createElement('button');
      inspectBtn.type = 'button';
      inspectBtn.className = 'ed-cta-btn';
      inspectBtn.textContent = 'Inspect Internals →';
      inspectBtn.addEventListener('click', () => {
        this.cb.onInspectInternals(copy.inspectKind!);
      });
      actions.appendChild(inspectBtn);
      panel.appendChild(actions);
    }
  }

  private renderCutawayEditorial(
    state: SceneState,
    activePart: string | null,
    cutaway: CutawaySceneGeometry | null,
  ): void {
    const panel = this.els.editorialPanel;
    panel.classList.remove('is-tour-mode');
    panel.innerHTML = '';

    // 1. If user clicked/hovered a specific tensor slab inside the cutaway, show its details!
    const partInfo: CutawayPartInfo | undefined =
      activePart && cutaway ? cutaway.partCatalog[activePart] : undefined;

    if (partInfo) {
      const header = document.createElement('div');
      header.className = 'ed-header';

      const title = document.createElement('h2');
      title.className = 'ed-title';
      title.textContent = partInfo.title;

      const badge = document.createElement('span');
      badge.className = 'ed-badge ed-badge-active';
      badge.textContent = partInfo.badge;
      header.append(title, badge);

      const p = document.createElement('p');
      p.className = 'ed-body';
      p.textContent = partInfo.summary;

      const list = document.createElement('ul');
      list.className = 'ed-specs';
      for (const item of partInfo.specs) {
        const li = document.createElement('li');
        li.textContent = item;
        list.appendChild(li);
      }

      const actions = document.createElement('div');
      actions.className = 'ed-actions';
      if (state.selectedPart !== null) {
        const clearBtn = document.createElement('button');
        clearBtn.type = 'button';
        clearBtn.className = 'ed-cta-btn ed-cta-secondary';
        clearBtn.textContent = 'All Layer Internals';
        clearBtn.addEventListener('click', () => this.cb.onClearSelection());
        actions.appendChild(clearBtn);
      }
      const backBtn = document.createElement('button');
      backBtn.type = 'button';
      backBtn.className = 'ed-cta-btn ed-cta-secondary';
      backBtn.textContent = '← 52-Layer Tower';
      backBtn.addEventListener('click', () => this.cb.onSelectMode('tower'));
      actions.appendChild(backBtn);

      panel.append(header, p, list);
      if (state.mode === 'cutaway_gqa') {
        panel.appendChild(this.buildGqaHeadSelector(state.selectedPart, state.hoveredPart));
      } else if (state.mode === 'cutaway_mamba') {
        panel.appendChild(
          this.buildMambaHeadSelector(state.mambaActiveHead, state.selectedPart, state.hoveredPart),
        );
      } else if (state.mode === 'cutaway_moe') {
        panel.appendChild(this.buildMoeInspectorControls(state, cutaway));
      }
      panel.appendChild(actions);
      return;
    }

    // 2. Otherwise show the rich Cutaway Overview card
    const isFp8 = state.precision === 'fp8_selective';
    const header = document.createElement('div');
    header.className = 'ed-header';

    const title = document.createElement('h2');
    title.className = 'ed-title';

    const badge = document.createElement('span');
    badge.className = 'ed-badge';

    let bodyText = '';
    let specs: string[] = [];

    if (state.mode === 'cutaway_mamba') {
      const isFeeder = state.mambaBayId === 42;
      title.textContent = isFeeder
        ? 'Mamba-2 Cutaway · Bay 42 (Feeder)'
        : 'Mamba-2 Cutaway · Bay 01';
      badge.textContent = isFeeder && isFp8 ? 'BF16 Preserved §4.2' : '47.1 MiB Fixed';
      if (isFeeder && isFp8) badge.classList.add('ed-badge-gold');

      bodyText = isFeeder
        ? 'Feeder preceding Bay 43 GQA — preserved in BF16 under NVIDIA §4.2. Runs a deterministic miniature Mamba-2 simulator (d=8, 16H/8G, dₕ=4, dₛ=4; Float32 tensor storage, rounded outputs) where sequence tensors grow with T while Hₜ₋₁ ⇄ Hₜ and the 64-head bank stay fixed.'
        : 'Runs a deterministic miniature Mamba-2 simulator (d=8, 16H/8G, dₕ=4, dₛ=4; Float32 tensor storage, rounded outputs). Sequence tensors (xBC_pre, z, x, B, C, Δt) grow with T while the 4-tap Conv1D buffer and Hₜ₋₁ ⇄ Hₜ SSM state remain strictly constant.';

      specs = [
        'Persistent storage (23 bays): 46.0 MiB SSM (FP32) + ~1.08 MiB Conv1D (BF16) ≈ 47.1 MiB fixed · H_history is browser-only playback',
        isFeeder
          ? 'Bay 42 Feeder metadata (§4.2): W_in, W_out, Conv1D preserved in BF16 (gold rim); live sim runs Float32'
          : `Bay 01 metadata (§4.2): W_in & W_out ${isFp8 ? 'FP8' : 'BF16'} · Conv1D ALWAYS BF16 · SSM FP32 (live sim runs Float32)`,
        'Gate FIRST (y·SiLU(z)) → 8-Group RMSNorm (ε=1e−5, norm_before_gate=False) · Select H00–H15 below.',
      ];
    } else if (state.mode === 'cutaway_moe') {
      const fTok = cutaway?.moeStats?.focusToken ?? Math.max(0, state.tokens - 1);
      const tokLabel = cutaway?.moeSim?.tokenLabels[fTok] ?? `t${fTok}`;
      title.textContent =
        state.moeBayId === 52
          ? 'MoE 128+1 Cutaway · Bay 52 (Final)'
          : 'MoE 128+1 Cutaway · Bay 02';
      badge.textContent = isFp8 ? 'FP8 Experts · FP32 Router' : `6 / 128 Active · t${fTok}`;
      if (isFp8) badge.classList.add('ed-badge-gold');

      bodyText =
        `Runs a deterministic miniature MoE simulator (d=8, 128 routed top-6, d_exp=4, d_shared=8 = 2×, ×2.5; Float32 storage, rounded outputs). For token t${fTok} "${tokLabel}", RMSNorm forks in parallel into the sparse 128-expert bank (non-gated ReLU², no gate_proj) and the always-on 2× shared expert.`;

      specs = [
        'Router: sigmoid(W_router·U) + e_score_correction_bias selects Top-6; unbiased σ gathers weights → norm × 2.5 (Router stays FP32 in FP8 Selective)',
        SHARED_EXPERT_NOTE + ' (added unscaled at 1.0× alongside Top-6 routed sum)',
        '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B dynamic token cache (strictly per-token independent)',
      ];
    } else {
      const perBayKiB = ((isFp8 ? 512 : 1024) * state.tokens) / 1024;
      title.textContent = `GQA Cutaway · Bay ${formatBayId(state.gqaBayId)}`;
      badge.textContent = isFp8 ? 'BF16 Compute · FP8 KV' : '32Q / 2KV · NoPE';
      if (isFp8) badge.classList.add('ed-badge-gold');

      bodyText =
        'Thirty-two query heads (Group 0: Q00–Q15 → KV₀; Group 1: Q16–Q31 → KV₁) drive a deterministic miniature simulator with Float32 tensor storage and rounded outputs (d=8, dₕ=4, NoPE) using educational token labels. Hover or click any cell for rounded numerical explanations.';

      specs = [
        `Shared KV cache: 2 KV heads × ${state.tokens} tokens = ${perBayKiB} KiB (${isFp8 ? '+0.5 KiB/tok FP8 storage' : '+1 KiB/tok BF16 storage'})`,
        isFp8
          ? 'Production precision metadata (§4.2): Q/K/V/O compute BF16; KV storage FP8 (miniature sim uses Float32 storage)'
          : 'Production precision metadata: BF16 weights/KV (toggle FP8 Selective for §4.2; miniature sim uses Float32 storage)',
        'Click any Q00–Q31 head below to switch its group’s live Scores & Softmax textures.',
      ];
    }

    header.append(title, badge);

    const p = document.createElement('p');
    p.className = 'ed-body';
    p.textContent = bodyText;

    const list = document.createElement('ul');
    list.className = 'ed-specs';
    for (const item of specs) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }

    const actions = document.createElement('div');
    actions.className = 'ed-actions';
    const backBtn = document.createElement('button');
    backBtn.type = 'button';
    backBtn.className = 'ed-cta-btn ed-cta-secondary';
    backBtn.textContent = '← Back to 52-Layer Tower';
    backBtn.addEventListener('click', () => this.cb.onSelectMode('tower'));
    actions.appendChild(backBtn);

    if (state.mode === 'cutaway_moe') {
      panel.append(header, p, this.buildMoeInspectorControls(state, cutaway), list);
    } else {
      panel.append(header, p, list);
      if (state.mode === 'cutaway_gqa') {
        panel.appendChild(this.buildGqaHeadSelector(state.selectedPart, state.hoveredPart));
      } else if (state.mode === 'cutaway_mamba') {
        panel.appendChild(
          this.buildMambaHeadSelector(state.mambaActiveHead, state.selectedPart, state.hoveredPart),
        );
      }
    }
    panel.appendChild(actions);
  }

  /** Compact Focused Token (t0..t_{T-1}) & Top-6 Winning Expert selector inside MoE inspector. */
  private buildMoeInspectorControls(
    state: SceneState,
    cutaway: CutawaySceneGeometry | null,
  ): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector';
    wrap.setAttribute('data-testid', 'moe-inspector-controls');

    const sim = cutaway?.moeSim ?? null;
    const focusTok = cutaway?.moeStats?.focusToken ?? Math.max(0, state.tokens - 1);
    const focusExp = cutaway?.moeStats?.focusExpert ?? cutaway?.moeStats?.activeExpertIds[0] ?? 0;

    // 1. Focused Token Row
    const tokTop = document.createElement('div');
    tokTop.className = 'qhead-top-row';
    const tokHeading = document.createElement('span');
    tokHeading.className = 'qhead-heading';
    const tokLabel = sim?.tokenLabels[focusTok] ?? `t${focusTok}`;
    tokHeading.textContent = `Focused Token: t${focusTok} "${tokLabel}" (1..${state.tokens})`;
    tokTop.appendChild(tokHeading);
    wrap.appendChild(tokTop);

    const tokGrid = document.createElement('div');
    tokGrid.className = 'moe-token-grid';
    for (let t = 0; t < state.tokens; t++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'moe-tok-btn';
      btn.dataset.moeToken = String(t);
      const lbl = sim?.tokenLabels[t] ?? '';
      btn.textContent = `t${t} ${lbl}`;
      btn.title = `Inspect routing for token t${t} "${lbl}"`;
      const isSel = t === focusTok;
      btn.classList.toggle('is-selected', isSel);
      btn.setAttribute('aria-pressed', String(isSel));
      btn.addEventListener('click', () => this.cb.onSelectMoeFocusToken(t));
      tokGrid.appendChild(btn);
    }
    wrap.appendChild(tokGrid);

    // 2. Top-6 Active Experts Row (clickable to sync detail chain)
    if (sim) {
      const expTop = document.createElement('div');
      expTop.className = 'qhead-group-label';
      expTop.textContent = `Top-6 Active Experts for t${focusTok} (×2.5 scaled, click to inspect):`;
      wrap.appendChild(expTop);

      const expGrid = document.createElement('div');
      expGrid.className = 'moe-topk-grid';
      for (let j = 0; j < sim.topK; j++) {
        const eid = sim.topk_indices[focusTok * sim.topK + j];
        const w = sim.topk_weights[focusTok * sim.topK + j];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'moe-exp-btn';
        btn.dataset.moeExpert = String(eid);
        const isSel = eid === focusExp;
        btn.classList.toggle('is-selected', isSel);
        btn.setAttribute('aria-pressed', String(isSel));
        btn.innerHTML = `<span>${formatExpertId(eid)}</span><span class="moe-exp-w">${w.toFixed(2)}</span>`;
        btn.addEventListener('click', () => this.cb.onSelectMoeExpert(eid));
        expGrid.appendChild(btn);
      }
      wrap.appendChild(expGrid);
    }

    return wrap;
  }

  /** Interactive Q00..Q31 head selector grouped into 16:1 KV banks inside GQA inspector. */
  private buildGqaHeadSelector(selectedPart: string | null, hoveredPart: string | null): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector';
    wrap.setAttribute('data-testid', 'gqa-head-selector');

    const activeHead = parseQueryHeadPartId(hoveredPart ?? selectedPart);
    const pinnedHead = parseQueryHeadPartId(selectedPart);

    const topRow = document.createElement('div');
    topRow.className = 'qhead-top-row';
    const label = document.createElement('span');
    label.className = 'qhead-heading';
    label.textContent =
      activeHead !== null
        ? `Query Head ${formatQueryHeadId(activeHead)} → KV${activeHead < 16 ? 0 : 1}`
        : 'Select Query Head (Q00–Q31)';
    topRow.appendChild(label);

    if (pinnedHead !== null) {
      const resetHead = document.createElement('button');
      resetHead.type = 'button';
      resetHead.className = 'qhead-clear-btn';
      resetHead.textContent = 'All 32';
      resetHead.addEventListener('click', () => this.cb.onSelectPart(null));
      topRow.appendChild(resetHead);
    }
    wrap.appendChild(topRow);

    const groups: readonly { title: string; start: number; end: number }[] = [
      { title: 'Group 0 → KV₀ (Q00–Q15)', start: 0, end: 15 },
      { title: 'Group 1 → KV₁ (Q16–Q31)', start: 16, end: 31 },
    ];

    for (const grp of groups) {
      const grpEl = document.createElement('div');
      grpEl.className = 'qhead-group';

      const grpTitle = document.createElement('div');
      grpTitle.className = 'qhead-group-label';
      grpTitle.textContent = grp.title;
      grpEl.appendChild(grpTitle);

      const grid = document.createElement('div');
      grid.className = 'qhead-grid';
      for (let i = grp.start; i <= grp.end; i++) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qhead-btn';
        btn.dataset.qhead = String(i);
        btn.textContent = formatQueryHeadId(i);
        const isSel = pinnedHead === i;
        const isActive = activeHead === i;
        btn.classList.toggle('is-selected', isSel);
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', String(isSel));
        btn.addEventListener('click', () => {
          this.cb.onSelectPart(isSel ? null : `q-head-${i}`);
        });
        grid.appendChild(btn);
      }
      grpEl.appendChild(grid);
      wrap.appendChild(grpEl);
    }

    return wrap;
  }

  /** Interactive H00..H15 head selector grouped into 8 B/C groups (G0..G7, 2 mini heads/group). */
  private buildMambaHeadSelector(
    activeMambaHead: number,
    selectedPart: string | null,
    hoveredPart: string | null,
  ): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector';
    wrap.setAttribute('data-testid', 'mamba-head-selector');

    const partHead = parseMambaHeadPartId(hoveredPart ?? selectedPart);
    const activeHead = partHead ?? activeMambaHead;
    const pinnedHead = parseMambaHeadPartId(selectedPart);

    const topRow = document.createElement('div');
    topRow.className = 'qhead-top-row';
    const label = document.createElement('span');
    label.className = 'qhead-heading';
    label.textContent = `SSM Head ${formatMambaHeadId(activeHead)} · Group G${mambaGroupForHead(activeHead)} (16 mini / 64 prod)`;
    topRow.appendChild(label);

    if (pinnedHead !== null) {
      const resetHead = document.createElement('button');
      resetHead.type = 'button';
      resetHead.className = 'qhead-clear-btn';
      resetHead.textContent = 'Clear';
      resetHead.addEventListener('click', () => this.cb.onSelectPart(null));
      topRow.appendChild(resetHead);
    }
    wrap.appendChild(topRow);

    const groupsGrid = document.createElement('div');
    groupsGrid.className = 'mhead-groups-grid';

    for (let g = 0; g < 8; g++) {
      const pair = document.createElement('div');
      pair.className = 'mhead-pair';

      const pairLabel = document.createElement('div');
      pairLabel.className = 'mhead-pair-label';
      pairLabel.textContent = `G${g}`;
      pair.appendChild(pairLabel);

      const row = document.createElement('div');
      row.className = 'mhead-pair-row';
      for (let local = 0; local < 2; local++) {
        const h = g * 2 + local;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qhead-btn';
        btn.dataset.mhead = String(h);
        btn.textContent = formatMambaHeadId(h);
        const isSel = pinnedHead === h || (pinnedHead === null && activeMambaHead === h);
        const isActive = activeHead === h;
        btn.classList.toggle('is-selected', isSel);
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', String(isSel));
        btn.addEventListener('click', () => {
          this.cb.onSelectPart(pinnedHead === h ? null : `mamba-head-${h}`);
        });
        row.appendChild(btn);
      }
      pair.appendChild(row);
      groupsGrid.appendChild(pair);
    }

    wrap.appendChild(groupsGrid);
    return wrap;
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns.length = 0;
  }
}
