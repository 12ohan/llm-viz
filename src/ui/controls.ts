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
  type SceneState,
  type TocSectionId,
  type ViewMode,
} from '../model/state';
import type { CutawayPartInfo, CutawaySceneGeometry } from '../scene/cutaways';

export interface ControlsCallbacks {
  onSelectMode: (mode: ViewMode) => void;
  onSelectSection: (section: TocSectionId) => void;
  onSelectMambaBay: (bayId: 1 | 42) => void;
  onSelectGqaBay: (bayId: 6 | 43) => void;
  onInspectInternals: (kind: BayKind, bayId?: number) => void;
  onTokensChange: (tokens: number) => void;
  onPrecisionChange: (mode: PrecisionMode) => void;
  onFitAll: () => void;
  onZoomSelectedBay: () => void;
  onReset: () => void;
  onStepBay: (delta: -1 | 1) => void;
  onClearSelection: () => void;
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
  gqaBayButtons: HTMLButtonElement[];
  miniNodes: SVGGElement[];
  towerSubbar: HTMLDivElement;
  mambaSubbar: HTMLDivElement;
  moeSubbar: HTMLDivElement;
  gqaSubbar: HTMLDivElement;
  floatingBadge: HTMLDivElement;
  editorialPanel: HTMLDivElement;
  tokenSlider: HTMLInputElement;
  tokenDec: HTMLButtonElement;
  tokenInc: HTMLButtonElement;
  tokenValue: HTMLOutputElement;
  precisionBf16: HTMLButtonElement;
  precisionFp8: HTMLButtonElement;
  memoryReadout: HTMLDivElement;
  bayStepRow: HTMLDivElement;
  bayPrev: HTMLButtonElement;
  bayNext: HTMLButtonElement;
  bayClear: HTMLButtonElement;
  zoomBayBtn: HTMLButtonElement;
  fitBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
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
    gqaBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-bay]')),
    miniNodes: Array.from(document.querySelectorAll<SVGGElement>('.mini-node[data-mini-section]')),
    towerSubbar: get<HTMLDivElement>('tower-subbar'),
    mambaSubbar: get<HTMLDivElement>('mamba-subbar'),
    moeSubbar: get<HTMLDivElement>('moe-subbar'),
    gqaSubbar: get<HTMLDivElement>('gqa-subbar'),
    floatingBadge: get<HTMLDivElement>('floating-badge'),
    editorialPanel: get<HTMLDivElement>('editorial-panel'),
    tokenSlider: get<HTMLInputElement>('token-slider'),
    tokenDec: get<HTMLButtonElement>('token-dec'),
    tokenInc: get<HTMLButtonElement>('token-inc'),
    tokenValue: get<HTMLOutputElement>('token-value'),
    precisionBf16: get<HTMLButtonElement>('precision-bf16'),
    precisionFp8: get<HTMLButtonElement>('precision-fp8'),
    memoryReadout: get<HTMLDivElement>('memory-readout'),
    bayStepRow: get<HTMLDivElement>('bay-step-row'),
    bayPrev: get<HTMLButtonElement>('bay-prev'),
    bayNext: get<HTMLButtonElement>('bay-next'),
    bayClear: get<HTMLButtonElement>('bay-clear'),
    zoomBayBtn: get<HTMLButtonElement>('zoom-bay-btn'),
    fitBtn: get<HTMLButtonElement>('fit-btn'),
    resetBtn: get<HTMLButtonElement>('reset-btn'),
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

    for (const btn of this.els.gqaBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.gqaBay ?? '6', 10);
        if (id === 6 || id === 43) this.cb.onSelectGqaBay(id);
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

    // Mamba & GQA representative bay toggles
    for (const btn of this.els.mambaBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.mambaBay) === state.mambaBayId);
    }
    for (const btn of this.els.gqaBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.gqaBay) === state.gqaBayId);
    }

    // Floating corner badge reflects current mode clearly
    if (state.mode === 'tower') {
      this.els.floatingBadge.textContent = 'Nemotron 3 Nano 30B-A3B · 3.2B active';
    } else if (state.mode === 'cutaway_mamba') {
      this.els.floatingBadge.textContent =
        state.mambaBayId === 42
          ? 'Cutaway: Mamba-2 · Bay 42 Feeder (BF16 §4.2)'
          : 'Cutaway: Mamba-2 · Bay 01 (2.05 MiB fixed SSM)';
    } else if (state.mode === 'cutaway_moe') {
      this.els.floatingBadge.textContent =
        'Cutaway: MoE 128+1 · Bay 02 (6 Active + Shared 3,712)';
    } else {
      this.els.floatingBadge.textContent = `Cutaway: GQA Attention · Bay ${formatBayId(state.gqaBayId)} (32Q / 2KV, NoPE)`;
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

    // Memory readout (with Part A #4 static weights vs dynamic sequence cache clarification!)
    const readoutKey = `${state.tokens}:${state.precision}`;
    if (readoutKey !== this.lastReadoutKey) {
      this.lastReadoutKey = readoutKey;
      const mem = memoryFor(this.spec, state.tokens, state.precision);
      const fmt = formatMemoryReadout(mem);

      this.els.memoryReadout.innerHTML = '';
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
        this.els.memoryReadout.appendChild(div);
      }
    }

    // Middle editorial panel
    const activeBayId = isTower ? state.hoveredBay ?? state.selectedBay : null;
    const activePart = !isTower ? state.hoveredPart ?? state.selectedPart : null;
    const editorialKey = `${state.mode}:${state.mambaBayId}:${state.gqaBayId}:${state.section}:${activeBayId ?? 'none'}:${state.selectedBay ?? 'none'}:${activePart ?? 'none'}:${state.selectedPart ?? 'none'}:${state.precision}:${state.tokens}`;

    if (editorialKey !== this.lastEditorialKey) {
      this.lastEditorialKey = editorialKey;
      if (isTower) {
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

  private renderTowerEditorial(
    section: TocSectionId,
    bayId: number | null,
    isPinned: boolean,
    precision: PrecisionMode,
    tokens: number,
  ): void {
    const panel = this.els.editorialPanel;
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

      panel.append(header, p, list, actions);
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
      badge.textContent = isFeeder && isFp8 ? 'BF16 Preserved §4.2' : '2.05 MiB Fixed';
      if (isFeeder && isFp8) badge.classList.add('ed-badge-gold');

      bodyText = isFeeder
        ? 'Feeder preceding Bay 43 GQA — preserved in BF16 (NVIDIA §4.2). Every compute projection, 4-tap Causal Conv1D, and the 64-head FP32 SSM state stay vivid BF16/FP32 while scrubbing T updates the cyan state cells in place.'
        : 'Input projection splits into three parallel paths: bypass gate z (4,096), Causal Conv1D (k=4, always BF16) → SiLU, and per-head decay Δt (64). Scrubbing T = 1 → 16 updates the 8×8 cyan SSM state array in place without growing physical memory.';

      specs = [
        isFeeder
          ? 'Bay 42 Feeder: In-Proj, Out-Proj, Conv1D stay BF16 in FP8 Selective (gold rim)'
          : `Bay 01 precision: In-Proj & Out-Proj ${isFp8 ? 'FP8 frosted' : 'BF16'} · Conv1D ALWAYS BF16 · SSM FP32`,
        'Recurrence h_t = A·h_{t−1} + B·x_t · 2.05 MiB/layer (47.1 MiB across 23 bays)',
        'Click any 3D tensor slab to inspect its exact dimensions and precision.',
      ];
    } else if (state.mode === 'cutaway_moe') {
      const activeIds = cutaway?.moeStats?.activeExpertIds ?? [];
      const activeList = activeIds.map((id) => `#${id}`).join(', ');
      title.textContent = 'MoE 128+1 Cutaway · Bay 02';
      badge.textContent = `6 / 128 Active · T=${state.tokens}`;

      bodyText =
        'Inside each MoE bay, experts execute in PARALLEL — never serialized. The sigmoid router activates 6 of 128 routed experts (2 non-gated ReLU² plates each: up_proj + down_proj, NO gate_proj) while 122 experts stay idle (95.3% skipped), running alongside the always-active double-width shared expert.';

      specs = [
        `Token T=${state.tokens} active routed experts: ${activeList || '6 of 128'}`,
        SHARED_EXPERT_NOTE,
        '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B token cache',
      ];
    } else {
      const perBayKiB = ((isFp8 ? 512 : 1024) * state.tokens) / 1024;
      title.textContent = `GQA Cutaway · Bay ${formatBayId(state.gqaBayId)}`;
      badge.textContent = isFp8 ? 'BF16 Compute · FP8 KV' : '32Q / 2KV · NoPE';
      if (isFp8) badge.classList.add('ed-badge-gold');

      bodyText =
        'Thirty-two query heads (2 groups × 16) attend across two 16× thinner KV heads with NoPE (No Positional Embeddings — token order arrives from preceding Mamba-2 layers). Scrubbing T physically grows both KV stacks and widens Attn Scores (32 × T).';

      specs = [
        `Live KV stacks: 2 KV heads × ${state.tokens} tokens = ${perBayKiB} KiB (${isFp8 ? '+0.5 KiB/tok FP8' : '+1 KiB/tok BF16'})`,
        isFp8
          ? 'FP8 Selective: Q/K/V/O compute stays BF16 (gold rim); ONLY KV-cache slices switch to FP8'
          : 'Toggle FP8 Selective to halve the KV slices while keeping Q/K/V/O compute in BF16',
        'Click Q Proj, K/V Proj, KV Head 0/1, Attn Scores, or O Proj to inspect.',
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

    panel.append(header, p, list, actions);
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns.length = 0;
  }
}
