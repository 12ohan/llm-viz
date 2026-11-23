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
  type NemotronSpec,
  type PrecisionMode,
} from '../model/spec';
import {
  MAX_TOKENS,
  MIN_TOKENS,
  type SceneState,
  type TocSectionId,
} from '../model/state';

export interface ControlsCallbacks {
  onSelectSection: (section: TocSectionId) => void;
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
}

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
  },
  moe: {
    title: 'Mixture of Experts (128 + 1)',
    badge: '23 bays · 6 routed + 1 shared',
    body:
      'A 128-score sigmoid router activates exactly six routed experts per token (amber-lit tiles), each with width 1,856 and squared ReLU. Beside the tray sits one always-active double-width shared expert (3,712) wired directly into the residual return.',
    specs: [
      'Routed experts: 6 active of 128 · width 2,688 → 1,856 → 2,688 (ReLU²)',
      'Shared expert: 1 double-width branch · 2,688 → 3,712 → 2,688 (checkpoint spec)',
      'Token history storage: 0 B (stateless feed-forward bank)',
    ],
  },
  gqa: {
    title: 'Grouped-Query Attention',
    badge: '6 bays · 06, 13, 20, 27, 34, 43',
    body:
      'Only six layers use full attention, standing out along the tower with periwinkle Q/K/V projection plates and two paired emerald/cyan KV-cache tracks extending forward. Thirty-two query heads share just two KV heads (16 query heads per KV pair).',
    specs: [
      'Q projection: 32 heads × 128 = 4,096 · K & V projections: 2 heads × 128 = 256 each',
      'Logical KV payload: +6 KiB/token in BF16 · +3 KiB/token in FP8 Selective',
      '52-layer pure GQA reference would cost +52 KiB/token (BF16) or +26 KiB/token (FP8)',
    ],
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
  tocButtons: HTMLButtonElement[];
  presetButtons: HTMLButtonElement[];
  miniNodes: SVGGElement[];
  editorialPanel: HTMLDivElement;
  tokenSlider: HTMLInputElement;
  tokenDec: HTMLButtonElement;
  tokenInc: HTMLButtonElement;
  tokenValue: HTMLOutputElement;
  precisionBf16: HTMLButtonElement;
  precisionFp8: HTMLButtonElement;
  memoryReadout: HTMLDivElement;
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
    tocButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.toc-btn[data-section]')),
    presetButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.pill-btn[data-preset]')),
    miniNodes: Array.from(document.querySelectorAll<SVGGElement>('.mini-node[data-mini-section]')),
    editorialPanel: get<HTMLDivElement>('editorial-panel'),
    tokenSlider: get<HTMLInputElement>('token-slider'),
    tokenDec: get<HTMLButtonElement>('token-dec'),
    tokenInc: get<HTMLButtonElement>('token-inc'),
    tokenValue: get<HTMLOutputElement>('token-value'),
    precisionBf16: get<HTMLButtonElement>('precision-bf16'),
    precisionFp8: get<HTMLButtonElement>('precision-fp8'),
    memoryReadout: get<HTMLDivElement>('memory-readout'),
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

  sync(state: SceneState): void {
    // Active TOC & floating presets
    for (const btn of this.els.tocButtons) {
      btn.classList.toggle('is-active', btn.dataset.section === state.section);
    }
    for (const btn of this.els.presetButtons) {
      btn.classList.toggle('is-active', btn.dataset.preset === state.section);
    }
    for (const node of this.els.miniNodes) {
      node.classList.toggle('is-active', node.dataset.miniSection === state.section);
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

    // Zoom bay button & Clear button
    const hasPinned = state.selectedBay !== null;
    this.els.bayClear.hidden = !hasPinned;
    this.els.zoomBayBtn.disabled = !hasPinned;
    this.els.zoomBayBtn.textContent = hasPinned
      ? `Zoom Bay ${formatBayId(state.selectedBay!)}`
      : 'Zoom Bay';

    // Memory readout
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
      ];
      for (const r of rows) {
        const div = document.createElement('div');
        div.className = r.cls;
        div.textContent = r.text;
        this.els.memoryReadout.appendChild(div);
      }
    }

    // Middle editorial panel (shows inspected bay when hovered/pinned, else active TOC section)
    const activeBayId = state.hoveredBay ?? state.selectedBay;
    const editorialKey = `${state.section}:${activeBayId ?? 'none'}:${state.selectedBay ?? 'none'}:${state.precision}:${state.tokens}`;
    if (editorialKey !== this.lastEditorialKey) {
      this.lastEditorialKey = editorialKey;
      this.renderEditorial(state.section, activeBayId, hasPinned, state.precision, state.tokens);
    }
  }

  private renderEditorial(
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
          narrative = `Bay ${formatBayId(bay.id)} is one of six Grouped-Query Attention layers. Thirty-two query heads read from two paired emerald/cyan KV tracks extending outward along depth. In FP8 Selective mode, the attention body stays BF16 while its KV cache switches to FP8.`;
        } else {
          storageLine = 'Memory: 0 B token history (stateless MoE layer)';
          narrative = `Bay ${formatBayId(bay.id)} routes each token through a 128-score sigmoid gate to activate six routed experts (width 1,856, ReLU²) in amber, summed alongside the always-active double-width shared expert (width 3,712).`;
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
        for (const item of [computeLine, storageLine, PARAMETER_CAPTION]) {
          const li = document.createElement('li');
          li.textContent = item;
          list.appendChild(li);
        }

        panel.append(header, sub, p, list);
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
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns.length = 0;
  }
}
