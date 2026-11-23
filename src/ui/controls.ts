import { formatMemoryReadout, memoryFor } from '../model/memory';
import {
  bayKindLabel,
  bodyPrecision,
  conv1dPrecision,
  formatBayId,
  formatBracketCaption,
  kvCachePrecision,
  ssmStateDtype,
  type NemotronSpec,
  type PrecisionMode,
} from '../model/spec';
import { MAX_TOKENS, MIN_TOKENS, type SceneState } from '../model/state';

export interface ControlsCallbacks {
  onTokensChange: (tokens: number) => void;
  onPrecisionChange: (mode: PrecisionMode) => void;
  onReset: () => void;
  onStepBay: (delta: -1 | 1) => void;
  onClearSelection: () => void;
}

export interface ControlsElements {
  tokenSlider: HTMLInputElement;
  tokenDec: HTMLButtonElement;
  tokenInc: HTMLButtonElement;
  tokenValue: HTMLOutputElement;
  precisionBf16: HTMLButtonElement;
  precisionFp8: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  memoryReadout: HTMLDivElement;
  bayInspector: HTMLDivElement;
  bayPrev: HTMLButtonElement;
  bayNext: HTMLButtonElement;
  bayClear: HTMLButtonElement;
}

export function queryControlsElements(): ControlsElements {
  const get = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing required DOM element #${id}`);
    return el as T;
  };
  return {
    tokenSlider: get<HTMLInputElement>('token-slider'),
    tokenDec: get<HTMLButtonElement>('token-dec'),
    tokenInc: get<HTMLButtonElement>('token-inc'),
    tokenValue: get<HTMLOutputElement>('token-value'),
    precisionBf16: get<HTMLButtonElement>('precision-bf16'),
    precisionFp8: get<HTMLButtonElement>('precision-fp8'),
    resetBtn: get<HTMLButtonElement>('reset-btn'),
    memoryReadout: get<HTMLDivElement>('memory-readout'),
    bayInspector: get<HTMLDivElement>('bay-inspector'),
    bayPrev: get<HTMLButtonElement>('bay-prev'),
    bayNext: get<HTMLButtonElement>('bay-next'),
    bayClear: get<HTMLButtonElement>('bay-clear'),
  };
}

/**
 * Binds accessible controls, tabular memory readout, and keyboard bay inspector.
 */
export class MacroControls {
  private readonly spec: NemotronSpec;
  private readonly els: ControlsElements;
  private readonly cb: ControlsCallbacks;
  private readonly cleanupFns: (() => void)[] = [];

  // Cache last rendered text to avoid redundant DOM writes
  private lastReadoutKey = '';
  private lastInspectorKey = '';

  constructor(spec: NemotronSpec, els: ControlsElements, cb: ControlsCallbacks) {
    this.spec = spec;
    this.els = els;
    this.cb = cb;
    this.bindEvents();
  }

  private on<K extends keyof HTMLElementEventMap>(
    el: HTMLElement,
    type: K,
    handler: (ev: HTMLElementEventMap[K]) => void,
  ): void {
    el.addEventListener(type, handler as EventListener);
    this.cleanupFns.push(() => el.removeEventListener(type, handler as EventListener));
  }

  private bindEvents(): void {
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

    this.on(this.els.precisionBf16, 'click', () => {
      this.cb.onPrecisionChange('bf16');
    });

    this.on(this.els.precisionFp8, 'click', () => {
      this.cb.onPrecisionChange('fp8_selective');
    });

    this.on(this.els.resetBtn, 'click', () => {
      this.cb.onReset();
    });

    this.on(this.els.bayPrev, 'click', () => {
      this.cb.onStepBay(-1);
    });

    this.on(this.els.bayNext, 'click', () => {
      this.cb.onStepBay(1);
    });

    this.on(this.els.bayClear, 'click', () => {
      this.cb.onClearSelection();
    });
  }

  sync(state: SceneState): void {
    // Sync slider and precision toggle
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

    // Sync tabular memory readout (§1 & §5)
    const readoutKey = `${state.tokens}:${state.precision}`;
    if (readoutKey !== this.lastReadoutKey) {
      this.lastReadoutKey = readoutKey;
      const mem = memoryFor(this.spec, state.tokens, state.precision);
      const fmt = formatMemoryReadout(mem);

      this.els.memoryReadout.innerHTML = '';
      const items = [
        { text: fmt.mambaText, cls: 'readout-item readout-fixed' },
        { text: fmt.kvText, cls: 'readout-item readout-kv' },
        { text: fmt.referenceText, cls: 'readout-item readout-ref' },
      ];
      items.forEach((item, idx) => {
        if (idx > 0) {
          const sep = document.createElement('span');
          sep.className = 'readout-sep';
          sep.setAttribute('aria-hidden', 'true');
          sep.textContent = '·';
          this.els.memoryReadout.appendChild(sep);
        }
        const span = document.createElement('span');
        span.className = item.cls;
        span.textContent = item.text;
        this.els.memoryReadout.appendChild(span);
      });
    }

    // Sync bay inspector
    const activeBayId = state.hoveredBay ?? state.selectedBay;
    const isPinned = state.selectedBay !== null;
    this.els.bayClear.hidden = !isPinned;

    const inspectorKey = `${activeBayId ?? 'none'}:${state.selectedBay ?? 'none'}:${state.precision}:${state.tokens}`;
    if (inspectorKey !== this.lastInspectorKey) {
      this.lastInspectorKey = inspectorKey;
      this.renderInspector(activeBayId, isPinned, state.precision, state.tokens);
    }
  }

  private renderInspector(
    bayId: number | null,
    isPinned: boolean,
    precision: PrecisionMode,
    tokens: number,
  ): void {
    const container = this.els.bayInspector;
    container.innerHTML = '';

    if (bayId === null) {
      const legend = document.createElement('div');
      legend.className = 'inspector-legend';
      legend.innerHTML = [
        '<span class="legend-entry"><strong>M</strong> Mamba-2 · fixed-state register (23 bays)</span>',
        '<span class="legend-sep" aria-hidden="true">·</span>',
        '<span class="legend-entry"><strong>E</strong> MoE · 128 routed + 1 double-width shared (23 bays)</span>',
        '<span class="legend-sep" aria-hidden="true">·</span>',
        '<span class="legend-entry"><strong>*</strong> GQA · 32Q / 2KV cache tracks (6 bays: 06, 13, 20, 27, 34, 43)</span>',
      ].join('');
      const hint = document.createElement('div');
      hint.className = 'inspector-hint';
      hint.textContent =
        'Click any bay or use ← / → keys to inspect · Scroll to zoom · Drag to pan · Esc clears selection.';
      container.append(legend, hint);
      return;
    }

    const bay = this.spec.bays[bayId - 1];
    if (!bay) return;

    const bracket = this.spec.brackets[bay.bracket];
    const computePrec = bodyPrecision(bay, precision);
    const computeLabel =
      precision === 'bf16'
        ? 'Compute: BF16'
        : computePrec === 'bf16'
          ? 'Compute: BF16 (preserved in FP8 Selective)'
          : 'Compute: FP8 weights/activations';

    let storageLabel: string;
    if (bay.kind === 'M') {
      const ssm = ssmStateDtype(bay)!.toUpperCase();
      const conv = conv1dPrecision(bay, precision)!.toUpperCase();
      storageLabel = `Memory: SSM ${ssm} (64×64×128) · Conv1D ${conv} (6,144×4) — fixed`;
    } else if (bay.kind === '*') {
      const kvPrec = kvCachePrecision(bay, precision)!;
      const perLayerBytes = kvPrec === 'fp8' ? 512 : 1024;
      const perLayerTotalKiB = (perLayerBytes * tokens) / 1024;
      const perLayerSlope = kvPrec === 'fp8' ? '0.5 KiB/token' : '1 KiB/token';
      storageLabel = `KV cache: ${kvPrec.toUpperCase()} (${perLayerTotalKiB} KiB at ${tokens} tokens, +${perLayerSlope})`;
    } else {
      storageLabel = 'Memory: 0 B token history (stateless; 6 of 128 routed + 1 shared)';
    }

    const roleNote = bay.feeder
      ? `Feeder immediately preceding attention bay ${formatBayId(bay.id + 1)}`
      : bay.kind === '*'
        ? 'Acquires K/V history (2 KV heads × 128 dim)'
        : bay.kind === 'M'
          ? 'Recurrent state update (in place)'
          : 'Sparse expert mixture (1,856 routed / 3,712 shared)';

    const topLine = document.createElement('div');
    topLine.className = 'inspector-primary';

    const idSpan = document.createElement('span');
    idSpan.className = 'inspector-bay-id';
    idSpan.textContent = `Bay ${formatBayId(bay.id)} / 52`;

    const kindSpan = document.createElement('span');
    kindSpan.className = 'inspector-kind';
    kindSpan.textContent = `${bay.kind} · ${bayKindLabel(bay.kind)}`;

    const spanBadge = document.createElement('span');
    spanBadge.className = 'inspector-bracket';
    spanBadge.textContent = formatBracketCaption(bracket);

    const statusSpan = document.createElement('span');
    statusSpan.className = 'inspector-status';
    statusSpan.textContent = isPinned ? 'Pinned' : 'Hover';

    topLine.append(idSpan, kindSpan, spanBadge, statusSpan);

    const detailLine = document.createElement('div');
    detailLine.className = 'inspector-details';
    detailLine.textContent = `${roleNote} · ${computeLabel} · ${storageLabel}`;

    container.append(topLine, detailLine);
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns.length = 0;
  }
}
