import {
  formatBytesAdaptive,
  formatMiB,
  formatTokenCount,
  logPositionFromTokens,
  MAX_SCALING_TOKENS,
  memoryCrossovers,
  memoryFor,
  tokensFromLogPosition,
} from '../model/memory';
import type { NemotronSpec, PrecisionMode } from '../model/spec';

export interface MemoryChartCallbacks {
  onClose: () => void;
  onPrecisionChange: (mode: PrecisionMode) => void;
  onProbeTokensChange: (tokens: number) => void;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const QUICK_JUMPS: readonly { label: string; tokens: number }[] = [
  { label: '1K', tokens: 1024 },
  { label: '8K', tokens: 8192 },
  { label: '32K', tokens: 32_768 },
  { label: '128K', tokens: 131_072 },
  { label: '256K', tokens: 262_144 },
  { label: '1M', tokens: MAX_SCALING_TOKENS },
];

const VB_W = 680;
const VB_H = 310;
const PAD_L = 68;
const PAD_R = 152;
const PAD_T = 24;
const PAD_B = 36;
const PLOT_W = VB_W - PAD_L - PAD_R;
const PLOT_H = VB_H - PAD_T - PAD_B;

const MIN_Y_BYTES = 16 * 1024; // 16 KiB
const MAX_Y_BYTES = 80 * 1024 * 1024 * 1024; // 80 GiB
const LOG_MAX_T = Math.log(MAX_SCALING_TOKENS);
const LOG_MIN_Y = Math.log(MIN_Y_BYTES);
const LOG_SPAN_Y = Math.log(MAX_Y_BYTES) - LOG_MIN_Y;

function mapX(tokens: number): number {
  const clamped = Math.max(1, Math.min(MAX_SCALING_TOKENS, tokens));
  return PAD_L + (Math.log(clamped) / LOG_MAX_T) * PLOT_W;
}

function mapY(bytes: number): number {
  const clamped = Math.max(MIN_Y_BYTES, Math.min(MAX_Y_BYTES, bytes));
  const norm = (Math.log(clamped) - LOG_MIN_Y) / LOG_SPAN_Y;
  return PAD_T + (1 - norm) * PLOT_H;
}

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  if (text !== undefined) el.textContent = text;
  return el;
}

/**
 * Zero-dependency inline SVG log-log Memory Scaling (1 -> 1M tokens) chart drawer
 * driven directly by src/model/memory.ts.
 */
export class MemoryScalingDrawer {
  private readonly spec: NemotronSpec;
  private readonly host: HTMLElement;
  private readonly cb: MemoryChartCallbacks;
  private readonly cleanupFns: (() => void)[] = [];

  private readonly precisionBf16Btn: HTMLButtonElement;
  private readonly precisionFp8Btn: HTMLButtonElement;
  private readonly closeBtn: HTMLButtonElement;
  private readonly chartHost: HTMLDivElement;
  private readonly probeSlider: HTMLInputElement;
  private readonly probeOutput: HTMLOutputElement;
  private readonly jumpButtons: HTMLButtonElement[] = [];
  private readonly statsGrid: HTMLDivElement;

  private lastKey = '';

  constructor(spec: NemotronSpec, host: HTMLElement, cb: MemoryChartCallbacks) {
    this.spec = spec;
    this.host = host;
    this.cb = cb;

    this.host.innerHTML = '';

    const backdrop = document.createElement('div');
    backdrop.className = 'mem-drawer-backdrop';
    this.on(backdrop, 'click', () => this.cb.onClose());

    const card = document.createElement('section');
    card.className = 'mem-drawer-card';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-modal', 'true');
    card.setAttribute('aria-labelledby', 'mem-drawer-title');

    // --- Header ---
    const header = document.createElement('div');
    header.className = 'mem-drawer-header';

    const titleWrap = document.createElement('div');
    titleWrap.className = 'mem-drawer-title-wrap';

    const title = document.createElement('h2');
    title.id = 'mem-drawer-title';
    title.className = 'mem-drawer-title';
    title.textContent = 'Sequence State Memory Scaling (1 → 1M Tokens)';

    const subtitle = document.createElement('p');
    subtitle.className = 'mem-drawer-subtitle';
    subtitle.textContent =
      'Log-log comparison of Nemotron 3 Nano (47.1 MiB Mamba-2 fixed + 6 GQA KV layers) vs. a 52-layer pure GQA Transformer.';

    titleWrap.append(title, subtitle);

    const headerActions = document.createElement('div');
    headerActions.className = 'mem-drawer-actions';

    const seg = document.createElement('div');
    seg.className = 'segmented-control mem-drawer-seg';
    seg.setAttribute('role', 'radiogroup');
    seg.setAttribute('aria-label', 'Chart precision mode');

    this.precisionBf16Btn = document.createElement('button');
    this.precisionBf16Btn.type = 'button';
    this.precisionBf16Btn.className = 'seg-btn';
    this.precisionBf16Btn.setAttribute('role', 'radio');
    this.precisionBf16Btn.textContent = 'BF16';
    this.on(this.precisionBf16Btn, 'click', () => this.cb.onPrecisionChange('bf16'));

    this.precisionFp8Btn = document.createElement('button');
    this.precisionFp8Btn.type = 'button';
    this.precisionFp8Btn.className = 'seg-btn';
    this.precisionFp8Btn.setAttribute('role', 'radio');
    this.precisionFp8Btn.textContent = 'FP8 Selective';
    this.on(this.precisionFp8Btn, 'click', () => this.cb.onPrecisionChange('fp8_selective'));

    seg.append(this.precisionBf16Btn, this.precisionFp8Btn);

    this.closeBtn = document.createElement('button');
    this.closeBtn.type = 'button';
    this.closeBtn.className = 'mem-close-btn';
    this.closeBtn.setAttribute('aria-label', 'Close Memory Scaling chart');
    this.closeBtn.textContent = '×';
    this.on(this.closeBtn, 'click', () => this.cb.onClose());

    headerActions.append(seg, this.closeBtn);
    header.append(titleWrap, headerActions);

    // --- Chart SVG container ---
    this.chartHost = document.createElement('div');
    this.chartHost.className = 'mem-chart-svg-wrap';

    // --- Interactive Probe Controls ---
    const probeBox = document.createElement('div');
    probeBox.className = 'mem-probe-box';

    const probeTop = document.createElement('div');
    probeTop.className = 'mem-probe-top';

    const probeLabel = document.createElement('label');
    probeLabel.className = 'mem-probe-label';
    probeLabel.htmlFor = 'mem-probe-slider';
    probeLabel.textContent = 'Context length probe (log scale)';

    this.probeOutput = document.createElement('output');
    this.probeOutput.className = 'mem-probe-output';
    this.probeOutput.setAttribute('for', 'mem-probe-slider');

    probeTop.append(probeLabel, this.probeOutput);

    this.probeSlider = document.createElement('input');
    this.probeSlider.id = 'mem-probe-slider';
    this.probeSlider.type = 'range';
    this.probeSlider.min = '0';
    this.probeSlider.max = '1000';
    this.probeSlider.step = '1';
    this.probeSlider.className = 'mem-probe-slider';
    this.probeSlider.setAttribute('aria-label', 'Context length from 1 to 1,048,576 tokens');
    this.on(this.probeSlider, 'input', () => {
      const norm = Number(this.probeSlider.value) / 1000;
      this.cb.onProbeTokensChange(tokensFromLogPosition(norm));
    });

    const jumpRow = document.createElement('div');
    jumpRow.className = 'mem-jump-row';
    const jumpLabel = document.createElement('span');
    jumpLabel.className = 'mem-jump-label';
    jumpLabel.textContent = 'Quick jump:';
    jumpRow.appendChild(jumpLabel);

    for (const j of QUICK_JUMPS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'mem-jump-btn';
      btn.dataset.tokens = String(j.tokens);
      btn.textContent = j.label;
      this.on(btn, 'click', () => this.cb.onProbeTokensChange(j.tokens));
      this.jumpButtons.push(btn);
      jumpRow.appendChild(btn);
    }

    this.statsGrid = document.createElement('div');
    this.statsGrid.className = 'mem-stats-grid';
    this.statsGrid.setAttribute('data-testid', 'memory-probe-stats');

    probeBox.append(probeTop, this.probeSlider, jumpRow, this.statsGrid);

    // --- Paper Footnote ---
    const footnote = document.createElement('footer');
    footnote.className = 'mem-footnote';
    footnote.innerHTML =
      '<strong>Benchmark context</strong> (NVIDIA Technical Report, <a href="https://arxiv.org/abs/2512.20848" target="_blank" rel="noopener noreferrer">arXiv:2512.20848v1</a>): ' +
      'RULER-100 @ 1M = <strong>86.34</strong> (reasoning-off, Table 3); up to <strong>3.3× throughput</strong> vs. Qwen3-30B-A3B-Thinking &amp; <strong>2.2×</strong> vs. GPT-OSS-20B on a single H200 at 8K input / 16K output (Figure 1, noting cross-model precision differences across published baselines).';

    card.append(header, this.chartHost, probeBox, footnote);
    this.host.append(backdrop, card);
  }

  private on<K extends keyof HTMLElementEventMap>(
    el: Element,
    type: K,
    handler: (ev: HTMLElementEventMap[K]) => void,
  ): void {
    el.addEventListener(type, handler as EventListener);
    this.cleanupFns.push(() => el.removeEventListener(type, handler as EventListener));
  }

  sync(open: boolean, precision: PrecisionMode, probeTokens: number): void {
    this.host.hidden = !open;
    if (!open) return;

    const clampedTokens = Math.max(1, Math.min(MAX_SCALING_TOKENS, Math.round(probeTokens)));
    const key = `${precision}:${clampedTokens}`;
    if (key === this.lastKey) return;
    this.lastKey = key;

    const isBf16 = precision === 'bf16';
    this.precisionBf16Btn.classList.toggle('is-active', isBf16);
    this.precisionBf16Btn.setAttribute('aria-checked', String(isBf16));
    this.precisionFp8Btn.classList.toggle('is-active', !isBf16);
    this.precisionFp8Btn.setAttribute('aria-checked', String(!isBf16));

    const sliderPos = String(Math.round(logPositionFromTokens(clampedTokens) * 1000));
    if (this.probeSlider.value !== sliderPos) {
      this.probeSlider.value = sliderPos;
    }
    this.probeOutput.textContent = `${formatTokenCount(clampedTokens)} tokens`;

    for (const btn of this.jumpButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.tokens) === clampedTokens);
    }

    this.renderSvg(precision, clampedTokens);
    this.renderProbeStats(precision, clampedTokens);
  }

  private renderSvg(precision: PrecisionMode, probeTokens: number): void {
    const cross = memoryCrossovers(this.spec, precision);
    const probeMem = memoryFor(this.spec, probeTokens, precision);
    const isFp8 = precision === 'fp8_selective';

    const svg = svgEl('svg', {
      viewBox: `0 0 ${VB_W} ${VB_H}`,
      class: 'mem-svg',
      role: 'img',
      'aria-label': `Log-log memory scaling chart from 1 to 1M tokens in ${isFp8 ? 'FP8 Selective' : 'BF16'} mode`,
    });

    // Background plot area
    svg.appendChild(
      svgEl('rect', {
        x: PAD_L,
        y: PAD_T,
        width: PLOT_W,
        height: PLOT_H,
        fill: '#FAFBFC',
        stroke: '#DFE4EC',
        'stroke-width': 1,
      }),
    );

    // X gridlines & labels
    const xTicks: readonly { tokens: number; label: string }[] = [
      { tokens: 1, label: '1' },
      { tokens: 10, label: '10' },
      { tokens: 100, label: '100' },
      { tokens: 1024, label: '1K' },
      { tokens: 8192, label: '8K' },
      { tokens: 32_768, label: '32K' },
      { tokens: 131_072, label: '128K' },
      { tokens: MAX_SCALING_TOKENS, label: '1M' },
    ];
    for (const tick of xTicks) {
      const x = mapX(tick.tokens);
      svg.appendChild(
        svgEl('line', {
          x1: x.toFixed(1),
          y1: PAD_T,
          x2: x.toFixed(1),
          y2: PAD_T + PLOT_H,
          stroke: tick.tokens === MAX_SCALING_TOKENS ? '#CBD5E1' : '#E8EDF5',
          'stroke-width': 1,
        }),
      );
      svg.appendChild(
        svgEl(
          'text',
          {
            x: x.toFixed(1),
            y: PAD_T + PLOT_H + 16,
            'text-anchor': 'middle',
            fill: '#475569',
            'font-size': 10,
            'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
          },
          tick.label,
        ),
      );
    }

    // X axis title
    svg.appendChild(
      svgEl(
        'text',
        {
          x: PAD_L + PLOT_W * 0.5,
          y: VB_H - 4,
          'text-anchor': 'middle',
          fill: '#64748B',
          'font-size': 10,
          'font-weight': 600,
        },
        'Cached sequence length (tokens, log scale)',
      ),
    );

    // Y gridlines & labels
    const yTicks: readonly { bytes: number; label: string; isMambaBase?: boolean }[] = [
      { bytes: 32 * 1024, label: '32 KiB' },
      { bytes: 256 * 1024, label: '256 KiB' },
      { bytes: 4 * 1024 * 1024, label: '4 MiB' },
      { bytes: cross.at1M.fixedMambaBytes, label: '47.1 MiB', isMambaBase: true },
      { bytes: 512 * 1024 * 1024, label: '512 MiB' },
      { bytes: 4 * 1024 * 1024 * 1024, label: '4 GiB' },
      { bytes: 64 * 1024 * 1024 * 1024, label: '64 GiB' },
    ];
    for (const tick of yTicks) {
      const y = mapY(tick.bytes);
      svg.appendChild(
        svgEl('line', {
          x1: PAD_L,
          y1: y.toFixed(1),
          x2: PAD_L + PLOT_W,
          y2: y.toFixed(1),
          stroke: tick.isMambaBase ? '#38BDF8' : '#E8EDF5',
          'stroke-width': tick.isMambaBase ? 1.2 : 1,
          'stroke-dasharray': tick.isMambaBase ? '4 3' : 'none',
        }),
      );
      svg.appendChild(
        svgEl(
          'text',
          {
            x: PAD_L - 8,
            y: (y + 3.5).toFixed(1),
            'text-anchor': 'end',
            fill: tick.isMambaBase ? '#0284C7' : '#475569',
            'font-size': 10,
            'font-weight': tick.isMambaBase ? 700 : 500,
            'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
          },
          tick.label,
        ),
      );
    }

    // Build smooth sample curves (96 log-spaced points + exact milestone tokens)
    const sampleSet = new Set<number>([
      1,
      cross.beatsGqaTokens,
      cross.kvExceedsMambaTokens,
      probeTokens,
      MAX_SCALING_TOKENS,
    ]);
    for (let i = 0; i <= 96; i++) {
      sampleSet.add(tokensFromLogPosition(i / 96));
    }
    const sampleTokens = Array.from(sampleSet).sort((a, b) => a - b);

    const gqaPts: string[] = [];
    const nemotronPts: string[] = [];
    const savingsUpper: string[] = [];
    const savingsLower: string[] = [];

    for (const t of sampleTokens) {
      const m = memoryFor(this.spec, t, precision);
      const x = mapX(t);
      const yGqa = mapY(m.referenceBytes);
      const yNem = mapY(m.hybridTotalBytes);
      gqaPts.push(`${x.toFixed(1)},${yGqa.toFixed(1)}`);
      nemotronPts.push(`${x.toFixed(1)},${yNem.toFixed(1)}`);

      if (t >= cross.beatsGqaTokens) {
        savingsUpper.push(`${x.toFixed(1)},${yGqa.toFixed(1)}`);
        savingsLower.unshift(`${x.toFixed(1)},${yNem.toFixed(1)}`);
      }
    }

    // Shaded savings wedge where Nemotron beats 52-layer GQA
    if (savingsUpper.length > 1) {
      svg.appendChild(
        svgEl('polygon', {
          points: [...savingsUpper, ...savingsLower].join(' '),
          fill: 'rgba(22, 163, 74, 0.10)',
        }),
      );
    }

    // 52-Layer GQA Reference line
    svg.appendChild(
      svgEl('polyline', {
        points: gqaPts.join(' '),
        fill: 'none',
        stroke: '#64748B',
        'stroke-width': 2.1,
        'stroke-dasharray': '6 4',
      }),
    );

    // Nemotron 3 Nano Total line
    svg.appendChild(
      svgEl('polyline', {
        points: nemotronPts.join(' '),
        fill: 'none',
        stroke: '#1B2340',
        'stroke-width': 2.6,
      }),
    );

    // Subtle Mamba fixed label inside left plateau
    const mambaBaseY = mapY(cross.at1M.fixedMambaBytes);
    svg.appendChild(
      svgEl(
        'text',
        {
          x: PAD_L + 8,
          y: (mambaBaseY - 7).toFixed(1),
          fill: '#0284C7',
          'font-size': 9.5,
          'font-weight': 600,
        },
        'Mamba-2 fixed floor (47.1 MiB · 23 bays)',
      ),
    );

    // --- Milestone Dot 1: Beats 52-layer GQA ---
    const beatsMem = memoryFor(this.spec, cross.beatsGqaTokens, precision);
    const bx = mapX(cross.beatsGqaTokens);
    const by = mapY(beatsMem.hybridTotalBytes);

    svg.appendChild(
      svgEl('line', {
        x1: bx.toFixed(1),
        y1: by.toFixed(1),
        x2: (bx - 16).toFixed(1),
        y2: (by - 28).toFixed(1),
        stroke: '#16A34A',
        'stroke-width': 1.1,
      }),
    );
    svg.appendChild(
      svgEl('circle', {
        cx: bx.toFixed(1),
        cy: by.toFixed(1),
        r: 4.5,
        fill: '#16A34A',
        stroke: '#FFFFFF',
        'stroke-width': 1.6,
      }),
    );
    const tag1X = Math.max(PAD_L + 4, bx - 134);
    const tag1Y = by - 50;
    svg.appendChild(
      svgEl('rect', {
        x: tag1X.toFixed(1),
        y: tag1Y.toFixed(1),
        width: 128,
        height: 28,
        rx: 4,
        fill: '#FFFFFF',
        stroke: '#16A34A',
        'stroke-width': 1,
      }),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: (tag1X + 6).toFixed(1),
          y: (tag1Y + 12).toFixed(1),
          fill: '#15803D',
          'font-size': 9.2,
          'font-weight': 700,
        },
        'Beats 52-layer GQA',
      ),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: (tag1X + 6).toFixed(1),
          y: (tag1Y + 23).toFixed(1),
          fill: '#1E2430',
          'font-size': 9.2,
          'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
        },
        `${cross.beatsGqaTokens.toLocaleString('en-US')} tokens`,
      ),
    );

    // --- Milestone Dot 2: KV exceeds fixed Mamba state ---
    const kvMem = memoryFor(this.spec, cross.kvExceedsMambaTokens, precision);
    const kx = mapX(cross.kvExceedsMambaTokens);
    const ky = mapY(kvMem.hybridTotalBytes);

    svg.appendChild(
      svgEl('line', {
        x1: kx.toFixed(1),
        y1: ky.toFixed(1),
        x2: (kx + 12).toFixed(1),
        y2: (ky + 32).toFixed(1),
        stroke: '#D97706',
        'stroke-width': 1.1,
      }),
    );
    svg.appendChild(
      svgEl('circle', {
        cx: kx.toFixed(1),
        cy: ky.toFixed(1),
        r: 4.5,
        fill: '#D97706',
        stroke: '#FFFFFF',
        'stroke-width': 1.6,
      }),
    );
    const tag2X = Math.min(PAD_L + PLOT_W - 134, kx + 4);
    const tag2Y = ky + 32;
    svg.appendChild(
      svgEl('rect', {
        x: tag2X.toFixed(1),
        y: tag2Y.toFixed(1),
        width: 132,
        height: 28,
        rx: 4,
        fill: '#FFFFFF',
        stroke: '#D97706',
        'stroke-width': 1,
      }),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: (tag2X + 6).toFixed(1),
          y: (tag2Y + 12).toFixed(1),
          fill: '#B45309',
          'font-size': 9.2,
          'font-weight': 700,
        },
        'KV > Mamba fixed',
      ),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: (tag2X + 6).toFixed(1),
          y: (tag2Y + 23).toFixed(1),
          fill: '#1E2430',
          'font-size': 9.2,
          'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
        },
        `${cross.kvExceedsMambaTokens.toLocaleString('en-US')} tokens`,
      ),
    );

    // --- 1M Endpoint Callouts in right gutter ---
    const endX = mapX(MAX_SCALING_TOKENS);
    const endYGqa = mapY(cross.at1M.referenceBytes);
    const endYNem = mapY(cross.at1M.hybridTotalBytes);

    svg.appendChild(
      svgEl('circle', {
        cx: endX.toFixed(1),
        cy: endYGqa.toFixed(1),
        r: 4,
        fill: '#64748B',
        stroke: '#FFFFFF',
        'stroke-width': 1.5,
      }),
    );
    svg.appendChild(
      svgEl('circle', {
        cx: endX.toFixed(1),
        cy: endYNem.toFixed(1),
        r: 4.5,
        fill: '#1B2340',
        stroke: '#FFFFFF',
        'stroke-width': 1.5,
      }),
    );

    // Right gutter labels for 1M endpoints
    svg.appendChild(
      svgEl(
        'text',
        {
          x: endX + 10,
          y: (endYGqa - 2).toFixed(1),
          fill: '#475569',
          'font-size': 9.5,
          'font-weight': 700,
        },
        '52-Layer GQA @ 1M',
      ),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: endX + 10,
          y: (endYGqa + 10).toFixed(1),
          fill: '#475569',
          'font-size': 10.5,
          'font-weight': 700,
          'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
        },
        formatBytesAdaptive(cross.at1M.referenceBytes),
      ),
    );

    svg.appendChild(
      svgEl(
        'text',
        {
          x: endX + 10,
          y: (endYNem - 2).toFixed(1),
          fill: '#1B2340',
          'font-size': 9.5,
          'font-weight': 700,
        },
        'Nemotron 3 Nano @ 1M',
      ),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: endX + 10,
          y: (endYNem + 10).toFixed(1),
          fill: '#1B2340',
          'font-size': 10.5,
          'font-weight': 700,
          'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
        },
        `${formatBytesAdaptive(cross.at1M.hybridTotalBytes)} (~${cross.savingsRatioAt1M.toFixed(1)}× less)`,
      ),
    );

    // Savings bracket connecting 1M endpoints
    const brX = endX + 4;
    svg.appendChild(
      svgEl('line', {
        x1: brX,
        y1: (endYGqa + 6).toFixed(1),
        x2: brX,
        y2: (endYNem - 6).toFixed(1),
        stroke: '#16A34A',
        'stroke-width': 2,
      }),
    );

    // --- Interactive Probe Vertical Line & Dots ---
    const px = mapX(probeTokens);
    const pyNem = mapY(probeMem.hybridTotalBytes);
    const pyGqa = mapY(probeMem.referenceBytes);

    svg.appendChild(
      svgEl('line', {
        x1: px.toFixed(1),
        y1: PAD_T,
        x2: px.toFixed(1),
        y2: PAD_T + PLOT_H,
        stroke: '#E04F33',
        'stroke-width': 1.3,
        'stroke-dasharray': '3 2',
      }),
    );
    svg.appendChild(
      svgEl('circle', {
        cx: px.toFixed(1),
        cy: pyGqa.toFixed(1),
        r: 4,
        fill: '#64748B',
        stroke: '#E04F33',
        'stroke-width': 1.5,
      }),
    );
    svg.appendChild(
      svgEl('circle', {
        cx: px.toFixed(1),
        cy: pyNem.toFixed(1),
        r: 5,
        fill: '#E04F33',
        stroke: '#FFFFFF',
        'stroke-width': 1.8,
      }),
    );

    // Top-left legend inside plot
    const legX = PAD_L + 10;
    const legY = PAD_T + 14;
    svg.appendChild(
      svgEl('line', {
        x1: legX,
        y1: legY,
        x2: legX + 18,
        y2: legY,
        stroke: '#1B2340',
        'stroke-width': 2.6,
      }),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: legX + 24,
          y: legY + 3.5,
          fill: '#1E2430',
          'font-size': 10,
          'font-weight': 600,
        },
        `Nemotron 3 Nano (${isFp8 ? 'FP8 KV +3 KiB/tok' : 'BF16 KV +6 KiB/tok'})`,
      ),
    );
    svg.appendChild(
      svgEl('line', {
        x1: legX,
        y1: legY + 15,
        x2: legX + 18,
        y2: legY + 15,
        stroke: '#64748B',
        'stroke-width': 2,
        'stroke-dasharray': '5 3',
      }),
    );
    svg.appendChild(
      svgEl(
        'text',
        {
          x: legX + 24,
          y: legY + 18.5,
          fill: '#475569',
          'font-size': 10,
          'font-weight': 500,
        },
        `52-Layer GQA Ref (${isFp8 ? '+26 KiB/tok' : '+52 KiB/tok'})`,
      ),
    );

    this.chartHost.innerHTML = '';
    this.chartHost.appendChild(svg);
  }

  private renderProbeStats(precision: PrecisionMode, probeTokens: number): void {
    const m = memoryFor(this.spec, probeTokens, precision);
    const ratio = m.referenceBytes / m.hybridTotalBytes;
    const savingsText =
      ratio >= 1
        ? `${ratio.toFixed(2)}× smaller`
        : `${(m.hybridTotalBytes / m.referenceBytes).toFixed(1)}× fixed base`;

    this.statsGrid.innerHTML = '';
    const cards: readonly { label: string; value: string; bytes: string; tone?: string }[] = [
      {
        label: 'Mamba-2 fixed (23 bays)',
        value: formatMiB(m.fixedMambaBytes, 1),
        bytes: `${m.fixedMambaBytes.toLocaleString('en-US')} B`,
        tone: 'cyan',
      },
      {
        label: `6 GQA KV (${precision === 'fp8_selective' ? 'FP8' : 'BF16'})`,
        value: formatBytesAdaptive(m.kvBytes),
        bytes: `${m.kvBytes.toLocaleString('en-US')} B`,
      },
      {
        label: 'Nemotron 3 Nano total',
        value: formatBytesAdaptive(m.hybridTotalBytes),
        bytes: `${m.hybridTotalBytes.toLocaleString('en-US')} B`,
        tone: 'navy',
      },
      {
        label: '52-layer GQA ref',
        value: formatBytesAdaptive(m.referenceBytes),
        bytes: `${savingsText}`,
        tone: ratio >= 1 ? 'green' : 'muted',
      },
    ];

    for (const c of cards) {
      const box = document.createElement('div');
      box.className = `mem-stat-cell${c.tone ? ` is-${c.tone}` : ''}`;

      const lbl = document.createElement('div');
      lbl.className = 'mem-stat-label';
      lbl.textContent = c.label;

      const val = document.createElement('div');
      val.className = 'mem-stat-value';
      val.textContent = c.value;

      const sub = document.createElement('div');
      sub.className = 'mem-stat-sub';
      sub.textContent = c.bytes;

      box.append(lbl, val, sub);
      this.statsGrid.appendChild(box);
    }
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns.length = 0;
  }
}
