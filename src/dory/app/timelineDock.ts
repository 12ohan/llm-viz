/**
 * Editorial Trajectory, Elastic-Depth & Contract A Self-Spec Timeline Dock
 * for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/app/timelineDock.ts`).
 *
 * Exports:
 *   - Pure Node-testable helpers:
 *       * `DORY_PRIMARY_LOOP_CHOICES`, `DORY_EXTRAPOLATION_LOOP_CHOICES`, `DORY_ALL_LOOP_CHOICES`
 *       * `getDoryPromptPreset`
 *       * `formatDoryExecutedDepthLabel`
 *       * `computeDoryBranchOptionsSignature`
 *       * `buildDoryTimelineDockState`
 *   - Accessible DOM dock controller (`createDoryTimelineDock`):
 *       * Strict in-place DOM identity across `update(state)` for all sliders (`scrubber`,
 *         `temperature`), selects (`prompt`, `branch` rebuilt only on signature change),
 *         and toggle buttons — never drops pointer drag or keyboard focus mid-interaction.
 *       * Elastic loop depth `k in {1, 2, 3, 4}` (plus optional base extrapolation `{6, 8}`)
 *         roving-tabindex tablist triggering `D6`/`DR18` immutable branch fork/re-prefill.
 *       * Dual-axis KV policy toggle (`per_loop` vs. `last_loop`) triggering `DR18` clean fork.
 *       * Reconstructed Paper §6.2 Contract A self-speculation mode toggle & sampling controls
 *         (`DR23`: `k = 1 -> 4`, `per_loop`, `104` Dual-Coda banks).
 *       * Live executed-depth, `D10` `3.00 MiB` LRU pool, `<= 12.00 MiB` memory audit & `DR18`
 *         immutable fork notice banner.
 *
 * Strictly `D4`-compliant: ZERO imports from Gate 4b/4c Nano modules, sibling apps, or
 * ambient RNG. Reuses ONLY `src/shared/{a11y,format}.ts`.
 */

import { handleIndexedTablistKeydown, syncRovingTabindex } from '../../shared/a11y';
import { formatMiB } from '../../shared/format';
import { computeEffectivePairedDepth, computeExecutedDepth, computeKvBankCounts } from '../spec';
import {
  DORY_DEFAULT_GREEDY_SAMPLING,
  DORY_MINI_CONFIG,
  DORY_SPEC_CONTRACT_A_BADGE,
  type DoryExecutionMode,
  type DoryKvCacheMode,
  type DorySamplingConfig,
  type DorySession,
  type DorySupportedLoopCount,
} from '../types';
import {
  DORY_DEFAULT_PROMPT_PRESET_ID,
  DORY_PROMPT_PRESETS,
  type DoryPromptPreset,
  type DoryPromptPresetId,
  type DoryTimelineBranchSummary,
  type DoryTimelineDockCallbacks,
  type DoryTimelineDockHandle,
  type DoryTimelineDockState,
} from './types';

// ============================================================================
// PURE NODE-TESTABLE CONSTANTS & STATE BUILDERS
// ============================================================================

export const DORY_PRIMARY_LOOP_CHOICES: readonly [1, 2, 3, 4] = Object.freeze([
  1, 2, 3, 4,
] as const);

export const DORY_EXTRAPOLATION_LOOP_CHOICES: readonly [6, 8] = Object.freeze([6, 8] as const);

export const DORY_ALL_LOOP_CHOICES: readonly DorySupportedLoopCount[] = Object.freeze([
  1, 2, 3, 4, 6, 8,
] as const);

const PROMPT_PRESET_BY_ID: ReadonlyMap<DoryPromptPresetId, DoryPromptPreset> = new Map(
  DORY_PROMPT_PRESETS.map((p) => [p.id, p]),
);

/** Look up a canonical deterministic prompt preset by ID (falls back cleanly to default). */
export function getDoryPromptPreset(presetId: DoryPromptPresetId): DoryPromptPreset {
  return (
    PROMPT_PRESET_BY_ID.get(presetId) ??
    PROMPT_PRESET_BY_ID.get(DORY_DEFAULT_PROMPT_PRESET_ID) ??
    DORY_PROMPT_PRESETS[0]!
  );
}

/**
 * Formats a concise executed-depth & KV-bank telemetry badge strictly matching `D1` physical-72
 * indexing (`32 + 40k` executed sublayers, paired `transformers` correspondence, and logical
 * `36..176` or `104` Contract A bank count).
 */
export function formatDoryExecutedDepthLabel(
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  executionMode: DoryExecutionMode,
): string {
  const execDepth = computeExecutedDepth(kLoops, DORY_MINI_CONFIG);
  const pairedDepth = computeEffectivePairedDepth(kLoops, DORY_MINI_CONFIG);
  const banks = computeKvBankCounts(kLoops, kvCacheMode, executionMode === 'self_spec');
  if (executionMode === 'self_spec') {
    return `${execDepth}L (${pairedDepth} paired) · ${banks.totalBanks} Contract A banks`;
  }
  const modeTag = kvCacheMode === 'per_loop' ? 'per_loop' : 'last_loop';
  return `${execDepth}L (${pairedDepth} paired) · ${banks.totalBanks} ${modeTag} banks`;
}

/**
 * Compact structural signature for `<select>` branch options so `createDoryTimelineDock`
 * preserves existing `<option>` DOM nodes whenever only the active step, cursor, or playback
 * state changes.
 */
export function computeDoryBranchOptionsSignature(
  branches: readonly DoryTimelineBranchSummary[],
): string {
  const parts = new Array<string>(branches.length);
  for (let i = 0; i < branches.length; i += 1) {
    const b = branches[i]!;
    parts[i] =
      `${b.branchId}|${b.label}|${b.forkReason}|${b.kLoops}|${b.kvCacheMode}|` +
      `${b.executionMode}|${b.committedLen}|${b.recordCount}|${b.kvBankCount}|${b.haltReason ?? ''}`;
  }
  return parts.join(';;');
}

function toSupportedLoopCount(kLoops: number): DorySupportedLoopCount {
  if (kLoops === 1 || kLoops === 2 || kLoops === 3 || kLoops === 4 || kLoops === 6 || kLoops === 8) {
    return kLoops;
  }
  return DORY_MINI_CONFIG.defaultRecurrentLoops;
}

export interface BuildDoryTimelineDockStateOptions {
  readonly promptPresetId?: DoryPromptPresetId;
  readonly allowExtrapolationDepths?: boolean;
  readonly isPlaying?: boolean;
  readonly tourActive?: boolean;
  readonly activeRecordId?: string | null;
  readonly dr18ForkNotice?: string | null;
  /** Optional pending sampling override (e.g. unsaved slider change prior to next step fork). */
  readonly sampling?: DorySamplingConfig;
}

/**
 * Pure Node-testable state builder summarizing `DorySession`'s active branch, cursor position,
 * elastic loop budget `k`, KV cache policy, Contract A self-spec state, `DR18` notice, and
 * `D10` `<= 12.00 MiB` memory audit.
 */
export function buildDoryTimelineDockState(
  session: DorySession,
  opts: BuildDoryTimelineDockStateOptions = {},
): DoryTimelineDockState {
  const promptPresetId = opts.promptPresetId ?? DORY_DEFAULT_PROMPT_PRESET_ID;
  const isPlaying = opts.isPlaying === true;
  const tourActive = opts.tourActive === true;

  const activeRecord =
    opts.activeRecordId !== undefined
      ? opts.activeRecordId !== null
        ? session.getRecord(opts.activeRecordId)
        : null
      : session.getActiveRecord();

  const activeBranch =
    activeRecord !== null ? session.getBranch(activeRecord.branchId) : session.getActiveBranch();

  const activeBranchId = activeBranch?.branchId ?? null;
  const activeRecordId = activeRecord?.recordId ?? null;
  const activeRecordKind = activeRecord?.kind ?? null;

  const kLoops = toSupportedLoopCount(
    activeRecord?.kLoops ?? activeBranch?.config.kLoops ?? DORY_MINI_CONFIG.defaultRecurrentLoops,
  );
  const allowExtrapolationDepths =
    opts.allowExtrapolationDepths === true || kLoops > 4;
  const kvCacheMode: DoryKvCacheMode =
    activeRecord?.kvCacheMode ??
    activeBranch?.config.kvCacheMode ??
    DORY_MINI_CONFIG.defaultKvCacheMode;
  const executionMode: DoryExecutionMode =
    activeRecord?.executionMode ?? activeBranch?.config.executionMode ?? 'standard';
  const sampling =
    opts.sampling ??
    activeRecord?.sampling ??
    activeBranch?.config.sampling ??
    DORY_DEFAULT_GREEDY_SAMPLING;

  const recordIds = activeBranch?.recordIds ?? [];
  const totalRecordsInBranch = recordIds.length;
  const rawIdx = activeRecordId !== null ? recordIds.indexOf(activeRecordId) : -1;
  const currentStepIndex = rawIdx >= 0 ? rawIdx : Math.max(0, totalRecordsInBranch - 1);
  const isHistoricalCursor =
    totalRecordsInBranch > 0 && currentStepIndex < totalRecordsInBranch - 1;

  const committedLen = activeRecord?.committedLen ?? activeBranch?.committedLen ?? 0;
  const maxSeqLen = session.params.config.maxPositionEmbeddings;
  const slidingWindow = session.params.config.slidingWindow;
  const haltReason = isHistoricalCursor
    ? (activeRecord?.haltReason ?? null)
    : (activeBranch?.haltReason ?? null);
  const cursorPendingToken = isHistoricalCursor
    ? (activeRecord?.pendingToken ?? null)
    : (activeBranch?.pendingToken ?? null);

  const memoryAudit = session.measureMemory();

  const canStepBack = !tourActive && totalRecordsInBranch > 0 && currentStepIndex > 0;
  const canStepForward =
    !tourActive && totalRecordsInBranch > 0 && currentStepIndex < totalRecordsInBranch - 1;

  const hasRecordBudget = memoryAudit.recordCount < session.bounds.maxRecords;
  const canExecuteStep =
    !tourActive &&
    activeBranch !== null &&
    hasRecordBudget &&
    haltReason === null &&
    cursorPendingToken !== null &&
    committedLen < maxSeqLen;

  const allBranches = session.listBranches();
  const branches: DoryTimelineBranchSummary[] = allBranches.map((b) =>
    Object.freeze({
      branchId: b.branchId,
      label: b.label,
      forkReason: b.forkReason,
      kLoops: b.config.kLoops,
      kvCacheMode: b.config.kvCacheMode,
      executionMode: b.config.executionMode,
      committedLen: b.committedLen,
      recordCount: b.recordIds.length,
      kvBankCount: b.kvBankCount,
      kvBytes: b.kvBytes,
      haltReason: b.haltReason,
      isActive: b.branchId === activeBranchId,
    }),
  );

  const executedDepthLabel = formatDoryExecutedDepthLabel(kLoops, kvCacheMode, executionMode);

  return Object.freeze({
    promptPresetId,
    kLoops,
    allowExtrapolationDepths,
    kvCacheMode,
    executionMode,
    sampling,
    activeBranchId,
    activeRecordId,
    activeRecordKind,
    currentStepIndex,
    totalRecordsInBranch,
    committedLen,
    maxSeqLen,
    slidingWindow,
    canStepBack,
    canStepForward,
    canExecuteStep,
    haltReason,
    isHistoricalCursor,
    isPlaying,
    tourActive,
    executedDepthLabel,
    dr18ForkNotice: opts.dr18ForkNotice ?? null,
    branches: Object.freeze(branches),
    memoryAudit,
  });
}

// ============================================================================
// ACCESSIBLE EDITORIAL DOM TIMELINE DOCK CONTROLLER
// ============================================================================

function makeEl<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function makeBtn(
  className: string,
  text: string,
  testId: string,
  ariaLabel?: string,
  title?: string,
): HTMLButtonElement {
  const btn = makeEl('button', className, text);
  btn.type = 'button';
  btn.dataset.testid = testId;
  if (ariaLabel) btn.setAttribute('aria-label', ariaLabel);
  if (title) btn.title = title;
  return btn;
}

function syncRangeInputInPlace(input: HTMLInputElement, nextValue: string): void {
  if (typeof document !== 'undefined' && document.activeElement === input) {
    return;
  }
  if (input.value !== nextValue) {
    input.value = nextValue;
  }
}

/**
 * Mounts the accessible Dory1.0-4B editorial bottom timeline dock into `container`.
 * Preserves exact DOM node identity across every `update(state)` call for all sliders,
 * tablists, toggles, and selects (rebuilding branch `<option>`s strictly when the branch
 * signature changes).
 */
export function createDoryTimelineDock(
  container: HTMLElement,
  callbacks: DoryTimelineDockCallbacks,
): DoryTimelineDockHandle {
  const root = makeEl('section', 'dory-timeline-dock');
  root.dataset.testid = 'dory-timeline-dock';
  root.setAttribute(
    'aria-label',
    'Dory1.0-4B elastic loop depth, dual-axis KV policy, Contract A self-spec, and trajectory dock',
  );

  // --------------------------------------------------------------------------
  // ROW 1: ELASTIC LOOP DEPTH `k` (DR18), KV POLICY, CONTRACT A SELF-SPEC & PROMPT
  // --------------------------------------------------------------------------

  const primaryRow = makeEl('div', 'dory-dock-row dory-dock-row--config');

  // 1a. Elastic Loop Depth `k` Tablist (`1, 2, 3, 4` + optional base extrapolation `6, 8`)
  const depthWrap = makeEl('div', 'dory-dock-group dory-dock-depth-group');
  const depthCaption = makeEl('span', 'dory-dock-group-label', 'Loops k');

  const kTablist = makeEl('div', 'dory-dock-segmented dory-dock-k-tablist');
  kTablist.setAttribute('role', 'tablist');
  kTablist.setAttribute('aria-label', 'Recurrent loop depth k (DR18 immutable branch fork)');

  const kButtons: HTMLButtonElement[] = [];
  let visibleKChoices: DorySupportedLoopCount[] = [...DORY_PRIMARY_LOOP_CHOICES];
  let activeVisibleKIdx = 1; // default k=2

  for (let i = 0; i < DORY_ALL_LOOP_CHOICES.length; i += 1) {
    const kVal = DORY_ALL_LOOP_CHOICES[i]!;
    const isExtrap = kVal > 4;
    const btn = makeBtn(
      `dory-dock-seg-btn dory-dock-k-btn${isExtrap ? ' dory-dock-k-btn--extrap' : ''}`,
      `k=${kVal}`,
      `dory-dock-k-${kVal}`,
      isExtrap
        ? `Base-model extrapolation loop depth k=${kVal} (triggers DR18 immutable fork)`
        : `Trained loop depth k=${kVal} (triggers DR18 immutable fork)`,
      isExtrap
        ? `Base extrapolation k=${kVal} (${computeExecutedDepth(kVal)}L executed) — forks a clean DR18 branch`
        : `Trained loop depth k=${kVal} (${computeExecutedDepth(kVal)}L executed) — forks a clean DR18 branch`,
    );
    btn.setAttribute('role', 'tab');
    btn.dataset.kLoops = String(kVal);
    if (isExtrap) {
      btn.hidden = true;
    }
    btn.addEventListener('click', () => {
      callbacks.onSelectKLoops(kVal);
    });
    btn.addEventListener('keydown', (ev) => {
      const visibleBtns = kButtons.filter((b) => !b.hidden);
      const curIdx = visibleBtns.indexOf(btn);
      handleIndexedTablistKeydown(
        ev,
        visibleBtns,
        curIdx >= 0 ? curIdx : activeVisibleKIdx,
        (nextIdx) => {
          const nextK = visibleKChoices[nextIdx];
          if (nextK !== undefined) {
            activeVisibleKIdx = nextIdx;
            callbacks.onSelectKLoops(nextK);
          }
        },
      );
    });
    kButtons.push(btn);
    kTablist.appendChild(btn);
  }
  syncRovingTabindex(kButtons.slice(0, 4), 1);

  const extrapLabel = makeEl('label', 'dory-dock-extrap-toggle');
  extrapLabel.title =
    'Show base-model extrapolation loop budgets k=6 and k=8 (bounded <= 12 MiB by D10 3.00 MiB LRU pool)';
  const extrapCheckbox = makeEl('input');
  extrapCheckbox.type = 'checkbox';
  extrapCheckbox.dataset.testid = 'dory-dock-extrap-toggle';
  extrapCheckbox.setAttribute('aria-label', 'Show base extrapolation depths k=6 and k=8');
  extrapCheckbox.addEventListener('change', () => {
    callbacks.onToggleExtrapolationDepths(extrapCheckbox.checked);
  });
  extrapLabel.append(extrapCheckbox, makeEl('span', undefined, '+6,8'));

  depthWrap.append(depthCaption, kTablist, extrapLabel);

  // 1b. Dual-Axis KV Cache Policy Toggle (`per_loop` vs. `last_loop`, DR17/DR18)
  const kvGroup = makeEl('div', 'dory-dock-segmented dory-dock-kv-group');
  kvGroup.setAttribute('role', 'group');
  kvGroup.setAttribute('aria-label', 'KV cache policy (DR18 immutable branch fork)');

  const perLoopBtn = makeBtn(
    'dory-dock-seg-btn dory-dock-toggle-btn',
    'per_loop',
    'dory-dock-kv-per_loop',
    'per_loop KV cache policy (16 + 20k independent banks; partition-invariant)',
    'Retains independent KV banks for every recurrent loop t=1..k (triggers DR18 clean fork on switch)',
  );
  perLoopBtn.addEventListener('click', () => callbacks.onSelectKvCacheMode('per_loop'));

  const lastLoopBtn = makeBtn(
    'dory-dock-seg-btn dory-dock-toggle-btn',
    'last_loop',
    'dory-dock-kv-last_loop',
    'last_loop KV cache policy (36 frozen banks; partition-dependent for k >= 2 per DR17)',
    'Freezes KV footprint at 36 banks by committing final-loop KV only (partition-dependent for k >= 2, DR17; triggers DR18 clean fork)',
  );
  lastLoopBtn.addEventListener('click', () => callbacks.onSelectKvCacheMode('last_loop'));

  kvGroup.append(perLoopBtn, lastLoopBtn);

  // 1c. Contract A Self-Speculative Execution Mode Toggle (`standard` vs. `self_spec`, DR23)
  const modeGroup = makeEl('div', 'dory-dock-segmented dory-dock-mode-group');
  modeGroup.setAttribute('role', 'group');
  modeGroup.setAttribute('aria-label', 'Execution mode: standard decode or Contract A self-speculation');

  const standardModeBtn = makeBtn(
    'dory-dock-seg-btn dory-dock-toggle-btn',
    'Standard',
    'dory-dock-mode-standard',
    'Standard autoregressive decode at selected loop depth k',
    'Standard 1-token causal decode at active k and KV policy',
  );
  standardModeBtn.addEventListener('click', () => callbacks.onSelectExecutionMode('standard'));

  const selfSpecModeBtn = makeBtn(
    'dory-dock-seg-btn dory-dock-toggle-btn dory-dock-toggle-btn--spec',
    'Self-Spec §6.2',
    'dory-dock-mode-self_spec',
    'Contract A Paper §6.2 self-speculative decoding (k=1 draft gamma=4 -> k=4 verify, 104 Dual-Coda banks, per_loop only)',
    DORY_SPEC_CONTRACT_A_BADGE,
  );
  selfSpecModeBtn.addEventListener('click', () => callbacks.onSelectExecutionMode('self_spec'));

  modeGroup.append(standardModeBtn, selfSpecModeBtn);

  // 1d. Sampling Temperature Slider (In-Place DOM Identity — supports Greedy T=0 & Stochastic Contract A!)
  const tempWrap = makeEl('div', 'dory-dock-group dory-dock-temp-group');
  const tempLabel = makeEl('span', 'dory-dock-temp-label', 'T=0.00 (Greedy)');
  tempLabel.dataset.testid = 'dory-dock-temp-readout';

  const tempSlider = makeEl('input', 'dory-dock-temp-slider');
  tempSlider.type = 'range';
  tempSlider.min = '0';
  tempSlider.max = '1.5';
  tempSlider.step = '0.05';
  tempSlider.value = '0';
  tempSlider.dataset.testid = 'dory-dock-temp-slider';
  tempSlider.setAttribute('aria-label', 'Sampling temperature (0.00 greedy to 1.50 stochastic)');
  tempSlider.addEventListener('input', () => {
    const parsed = Number.parseFloat(tempSlider.value);
    if (!Number.isFinite(parsed)) return;
    const clamped = Math.max(0, Math.min(1.5, Math.round(parsed * 20) / 20));
    tempLabel.textContent =
      clamped === 0 ? 'T=0.00 (Greedy)' : `T=${clamped.toFixed(2)} (Stoch)`;
    callbacks.onUpdateSampling({
      temperature: clamped,
      topK: 0,
      topP: clamped === 0 ? 1.0 : 0.95,
    });
  });
  tempWrap.append(tempLabel, tempSlider);

  // 1e. Deterministic Prompt Preset Selector (Created Once)
  const promptSelect = makeEl('select', 'dory-dock-select dory-dock-prompt-select');
  promptSelect.dataset.testid = 'dory-dock-prompt-select';
  promptSelect.setAttribute('aria-label', 'Deterministic miniature prompt preset');
  for (let i = 0; i < DORY_PROMPT_PRESETS.length; i += 1) {
    const preset = DORY_PROMPT_PRESETS[i]!;
    const opt = makeEl('option', undefined, preset.shortLabel);
    opt.value = preset.id;
    opt.title = preset.description;
    promptSelect.appendChild(opt);
  }
  promptSelect.value = DORY_DEFAULT_PROMPT_PRESET_ID;
  promptSelect.addEventListener('change', () => {
    const nextId = promptSelect.value as DoryPromptPresetId;
    if (PROMPT_PRESET_BY_ID.has(nextId)) {
      callbacks.onSelectPromptPreset(nextId);
    }
  });

  primaryRow.append(depthWrap, kvGroup, modeGroup, tempWrap, promptSelect);

  // --------------------------------------------------------------------------
  // ROW 2: TRAJECTORY TRANSPORT, SCRUBBER, BRANCH SELECT/FORK & LIVE TELEMETRY
  // --------------------------------------------------------------------------

  const transportRow = makeEl('div', 'dory-dock-row dory-dock-row--transport');

  // 2a. Transport & Scrubber (In-Place DOM Identity)
  const transportGroup = makeEl('div', 'dory-dock-transport');

  const stepBackBtn = makeBtn(
    'dory-dock-btn',
    '◀',
    'dory-dock-step-back',
    'Step backward in active branch',
  );
  stepBackBtn.addEventListener('click', () => callbacks.onStepBack());

  const playBtn = makeBtn(
    'dory-dock-btn',
    '▶ Play',
    'dory-dock-play-pause',
    'Play or pause trajectory replay',
  );
  playBtn.setAttribute('aria-pressed', 'false');
  playBtn.addEventListener('click', () => callbacks.onTogglePlay());

  const stepFwdBtn = makeBtn(
    'dory-dock-btn',
    '▶',
    'dory-dock-step-forward',
    'Step forward in active branch',
  );
  stepFwdBtn.addEventListener('click', () => callbacks.onStepForward());

  const scrubber = makeEl('input', 'dory-dock-scrubber');
  scrubber.type = 'range';
  scrubber.min = '0';
  scrubber.max = '0';
  scrubber.value = '0';
  scrubber.dataset.testid = 'dory-dock-scrubber';
  scrubber.setAttribute('aria-label', 'Branch trajectory step scrubber');
  scrubber.addEventListener('input', () => {
    const idx = Number.parseInt(scrubber.value, 10);
    if (Number.isFinite(idx)) {
      callbacks.onScrubIndex(idx);
    }
  });

  const stepReadout = makeEl('span', 'dory-dock-step-readout', 'Step 0 / 0');
  stepReadout.dataset.testid = 'dory-dock-step-readout';

  const executeBtn = makeBtn(
    'dory-dock-btn dory-dock-btn--primary',
    '+ Step',
    'dory-dock-execute-step',
    'Execute next decode step or Contract A self-speculative cycle',
  );
  executeBtn.addEventListener('click', () => callbacks.onExecuteNextStep());

  transportGroup.append(stepBackBtn, playBtn, stepFwdBtn, scrubber, stepReadout, executeBtn);

  // 2b. Immutable Branch Switcher, Fork & Reset
  const branchGroup = makeEl('div', 'dory-dock-branches');

  const branchSelect = makeEl('select', 'dory-dock-select dory-dock-branch-select');
  branchSelect.dataset.testid = 'dory-dock-branch-select';
  branchSelect.setAttribute('aria-label', 'Active immutable session branch');
  branchSelect.addEventListener('change', () => {
    if (branchSelect.value) {
      callbacks.onSwitchBranch(branchSelect.value);
    }
  });

  const forkBtn = makeBtn(
    'dory-dock-btn',
    'Fork',
    'dory-dock-fork-branch',
    'Fork an immutable copy-on-write branch at current cursor',
  );
  forkBtn.addEventListener('click', () => callbacks.onForkBranch());

  const resetBtn = makeBtn(
    'dory-dock-btn',
    'Reset',
    'dory-dock-reset-session',
    'Reset session to a fresh branch from the active prompt preset',
  );
  resetBtn.addEventListener('click', () => callbacks.onResetSession());

  branchGroup.append(branchSelect, forkBtn, resetBtn);

  // 2c. Live Depth, KV & D10 Memory Telemetry Pills
  const telemetryGroup = makeEl('div', 'dory-dock-telemetry');

  const depthPill = makeEl(
    'span',
    'dory-dock-pill dory-dock-pill--depth',
    '112L (56 paired) · 56 per_loop banks',
  );
  depthPill.dataset.testid = 'dory-dock-depth-pill';

  const memoryPill = makeEl(
    'span',
    'dory-dock-pill dory-dock-pill--memory',
    '0.00 / 12.00 MiB',
  );
  memoryPill.dataset.testid = 'dory-dock-memory-pill';

  telemetryGroup.append(depthPill, memoryPill);

  transportRow.append(transportGroup, branchGroup, telemetryGroup);

  // --------------------------------------------------------------------------
  // OPTIONAL DR18 IMMUTABLE FORK NOTICE BAR (IN-PLACE STATUS)
  // --------------------------------------------------------------------------

  const dr18NoticeEl = makeEl('div', 'dory-dock-fork-notice dory-dock-dr18-notice');
  dr18NoticeEl.dataset.testid = 'dory-dock-dr18-notice';
  dr18NoticeEl.setAttribute('role', 'status');
  dr18NoticeEl.setAttribute('aria-live', 'polite');
  dr18NoticeEl.hidden = true;

  root.append(primaryRow, transportRow, dr18NoticeEl);
  container.appendChild(root);

  let disposed = false;
  let lastBranchSignature = '';

  function update(state: DoryTimelineDockState): void {
    if (disposed) return;

    root.classList.toggle('is-tour-locked', state.tourActive);
    root.classList.toggle('is-self-spec', state.executionMode === 'self_spec');

    // 1. Elastic loop depth tablist (show k=6,8 only when allowExtrapolationDepths is active)
    const showExtrap = state.allowExtrapolationDepths || state.kLoops > 4;
    extrapCheckbox.checked = showExtrap;
    extrapCheckbox.disabled = state.tourActive;

    visibleKChoices = showExtrap ? [...DORY_ALL_LOOP_CHOICES] : [...DORY_PRIMARY_LOOP_CHOICES];
    const visibleButtons: HTMLButtonElement[] = [];
    for (let i = 0; i < DORY_ALL_LOOP_CHOICES.length; i += 1) {
      const kVal = DORY_ALL_LOOP_CHOICES[i]!;
      const btn = kButtons[i]!;
      const hidden = kVal > 4 && !showExtrap;
      btn.hidden = hidden;
      btn.disabled = state.tourActive || hidden;
      btn.classList.toggle('is-active', kVal === state.kLoops);
      if (!hidden) {
        visibleButtons.push(btn);
      }
    }
    const foundKIdx = visibleKChoices.indexOf(state.kLoops);
    activeVisibleKIdx = foundKIdx >= 0 ? foundKIdx : 0;
    syncRovingTabindex(visibleButtons, activeVisibleKIdx);

    // 2. Dual-axis KV cache policy buttons (`per_loop` vs. `last_loop`)
    const isPerLoop = state.kvCacheMode === 'per_loop';
    perLoopBtn.classList.toggle('is-active', isPerLoop);
    perLoopBtn.setAttribute('aria-pressed', isPerLoop ? 'true' : 'false');
    perLoopBtn.disabled = state.tourActive;

    lastLoopBtn.classList.toggle('is-active', !isPerLoop);
    lastLoopBtn.setAttribute('aria-pressed', !isPerLoop ? 'true' : 'false');
    lastLoopBtn.disabled = state.tourActive;

    // 3. Execution mode buttons (`standard` vs. Contract A `self_spec`)
    const isSpec = state.executionMode === 'self_spec';
    standardModeBtn.classList.toggle('is-active', !isSpec);
    standardModeBtn.setAttribute('aria-pressed', !isSpec ? 'true' : 'false');
    standardModeBtn.disabled = state.tourActive;

    selfSpecModeBtn.classList.toggle('is-active', isSpec);
    selfSpecModeBtn.setAttribute('aria-pressed', isSpec ? 'true' : 'false');
    selfSpecModeBtn.disabled = state.tourActive;

    // 4. Sampling temperature slider (preserved strictly in-place)
    const tempVal = Number.isFinite(state.sampling.temperature)
      ? Math.max(0, Math.min(1.5, state.sampling.temperature))
      : 0;
    tempSlider.disabled = state.tourActive;
    syncRangeInputInPlace(tempSlider, String(tempVal));
    tempLabel.textContent =
      tempVal === 0 ? 'T=0.00 (Greedy)' : `T=${tempVal.toFixed(2)} (Stoch)`;

    // 5. Prompt preset selector
    promptSelect.disabled = state.tourActive;
    if (promptSelect.value !== state.promptPresetId) {
      promptSelect.value = state.promptPresetId;
    }

    // 6. Transport buttons & trajectory scrubber (preserved strictly in-place)
    stepBackBtn.disabled = state.tourActive || !state.canStepBack;
    stepFwdBtn.disabled = state.tourActive || !state.canStepForward;
    playBtn.disabled = state.tourActive || state.totalRecordsInBranch <= 1;
    playBtn.textContent = state.isPlaying ? '⏸ Pause' : '▶ Play';
    playBtn.setAttribute('aria-pressed', state.isPlaying ? 'true' : 'false');

    const maxIdx = Math.max(0, state.totalRecordsInBranch - 1);
    const clampedIdx = Math.max(0, Math.min(maxIdx, state.currentStepIndex));
    if (scrubber.max !== String(maxIdx)) {
      scrubber.max = String(maxIdx);
    }
    syncRangeInputInPlace(scrubber, String(clampedIdx));
    scrubber.disabled = state.tourActive || state.totalRecordsInBranch <= 1;

    const displayCur = state.totalRecordsInBranch > 0 ? clampedIdx + 1 : 0;
    const histSuffix = state.isHistoricalCursor ? ' · Hist' : '';
    stepReadout.textContent = `Step ${displayCur} / ${state.totalRecordsInBranch}${histSuffix}`;

    executeBtn.disabled = state.tourActive || !state.canExecuteStep;
    executeBtn.textContent = isSpec
      ? state.haltReason
        ? `Halted (${state.haltReason})`
        : '+ Spec Cycle (γ=4)'
      : state.haltReason
        ? `Halted (${state.haltReason})`
        : '+ Step';

    // 7. Branch switcher `<select>`: rebuild `<option>` elements ONLY when signature changes
    const nextBranchSig = computeDoryBranchOptionsSignature(state.branches);
    if (nextBranchSig !== lastBranchSignature) {
      lastBranchSignature = nextBranchSig;
      branchSelect.innerHTML = '';
      for (let i = 0; i < state.branches.length; i += 1) {
        const b = state.branches[i]!;
        const opt = document.createElement('option');
        opt.value = b.branchId;
        opt.textContent = `${b.label} · ${b.committedLen}tok (${b.recordCount} rec)`;
        branchSelect.appendChild(opt);
      }
    }
    const targetBranchVal = state.activeBranchId ?? '';
    if (branchSelect.value !== targetBranchVal) {
      branchSelect.value = targetBranchVal;
    }
    branchSelect.disabled = state.tourActive || state.branches.length === 0;

    forkBtn.disabled = state.tourActive || state.activeRecordId === null;
    resetBtn.disabled = state.tourActive;

    // 8. Telemetry pills (`D1` depth/banks + `D10` `<= 12.00 MiB` memory + LRU pool)
    const evictTag =
      state.committedLen > state.slidingWindow
        ? ` · SWA evicted ${state.committedLen - state.slidingWindow}`
        : '';
    depthPill.textContent =
      `${state.executedDepthLabel} · pos ${state.committedLen}/${state.maxSeqLen} (W=${state.slidingWindow}${evictTag})`;
    depthPill.classList.toggle('is-extrap', state.kLoops > 4);
    depthPill.classList.toggle('is-spec', isSpec);

    const audit = state.memoryAudit;
    const usedLabel = formatMiB(audit.totalWithWeightsBytes, 2);
    const capLabel = formatMiB(audit.maxSessionRetainedBytes, 2);
    const lruLabel = formatMiB(audit.checkpointPoolBytes, 2);
    memoryPill.textContent =
      `${usedLabel} / ${capLabel} · ${audit.branchCount} br · ${audit.recordCount} rec · LRU ${lruLabel}/3.00 MiB`;
    memoryPill.classList.toggle(
      'is-over-budget',
      !audit.withinTotalBudgetWithWeights || !audit.withinCheckpointPoolBudget,
    );

    // 9. DR18 fork notice status banner
    if (state.dr18ForkNotice && state.dr18ForkNotice.length > 0) {
      dr18NoticeEl.hidden = false;
      dr18NoticeEl.textContent = state.dr18ForkNotice;
    } else {
      dr18NoticeEl.hidden = true;
      dr18NoticeEl.textContent = '';
    }
  }

  function getRenderedBranchSignature(): string {
    return lastBranchSignature;
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    if (root.parentNode === container) {
      container.removeChild(root);
    }
  }

  return {
    root,
    update,
    getRenderedBranchSignature,
    dispose,
  };
}
