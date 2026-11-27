/**
 * Editorial Trajectory & Elastic-MoT Timeline Dock (`src/uni/app/timelineDock.ts`)
 *
 * Exports:
 *   - Pure Node-testable helpers:
 *       * `UNI_DOCK_TASK_CHOICES`
 *       * `taskChoiceToSessionTaskMode`
 *       * `sessionTaskModeToDockChoice`
 *       * `buildUniTimelineDockState`
 *   - Accessible DOM dock controller (`createUniTimelineDock`):
 *       * 4-task switcher tablist (using localized `a11yHelpers.ts`)
 *       * Step back/forward, play/pause, trajectory range scrubber, step-execute button
 *       * Branch `<select>` + branch fork button
 *       * `D5` `HF-Faithful` vs `Elastic-Pruned` prefill toggle
 *       * Live memory & layer/KV telemetry pills
 *
 * Strictly zero imports from `src/ui/controls.ts` (D4), `src/twotower/`, or `src/diffusion/`.
 */

import { buildUniElasticD5Comparison } from '../archReadouts';
import type { UniSession } from '../session';
import { formatSiIecBytes } from '../spec';
import {
  UNI_MINI_CONFIG,
  UNI_MINI_EOS_TOKEN_ID,
  UNI_MINI_MASK_TOKEN_ID,
  type UniEditGuidanceMode,
  type UniElasticPrefillMode,
  type UniExecutionViewId,
  type UniGainPresetId,
  type UniScenePresetId,
  type UniSessionBranch,
  type UniSessionRecord,
  type UniSessionTaskMode,
} from '../types';
import { handleUniTablistKeydown, syncRovingTabindex } from './a11yHelpers';
import type {
  UniDockTaskChoice,
  UniTimelineBranchSummary,
  UniTimelineDockCallbacks,
  UniTimelineDockState,
} from './types';

export const UNI_DOCK_TASK_CHOICES: readonly {
  readonly id: UniDockTaskChoice;
  readonly label: string;
  readonly shortLabel: string;
  readonly taskMode: UniSessionTaskMode;
}[] = Object.freeze([
  Object.freeze({
    id: 'understanding',
    label: 'Visual Understanding',
    shortLabel: 'Understand',
    taskMode: 'visual_understanding',
  }),
  Object.freeze({
    id: 'grounding',
    label: 'Object Grounding (<LOC>)',
    shortLabel: 'Ground',
    taskMode: 'object_grounding',
  }),
  Object.freeze({
    id: 't2i',
    label: 'Text-to-Image (T2I)',
    shortLabel: 'T2I',
    taskMode: 'text_to_image',
  }),
  Object.freeze({
    id: 'image_edit',
    label: 'Dual-CFG Image Edit',
    shortLabel: 'Edit',
    taskMode: 'image_edit',
  }),
]);

export function taskChoiceToSessionTaskMode(choice: UniDockTaskChoice): UniSessionTaskMode {
  switch (choice) {
    case 'understanding':
      return 'visual_understanding';
    case 'grounding':
      return 'object_grounding';
    case 't2i':
      return 'text_to_image';
    case 'image_edit':
      return 'image_edit';
  }
}

export function sessionTaskModeToDockChoice(mode: UniSessionTaskMode): UniDockTaskChoice {
  switch (mode) {
    case 'text_understanding':
    case 'visual_understanding':
      return 'understanding';
    case 'object_grounding':
      return 'grounding';
    case 'text_to_image':
      return 't2i';
    case 'image_edit':
      return 'image_edit';
  }
}

const UNI_DEFAULT_UNDERSTANDING_MAX_NEW_TOKENS = 8;

/**
 * Compact structural signature for `<select>` branch options so `createUniTimelineDock`
 * avoids clearing/rebuilding `<option>` nodes when only the active step or playback changes.
 */
export function computeBranchOptionsSignature(
  branches: readonly UniTimelineBranchSummary[],
): string {
  const parts = new Array<string>(branches.length);
  for (let i = 0; i < branches.length; i++) {
    const b = branches[i]!;
    parts[i] = `${b.branchId}|${b.label}|${b.recordCount}`;
  }
  return parts.join(';;');
}

function resolveBranchInitialPromptLen(
  session: UniSession,
  branch: UniSessionBranch,
): number {
  let cursorBranch: UniSessionBranch | undefined = branch;
  let hops = 0;
  while (cursorBranch && hops < 16) {
    if (cursorBranch.recordIds.length > 0) {
      const firstRec = session.getRecord(cursorBranch.recordIds[0]!);
      if (firstRec && firstRec.kind === 'prefill') {
        return firstRec.routing.length;
      }
    }
    if (!cursorBranch.parentBranchId) break;
    cursorBranch = session.getBranch(cursorBranch.parentBranchId);
    hops += 1;
  }
  if (branch.recordIds.length > 0) {
    const firstRec = session.getRecord(branch.recordIds[0]!);
    if (firstRec) return firstRec.positionIds[0] ?? 0;
  }
  return 0;
}

function canBranchExecuteAnotherStep(
  session: UniSession,
  branch: UniSessionBranch | undefined,
  lastRecord: UniSessionRecord | undefined,
  tourActive: boolean,
  maxRecords: number,
  totalSessionRecords: number,
  opts?: {
    readonly maxNewTokens?: number;
    readonly branchFinished?: boolean;
  },
): boolean {
  if (!branch || tourActive) return false;
  if (opts?.branchFinished === true) return false;
  if (totalSessionRecords >= maxRecords) return false;

  // Respect explicit completion signals if exposed by UniSession or a branch wrapper
  const maybeFinishedFn = (
    session as unknown as { isBranchFinished?: (branchId: string) => boolean }
  ).isBranchFinished;
  if (typeof maybeFinishedFn === 'function' && maybeFinishedFn.call(session, branch.branchId)) {
    return false;
  }
  if ((branch as unknown as { finished?: boolean }).finished === true) {
    return false;
  }

  if (branch.taskMode === 'text_to_image' || branch.taskMode === 'image_edit') {
    const totalRounds = branch.schedule?.totalRounds ?? 0;
    if (totalRounds <= 0) return false;
    if (lastRecord?.kind === 'image_denoise' && lastRecord.imageState) {
      return lastRecord.imageState.roundIndex + 1 < totalRounds;
    }
    return branch.recordIds.length < 1 + totalRounds;
  }

  if (branch.taskMode === 'object_grounding') {
    if (!lastRecord || lastRecord.kind === 'prefill') return true;
    const tokens = lastRecord.textState?.tokenIds;
    if (!tokens) return false;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === UNI_MINI_MASK_TOKEN_ID) return true;
    }
    return false;
  }

  // `text_understanding` | `visual_understanding`
  const effectiveMaxNewTokens = opts?.maxNewTokens ?? UNI_DEFAULT_UNDERSTANDING_MAX_NEW_TOKENS;
  if (effectiveMaxNewTokens <= 0) return false;

  const initialPromptLen = resolveBranchInitialPromptLen(session, branch);
  const generatedTokens = Math.max(0, branch.prefixCache.seqLen - initialPromptLen);

  if (!lastRecord || lastRecord.kind === 'prefill') {
    const defaultBlockLen = 4;
    return (
      generatedTokens < effectiveMaxNewTokens &&
      branch.prefixCache.seqLen + defaultBlockLen <= branch.prefixCache.capacity
    );
  }

  const tokens = lastRecord.textState?.tokenIds;
  const blockLength = tokens && tokens.length > 0 ? tokens.length : 4;
  let hasRemainingMask = false;
  if (tokens) {
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === UNI_MINI_EOS_TOKEN_ID) return false;
      if (tokens[i] === UNI_MINI_MASK_TOKEN_ID) {
        hasRemainingMask = true;
      }
    }
  }

  // Mid-block denoising steps still have [MASK] slots to unmask within the active block
  if (hasRemainingMask) {
    return true;
  }

  // Block just committed: check maxNewTokens budget, partial-block budget truncation, and KV room
  if (generatedTokens >= effectiveMaxNewTokens) {
    return false;
  }
  const blockEndPos = (lastRecord.positionIds[0] ?? initialPromptLen) + blockLength;
  if (blockEndPos > branch.prefixCache.seqLen) {
    // Commit cropped the block short (EOS or custom maxNewTokens < blockLength boundary)
    return false;
  }
  if (branch.prefixCache.seqLen + blockLength > branch.prefixCache.capacity) {
    return false;
  }
  return true;
}

function buildLayerTelemetrySummary(
  executionViewId: UniExecutionViewId,
  prefillMode: UniElasticPrefillMode,
  branch: UniSessionBranch | undefined,
): string {
  const prefixTokens = Math.max(0, branch?.prefixCache.seqLen ?? 0);
  const d5 = buildUniElasticD5Comparison(executionViewId, prefixTokens);
  const wastedPct = (d5.wastedPrefixKvFraction * 100).toFixed(1);

  if (executionViewId === 'text_to_image' || executionViewId === 'image_edit') {
    if (prefillMode === 'elastic_pruned') {
      return `${d5.elasticPrunedLayersExecuted}L U + 34L G · KV ${prefixTokens} tok (${formatSiIecBytes(d5.miniElasticPrunedPrefixKvBytes)}, 0 wasted)`;
    }
    return `${d5.hfFaithfulLayersExecuted}L U + 34L G · KV ${prefixTokens} tok (${wastedPct}% split-U waste)`;
  }

  return `34L U · KV ${prefixTokens}/${UNI_MINI_CONFIG.kvCapacity} tok · B_cfg=${d5.cfgBranches}`;
}

/**
 * Pure Node-testable state builder summarizing the active `UniSession` trajectory,
 * branches, `D5` prefill mode, `Set<ArrayBufferLike>` memory audit, and layer/KV telemetry.
 */
export function buildUniTimelineDockState(
  session: UniSession,
  opts?: {
    readonly isPlaying?: boolean;
    readonly tourActive?: boolean;
    readonly activeRecordId?: string | null;
    readonly maxNewTokens?: number;
    readonly branchFinished?: boolean;
  },
): UniTimelineDockState {
  const isPlaying = opts?.isPlaying === true;
  const tourActive = opts?.tourActive === true;
  const activeRecordId =
    opts?.activeRecordId !== undefined ? opts.activeRecordId : session.getActiveRecordId();

  const activeRecord = activeRecordId ? session.getRecord(activeRecordId) : undefined;
  const activeBranchId = activeRecord ? activeRecord.branchId : session.getActiveBranchId();
  const activeBranch = activeBranchId ? session.getBranch(activeBranchId) : undefined;

  const taskMode: UniSessionTaskMode = activeBranch?.taskMode ?? 'text_to_image';
  const taskChoice = sessionTaskModeToDockChoice(taskMode);
  const executionViewId: UniExecutionViewId = taskMode;
  const prefillMode: UniElasticPrefillMode = activeBranch?.prefillMode ?? 'elastic_pruned';
  const gainPresetId: UniGainPresetId =
    activeBranch?.gainPresetId ?? UNI_MINI_CONFIG.defaultGainPreset;
  const scenePresetId: UniScenePresetId =
    activeBranch?.scenePresetId ??
    (taskMode === 'image_edit' ? 'harbor_panorama_16x8' : 'sunlit_cat_16x16');
  const editGuidanceMode: UniEditGuidanceMode = activeBranch?.guidanceMode ?? 'legacy';

  const recordIds = activeBranch?.recordIds ?? [];
  const totalRecordsInBranch = recordIds.length;
  const rawIdx = activeRecordId ? recordIds.indexOf(activeRecordId) : -1;
  const currentStepIndex = rawIdx >= 0 ? rawIdx : Math.max(0, totalRecordsInBranch - 1);

  const canStepBack = !tourActive && totalRecordsInBranch > 0 && currentStepIndex > 0;
  const canStepForward =
    !tourActive && totalRecordsInBranch > 0 && currentStepIndex < totalRecordsInBranch - 1;

  const lastRecordId =
    totalRecordsInBranch > 0 ? recordIds[totalRecordsInBranch - 1]! : undefined;
  const lastRecord = lastRecordId ? session.getRecord(lastRecordId) : undefined;
  const memoryAudit = session.auditMemory();

  const canExecuteStep = canBranchExecuteAnotherStep(
    session,
    activeBranch,
    lastRecord,
    tourActive,
    memoryAudit.maxRecords,
    memoryAudit.recordCount,
    {
      ...(opts?.maxNewTokens !== undefined ? { maxNewTokens: opts.maxNewTokens } : {}),
      ...(opts?.branchFinished !== undefined ? { branchFinished: opts.branchFinished } : {}),
    },
  );

  const allBranches = session.getAllBranches();
  const branches: UniTimelineBranchSummary[] = allBranches.map((b) =>
    Object.freeze({
      branchId: b.branchId,
      label: b.label,
      taskMode: b.taskMode,
      prefillMode: b.prefillMode,
      recordCount: b.recordIds.length,
      isActive: b.branchId === activeBranchId,
    }),
  );

  const layerSummary = buildLayerTelemetrySummary(executionViewId, prefillMode, activeBranch);

  return Object.freeze({
    taskChoice,
    taskMode,
    executionViewId,
    prefillMode,
    gainPresetId,
    scenePresetId,
    editGuidanceMode,
    activeBranchId,
    activeRecordId,
    currentStepIndex,
    totalRecordsInBranch,
    canStepBack,
    canStepForward,
    canExecuteStep,
    isPlaying,
    tourActive,
    branches: Object.freeze(branches),
    memoryAudit,
    layerSummary,
  });
}

function makeButton(
  className: string,
  text: string,
  testId: string,
  ariaLabel?: string,
): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = className;
  btn.textContent = text;
  btn.dataset.testid = testId;
  if (ariaLabel) btn.setAttribute('aria-label', ariaLabel);
  return btn;
}

/**
 * Mount the accessible editorial bottom timeline dock into `container`, exposing
 * `update(state)` and `dispose()`.
 */
export function createUniTimelineDock(
  container: HTMLElement,
  callbacks: UniTimelineDockCallbacks,
): { update(state: UniTimelineDockState): void; dispose(): void } {
  const root = document.createElement('section');
  root.className = 'uni-timeline-dock';
  root.dataset.testid = 'uni-timeline-dock';
  root.setAttribute('aria-label', 'NLD-Uni trajectory and elastic prefill dock');

  // 1. 4-Task Switcher Tablist
  const taskTablist = document.createElement('div');
  taskTablist.className = 'uni-dock-tasks';
  taskTablist.setAttribute('role', 'tablist');
  taskTablist.setAttribute('aria-label', 'NLD-Uni task mode');

  const taskButtons: HTMLButtonElement[] = [];
  let activeTaskIdx = 0;

  for (let i = 0; i < UNI_DOCK_TASK_CHOICES.length; i++) {
    const choice = UNI_DOCK_TASK_CHOICES[i]!;
    const btn = makeButton(
      'uni-dock-task-btn',
      choice.shortLabel,
      `uni-dock-task-${choice.id}`,
      choice.label,
    );
    btn.setAttribute('role', 'tab');
    btn.title = choice.label;
    btn.addEventListener('click', () => {
      activeTaskIdx = i;
      syncRovingTabindex(taskButtons, activeTaskIdx);
      callbacks.onSelectTaskChoice(choice.id);
    });
    btn.addEventListener('keydown', (ev) => {
      handleUniTablistKeydown(ev, taskButtons, activeTaskIdx, (nextIdx) => {
        activeTaskIdx = nextIdx;
        callbacks.onSelectTaskChoice(UNI_DOCK_TASK_CHOICES[nextIdx]!.id);
      });
    });
    taskButtons.push(btn);
    taskTablist.appendChild(btn);
  }
  syncRovingTabindex(taskButtons, 0);

  // 2. Playback & Trajectory Scrubber
  const transportGroup = document.createElement('div');
  transportGroup.className = 'uni-dock-transport';

  const stepBackBtn = makeButton('uni-dock-btn', '◀', 'uni-dock-step-back', 'Step backward');
  stepBackBtn.addEventListener('click', () => callbacks.onStepBack());

  const playBtn = makeButton('uni-dock-btn', '▶ Play', 'uni-dock-play-pause', 'Play or pause');
  playBtn.addEventListener('click', () => callbacks.onTogglePlay());

  const stepFwdBtn = makeButton('uni-dock-btn', '▶', 'uni-dock-step-forward', 'Step forward');
  stepFwdBtn.addEventListener('click', () => callbacks.onStepForward());

  const scrubber = document.createElement('input');
  scrubber.type = 'range';
  scrubber.className = 'uni-dock-scrubber';
  scrubber.dataset.testid = 'uni-dock-scrubber';
  scrubber.min = '0';
  scrubber.max = '0';
  scrubber.value = '0';
  scrubber.setAttribute('aria-label', 'Trajectory step');
  scrubber.addEventListener('input', () => {
    const idx = Number.parseInt(scrubber.value, 10);
    if (Number.isFinite(idx)) {
      callbacks.onScrubIndex(idx);
    }
  });

  const stepLabel = document.createElement('span');
  stepLabel.className = 'uni-dock-step-readout';
  stepLabel.dataset.testid = 'uni-dock-step-readout';
  stepLabel.textContent = 'Step 0 / 0';

  const executeBtn = makeButton(
    'uni-dock-btn uni-dock-btn--primary',
    '+ Step',
    'uni-dock-execute-step',
    'Execute next denoising or decoding step',
  );
  executeBtn.addEventListener('click', () => callbacks.onExecuteNextStep());

  transportGroup.append(stepBackBtn, playBtn, stepFwdBtn, scrubber, stepLabel, executeBtn);

  // 3. Branch Select & Fork + D5 Prefill Mode Toggle
  const branchGroup = document.createElement('div');
  branchGroup.className = 'uni-dock-branches';

  const branchSelect = document.createElement('select');
  branchSelect.className = 'uni-dock-branch-select';
  branchSelect.dataset.testid = 'uni-dock-branch-select';
  branchSelect.setAttribute('aria-label', 'Active session branch');
  branchSelect.addEventListener('change', () => {
    if (branchSelect.value) {
      callbacks.onSwitchBranch(branchSelect.value);
    }
  });

  const forkBtn = makeButton('uni-dock-btn', 'Fork', 'uni-dock-fork-branch', 'Fork branch at cursor');
  forkBtn.addEventListener('click', () => callbacks.onForkBranch());

  const prefillGroup = document.createElement('div');
  prefillGroup.className = 'uni-dock-prefill-group';
  prefillGroup.setAttribute('role', 'group');
  prefillGroup.setAttribute('aria-label', 'D5 Elastic Prefill Mode');

  const hfBtn = makeButton(
    'uni-dock-toggle-btn',
    'HF-Faithful',
    'uni-dock-prefill-hf_faithful',
    'HF-Faithful 34L U prefill',
  );
  hfBtn.addEventListener('click', () => callbacks.onSelectPrefillMode('hf_faithful'));

  const prunedBtn = makeButton(
    'uni-dock-toggle-btn',
    'Elastic-Pruned',
    'uni-dock-prefill-elastic_pruned',
    'Elastic-Pruned 18L U prefill',
  );
  prunedBtn.addEventListener('click', () => callbacks.onSelectPrefillMode('elastic_pruned'));

  prefillGroup.append(hfBtn, prunedBtn);
  branchGroup.append(branchSelect, forkBtn, prefillGroup);

  // 4. Live Memory & Layer Telemetry Pills
  const telemetryGroup = document.createElement('div');
  telemetryGroup.className = 'uni-dock-telemetry';

  const memoryPill = document.createElement('span');
  memoryPill.className = 'uni-dock-pill uni-dock-pill--memory';
  memoryPill.dataset.testid = 'uni-dock-memory-pill';
  memoryPill.textContent = '0.00 / 12.00 MiB';

  const layerPill = document.createElement('span');
  layerPill.className = 'uni-dock-pill uni-dock-pill--layer';
  layerPill.dataset.testid = 'uni-dock-layer-pill';
  layerPill.textContent = '34L Elastic-MoT';

  telemetryGroup.append(memoryPill, layerPill);

  root.append(taskTablist, transportGroup, branchGroup, telemetryGroup);
  container.appendChild(root);

  let disposed = false;
  let lastBranchOptionsSignature = '';

  function update(state: UniTimelineDockState): void {
    if (disposed) return;

    const foundIdx = UNI_DOCK_TASK_CHOICES.findIndex((c) => c.id === state.taskChoice);
    activeTaskIdx = foundIdx >= 0 ? foundIdx : 0;
    syncRovingTabindex(taskButtons, activeTaskIdx);
    for (let i = 0; i < taskButtons.length; i++) {
      taskButtons[i]!.classList.toggle('is-active', i === activeTaskIdx);
      taskButtons[i]!.disabled = state.tourActive;
    }

    stepBackBtn.disabled = state.tourActive || !state.canStepBack;
    stepFwdBtn.disabled = state.tourActive || !state.canStepForward;
    executeBtn.disabled = !state.canExecuteStep || state.tourActive;
    forkBtn.disabled = state.tourActive || state.activeRecordId === null;
    playBtn.disabled = state.tourActive || state.totalRecordsInBranch <= 1;
    playBtn.textContent = state.isPlaying ? '⏸ Pause' : '▶ Play';
    playBtn.setAttribute('aria-pressed', state.isPlaying ? 'true' : 'false');

    const maxIdx = Math.max(0, state.totalRecordsInBranch - 1);
    scrubber.min = '0';
    scrubber.max = String(maxIdx);
    scrubber.value = String(Math.max(0, Math.min(maxIdx, state.currentStepIndex)));
    scrubber.disabled = state.tourActive || state.totalRecordsInBranch <= 1;

    const displayCur = state.totalRecordsInBranch > 0 ? state.currentStepIndex + 1 : 0;
    stepLabel.textContent = `Step ${displayCur} / ${state.totalRecordsInBranch}`;

    // Rebuild branch <option> nodes ONLY when the branch list signature changes
    const nextBranchSig = computeBranchOptionsSignature(state.branches);
    if (nextBranchSig !== lastBranchOptionsSignature) {
      lastBranchOptionsSignature = nextBranchSig;
      branchSelect.innerHTML = '';
      for (let i = 0; i < state.branches.length; i++) {
        const b = state.branches[i]!;
        const opt = document.createElement('option');
        opt.value = b.branchId;
        opt.textContent = `${b.label} (${b.recordCount})`;
        branchSelect.appendChild(opt);
      }
    }
    branchSelect.value = state.activeBranchId ?? '';
    branchSelect.disabled = state.tourActive || state.branches.length === 0;

    const isHf = state.prefillMode === 'hf_faithful';
    hfBtn.classList.toggle('is-active', isHf);
    hfBtn.setAttribute('aria-pressed', isHf ? 'true' : 'false');
    hfBtn.disabled = state.tourActive;
    prunedBtn.classList.toggle('is-active', !isHf);
    prunedBtn.setAttribute('aria-pressed', !isHf ? 'true' : 'false');
    prunedBtn.disabled = state.tourActive;

    const usedMiB = (state.memoryAudit.totalRetainedBytes / (1024 * 1024)).toFixed(2);
    const maxMiB = (state.memoryAudit.maxAllowedBytes / (1024 * 1024)).toFixed(2);
    memoryPill.textContent =
      `${usedMiB} / ${maxMiB} MiB · ${state.memoryAudit.branchCount}/${state.memoryAudit.maxBranches} br · ${state.memoryAudit.recordCount}/${state.memoryAudit.maxRecords} rec`;
    memoryPill.classList.toggle('is-over-budget', !state.memoryAudit.withinBudget);

    layerPill.textContent = state.layerSummary;
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    if (root.parentNode === container) {
      container.removeChild(root);
    }
  }

  return {
    update,
    dispose,
  };
}
