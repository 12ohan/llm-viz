/**
 * Diagnostic Session Workbench (`src/twotower/app/diagnosticUi.ts`)
 *
 * Pure TypeScript + DOM engineering diagnostic UI wired strictly to the headless
 * `TwoTowerSession` APIs (zero runtime dependencies, zero Nano UI/scene/render imports,
 * zero 3D WebGL).
 */

import {
  DEFAULT_TWOTOWER_PROMPT_TOKENS,
  TWOTOWER_CONFIDENCE_PRESETS,
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  deleteTwoTowerSessionBranch,
  forkTwoTowerSessionBranch,
  getActiveBranch,
  getActiveRecord,
  getLatestContextExecution,
  getSessionCapabilities,
  initTwoTowerDiffusionBlock,
  measureTwoTowerSessionRetainedBytes,
  modulationOrderForBayKind,
  navigateTwoTowerSession,
  navigateTwoTowerSessionBack,
  navigateTwoTowerSessionForward,
  pruneInactiveSessionTraces,
  restartTwoTowerSession,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  stepTwoTowerSessionNext,
  switchTwoTowerSessionBranch,
  updateTwoTowerSessionConfig,
  type SamplerSlotDecision,
  type SamplerStepPresentation,
  type TwoTowerContextLayerTrace,
  type TwoTowerDenoiserLayerTrace,
  type TwoTowerExecutionMode,
  type TwoTowerSession,
  type TwoTowerSessionCapabilities,
  type TwoTowerSessionRecord,
  type TwoTowerVariateProvenance,
} from '../index';

interface PromptPreset {
  readonly id: 'default_8' | 'short_4' | 'single_1';
  readonly label: string;
  readonly tokenIds: readonly number[];
}

const PROMPT_PRESETS: readonly PromptPreset[] = Object.freeze([
  Object.freeze({
    id: 'default_8' as const,
    label: 'Default 8 tokens',
    tokenIds: DEFAULT_TWOTOWER_PROMPT_TOKENS,
  }),
  Object.freeze({
    id: 'short_4' as const,
    label: 'Short 4 tokens',
    tokenIds: Object.freeze([0, 1, 2, 3]),
  }),
  Object.freeze({
    id: 'single_1' as const,
    label: 'Single token (S=1)',
    tokenIds: Object.freeze([0]),
  }),
]);

type StatusSeverity = 'neutral' | 'info' | 'warn' | 'error';

interface StatusBannerState {
  readonly severity: StatusSeverity;
  readonly text: string;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  textContent?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (textContent !== undefined) node.textContent = textContent;
  return node;
}

function formatBytes(bytes: number): string {
  const exact = `${bytes.toLocaleString('en-US')} B`;
  if (bytes >= 1024 * 1024) {
    return `${exact} (${(bytes / (1024 * 1024)).toFixed(2)} MiB)`;
  }
  if (bytes >= 1024) {
    return `${exact} (${(bytes / 1024).toFixed(2)} KiB)`;
  }
  return exact;
}

function computeRms(arr: Float32Array): number {
  if (arr.length === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    sumSq += arr[i] * arr[i];
  }
  return Math.sqrt(sumSq / arr.length);
}

function computeL2(arr: Float32Array): number {
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    sumSq += arr[i] * arr[i];
  }
  return Math.sqrt(sumSq);
}

function formatConfidencePresetLabel(threshold: number): string {
  for (const preset of TWOTOWER_CONFIDENCE_PRESETS) {
    if (Math.abs(threshold - preset.threshold) < 1e-6) {
      return preset.label;
    }
  }
  return `Custom (γ = ${threshold.toFixed(2)})`;
}

function formatModeLabel(mode: TwoTowerExecutionMode): string {
  switch (mode) {
    case 'mask_diffusion':
      return 'Mask-Diffusion';
    case 'ar':
      return 'AR';
    case 'mock_ar':
      return 'Mock-AR';
  }
}

function phaseBadgeVariant(phase: TwoTowerSessionRecord['lifecyclePhase']): string {
  switch (phase) {
    case 'ready':
      return 'tt-badge--emerald';
    case 'block_initialized':
    case 'block_refining':
      return 'tt-badge--amber';
    case 'refinement_denoised':
      return 'tt-badge--violet';
    case 'completed_uncommitted':
      return 'tt-badge--cyan';
    case 'eos_terminated':
    case 'capacity_exhausted':
      return 'tt-badge--rose';
  }
}

function decisionBadgeVariant(decision: SamplerSlotDecision): string {
  switch (decision) {
    case 'committed_threshold':
      return 'tt-badge--emerald';
    case 'committed_progress_floor':
    case 'committed_budget_floor':
    case 'committed_final_step':
      return 'tt-badge--cyan';
    case 'locked_prior':
      return 'tt-badge--indigo';
    case 'remasked':
      return 'tt-badge--amber';
  }
}

function formatBayTelemetrySummary(trace: TwoTowerContextLayerTrace | TwoTowerDenoiserLayerTrace): string {
  const rms = computeRms(trace.sim.Y);
  if (trace.kind === '*') {
    const sim = trace.sim;
    return `kind=* (GQA) | outRMS=${rms.toFixed(4)} | prefix=${sim.prefixTokens}, query=${sim.tokens}, attendedKV=${sim.keyTokens} [${sim.tokens}×${sim.keyTokens}] | mask=${sim.attentionMaskMode}`;
  }
  if (trace.kind === 'E') {
    const sim = trace.sim;
    const lastTok = Math.max(0, sim.tokens - 1);
    const experts: string[] = [];
    for (let k = 0; k < sim.topK; k++) {
      const expId = sim.topk_indices[lastTok * sim.topK + k];
      experts.push(`E${String(expId).padStart(3, '0')}`);
    }
    return `kind=E (MoE) | outRMS=${rms.toFixed(4)} | last-tok Top-${sim.topK}=[${experts.join(', ')}] | router=${sim.routerPrecisionPolicy}`;
  }
  const sim = trace.sim;
  const hNorm = computeL2(sim.H);
  return `kind=M (Mamba-2) | outRMS=${rms.toFixed(4)} | processed=${sim.startTokenOffset}→${sim.startTokenOffset + sim.tokens} (Δ=${sim.tokens}) | seeded=${sim.hasInitialState ? 'yes' : 'no'} | ‖H‖₂=${hNorm.toFixed(4)}`;
}

function findLatestConfidenceCommitForBlock(
  session: TwoTowerSession,
  blockId: number,
): {
  readonly presentation: SamplerStepPresentation;
  readonly variateProvenance: TwoTowerVariateProvenance;
  readonly refinementIndex: number;
  readonly isFromActiveRecord: boolean;
} | null {
  const branch = getActiveBranch(session);
  const active = branch.records[branch.cursorIndex];
  if (active.kind === 'diffusion_confidence_commit') {
    return {
      presentation: active.samplerPresentation,
      variateProvenance: active.variateProvenance,
      refinementIndex: active.refinementIndex,
      isFromActiveRecord: true,
    };
  }
  for (let i = branch.cursorIndex - 1; i >= 0; i--) {
    const rec = branch.records[i];
    if (rec.kind === 'diffusion_confidence_commit' && rec.blockConfig.blockId === blockId) {
      return {
        presentation: rec.samplerPresentation,
        variateProvenance: rec.variateProvenance,
        refinementIndex: rec.refinementIndex,
        isFromActiveRecord: false,
      };
    }
    if (rec.activeBlock === null) break;
  }
  return null;
}

function describeActionTooltips(
  session: TwoTowerSession,
  record: TwoTowerSessionRecord,
  caps: TwoTowerSessionCapabilities,
): {
  readonly stepAr: string;
  readonly stepMockAr: string;
  readonly initBlock: string;
  readonly runRefinement: string;
  readonly commitConfidence: string;
  readonly commitBlock: string;
  readonly stepNext: string;
} {
  const autoForkNote = caps.willAutoForkOnMutatingStep
    ? ' (rewound cursor: will auto-fork a new branch)'
    : '';
  const termNote = caps.isTerminated
    ? `Disabled: branch terminated (${caps.terminationReason}). Navigate back, fork, or restart.`
    : null;
  const maxBranchNote =
    !caps.isAtBranchTail && session.bounds.maxBranches < 2
      ? 'Disabled: cursor is rewound and maxBranches < 2.'
      : null;
  const globalBlock = termNote ?? maxBranchNote;

  const stepAr = caps.canStepAr
    ? `Execute 1 causal Context Tower AR step at pos ${caps.committedPrefixLen}${autoForkNote}.`
    : globalBlock ??
      (caps.hasActiveBlock
        ? `Disabled: diffusion block b${record.activeBlock!.blockId} is active (${caps.activeBlockStatus}). Commit or rewind first.`
        : `Disabled: KV capacity exhausted (${caps.committedPrefixLen}/${caps.kvCapacity}).`);

  const stepMockAr = caps.canStepMockAr
    ? `Execute 1 unmodulated Denoiser Mock-AR step (KV[:-1] + mambaS2, tModel=null) at pos ${caps.committedPrefixLen}${autoForkNote}.`
    : globalBlock ??
      (caps.hasActiveBlock
        ? `Disabled: diffusion block b${record.activeBlock!.blockId} is active (${caps.activeBlockStatus}). Commit or rewind first.`
        : `Disabled: KV capacity exhausted (${caps.committedPrefixLen}/${caps.kvCapacity}).`);

  const initBlock = caps.canInitDiffusionBlock
    ? `Initialize masked block [MASK]^${caps.pendingBlockSize} at pos [${caps.committedPrefixLen}..${caps.committedPrefixLen + caps.pendingBlockSize - 1}]${autoForkNote}.`
    : globalBlock ??
      (caps.hasActiveBlock
        ? `Disabled: diffusion block b${record.activeBlock!.blockId} is already active (${caps.activeBlockStatus}).`
        : `Disabled: P + B (${caps.committedPrefixLen} + ${caps.pendingBlockSize} = ${caps.committedPrefixLen + caps.pendingBlockSize}) exceeds kvCapacity (${caps.kvCapacity}). Reduce B or restart.`);

  const runRefinement = caps.canRunDiffusionRefinement
    ? `Run Denoiser forward pass r=${record.activeBlock!.completedRefinements} on block b${record.activeBlock!.blockId} (leaves xt unchanged for inspection)${autoForkNote}.`
    : globalBlock ??
      (!caps.hasActiveBlock
        ? 'Disabled: initialize a diffusion block first.'
        : caps.activeBlockStatus === 'awaiting_confidence_commit'
          ? 'Disabled: current denoised refinement is awaiting Commit Confidence.'
          : caps.activeBlockStatus === 'completed_uncommitted'
            ? 'Disabled: block has 0 masked slots left — use Commit Block -> Context.'
            : 'Disabled: block refinement budget K reached.');

  const commitConfidence = caps.canCommitRefinementConfidence
    ? `Apply MDLM constraints and confidence unmasking (γ=${session.pendingConfig.confidenceThreshold.toFixed(2)}, T=${session.pendingConfig.temperature}) to pending refinement r=${record.activeBlock!.pendingRefinementIndex}${autoForkNote}.`
    : globalBlock ??
      (!caps.hasActiveBlock
        ? 'Disabled: no active diffusion block.'
        : `Disabled: run a Denoiser pass first (status: ${caps.activeBlockStatus}).`);

  const commitBlock = caps.canCommitActiveBlockToContext
    ? `Commit completed block b${record.activeBlock!.blockId} (${record.activeBlock!.blockConfig.blockSize} tokens) into persistent Context Tower cache${autoForkNote}.`
    : globalBlock ??
      (!caps.hasActiveBlock
        ? 'Disabled: no active diffusion block.'
        : `Disabled: block b${record.activeBlock!.blockId} still has ${record.activeBlock!.nMasked} masked token(s) (status: ${caps.activeBlockStatus}).`);

  const stepNext = caps.canStepNext
    ? `Execute next legal transition (${caps.nextTransitionKind}) for mode ${formatModeLabel(session.pendingConfig.mode)}${autoForkNote}.`
    : globalBlock ??
      `Disabled: no legal transition in mode ${formatModeLabel(session.pendingConfig.mode)} (remaining KV ${caps.remainingKvCapacity}, pending B=${caps.pendingBlockSize}).`;

  return {
    stepAr,
    stepMockAr,
    initBlock,
    runRefinement,
    commitConfidence,
    commitBlock,
    stepNext,
  };
}

export interface DiagnosticWorkbenchController {
  syncSession(nextSession: TwoTowerSession, statusText?: string): void;
  setInspectedBay(bayId: number): void;
  getSession(): TwoTowerSession;
}

export interface DiagnosticWorkbenchOptions {
  readonly initialSession?: TwoTowerSession;
  readonly onSessionChange?: (nextSession: TwoTowerSession, statusText?: string) => void;
  readonly onInspectedBayChange?: (bayId: number) => void;
}

export function mountTwoTowerDiagnosticWorkbench(
  root: HTMLElement,
  options?: DiagnosticWorkbenchOptions,
): DiagnosticWorkbenchController {
  let session: TwoTowerSession =
    options?.initialSession ?? createTwoTowerSession(createDefaultTwoTowerModel());
  let selectedPromptPresetId: PromptPreset['id'] = 'default_8';
  let inspectedBayId = 6; // Default to Bay 6 (GQA *, eagerly retained on initial prefill)
  let statusBanner: StatusBannerState = {
    severity: 'info',
    text: 'Session initialized via split-segment Context prefill (P=8 / 24, B=4, K=4, γ=0.22 mini default).',
  };

  root.textContent = '';

  const workbench = el('div', 'tt-workbench');
  workbench.setAttribute('role', 'region');
  workbench.setAttribute('aria-label', 'Nemotron-Labs-TwoTower Diagnostic Session Workbench');

  // ---------------------------------------------------------------------------
  // 1. Header & Epistemic Disclosure Banner
  // ---------------------------------------------------------------------------
  const header = el('header', 'tt-header');
  const titleGroup = el('div', 'tt-header-title-group');
  const h1 = el(
    'h1',
    'tt-title',
    'Nemotron-Labs-TwoTower — Diagnostic Session Workbench',
  );
  const shortRev = TWOTOWER_SPEC.provenance.hfRevision.slice(0, 8);
  const subtitle = el(
    'div',
    'tt-subtitle',
    `Dual-Tower 2×52L (23M · 23E · 6*) · 29 Aligned Stateful Transfer Bays · HF rev ${shortRev} · arXiv:${TWOTOWER_SPEC.provenance.paperArxivId}`,
  );
  titleGroup.append(h1, subtitle);

  const headerStatus = el('div', 'tt-header-status');
  const phaseBadge = el('span', 'tt-badge');
  phaseBadge.dataset.testid = 'lifecycle-phase';

  const capacityBadge = el('span', 'tt-badge tt-badge--cyan');
  capacityBadge.dataset.testid = 'capacity-meter';

  const capacityWarningBadge = el('span', 'tt-badge tt-badge--rose');
  capacityWarningBadge.dataset.testid = 'capacity-warning';
  capacityWarningBadge.hidden = true;

  headerStatus.append(phaseBadge, capacityBadge, capacityWarningBadge);
  header.append(titleGroup, headerStatus);

  const epistemicBanner = el('section', 'tt-epistemic');
  epistemicBanner.dataset.testid = 'epistemic-banner';
  epistemicBanner.setAttribute('aria-label', 'Epistemic simulator disclosure');
  epistemicBanner.innerHTML =
    `<span><strong>Miniature Float32 Educational Simulator:</strong> ` +
    `<code>d = ${TWOTOWER_MINI_CONFIG.d}</code>, ` +
    `<code>V = ${TWOTOWER_MINI_CONFIG.vocabSize}</code>, ` +
    `<code>L = ${TWOTOWER_SPEC.layerAlignment.totalLayers}</code>, ` +
    `<code>kvCapacity = ${TWOTOWER_MINI_CONFIG.kvCapacity}</code>, deterministic synthetic seeded weights ` +
    `— <strong>NOT</strong> 63.2B checkpoint inference (pinned HF metadata revision <code>${shortRev}</code>).</span>` +
    `<span><strong>Threshold Calibration:</strong> <code>γ = 0.22</code> is a calibrated miniature educational default ` +
    `alongside <code>0.80</code> Paper and <code>0.90</code> Reference Code presets.</span>`;

  // ---------------------------------------------------------------------------
  // 2. Configuration & Prompt Bar (Strictly Future Transitions)
  // ---------------------------------------------------------------------------
  const configPanel = el('section', 'tt-panel');
  configPanel.setAttribute('aria-label', 'Session configuration and prompt presets');

  const configHead = el('div', 'tt-panel-head');
  configHead.append(
    el('h2', 'tt-panel-title', 'Configuration & Prompt Bar'),
    el(
      'span',
      'tt-panel-note',
      'Changes apply strictly to future transitions & blocks — historical records and active block configs are never retroactively altered.',
    ),
  );

  const configGrid = el('div', 'tt-config-grid');

  // Execution Mode
  const modeField = el('div', 'tt-field');
  const modeLabel = el('label', undefined, 'Execution Mode');
  modeLabel.htmlFor = 'tt-mode-select';
  const modeSelect = el('select', 'tt-select');
  modeSelect.id = 'tt-mode-select';
  modeSelect.dataset.testid = 'mode-select';
  for (const [val, lbl] of [
    ['mask_diffusion', 'Mask-Diffusion'],
    ['ar', 'AR'],
    ['mock_ar', 'Mock-AR'],
  ] as const) {
    const opt = el('option', undefined, lbl);
    opt.value = val;
    modeSelect.append(opt);
  }
  modeField.append(modeLabel, modeSelect);

  // Prompt Preset + Apply & Restart
  const promptField = el('div', 'tt-field');
  const promptLabel = el('label', undefined, 'Prompt Preset');
  promptLabel.htmlFor = 'tt-prompt-select';
  const promptRow = el('div', 'tt-field-row');
  const promptSelect = el('select', 'tt-select');
  promptSelect.id = 'tt-prompt-select';
  promptSelect.dataset.testid = 'prompt-select';
  for (const p of PROMPT_PRESETS) {
    const opt = el('option', undefined, p.label);
    opt.value = p.id;
    promptSelect.append(opt);
  }
  const restartBtn = el('button', 'tt-btn tt-btn--primary', 'Apply & Restart');
  restartBtn.type = 'button';
  restartBtn.dataset.testid = 'btn-restart';
  promptRow.append(promptSelect, restartBtn);
  promptField.append(promptLabel, promptRow);

  // Block Width B
  const blockSizeField = el('div', 'tt-field');
  const blockSizeLabel = el('label', undefined, 'Block Width B');
  blockSizeLabel.htmlFor = 'tt-block-size-select';
  const blockSizeSelect = el('select', 'tt-select');
  blockSizeSelect.id = 'tt-block-size-select';
  blockSizeSelect.dataset.testid = 'block-size-select';
  for (const bSize of TWOTOWER_MINI_CONFIG.supportedBlockSizes) {
    const opt = el('option', undefined, String(bSize));
    opt.value = String(bSize);
    blockSizeSelect.append(opt);
  }
  blockSizeField.append(blockSizeLabel, blockSizeSelect);

  // Independent Refinement Budget K
  const stepsField = el('div', 'tt-field');
  const stepsLabel = el('label', undefined, 'Steps K (1..16)');
  stepsLabel.htmlFor = 'tt-steps-input';
  const stepsInput = el('input', 'tt-input tt-input--sm');
  stepsInput.id = 'tt-steps-input';
  stepsInput.type = 'number';
  stepsInput.min = '1';
  stepsInput.max = '16';
  stepsInput.step = '1';
  stepsInput.dataset.testid = 'steps-input';
  stepsField.append(stepsLabel, stepsInput);

  // Confidence Threshold γ + 3 Presets
  const thresholdField = el('div', 'tt-field');
  const thresholdLabel = el('label', undefined, 'Confidence Threshold γ');
  thresholdLabel.htmlFor = 'tt-threshold-input';
  const thresholdRow = el('div', 'tt-field-row');
  const thresholdInput = el('input', 'tt-input tt-input--sm');
  thresholdInput.id = 'tt-threshold-input';
  thresholdInput.type = 'number';
  thresholdInput.min = '0';
  thresholdInput.max = '1';
  thresholdInput.step = '0.01';
  thresholdInput.dataset.testid = 'threshold-input';

  const presetMiniBtn = el('button', 'tt-btn tt-btn--preset', 'Mini Default 0.22');
  presetMiniBtn.type = 'button';
  presetMiniBtn.dataset.testid = 'preset-mini';
  presetMiniBtn.title = TWOTOWER_CONFIDENCE_PRESETS[0].provenanceNote;

  const presetPaperBtn = el('button', 'tt-btn tt-btn--preset', 'Paper 0.80');
  presetPaperBtn.type = 'button';
  presetPaperBtn.dataset.testid = 'preset-paper';
  presetPaperBtn.title = TWOTOWER_CONFIDENCE_PRESETS[1].provenanceNote;

  const presetCodeBtn = el('button', 'tt-btn tt-btn--preset', 'Code 0.90');
  presetCodeBtn.type = 'button';
  presetCodeBtn.dataset.testid = 'preset-code';
  presetCodeBtn.title = TWOTOWER_CONFIDENCE_PRESETS[2].provenanceNote;

  thresholdRow.append(thresholdInput, presetMiniBtn, presetPaperBtn, presetCodeBtn);
  thresholdField.append(thresholdLabel, thresholdRow);

  // Temperature T
  const tempField = el('div', 'tt-field');
  const tempLabel = el('label', undefined, 'Temp T (0=Greedy)');
  tempLabel.htmlFor = 'tt-temperature-input';
  const tempInput = el('input', 'tt-input tt-input--sm');
  tempInput.id = 'tt-temperature-input';
  tempInput.type = 'number';
  tempInput.min = '0';
  tempInput.step = '0.1';
  tempInput.dataset.testid = 'temperature-input';
  tempField.append(tempLabel, tempInput);

  // Sampling Seed
  const seedField = el('div', 'tt-field');
  const seedLabel = el('label', undefined, 'Sampling Seed');
  seedLabel.htmlFor = 'tt-seed-input';
  const seedInput = el('input', 'tt-input tt-input--md');
  seedInput.id = 'tt-seed-input';
  seedInput.type = 'number';
  seedInput.step = '1';
  seedInput.dataset.testid = 'seed-input';
  seedField.append(seedLabel, seedInput);

  configGrid.append(
    modeField,
    promptField,
    blockSizeField,
    stepsField,
    thresholdField,
    tempField,
    seedField,
  );
  configPanel.append(configHead, configGrid);

  // ---------------------------------------------------------------------------
  // 3. Explicit Lifecycle Action Bar & History / Branching Bar
  // ---------------------------------------------------------------------------
  const actionPanel = el('section', 'tt-panel');
  actionPanel.setAttribute('aria-label', 'Explicit lifecycle transitions and history controls');

  const actionRows = el('div', 'tt-action-rows');

  const transitionRow = el('div', 'tt-action-row');
  transitionRow.append(el('span', 'tt-action-label', 'Transitions'));

  const btnStepAr = el('button', 'tt-btn', 'Step AR');
  btnStepAr.type = 'button';
  btnStepAr.dataset.testid = 'btn-step-ar';

  const btnStepMockAr = el('button', 'tt-btn', 'Step Mock-AR');
  btnStepMockAr.type = 'button';
  btnStepMockAr.dataset.testid = 'btn-step-mock-ar';

  const btnInitBlock = el('button', 'tt-btn tt-btn--accent', 'Init Block [MASK]^B');
  btnInitBlock.type = 'button';
  btnInitBlock.dataset.testid = 'btn-init-block';

  const btnRunRefinement = el('button', 'tt-btn tt-btn--accent', 'Run Denoiser Pass');
  btnRunRefinement.type = 'button';
  btnRunRefinement.dataset.testid = 'btn-run-refinement';

  const btnCommitConfidence = el('button', 'tt-btn tt-btn--accent', 'Commit Confidence');
  btnCommitConfidence.type = 'button';
  btnCommitConfidence.dataset.testid = 'btn-commit-confidence';

  const btnCommitBlock = el('button', 'tt-btn tt-btn--emerald', 'Commit Block -> Context');
  btnCommitBlock.type = 'button';
  btnCommitBlock.dataset.testid = 'btn-commit-block';

  const btnStepNext = el('button', 'tt-btn tt-btn--primary', 'Next Step');
  btnStepNext.type = 'button';
  btnStepNext.dataset.testid = 'btn-step-next';

  transitionRow.append(
    btnStepAr,
    btnStepMockAr,
    el('span', 'tt-divider'),
    btnInitBlock,
    btnRunRefinement,
    btnCommitConfidence,
    btnCommitBlock,
    el('span', 'tt-divider'),
    btnStepNext,
  );

  const historyRow = el('div', 'tt-action-row');
  historyRow.append(el('span', 'tt-action-label', 'History'));

  const btnNavBack = el('button', 'tt-btn', '<- Back');
  btnNavBack.type = 'button';
  btnNavBack.dataset.testid = 'btn-nav-back';

  const btnNavForward = el('button', 'tt-btn', 'Forward ->');
  btnNavForward.type = 'button';
  btnNavForward.dataset.testid = 'btn-nav-forward';

  const btnForkBranch = el('button', 'tt-btn', 'Fork Branch');
  btnForkBranch.type = 'button';
  btnForkBranch.dataset.testid = 'btn-fork-branch';

  const branchSelect = el('select', 'tt-select');
  branchSelect.dataset.testid = 'branch-select';
  branchSelect.setAttribute('aria-label', 'Active session branch');

  const btnDeleteBranch = el('button', 'tt-btn tt-btn--danger', 'Delete Branch');
  btnDeleteBranch.type = 'button';
  btnDeleteBranch.dataset.testid = 'btn-delete-branch';

  const btnPruneTraces = el('button', 'tt-btn', 'Prune Inactive Traces');
  btnPruneTraces.type = 'button';
  btnPruneTraces.dataset.testid = 'btn-prune-traces';

  const autoForkPill = el('span', 'tt-badge tt-badge--amber', 'Rewound: next step auto-forks');
  autoForkPill.hidden = true;

  historyRow.append(
    btnNavBack,
    btnNavForward,
    el('span', 'tt-divider'),
    btnForkBranch,
    branchSelect,
    btnDeleteBranch,
    el('span', 'tt-divider'),
    btnPruneTraces,
    autoForkPill,
  );

  const statusMessageEl = el('div', 'tt-status-message');
  statusMessageEl.dataset.testid = 'status-message';
  statusMessageEl.setAttribute('role', 'status');
  statusMessageEl.setAttribute('aria-live', 'polite');

  actionRows.append(transitionRow, historyRow, statusMessageEl);
  actionPanel.append(actionRows);

  // ---------------------------------------------------------------------------
  // 4. Main Two-Column Workbench
  // ---------------------------------------------------------------------------
  const mainGrid = el('div', 'tt-main-grid');
  const leftCol = el('div', 'tt-col');
  const rightCol = el('div', 'tt-col');

  // 4A. Token Tape Panel
  const tapePanel = el('section', 'tt-panel');
  const tapeHead = el('div', 'tt-panel-head');
  const tapeSummary = el('span', 'tt-panel-note');
  tapeHead.append(el('h2', 'tt-panel-title', 'Token Tape (Context + Active Block + Free KV)'), tapeSummary);

  const tapeLegend = el('div', 'tt-tape-legend');
  for (const [cls, txt] of [
    ['tt-swatch--context', 'Context Committed'],
    ['tt-swatch--mask', 'Active [MASK]'],
    ['tt-swatch--new', 'Newly Unmasked'],
    ['tt-swatch--locked', 'Locked Prior'],
    ['tt-swatch--uncommitted', 'Completed (Uncommitted)'],
    ['tt-swatch--free', 'Free KV Slot'],
  ] as const) {
    const item = el('span', 'tt-legend-item');
    item.append(el('span', `tt-swatch ${cls}`), document.createTextNode(txt));
    tapeLegend.append(item);
  }

  const tokenTapeContainer = el('div', 'tt-token-tape');
  tokenTapeContainer.dataset.testid = 'token-tape';
  tapePanel.append(tapeHead, tapeLegend, tokenTapeContainer);

  // 4B. Sampler Presentation Breakdown Panel
  const samplerPanel = el('section', 'tt-panel');
  samplerPanel.dataset.testid = 'sampler-panel';
  const samplerHead = el('div', 'tt-panel-head');
  const samplerTitleRight = el('span', 'tt-panel-note');
  samplerHead.append(
    el('h2', 'tt-panel-title', 'Sampler Presentation Breakdown (MDLM Constrained + Unscaled Confidence)'),
    samplerTitleRight,
  );
  const samplerBody = el('div');
  samplerPanel.append(samplerHead, samplerBody);

  // 4C. Branch History List Panel
  const historyPanel = el('section', 'tt-panel');
  const historyHead = el('div', 'tt-panel-head');
  const historyBranchInfo = el('span', 'tt-panel-note');
  historyHead.append(el('h2', 'tt-panel-title', 'Active Branch Transition History'), historyBranchInfo);
  const historyListContainer = el('div', 'tt-history-list');
  historyListContainer.dataset.testid = 'history-list';
  historyPanel.append(historyHead, historyListContainer);

  leftCol.append(tapePanel, samplerPanel, historyPanel);

  // 4D. Identity, Provenance & Trace Origin Inspector Panel
  const inspectorPanel = el('section', 'tt-panel');
  inspectorPanel.dataset.testid = 'inspector-panel';

  const inspectorHead = el('div', 'tt-panel-head');
  const baySelectorWrap = el('div', 'tt-field-row');
  const baySelectorLabel = el('label', 'tt-panel-note', 'Inspect Bay:');
  baySelectorLabel.htmlFor = 'tt-trace-bay-select';
  const traceBaySelect = el('select', 'tt-select');
  traceBaySelect.id = 'tt-trace-bay-select';
  traceBaySelect.dataset.testid = 'trace-bay-select';
  for (const entry of TWOTOWER_SPEC.layerAlignment.entries) {
    const kindName = entry.kind === 'M' ? 'Mamba-2 (M)' : entry.kind === '*' ? 'GQA (*)' : 'MoE (E)';
    const transferTag = entry.transfersState ? entry.transferKind : 'stateless';
    const opt = el(
      'option',
      undefined,
      `Bay ${String(entry.bayId).padStart(2, '0')} · ${kindName} · ${transferTag}`,
    );
    opt.value = String(entry.bayId);
    traceBaySelect.append(opt);
  }
  traceBaySelect.value = String(inspectedBayId);
  baySelectorWrap.append(baySelectorLabel, traceBaySelect);

  inspectorHead.append(
    el('h2', 'tt-panel-title', 'Identity, Provenance & Trace Origin Inspector'),
    baySelectorWrap,
  );

  const identityGrid = el('dl', 'tt-kv-grid');
  const contextTraceCard = el('div', 'tt-subcard');
  const persistentCacheCard = el('div', 'tt-subcard');
  const denoiserScratchCard = el('div', 'tt-subcard');

  inspectorPanel.append(
    inspectorHead,
    identityGrid,
    contextTraceCard,
    persistentCacheCard,
    denoiserScratchCard,
  );

  // 4E. Retained Session Memory Panel
  const memoryPanel = el('section', 'tt-panel');
  memoryPanel.dataset.testid = 'memory-panel';
  const memoryHead = el('div', 'tt-panel-head');
  memoryHead.append(
    el('h2', 'tt-panel-title', 'Retained Session Memory (Deduplicated Typed-Array Accounting)'),
    el('span', 'tt-panel-note', 'Excludes read-only shared weights'),
  );
  const memoryGrid = el('dl', 'tt-mem-grid');
  memoryPanel.append(memoryHead, memoryGrid);

  rightCol.append(inspectorPanel, memoryPanel);
  mainGrid.append(leftCol, rightCol);

  workbench.append(header, epistemicBanner, configPanel, actionPanel, mainGrid);
  root.append(workbench);

  // ---------------------------------------------------------------------------
  // Safe Action Wrapper (Never Uncaught Throw!)
  // ---------------------------------------------------------------------------
  function setStatus(severity: StatusSeverity, text: string): void {
    statusBanner = { severity, text };
  }

  function runSafe(actionLabel: string, fn: () => void): void {
    const prevSession = session;
    try {
      fn();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus('error', `${actionLabel}: ${msg}`);
    }
    render();
    if (session !== prevSession) {
      options?.onSessionChange?.(session, statusBanner.text);
    }
  }

  // ---------------------------------------------------------------------------
  // Event Wiring
  // ---------------------------------------------------------------------------
  modeSelect.addEventListener('change', () => {
    runSafe('Mode update failed', () => {
      const mode = modeSelect.value as TwoTowerExecutionMode;
      session = updateTwoTowerSessionConfig(session, { mode });
      setStatus(
        'info',
        `Future execution mode set to ${formatModeLabel(mode)} (historical records unchanged).`,
      );
    });
  });

  promptSelect.addEventListener('change', () => {
    selectedPromptPresetId = promptSelect.value as PromptPreset['id'];
    setStatus(
      'neutral',
      `Selected prompt preset "${PROMPT_PRESETS.find((p) => p.id === selectedPromptPresetId)?.label}". Click "Apply & Restart" to re-prefill.`,
    );
    render();
  });

  restartBtn.addEventListener('click', () => {
    runSafe('Restart failed', () => {
      const preset =
        PROMPT_PRESETS.find((p) => p.id === selectedPromptPresetId) ?? PROMPT_PRESETS[0];
      session = restartTwoTowerSession(session, { promptTokenIds: preset.tokenIds });
      setStatus(
        'info',
        `Session restarted with prompt preset "${preset.label}" (P=${preset.tokenIds.length} / ${session.kvCapacity}).`,
      );
    });
  });

  blockSizeSelect.addEventListener('change', () => {
    runSafe('Block width update failed', () => {
      const blockSize = Number(blockSizeSelect.value);
      session = updateTwoTowerSessionConfig(session, { blockSize });
      setStatus(
        'info',
        `Future block width set to B=${blockSize} (applies to newly initialized blocks only).`,
      );
    });
  });

  const applyStepsValue = (strictEmptyError: boolean): void => {
    const raw = stepsInput.value.trim();
    if (raw === '') {
      if (strictEmptyError) {
        setStatus('error', 'Invalid Steps K: enter an integer in [1..16].');
        render();
      }
      return;
    }
    const parsed = Number(raw);
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > 16) {
      setStatus('error', `Invalid Steps K (${raw}): must be an integer in [1..16].`);
      render();
      return;
    }
    runSafe('Steps K update failed', () => {
      session = updateTwoTowerSessionConfig(session, { stepsPerBlock: parsed });
      setStatus(
        'info',
        `Future refinement budget set to K=${parsed} (independent of B=${session.pendingConfig.blockSize}).`,
      );
    });
  };
  stepsInput.addEventListener('input', () => applyStepsValue(false));
  stepsInput.addEventListener('change', () => applyStepsValue(true));

  const applyThresholdValue = (strictEmptyError: boolean): void => {
    const raw = thresholdInput.value.trim();
    if (raw === '') {
      if (strictEmptyError) {
        setStatus('error', 'Invalid confidence threshold γ: enter a finite number in [0, 1].');
        render();
      }
      return;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
      setStatus('error', `Invalid confidence threshold γ (${raw}): must be finite in [0, 1].`);
      render();
      return;
    }
    runSafe('Threshold γ update failed', () => {
      session = updateTwoTowerSessionConfig(session, { confidenceThreshold: parsed });
      setStatus(
        'info',
        `Future confidence threshold set to γ=${parsed.toFixed(2)} (${formatConfidencePresetLabel(parsed)}).`,
      );
    });
  };
  thresholdInput.addEventListener('input', () => applyThresholdValue(false));
  thresholdInput.addEventListener('change', () => applyThresholdValue(true));

  const applyThresholdPreset = (value: number, label: string): void => {
    runSafe('Preset update failed', () => {
      session = updateTwoTowerSessionConfig(session, { confidenceThreshold: value });
      thresholdInput.value = value.toFixed(2);
      setStatus('info', `Applied confidence threshold preset: ${label} (γ=${value.toFixed(2)}).`);
    });
  };
  presetMiniBtn.addEventListener('click', () =>
    applyThresholdPreset(0.22, 'Miniature Educational Default'),
  );
  presetPaperBtn.addEventListener('click', () => applyThresholdPreset(0.8, 'Paper Default'));
  presetCodeBtn.addEventListener('click', () =>
    applyThresholdPreset(0.9, 'Reference Code Default'),
  );

  const applyTemperatureValue = (strictEmptyError: boolean): void => {
    const raw = tempInput.value.trim();
    if (raw === '') {
      if (strictEmptyError) {
        setStatus('error', 'Invalid temperature T: enter a finite number >= 0.');
        render();
      }
      return;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
      setStatus('error', `Invalid temperature T (${raw}): must be finite >= 0.`);
      render();
      return;
    }
    runSafe('Temperature update failed', () => {
      session = updateTwoTowerSessionConfig(session, { temperature: parsed });
      setStatus(
        'info',
        parsed <= 0
          ? 'Future sampling set to T=0.0 (deterministic greedy argmax).'
          : `Future sampling set to T=${parsed} (deterministic keyed Gumbel-max, seed=${session.pendingConfig.samplingSeed}).`,
      );
    });
  };
  tempInput.addEventListener('input', () => applyTemperatureValue(false));
  tempInput.addEventListener('change', () => applyTemperatureValue(true));

  const applySeedValue = (strictEmptyError: boolean): void => {
    const raw = seedInput.value.trim();
    if (raw === '') {
      if (strictEmptyError) {
        setStatus('error', 'Invalid sampling seed: enter a finite integer.');
        render();
      }
      return;
    }
    const parsed = Number(raw);
    if (!Number.isInteger(parsed)) {
      setStatus('error', `Invalid sampling seed (${raw}): must be an integer.`);
      render();
      return;
    }
    runSafe('Sampling seed update failed', () => {
      session = updateTwoTowerSessionConfig(session, { samplingSeed: parsed });
      setStatus('info', `Future deterministic sampling seed set to ${parsed}.`);
    });
  };
  seedInput.addEventListener('input', () => applySeedValue(false));
  seedInput.addEventListener('change', () => applySeedValue(true));

  // Transition Buttons
  btnStepAr.addEventListener('click', () => {
    runSafe('Step AR failed', () => {
      session = stepTwoTowerSessionAr(session);
      const rec = getActiveRecord(session);
      if (rec.kind === 'ar_step') {
        setStatus(
          'info',
          `AR step committed "${rec.selectedTokenLabel}" (id=${rec.selectedTokenId}) at pos ${rec.identity.absoluteTokenStart}. Context ${rec.committedPrefixLen}/${rec.kvCapacity}.`,
        );
      }
    });
  });

  btnStepMockAr.addEventListener('click', () => {
    runSafe('Step Mock-AR failed', () => {
      session = stepTwoTowerSessionMockAr(session);
      const rec = getActiveRecord(session);
      if (rec.kind === 'mock_ar_step') {
        setStatus(
          'info',
          `Mock-AR step (unmodulated Denoiser, tModel=null) committed "${rec.selectedTokenLabel}" (id=${rec.selectedTokenId}) at pos ${rec.identity.absoluteTokenStart}. Context ${rec.committedPrefixLen}/${rec.kvCapacity}.`,
        );
      }
    });
  });

  btnInitBlock.addEventListener('click', () => {
    runSafe('Init Block failed', () => {
      session = initTwoTowerDiffusionBlock(session);
      const rec = getActiveRecord(session);
      if (rec.kind === 'diffusion_block_init') {
        setStatus(
          'info',
          `Initialized block b${rec.blockConfig.blockId} ([MASK]^${rec.blockConfig.blockSize}, K=${rec.blockConfig.stepsPerBlock}) at pos [${rec.identity.absoluteTokenRange[0]}..${rec.identity.absoluteTokenRange[1]}]. Context stays ${rec.committedPrefixLen}/${rec.kvCapacity}.`,
        );
      }
    });
  });

  btnRunRefinement.addEventListener('click', () => {
    runSafe('Run Denoiser Pass failed', () => {
      session = runTwoTowerDiffusionRefinement(session);
      const rec = getActiveRecord(session);
      if (rec.kind === 'diffusion_refinement') {
        setStatus(
          'info',
          `Denoiser pass b${rec.blockConfig.blockId}/r${rec.refinementIndex} complete (tModel=${rec.tModel.toFixed(3)}). xt unchanged; ready for Commit Confidence.`,
        );
      }
    });
  });

  btnCommitConfidence.addEventListener('click', () => {
    runSafe('Commit Confidence failed', () => {
      session = commitTwoTowerRefinementConfidence(session);
      const rec = getActiveRecord(session);
      if (rec.kind === 'diffusion_confidence_commit') {
        const p = rec.samplerPresentation;
        setStatus(
          'info',
          `Confidence commit b${rec.blockConfig.blockId}/r${rec.refinementIndex}: unmasked ${p.tokensCommittedCount} (${p.commitReason}), remaining masked=${p.nMaskedAfter}. Phase: ${rec.lifecyclePhase}.`,
        );
      }
    });
  });

  btnCommitBlock.addEventListener('click', () => {
    runSafe('Commit Block failed', () => {
      session = commitTwoTowerActiveBlockToContext(session);
      const rec = getActiveRecord(session);
      if (rec.kind === 'diffusion_block_commit') {
        setStatus(
          'info',
          `Committed block b${rec.blockConfig.blockId} (${rec.committedBlockTokens.length} tokens, NFE=${rec.nfe}) to Context Tower (${rec.committedPrefixLen}/${rec.kvCapacity}).`,
        );
      }
    });
  });

  btnStepNext.addEventListener('click', () => {
    runSafe('Next Step failed', () => {
      session = stepTwoTowerSessionNext(session);
      const rec = getActiveRecord(session);
      setStatus(
        'info',
        `Stepped to #${rec.identity.recordIndex} (${rec.kind}) -> phase ${rec.lifecyclePhase}.`,
      );
    });
  });

  // History & Branch Controls
  btnNavBack.addEventListener('click', () => {
    runSafe('Navigate Back failed', () => {
      session = navigateTwoTowerSessionBack(session);
      const branch = getActiveBranch(session);
      const rec = getActiveRecord(session);
      setStatus(
        'info',
        `Rewound to #${branch.cursorIndex} (${rec.kind}) on ${branch.branchId}. Stepping here will non-destructively auto-fork.`,
      );
    });
  });

  btnNavForward.addEventListener('click', () => {
    runSafe('Navigate Forward failed', () => {
      session = navigateTwoTowerSessionForward(session);
      const branch = getActiveBranch(session);
      const rec = getActiveRecord(session);
      setStatus('info', `Advanced cursor to #${branch.cursorIndex} (${rec.kind}) on ${branch.branchId}.`);
    });
  });

  btnForkBranch.addEventListener('click', () => {
    runSafe('Fork Branch failed', () => {
      const parentBranch = getActiveBranch(session);
      session = forkTwoTowerSessionBranch(session);
      const nextBranch = getActiveBranch(session);
      setStatus(
        'info',
        `Forked "${nextBranch.label}" (${nextBranch.branchId}) from ${parentBranch.branchId}@#${parentBranch.cursorIndex}.`,
      );
    });
  });

  branchSelect.addEventListener('change', () => {
    runSafe('Switch Branch failed', () => {
      const targetId = branchSelect.value;
      session = switchTwoTowerSessionBranch(session, targetId);
      const branch = getActiveBranch(session);
      setStatus(
        'info',
        `Switched active branch to "${branch.label}" (${branch.branchId}, cursor #${branch.cursorIndex}/${branch.records.length - 1}).`,
      );
    });
  });

  btnDeleteBranch.addEventListener('click', () => {
    runSafe('Delete Branch failed', () => {
      // Active-branch-safe deletion: pick the most recently added inactive branch.
      let target = null;
      for (let i = session.branches.length - 1; i >= 0; i--) {
        if (session.branches[i].branchId !== session.activeBranchId) {
          target = session.branches[i];
          break;
        }
      }
      if (!target) {
        setStatus('warn', 'No inactive branch available to delete.');
        return;
      }
      session = deleteTwoTowerSessionBranch(session, target.branchId);
      setStatus('info', `Deleted inactive branch "${target.label}" (${target.branchId}).`);
    });
  });

  btnPruneTraces.addEventListener('click', () => {
    runSafe('Prune Traces failed', () => {
      const before = measureTwoTowerSessionRetainedBytes(session).totalRetainedBytes;
      session = pruneInactiveSessionTraces(session);
      const after = measureTwoTowerSessionRetainedBytes(session).totalRetainedBytes;
      const freed = Math.max(0, before - after);
      setStatus(
        'info',
        `Pruned inactive traces: ${formatBytes(before)} -> ${formatBytes(after)} (freed ${formatBytes(freed)}; lazy 0-ULP reconstruction preserved).`,
      );
    });
  });

  traceBaySelect.addEventListener('change', () => {
    inspectedBayId = Number(traceBaySelect.value);
    render();
    options?.onInspectedBayChange?.(inspectedBayId);
  });

  // ---------------------------------------------------------------------------
  // Render Sub-Sections
  // ---------------------------------------------------------------------------
  function renderTokenTape(record: TwoTowerSessionRecord): void {
    tokenTapeContainer.textContent = '';
    const ctxState = record.contextSnapshot.ctxState;
    const P = record.committedPrefixLen;
    const kvCap = record.kvCapacity;

    const committedStartThisStep =
      record.kind === 'ar_step' ||
      record.kind === 'mock_ar_step' ||
      record.kind === 'diffusion_block_commit'
        ? record.identity.absoluteTokenRange[0]
        : -1;

    // 1. Committed Context prefix [0..P-1]
    for (let pos = 0; pos < P; pos++) {
      const tokId = ctxState.prefixTokenIds[pos];
      const tokLabel = ctxState.prefixTokenLabels[pos] ?? `id:${tokId}`;
      const isFresh = committedStartThisStep !== -1 && pos >= committedStartThisStep;
      const chip = el(
        'div',
        `tt-token-chip ${isFresh ? 'tt-token-chip--context-new' : 'tt-token-chip--context'}`,
      );
      chip.dataset.tokenRole = 'context';
      const top = el('div', 'tt-token-pos');
      top.append(el('span', undefined, `t${pos}`), el('span', undefined, `#${tokId}`));
      const word = el('div', 'tt-token-word', tokLabel);
      const tag = el('div', 'tt-token-tag', isFresh ? 'ctx+new' : 'ctx');
      chip.append(top, word, tag);
      tokenTapeContainer.append(chip);
    }

    // 2. Active diffusion block [P..P+B-1]
    const activeBlock = record.activeBlock;
    let nextFreeStart = P;
    if (activeBlock !== null) {
      const B = activeBlock.blockConfig.blockSize;
      nextFreeStart = P + B;
      const newlyCommittedSet = new Set<number>(
        record.kind === 'diffusion_confidence_commit' ? record.unmaskStep.committedIndices : [],
      );
      const isCompletedUncommitted = activeBlock.status === 'completed_uncommitted';

      for (let j = 0; j < B; j++) {
        const absPos = activeBlock.absoluteTokenStart + j;
        const tokId = activeBlock.currentTokens[j];
        const tokLabel = activeBlock.currentTokenLabels[j];
        const isMask = tokId === session.model.maskTokenId;
        const isNewlyCommitted = newlyCommittedSet.has(j);

        let cls = 'tt-token-chip--block-locked';
        let tagText = 'locked';
        let role = 'block-locked';

        if (isMask) {
          cls = 'tt-token-chip--mask';
          tagText = 'masked';
          role = 'block-mask';
        } else if (isCompletedUncommitted) {
          cls = 'tt-token-chip--block-complete';
          tagText = isNewlyCommitted ? 'done·uncomm' : 'uncommitted';
          role = 'block-complete-uncommitted';
        } else if (isNewlyCommitted) {
          cls = 'tt-token-chip--block-new';
          tagText = 'unmasked';
          role = 'block-new';
        }

        const chip = el('div', `tt-token-chip ${cls}`);
        chip.dataset.tokenRole = role;
        const top = el('div', 'tt-token-pos');
        top.append(el('span', undefined, `t${absPos}`), el('span', undefined, `s${j}`));
        const word = el('div', 'tt-token-word', tokLabel);
        const tag = el('div', 'tt-token-tag', tagText);
        chip.append(top, word, tag);
        tokenTapeContainer.append(chip);
      }
    }

    // 3. Remaining free KV capacity slots
    for (let pos = nextFreeStart; pos < kvCap; pos++) {
      const chip = el('div', 'tt-token-chip tt-token-chip--free');
      chip.dataset.tokenRole = 'free';
      const top = el('div', 'tt-token-pos');
      top.append(el('span', undefined, `t${pos}`));
      const word = el('div', 'tt-token-word', '—');
      const tag = el('div', 'tt-token-tag', 'free');
      chip.append(top, word, tag);
      tokenTapeContainer.append(chip);
    }

    tapeSummary.textContent =
      activeBlock !== null
        ? `Context P=${P} | Active block b${activeBlock.blockId} [${activeBlock.absoluteTokenRange[0]}..${activeBlock.absoluteTokenRange[1]}] (masked ${activeBlock.nMasked}/${activeBlock.blockConfig.blockSize}, uncommitted) | Free ${kvCap - nextFreeStart}`
        : `Context P=${P} | No active block | Free ${kvCap - P} / ${kvCap}`;
  }

  function renderSamplerPanel(record: TwoTowerSessionRecord): void {
    samplerBody.textContent = '';

    const activeBlock = record.activeBlock;
    const found =
      activeBlock !== null
        ? findLatestConfidenceCommitForBlock(session, activeBlock.blockId)
        : null;

    if (!found) {
      samplerTitleRight.textContent =
        record.kind === 'diffusion_refinement'
          ? `Block b${record.blockConfig.blockId} r${record.refinementIndex} denoised (awaiting Commit Confidence)`
          : 'Awaiting diffusion_confidence_commit';

      const empty = el('div', 'tt-empty');
      if (record.kind === 'diffusion_refinement') {
        empty.textContent =
          `Denoiser pass r=${record.refinementIndex} (tModel=${record.tModel.toFixed(3)}) completed with xt unchanged. ` +
          `Click "Commit Confidence" to inspect Raw LM-Head vs. Constrained MDLM vs. Unscaled Confidence decisions.`;
      } else if (record.kind === 'ar_step' || record.kind === 'mock_ar_step') {
        empty.textContent =
          `${record.kind === 'mock_ar_step' ? 'Mock-AR' : 'AR'} step committed "${record.selectedTokenLabel}" ` +
          `(id=${record.selectedTokenId}, prob=${(record.selectionProbs[record.selectedTokenId] * 100).toFixed(2)}%). ` +
          `MDLM confidence-unmasking breakdown populates during Mask-Diffusion confidence commit steps.`;
      } else {
        empty.textContent =
          'No confidence-unmasking step on this record yet. Run Init Block -> Run Denoiser Pass -> Commit Confidence to inspect per-slot MDLM sampler breakdown.';
      }
      samplerBody.append(empty);
      return;
    }

    const { presentation: pres, variateProvenance, refinementIndex, isFromActiveRecord } = found;
    const presetLabel = formatConfidencePresetLabel(pres.confidenceThreshold);
    samplerTitleRight.textContent = isFromActiveRecord
      ? `Live step r=${refinementIndex} (${pres.stepIndex + 1}/${pres.stepsPerBlock})`
      : `Latest commit on active block (r=${refinementIndex})`;

    const meta = el('div', 'tt-sampler-meta');
    meta.append(
      el('span', 'tt-badge tt-badge--amber', presetLabel),
      el(
        'span',
        'tt-badge tt-badge--violet',
        `step ${pres.stepIndex + 1}/${pres.stepsPerBlock} · rem=${pres.remainingSteps} · t=${pres.tModel.toFixed(3)}`,
      ),
      el(
        'span',
        'tt-badge tt-badge--cyan',
        `above γ: ${pres.numAboveThreshold} · floor: ${pres.minCommitFloor} · committed: ${pres.tokensCommittedCount}`,
      ),
      el('span', 'tt-badge tt-badge--emerald', `commitReason: ${pres.commitReason}`),
      el(
        'span',
        'tt-badge tt-badge--muted',
        `masked ${pres.nMaskedBefore} -> ${pres.nMaskedAfter}${pres.earlyCompleted ? ' (early complete)' : ''}`,
      ),
    );
    samplerBody.append(meta);

    // Deterministic variate provenance when T > 0
    if (pres.temperature > 0) {
      const varBox = el('div', 'tt-variate-box');
      varBox.style.marginTop = '6px';
      varBox.textContent =
        `Gumbel-max T=${pres.temperature} | variateMode=${variateProvenance.mode} | ` +
        `seed=${variateProvenance.samplingSeed ?? 'explicit'} | ` +
        `variateDigest=${variateProvenance.variateDigest ?? 'none'} | ` +
        `derivationKey=${variateProvenance.derivationKey ?? 'none'}`;
      samplerBody.append(varBox);
    }

    const tableWrap = el('div', 'tt-table-wrap');
    tableWrap.style.marginTop = '8px';
    const table = el('table', 'tt-table');

    const thead = el('thead');
    const headRow = el('tr');
    for (const colTitle of [
      'Slot / Abs Pos',
      'Before -> After',
      'Raw LM-Head Top-1',
      'Constrained MDLM Top-1',
      'Selected Candidate',
      'Unscaled Confidence',
      'Commitment Decision',
    ]) {
      headRow.append(el('th', undefined, colTitle));
    }
    thead.append(headRow);
    table.append(thead);

    const tbody = el('tbody');
    for (const slot of pres.slots) {
      const tr = el('tr');

      // 1. Slot / Abs Pos
      const tdSlot = el('td');
      tdSlot.textContent = `s${slot.slotIndex} · pos ${slot.absolutePosition ?? '—'}`;
      tr.append(tdSlot);

      // 2. Before -> After Token
      const tdTok = el('td');
      tdTok.textContent = `${slot.tokenLabelBefore} -> ${slot.tokenLabelAfter}`;
      const tokSub = el('span', 'tt-subcell', `id ${slot.tokenIdBefore} -> ${slot.tokenIdAfter}`);
      tdTok.append(tokSub);
      tr.append(tdTok);

      // 3. Raw LM-Head Top-1
      const tdRaw = el('td');
      tdRaw.textContent = `${slot.rawLmHead.top1TokenLabel} (${(slot.rawLmHead.top1Prob * 100).toFixed(2)}%)`;
      const rawSub = el(
        'span',
        'tt-subcell',
        `raw [MASK]=${(slot.rawLmHead.maskTokenProb * 100).toFixed(2)}%`,
      );
      tdRaw.append(rawSub);
      tr.append(tdRaw);

      // 4. Constrained MDLM Top-1
      const tdMdlm = el('td');
      const topMdlm = slot.constrainedMdlm.topK[0];
      tdMdlm.textContent = topMdlm
        ? `${topMdlm.tokenLabel} (${(topMdlm.prob * 100).toFixed(2)}%)`
        : '—';
      const mdlmSub = el(
        'span',
        'tt-subcell',
        slot.constrainedMdlm.isCarryOverDelta
          ? '[MASK]=0.0000 · carry-over delta'
          : '[MASK]=0.0000 (excluded)',
      );
      tdMdlm.append(mdlmSub);
      tr.append(tdMdlm);

      // 5. Selected Candidate
      const tdSel = el('td');
      tdSel.textContent = `${slot.selection.selectedTokenLabel} (#${slot.selection.selectedTokenId})`;
      const selBadge = el(
        'span',
        `tt-subcell`,
        slot.selection.samplingMode,
      );
      tdSel.append(selBadge);
      tr.append(tdSel);

      // 6. Unscaled Confidence (STRICTLY from confidenceDisplay.displayText — NEVER Infinity!)
      const tdConf = el('td');
      if (slot.confidenceDisplay.kind === 'locked_prior') {
        tdConf.append(
          el('span', 'tt-badge tt-badge--indigo', slot.confidenceDisplay.displayText),
        );
      } else {
        const badgeCls = slot.confidenceDisplay.exceedsThreshold
          ? 'tt-badge--emerald'
          : 'tt-badge--muted';
        tdConf.append(
          el('span', `tt-badge ${badgeCls}`, slot.confidenceDisplay.displayText),
        );
        tdConf.append(
          el(
            'span',
            'tt-subcell',
            slot.confidenceDisplay.exceedsThreshold
              ? `> γ (${pres.confidenceThreshold.toFixed(2)})`
              : `<= γ (${pres.confidenceThreshold.toFixed(2)})`,
          ),
        );
      }
      tr.append(tdConf);

      // 7. Commitment Decision
      const tdDec = el('td');
      tdDec.append(el('span', `tt-badge ${decisionBadgeVariant(slot.decision)}`, slot.decision));
      tr.append(tdDec);

      tbody.append(tr);
    }

    table.append(tbody);
    tableWrap.append(table);
    samplerBody.append(tableWrap);
  }

  function renderHistoryList(): void {
    const branch = getActiveBranch(session);
    historyBranchInfo.textContent =
      `${branch.label} (${branch.branchId}) · cursor #${branch.cursorIndex} / ${branch.records.length - 1}` +
      (branch.parentBranchId
        ? ` · forked from ${branch.parentBranchId}@#${branch.forkPointIndex}`
        : ' · root branch');

    historyListContainer.textContent = '';
    for (let idx = 0; idx < branch.records.length; idx++) {
      const rec = branch.records[idx];
      const isActive = idx === branch.cursorIndex;
      const row = el('button', `tt-history-row${isActive ? ' is-active' : ''}`);
      row.type = 'button';
      if (isActive) {
        row.setAttribute('aria-current', 'step');
      }
      const bTag = rec.identity.blockId !== null ? `b${rec.identity.blockId}` : '—';
      const rTag = rec.identity.refinementIndex !== null ? `/r${rec.identity.refinementIndex}` : '';
      const brCell = `${bTag}${rTag}`;
      const rangeCell = `pos[${rec.identity.absoluteTokenRange[0]}..${rec.identity.absoluteTokenRange[1]}]`;

      row.append(
        el('span', undefined, `#${idx}`),
        el('span', undefined, rec.kind),
        el('span', undefined, brCell),
        el('span', undefined, rangeCell),
        el('span', undefined, rec.lifecyclePhase),
      );

      row.addEventListener('click', () => {
        runSafe('History navigation failed', () => {
          session = navigateTwoTowerSession(session, idx);
          setStatus('info', `Jumped to record #${idx} (${rec.kind}) on ${branch.branchId}.`);
        });
      });

      historyListContainer.append(row);
    }
  }

  function renderInspector(record: TwoTowerSessionRecord): void {
    // 1. Identity Grid
    identityGrid.textContent = '';
    const id = record.identity;
    const cfg = record.configAtStep;
    const kvItems: ReadonlyArray<{ readonly label: string; readonly value: string; readonly full?: boolean }> = [
      { label: 'modelId', value: id.modelId, full: true },
      { label: 'twoTowerFingerprint', value: id.twoTowerFingerprint },
      { label: 'sessionId / branchId', value: `${id.sessionId} / ${id.branchId}` },
      { label: 'recordId', value: id.recordId, full: true },
      { label: 'parentRecordId', value: id.parentRecordId ?? 'null (root)', full: true },
      { label: 'sourceCacheDigest', value: id.sourceCacheDigest },
      { label: 'resultCacheDigest', value: id.resultCacheDigest },
      {
        label: 'blockId / refinementIndex',
        value: `block=${id.blockId ?? 'null'} · ref=${id.refinementIndex ?? 'null'}`,
      },
      {
        label: 'absoluteTokenRange',
        value: `[${id.absoluteTokenRange[0]}..${id.absoluteTokenRange[1]}] (start=${id.absoluteTokenStart})`,
      },
      {
        label: 'configAtStep (frozen)',
        value: `mode=${cfg.mode} · B=${cfg.blockSize} · K=${cfg.stepsPerBlock} · γ=${cfg.confidenceThreshold.toFixed(2)} · T=${cfg.temperature} · seed=${cfg.samplingSeed} · eos=[${cfg.eosTokenIds.join(',')}]`,
        full: true,
      },
    ];
    for (const item of kvItems) {
      const box = el('div', `tt-kv-item${item.full ? ' tt-kv-item--full' : ''}`);
      box.append(el('dt', undefined, item.label), el('dd', undefined, item.value));
      identityGrid.append(box);
    }

    // 2. Context Execution Traces Sub-Card
    contextTraceCard.textContent = '';
    const latestCtx = getLatestContextExecution(session);
    const ctxHead = el('div', 'tt-subcard-head');
    ctxHead.append(
      el('span', 'tt-subcard-title', `Context Execution Traces (Bay ${String(inspectedBayId).padStart(2, '0')})`),
      el(
        'span',
        'tt-badge tt-badge--cyan',
        latestCtx
          ? `from #${latestCtx.branchCursorIndex} ${latestCtx.record.kind} (${latestCtx.execution.segments.length} seg)`
          : 'none',
      ),
    );
    contextTraceCard.append(ctxHead);

    if (latestCtx) {
      const segList = el('div', 'tt-segment-list');
      for (const seg of latestCtx.execution.segments) {
        const trace = seg.getLayerTrace(inspectedBayId);
        const originBadgeCls =
          trace.traceOrigin === 'context_computed' ? 'tt-badge--emerald' : 'tt-badge--amber';

        const segItem = el('div', 'tt-segment-item');
        const segTop = el('div', 'tt-segment-top');
        segTop.append(
          el(
            'span',
            undefined,
            `seg ${seg.segmentIndex + 1}/${seg.segmentCount}: ${seg.segmentRole} · range [${seg.tokenRange[0]}..${seg.tokenRange[1]}] (${seg.tokens} tok)`,
          ),
          el('span', `tt-badge ${originBadgeCls}`, trace.traceOrigin),
        );
        const segExec = el(
          'div',
          'tt-segment-detail',
          `stepExecutionId=${seg.stepExecutionId} · prefix=${seg.prefixTokens} -> total=${seg.totalTokens}`,
        );
        const segTel = el('div', 'tt-segment-detail', formatBayTelemetrySummary(trace));
        segItem.append(segTop, segExec, segTel);
        segList.append(segItem);
      }
      contextTraceCard.append(segList);
    }

    // 3. Persistent Cache Sub-Card
    persistentCacheCard.textContent = '';
    const snap = record.contextSnapshot;
    const ctxState = snap.ctxState;

    let mambaFixedBytes = 0;
    for (const m of ctxState.mambaStates.values()) {
      mambaFixedBytes += m.convState.byteLength + m.ssmState.byteLength;
    }
    let gqaAllocatedBytes = 0;
    let gqaPopulatedBytes = 0;
    for (const g of ctxState.gqaCaches.values()) {
      gqaAllocatedBytes += g.K.byteLength + g.V.byteLength;
      gqaPopulatedBytes += 2 * g.nKv * g.dHead * g.seqLen * 4;
    }

    const cacheHead = el('div', 'tt-subcard-head');
    cacheHead.append(
      el('span', 'tt-subcard-title', 'Persistent Context Cache Ownership'),
      el('span', 'tt-badge tt-badge--cyan', snap.stateOrigin),
    );

    const cacheDl = el('dl', 'tt-kv-grid');
    for (const [label, value, full] of [
      [
        `${ctxState.mambaStates.size} Mamba-2 Fixed Bays`,
        `${formatBytes(mambaFixedBytes)} (fixed; +mambaS2 for Mock-AR)`,
        false,
      ],
      [
        `${ctxState.gqaCaches.size} GQA KV Bays (cap=${ctxState.maxTokens})`,
        `${ctxState.processedTokens}/${ctxState.maxTokens} tok · ${gqaPopulatedBytes.toLocaleString('en-US')} / ${gqaAllocatedBytes.toLocaleString('en-US')} B`,
        false,
      ],
      [`${ctxState.moeLayerIds.length} Stateless MoE Bays`, '0 B (no sequence cache)', false],
      ['snapshotDigest', snap.snapshotDigest, false],
    ] as const) {
      const box = el('div', `tt-kv-item${full ? ' tt-kv-item--full' : ''}`);
      box.append(el('dt', undefined, label), el('dd', undefined, value));
      cacheDl.append(box);
    }
    persistentCacheCard.append(cacheHead, cacheDl);

    // 4. Denoiser Scratch Sub-Card
    denoiserScratchCard.textContent = '';
    const pass = record.denoiserPass;
    const denHead = el('div', 'tt-subcard-head');
    denHead.append(
      el('span', 'tt-subcard-title', `Denoiser Scratchpad (Bay ${String(inspectedBayId).padStart(2, '0')})`),
      pass
        ? el('span', 'tt-badge tt-badge--violet', 'denoiser_scratch')
        : el('span', 'tt-badge tt-badge--muted', 'inactive on this record'),
    );
    denoiserScratchCard.append(denHead);

    if (pass) {
      const bayTrace = pass.getLayerTrace(inspectedBayId);
      const bayMod = pass.getLayerModulation(inspectedBayId);
      const modOrder = modulationOrderForBayKind(bayTrace.kind);
      const modText =
        bayMod === null
          ? 'bypassed (unmodulated Mock-AR)'
          : `‖shift‖₂=${computeL2(bayMod.shift).toFixed(4)} · ‖scale‖₂=${computeL2(bayMod.scale).toFixed(4)} · ‖gate‖₂=${computeL2(bayMod.gate).toFixed(4)} (${modOrder})`;

      const denDl = el('dl', 'tt-kv-grid');
      for (const [label, value, full] of [
        ['passExecutionId', pass.passExecutionId, true],
        ['mode / tModel', `${pass.mode} · tModel=${pass.tModel === null ? 'null' : pass.tModel.toFixed(4)}`, false],
        ['traceOrigin', bayTrace.traceOrigin, false],
        ['Bay adaLN Modulation', modText, true],
        ['Inspected Bay Telemetry', formatBayTelemetrySummary(bayTrace), true],
        [
          'Scratchpad Lifecycle',
          'Ephemeral scratchpad KV & Mamba state discarded immediately after pass (never mutates Context persistent_cache).',
          true,
        ],
      ] as const) {
        const box = el('div', `tt-kv-item${full ? ' tt-kv-item--full' : ''}`);
        box.append(el('dt', undefined, label), el('dd', undefined, value));
        denDl.append(box);
      }
      denoiserScratchCard.append(denDl);
    } else {
      denoiserScratchCard.append(
        el(
          'div',
          'tt-empty',
          'No Denoiser forward pass on this record. Scratchpad KV/Mamba tensors exist only during diffusion_refinement, diffusion_confidence_commit, or mock_ar_step and are discarded after each pass.',
        ),
      );
    }
  }

  function renderMemoryPanel(): void {
    const mem = measureTwoTowerSessionRetainedBytes(session);
    memoryGrid.textContent = '';

    const items: ReadonlyArray<{
      readonly label: string;
      readonly value: string;
      readonly testId?: string;
    }> = [
      {
        label: 'totalRetainedBytes',
        value: formatBytes(mem.totalRetainedBytes),
        testId: 'mem-total-retained',
      },
      {
        label: 'contextSnapshotBytes',
        value: formatBytes(mem.contextSnapshotBytes),
      },
      {
        label: 'blockConditioningBytes',
        value: formatBytes(mem.blockConditioningBytes),
      },
      {
        label: 'uniqueBlockConditioningCount',
        value: String(mem.uniqueBlockConditioningCount),
        testId: 'mem-unique-block-conditioning-count',
      },
      {
        label: 'contextTraceBytes',
        value: formatBytes(mem.contextTraceBytes),
      },
      {
        label: 'denoiserTraceBytes',
        value: formatBytes(mem.denoiserTraceBytes),
      },
      {
        label: 'samplerAndSelectionBytes',
        value: formatBytes(mem.samplerAndSelectionBytes),
      },
      {
        label: 'sharedModelWeightBytes',
        value: formatBytes(mem.sharedModelWeightBytes),
      },
      {
        label: 'records / branches',
        value: `${mem.uniqueRecordCount} rec / ${mem.branchCount} br`,
        testId: 'mem-records-branches',
      },
    ];

    for (const item of items) {
      const cell = el('div', 'tt-mem-cell');
      const dt = el('dt', undefined, item.label);
      const dd = el('dd', undefined, item.value);
      if (item.testId) dd.dataset.testid = item.testId;
      cell.append(dt, dd);
      memoryGrid.append(cell);
    }
  }

  // ---------------------------------------------------------------------------
  // Master Render Loop
  // ---------------------------------------------------------------------------
  function render(): void {
    const record = getActiveRecord(session);
    const caps = getSessionCapabilities(session);
    const cfg = session.pendingConfig;
    const activeEl = document.activeElement;

    // 1. Header Status Badges
    phaseBadge.className = `tt-badge ${phaseBadgeVariant(record.lifecyclePhase)}`;
    phaseBadge.textContent = record.lifecyclePhase;

    const freeSlots = caps.remainingKvCapacity;
    capacityBadge.textContent = `Context ${caps.committedPrefixLen} / ${caps.kvCapacity} (${freeSlots} free)`;

    if (record.lifecyclePhase === 'eos_terminated') {
      capacityWarningBadge.hidden = false;
      capacityWarningBadge.textContent = `EOS terminated (P=${caps.committedPrefixLen})`;
    } else if (record.lifecyclePhase === 'capacity_exhausted') {
      capacityWarningBadge.hidden = false;
      capacityWarningBadge.textContent = `Capacity exhausted (${caps.committedPrefixLen}/${caps.kvCapacity})`;
    } else if (
      !caps.hasActiveBlock &&
      caps.committedPrefixLen + caps.pendingBlockSize > caps.kvCapacity
    ) {
      capacityWarningBadge.hidden = false;
      capacityWarningBadge.textContent = `Block overflow: P+B (${caps.committedPrefixLen}+${caps.pendingBlockSize}) > ${caps.kvCapacity}`;
    } else {
      capacityWarningBadge.hidden = true;
    }

    // 2. Sync Config Inputs (avoid clobbering focused numeric input during typing)
    modeSelect.value = cfg.mode;
    promptSelect.value = selectedPromptPresetId;
    blockSizeSelect.value = String(cfg.blockSize);
    if (activeEl !== stepsInput) {
      stepsInput.value = String(cfg.stepsPerBlock);
    }
    if (activeEl !== thresholdInput) {
      thresholdInput.value = cfg.confidenceThreshold.toFixed(2);
    }
    if (activeEl !== tempInput) {
      tempInput.value = String(cfg.temperature);
    }
    if (activeEl !== seedInput) {
      seedInput.value = String(cfg.samplingSeed);
    }

    presetMiniBtn.classList.toggle('is-active', Math.abs(cfg.confidenceThreshold - 0.22) < 1e-6);
    presetPaperBtn.classList.toggle('is-active', Math.abs(cfg.confidenceThreshold - 0.8) < 1e-6);
    presetCodeBtn.classList.toggle('is-active', Math.abs(cfg.confidenceThreshold - 0.9) < 1e-6);

    // 3. Explicit Transition Buttons strictly driven by getSessionCapabilities(session)
    const tips = describeActionTooltips(session, record, caps);
    btnStepAr.disabled = !caps.canStepAr;
    btnStepAr.title = tips.stepAr;

    btnStepMockAr.disabled = !caps.canStepMockAr;
    btnStepMockAr.title = tips.stepMockAr;

    btnInitBlock.disabled = !caps.canInitDiffusionBlock;
    btnInitBlock.textContent = `Init Block [MASK]^${cfg.blockSize}`;
    btnInitBlock.title = tips.initBlock;

    btnRunRefinement.disabled = !caps.canRunDiffusionRefinement;
    btnRunRefinement.title = tips.runRefinement;

    btnCommitConfidence.disabled = !caps.canCommitRefinementConfidence;
    btnCommitConfidence.title = tips.commitConfidence;

    btnCommitBlock.disabled = !caps.canCommitActiveBlockToContext;
    btnCommitBlock.title = tips.commitBlock;

    btnStepNext.disabled = !caps.canStepNext;
    btnStepNext.textContent = caps.nextTransitionKind
      ? `Next Step (${caps.nextTransitionKind})`
      : 'Next Step';
    btnStepNext.title = tips.stepNext;

    // 4. History & Branching Controls
    btnNavBack.disabled = !caps.canNavigateBack;
    btnNavBack.title = caps.canNavigateBack
      ? 'Navigate to previous record on active branch'
      : 'Disabled: already at root context_prefill record (#0)';

    btnNavForward.disabled = !caps.canNavigateForward;
    btnNavForward.title = caps.canNavigateForward
      ? 'Navigate forward on active branch'
      : 'Disabled: already at branch tail';

    btnForkBranch.disabled = !caps.canForkBranch;
    btnForkBranch.title = caps.canForkBranch
      ? 'Fork a new independent branch from the current cursor'
      : 'Disabled: maxBranches < 2';

    branchSelect.textContent = '';
    for (const br of session.branches) {
      const opt = el(
        'option',
        undefined,
        `${br.label} (${br.branchId} · ${br.records.length} rec @#${br.cursorIndex})`,
      );
      opt.value = br.branchId;
      branchSelect.append(opt);
    }
    branchSelect.value = session.activeBranchId;

    btnDeleteBranch.disabled = !caps.canDeleteAnyBranch;
    if (caps.canDeleteAnyBranch) {
      let targetId = '';
      for (let i = session.branches.length - 1; i >= 0; i--) {
        if (session.branches[i].branchId !== session.activeBranchId) {
          targetId = session.branches[i].branchId;
          break;
        }
      }
      btnDeleteBranch.title = `Delete inactive branch ${targetId} (active branch is protected)`;
    } else {
      btnDeleteBranch.title = 'Disabled: cannot delete the only remaining branch';
    }

    autoForkPill.hidden = !caps.willAutoForkOnMutatingStep;

    // 5. Status Banner
    statusMessageEl.className = `tt-status-message${
      statusBanner.severity === 'error'
        ? ' is-error'
        : statusBanner.severity === 'info'
          ? ' is-info'
          : statusBanner.severity === 'warn'
            ? ' is-warn'
            : ''
    }`;
    statusMessageEl.textContent = statusBanner.text;

    // 6. Panels (render inspector BEFORE memory panel so LRU trace bytes match live UI state)
    renderTokenTape(record);
    renderSamplerPanel(record);
    renderHistoryList();
    renderInspector(record);
    renderMemoryPanel();
  }

  render();

  return {
    syncSession(nextSession: TwoTowerSession, statusText?: string): void {
      session = nextSession;
      if (statusText) {
        statusBanner = { severity: 'info', text: statusText };
      }
      render();
    },
    setInspectedBay(bayId: number): void {
      const clamped = Math.max(1, Math.min(52, Math.trunc(bayId)));
      if (clamped !== inspectedBayId) {
        inspectedBayId = clamped;
        traceBaySelect.value = String(clamped);
        render();
      }
    },
    getSession(): TwoTowerSession {
      return session;
    },
  };
}
