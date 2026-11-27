/**
 * Unified Tri-Mode Event Timeline Dock & Pure Headless A11y Helpers
 * (`src/diffusion/app/timelineDock.ts`)
 *
 * Exports:
 *   - Pure headless functions (zero DOM dependencies inside these functions so
 *     Node-only Vitest suites can import and verify them directly):
 *       * `formatTimelineRecordKindBadge`
 *       * `buildDiffusionTimelineDockState`
 *       * `computeRovingTabIndex`
 *       * `resolveDiffusionGlobalKeyAction`
 *       * `isReducedMotionPreferred`
 *   - Compact horizontal editorial DOM controller:
 *       * `createDiffusionTimelineDock`
 *
 * Deterministic, sibling-isolated implementation with strictly finite UI strings.
 */

import { shouldDeferActivationKey as shouldDeferGlobalShortcut } from '../../shared/a11y';
import type { DiffusionSession } from '../session';
import type {
  DiffusionTimelineBranchViewModel,
  DiffusionTimelineDockState,
  DiffusionTimelineNodeViewModel,
} from '../story/types';
import type {
  DiffusionExecutionMode,
  DiffusionSessionRecord,
  DiffusionSessionRecordKind,
} from '../types';

// ============================================================================
// PURE HEADLESS TIMELINE & A11Y HELPERS
// ============================================================================

export function formatTimelineRecordKindBadge(kind: DiffusionSessionRecordKind): string {
  switch (kind) {
    case 'prefill':
      return 'Prefill';
    case 'ar_step':
      return 'AR Step';
    case 'dlm_block':
      return 'DLM Block';
    case 'spec_cycle':
      return 'Spec Cycle';
  }
}

function formatModeBadgeLabel(mode: DiffusionExecutionMode, causalContext: boolean): string {
  switch (mode) {
    case 'ar':
      return 'AR';
    case 'dlm':
      return causalContext ? 'DLM Causal' : 'DLM Bidir';
    case 'linear_spec':
      return 'Linear Spec';
    case 'linear_spec_lora':
      return 'Spec + LoRA';
  }
}

function buildShortNodeLabel(record: DiffusionSessionRecord): string {
  const idx = `#${record.recordIndex}`;
  if (record.kind === 'prefill') {
    return `${idx} Prefill (${record.kvSnapshot.seqLen})`;
  }
  if (record.kind === 'ar_step') {
    return `${idx} AR +${record.stepEmittedTokenIds.length}`;
  }
  if (record.kind === 'dlm_block') {
    const passes = record.dlmMeta?.denoisePassCount ?? 0;
    return `${idx} DLM (${passes}p)`;
  }
  if (record.specMeta) {
    const proposals = Math.max(0, record.specMeta.draftedBlockTokens.length - 1);
    return `${idx} Spec ${record.specMeta.matchedDraftCount}/${proposals}`;
  }
  return `${idx} Spec`;
}

function buildGuardSummary(record: DiffusionSessionRecord): string | null {
  const parts: string[] = [];
  if (record.guardFlags.draftGuardTriggered !== 'none') {
    parts.push(`D18:${record.guardFlags.draftGuardTriggered}`);
  }
  if (record.guardFlags.residualMasksCommittedCount > 0) {
    parts.push(`Residual [MASK]×${record.guardFlags.residualMasksCommittedCount}`);
  }
  if (record.guardFlags.trailingMasksAfterEosCount > 0) {
    parts.push(`Trailing [MASK]×${record.guardFlags.trailingMasksAfterEosCount}`);
  }
  if (record.guardFlags.capacityExhausted) {
    parts.push(`KV Cap ${record.kvSnapshot.seqLen}/${record.kvSnapshot.capacity}`);
  }
  if (parts.length === 0 && record.guardFlags.warnings.length > 0) {
    parts.push(record.guardFlags.warnings[0]!);
  }
  return parts.length > 0 ? parts.join(' · ') : null;
}

export function buildDiffusionTimelineDockState(
  session: DiffusionSession,
  options?: { readonly isReadOnly?: boolean },
): DiffusionTimelineDockState {
  const isReadOnly = options?.isReadOnly === true;
  const state = session.getState();
  const memoryReport = session.measureMemory();

  const forkRecordIds = new Set<string>();
  for (const branch of state.branches) {
    if (branch.forkRecordId !== null) {
      forkRecordIds.add(branch.forkRecordId);
    }
  }

  const branchModels: DiffusionTimelineBranchViewModel[] = [];
  for (const branch of state.branches) {
    const isActiveBranch = branch.branchId === state.activeBranchId;
    const records = session.getBranchRecords(branch.branchId);
    const nodes: DiffusionTimelineNodeViewModel[] = records.map((rec) => {
      const guardSummary = buildGuardSummary(rec);
      const hasGuardWarning = guardSummary !== null;
      const loraActiveOnDraft =
        rec.configAtStep.mode === 'linear_spec_lora' && rec.kind === 'spec_cycle';
      return Object.freeze({
        recordId: rec.recordId,
        parentRecordId: rec.parentRecordId,
        branchId: rec.branchId,
        recordIndex: rec.recordIndex,
        kind: rec.kind,
        kindBadge: formatTimelineRecordKindBadge(rec.kind),
        shortLabel: buildShortNodeLabel(rec),
        mode: rec.configAtStep.mode,
        modeLabel: formatModeBadgeLabel(rec.configAtStep.mode, rec.configAtStep.causalContext),
        causalContext: rec.configAtStep.causalContext,
        kvProvenance: rec.kvMaskProvenance,
        kvSeqLen: rec.kvSnapshot.seqLen,
        kvCapacity: rec.kvSnapshot.capacity,
        emittedCount: rec.stepEmittedTokenIds.length,
        totalGeneratedCount: rec.generatedTokenIds.length,
        lifecyclePhase: rec.lifecyclePhase,
        counters: rec.counters,
        hasGuardWarning,
        guardSummary,
        loraActiveOnDraft,
        isActive: isActiveBranch && rec.recordId === state.activeRecordId,
        isOnActiveBranch: isActiveBranch,
        isForkSource: forkRecordIds.has(rec.recordId),
      });
    });

    const tipRecordId = nodes.length > 0 ? nodes[nodes.length - 1]!.recordId : null;
    const canDelete = !isReadOnly && state.branches.length > 1 && !isActiveBranch;

    branchModels.push(
      Object.freeze({
        branchId: branch.branchId,
        parentBranchId: branch.parentBranchId,
        forkRecordId: branch.forkRecordId,
        isActive: isActiveBranch,
        canDelete,
        nodeCount: nodes.length,
        tipRecordId,
        nodes: Object.freeze(nodes),
      }),
    );
  }

  const maxBytes = Math.max(1, memoryReport.maxRetainedBytes);
  const rawPct = (memoryReport.totalRetainedBytes / maxBytes) * 100;
  const memoryBarPct = Number.isFinite(rawPct) ? Math.max(0, Math.min(100, rawPct)) : 0;

  const usedMiB = memoryReport.totalRetainedBytes / (1024 * 1024);
  const capMiB = memoryReport.maxRetainedBytes / (1024 * 1024);
  const memoryLabel =
    `Branches ${state.branches.length}/${state.bounds.maxBranches} · ` +
    `Records ${state.records.size}/${state.bounds.maxRecords} · ` +
    `${usedMiB.toFixed(2)} / ${capMiB.toFixed(2)} MiB`;

  return Object.freeze({
    branches: Object.freeze(branchModels),
    activeBranchId: state.activeBranchId,
    activeRecordId: state.activeRecordId,
    branchCount: state.branches.length,
    maxBranches: state.bounds.maxBranches,
    recordCount: state.records.size,
    maxRecords: state.bounds.maxRecords,
    memoryReport,
    memoryBarPct,
    memoryLabel,
    withinMemoryCap: memoryReport.withinBudget,
    isReadOnly,
  });
}

/**
 * Pure headless WAI-ARIA roving tabindex resolver for horizontal tablists & strips.
 */
export function computeRovingTabIndex(
  currentIndex: number,
  count: number,
  key: string,
): number | null {
  if (!Number.isInteger(count) || count <= 0) return null;
  const cur = Math.max(0, Math.min(count - 1, Math.trunc(currentIndex)));
  switch (key) {
    case 'ArrowRight':
      return (cur + 1) % count;
    case 'ArrowLeft':
      return (cur - 1 + count) % count;
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    default:
      return null;
  }
}

function isEditableOrTablistTarget(target: EventTarget | null): boolean {
  if (!target || typeof target !== 'object') return false;
  const node = target as Partial<HTMLElement>;
  if (typeof node.closest === 'function') {
    if (
      node.closest(
        'input, textarea, select, [contenteditable="true"], [contenteditable=""], [role="tablist"], [role="tab"]',
      ) !== null
    ) {
      return true;
    }
  }
  let cur: Partial<HTMLElement> | null = node;
  while (cur) {
    const tag = (cur.tagName ?? '').toUpperCase();
    const role = typeof cur.getAttribute === 'function' ? cur.getAttribute('role') : null;
    if (
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      cur.isContentEditable === true ||
      role === 'tab' ||
      role === 'tablist'
    ) {
      return true;
    }
    cur = (cur.parentNode as Partial<HTMLElement> | null) ?? null;
  }
  return false;
}

export type DiffusionGlobalKeyAction =
  | 'none'
  | 'tour_next'
  | 'tour_prev'
  | 'close_inspector'
  | 'exit_tour'
  | 'clear_pin';

/**
 * Pure headless global keyboard shortcut resolver honoring `shouldDeferGlobalShortcut`
 * from `../../ui/controls`, editable form controls, tablists, and Escape precedence.
 */
export function resolveDiffusionGlobalKeyAction(
  key: string,
  opts: {
    readonly target: EventTarget | null;
    readonly tourActive: boolean;
    readonly inspectorOpen: boolean;
    readonly hasPinnedCell: boolean;
  },
): DiffusionGlobalKeyAction {
  if (shouldDeferGlobalShortcut(key, opts.target)) {
    return 'none';
  }

  if (key === 'Escape') {
    if (opts.inspectorOpen) return 'close_inspector';
    if (opts.hasPinnedCell) return 'clear_pin';
    if (opts.tourActive) return 'exit_tour';
    return 'none';
  }

  if (key === 'ArrowRight' || key === 'ArrowLeft') {
    if (!opts.tourActive) return 'none';
    if (isEditableOrTablistTarget(opts.target)) return 'none';
    return key === 'ArrowRight' ? 'tour_next' : 'tour_prev';
  }

  return 'none';
}

/**
 * Pure headless reduced-motion detector with injectable `matchMediaFn` for Node tests.
 */
export function isReducedMotionPreferred(
  matchMediaFn?: (query: string) => { matches: boolean },
): boolean {
  try {
    if (typeof matchMediaFn === 'function') {
      return Boolean(matchMediaFn('(prefers-reduced-motion: reduce)')?.matches);
    }
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      return Boolean(window.matchMedia('(prefers-reduced-motion: reduce)')?.matches);
    }
  } catch {
    return false;
  }
  return false;
}

// ============================================================================
// EDITORIAL HORIZONTAL TIMELINE DOCK DOM CONTROLLER
// ============================================================================

export interface DiffusionTimelineDockCallbacks {
  onSeekRecord(recordId: string): void;
  onDeleteBranch(branchId: string): void;
}

export interface DiffusionTimelineDockController {
  update(state: DiffusionTimelineDockState): void;
  getElement(): HTMLElement;
  dispose(): void;
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

function shortForkLabel(forkRecordId: string): string {
  const parts = forkRecordId.split(':');
  return parts.length >= 2 ? `${parts[0]}:${parts[1]}` : forkRecordId;
}

export function createDiffusionTimelineDock(
  host: HTMLElement,
  callbacks: DiffusionTimelineDockCallbacks,
): DiffusionTimelineDockController {
  const root = el('section', 'df-timeline-dock');
  root.dataset.testid = 'diffusion-timeline-dock';
  root.setAttribute('aria-label', 'Unified tri-mode event timeline');

  const header = el('div', 'df-timeline-header');
  const titleWrap = el('div', 'df-timeline-title-wrap');
  const titleEl = el('span', 'df-timeline-title', 'Event Timeline');
  const readOnlyBadge = el(
    'span',
    'df-timeline-readonly-badge',
    'Tour Session Timeline (Read-Only · Free-Explore Preserved)',
  );
  readOnlyBadge.dataset.testid = 'diffusion-timeline-readonly-badge';
  readOnlyBadge.hidden = true;
  titleWrap.append(titleEl, readOnlyBadge);

  const memoryWrap = el('div', 'df-memory-meter');
  memoryWrap.dataset.testid = 'diffusion-memory-meter';
  const memoryText = el('span', 'df-memory-meter-label', 'Branches 1/12 · Records 0/128 · 0.00 / 12.00 MiB');
  const memoryTrack = el('div', 'df-memory-bar-track');
  memoryTrack.setAttribute('aria-hidden', 'true');
  const memoryFill = el('div', 'df-memory-bar-fill');
  memoryTrack.appendChild(memoryFill);
  memoryWrap.append(memoryText, memoryTrack);

  header.append(titleWrap, memoryWrap);

  const lanesWrap = el('div', 'df-timeline-lanes');
  lanesWrap.setAttribute('role', 'list');

  root.append(header, lanesWrap);
  host.appendChild(root);

  let disposed = false;

  function update(state: DiffusionTimelineDockState): void {
    if (disposed) return;

    root.classList.toggle('is-readonly', state.isReadOnly);
    readOnlyBadge.hidden = !state.isReadOnly;

    memoryText.textContent = state.memoryLabel;
    memoryFill.style.width = `${state.memoryBarPct.toFixed(1)}%`;
    memoryFill.className = 'df-memory-bar-fill';
    if (!state.withinMemoryCap || state.memoryBarPct >= 90) {
      memoryFill.classList.add('is-danger');
    } else if (state.memoryBarPct >= 70) {
      memoryFill.classList.add('is-warn');
    }

    lanesWrap.innerHTML = '';

    for (const branch of state.branches) {
      const row = el('div', 'df-timeline-branch');
      row.dataset.testid = `diffusion-timeline-branch-${branch.branchId}`;
      row.setAttribute('role', 'listitem');
      row.classList.toggle('is-active', branch.isActive);

      const meta = el('div', 'df-timeline-branch-meta');
      const idChip = el('span', 'df-timeline-branch-id', branch.branchId);
      meta.appendChild(idChip);

      if (branch.forkRecordId) {
        const forkTag = el('span', 'df-timeline-fork-tag', `↳ ${shortForkLabel(branch.forkRecordId)}`);
        forkTag.title = `Forked from ${branch.forkRecordId} (${branch.parentBranchId ?? 'root'})`;
        meta.appendChild(forkTag);
      }

      const delBtn = el('button', 'df-timeline-del-btn', '×');
      delBtn.type = 'button';
      delBtn.dataset.testid = `diffusion-timeline-delete-${branch.branchId}`;
      delBtn.setAttribute('aria-label', `Delete branch ${branch.branchId}`);
      delBtn.title = branch.canDelete
        ? `Delete inactive branch ${branch.branchId}`
        : state.isReadOnly
          ? 'Tour timeline is read-only'
          : 'Active or sole remaining branch cannot be deleted';
      delBtn.disabled = state.isReadOnly || !branch.canDelete;
      delBtn.addEventListener('click', () => {
        if (!state.isReadOnly && branch.canDelete) {
          callbacks.onDeleteBranch(branch.branchId);
        }
      });
      meta.appendChild(delBtn);

      const nodesTrack = el('div', 'df-timeline-nodes');
      for (const node of branch.nodes) {
        const btn = el('button', 'df-timeline-node');
        btn.type = 'button';
        btn.dataset.testid = `diffusion-timeline-node-${node.recordId}`;
        btn.classList.add(`df-node--${node.kind}`);
        if (node.mode === 'linear_spec_lora') btn.classList.add('is-lora');
        if (node.kvProvenance === 'block_bidirectional') btn.classList.add('is-bidir');
        if (node.isActive) {
          btn.classList.add('is-active');
          btn.setAttribute('aria-current', 'step');
        }
        if (node.isForkSource) btn.classList.add('is-fork-source');
        if (node.hasGuardWarning) btn.classList.add('has-warning');

        btn.disabled = state.isReadOnly;
        btn.title =
          `${node.kindBadge} (${node.modeLabel}) · KV ${node.kvSeqLen}/${node.kvCapacity} (${node.kvProvenance}) · ` +
          `Actual ${node.counters.actualForwardPasses} / Dec ${node.counters.decodeForwardPasses} / Ref ${node.counters.referenceMethodNfe}` +
          (node.guardSummary ? ` · Guard: ${node.guardSummary}` : '');

        const kindSpan = el('span', 'df-node-kind', node.kindBadge);
        const modeSpan = el('span', 'df-node-mode', node.modeLabel);
        const kvSpan = el('span', 'df-node-kv', `KV ${node.kvSeqLen}`);
        btn.append(kindSpan, modeSpan, kvSpan);

        if (node.hasGuardWarning) {
          const warnSpan = el('span', 'df-node-warn', '!');
          warnSpan.title = node.guardSummary ?? 'Guard warning';
          btn.appendChild(warnSpan);
        }

        btn.addEventListener('click', () => {
          if (!state.isReadOnly) {
            callbacks.onSeekRecord(node.recordId);
          }
        });
        nodesTrack.appendChild(btn);
      }

      row.append(meta, nodesTrack);
      lanesWrap.appendChild(row);
    }
  }

  return {
    update,
    getElement(): HTMLElement {
      return root;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      root.remove();
    },
  };
}
