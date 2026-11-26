import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  createDefaultDiffusionModel,
  createDiffusionSession,
  findInvalidUiNumericStrings,
  type DiffusionSessionRecordKind,
} from '../src/diffusion';
import {
  buildDiffusionTimelineDockState,
  computeRovingTabIndex,
  formatTimelineRecordKindBadge,
  isReducedMotionPreferred,
  resolveDiffusionGlobalKeyAction,
} from '../src/diffusion/app/timelineDock';
import { handleTablistKeydown, shouldDeferGlobalShortcut } from '../src/ui/controls';

interface DuckTabButton {
  hidden: boolean;
  disabled: boolean;
  focusCount: number;
  attributes: Record<string, string>;
  getAttribute(name: string): string | null;
  setAttribute(name: string, value: string): void;
  focus(): void;
  contains(node: Node | null): boolean;
}

function makeTabStub(opts?: {
  selected?: boolean;
  hidden?: boolean;
  disabled?: boolean;
}): DuckTabButton {
  const attributes: Record<string, string> = {
    role: 'tab',
    tabindex: opts?.selected ? '0' : '-1',
    'aria-selected': opts?.selected ? 'true' : 'false',
  };
  const stub: DuckTabButton = {
    hidden: opts?.hidden ?? false,
    disabled: opts?.disabled ?? false,
    focusCount: 0,
    attributes,
    getAttribute(name: string) {
      return Object.prototype.hasOwnProperty.call(attributes, name) ? attributes[name]! : null;
    },
    setAttribute(name: string, value: string) {
      attributes[name] = value;
    },
    focus() {
      stub.focusCount += 1;
    },
    contains(node: Node | null) {
      return (node as unknown) === stub;
    },
  };
  return stub;
}

function makeDomTargetStub(opts: {
  tagName: string;
  role?: string | null;
  href?: string | null;
  tabindex?: string | null;
  id?: string;
  isContentEditable?: boolean;
  parentNode?: unknown;
}): EventTarget {
  return {
    tagName: opts.tagName,
    id: opts.id ?? '',
    isContentEditable: opts.isContentEditable ?? false,
    parentNode: opts.parentNode ?? null,
    getAttribute(name: string) {
      if (name === 'role') return opts.role ?? null;
      if (name === 'href') return opts.href ?? null;
      if (name === 'tabindex') return opts.tabindex ?? null;
      return null;
    },
  } as unknown as EventTarget;
}

describe('DF4 Unified Tri-Mode Event Timeline View-Model & WAI-ARIA A11y', () => {
  it('models all 4 record kinds, copy-on-step forks, seeking, branch deletion, and read-only tour state', () => {
    const model = createDefaultDiffusionModel();
    const session = createDiffusionSession(model, {
      config: { mode: 'ar', causalContext: true, blockLength: 4, maxNewTokens: 12, temperature: 0 },
    });

    // Branch 1: prefill -> ar_step
    const rPrefill = session.step();
    const rAr = session.step();
    const branch1Id = rPrefill.branchId;

    // Rewind to prefill and fork Branch 2 in DLM Causal mode
    session.seekRecord(rPrefill.recordId);
    session.updateConfig({
      mode: 'dlm',
      causalContext: true,
      blockLength: 4,
      maxNewTokens: 12,
      temperature: 0,
      samplerAThreshold: null,
    });
    const rDlm = session.step();
    const branch2Id = rDlm.branchId;

    // Rewind to prefill and fork Branch 3 in Linear Spec + LoRA mode
    session.seekRecord(rPrefill.recordId);
    session.updateConfig({
      mode: 'linear_spec_lora',
      causalContext: true,
      blockLength: 4,
      maxNewTokens: 12,
      temperature: 0,
      samplerBThreshold: 0.0,
    });
    const rSpec = session.step();
    const branch3Id = rSpec.branchId;

    expect(rPrefill.kind).toBe('prefill');
    expect(rAr.kind).toBe('ar_step');
    expect(rDlm.kind).toBe('dlm_block');
    expect(rSpec.kind).toBe('spec_cycle');
    expect(new Set([branch1Id, branch2Id, branch3Id]).size).toBe(3);

    const expectedBadges: Readonly<Record<DiffusionSessionRecordKind, string>> = {
      prefill: 'Prefill',
      ar_step: 'AR Step',
      dlm_block: 'DLM Block',
      spec_cycle: 'Spec Cycle',
    };
    for (const [kind, badge] of Object.entries(expectedBadges) as [
      DiffusionSessionRecordKind,
      string,
    ][]) {
      expect(formatTimelineRecordKindBadge(kind)).toBe(badge);
    }

    const dockState = buildDiffusionTimelineDockState(session);
    expect(dockState.branchCount).toBe(3);
    expect(dockState.branches).toHaveLength(3);
    expect(dockState.recordCount).toBe(4);
    expect(dockState.activeBranchId).toBe(branch3Id);
    expect(dockState.activeRecordId).toBe(rSpec.recordId);
    expect(dockState.isReadOnly).toBe(false);
    expect(dockState.memoryReport.withinBudget).toBe(true);
    expect(dockState.withinMemoryCap).toBe(true);
    expect(dockState.memoryBarPct).toBeGreaterThanOrEqual(0);
    expect(dockState.memoryBarPct).toBeLessThanOrEqual(100);
    expect(findInvalidUiNumericStrings(dockState)).toEqual([]);

    const allNodes = dockState.branches.flatMap((b) => b.nodes);
    // Each forked branch inherits the prefix record (rPrefill), giving 2 nodes per branch = 6 lane nodes for 4 unique records
    expect(allNodes).toHaveLength(6);
    expect(new Set(allNodes.map((n) => n.recordId)).size).toBe(4);

    // Exactly one node is active: rSpec on branch3Id
    const activeNodes = allNodes.filter((n) => n.isActive);
    expect(activeNodes).toHaveLength(1);
    expect(activeNodes[0]!.recordId).toBe(rSpec.recordId);
    expect(activeNodes[0]!.isOnActiveBranch).toBe(true);

    // Fork source marker only on rPrefill (across its branch lane appearances)
    const prefillNodes = allNodes.filter((n) => n.recordId === rPrefill.recordId);
    expect(prefillNodes).toHaveLength(3);
    expect(prefillNodes.every((n) => n.isForkSource)).toBe(true);
    for (const n of allNodes.filter((x) => x.recordId !== rPrefill.recordId)) {
      expect(n.isForkSource).toBe(false);
    }

    // Exact kindBadge on each node
    for (const n of allNodes) {
      expect(n.kindBadge).toBe(expectedBadges[n.kind]);
    }

    // canDelete: false on active branch, true on inactive branches
    for (const b of dockState.branches) {
      expect(b.canDelete).toBe(b.branchId !== branch3Id);
    }

    // Seeking to rAr updates activeRecordId / activeBranchId / isActive / canDelete
    session.seekRecord(rAr.recordId);
    const soughtState = buildDiffusionTimelineDockState(session);
    expect(soughtState.activeRecordId).toBe(rAr.recordId);
    expect(soughtState.activeBranchId).toBe(branch1Id);
    const soughtActiveNodes = soughtState.branches.flatMap((b) => b.nodes).filter((n) => n.isActive);
    expect(soughtActiveNodes).toHaveLength(1);
    expect(soughtActiveNodes[0]!.recordId).toBe(rAr.recordId);
    for (const b of soughtState.branches) {
      expect(b.canDelete).toBe(b.branchId !== branch1Id);
    }

    // Deleting inactive branch2 updates branchCount, recordCount, and canDelete
    expect(session.deleteBranch(branch2Id)).toBe(true);
    const afterDelete = buildDiffusionTimelineDockState(session);
    expect(afterDelete.branchCount).toBe(2);
    expect(afterDelete.branches).toHaveLength(2);
    expect(afterDelete.recordCount).toBe(3);
    expect(afterDelete.branches.find((b) => b.branchId === branch2Id)).toBeUndefined();
    expect(afterDelete.branches.find((b) => b.branchId === branch1Id)?.canDelete).toBe(false);
    expect(afterDelete.branches.find((b) => b.branchId === branch3Id)?.canDelete).toBe(true);

    // Read-only (Tour mode) forces isReadOnly === true and canDelete === false across ALL branches
    const readOnlyState = buildDiffusionTimelineDockState(session, { isReadOnly: true });
    expect(readOnlyState.isReadOnly).toBe(true);
    expect(readOnlyState.branches.every((b) => b.canDelete === false)).toBe(true);
    expect(findInvalidUiNumericStrings(readOnlyState)).toEqual([]);

    // Deleting the remaining inactive branch leaves 1 branch with canDelete === false
    expect(session.deleteBranch(branch3Id)).toBe(true);
    const singleBranchState = buildDiffusionTimelineDockState(session);
    expect(singleBranchState.branchCount).toBe(1);
    expect(singleBranchState.branches[0]!.canDelete).toBe(false);
  });

  it('computes WAI-ARIA roving tabindex across counts 0, 3, 5, 7 with wrap-around and ignores unrelated keys', () => {
    expect(computeRovingTabIndex(0, 0, 'ArrowRight')).toBeNull();
    expect(computeRovingTabIndex(0, -1, 'ArrowLeft')).toBeNull();
    expect(computeRovingTabIndex(1, 5, 'Enter')).toBeNull();
    expect(computeRovingTabIndex(1, 5, 'Tab')).toBeNull();
    expect(computeRovingTabIndex(1, 5, 'x')).toBeNull();

    for (const count of [3, 5, 7]) {
      expect(computeRovingTabIndex(0, count, 'ArrowRight')).toBe(1);
      expect(computeRovingTabIndex(count - 1, count, 'ArrowRight')).toBe(0);
      expect(computeRovingTabIndex(0, count, 'ArrowLeft')).toBe(count - 1);
      expect(computeRovingTabIndex(count - 1, count, 'ArrowLeft')).toBe(count - 2);
      expect(computeRovingTabIndex(count - 1, count, 'Home')).toBe(0);
      expect(computeRovingTabIndex(0, count, 'End')).toBe(count - 1);
    }
  });

  it('handles handleTablistKeydown with duck-typed button stubs, roving tabindex, focus movement, and hidden/disabled skipping', () => {
    const tabs = [
      makeTabStub({ selected: true }),
      makeTabStub({ disabled: true }),
      makeTabStub({ hidden: true }),
      makeTabStub(),
      makeTabStub(),
    ];
    const buttons = tabs as unknown as HTMLButtonElement[];

    let selectedIdx = -1;
    let selectedTab: HTMLButtonElement | null = null;
    let prevented = 0;
    let stopped = 0;

    // ArrowRight from index 0 skips disabled (1) and hidden (2) and lands on index 3
    const handled1 = handleTablistKeydown(
      buttons,
      {
        key: 'ArrowRight',
        target: tabs[0] as unknown as EventTarget,
        preventDefault: () => {
          prevented += 1;
        },
        stopPropagation: () => {
          stopped += 1;
        },
      },
      (tab, idx) => {
        selectedTab = tab;
        selectedIdx = idx;
      },
    );

    expect(handled1).toBe(true);
    expect(prevented).toBe(1);
    expect(stopped).toBe(1);
    expect(selectedIdx).toBe(3);
    expect(selectedTab).toBe(buttons[3]);
    expect(tabs[3]!.focusCount).toBe(1);
    expect(tabs.map((t) => t.getAttribute('tabindex'))).toEqual(['-1', '-1', '-1', '0', '-1']);

    // Update aria-selected as app controller does and verify ArrowLeft from 0 wraps to 4
    tabs[0]!.setAttribute('aria-selected', 'false');
    tabs[3]!.setAttribute('aria-selected', 'true');
    const handled2 = handleTablistKeydown(
      buttons,
      {
        key: 'ArrowLeft',
        target: tabs[0] as unknown as EventTarget,
        preventDefault: () => {
          prevented += 1;
        },
        stopPropagation: () => {
          stopped += 1;
        },
      },
      (_tab, idx) => {
        selectedIdx = idx;
      },
    );
    expect(handled2).toBe(true);
    expect(selectedIdx).toBe(4);
    expect(tabs[4]!.focusCount).toBe(1);
    expect(tabs.map((t) => t.getAttribute('tabindex'))).toEqual(['-1', '-1', '-1', '-1', '0']);

    // Home returns to index 0, End returns to index 4
    handleTablistKeydown(
      buttons,
      {
        key: 'Home',
        target: tabs[4] as unknown as EventTarget,
        preventDefault: () => {},
        stopPropagation: () => {},
      },
      (_tab, idx) => {
        selectedIdx = idx;
      },
    );
    expect(selectedIdx).toBe(0);
    expect(tabs[0]!.getAttribute('tabindex')).toBe('0');

    handleTablistKeydown(
      buttons,
      {
        key: 'End',
        target: tabs[0] as unknown as EventTarget,
        preventDefault: () => {},
        stopPropagation: () => {},
      },
      (_tab, idx) => {
        selectedIdx = idx;
      },
    );
    expect(selectedIdx).toBe(4);
    expect(tabs[4]!.getAttribute('tabindex')).toBe('0');

    // Unrelated key does not intercept or fire callback
    let extraCalls = 0;
    const unhandled = handleTablistKeydown(
      buttons,
      {
        key: 'Enter',
        target: tabs[4] as unknown as EventTarget,
        preventDefault: () => {
          extraCalls += 1;
        },
        stopPropagation: () => {
          extraCalls += 1;
        },
      },
      () => {
        extraCalls += 1;
      },
    );
    expect(unhandled).toBe(false);
    expect(extraCalls).toBe(0);
  });

  it('resolves Escape precedence, tour arrow gating, and Space/Enter deferral via shouldDeferGlobalShortcut', () => {
    const canvasStub = makeDomTargetStub({ tagName: 'CANVAS', id: 'gl-canvas' });
    const divStub = makeDomTargetStub({ tagName: 'DIV' });
    const inputStub = makeDomTargetStub({ tagName: 'INPUT' });
    const textareaStub = makeDomTargetStub({ tagName: 'TEXTAREA' });
    const selectStub = makeDomTargetStub({ tagName: 'SELECT' });
    const buttonStub = makeDomTargetStub({ tagName: 'BUTTON' });
    const tabStub = makeDomTargetStub({
      tagName: 'BUTTON',
      role: 'tab',
      parentNode: makeDomTargetStub({ tagName: 'DIV', role: 'tablist' }),
    });
    const tablistChildStub = makeDomTargetStub({
      tagName: 'SPAN',
      parentNode: makeDomTargetStub({ tagName: 'DIV', role: 'tablist' }),
    });

    // Escape precedence: inspector -> pinned cell -> tour -> none
    expect(
      resolveDiffusionGlobalKeyAction('Escape', {
        target: divStub,
        tourActive: true,
        inspectorOpen: true,
        hasPinnedCell: true,
      }),
    ).toBe('close_inspector');
    expect(
      resolveDiffusionGlobalKeyAction('Escape', {
        target: divStub,
        tourActive: true,
        inspectorOpen: false,
        hasPinnedCell: true,
      }),
    ).toBe('clear_pin');
    expect(
      resolveDiffusionGlobalKeyAction('Escape', {
        target: divStub,
        tourActive: true,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('exit_tour');
    expect(
      resolveDiffusionGlobalKeyAction('Escape', {
        target: divStub,
        tourActive: false,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('none');

    // ArrowRight / ArrowLeft operate ONLY when tourActive === true and target is not editable/tablist
    expect(
      resolveDiffusionGlobalKeyAction('ArrowRight', {
        target: divStub,
        tourActive: true,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('tour_next');
    expect(
      resolveDiffusionGlobalKeyAction('ArrowLeft', {
        target: canvasStub,
        tourActive: true,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('tour_prev');
    expect(
      resolveDiffusionGlobalKeyAction('ArrowRight', {
        target: divStub,
        tourActive: false,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('none');

    for (const blockedTarget of [inputStub, textareaStub, selectStub, tabStub, tablistChildStub]) {
      expect(
        resolveDiffusionGlobalKeyAction('ArrowRight', {
          target: blockedTarget,
          tourActive: true,
          inspectorOpen: false,
          hasPinnedCell: false,
        }),
      ).toBe('none');
      expect(
        resolveDiffusionGlobalKeyAction('ArrowLeft', {
          target: blockedTarget,
          tourActive: true,
          inspectorOpen: false,
          hasPinnedCell: false,
        }),
      ).toBe('none');
    }

    // Space / Enter on interactive button/tab controls defer cleanly
    expect(shouldDeferGlobalShortcut(' ', buttonStub)).toBe(true);
    expect(shouldDeferGlobalShortcut('Enter', tabStub)).toBe(true);
    expect(shouldDeferGlobalShortcut(' ', canvasStub)).toBe(false);
    expect(shouldDeferGlobalShortcut('Escape', buttonStub)).toBe(false);

    expect(
      resolveDiffusionGlobalKeyAction(' ', {
        target: buttonStub,
        tourActive: true,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('none');
    expect(
      resolveDiffusionGlobalKeyAction('Enter', {
        target: tabStub,
        tourActive: true,
        inspectorOpen: false,
        hasPinnedCell: false,
      }),
    ).toBe('none');
  });

  it('detects reduced motion preference cleanly with mocked matchMedia and Node fallbacks', () => {
    const queries: string[] = [];
    expect(
      isReducedMotionPreferred((query) => {
        queries.push(query);
        return { matches: true };
      }),
    ).toBe(true);
    expect(queries).toEqual(['(prefers-reduced-motion: reduce)']);

    expect(isReducedMotionPreferred(() => ({ matches: false }))).toBe(false);
    expect(isReducedMotionPreferred()).toBe(false);
    expect(
      isReducedMotionPreferred(() => {
        throw new Error('matchMedia unavailable');
      }),
    ).toBe(false);
  });

  it('passes static a11y and sibling-isolation contracts across spatialApp.ts, spatial.css, and walkthrough.ts', () => {
    const rootDir = path.resolve(process.cwd());
    const spatialAppSrc = fs.readFileSync(
      path.join(rootDir, 'src/diffusion/app/spatialApp.ts'),
      'utf8',
    );
    const spatialCssSrc = fs.readFileSync(
      path.join(rootDir, 'src/diffusion/app/spatial.css'),
      'utf8',
    );
    const walkthroughSrc = fs.readFileSync(
      path.join(rootDir, 'src/diffusion/story/walkthrough.ts'),
      'utf8',
    );

    expect(spatialAppSrc).toContain("setAttribute('role', 'tablist')");
    expect(spatialAppSrc).toContain("setAttribute('aria-live', 'polite')");
    expect(spatialAppSrc).not.toContain('src/twotower/');
    expect(spatialAppSrc).not.toContain('Math.random');

    expect(spatialCssSrc).toContain('@media (prefers-reduced-motion: reduce)');

    expect(walkthroughSrc).not.toContain('src/twotower/');
    expect(walkthroughSrc).not.toContain('Math.random');
    expect(walkthroughSrc).not.toMatch(/\bdocument\./);
    expect(walkthroughSrc).not.toMatch(/\bwindow\./);
  });
});
