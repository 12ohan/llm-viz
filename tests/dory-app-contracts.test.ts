/**
 * Milestone DORY6 — Slice 6C Headless App Contract & Isolation Guardrail Test Suite
 * (`tests/dory-app-contracts.test.ts`)
 *
 * Exhaustively verifies in headless Node (zero real DOM / WebGL required):
 *   1. `spatialApp.ts`:
 *      - `createIdleRafScheduler`: idle rest `isActive() === false`, one-shot wake & settle,
 *        re-entrant `requestFrame()`, continuous reasons (`camera_fly`, `playback`) cancellation,
 *        `dtMs` clamping, and clean `dispose()`.
 *      - `createAtlasUploadTracker`: `null` & repeated `atlasKey` dedup (`0` redundant GPU uploads),
 *        round-trip across `null` (View 1 Macro), distinct key increment, and `reset()` on context restore.
 *      - `buildDorySwitcherEntries`: pre-`DORY7` standalone `'dory'` append without mutating
 *        `MODEL_REGISTRY`, and zero duplicates once `'dory'` is present.
 *      - `resolveDoryGlobalKeyAction`: shortcuts `1..7`, `t/r/i/c/?`, tour navigation, modal Escape,
 *        modifier suppression, and editable-target deferral (`INPUT`, `TEXTAREA`, `SELECT`, `textbox`, `combobox`).
 *      - `resolveDoryPickInteraction`: all 7 `D7` views (`macro`, `hypersphere`, `gqa_rope`,
 *        `swiglu`, `kv_compare`, `self_spec`, `bookends`) including loop clamping & layer parity guards.
 *   2. `spatialOverlay.ts`:
 *      - `DORY_MAX_OVERLAY_CELL_NUM_NODES === 64`, `DORY_MAX_OVERLAY_AXIS_HEADER_NODES === 24`,
 *        strict `computeDoryOverlayCellNumBudget` (`<= 64`) & `computeDoryOverlayAxisHeaderBudget` (`<= 24`).
 *      - Static zero-allocation check inside `DorySpatialOverlay.update()` & all post-mount helpers.
 *      - Mocked-DOM execution of `createDorySpatialOverlay` on a zoomed `16x8 = 128`-cell grid asserting
 *        `<= 64` rendered cell-number DOM nodes, `<= 24` axis headers, and `0` post-mount `createElement`
 *        or `createElementNS` calls across `update()`, `clear()`, and `dispose()`.
 *   3. `charts.ts`:
 *      - All 6 pure headless 2D `<svg>` charts (`spanning_subspace`, `exploratory_pca`,
 *        `dual_rope_phasors` SWA & Full, `kv_scaling_256k`, `self_spec_probs` rejection & full-accept,
 *        `tbptl_schematic`) and `buildChartsForDoryScene` dispatch across all 7 views.
 *      - Well-formed standalone `<svg>` markup, finite viewBox/coordinates, zero `NaN`/`Infinity`/`undefined`,
 *        and mandatory epistemic disclosures (`DR6`, `DR7` anti-SLERP, Finding 9 PCA `8D`-metrics & `120 deg`
 *        anchor-map caveat, `S`-invariant `(4+5k)/9` & `S=1,024` `4x` knee, `DR19`/`DR23` Contract A `104`
 *        banks, and `READ-ONLY · NO GRADIENTS` TBPTL depth-1).
 *      - Mocked-DOM `createDoryChartsPanel` signature deduplication & clean disposal.
 *   4. `timelineDock.ts`:
 *      - Prompt presets (`DORY_PROMPT_PRESETS`, zero EOS, SWA crossing flags), `formatDoryExecutedDepthLabel`
 *        across `k in {1,2,4,8}` × `{per_loop, last_loop, self_spec}`, `buildDoryTimelineDockState`,
 *        `computeDoryBranchOptionsSignature`, and mocked-DOM `createDoryTimelineDock` verifying `tourActive`
 *        locks, in-place slider DOM identity, and `<option>` preservation on unchanged branch signatures.
 *   5. Recursive Static Isolation Guardrails (`D4`, Gate 4b/4c, `DR22`):
 *      - Walks EVERY `.ts` file under `src/dory/` recursively via TypeScript's AST parser AND comment-stripped
 *        scanning; asserts `0` imports from Gate 4b/4c forbidden Nano modules (`src/main.ts`,
 *        `ui/{controls,overlay,memoryChart}`, `story/walkthrough`, `model/{spec,memory,state,rcPlayback}`,
 *        `scene/{spine,sheetInspector,hybridPick,*Cutaway,*Spatial,accumulation}`,
 *        `math/{gqaSim,mambaSim,moeSim,hybridFlow,incrementalDecode}`), `0` sibling app imports
 *        (`twotower`, `diffusion`, `uni`), `0` bare runtime npm imports, strict positive conformance to the
 *        `D4` shared leaf allowlist, `0` `Math.random(` calls, and `0` external CSS `@import`/`url()` leaks.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import ts from 'typescript';
import { describe, expect, it, vi } from 'vitest';
import { buildDoryArchReadouts } from '../src/dory/archReadouts';
import {
  buildChartsForDoryScene,
  createDoryChartsPanel,
  DORY_PCA_COMPACT_BADGE,
  renderDualRopePhasorsSvg,
  renderExploratoryPcaSvg,
  renderProductionKvScalingSvg,
  renderSelfSpecProbsSvg,
  renderSpanningSubspaceSvg,
  renderTbptlSchematicSvg,
} from '../src/dory/app/charts';
import {
  buildDorySwitcherEntries,
  createAtlasUploadTracker,
  createIdleRafScheduler,
  resolveDoryGlobalKeyAction,
  resolveDoryPickInteraction,
} from '../src/dory/app/spatialApp';
import {
  computeDoryOverlayAxisHeaderBudget,
  computeDoryOverlayCellNumBudget,
  createDorySpatialOverlay,
  DORY_MAX_OVERLAY_AXIS_HEADER_NODES,
  DORY_MAX_OVERLAY_CELL_NUM_NODES,
} from '../src/dory/app/spatialOverlay';
import {
  buildDoryTimelineDockState,
  computeDoryBranchOptionsSignature,
  createDoryTimelineDock,
  DORY_ALL_LOOP_CHOICES,
  DORY_EXTRAPOLATION_LOOP_CHOICES,
  DORY_PRIMARY_LOOP_CHOICES,
  formatDoryExecutedDepthLabel,
  getDoryPromptPreset,
} from '../src/dory/app/timelineDock';
import {
  DORY_DEFAULT_PROMPT_PRESET_ID,
  DORY_PROMPT_PRESETS,
  DORY_STANDALONE_MODEL_META,
  type DorySvgChartKind,
  type DorySvgChartRenderResult,
} from '../src/dory/app/types';
import { getDefaultDoryModelParameters } from '../src/dory/parameters';
import { buildDoryDualRopeTables } from '../src/dory/rope';
import { buildDorySpatialScene } from '../src/dory/scene/index';
import {
  DEFAULT_DORY_SPATIAL_FOCUS_STATE,
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_SPATIAL_VIEW_MODES,
  DORY_TBPTL_READONLY_BADGE,
  type DoryPickIdentity,
  type DorySelfSpecViewMetadata,
  type DorySpatialScene,
  type DorySpatialViewMode,
} from '../src/dory/scene/types';
import { createDorySession } from '../src/dory/session';
import {
  DORY_DEFAULT_GREEDY_SAMPLING,
  DORY_EOS_TOKEN_IDS,
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  DORY_SPEC_CONTRACT_A_BADGE,
  type DorySession,
} from '../src/dory/types';
import { buildCameraSnapshot, fitOpBoundsCamera, makeViewportMetrics } from '../src/render/camera';
import { MODEL_REGISTRY, type ModelEntryMeta } from '../src/shared/modelRegistry';

const SHARED_PARAMS = getDefaultDoryModelParameters();
const SHARED_ROPE = buildDoryDualRopeTables(SHARED_PARAMS.config);

function toPosix(p: string): string {
  return p.split(sep).join('/');
}

function stripTsComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:\\])\/\/.*$/gm, '$1');
}

function collectTsFilesRecursive(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...collectTsFilesRecursive(full));
    } else if (entry.endsWith('.ts')) {
      out.push(full);
    }
  }
  return out.sort();
}

function extractTsModuleSpecifiers(code: string, fileName: string): readonly string[] {
  const sf = ts.createSourceFile(
    fileName,
    code,
    ts.ScriptTarget.Latest,
    false,
    ts.ScriptKind.TS,
  );
  const specifiers: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (
      ts.isExportDeclaration(node) &&
      node.moduleSpecifier &&
      ts.isStringLiteral(node.moduleSpecifier)
    ) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword &&
      node.arguments.length >= 1 &&
      ts.isStringLiteralLike(node.arguments[0]!)
    ) {
      specifiers.push(node.arguments[0]!.text);
    } else if (
      ts.isImportTypeNode(node) &&
      ts.isLiteralTypeNode(node.argument) &&
      ts.isStringLiteral(node.argument.literal)
    ) {
      specifiers.push(node.argument.literal.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return specifiers;
}

function findAstMathRandomAccesses(code: string, fileName: string): number {
  const sf = ts.createSourceFile(
    fileName,
    code,
    ts.ScriptTarget.Latest,
    false,
    ts.ScriptKind.TS,
  );
  let count = 0;
  const visit = (node: ts.Node): void => {
    if (
      ts.isPropertyAccessExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === 'Math' &&
      node.name.text === 'random'
    ) {
      count += 1;
    } else if (
      ts.isElementAccessExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === 'Math' &&
      ts.isStringLiteralLike(node.argumentExpression) &&
      node.argumentExpression.text === 'random'
    ) {
      count += 1;
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return count;
}

function createSampleSession(opts?: {
  readonly kLoops?: 1 | 2 | 3 | 4 | 6 | 8;
  readonly kvCacheMode?: 'per_loop' | 'last_loop';
  readonly executionMode?: 'standard' | 'self_spec';
  readonly steps?: number;
}): DorySession {
  const preset = getDoryPromptPreset(DORY_DEFAULT_PROMPT_PRESET_ID);
  const execMode = opts?.executionMode ?? 'standard';
  const session = createDorySession({
    params: SHARED_PARAMS,
    ropeTables: SHARED_ROPE,
    initialBranch: {
      promptTokens: preset.tokens,
      kLoops: execMode === 'self_spec' ? 4 : (opts?.kLoops ?? 2),
      kvCacheMode: execMode === 'self_spec' ? 'per_loop' : (opts?.kvCacheMode ?? 'per_loop'),
      executionMode: execMode,
      sampling: DORY_DEFAULT_GREEDY_SAMPLING,
    },
  });
  const steps = opts?.steps ?? 1;
  for (let i = 0; i < steps; i += 1) {
    session.step();
  }
  return session;
}

function makePickIdentity(overrides: Partial<DoryPickIdentity> = {}): DoryPickIdentity {
  return {
    viewMode: 'macro',
    physicalLayerIdx: null,
    pairedLayerIdx: null,
    executedLayerIdx: null,
    stage: null,
    sublayerKind: null,
    loopIteration: null,
    partId: null,
    tensorKey: null,
    qHeadIndex: null,
    kvHeadIndex: null,
    rotaryPlane: null,
    kvBankSlotId: null,
    kvBankKind: null,
    kvPolicy: null,
    cacheRole: null,
    specOffset: null,
    tokenIndex: null,
    absTokenPos: null,
    row: null,
    col: null,
    modelWeightId: SHARED_PARAMS.modelWeightId,
    semanticTag: null,
    ...overrides,
  };
}

interface MockDomNode {
  tagName: string;
  type?: string;
  className: string;
  textContent: string;
  value: string;
  min: string;
  max: string;
  step: string;
  checked: boolean;
  disabled: boolean;
  hidden: boolean;
  tabIndex: number;
  title: string;
  style: Record<string, string>;
  dataset: Record<string, string>;
  attributes: Record<string, string>;
  classes: Set<string>;
  children: MockDomNode[];
  parentNode: MockDomNode | null;
  classList: {
    toggle(cls: string, force?: boolean): boolean;
    add(...cls: string[]): void;
    remove(...cls: string[]): void;
    contains(cls: string): boolean;
  };
  setAttribute(k: string, v: string): void;
  getAttribute(k: string): string | null;
  addEventListener(type: string, fn: (...args: unknown[]) => void): void;
  appendChild(child: MockDomNode): MockDomNode;
  append(...nodes: MockDomNode[]): void;
  removeChild(child: MockDomNode): MockDomNode;
  focus(): void;
  getContext?: () => null;
  innerHTML: string;
}

function createMockDomHarness(): {
  readonly makeNode: (tag: string) => MockDomNode;
  readonly getCreateElementCount: () => number;
  readonly getCreateElementNsCount: () => number;
  readonly resetCounts: () => void;
  readonly findByTestId: (rootNode: MockDomNode, id: string) => MockDomNode | null;
  readonly stubGlobalDocument: () => void;
} {
  let createElementCount = 0;
  let createElementNsCount = 0;

  const buildNode = (tag: string): MockDomNode => {
    const classes = new Set<string>();
    let rawInnerHtml = '';
    const node: MockDomNode = {
      tagName: tag.toUpperCase(),
      className: '',
      textContent: '',
      value: '',
      min: '0',
      max: '0',
      step: '1',
      checked: false,
      disabled: false,
      hidden: false,
      tabIndex: 0,
      title: '',
      style: {},
      dataset: {},
      attributes: {},
      classes,
      children: [],
      parentNode: null,
      classList: {
        toggle(cls: string, force?: boolean): boolean {
          const next = force !== undefined ? force : !classes.has(cls);
          if (next) classes.add(cls);
          else classes.delete(cls);
          return next;
        },
        add(...cls: string[]): void {
          for (const c of cls) classes.add(c);
        },
        remove(...cls: string[]): void {
          for (const c of cls) classes.delete(c);
        },
        contains(cls: string): boolean {
          return classes.has(cls);
        },
      },
      setAttribute(k: string, v: string): void {
        node.attributes[k] = v;
      },
      getAttribute(k: string): string | null {
        return node.attributes[k] ?? null;
      },
      addEventListener(): void {
        // no-op in headless Node mock
      },
      appendChild(child: MockDomNode): MockDomNode {
        child.parentNode = node;
        node.children.push(child);
        return child;
      },
      append(...nodes: MockDomNode[]): void {
        for (const c of nodes) node.appendChild(c);
      },
      removeChild(child: MockDomNode): MockDomNode {
        const idx = node.children.indexOf(child);
        if (idx >= 0) node.children.splice(idx, 1);
        child.parentNode = null;
        return child;
      },
      focus(): void {
        // no-op
      },
      ...(tag.toLowerCase() === 'canvas' ? { getContext: () => null } : {}),
      get innerHTML(): string {
        return rawInnerHtml;
      },
      set innerHTML(v: string) {
        rawInnerHtml = v;
        if (v === '') {
          for (const c of node.children) c.parentNode = null;
          node.children.length = 0;
        }
      },
    };
    return node;
  };

  const findByTestId = (rootNode: MockDomNode, id: string): MockDomNode | null => {
    if (rootNode.dataset.testid === id) return rootNode;
    for (const c of rootNode.children) {
      const hit = findByTestId(c, id);
      if (hit) return hit;
    }
    return null;
  };

  return {
    makeNode: (tag: string) => buildNode(tag),
    getCreateElementCount: () => createElementCount,
    getCreateElementNsCount: () => createElementNsCount,
    resetCounts: () => {
      createElementCount = 0;
      createElementNsCount = 0;
    },
    findByTestId,
    stubGlobalDocument: () => {
      vi.stubGlobal('document', {
        activeElement: null,
        createElement: (tag: string) => {
          createElementCount += 1;
          return buildNode(tag);
        },
        createElementNS: (_ns: string, tag: string) => {
          createElementNsCount += 1;
          return buildNode(tag);
        },
      });
    },
  };
}

function assertCleanSvgChart(chart: DorySvgChartRenderResult, expectedKind: DorySvgChartKind): void {
  expect(chart.kind).toBe(expectedKind);
  expect(chart.title.length).toBeGreaterThan(0);
  expect(chart.ariaLabel.length).toBeGreaterThan(0);
  expect(chart.width).toBeGreaterThan(0);
  expect(chart.height).toBeGreaterThan(0);
  expect(chart.epistemicCaveats.length).toBeGreaterThan(0);

  const svg = chart.svgMarkup;
  expect(svg.startsWith('<svg ')).toBe(true);
  expect(svg.endsWith('</svg>')).toBe(true);
  expect(svg).toContain('xmlns="http://www.w3.org/2000/svg"');
  expect(svg).toContain(`viewBox="0 0 ${chart.width} ${chart.height}"`);
  expect(svg).toContain('role="img"');
  expect(svg).toContain(`dory-chart-svg--${expectedKind}`);

  // Zero NaN, Infinity, or undefined leaks anywhere in SVG markup or metadata
  for (const text of [
    svg,
    chart.title,
    chart.badge ?? '',
    chart.ariaLabel,
    ...chart.epistemicCaveats,
  ]) {
    expect(text).not.toMatch(/\bNaN\b/);
    expect(text).not.toMatch(/Infinity/i);
    expect(text).not.toMatch(/\bundefined\b/);
  }
}

// ============================================================================
// 1. `spatialApp.ts` PURE STATE MACHINES & INTERACTION HELPERS
// ============================================================================

describe('spatialApp.ts pure Node state machines (idle-RAF, atlas dedup, switcher, keys & picks)', () => {
  it('createIdleRafScheduler wakes on requestFrame, coalesces duplicates, settles to rafActive=false, and cancels cleanly', () => {
    let nextHandle = 1;
    const queued = new Map<number, (nowMs: number) => void>();
    const cancelled: number[] = [];
    const frames: { nowMs: number; dtMs: number }[] = [];
    let reRequestOnce = false;

    const scheduler = createIdleRafScheduler({
      requestFrameFn: (cb) => {
        const h = nextHandle++;
        queued.set(h, cb);
        return h;
      },
      cancelFrameFn: (h) => {
        cancelled.push(h);
        queued.delete(h);
      },
      onFrame: (nowMs, dtMs) => {
        frames.push({ nowMs, dtMs });
        if (reRequestOnce) {
          reRequestOnce = false;
          scheduler.requestFrame();
        }
      },
    });

    // At rest initially
    expect(scheduler.isActive()).toBe(false);
    expect(scheduler.getContinuousReasons()).toEqual([]);
    expect(queued.size).toBe(0);

    // Duplicate requestFrame() coalesces into a single queued callback
    scheduler.requestFrame();
    scheduler.requestFrame();
    expect(scheduler.isActive()).toBe(true);
    expect(queued.size).toBe(1);

    // Execute tick -> immediately settles back to idle (`rafActive === false`) with zero spin
    const [h1, cb1] = Array.from(queued.entries())[0]!;
    queued.delete(h1);
    cb1(100);
    expect(frames).toEqual([{ nowMs: 100, dtMs: 0 }]);
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);

    // Re-entrant requestFrame() inside onFrame schedules exactly one follow-up tick, then settles
    reRequestOnce = true;
    scheduler.requestFrame();
    const [h2, cb2] = Array.from(queued.entries())[0]!;
    queued.delete(h2);
    cb2(116);
    expect(scheduler.isActive()).toBe(true);
    expect(queued.size).toBe(1);

    const [h3, cb3] = Array.from(queued.entries())[0]!;
    queued.delete(h3);
    cb3(132);
    expect(frames[frames.length - 1]).toEqual({ nowMs: 132, dtMs: 16 });
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);

    // Multiple continuous reasons keep running until ALL reasons are cleared; dtMs clamped to [0, 100]
    scheduler.setContinuousReason('camera_fly', true);
    scheduler.setContinuousReason('playback', true);
    expect(scheduler.isActive()).toBe(true);
    expect([...scheduler.getContinuousReasons()].sort()).toEqual(['camera_fly', 'playback']);

    const [h4, cb4] = Array.from(queued.entries())[0]!;
    queued.delete(h4);
    cb4(500);
    expect(frames[frames.length - 1]).toEqual({ nowMs: 500, dtMs: 0 });
    expect(scheduler.isActive()).toBe(true);
    expect(queued.size).toBe(1);

    // Large timestamp jump clamps dtMs to 100 ms
    const [h5, cb5] = Array.from(queued.entries())[0]!;
    queued.delete(h5);
    cb5(1250);
    expect(frames[frames.length - 1]).toEqual({ nowMs: 1250, dtMs: 100 });

    // Clearing one reason leaves the other active
    scheduler.setContinuousReason('camera_fly', false);
    expect(scheduler.isActive()).toBe(true);
    expect(scheduler.getContinuousReasons()).toEqual(['playback']);

    // Clearing the final reason immediately cancels queued RAF and returns to idle rest
    scheduler.setContinuousReason('playback', false);
    expect(scheduler.isActive()).toBe(false);
    expect(scheduler.getContinuousReasons()).toEqual([]);
    expect(queued.size).toBe(0);
    expect(cancelled.length).toBeGreaterThanOrEqual(1);

    // Dispose cancels pending frame and permanently ignores subsequent wake attempts
    scheduler.requestFrame();
    expect(scheduler.isActive()).toBe(true);
    scheduler.dispose();
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);

    scheduler.requestFrame();
    scheduler.setContinuousReason('playback', true);
    expect(scheduler.isActive()).toBe(false);
    expect(queued.size).toBe(0);
  });

  it('createAtlasUploadTracker performs 0 redundant uploads on null or repeated atlasKey and resets cleanly on context restore', () => {
    const tracker = createAtlasUploadTracker();

    expect(tracker.getLastAtlasKey()).toBeNull();
    expect(tracker.getUploadCount()).toBe(0);

    // Null atlas causes 0 uploads
    expect(tracker.shouldUpload(null)).toBe(false);
    expect(tracker.getUploadCount()).toBe(0);
    expect(tracker.getLastAtlasKey()).toBeNull();

    // First real dual-hash RG32F key uploads once
    const gqaKey = 'gqa_rope:256x192:a1b2c3d4:e5f60718';
    expect(tracker.shouldUpload(gqaKey)).toBe(true);
    expect(tracker.getUploadCount()).toBe(1);
    expect(tracker.getLastAtlasKey()).toBe(gqaKey);

    // Repeated identical key -> 0 redundant uploads
    expect(tracker.shouldUpload(gqaKey)).toBe(false);
    expect(tracker.getUploadCount()).toBe(1);

    // Round-trip through null atlas and back to same key -> still 0 redundant uploads
    expect(tracker.shouldUpload(null)).toBe(false);
    expect(tracker.shouldUpload(gqaKey)).toBe(false);
    expect(tracker.getUploadCount()).toBe(1);

    // Different key increments uploadCount
    const specKey = 'self_spec:256x128:11223344:55667788';
    expect(tracker.shouldUpload(specKey)).toBe(true);
    expect(tracker.getUploadCount()).toBe(2);
    expect(tracker.getLastAtlasKey()).toBe(specKey);

    // WebGL2 context-lost reset clears lastAtlasKey (preserving total counter) so restore re-uploads
    tracker.reset();
    expect(tracker.getLastAtlasKey()).toBeNull();
    expect(tracker.getUploadCount()).toBe(2);
    expect(tracker.shouldUpload(specKey)).toBe(true);
    expect(tracker.getUploadCount()).toBe(3);
  });

  it('buildDorySwitcherEntries appends standalone Dory metadata without mutating MODEL_REGISTRY or duplicating post-DORY7', () => {
    const beforeLen = MODEL_REGISTRY.length;
    const entries = buildDorySwitcherEntries();

    expect(MODEL_REGISTRY.length).toBe(beforeLen);
    expect(Object.isFrozen(entries)).toBe(true);
    const doryHits = entries.filter((e) => (e.id as string) === 'dory');
    expect(doryHits).toHaveLength(1);
    expect(doryHits[0]!.pathSuffix).toBe(DORY_STANDALONE_MODEL_META.pathSuffix);
    expect(doryHits[0]!.testId).toBe('nav-to-dory');

    // Passing a registry that already contains `'dory'` returns it directly with zero duplicates
    const postDory7 = buildDorySwitcherEntries(entries as readonly ModelEntryMeta[]);
    expect(postDory7).toBe(entries);
    expect(postDory7.filter((e) => (e.id as string) === 'dory')).toHaveLength(1);
  });

  it('resolveDoryGlobalKeyAction routes 1-7 views, tour keys, modal Escape, and defers cleanly on editable inputs', () => {
    const canvasTarget = { tagName: 'CANVAS' } as unknown as EventTarget;

    // 1..7 select all 7 Dory spatial views in order
    for (let i = 0; i < DORY_SPATIAL_VIEW_MODES.length; i += 1) {
      const action = resolveDoryGlobalKeyAction(String(i + 1), {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      });
      expect(action).toEqual({ type: 'select_view', viewMode: DORY_SPATIAL_VIEW_MODES[i]!.id });
    }

    expect(
      resolveDoryGlobalKeyAction('t', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'toggle_tour' });
    expect(
      resolveDoryGlobalKeyAction('R', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'reset_camera' });
    expect(
      resolveDoryGlobalKeyAction('i', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'toggle_inspector' });
    expect(
      resolveDoryGlobalKeyAction('c', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'toggle_charts' });
    expect(
      resolveDoryGlobalKeyAction('?', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'open_shortcuts' });

    // Tour active routes ArrowRight, ArrowLeft, and Escape
    expect(
      resolveDoryGlobalKeyAction('ArrowRight', {
        target: canvasTarget,
        tourActive: true,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'tour_next' });
    expect(
      resolveDoryGlobalKeyAction('ArrowLeft', {
        target: canvasTarget,
        tourActive: true,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'tour_prev' });
    expect(
      resolveDoryGlobalKeyAction('Escape', {
        target: canvasTarget,
        tourActive: true,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'exit_tour' });
    expect(
      resolveDoryGlobalKeyAction('ArrowRight', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'none' });

    // Shortcuts modal traps everything except Escape
    expect(
      resolveDoryGlobalKeyAction('Escape', {
        target: canvasTarget,
        tourActive: true,
        shortcutsModalOpen: true,
      }),
    ).toEqual({ type: 'close_shortcuts' });
    expect(
      resolveDoryGlobalKeyAction('1', {
        target: canvasTarget,
        tourActive: false,
        shortcutsModalOpen: true,
      }),
    ).toEqual({ type: 'none' });

    // Modifier keys suppressed
    expect(
      resolveDoryGlobalKeyAction('r', {
        target: canvasTarget,
        metaKey: true,
        tourActive: false,
        shortcutsModalOpen: false,
      }),
    ).toEqual({ type: 'none' });

    // Editable controls defer ALL shortcuts (including Escape in ledger search input!)
    for (const editableTarget of [
      { tagName: 'INPUT' },
      { tagName: 'TEXTAREA' },
      { tagName: 'SELECT' },
      { tagName: 'DIV', isContentEditable: true },
      { tagName: 'DIV', getAttribute: (k: string) => (k === 'role' ? 'textbox' : null) },
      { tagName: 'DIV', getAttribute: (k: string) => (k === 'role' ? 'combobox' : null) },
    ]) {
      expect(
        resolveDoryGlobalKeyAction('Escape', {
          target: editableTarget as unknown as EventTarget,
          tourActive: true,
          shortcutsModalOpen: false,
        }),
      ).toEqual({ type: 'none' });
      expect(
        resolveDoryGlobalKeyAction('2', {
          target: editableTarget as unknown as EventTarget,
          tourActive: false,
          shortcutsModalOpen: false,
        }),
      ).toEqual({ type: 'none' });
    }
  });

  it('resolveDoryPickInteraction maps ray-picked identities cleanly across all 7 spatial views', () => {
    // View 1 (Macro) & View 2 (Hypersphere): physicalLayerIdx + clamped loopIteration
    expect(
      resolveDoryPickInteraction(
        'macro',
        makePickIdentity({ viewMode: 'macro', physicalLayerIdx: 16, loopIteration: 9 }),
        4,
      ),
    ).toEqual({ selectedPhysicalLayerIdx: 16, selectedLoopIteration: 4 });

    expect(
      resolveDoryPickInteraction(
        'hypersphere',
        makePickIdentity({ viewMode: 'hypersphere', physicalLayerIdx: 22, loopIteration: 2 }),
        4,
      ),
    ).toEqual({ selectedPhysicalLayerIdx: 22, selectedLoopIteration: 2 });

    // View 3 (GQA & RoPE): only even attention sublayers accepted; extracts qHead, plane, score row/col
    expect(
      resolveDoryPickInteraction(
        'gqa_rope',
        makePickIdentity({
          viewMode: 'gqa_rope',
          physicalLayerIdx: 22,
          qHeadIndex: 5,
          rotaryPlane: 3,
          tensorKey: 'attn_probs',
          row: 7,
          col: 4,
        }),
        4,
      ),
    ).toEqual({
      gqaLayerIdx: 22,
      selectedQHead: 5,
      selectedRotaryPlane: 3,
      selectedQueryRow: 7,
      selectedKeyCol: 4,
    });
    // Odd layer ignored for gqaLayerIdx
    expect(
      resolveDoryPickInteraction(
        'gqa_rope',
        makePickIdentity({ viewMode: 'gqa_rope', physicalLayerIdx: 17 }),
        4,
      ),
    ).toBeNull();

    // View 4 (SwiGLU): only odd MLP sublayers accepted; routes channel (0..23) vs outputDim (0..7)
    expect(
      resolveDoryPickInteraction(
        'swiglu',
        makePickIdentity({
          viewMode: 'swiglu',
          physicalLayerIdx: 17,
          tensorKey: 'ffn_sub_norm',
          col: 19,
        }),
        4,
      ),
    ).toEqual({ swigluLayerIdx: 17, swigluChannel: 19 });
    expect(
      resolveDoryPickInteraction(
        'swiglu',
        makePickIdentity({ viewMode: 'swiglu', tensorKey: 'mlp_out', col: 6 }),
        4,
      ),
    ).toEqual({ swigluOutputDim: 6 });

    // View 5 (KV Compare): even attention layers + loop clamping
    expect(
      resolveDoryPickInteraction(
        'kv_compare',
        makePickIdentity({ viewMode: 'kv_compare', physicalLayerIdx: 20, loopIteration: 3 }),
        4,
      ),
    ).toEqual({ kvCompareLayerIdx: 20, selectedLoopIteration: 3 });

    // View 6 (Self-Spec): specOffset + even attention layer
    expect(
      resolveDoryPickInteraction(
        'self_spec',
        makePickIdentity({ viewMode: 'self_spec', physicalLayerIdx: 56, specOffset: 2 }),
        4,
      ),
    ).toEqual({ selfSpecAttentionLayerIdx: 56, selectedVerifyOffset: 2 });

    // View 7 (Bookends): 4 read-only checkpoint cards + untied vocab token lookup
    expect(
      resolveDoryPickInteraction(
        'bookends',
        makePickIdentity({ viewMode: 'bookends', partId: 'bookend_ckpt_nvfp4' }),
        4,
      ),
    ).toEqual({ selectedCheckpointFormat: 'nvfp4' });
    expect(
      resolveDoryPickInteraction(
        'bookends',
        makePickIdentity({ viewMode: 'bookends', partId: 'bookend_ckpt_q5_k_m' }),
        4,
      ),
    ).toEqual({ selectedCheckpointFormat: 'q5_k_m' });
    expect(
      resolveDoryPickInteraction(
        'bookends',
        makePickIdentity({ viewMode: 'bookends', tokenIndex: 12 }),
        4,
      ),
    ).toEqual({ selectedVocabTokenId: 12 });
  });
});

// ============================================================================
// 2. `spatialOverlay.ts` CONTRACTS (<= 64 CAP & ZERO POST-MOUNT ALLOCATIONS)
// ============================================================================

describe('spatialOverlay.ts contracts (<= 64 cell-num cap, <= 24 axis headers & zero per-frame DOM creation)', () => {
  it('enforces DORY_MAX_OVERLAY_CELL_NUM_NODES === 64 and DORY_MAX_OVERLAY_AXIS_HEADER_NODES === 24 strictly', () => {
    expect(DORY_MAX_OVERLAY_CELL_NUM_NODES).toBe(64);
    expect(DORY_MAX_OVERLAY_AXIS_HEADER_NODES).toBe(24);

    // Cell-number cap strictly <= 64
    expect(computeDoryOverlayCellNumBudget(256)).toBe(64);
    expect(computeDoryOverlayCellNumBudget(256, 128)).toBe(64);
    expect(computeDoryOverlayCellNumBudget(64, 64)).toBe(64);
    expect(computeDoryOverlayCellNumBudget(48, 32)).toBe(32);
    expect(computeDoryOverlayCellNumBudget(12.9, 64)).toBe(12);
    expect(computeDoryOverlayCellNumBudget(-5)).toBe(0);
    expect(computeDoryOverlayCellNumBudget(Number.NaN)).toBe(0);
    expect(computeDoryOverlayCellNumBudget(Number.POSITIVE_INFINITY)).toBe(0);

    // Axis-header cap strictly <= 24
    expect(computeDoryOverlayAxisHeaderBudget(100)).toBe(24);
    expect(computeDoryOverlayAxisHeaderBudget(100, 48)).toBe(24);
    expect(computeDoryOverlayAxisHeaderBudget(24, 24)).toBe(24);
    expect(computeDoryOverlayAxisHeaderBudget(18, 10)).toBe(10);
    expect(computeDoryOverlayAxisHeaderBudget(-3)).toBe(0);
    expect(computeDoryOverlayAxisHeaderBudget(Number.NaN)).toBe(0);
  });

  it('statically contains ZERO createElement / createElementNS calls inside DorySpatialOverlay.update() and all post-mount methods', () => {
    const src = readFileSync(resolve(__dirname, '../src/dory/app/spatialOverlay.ts'), 'utf8');
    const stripped = stripTsComments(src);
    const updateIdx = stripped.indexOf('update(input: DorySpatialOverlayUpdateInput)');
    const factoryIdx = stripped.indexOf('export function createDorySpatialOverlay', updateIdx);
    expect(updateIdx).toBeGreaterThan(-1);
    expect(factoryIdx).toBeGreaterThan(updateIdx);
    const postMountBody = stripped.slice(updateIdx, factoryIdx);
    expect(postMountBody).not.toContain('createElement(');
    expect(postMountBody).not.toContain('createElementNS(');
  });

  it('allocates pools once at mount and enforces <= 64 cell-num nodes with 0 post-mount DOM creations across zoomed 128-cell grids', () => {
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();

    try {
      const host = harness.makeNode('div');
      const overlay = createDorySpatialOverlay(host as unknown as HTMLElement, {
        maxCellNumNodes: 128, // intentionally above 64 to prove clamping
        maxAxisHeaderNodes: 40, // intentionally above 24 to prove clamping
      });

      const mountHtmlCount = harness.getCreateElementCount();
      const mountSvgCount = harness.getCreateElementNsCount();
      expect(mountHtmlCount).toBeGreaterThan(64);
      expect(mountSvgCount).toBeGreaterThan(0);
      expect(overlay.getStats().poolCellNumCapacity).toBe(64);

      // Build a real View 3 (GQA & RoPE) scene and zoom tightly onto a synthetic 16x8 = 128-cell grid
      const session = createSampleSession({ kLoops: 2, steps: 2 });
      const baseScene = buildDorySpatialScene(session, 'gqa_rope', {
        ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
        gqaLayerIdx: 16,
      });

      const denseValues = new Float32Array(16 * 8);
      for (let i = 0; i < denseValues.length; i += 1) {
        denseValues[i] = ((i % 17) - 8) * 0.125;
      }

      const denseScene: DorySpatialScene = {
        ...baseScene,
        activeCellGrids: [
          {
            originWorld: [0, 0, 0],
            extent: [12, 0.04, 6],
            axisU: [1, 0, 0],
            axisN: [0, 1, 0],
            cols: 16,
            rows: 8,
            values: denseValues,
            activeCol: 3,
            activeRow: 2,
            format: 'signed',
            colLabels: Array.from({ length: 16 }, (_, c) => `k${c}`),
            rowLabels: Array.from({ length: 8 }, (_, r) => `q${r}`),
          },
        ],
      };

      const metrics = makeViewportMetrics(0, 0, 1600, 1000, 1, 0, 0);
      const fitBounds = { min: [-6, -0.5, -3] as const, max: [6, 0.5, 3] as const };
      const pose = fitOpBoundsCamera(fitBounds, metrics, {
        azimuthDeg: 0,
        elevationDeg: 85,
        fillFraction: 0.94,
      });
      const snapshot = buildCameraSnapshot(pose, fitBounds, metrics);

      // Reset counters after mount: EVERY subsequent update / clear / dispose MUST allocate 0 nodes
      harness.resetCounts();

      const stats1 = overlay.update({
        scene: denseScene,
        snapshot,
        hoveredCell: null,
        selectedPartId: null,
        showCellNumbers: true,
      });

      expect(harness.getCreateElementCount()).toBe(0);
      expect(harness.getCreateElementNsCount()).toBe(0);
      expect(stats1.poolCellNumCapacity).toBe(64);
      expect(stats1.renderedCellNumNodes).toBeGreaterThan(0);
      expect(stats1.renderedCellNumNodes).toBeLessThanOrEqual(64);
      expect(stats1.renderedAxisHeaderNodes).toBeLessThanOrEqual(24);

      // Turning off showCellNumbers hides all numeric cell nodes with 0 allocations
      const statsHidden = overlay.update({
        scene: denseScene,
        snapshot,
        showCellNumbers: false,
      });
      expect(statsHidden.renderedCellNumNodes).toBe(0);
      expect(harness.getCreateElementCount()).toBe(0);
      expect(harness.getCreateElementNsCount()).toBe(0);

      overlay.clear();
      expect(overlay.getStats().renderedCellNumNodes).toBe(0);
      overlay.dispose();
      expect(harness.getCreateElementCount()).toBe(0);
      expect(harness.getCreateElementNsCount()).toBe(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

// ============================================================================
// 3. `charts.ts` CONTRACTS (ALL 6 HEADLESS `<svg>` CHARTS & 7-VIEW DISPATCH)
// ============================================================================

describe('charts.ts contracts (6 headless editorial <svg> charts, epistemic badges & 7-view dispatch)', () => {
  const stdSession = createSampleSession({ kLoops: 4, steps: 2 });
  const specSession = createSampleSession({ executionMode: 'self_spec', steps: 1 });

  it('buildChartsForDoryScene dispatches exact chart kinds across all 7 D7 views with clean SVG & zero NaN/Infinity', () => {
    const readouts = buildDoryArchReadouts({
      activeKLoops: 4,
      activeKvCacheMode: 'per_loop',
      liveSessionMemory: stdSession.measureMemory(),
    });

    const expectedByView: Readonly<Record<DorySpatialViewMode, readonly DorySvgChartKind[]>> = {
      macro: [],
      hypersphere: ['spanning_subspace', 'exploratory_pca'],
      gqa_rope: ['dual_rope_phasors'],
      swiglu: [],
      kv_compare: ['kv_scaling_256k'],
      self_spec: ['self_spec_probs'],
      bookends: ['tbptl_schematic'],
    };

    const seenKinds = new Set<DorySvgChartKind>();

    for (const viewDesc of DORY_SPATIAL_VIEW_MODES) {
      const targetSession = viewDesc.id === 'self_spec' ? specSession : stdSession;
      const scene = buildDorySpatialScene(targetSession, viewDesc.id, {
        ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
        selectedPhysicalLayerIdx: 16,
        selectedLoopIteration: 2,
      });
      const charts = buildChartsForDoryScene(scene, readouts);
      expect(charts.map((c) => c.kind)).toEqual(expectedByView[viewDesc.id]);
      for (const chart of charts) {
        assertCleanSvgChart(chart, chart.kind);
        seenKinds.add(chart.kind);
      }
    }

    expect(seenKinds.size).toBe(6);
  });

  it('enforces mandatory domain disclosures across every individual SVG chart renderer', () => {
    const readouts = buildDoryArchReadouts({
      activeKLoops: 4,
      activeKvCacheMode: 'per_loop',
      liveSessionMemory: stdSession.measureMemory(),
    });

    // 1 & 2. View 2 (`hypersphere`, phys L16 planted >1 alpha channels): anti-SLERP + Exploratory PCA + 120-deg caveat
    const hyperScene = buildDorySpatialScene(stdSession, 'hypersphere', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 2,
      includeCounterfactualCoda: true,
      counterfactualLoop: 1,
    });
    expect(hyperScene.hypersphereMetadata).toBeDefined();

    const spanChart = renderSpanningSubspaceSvg(hyperScene.hypersphereMetadata!.spanningSubspace, {
      width: 420,
      height: 250,
    });
    assertCleanSvgChart(spanChart, 'spanning_subspace');
    expect(spanChart.width).toBe(420);
    expect(spanChart.height).toBe(250);
    expect(spanChart.svgMarkup).toContain('NOT TRIG SLERP');
    expect(spanChart.svgMarkup).toContain('λ=1');
    expect(spanChart.epistemicCaveats.join(' ')).toContain(DORY_ANTI_SLERP_PLAQUE);
    expect(spanChart.epistemicCaveats.join(' ')).toContain(DORY_NLERP_MARKER);
    expect(hyperScene.hypersphereMetadata!.spanningSubspace.extrapolatedChannels.length).toBeGreaterThanOrEqual(2);
    expect(spanChart.epistemicCaveats.join(' ')).toContain('Unclamped extrapolation');

    const pcaChart = renderExploratoryPcaSvg(hyperScene.hypersphereMetadata!);
    assertCleanSvgChart(pcaChart, 'exploratory_pca');
    expect(pcaChart.badge).toBe(DORY_PCA_COMPACT_BADGE);
    expect(pcaChart.svgMarkup).toContain('PROJECTED NORMS &lt;= 1; ALL METRICS COMPUTED IN 8D');
    expect(pcaChart.svgMarkup).toContain('Anchor Jacobian Caveat:');
    expect(pcaChart.svgMarkup).toContain('120°');
    expect(pcaChart.epistemicCaveats).toContain(DORY_EXPLORATORY_PCA_BADGE);
    expect(pcaChart.epistemicCaveats).toContain(DORY_ANCHOR_MAP_CAVEAT_BADGE);
    expect(pcaChart.epistemicCaveats.some((c) => c.includes('DIAGNOSTIC COUNTERFACTUAL'))).toBe(true);

    // 3. View 3 (`gqa_rope`): SWA (L16, theta_S=16, W=8) vs. Full (L22, theta_F=1024) + DR6 order
    const swaScene = buildDorySpatialScene(stdSession, 'gqa_rope', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      gqaLayerIdx: 16,
      selectedLoopIteration: 2,
    });
    const fullScene = buildDorySpatialScene(stdSession, 'gqa_rope', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      gqaLayerIdx: 22,
      selectedLoopIteration: 2,
    });
    const swaChart = renderDualRopePhasorsSvg(swaScene.gqaRopeMetadata!);
    const fullChart = renderDualRopePhasorsSvg(fullScene.gqaRopeMetadata!);
    assertCleanSvgChart(swaChart, 'dual_rope_phasors');
    assertCleanSvgChart(fullChart, 'dual_rope_phasors');
    expect(swaChart.svgMarkup).toContain('θ_S=16');
    expect(swaChart.svgMarkup).toContain('W=8');
    expect(swaChart.svgMarkup).toContain('RoPE -&gt; L2Norm (DR6)');
    for (const pLabel of ['P0', 'P1', 'P2', 'P3']) {
      expect(swaChart.svgMarkup).toContain(pLabel);
      expect(fullChart.svgMarkup).toContain(pLabel);
    }
    expect(fullChart.svgMarkup).toContain('θ_F=1024');
    expect(fullChart.badge).toContain('10,000,000');

    // 4. View 5 (`kv_compare`): 1 -> 256K BF16 scaling, S=1,024 4x knee & S-invariant (4+5k)/9 ratio
    const kvChart = renderProductionKvScalingSvg(
      {
        rows: readouts.productionKvScaling,
        activeKLoops: 4,
        activeKvCacheMode: 'per_loop',
        selectedSeqLen: 1024,
      },
      { width: 440, height: 260 },
    );
    assertCleanSvgChart(kvChart, 'kv_scaling_256k');
    expect(kvChart.badge).toContain('2.667x');
    expect(kvChart.svgMarkup).toContain('S=1,024 SWA Knee');
    expect(kvChart.svgMarkup).toContain('4x slope drop');
    expect(kvChart.svgMarkup).toContain('256K');
    expect(kvChart.epistemicCaveats.some((c) => c.includes('(4+5k)/9'))).toBe(true);

    // Dimension clamping safety against absurd / NaN dims
    const clampedKvChart = renderProductionKvScalingSvg(
      { rows: readouts.productionKvScaling, activeKLoops: 2, activeKvCacheMode: 'last_loop' },
      { width: Number.NaN, height: 50 },
    );
    assertCleanSvgChart(clampedKvChart, 'kv_scaling_256k');
    expect(clampedKvChart.width).toBe(392);
    expect(clampedKvChart.height).toBe(175);
    expect(clampedKvChart.badge).toContain('1.556x');

    // 5. View 6 (`self_spec`): Contract A 104 banks, Paper §6.2 badge, binary64 rule & cap_swa=16 ring
    const specScene = buildDorySpatialScene(specSession, 'self_spec', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
    });
    expect(specScene.selfSpecMetadata).toBeDefined();
    const specMeta = specScene.selfSpecMetadata!;
    const specChart = renderSelfSpecProbsSvg(specMeta, { width: 480, height: 260 });
    assertCleanSvgChart(specChart, 'self_spec_probs');
    expect(specChart.badge).toBe(DORY_SPEC_CONTRACT_A_BADGE);
    expect(specChart.svgMarkup).toContain('Reconstructed from Paper §6.2');
    expect(specChart.svgMarkup).toContain('cap_swa=16 Ring:');
    expect(specChart.svgMarkup).toContain('L1 20 (Reused)');
    expect(specChart.svgMarkup).toContain('L2-4 60 (Deep)');
    expect(specChart.svgMarkup).toContain('Deep 8');
    expect(specChart.svgMarkup).toContain('Shal 8');
    expect(specChart.epistemicCaveats.some((c) => c.includes('output_shallow'))).toBe(true);

    // Exercise BOTH the full-accept ("No Residual Needed") and rejection ("Z=... · 0 extra Shal Coda") branches
    const fullAcceptMeta: DorySelfSpecViewMetadata = {
      ...specMeta,
      residualTopTokens: Object.freeze([]),
      shallowCodaCatchupExecuted: true,
    };
    const fullAcceptChart = renderSelfSpecProbsSvg(fullAcceptMeta);
    assertCleanSvgChart(fullAcceptChart, 'self_spec_probs');
    expect(fullAcceptChart.svgMarkup).toContain('No Residual Needed');
    expect(fullAcceptChart.svgMarkup).toContain('1-tok');

    const rejectedMeta: DorySelfSpecViewMetadata = {
      ...specMeta,
      shallowCodaCatchupExecuted: false,
      cycleSummary: {
        ...specMeta.cycleSummary,
        residualMassZ: 0.4125,
      },
      residualTopTokens: Object.freeze([
        {
          tokenId: 25,
          tokenText: 'loop',
          targetProb: 0.55,
          proposalProb: 0.20,
          residualProb: 0.8485,
        },
        {
          tokenId: 29,
          tokenText: 'sphere',
          targetProb: 0.15,
          proposalProb: 0.0875,
          residualProb: 0.1515,
        },
      ]),
    };
    const rejectedChart = renderSelfSpecProbsSvg(rejectedMeta);
    assertCleanSvgChart(rejectedChart, 'self_spec_probs');
    expect(rejectedChart.svgMarkup).toContain('Z=0.4125 · 0 extra Shal Coda');
    expect(rejectedChart.svgMarkup).toContain('loop');

    // 6. View 7 (`bookends`): strictly READ-ONLY · NO GRADIENTS TBPTL depth-1 & 5 stages
    const tbptlChart = renderTbptlSchematicSvg(readouts.tbptlSchematic);
    assertCleanSvgChart(tbptlChart, 'tbptl_schematic');
    expect(tbptlChart.badge).toBe(DORY_TBPTL_READONLY_BADGE);
    expect(tbptlChart.svgMarkup).toContain('READ-ONLY · NO GRADIENTS');
    expect(tbptlChart.svgMarkup).toContain('TBPTL DEPTH=1');
    expect(tbptlChart.svgMarkup).toContain('EARLY-EXIT LOSSES=0');
    expect(tbptlChart.svgMarkup).toContain('+x Anchor');
    expect(tbptlChart.svgMarkup).toContain('0 fake quant toggles');
    for (const s of ['S1', 'S2', 'S3', 'S4', 'S5']) {
      expect(tbptlChart.svgMarkup).toContain(`>${s}<`);
    }
  });

  it('createDoryChartsPanel deduplicates identical scene renders and clears/disposes cleanly', () => {
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();

    try {
      const host = harness.makeNode('div');
      const panel = createDoryChartsPanel(host as unknown as HTMLElement);
      const readouts = buildDoryArchReadouts({ activeKLoops: 4, activeKvCacheMode: 'per_loop' });
      const hyperScene = buildDorySpatialScene(stdSession, 'hypersphere', {
        ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
        selectedPhysicalLayerIdx: 16,
      });
      const macroScene = buildDorySpatialScene(stdSession, 'macro', DEFAULT_DORY_SPATIAL_FOCUS_STATE);

      harness.resetCounts();
      const kinds1 = panel.renderForScene(hyperScene, readouts);
      expect(kinds1).toEqual(['spanning_subspace', 'exploratory_pca']);
      expect(panel.getRenderedChartKinds()).toEqual(['spanning_subspace', 'exploratory_pca']);
      const firstRenderCreates = harness.getCreateElementCount();
      expect(firstRenderCreates).toBeGreaterThan(0);

      // Re-rendering identical scene hits signature cache -> 0 new DOM nodes allocated
      const kinds2 = panel.renderForScene(hyperScene, readouts);
      expect(kinds2).toEqual(['spanning_subspace', 'exploratory_pca']);
      expect(harness.getCreateElementCount()).toBe(firstRenderCreates);

      // Rendering a chartless view (Macro) clears and hides the panel
      const kinds3 = panel.renderForScene(macroScene, readouts);
      expect(kinds3).toEqual([]);
      expect(panel.getRenderedChartKinds()).toEqual([]);

      panel.dispose();
      expect(panel.getRenderedChartKinds()).toEqual([]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

// ============================================================================
// 4. `timelineDock.ts` CONTRACTS (DEPTH/POLICY TELEMETRY, TOUR LOCK & IN-PLACE DOM)
// ============================================================================

describe('timelineDock.ts contracts (presets, depth/KV badges, DR18 state & DOM identity)', () => {
  it('validates all 4 deterministic prompt presets and depth/KV bank label formatting', () => {
    expect(DORY_PRIMARY_LOOP_CHOICES).toEqual([1, 2, 3, 4]);
    expect(DORY_EXTRAPOLATION_LOOP_CHOICES).toEqual([6, 8]);
    expect(DORY_ALL_LOOP_CHOICES).toEqual([1, 2, 3, 4, 6, 8]);
    expect(DORY_PROMPT_PRESETS).toHaveLength(4);

    for (const preset of DORY_PROMPT_PRESETS) {
      expect(getDoryPromptPreset(preset.id)).toBe(preset);
      expect(preset.tokens.length).toBeGreaterThan(0);
      expect(preset.tokens.length).toBeLessThanOrEqual(DORY_MINI_CONFIG.maxPositionEmbeddings);
      expect(preset.crossesSwaWindow).toBe(preset.tokens.length > DORY_MINI_CONFIG.slidingWindow);
      for (const tok of preset.tokens) {
        expect(tok).toBeGreaterThanOrEqual(0);
        expect(tok).toBeLessThan(DORY_MINI_CONFIG.vocabSize);
        expect(DORY_EOS_TOKEN_IDS).not.toContain(tok);
      }
      const partitionSum = preset.defaultChunkPartition.reduce((acc, n) => acc + n, 0);
      expect(partitionSum).toBe(preset.tokens.length);
    }

    // Physical-72 + paired transformers depth and bank counts
    expect(formatDoryExecutedDepthLabel(1, 'per_loop', 'standard')).toBe(
      '72L (36 paired) · 36 per_loop banks',
    );
    expect(formatDoryExecutedDepthLabel(2, 'per_loop', 'standard')).toBe(
      '112L (56 paired) · 56 per_loop banks',
    );
    expect(formatDoryExecutedDepthLabel(4, 'last_loop', 'standard')).toBe(
      '192L (96 paired) · 36 last_loop banks',
    );
    expect(formatDoryExecutedDepthLabel(4, 'per_loop', 'self_spec')).toBe(
      '192L (96 paired) · 104 Contract A banks',
    );
    expect(formatDoryExecutedDepthLabel(8, 'per_loop', 'standard')).toBe(
      '352L (176 paired) · 176 per_loop banks',
    );
  });

  it('buildDoryTimelineDockState summarizes session cursor, DR18 notice, tour locks & <= 12 MiB memory audit', () => {
    const session = createSampleSession({ kLoops: 2, steps: 2 });
    const branch = session.getActiveBranch()!;

    const liveState = buildDoryTimelineDockState(session, {
      dr18ForkNotice: 'DR18: forked clean k=2 branch',
    });
    expect(liveState.kLoops).toBe(2);
    expect(liveState.kvCacheMode).toBe('per_loop');
    expect(liveState.executionMode).toBe('standard');
    expect(liveState.totalRecordsInBranch).toBe(3); // prefill + 2 decode steps
    expect(liveState.currentStepIndex).toBe(2);
    expect(liveState.isHistoricalCursor).toBe(false);
    expect(liveState.canStepBack).toBe(true);
    expect(liveState.canStepForward).toBe(false);
    expect(liveState.canExecuteStep).toBe(true);
    expect(liveState.executedDepthLabel).toBe('112L (56 paired) · 56 per_loop banks');
    expect(liveState.dr18ForkNotice).toBe('DR18: forked clean k=2 branch');
    expect(liveState.memoryAudit.withinTotalBudgetWithWeights).toBe(true);
    expect(liveState.memoryAudit.withinCheckpointPoolBudget).toBe(true);

    // Historical cursor at record 0 enables step forward and flags isHistoricalCursor
    const histState = buildDoryTimelineDockState(session, {
      activeRecordId: branch.recordIds[0]!,
    });
    expect(histState.isHistoricalCursor).toBe(true);
    expect(histState.canStepBack).toBe(false);
    expect(histState.canStepForward).toBe(true);

    // tourActive === true locks transport and execution
    const tourState = buildDoryTimelineDockState(session, { tourActive: true });
    expect(tourState.tourActive).toBe(true);
    expect(tourState.canStepBack).toBe(false);
    expect(tourState.canStepForward).toBe(false);
    expect(tourState.canExecuteStep).toBe(false);
  });

  it('createDoryTimelineDock preserves slider DOM identity & branch <option> nodes across updates and locks during tourActive', () => {
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();

    try {
      const container = harness.makeNode('div');
      const dock = createDoryTimelineDock(container as unknown as HTMLElement, {
        onStepBack: () => {},
        onStepForward: () => {},
        onScrubIndex: () => {},
        onTogglePlay: () => {},
        onExecuteNextStep: () => {},
        onSelectKLoops: () => {},
        onToggleExtrapolationDepths: () => {},
        onSelectKvCacheMode: () => {},
        onSelectExecutionMode: () => {},
        onSelectPromptPreset: () => {},
        onUpdateSampling: () => {},
        onForkBranch: () => {},
        onSwitchBranch: () => {},
        onResetSession: () => {},
      });

      const session = createSampleSession({ kLoops: 2, steps: 2 });
      const b1 = session.getActiveBranch()!;
      const state1 = buildDoryTimelineDockState(session);

      harness.resetCounts();
      dock.update(state1);

      const branchSelect = harness.findByTestId(container, 'dory-dock-branch-select')!;
      const scrubber = harness.findByTestId(container, 'dory-dock-scrubber')!;
      const tempSlider = harness.findByTestId(container, 'dory-dock-temp-slider')!;
      expect(branchSelect.children).toHaveLength(1);
      const firstOptRef = branchSelect.children[0]!;
      expect(dock.getRenderedBranchSignature()).toBe(computeDoryBranchOptionsSignature(state1.branches));

      // Re-updating with same branch signature (cursor scrub / play toggle) allocates 0 DOM nodes & preserves refs
      harness.resetCounts();
      const stateScrub = buildDoryTimelineDockState(session, {
        isPlaying: true,
        activeRecordId: b1.recordIds[0]!,
      });
      dock.update(stateScrub);
      expect(harness.getCreateElementCount()).toBe(0);
      expect(branchSelect.children[0]).toBe(firstOptRef);
      expect(harness.findByTestId(container, 'dory-dock-scrubber')).toBe(scrubber);
      expect(harness.findByTestId(container, 'dory-dock-temp-slider')).toBe(tempSlider);

      // tourActive === true disables every interactive dock control
      dock.update(buildDoryTimelineDockState(session, { tourActive: true }));
      for (const testId of [
        'dory-dock-k-1',
        'dory-dock-k-2',
        'dory-dock-k-3',
        'dory-dock-k-4',
        'dory-dock-extrap-toggle',
        'dory-dock-kv-per_loop',
        'dory-dock-kv-last_loop',
        'dory-dock-mode-standard',
        'dory-dock-mode-self_spec',
        'dory-dock-temp-slider',
        'dory-dock-prompt-select',
        'dory-dock-step-back',
        'dory-dock-play-pause',
        'dory-dock-step-forward',
        'dory-dock-scrubber',
        'dory-dock-execute-step',
        'dory-dock-branch-select',
        'dory-dock-fork-branch',
        'dory-dock-reset-session',
      ]) {
        expect(harness.findByTestId(container, testId)!.disabled).toBe(true);
      }

      // DR18 config switch forks a clean branch -> signature changes -> <option>s rebuild cleanly
      const switchedRec = session.switchDepthOrPolicy({ kLoops: 4, kvCacheMode: 'last_loop' });
      const state2 = buildDoryTimelineDockState(session, {
        dr18ForkNotice: 'Forked k=4 last_loop branch',
      });
      dock.update(state2);
      expect(branchSelect.children).toHaveLength(2);
      expect(branchSelect.value).toBe(switchedRec.branchId);
      const dr18El = harness.findByTestId(container, 'dory-dock-dr18-notice')!;
      expect(dr18El.hidden).toBe(false);
      expect(dr18El.textContent).toContain('Forked k=4 last_loop');

      dock.dispose();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

// ============================================================================
// 5. RECURSIVE STATIC ISOLATION GUARDRAILS ACROSS ALL `src/dory/**` (`D4`, GATE 4b/4c)
// ============================================================================

describe('Static architectural isolation & clean-room guardrails across all src/dory/** files', () => {
  const repoRoot = resolve(__dirname, '..');
  const doryRoot = resolve(repoRoot, 'src/dory');
  const doryTsFiles = collectTsFilesRecursive(doryRoot);

  it('recursively discovers all src/dory/**/*.ts modules across core, scene, story, and app slices', () => {
    expect(doryTsFiles.length).toBeGreaterThanOrEqual(30);
    const rels = new Set(doryTsFiles.map((f) => toPosix(relative(doryRoot, f))));
    expect(rels.has('types.ts')).toBe(true);
    expect(rels.has('forward.ts')).toBe(true);
    expect(rels.has('speculative.ts')).toBe(true);
    expect(rels.has('session.ts')).toBe(true);
    expect(rels.has('scene/index.ts')).toBe(true);
    expect(rels.has('story/walkthrough.ts')).toBe(true);
    expect(rels.has('app/spatialApp.ts')).toBe(true);
    expect(rels.has('app/spatialOverlay.ts')).toBe(true);
    expect(rels.has('app/charts.ts')).toBe(true);
    expect(rels.has('app/timelineDock.ts')).toBe(true);
    expect(rels.has('app/main.ts')).toBe(true);
  });

  it('has ZERO imports from Gate 4b/4c forbidden Nano modules, ZERO sibling app imports, ZERO bare npm imports, and strictly obeys the D4 shared leaf allowlist', () => {
    // Explicit Gate 4b (Nano UI / controller / walkthrough / exclusive cutaways) forbidden modules
    const gate4bForbiddenRe =
      /^src\/(?:main|ui\/(?:controls|overlay|memoryChart)|story\/walkthrough|model\/rcPlayback|scene\/(?:gqaCutaway|mambaCutaway)|ui\/.+|story\/.+)$/i;

    // Explicit Gate 4c (Nano / TwoTower hybrid core, sheetInspector, model, math) forbidden modules
    const gate4cForbiddenRe =
      /^src\/(?:scene\/(?:spine|sheetInspector|hybridPick|embeddingCutaway|outputCutaway|moeCutaway|gqaSpatial|mambaSpatial|moeSpatial|accumulation)|model\/(?:spec|memory|state)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode))$/i;

    // Sibling visualizer apps strictly forbidden
    const siblingAppForbiddenRe = /^src\/(?:twotower|diffusion|uni)(?:\/|$)/i;

    // Positive D4 allowlist for any import outside `src/dory/`
    const allowedExternalModules = new Set<string>([
      'src/render/device',
      'src/render/camera',
      'src/render/primitives',
      'src/render/solids',
      'src/render/lines',
      'src/render/linePacking',
      'src/render/ribbons',
      'src/render/sheetFrame',
      'src/render/sheets',
      'src/render/pick',
      'src/scene/atlasPacker',
      'src/math/formulaFormat',
      'src/shared/a11y',
      'src/shared/format',
      'src/shared/modelRegistry',
      'src/shared/modelSwitcher',
      'src/shared/overlayLayout',
    ]);

    const violations: string[] = [];

    for (const filePath of doryTsFiles) {
      const raw = readFileSync(filePath, 'utf8');
      const relFile = toPosix(relative(repoRoot, filePath));
      const specifiers = extractTsModuleSpecifiers(raw, filePath);

      for (const spec of specifiers) {
        if (!spec.startsWith('.')) {
          violations.push(`${relFile}: forbidden non-relative / runtime npm import "${spec}"`);
          continue;
        }

        const resolvedPath = toPosix(relative(repoRoot, resolve(dirname(filePath), spec)));
        const normalizedNoExt = resolvedPath.replace(/\.(?:[cm]?ts|[cm]?js|tsx|jsx)$/i, '');

        if (normalizedNoExt.startsWith('src/dory/')) {
          continue;
        }

        if (gate4bForbiddenRe.test(normalizedNoExt)) {
          violations.push(`${relFile}: Gate 4b forbidden Nano UI/controller import "${normalizedNoExt}"`);
        } else if (gate4cForbiddenRe.test(normalizedNoExt)) {
          violations.push(`${relFile}: Gate 4c forbidden Nano hybrid-core import "${normalizedNoExt}"`);
        } else if (siblingAppForbiddenRe.test(normalizedNoExt)) {
          violations.push(`${relFile}: forbidden sibling app import "${normalizedNoExt}"`);
        } else if (!allowedExternalModules.has(normalizedNoExt)) {
          violations.push(`${relFile}: import "${normalizedNoExt}" is outside the D4 shared leaf allowlist`);
        }
      }
    }

    expect(violations).toEqual([]);
  });

  it('contains ZERO Math.random( calls anywhere under src/dory/**/*.ts (AST + text verification) and zero external CSS imports', () => {
    const violations: string[] = [];

    for (const filePath of doryTsFiles) {
      const raw = readFileSync(filePath, 'utf8');
      const stripped = stripTsComments(raw);
      const relFile = toPosix(relative(repoRoot, filePath));

      if (/\bMath\.random\s*\(/.test(stripped)) {
        violations.push(`${relFile}: forbidden Math.random( text match`);
      }
      if (findAstMathRandomAccesses(raw, filePath) > 0) {
        violations.push(`${relFile}: forbidden Math.random AST property access`);
      }
    }

    // Also verify src/dory/app/spatial.css does not import sibling CSS or remote resources
    const cssPath = resolve(doryRoot, 'app/spatial.css');
    const cssStripped = readFileSync(cssPath, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    if (/@import\b/i.test(cssStripped)) {
      violations.push('src/dory/app/spatial.css: forbidden @import rule');
    }
    if (/url\(\s*['"]?https?:/i.test(cssStripped)) {
      violations.push('src/dory/app/spatial.css: forbidden remote http(s) url(...)');
    }

    expect(violations).toEqual([]);
  });
});
