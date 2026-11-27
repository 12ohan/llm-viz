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
  createDorySpatialApp,
  createIdleRafScheduler,
  resolveActiveDorySessionConfig,
  resolveDoryGlobalKeyAction,
  resolveDoryPickInteraction,
} from '../src/dory/app/spatialApp';
import {
  computeDoryOverlayAxisHeaderBudget,
  computeDoryOverlayCellNumBudget,
  computeDoryPickIdentityKey,
  createDorySpatialOverlay,
  DORY_MAX_OVERLAY_AXIS_HEADER_NODES,
  DORY_MAX_OVERLAY_CELL_NUM_NODES,
  resolveActiveDoryPopoverSpec,
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
import {
  buildCameraSnapshot,
  fitOpBoundsCamera,
  makeViewportMetrics,
  projectWorldToHostCss,
} from '../src/render/camera';
import { pickHitToDoryCell } from '../src/dory/scene/provenance';
import type { PickBox, PickHit } from '../src/render/pick';
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
  id: string;
  href: string;
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
  clientWidth: number;
  clientHeight: number;
  style: Record<string, string>;
  dataset: Record<string, string>;
  attributes: Record<string, string>;
  classes: Set<string>;
  children: MockDomNode[];
  parentNode: MockDomNode | null;
  listeners: Map<string, Set<(...args: unknown[]) => void>>;
  classList: {
    toggle(cls: string, force?: boolean): boolean;
    add(...cls: string[]): void;
    remove(...cls: string[]): void;
    contains(cls: string): boolean;
  };
  setAttribute(k: string, v: string): void;
  getAttribute(k: string): string | null;
  addEventListener(type: string, fn: (...args: unknown[]) => void): void;
  removeEventListener(type: string, fn: (...args: unknown[]) => void): void;
  dispatchEvent(type: string, event?: Record<string, unknown>): void;
  click(): void;
  contains(target: unknown): boolean;
  querySelectorAll(selector: string): MockDomNode[];
  getBoundingClientRect(): {
    left: number;
    top: number;
    right: number;
    bottom: number;
    width: number;
    height: number;
  };
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
  readonly stubGlobalWindow: (opts?: {
    readonly width?: number;
    readonly height?: number;
    readonly onRaf?: (cb: (nowMs: number) => void) => number;
    readonly onCancelRaf?: (h: number) => void;
  }) => { readonly flushRaf: (nowMs?: number) => void };
} {
  let createElementCount = 0;
  let createElementNsCount = 0;

  const buildNode = (tag: string): MockDomNode => {
    const classes = new Set<string>();
    const listeners = new Map<string, Set<(...args: unknown[]) => void>>();
    let rawInnerHtml = '';
    const node: MockDomNode = {
      tagName: tag.toUpperCase(),
      id: '',
      href: '',
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
      clientWidth: 1280,
      clientHeight: 720,
      style: {},
      dataset: {},
      attributes: {},
      classes,
      children: [],
      parentNode: null,
      listeners,
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
      addEventListener(type: string, fn: (...args: unknown[]) => void): void {
        let set = listeners.get(type);
        if (!set) {
          set = new Set();
          listeners.set(type, set);
        }
        set.add(fn);
      },
      removeEventListener(type: string, fn: (...args: unknown[]) => void): void {
        listeners.get(type)?.delete(fn);
      },
      dispatchEvent(type: string, event: Record<string, unknown> = {}): void {
        const set = listeners.get(type);
        if (!set) return;
        const payload = {
          type,
          target: node,
          preventDefault: () => {},
          stopPropagation: () => {},
          ...event,
        };
        for (const fn of Array.from(set)) {
          fn(payload);
        }
      },
      click(): void {
        if (node.disabled) return;
        node.dispatchEvent('click');
      },
      contains(target: unknown): boolean {
        if (target === node) return true;
        for (const c of node.children) {
          if (c.contains(target)) return true;
        }
        return false;
      },
      querySelectorAll(selector: string): MockDomNode[] {
        const cls = selector.startsWith('.') ? selector.slice(1) : null;
        if (!cls) return [];
        const matches: MockDomNode[] = [];
        const walk = (cur: MockDomNode): void => {
          if (cur.classes.has(cls) || cur.className.split(/\s+/).includes(cls)) {
            matches.push(cur);
          }
          for (const c of cur.children) walk(c);
        };
        for (const c of node.children) walk(c);
        return matches;
      },
      getBoundingClientRect() {
        return {
          left: 0,
          top: 0,
          right: node.clientWidth,
          bottom: node.clientHeight,
          width: node.clientWidth,
          height: node.clientHeight,
        };
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
        addEventListener: () => {},
        removeEventListener: () => {},
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
    stubGlobalWindow: (opts) => {
      let nextRafHandle = 1;
      const rafQueue = new Map<number, (nowMs: number) => void>();
      const reqRaf =
        opts?.onRaf ??
        ((cb: (nowMs: number) => void) => {
          const h = nextRafHandle++;
          rafQueue.set(h, cb);
          return h;
        });
      const cancelRaf =
        opts?.onCancelRaf ??
        ((h: number) => {
          rafQueue.delete(h);
        });
      vi.stubGlobal('requestAnimationFrame', reqRaf);
      vi.stubGlobal('cancelAnimationFrame', cancelRaf);
      vi.stubGlobal('window', {
        innerWidth: opts?.width ?? 1280,
        innerHeight: opts?.height ?? 720,
        devicePixelRatio: 1,
        addEventListener: () => {},
        removeEventListener: () => {},
        matchMedia: () => ({ matches: false }),
      });
      return {
        flushRaf: (nowMs = 16) => {
          const batch = Array.from(rafQueue.entries());
          rafQueue.clear();
          for (const [, cb] of batch) {
            cb(nowMs);
          }
        },
      };
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

  it('resolveDoryPickInteraction routes REAL buildDorySpatialScene() PickBoxes across all 7 spatial views (zero fabricated tensorKeys)', () => {
    const pickBoxToIdentity = (
      scene: DorySpatialScene,
      matcher: (pb: PickBox) => boolean,
      hitCell?: { readonly row?: number; readonly col?: number },
    ): DoryPickIdentity => {
      const pb = scene.pickBoxes.find(matcher);
      expect(pb).toBeDefined();
      const hit: PickHit = {
        bayId: pb!.bayId,
        ...(pb!.partId !== undefined ? { partId: pb!.partId } : {}),
        ...(pb!.headIndex !== undefined ? { headIndex: pb!.headIndex } : {}),
        ...(pb!.kvHeadIndex !== undefined ? { kvHeadIndex: pb!.kvHeadIndex } : {}),
        ...(pb!.groupIndex !== undefined ? { groupIndex: pb!.groupIndex } : {}),
        ...(pb!.tensorKey !== undefined ? { tensorKey: pb!.tensorKey } : {}),
        ...(pb!.tokenIndex !== undefined ? { tokenIndex: pb!.tokenIndex } : {}),
        ...(pb!.tapIndex !== undefined ? { tapIndex: pb!.tapIndex } : {}),
        ...(pb!.stateTokenIndex !== undefined ? { stateTokenIndex: pb!.stateTokenIndex } : {}),
        ...(pb!.towerRole !== undefined ? { towerRole: pb!.towerRole } : {}),
        ...(pb!.executionId !== undefined ? { executionId: pb!.executionId } : {}),
        ...(pb!.traceOrigin !== undefined ? { traceOrigin: pb!.traceOrigin } : {}),
        ...(pb!.semanticTag !== undefined ? { semanticTag: pb!.semanticTag } : {}),
        ...(pb!.absoluteTokenOffset !== undefined
          ? { absoluteTokenOffset: pb!.absoluteTokenOffset }
          : {}),
        ...(hitCell?.row !== undefined ? { hitRow: hitCell.row } : {}),
        ...(hitCell?.col !== undefined ? { hitCol: hitCell.col } : {}),
        t: 1,
        point: [
          (pb!.min[0] + pb!.max[0]) * 0.5,
          (pb!.min[1] + pb!.max[1]) * 0.5,
          (pb!.min[2] + pb!.max[2]) * 0.5,
        ],
        isFascia: false,
      };
      const decoded = pickHitToDoryCell(hit, scene.viewMode);
      expect(decoded).not.toBeNull();
      return decoded!;
    };

    const k4Session = createSampleSession({ kLoops: 4, steps: 0 });
    const specSession = createSampleSession({ executionMode: 'self_spec', steps: 1 });

    // View 1 (Macro — unrolled k=4): real recurrent sublayer box routes physicalLayerIdx & loopIteration
    const macroScene = buildDorySpatialScene(k4Session, 'macro', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      macroDisplayMode: 'unrolled',
      selectedLoopIteration: 3,
    });
    const macroPick = pickBoxToIdentity(
      macroScene,
      (pb) => pb.bayId === 17 && (pb.semanticTag ?? '').includes('loop=3'),
    );
    expect(resolveDoryPickInteraction('macro', macroPick, 4)).toEqual({
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 3,
    });
    // Loop clamping when kLoops < picked.loopIteration
    expect(resolveDoryPickInteraction('macro', macroPick, 2)).toEqual({
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 2,
    });

    // View 2 (Hypersphere): real sublayer sheet/solid routes physicalLayerIdx & loopIteration
    const hyperScene = buildDorySpatialScene(k4Session, 'hypersphere', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      selectedPhysicalLayerIdx: 22,
      selectedLoopIteration: 2,
    });
    const hyperPick = pickBoxToIdentity(
      hyperScene,
      (pb) => pb.bayId === 23 && (pb.semanticTag ?? '').includes('loop=2'),
    );
    expect(resolveDoryPickInteraction('hypersphere', hyperPick, 4)).toEqual({
      selectedPhysicalLayerIdx: 22,
      selectedLoopIteration: 2,
    });

    // View 3 (GQA & RoPE): REAL 'scores' and 'probs' plates route layer, qHead, queryRow & keyCol
    const gqaScene = buildDorySpatialScene(k4Session, 'gqa_rope', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      gqaLayerIdx: 22,
      selectedQHead: 0,
      selectedLoopIteration: 1,
    });
    const scorePick = pickBoxToIdentity(
      gqaScene,
      (pb) => pb.partId === 'gqa_scores_h5' && pb.tensorKey === 'scores',
      { row: 3, col: 2 },
    );
    expect(resolveDoryPickInteraction('gqa_rope', scorePick, 4)).toEqual({
      gqaLayerIdx: 22,
      selectedQHead: 5,
      selectedQueryRow: 3,
      selectedKeyCol: 2,
    });
    const probPick = pickBoxToIdentity(
      gqaScene,
      (pb) => pb.partId === 'gqa_probs_h7' && pb.tensorKey === 'probs',
      { row: 4, col: 1 },
    );
    expect(resolveDoryPickInteraction('gqa_rope', probPick, 4)).toEqual({
      gqaLayerIdx: 22,
      selectedQHead: 7,
      selectedQueryRow: 4,
      selectedKeyCol: 1,
    });
    // Odd layer ignored for gqaLayerIdx
    expect(
      resolveDoryPickInteraction(
        'gqa_rope',
        makePickIdentity({ viewMode: 'gqa_rope', physicalLayerIdx: 17 }),
        4,
      ),
    ).toBeNull();

    // View 4 (SwiGLU): REAL scene tensorKeys route channel (0..23), down_proj (both!), neg suv markers & outputDim (0..7)
    const swigluScene = buildDorySpatialScene(k4Session, 'swiglu', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      swigluLayerIdx: 17,
      selectedLoopIteration: 1,
    });
    for (const [partId, expectedKey, testCol] of [
      ['swiglu_gate_pre', 'gate_proj', 11],
      ['swiglu_up_pre', 'up_proj', 14],
      ['swiglu_suv_gate_eff', 'suv_gate', 5],
      ['swiglu_suv_up_eff', 'suv_up', 9],
      ['swiglu_gate_scaled', 'gate_scaled', 13],
      ['swiglu_up_scaled', 'up_scaled', 16],
      ['swiglu_gate_silu', 'gate_silu', 18],
      ['swiglu_product', 'swiglu_product', 20],
      ['swiglu_ffn_sub_norm', 'ffn_sub_norm', 19],
      ['swiglu_accum_terms', 'down_proj_accum', 23],
    ] as const) {
      const chPick = pickBoxToIdentity(
        swigluScene,
        (pb) => pb.partId === partId && pb.tensorKey === expectedKey,
        { row: 0, col: testCol },
      );
      expect(resolveDoryPickInteraction('swiglu', chPick, 4)).toEqual({
        swigluLayerIdx: 17,
        swigluChannel: testCol,
      });
    }
    // W_down [8, 24] updates BOTH swigluChannel (col) and swigluOutputDim (row)
    const downWeightPick = pickBoxToIdentity(
      swigluScene,
      (pb) => pb.partId === 'swiglu_down_proj_weight' && pb.tensorKey === 'down_proj',
      { row: 5, col: 17 },
    );
    expect(resolveDoryPickInteraction('swiglu', downWeightPick, 4)).toEqual({
      swigluLayerIdx: 17,
      swigluChannel: 17,
      swigluOutputDim: 5,
    });
    // Raised 3D negative signed-suv marker carries channel=<ch> in semanticTag (no grid cell)
    const negSuvPb = swigluScene.pickBoxes.find((pb) => pb.partId?.startsWith('swiglu_neg_suv_'));
    if (negSuvPb) {
      const expectedNegCh = Number.parseInt(negSuvPb.partId!.split('_').pop()!, 10);
      const negSuvPick = pickBoxToIdentity(swigluScene, (pb) => pb.partId === negSuvPb.partId);
      expect(resolveDoryPickInteraction('swiglu', negSuvPick, 4)).toEqual({
        swigluLayerIdx: 17,
        swigluChannel: expectedNegCh,
      });
    }
    // Output-dim (0..7) sheets route swigluOutputDim cleanly
    for (const [partId, expectedKey, testCol] of [
      ['swiglu_down_proj_out', 'down_proj_out', 6],
      ['swiglu_mlp_alpha_eff', 'mlp_alpha', 3],
      ['swiglu_nlerp_output', 'nlerp_output', 7],
      ['swiglu_h_in', 'nlerp_a_norm', 4],
    ] as const) {
      const outPick = pickBoxToIdentity(
        swigluScene,
        (pb) => pb.partId === partId && pb.tensorKey === expectedKey,
        { row: 0, col: testCol },
      );
      expect(resolveDoryPickInteraction('swiglu', outPick, 4)).toEqual({
        swigluLayerIdx: 17,
        swigluOutputDim: testCol,
      });
    }

    // View 5 (KV Compare): real even attention pick box routes kvCompareLayerIdx & selectedLoopIteration
    const kvScene = buildDorySpatialScene(k4Session, 'kv_compare', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      kvCompareLayerIdx: 20,
      selectedLoopIteration: 3,
    });
    const kvPick = pickBoxToIdentity(
      kvScene,
      (pb) => pb.bayId === 21 && (pb.semanticTag ?? '').includes('loop=3'),
    );
    expect(resolveDoryPickInteraction('kv_compare', kvPick, 4)).toEqual({
      kvCompareLayerIdx: 20,
      selectedLoopIteration: 3,
    });

    // View 6 (Self-Spec): real verify-offset & attention PickBoxes route selectedVerifyOffset & selfSpecAttentionLayerIdx
    const specScene = buildDorySpatialScene(specSession, 'self_spec', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      selfSpecAttentionLayerIdx: 56,
      selectedVerifyOffset: 0,
    });
    const specVerifyPick = pickBoxToIdentity(
      specScene,
      (pb) => (pb.semanticTag ?? '').includes('spec=1'),
    );
    expect(resolveDoryPickInteraction('self_spec', specVerifyPick, 4)?.selectedVerifyOffset).toBe(1);
    const specAttnPick = pickBoxToIdentity(specScene, (pb) => pb.bayId === 57);
    expect(
      resolveDoryPickInteraction('self_spec', specAttnPick, 4)?.selfSpecAttentionLayerIdx,
    ).toBe(56);

    // View 7 (Bookends): REAL PickBoxes for checkpoints, top-12 bars (vocabTokenId, NOT rank!),
    // 1x64 strips (picked.col, NOT tokenIndex=0!), 64x8 matrices (picked.row), seq sheets, and inert boxes
    const bookendsScene = buildDorySpatialScene(k4Session, 'bookends', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      selectedTokenOffset: 0,
    });
    expect(
      resolveDoryPickInteraction(
        'bookends',
        pickBoxToIdentity(bookendsScene, (pb) => pb.partId === 'bookend_ckpt_nvfp4'),
        4,
      ),
    ).toEqual({ selectedCheckpointFormat: 'nvfp4' });
    expect(
      resolveDoryPickInteraction(
        'bookends',
        pickBoxToIdentity(bookendsScene, (pb) => pb.partId === 'bookend_ckpt_q5_k_m'),
        4,
      ),
    ).toEqual({ selectedCheckpointFormat: 'q5_k_m' });

    // Top-12 bar rank=2 MUST decode vocabTokenId from semanticTag, NOT rank=2!
    const topBarRow2 = bookendsScene.bookendsMetadata!.topLogitRows[2]!;
    const topBarPick = pickBoxToIdentity(
      bookendsScene,
      (pb) => pb.partId === 'bookend_top_bar_2',
    );
    expect(topBarPick.tokenIndex).toBe(2); // rank is stored in tokenIndex
    expect(resolveDoryPickInteraction('bookends', topBarPick, 4)).toEqual({
      selectedVocabTokenId: topBarRow2.tokenId,
    });

    // 1x64 vocab strips MUST route picked.col (e.g. col=42), NEVER picked.tokenIndex (which is 0)!
    for (const [partId, tensorKey, col] of [
      ['bookend_raw_logits', 'raw_logits', 42],
      ['bookend_logit_scale_raw', 'logit_scale_raw', 19],
      ['bookend_logit_scale_eff', 'logit_scale_eff', 53],
      ['bookend_scaled_logits', 'scaled_logits', 31],
      ['bookend_softmax_probs', 'softmax_probs', 61],
    ] as const) {
      const stripPick = pickBoxToIdentity(
        bookendsScene,
        (pb) => pb.partId === partId && pb.tensorKey === tensorKey,
        { row: 0, col },
      );
      expect(resolveDoryPickInteraction('bookends', stripPick, 4)).toEqual({
        selectedVocabTokenId: col,
      });
    }

    // 64x8 vocab matrices (embeddings & untied lm_head) MUST route picked.row (vocab ID)
    for (const [partId, tensorKey, row] of [
      ['bookend_embed_weight', 'embeddings', 37],
      ['bookend_lm_head_weight', 'lm_head', 49],
    ] as const) {
      const matPick = pickBoxToIdentity(
        bookendsScene,
        (pb) => pb.partId === partId && pb.tensorKey === tensorKey,
        { row, col: 3 },
      );
      expect(resolveDoryPickInteraction('bookends', matPick, 4)).toEqual({
        selectedVocabTokenId: row,
      });
    }

    // Sequence sheets route selectedTokenOffset (NEVER corrupting selectedVocabTokenId)
    const embedLookupPick = pickBoxToIdentity(
      bookendsScene,
      (pb) => pb.partId === 'bookend_embed_tokens' && pb.tensorKey === 'embed_lookup',
      { row: 3, col: 2 },
    );
    expect(resolveDoryPickInteraction('bookends', embedLookupPick, 4)).toEqual({
      selectedTokenOffset: 3,
    });
    const codaHiddenPick = pickBoxToIdentity(
      bookendsScene,
      (pb) => pb.partId === 'bookend_coda_hidden' && pb.tensorKey === 'coda_hidden',
      { row: 2, col: 5 },
    );
    expect(resolveDoryPickInteraction('bookends', codaHiddenPick, 4)).toEqual({
      selectedTokenOffset: 2,
    });

    // Stage/active-row/bridge/TBPTL PickBoxes MUST NOT corrupt selectedVocabTokenId
    for (const partId of [
      'bookend_top12_stages',
      'bookend_embed_active_row',
      'bookend_lm_head_active_row',
      'bookend_backbone_bridge',
      'bookend_tbptl_stop_grad',
    ]) {
      const inertPick = pickBoxToIdentity(bookendsScene, (pb) => pb.partId === partId, {
        row: 0,
        col: 1,
      });
      expect(resolveDoryPickInteraction('bookends', inertPick, 4)).toBeNull();
    }
  });

  it('sampling temperature slider preserves pending readout & thumb value across refresh/blur and forks on +Step', () => {
    // 1. Pure dock state builder respects pending sampling override
    const baseSess = createSampleSession({ kLoops: 2, steps: 1 });
    const pendingSampling = Object.freeze({ temperature: 0.65, topK: 0, topP: 0.95 });
    const dockStateWithPending = buildDoryTimelineDockState(baseSess, {
      sampling: pendingSampling,
    });
    expect(dockStateWithPending.sampling).toEqual(pendingSampling);
    const dockStateDefault = buildDoryTimelineDockState(baseSess);
    expect(dockStateDefault.sampling.temperature).toBe(0);

    // 2. Full mocked-DOM app regression: dragging temp slider MUST NOT clobber readout or snap back on blur
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();
    harness.stubGlobalWindow();

    try {
      const host = harness.makeNode('div');
      const app = createDorySpatialApp({
        host: host as unknown as HTMLElement,
        initialViewMode: 'macro',
        initialKLoops: 2,
        attachDebugHandle: true,
      });

      const tempSlider = harness.findByTestId(host, 'dory-dock-temp-slider')!;
      const tempReadout = harness.findByTestId(host, 'dory-dock-temp-readout')!;
      const execBtn = harness.findByTestId(host, 'dory-dock-execute-step')!;
      expect(tempReadout.textContent).toBe('T=0.00 (Greedy)');
      expect(tempSlider.value).toBe('0');

      // Drag slider to 0.65 while unfocused (simulating both input & post-blur sync)
      tempSlider.value = '0.65';
      tempSlider.dispatchEvent('input');

      expect(tempReadout.textContent).toBe('T=0.65 (Stoch)');
      expect(tempSlider.value).toBe('0.65');
      expect(app.getState().sampling.temperature).toBeCloseTo(0.65, 6);

      // Switching views refreshes the dock; pending sampling MUST remain intact (no snap-back to 0!)
      app.debug.setView('bookends');
      expect(tempReadout.textContent).toBe('T=0.65 (Stoch)');
      expect(tempSlider.value).toBe('0.65');

      // Executing '+ Step' forks a clean branch carrying T=0.65
      const branchesBefore = app.debug.branchCount;
      execBtn.click();
      expect(app.debug.branchCount).toBe(branchesBefore + 1);
      expect(tempReadout.textContent).toBe('T=0.65 (Stoch)');
      expect(tempSlider.value).toBe('0.65');

      app.dispose();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('exiting Guided Tour via debug config setters or view tabs restores Free-Explore focus and unlocks scene & inspector', () => {
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();
    harness.stubGlobalWindow();

    try {
      const host = harness.makeNode('div');
      const app = createDorySpatialApp({
        host: host as unknown as HTMLElement,
        initialViewMode: 'macro',
        initialKLoops: 2,
        initialKvCacheMode: 'per_loop',
        initialExecutionMode: 'standard',
        attachDebugHandle: true,
      });

      // Set distinctive Free-Explore focus before entering tour
      app.debug.updateFocus({
        selectedPhysicalLayerIdx: 22,
        swigluChannel: 13,
        selectedCheckpointFormat: 'nvfp4',
      });
      const initialBranchCount = app.debug.branchCount;

      const assertTourLockedOnBeat11 = (): void => {
        const snap = app.debug.startTour(11);
        expect(snap.tourActive).toBe(true);
        expect(snap.viewMode).toBe('self_spec');
        expect(app.getCurrentScene().viewMode).toBe('self_spec');
        const lockNotes = host.querySelectorAll('.dory-tour-lock-note');
        expect(lockNotes.length).toBeGreaterThan(0);
      };

      // 1. debug.setKLoops(2) ALREADY matches freeSession -> MUST exit tour, restore Free-Explore view/focus, unlock inspector, zero spurious forks
      assertTourLockedOnBeat11();
      const afterK = app.debug.setKLoops(2);
      expect(afterK.tourActive).toBe(false);
      expect(afterK.viewMode).toBe('macro');
      expect(afterK.kLoops).toBe(2);
      expect(afterK.executionMode).toBe('standard');
      expect(afterK.branchCount).toBe(initialBranchCount);
      expect(app.getCurrentScene().viewMode).toBe('macro');
      expect(app.getState().focus.selectedPhysicalLayerIdx).toBe(22);
      expect(app.getState().focus.swigluChannel).toBe(13);
      expect(host.querySelectorAll('.dory-tour-lock-note')).toHaveLength(0);
      expect(harness.findByTestId(host, 'dory-layer-slider')!.disabled).toBe(false);

      // 2. debug.setKvCacheMode('per_loop') ALREADY matches freeSession -> MUST unlock cleanly
      assertTourLockedOnBeat11();
      const afterKv = app.debug.setKvCacheMode('per_loop');
      expect(afterKv.tourActive).toBe(false);
      expect(afterKv.viewMode).toBe('macro');
      expect(afterKv.kvCacheMode).toBe('per_loop');
      expect(afterKv.branchCount).toBe(initialBranchCount);
      expect(app.getCurrentScene().viewMode).toBe('macro');
      expect(host.querySelectorAll('.dory-tour-lock-note')).toHaveLength(0);
      expect(harness.findByTestId(host, 'dory-layer-slider')!.disabled).toBe(false);

      // 3. debug.setExecutionMode('standard') ALREADY matches freeSession -> MUST unlock cleanly
      assertTourLockedOnBeat11();
      const afterExec = app.debug.setExecutionMode('standard');
      expect(afterExec.tourActive).toBe(false);
      expect(afterExec.viewMode).toBe('macro');
      expect(afterExec.executionMode).toBe('standard');
      expect(afterExec.branchCount).toBe(initialBranchCount);
      expect(app.getCurrentScene().viewMode).toBe('macro');
      expect(host.querySelectorAll('.dory-tour-lock-note')).toHaveLength(0);
      expect(harness.findByTestId(host, 'dory-layer-slider')!.disabled).toBe(false);

      // 4. Exiting Guided Tour via clicking a View Tab restores Free-Explore focus without tour contamination
      assertTourLockedOnBeat11();
      const swigluTab = harness.findByTestId(host, 'dory-view-tab-swiglu')!;
      swigluTab.click();
      const afterTab = app.debug.getSnapshot();
      expect(afterTab.tourActive).toBe(false);
      expect(afterTab.viewMode).toBe('swiglu');
      expect(app.getCurrentScene().viewMode).toBe('swiglu');
      expect(app.getState().focus.swigluChannel).toBe(13);
      expect(app.getState().focus.selectedCheckpointFormat).toBe('nvfp4');
      expect(host.querySelectorAll('.dory-tour-lock-note')).toHaveLength(0);
      expect(harness.findByTestId(host, 'dory-swiglu-channel-slider')!.disabled).toBe(false);

      app.dispose();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('View 6 synthesized preview preserves active session telemetry in debug/getState and exposes reachable dory-activate-selfspec-btn', () => {
    // 1. Pure helper matches active session config across standard, last_loop, and live self_spec
    const stdSess = createSampleSession({ kLoops: 2, kvCacheMode: 'per_loop', executionMode: 'standard' });
    expect(resolveActiveDorySessionConfig(stdSess)).toEqual({
      kLoops: 2,
      kvCacheMode: 'per_loop',
      executionMode: 'standard',
    });

    const lastLoopSess = createSampleSession({ kLoops: 4, kvCacheMode: 'last_loop', executionMode: 'standard' });
    expect(resolveActiveDorySessionConfig(lastLoopSess)).toEqual({
      kLoops: 4,
      kvCacheMode: 'last_loop',
      executionMode: 'standard',
    });

    const specSess = createSampleSession({ executionMode: 'self_spec' });
    expect(resolveActiveDorySessionConfig(specSess)).toEqual({
      kLoops: 4,
      kvCacheMode: 'per_loop',
      executionMode: 'self_spec',
    });

    // 2. Full mocked-DOM app regression: viewing View 6 on a standard k=2 session MUST NOT pollute
    //    window.__dorySpatialDebug / getState() and MUST render a working dory-activate-selfspec-btn
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();
    harness.stubGlobalWindow();

    try {
      const host = harness.makeNode('div');
      const app = createDorySpatialApp({
        host: host as unknown as HTMLElement,
        initialViewMode: 'macro',
        initialKLoops: 2,
        initialKvCacheMode: 'per_loop',
        initialExecutionMode: 'standard',
        attachDebugHandle: true,
      });

      const initialBranchId = app.debug.activeBranchId!;
      expect(initialBranchId).toBeTruthy();

      // Switch to View 6 (`self_spec`) while session remains standard k=2 per_loop (56 banks)
      const previewSnap = app.debug.setView('self_spec');
      const previewScene = app.getCurrentScene();
      const previewState = app.getState();

      // View 6 renders its synthesized Contract A preview on the scene...
      expect(previewScene.viewMode).toBe('self_spec');
      expect(previewScene.selfSpecMetadata?.isSynthesizedPreview).toBe(true);
      expect(previewScene.executionMode).toBe('self_spec');

      // ...while debug snapshot & controller state faithfully report the active standard session!
      expect(previewSnap.viewMode).toBe('self_spec');
      expect(previewSnap.executionMode).toBe('standard');
      expect(previewSnap.kLoops).toBe(2);
      expect(previewSnap.kvCacheMode).toBe('per_loop');
      expect(previewSnap.executedDepth).toBe(112);
      expect(previewSnap.kvBankCount).toBe(56);

      expect(previewState.executionMode).toBe('standard');
      expect(previewState.kLoops).toBe(2);
      expect(previewState.kvCacheMode).toBe('per_loop');

      // Inspector MUST render the reachable 'Fork Live Contract A Session' button
      const activateBtn = harness.findByTestId(host, 'dory-activate-selfspec-btn');
      expect(activateBtn).not.toBeNull();
      expect(activateBtn!.disabled).toBe(false);

      // Clicking the button forks a live Contract A self_spec branch (k=4, per_loop, 104 banks)
      activateBtn!.click();

      const liveSnap = app.debug.getSnapshot();
      const liveState = app.getState();
      const liveScene = app.getCurrentScene();

      expect(liveScene.selfSpecMetadata?.isSynthesizedPreview).toBe(false);
      expect(liveSnap.executionMode).toBe('self_spec');
      expect(liveSnap.kLoops).toBe(4);
      expect(liveSnap.kvCacheMode).toBe('per_loop');
      expect(liveSnap.executedDepth).toBe(192);
      expect(liveSnap.kvBankCount).toBe(104);
      expect(liveSnap.branchCount).toBe(2);

      expect(liveState.executionMode).toBe('self_spec');
      expect(liveState.kLoops).toBe(4);
      expect(liveState.kvCacheMode).toBe('per_loop');

      // Once live self_spec is active, the fork button is hidden cleanly via nextControlsKey rebuild
      expect(harness.findByTestId(host, 'dory-activate-selfspec-btn')).toBeNull();

      // Switching back to the original standard branch on View 6 restores standard telemetry & button
      const backSnap = app.debug.switchBranch(initialBranchId);
      expect(backSnap.executionMode).toBe('standard');
      expect(backSnap.kLoops).toBe(2);
      expect(backSnap.kvBankCount).toBe(56);
      expect(harness.findByTestId(host, 'dory-activate-selfspec-btn')).not.toBeNull();

      app.dispose();
    } finally {
      vi.unstubAllGlobals();
    }
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

  it('prevents stale baked activeCellPopover leak across View 3 click -> hover other cell -> pointerleave lifecycle', () => {
    const harness = createMockDomHarness();
    harness.stubGlobalDocument();
    const win = harness.stubGlobalWindow({ width: 1280, height: 720 });

    try {
      // 1. Overlay-level mocked-DOM regression with a scene carrying a baked activeCellPopover
      const session = createSampleSession({ kLoops: 2, steps: 2 });
      const baseScene = buildDorySpatialScene(session, 'gqa_rope', {
        ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
        gqaLayerIdx: 16,
      });

      const scoreBox = baseScene.pickBoxes.find(
        (pb) => pb.tensorKey === 'scores' && pb.partId !== undefined && pb.frame !== undefined,
      )!;
      const probBox = baseScene.pickBoxes.find(
        (pb) => pb.tensorKey === 'probs' && pb.partId !== undefined && pb.frame !== undefined,
      )!;
      expect(scoreBox).toBeDefined();
      expect(probBox).toBeDefined();

      const cellA = pickHitToDoryCell(
        {
          bayId: scoreBox.bayId,
          partId: scoreBox.partId,
          headIndex: scoreBox.headIndex,
          kvHeadIndex: scoreBox.kvHeadIndex,
          groupIndex: scoreBox.groupIndex,
          tensorKey: scoreBox.tensorKey,
          tokenIndex: scoreBox.tokenIndex,
          tapIndex: scoreBox.tapIndex,
          stateTokenIndex: scoreBox.stateTokenIndex,
          semanticTag: scoreBox.semanticTag,
          hitRow: 0,
          hitCol: 0,
          t: 1,
          point: scoreBox.frame!.center,
          isFascia: false,
        },
        'gqa_rope',
      )!;
      const cellB = pickHitToDoryCell(
        {
          bayId: probBox.bayId,
          partId: probBox.partId,
          headIndex: probBox.headIndex,
          kvHeadIndex: probBox.kvHeadIndex,
          groupIndex: probBox.groupIndex,
          tensorKey: probBox.tensorKey,
          tokenIndex: probBox.tokenIndex,
          tapIndex: probBox.tapIndex,
          stateTokenIndex: probBox.stateTokenIndex,
          semanticTag: probBox.semanticTag,
          hitRow: 0,
          hitCol: 0,
          t: 1,
          point: probBox.frame!.center,
          isFascia: false,
        },
        'gqa_rope',
      )!;
      expect(computeDoryPickIdentityKey(cellA)).not.toBe(computeDoryPickIdentityKey(cellB));

      // Build scene with cellA baked into scene.activeCellPopover (simulating click -> rebuildSceneAndRefresh)
      const sceneBakedA = buildDorySpatialScene(session, 'gqa_rope', {
        ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
        gqaLayerIdx: 16,
        hoveredCell: cellA,
      });
      expect(sceneBakedA.activeCellPopover).not.toBeNull();
      const bakedLabelA = sceneBakedA.activeCellPopover!.tensorLabel;

      // Pure resolver checks
      expect(resolveActiveDoryPopoverSpec(sceneBakedA, cellA, cellA)).toBe(
        sceneBakedA.activeCellPopover,
      );
      const specForB = resolveActiveDoryPopoverSpec(sceneBakedA, cellB, cellA);
      expect(specForB).not.toBeNull();
      expect(specForB).not.toBe(sceneBakedA.activeCellPopover);
      expect(specForB!.tensorLabel).toBe(cellB.tensorKey ?? cellB.partId);
      expect(resolveActiveDoryPopoverSpec(sceneBakedA, null, cellA)).toBeNull();
      // Even without explicit bakedPopoverCell, mismatch on cellB & null hover are caught cleanly
      const specForBNoHint = resolveActiveDoryPopoverSpec(sceneBakedA, cellB);
      expect(specForBNoHint).not.toBeNull();
      expect(specForBNoHint).not.toBe(sceneBakedA.activeCellPopover);
      expect(resolveActiveDoryPopoverSpec(sceneBakedA, null)).toBeNull();

      const ovHost = harness.makeNode('div');
      const overlay = createDorySpatialOverlay(ovHost as unknown as HTMLElement);
      const popoverEl = harness.findByTestId(ovHost, 'dory-cell-popover')!;
      expect(popoverEl).not.toBeNull();

      const metrics = makeViewportMetrics(0, 0, 1280, 720, 1, 0, 0);
      const pose = fitOpBoundsCamera(sceneBakedA.cameraPreset.fitBounds, metrics, {
        azimuthDeg: sceneBakedA.cameraPreset.azimuthDeg,
        elevationDeg: sceneBakedA.cameraPreset.elevationDeg,
        fillFraction: 0.82,
      });
      const snapshot = buildCameraSnapshot(pose, sceneBakedA.fitBounds, metrics);

      harness.resetCounts();

      // (a) Hover matches baked cellA -> displays baked activeCellPopover
      const sA = overlay.update({
        scene: sceneBakedA,
        snapshot,
        hoveredCell: cellA,
        bakedPopoverCell: cellA,
        selectedPartId: cellA.partId,
      });
      expect(sA.popoverVisible).toBe(true);
      expect(popoverEl.style.display).toBe('flex');
      expect(popoverEl.children[0]!.children[0]!.textContent).toBe(bakedLabelA);

      // (b) Hover moves to cellB without rebuilding scene -> displays cellB fallback (NOT stale cellA!)
      const sB = overlay.update({
        scene: sceneBakedA,
        snapshot,
        hoveredCell: cellB,
        bakedPopoverCell: cellA,
        selectedPartId: cellA.partId,
      });
      expect(sB.popoverVisible).toBe(true);
      expect(popoverEl.style.display).toBe('flex');
      expect(popoverEl.children[0]!.children[0]!.textContent).toBe(cellB.tensorKey ?? cellB.partId);
      expect(popoverEl.children[0]!.children[0]!.textContent).not.toBe(bakedLabelA);

      // (c) Pointer leaves canvas (`hoveredCell: null`) while `selectedPartId` remains set -> hides cleanly!
      const sLeave = overlay.update({
        scene: sceneBakedA,
        snapshot,
        hoveredCell: null,
        bakedPopoverCell: cellA,
        selectedPartId: cellA.partId,
      });
      expect(sLeave.popoverVisible).toBe(false);
      expect(popoverEl.style.display).toBe('none');

      // Strictly 0 post-mount DOM allocations
      expect(harness.getCreateElementCount()).toBe(0);
      expect(harness.getCreateElementNsCount()).toBe(0);
      overlay.dispose();

      // 2. End-to-end mocked-DOM `createDorySpatialApp` pointer click -> hover other -> pointerleave on View 3
      const appHost = harness.makeNode('div');
      const app = createDorySpatialApp({
        host: appHost as unknown as HTMLElement,
        initialViewMode: 'gqa_rope',
        attachDebugHandle: false,
      });
      const canvasEl = harness.findByTestId(appHost, 'dory-webgl-canvas')!;
      const appPopoverEl = harness.findByTestId(appHost, 'dory-cell-popover')!;
      const appScene = app.getCurrentScene();
      const appPose = app.getCameraPose()!;
      const appMetrics = makeViewportMetrics(0, 0, 1280, 720, 1, 0, 0);
      const appSnap = buildCameraSnapshot(appPose, appScene.fitBounds, appMetrics);

      // Find two visible pickable sheets on View 3 and project their centers into client CSS coords
      const box1 = appScene.pickBoxes.find(
        (pb) => pb.tensorKey === 'scores' && pb.frame !== undefined,
      )!;
      const box2 = appScene.pickBoxes.find(
        (pb) => pb.tensorKey === 'probs' && pb.frame !== undefined,
      )!;
      const pt1 = projectWorldToHostCss(box1.frame!.center, appSnap);
      const pt2 = projectWorldToHostCss(box2.frame!.center, appSnap);
      expect(pt1.visible).toBe(true);
      expect(pt2.visible).toBe(true);

      // Hover box1, click box1 (triggers rebuildSceneAndRefresh baking currentScene.activeCellPopover)
      canvasEl.dispatchEvent('pointermove', { clientX: pt1.x, clientY: pt1.y, button: 0, pointerId: 1 });
      win.flushRaf(16);
      canvasEl.dispatchEvent('pointerdown', { clientX: pt1.x, clientY: pt1.y, button: 0, pointerId: 1 });
      canvasEl.dispatchEvent('pointerup', { clientX: pt1.x, clientY: pt1.y, button: 0, pointerId: 1 });
      win.flushRaf(32);

      const postClickScene = app.getCurrentScene();
      expect(postClickScene.activeCellPopover).not.toBeNull();
      expect(appPopoverEl.style.display).toBe('flex');
      const clickedTensorLabel = appPopoverEl.children[0]!.children[0]!.textContent;

      // Move pointer to box2 (without clicking) -> popover must update to box2, not stick on box1
      canvasEl.dispatchEvent('pointermove', { clientX: pt2.x, clientY: pt2.y, button: 0, pointerId: 1 });
      win.flushRaf(48);
      expect(appPopoverEl.style.display).toBe('flex');
      expect(appPopoverEl.children[0]!.children[0]!.textContent).not.toBe(clickedTensorLabel);

      // Leave canvas -> popover MUST hide cleanly (display === 'none') even with baked activeCellPopover
      canvasEl.dispatchEvent('pointerleave', { pointerId: 1 });
      win.flushRaf(64);
      expect(appPopoverEl.style.display).toBe('none');

      app.dispose();
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

    // 7. Regression: Exploratory 2D PCA at k=8 (selectedLoopIteration=8) & compact 180px height
    //    always includes selected row `t8` and keeps caveat box clamped inside SVG height
    const k8Sess = createSampleSession({ kLoops: 8, steps: 0 });
    const k8HyperScene = buildDorySpatialScene(k8Sess, 'hypersphere', {
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 8,
    });
    const k8PcaDefault = renderExploratoryPcaSvg(k8HyperScene.hypersphereMetadata!);
    assertCleanSvgChart(k8PcaDefault, 'exploratory_pca');
    expect(k8PcaDefault.svgMarkup).toContain('t8 ');
    const k8PcaCompact = renderExploratoryPcaSvg(k8HyperScene.hypersphereMetadata!, {
      width: 340,
      height: 180,
    });
    assertCleanSvgChart(k8PcaCompact, 'exploratory_pca');
    expect(k8PcaCompact.svgMarkup).toContain('t8 ');
    for (const rectMatch of k8PcaCompact.svgMarkup.matchAll(
      /<rect[^>]*\by="([0-9.]+)"[^>]*\bheight="([0-9.]+)"/g,
    )) {
      const y = Number.parseFloat(rectMatch[1]!);
      const h = Number.parseFloat(rectMatch[2]!);
      expect(y + h).toBeLessThanOrEqual(180);
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
    expect(rels.has('story/guidedTour.ts')).toBe(true);
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

    // Raw specifier collision guard: ensure no import/export specifier under `src/dory/`
    // matches Gate 4b/4c Nano module substrings (e.g. `../story/walkthrough` matching `story/walkthrough`).
    const rawForbiddenSpecifierRe =
      /(?:^|\/)(?:ui\/(?:controls|overlay|memoryChart)|scene\/(?:spine|sheetInspector)|model\/(?:spec|memory|state|rcPlayback)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)|story\/walkthrough|src\/main)(?:\.ts)?$/i;

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

        if (rawForbiddenSpecifierRe.test(spec)) {
          violations.push(`${relFile}: raw specifier "${spec}" collides with a Gate 4b/4c forbidden pattern`);
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

  it('guarantees every static dory-* class and state toggle in src/dory/app/*.ts matches a selector in spatial.css and styles #dory-app full-height', () => {
    const appDir = resolve(doryRoot, 'app');
    const cssPath = resolve(appDir, 'spatial.css');
    const cssStripped = readFileSync(cssPath, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

    // 1. Default mount host `#dory-app` (and `#app`) must be styled full-height in root rule
    expect(cssStripped).toMatch(/html\s*,\s*body\s*,[\s\S]*?#dory-app\b[\s\S]*?\{[\s\S]*?height\s*:\s*100%/);
    expect(cssStripped).toMatch(/html\s*,\s*body\s*,[\s\S]*?#app\b[\s\S]*?\{[\s\S]*?height\s*:\s*100%/);

    // 2. Collect all class selectors defined in spatial.css
    const cssClasses = new Set<string>();
    for (const m of cssStripped.matchAll(/\.([a-zA-Z0-9_-]+)/g)) {
      cssClasses.add(m[1]!);
    }

    // 3. AST-extract every class name emitted in src/dory/app/*.ts
    const extractStringFragments = (node: ts.Node | undefined): string[] => {
      if (!node) return [];
      if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
        return [node.text];
      }
      if (ts.isTemplateExpression(node)) {
        const out = [node.head.text];
        for (const span of node.templateSpans) {
          out.push(...extractStringFragments(span.expression));
          out.push(span.literal.text);
        }
        return out;
      }
      if (ts.isConditionalExpression(node)) {
        return [
          ...extractStringFragments(node.whenTrue),
          ...extractStringFragments(node.whenFalse),
        ];
      }
      if (
        ts.isBinaryExpression(node) &&
        node.operatorToken.kind === ts.SyntaxKind.PlusToken
      ) {
        return [
          ...extractStringFragments(node.left),
          ...extractStringFragments(node.right),
        ];
      }
      return [];
    };

    const appTsFiles = collectTsFilesRecursive(appDir);
    const missingClasses: string[] = [];

    for (const filePath of appTsFiles) {
      const relFile = toPosix(relative(repoRoot, filePath));
      const raw = readFileSync(filePath, 'utf8');
      const sf = ts.createSourceFile(filePath, raw, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);

      const checkClassString = (rawStr: string, pos: number): void => {
        const line = sf.getLineAndCharacterOfPosition(pos).line + 1;
        for (const token of rawStr.split(/\s+/)) {
          const cls = token.trim();
          if (!cls) continue;
          // Skip dynamic BEM prefix stems (e.g. `dory-badge--`, `dory-ov-tone--`, `dory-chart-badge--`)
          if (cls.endsWith('--')) continue;
          if (!/^(?:dory-|is-|has-)[a-z0-9_-]+$/.test(cls)) continue;
          if (!cssClasses.has(cls)) {
            missingClasses.push(`${relFile}:${line} -> .${cls}`);
          }
        }
      };

      const visit = (node: ts.Node): void => {
        const pos = node.getStart(sf);

        // DOM helper calls: el(tag, className), makeEl(tag, className), makeBtn(className, ...)
        if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
          const fn = node.expression.text;
          if ((fn === 'el' || fn === 'makeEl') && node.arguments.length >= 2) {
            for (const s of extractStringFragments(node.arguments[1])) checkClassString(s, pos);
          } else if (fn === 'makeBtn' && node.arguments.length >= 1) {
            for (const s of extractStringFragments(node.arguments[0])) checkClassString(s, pos);
          }
        }

        // Direct `.className = ...` assignments
        if (
          ts.isBinaryExpression(node) &&
          node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
          ts.isPropertyAccessExpression(node.left) &&
          node.left.name.text === 'className'
        ) {
          for (const s of extractStringFragments(node.right)) checkClassString(s, pos);
        }

        // `.setAttribute('class', ...)`
        if (
          ts.isCallExpression(node) &&
          ts.isPropertyAccessExpression(node.expression) &&
          node.expression.name.text === 'setAttribute' &&
          node.arguments.length >= 2 &&
          ts.isStringLiteral(node.arguments[0]!) &&
          node.arguments[0]!.text === 'class'
        ) {
          for (const s of extractStringFragments(node.arguments[1])) checkClassString(s, pos);
        }

        // `.classList.add/remove/toggle(...)`
        if (
          ts.isCallExpression(node) &&
          ts.isPropertyAccessExpression(node.expression) &&
          ts.isPropertyAccessExpression(node.expression.expression) &&
          node.expression.expression.name.text === 'classList'
        ) {
          const method = node.expression.name.text;
          if (method === 'add' || method === 'remove' || method === 'toggle') {
            const limit = method === 'toggle' ? 1 : node.arguments.length;
            for (let i = 0; i < limit; i += 1) {
              for (const s of extractStringFragments(node.arguments[i])) checkClassString(s, pos);
            }
          }
        }

        ts.forEachChild(node, visit);
      };

      visit(sf);
    }

    // 4. Verify all finite dynamic tone/chart badge variants exist in spatial.css
    const requiredDynamicClasses = [
      'dory-badge--prelude_indigo',
      'dory-badge--recurrent_teal',
      'dory-badge--coda_amber',
      'dory-badge--swa_cyan',
      'dory-badge--full_indigo',
      'dory-badge--nlerp_emerald',
      'dory-badge--spec_oxide',
      'dory-badge--readonly',
      'dory-badge--warn',
      'dory-badge--gold',
      'dory-badge--muted',
      'dory-chart-badge--spanning_subspace',
      'dory-chart-badge--exploratory_pca',
      'dory-chart-badge--dual_rope_phasors',
      'dory-chart-badge--kv_scaling_256k',
      'dory-chart-badge--self_spec_probs',
      'dory-chart-badge--tbptl_schematic',
    ];
    for (const cls of requiredDynamicClasses) {
      if (!cssClasses.has(cls)) {
        missingClasses.push(`required dynamic variant -> .${cls}`);
      }
    }

    expect(missingClasses).toEqual([]);
  });
});
