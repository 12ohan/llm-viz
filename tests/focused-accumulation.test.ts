import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  GQA_SIM_CONFIG,
  runGqaSim,
  type GqaCellRef,
  type GqaSimResult,
} from '../src/math/gqaSim';
import {
  MAMBA_SIM_CONFIG,
  readMambaConvTapValue,
  readMambaPriorSsmValue,
  runMambaSim,
  type MambaCellRef,
} from '../src/math/mambaSim';
import { runMoeSim, type MoeCellRef } from '../src/math/moeSim';
import { NEMOTRON_SPEC } from '../src/model/spec';
import { createInitialState, createPerspectivePose, type ViewMode } from '../src/model/state';
import {
  buildCameraSnapshot,
  makeViewportMetrics,
  type AABB,
} from '../src/render/camera';
import { FRAME_XZ_AXES } from '../src/render/sheetFrame';
import {
  SHEET_REVEAL_COL_MAJOR,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
} from '../src/render/sheets';
import {
  computeGqaFocusedAccumulation,
  computeMambaFocusedAccumulation,
  computeMoeFocusedAccumulation,
} from '../src/scene/accumulation';
import { buildMoECutaway, type CutawaySceneGeometry } from '../src/scene/cutaways';
import { fmtSigned } from '../src/scene/sheetInspector';
import {
  computeEditorialCacheKey,
  MacroControls,
  type ControlsCallbacks,
  type ControlsElements,
} from '../src/ui/controls';
import {
  computeSheetCellWorldPos,
  isCellHiddenByReveal,
  MacroOverlay,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../src/ui/overlay';

// Minimal deterministic DOM shim for end-to-end MacroOverlay reveal & accumulation checks
class MockClassList {
  private readonly set = new Set<string>();

  setFromString(value: string): void {
    this.set.clear();
    for (const tok of value.split(/\s+/)) {
      if (tok) this.set.add(tok);
    }
  }

  toString(): string {
    return [...this.set].join(' ');
  }

  add(...tokens: string[]): void {
    for (const t of tokens) if (t) this.set.add(t);
  }

  remove(...tokens: string[]): void {
    for (const t of tokens) this.set.delete(t);
  }

  toggle(token: string, force?: boolean): boolean {
    if (force === undefined) {
      if (this.set.has(token)) {
        this.set.delete(token);
        return false;
      }
      this.set.add(token);
      return true;
    }
    if (force) this.set.add(token);
    else this.set.delete(token);
    return force;
  }

  contains(token: string): boolean {
    return this.set.has(token);
  }
}

class MockElement {
  readonly tagName: string;
  id = '';
  type = '';
  title = '';
  value = '';
  disabled = false;
  hidden = false;
  private _textContent = '';
  private _innerHTML = '';
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  readonly classList = new MockClassList();
  readonly attributes = new Map<string, string>();
  readonly children: MockElement[] = [];

  constructor(tagName: string) {
    this.tagName = tagName;
  }

  get className(): string {
    return this.classList.toString();
  }

  set className(val: string) {
    this.classList.setFromString(val);
    this.attributes.set('class', val);
  }

  get textContent(): string {
    if (this.children.length === 0) return this._textContent;
    return this._textContent + this.children.map((c) => c.textContent).join('');
  }

  set textContent(val: string) {
    this._textContent = val;
    this._innerHTML = val;
    this.children.length = 0;
  }

  set innerHTML(val: string) {
    this._innerHTML = val;
    this._textContent = val;
    this.children.length = 0;
  }

  get innerHTML(): string {
    return this._innerHTML;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name === 'class') this.classList.setFromString(value);
    if (name.startsWith('data-')) {
      const camel = name
        .slice(5)
        .replace(/-([a-z])/g, (_, ch: string) => ch.toUpperCase());
      this.dataset[camel] = value;
    }
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  appendChild(child: MockElement): MockElement {
    this.children.push(child);
    return child;
  }

  append(...nodes: MockElement[]): void {
    for (const n of nodes) this.children.push(n);
  }

  prepend(...nodes: MockElement[]): void {
    this.children.unshift(...nodes);
  }

  addEventListener(): void {
    // no-op in headless unit tests
  }

  removeEventListener(): void {
    // no-op in headless unit tests
  }

  contains(other: unknown): boolean {
    if (other === this) return true;
    return this.children.some((c) => c.contains(other));
  }

  findAll(predicate: (el: MockElement) => boolean): MockElement[] {
    const out: MockElement[] = [];
    const visit = (node: MockElement) => {
      if (predicate(node)) out.push(node);
      for (const child of node.children) visit(child);
    };
    visit(this);
    return out;
  }

  querySelector(selector: string): MockElement | null {
    if (selector.startsWith('.')) {
      const cls = selector.slice(1);
      return this.findAll((el) => el.classList.contains(cls))[0] ?? null;
    }
    if (selector.startsWith('#')) {
      const id = selector.slice(1);
      return this.findAll((el) => el.id === id)[0] ?? null;
    }
    return null;
  }

  getContext(): null {
    return null;
  }
}

function installMockDocument(): { svg: MockElement; labelLayer: MockElement } {
  const head = new MockElement('head');
  const mockDoc = {
    head,
    getElementById(id: string): MockElement | null {
      return head.children.find((c) => c.id === id) ?? null;
    },
    createElement(tag: string): MockElement {
      return new MockElement(tag);
    },
    createElementNS(_ns: string, tag: string): MockElement {
      return new MockElement(tag);
    },
    addEventListener(): void {
      // no-op
    },
    removeEventListener(): void {
      // no-op
    },
  };
  (globalThis as { document?: unknown }).document = mockDoc;
  return {
    svg: new MockElement('svg'),
    labelLayer: new MockElement('div'),
  };
}

function makeMinimalSheetScene(): CutawaySceneGeometry {
  return {
    kind: 'cutaway_gqa',
    bayId: 6,
    solidData: new Float32Array(0),
    solidCount: 0,
    lineData: new Float32Array(0),
    lineCount: 0,
    ribbonData: new Float32Array(0),
    ribbonCount: 0,
    sheetData: new Float32Array(20),
    sheetCount: 1,
    solidPickBoxes: [],
    landmarks: [],
    kvDepthTicks: [],
    fitBounds: { min: [-4, -1, -4], max: [4, 2, 4] },
    partCatalog: {},
    gqaStats: {
      bayId: 6,
      qHeadCount: 32,
      qGroupCount: 2,
      qHeadsPerGroup: 16,
      kvHeadCount: 2,
      gqaRatio: 16,
      kvTokenSliceCount: 16,
      computeFp8: false,
      kvCacheFp8: false,
    },
  };
}

function makeIncrementalGqaSim(prefixLen: number): GqaSimResult {
  const full = runGqaSim(prefixLen + 1);
  const { nKv, dHead, maxTokens, d } = GQA_SIM_CONFIG;
  const KCache = new Float32Array(nKv * dHead * maxTokens);
  const VCache = new Float32Array(nKv * dHead * maxTokens);
  for (let row = 0; row < nKv * dHead; row++) {
    for (let k = 0; k < prefixLen; k++) {
      KCache[row * maxTokens + k] = full.K[row * full.keyTokens + k];
      VCache[row * maxTokens + k] = full.V[row * full.keyTokens + k];
    }
  }
  const XStep = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    XStep[i] = full.X[i * full.tokens + prefixLen];
  }
  return runGqaSim(1, {
    X: XStep,
    tokenIds: [full.tokenIds[prefixLen]],
    tokenLabels: [full.tokenLabels[prefixLen]],
    kvCacheIn: {
      seqLen: prefixLen,
      maxTokens,
      K: KCache,
      V: VCache,
      prefixTokenIds: full.tokenIds.subarray(0, prefixLen),
      prefixTokenLabels: full.tokenLabels.slice(0, prefixLen),
    },
  });
}

describe('RC1 Focused-Cell Accumulation Engine', () => {
  describe('GQA Focused Accumulation (Full Seq + Rectangular AR Decode)', () => {
    it('accumulates QKᵀ/√d across d=0..3 without spoiling final score and converges within 1e-5', () => {
      const sim = runGqaSim(8);
      const cell: GqaCellRef = { tensor: 'scores', headIndex: 3, kvHeadIndex: 0, row: 5, col: 2 };
      const exact = sim.Scores[(3 * sim.tokens + 5) * sim.keyTokens + 2];
      const exactText = fmtSigned(exact, 3);

      for (let dIdx = 0; dIdx < sim.dHead; dIdx++) {
        const progress = (dIdx + 0.4) / sim.dHead;
        const acc = computeGqaFocusedAccumulation(sim, cell, 2, progress);
        expect(acc).not.toBeNull();
        expect(acc!.active).toBe(true);
        expect(acc!.currentTermIndex).toBe(dIdx);
        expect(acc!.activeOperandIndex).toBe(dIdx);
        expect(acc!.totalTerms).toBe(4);

        if (dIdx < sim.dHead - 1) {
          expect(acc!.partialEquation).toContain('+ …');
          expect(acc!.partialEquation).toContain(`(running ${dIdx + 1}/4)`);
          expect(acc!.partialEquation).not.toContain(exactText);
          expect(acc!.popoverEquation).not.toContain(exactText);
          expect(acc!.valueBadgeText).not.toContain(exactText);
          expect(acc!.valueBadgeText).toContain(`(${dIdx + 1}/4)`);
          expect(acc!.badgeText).toBe(`Term ${dIdx + 1}/4 · Running`);
        } else {
          expect(Math.abs(acc!.partialSum - exact)).toBeLessThan(1e-5);
          expect(acc!.partialEquation).not.toContain('+ …');
          expect(acc!.valueBadgeText).toBe(`= ${exactText} (4/4)`);
        }
      }

      // Completion threshold >= 0.999 returns null for ordinary unrestricted inspection
      expect(computeGqaFocusedAccumulation(sim, cell, 2, 0.999)).toBeNull();
      expect(computeGqaFocusedAccumulation(sim, cell, 2, 1.0)).toBeNull();

      // Masked upper-triangle cell returns null
      const maskedCell: GqaCellRef = { tensor: 'scores', headIndex: 3, kvHeadIndex: 0, row: 2, col: 5 };
      expect(computeGqaFocusedAccumulation(sim, maskedCell, 2, 0.25)).toBeNull();
    });

    it('accumulates rectangular AR Decode QKᵀ/√d scores for both cached and new keys', () => {
      const arSim = makeIncrementalGqaSim(5); // P=5, S=1, keyTokens=6
      expect(arSim.prefixTokens).toBe(5);
      expect(arSim.tokens).toBe(1);
      expect(arSim.keyTokens).toBe(6);

      for (const k of [2, 5]) {
        const cell: GqaCellRef = { tensor: 'scores', headIndex: 18, kvHeadIndex: 1, row: 0, col: k };
        const exact = arSim.Scores[(18 * arSim.tokens + 0) * arSim.keyTokens + k];
        const exactText = fmtSigned(exact, 3);

        const early = computeGqaFocusedAccumulation(arSim, cell, 2, 0.1);
        expect(early).not.toBeNull();
        expect(early!.currentTermIndex).toBe(0);
        expect(early!.partialEquation).not.toContain(exactText);
        expect(early!.popoverEquation).not.toContain(exactText);
        expect(early!.valueBadgeText).not.toContain(exactText);

        const finalTerm = computeGqaFocusedAccumulation(arSim, cell, 2, 0.9);
        expect(finalTerm).not.toBeNull();
        expect(finalTerm!.currentTermIndex).toBe(3);
        expect(Math.abs(finalTerm!.partialSum - exact)).toBeLessThan(1e-5);
      }
    });

    it('accumulates Weighted V across causal keys in both Full Seq and AR Decode without spoiling', () => {
      // Full sequence: q=4 => keys k=0..4 (totalTerms=5)
      const sim = runGqaSim(8);
      const h = 5;
      const r = 2;
      const cell: GqaCellRef = {
        tensor: 'v_out',
        headIndex: h,
        kvHeadIndex: 0,
        row: 4,
        col: (h % 16) * sim.dHead + r,
      };
      const exact = sim.V_out[(h * sim.dHead + r) * sim.tokens + 4];
      const exactText = fmtSigned(exact, 3);

      for (let k = 0; k < 5; k++) {
        const acc = computeGqaFocusedAccumulation(sim, cell, 4, (k + 0.5) / 5);
        expect(acc).not.toBeNull();
        expect(acc!.totalTerms).toBe(5);
        expect(acc!.currentTermIndex).toBe(k);
        expect(acc!.activeOperandIndex).toBe(k);
        if (k < 4) {
          expect(acc!.partialEquation).not.toContain(exactText);
          expect(acc!.popoverEquation).not.toContain(exactText);
          expect(acc!.valueBadgeText).not.toContain(exactText);
        } else {
          expect(Math.abs(acc!.partialSum - exact)).toBeLessThan(1e-5);
        }
      }

      // Single-key q=0 in non-incremental mode has totalTerms=1 -> returns null
      const singleKeyCell: GqaCellRef = { tensor: 'v_out', headIndex: 0, kvHeadIndex: 0, row: 0, col: 0 };
      expect(computeGqaFocusedAccumulation(sim, singleKeyCell, 4, 0.5)).toBeNull();

      // AR Decode: P=5, S=1, q=0 => keyEnd=5, totalTerms=6
      const arSim = makeIncrementalGqaSim(5);
      const arH = 19;
      const arR = 1;
      const arCell: GqaCellRef = {
        tensor: 'v_out',
        headIndex: arH,
        kvHeadIndex: 1,
        row: 0,
        col: (arH % 16) * arSim.dHead + arR,
      };
      const arExact = arSim.V_out[(arH * arSim.dHead + arR) * arSim.tokens + 0];
      const arMid = computeGqaFocusedAccumulation(arSim, arCell, 4, 0.4);
      expect(arMid).not.toBeNull();
      expect(arMid!.totalTerms).toBe(6);
      expect(arMid!.currentTermIndex).toBe(2);
      expect(arMid!.valueBadgeText).not.toContain(fmtSigned(arExact, 3));

      const arFinal = computeGqaFocusedAccumulation(arSim, arCell, 4, 0.95);
      expect(arFinal).not.toBeNull();
      expect(arFinal!.currentTermIndex).toBe(5);
      expect(Math.abs(arFinal!.partialSum - arExact)).toBeLessThan(1e-5);
      expect(computeGqaFocusedAccumulation(arSim, arCell, 4, 0.999)).toBeNull();
    });
  });

  describe('Mamba-2 Focused Accumulation (Conv1D + Carried State, SSM Update, Readout + D)', () => {
    it('accumulates 4-tap Causal Conv1D newest-to-oldest (including carried conv state) + final SiLU', () => {
      // Build an incremental Mamba run with carried conv_buf state from a 5-token prefix
      const prefixSim = runMambaSim(5);
      const fullSim = runMambaSim(6);
      const XStep = new Float32Array(MAMBA_SIM_CONFIG.d);
      for (let i = 0; i < MAMBA_SIM_CONFIG.d; i++) {
        XStep[i] = fullSim.X[i * fullSim.tokens + 5];
      }
      const incSim = runMambaSim(1, {
        X: XStep,
        initialConvState: prefixSim.conv_buf,
        initialSsmState: prefixSim.H,
        startTokenOffset: 5,
        tokenIds: [fullSim.tokenIds[5]],
        tokenLabels: [fullSim.tokenLabels[5]],
      });

      const h = 2;
      const p = 1;
      const ch = h * incSim.dHead + p;
      const cell: MambaCellRef = { tensor: 'x', row: 0, col: p, headIndex: h, tokenIndex: 0 };
      const exactX = incSim.x[ch * incSim.tokens + 0];
      const exactText = fmtSigned(exactX, 3);

      // Verify carried taps are genuinely seeded for lags 1..3
      expect(readMambaConvTapValue(incSim, ch, 0, 3).sourceKind).toBe('current_step');
      expect(readMambaConvTapValue(incSim, ch, 0, 2).sourceKind).toBe('seeded_state');

      for (let lag = 0; lag < 4; lag++) {
        const acc = computeMambaFocusedAccumulation(incSim, cell, 2, (lag + 0.3) / 4);
        expect(acc).not.toBeNull();
        expect(acc!.active).toBe(true);
        expect(acc!.totalTerms).toBe(4);
        expect(acc!.currentTermIndex).toBe(lag);
        expect(acc!.activeOperandIndex).toBe(lag);

        if (lag < 3) {
          expect(acc!.partialEquation).toContain('+ …');
          expect(acc!.partialEquation).not.toContain(exactText);
          expect(acc!.popoverEquation).not.toContain(exactText);
          expect(acc!.valueBadgeText).not.toContain(exactText);
          if (lag > 0) {
            expect(acc!.partialSubEquation).toContain('carried');
          }
        } else {
          expect(acc!.partialEquation).toContain('SiLU(');
          expect(Math.abs(acc!.partialSum - exactX)).toBeLessThan(1e-5);
        }
      }

      // Also verify B, C, and conv_buf targets converge identically at final stage
      const bCell: MambaCellRef = { tensor: 'b', row: 0, col: 5, groupIndex: 1, tokenIndex: 0 };
      const bFinal = computeMambaFocusedAccumulation(incSim, bCell, 2, 0.9);
      expect(bFinal).not.toBeNull();
      expect(Math.abs(bFinal!.partialSum - incSim.B[5 * incSim.tokens + 0])).toBeLessThan(1e-5);

      const cCell: MambaCellRef = { tensor: 'c', row: 0, col: 9, groupIndex: 2, tokenIndex: 0 };
      const cFinal = computeMambaFocusedAccumulation(incSim, cCell, 2, 0.9);
      expect(cFinal).not.toBeNull();
      expect(Math.abs(cFinal!.partialSum - incSim.C[9 * incSim.tokens + 0])).toBeLessThan(1e-5);

      const bufCell: MambaCellRef = { tensor: 'conv_buf', row: 2, col: ch, tapIndex: 2, tokenIndex: 0 };
      const bufFinal = computeMambaFocusedAccumulation(incSim, bufCell, 2, 0.9);
      expect(bufFinal).not.toBeNull();
      expect(Math.abs(bufFinal!.partialSum - incSim.xBC[ch * incSim.tokens + 0])).toBeLessThan(1e-5);

      expect(computeMambaFocusedAccumulation(incSim, cell, 2, 0.999)).toBeNull();
    });

    it('accumulates 2-term SSM Update (decay dA·H_{t-1} then +Δt·x·B) including seeded prior state', () => {
      const prefixSim = runMambaSim(4);
      const fullSim = runMambaSim(5);
      const XStep = new Float32Array(MAMBA_SIM_CONFIG.d);
      for (let i = 0; i < MAMBA_SIM_CONFIG.d; i++) {
        XStep[i] = fullSim.X[i * fullSim.tokens + 4];
      }
      const incSim = runMambaSim(1, {
        X: XStep,
        initialConvState: prefixSim.conv_buf,
        initialSsmState: prefixSim.H,
        startTokenOffset: 4,
      });

      // Pick a seeded state cell with substantial recurrence update so prior, decayed, and final H_t are all distinct
      let h = 0;
      let p = 0;
      let n = 0;
      let bestDelta = -1;
      for (let chH = 0; chH < incSim.nHeads; chH++) {
        for (let chP = 0; chP < incSim.dHead; chP++) {
          for (let chN = 0; chN < incSim.dState; chN++) {
            const idx = (chH * incSim.dHead + chP) * incSim.dState + chN;
            const prev = readMambaPriorSsmValue(incSim, 0, idx).value;
            const decayed = Math.fround(incSim.dA[chH] * prev);
            const cur = incSim.H[idx];
            const delta = Math.min(Math.abs(cur - prev), Math.abs(cur - decayed));
            if (delta > bestDelta) {
              bestDelta = delta;
              h = chH;
              p = chP;
              n = chN;
            }
          }
        }
      }
      expect(bestDelta).toBeGreaterThan(0.05);

      const localIdx = (h * incSim.dHead + p) * incSim.dState + n;
      const prior = readMambaPriorSsmValue(incSim, 0, localIdx);
      expect(prior.isSeeded).toBe(true);

      const cell: MambaCellRef = { tensor: 'h_curr', row: p, col: n, headIndex: h, tokenIndex: 0 };
      const exactH = incSim.H[localIdx];
      const exactText = fmtSigned(exactH, 3);

      // Term 0: decay retention only
      const term0 = computeMambaFocusedAccumulation(incSim, cell, 3, 0.25);
      expect(term0).not.toBeNull();
      expect(term0!.totalTerms).toBe(2);
      expect(term0!.currentTermIndex).toBe(0);
      expect(term0!.activeOperandIndex).toBe(0);
      expect(Math.abs(term0!.partialSum - Math.fround(incSim.dA[h] * prior.value))).toBeLessThan(1e-5);
      expect(term0!.partialEquation).not.toContain(exactText);
      expect(term0!.popoverEquation).not.toContain(exactText);
      expect(term0!.valueBadgeText).not.toContain(exactText);

      // Term 1: adds Δt·x·B -> exact H_t
      const term1 = computeMambaFocusedAccumulation(incSim, cell, 3, 0.75);
      expect(term1).not.toBeNull();
      expect(term1!.currentTermIndex).toBe(1);
      expect(term1!.activeOperandIndex).toBe(1);
      expect(Math.abs(term1!.partialSum - exactH)).toBeLessThan(1e-5);

      expect(computeMambaFocusedAccumulation(incSim, cell, 3, 0.999)).toBeNull();
      expect(computeMambaFocusedAccumulation(incSim, { tensor: 'residual', row: 0, col: 0 }, 5, 0.25)).toBeNull();
    });

    it('accumulates SSM Readout H_t·C across n=0..3 plus skip +D·x without premature spoiling', () => {
      const sim = runMambaSim(6);
      const tok = 4;
      const h = 3;
      const p = 2;
      const col = h * sim.dHead + p;
      const cell: MambaCellRef = { tensor: 'y', row: tok, col, headIndex: h, tokenIndex: tok };
      const exactY = sim.y[col * sim.tokens + tok];
      const exactText = fmtSigned(exactY, 3);

      for (let idx = 0; idx < 5; idx++) {
        const acc = computeMambaFocusedAccumulation(sim, cell, 4, (idx + 0.5) / 5);
        expect(acc).not.toBeNull();
        expect(acc!.totalTerms).toBe(5);
        expect(acc!.currentTermIndex).toBe(idx);
        expect(acc!.activeOperandIndex).toBe(idx);

        if (idx < 4) {
          expect(acc!.partialEquation).toContain('+ …');
          expect(acc!.partialEquation).not.toContain(exactText);
          expect(acc!.popoverEquation).not.toContain(exactText);
          expect(acc!.valueBadgeText).not.toContain(exactText);
        } else {
          expect(acc!.currentTermExpr).toContain('+ D(');
          expect(Math.abs(acc!.partialSum - exactY)).toBeLessThan(1e-5);
        }
      }

      expect(computeMambaFocusedAccumulation(sim, cell, 4, 0.999)).toBeNull();
    });
  });

  describe('MoE (128+1) Focused Accumulation (Top-6 Denominator + Routed Sum / Merge)', () => {
    it('accumulates running unbiased sigmoid denominator Σσ on 1/6..5/6 and reveals final ×2.5 weight only at 6/6', () => {
      const sim = runMoeSim(8);
      const tok = 5;
      const slot = 1;
      const winnerId = sim.topk_indices[tok * sim.topK + slot];
      const cell: MoeCellRef = {
        tensor: 'topk_weights',
        row: 0,
        col: slot,
        tokenIndex: tok,
        expertId: winnerId,
        rankSlot: slot,
      };
      const exactW = sim.topk_weights[tok * sim.topK + slot];
      const exactText3 = exactW.toFixed(3);
      const exactText2 = exactW.toFixed(2);

      let expectedRawDenom = 0;
      let prevDenom = 0;
      for (let m = 0; m < sim.topK; m++) {
        expectedRawDenom += sim.topk_raw_scores[tok * sim.topK + m];
        const acc = computeMoeFocusedAccumulation(sim, cell, 2, (m + 0.4) / sim.topK, tok, winnerId);
        expect(acc).not.toBeNull();
        expect(acc!.totalTerms).toBe(6);
        expect(acc!.currentTermIndex).toBe(m);
        expect(acc!.activeOperandIndex).toBe(m);

        if (m < 5) {
          expect(acc!.isFinal).toBe(false);
          // Running unbiased sigmoid denominator strictly increases (never an inflated partial quotient)
          expect(acc!.partialSum).toBeCloseTo(Math.fround(expectedRawDenom), 5);
          expect(acc!.partialSum).toBeGreaterThan(prevDenom);
          prevDenom = acc!.partialSum;

          expect(acc!.partialEquation).toContain('Σσ =');
          expect(acc!.partialEquation).toContain('+ …');
          expect(acc!.partialEquation).toContain('/ Σ₆σ');
          expect(acc!.popoverEquation).toContain(`Σσ = ${expectedRawDenom.toFixed(3)} (${m + 1}/6)`);
          expect(acc!.valueBadgeText).toBe(`Σσ = ${expectedRawDenom.toFixed(3)} (${m + 1}/6)`);

          // Zero leakage of completed normalized weight anywhere while !isFinal
          for (const field of [
            acc!.partialEquation,
            acc!.partialSubEquation,
            acc!.popoverEquation,
            acc!.valueBadgeText,
            acc!.currentTermExpr,
          ]) {
            expect(field).not.toContain(exactText3);
            expect(field).not.toContain(exactText2);
          }
        } else {
          expect(acc!.isFinal).toBe(true);
          expect(Math.abs(acc!.partialSum - exactW)).toBeLessThan(1e-5);
          expect(acc!.valueBadgeText).toBe(`w = ${exactText3} (6/6)`);
          expect(acc!.partialEquation).toContain(`= ${exactText3} (6/6)`);
          expect(acc!.popoverEquation).toContain(`w=${exactText3}`);
        }
      }

      expect(computeMoeFocusedAccumulation(sim, cell, 2, 0.999, tok, winnerId)).toBeNull();
    });

    it('accumulates 6 weighted routed expert branches for routed_sum (step 3) and moe_out (step 5)', () => {
      const sim = runMoeSim(8);
      const tok = 6;
      const ch = 3;
      const focusExp = sim.topk_indices[tok * sim.topK];

      // routed_sum at step 3
      const routedCell: MoeCellRef = { tensor: 'routed_sum', row: tok, col: ch, tokenIndex: tok };
      const exactRouted = sim.y_routed[ch * sim.tokens + tok];
      const exactRoutedText = fmtSigned(exactRouted, 3);

      for (let j = 0; j < sim.topK; j++) {
        const acc = computeMoeFocusedAccumulation(sim, routedCell, 3, (j + 0.5) / 6, tok, focusExp);
        expect(acc).not.toBeNull();
        expect(acc!.totalTerms).toBe(6);
        expect(acc!.currentTermIndex).toBe(j);
        expect(acc!.activeOperandIndex).toBe(j);

        if (j < 5) {
          expect(acc!.partialEquation).not.toContain(exactRoutedText);
          expect(acc!.popoverEquation).not.toContain(exactRoutedText);
          expect(acc!.valueBadgeText).not.toContain(exactRoutedText);
        } else {
          expect(Math.abs(acc!.partialSum - exactRouted)).toBeLessThan(1e-5);
        }
      }

      // moe_out at step 5 merges + y_shared on final stage 6/6
      const mergeCell: MoeCellRef = { tensor: 'moe_out', row: tok, col: ch, tokenIndex: tok };
      const exactMerge = sim.y_moe[ch * sim.tokens + tok];
      const exactMergeText = fmtSigned(exactMerge, 3);

      const midMerge = computeMoeFocusedAccumulation(sim, mergeCell, 5, 0.4, tok, focusExp);
      expect(midMerge).not.toBeNull();
      expect(midMerge!.currentTermIndex).toBe(2);
      expect(midMerge!.partialEquation).not.toContain(exactMergeText);
      expect(midMerge!.popoverEquation).not.toContain(exactMergeText);
      expect(midMerge!.valueBadgeText).not.toContain(exactMergeText);

      const finalMerge = computeMoeFocusedAccumulation(sim, mergeCell, 5, 0.95, tok, focusExp);
      expect(finalMerge).not.toBeNull();
      expect(finalMerge!.currentTermIndex).toBe(5);
      expect(Math.abs(finalMerge!.partialSum - exactMerge)).toBeLessThan(1e-5);

      expect(computeMoeFocusedAccumulation(sim, routedCell, 3, 0.999, tok, focusExp)).toBeNull();
      expect(computeMoeFocusedAccumulation(sim, mergeCell, 5, 1.0, tok, focusExp)).toBeNull();
    });
  });
});

describe('RC1 Non-Spoiling Overlay Reveal Guards & Accumulating State', () => {
  afterEach(() => {
    delete (globalThis as { document?: unknown }).document;
  });

  describe('isCellHiddenByReveal', () => {
    it('keeps all cells visible when revealMode is omitted or NONE (0)', () => {
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 4; c++) {
          expect(isCellHiddenByReveal(c, r, {})).toBe(false);
          expect(isCellHiddenByReveal(c, r, { revealMode: SHEET_REVEAL_NONE })).toBe(false);
        }
      }
    });

    it('hides every cell in UNCOMPUTED mode (3)', () => {
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 4; c++) {
          expect(isCellHiddenByReveal(c, r, { revealMode: SHEET_REVEAL_UNCOMPUTED })).toBe(true);
        }
      }
    });

    it('masks strictly future cells in ROW_MAJOR mode (1) while keeping computed + frontier visible', () => {
      const grid = {
        revealMode: SHEET_REVEAL_ROW_MAJOR,
        revealColFrontier: 2,
        revealRowFrontier: 1,
      };
      // Row 0: fully computed -> visible
      expect(isCellHiddenByReveal(0, 0, grid)).toBe(false);
      expect(isCellHiddenByReveal(3, 0, grid)).toBe(false);
      // Row 1: cols 0..1 computed, col 2 frontier -> visible; col 3 future -> hidden
      expect(isCellHiddenByReveal(0, 1, grid)).toBe(false);
      expect(isCellHiddenByReveal(1, 1, grid)).toBe(false);
      expect(isCellHiddenByReveal(2, 1, grid)).toBe(false);
      expect(isCellHiddenByReveal(3, 1, grid)).toBe(true);
      // Row 2: strictly future -> all hidden
      expect(isCellHiddenByReveal(0, 2, grid)).toBe(true);
      expect(isCellHiddenByReveal(2, 2, grid)).toBe(true);
      expect(isCellHiddenByReveal(3, 2, grid)).toBe(true);
    });

    it('masks strictly future cells in COL_MAJOR mode (2) while keeping computed + frontier visible', () => {
      const grid = {
        revealMode: SHEET_REVEAL_COL_MAJOR,
        revealColFrontier: 1,
        revealRowFrontier: 1,
      };
      // Col 0: fully computed -> visible
      expect(isCellHiddenByReveal(0, 0, grid)).toBe(false);
      expect(isCellHiddenByReveal(0, 2, grid)).toBe(false);
      // Col 1: row 0 computed, row 1 frontier -> visible; row 2 future -> hidden
      expect(isCellHiddenByReveal(1, 0, grid)).toBe(false);
      expect(isCellHiddenByReveal(1, 1, grid)).toBe(false);
      expect(isCellHiddenByReveal(1, 2, grid)).toBe(true);
      // Cols 2..3: strictly future -> all hidden
      expect(isCellHiddenByReveal(2, 0, grid)).toBe(true);
      expect(isCellHiddenByReveal(3, 1, grid)).toBe(true);
    });
  });

  describe('MacroOverlay non-spoiling cell value override & popover accumulating class', () => {
    const bounds: AABB = { min: [-6, -2, -6], max: [6, 4, 6] };

    it('suppresses uncomputed future cell numbers, renders activeValueOverride with .is-accumulating, and toggles popover class', () => {
      const { svg, labelLayer } = installMockDocument();
      const overlay = new MacroOverlay(
        svg as unknown as SVGSVGElement,
        labelLayer as unknown as HTMLDivElement,
      );

      const scene = makeMinimalSheetScene();
      const metrics = makeViewportMetrics(0, 0, 1440, 900, 1);
      const closeSnap = buildCameraSnapshot(
        createPerspectivePose([0, 0, 0], 4.0, 8, 52, 36),
        bounds,
        metrics,
      );

      // 3x3 grid with frontier at (col=1, row=1):
      // computed: (0,0)=0.11, (1,0)=0.22, (2,0)=0.33, (0,1)=0.44
      // frontier: (1,1) final=8.88 overridden by running partial sum 1.23
      // future (MUST NEVER LEAK): (2,1)=9.91, (0,2)=9.92, (1,2)=9.93, (2,2)=9.94
      const grid: ActiveSheetCellGrid = {
        originWorld: [0, 0, 0],
        extent: [2.4, 0.06, 1.8],
        axisU: FRAME_XZ_AXES.axisU,
        axisN: FRAME_XZ_AXES.axisN,
        cols: 3,
        rows: 3,
        values: new Float32Array([0.11, 0.22, 0.33, 0.44, 8.88, 9.91, 9.92, 9.93, 9.94]),
        format: 'signed',
        activeCol: 1,
        activeRow: 1,
        revealMode: SHEET_REVEAL_ROW_MAJOR,
        revealColFrontier: 1,
        revealRowFrontier: 1,
        activeValueOverride: 1.23,
      };

      const popover: CellPopoverSpec = {
        world: computeSheetCellWorldPos(grid, 1, 1),
        tensorLabel: 'Scores · Q00',
        coordLabel: '[q=1, k=1]',
        badge: 'Term 2/4 · Running',
        equation: 'd1: (+0.52·+0.61)/2 → Σ = +1.230',
        valueText: '≈ +1.230 (2/4)',
        isAccumulating: true,
      };

      overlay.updateCutaway({
        scene,
        snapshot: closeSnap,
        tokens: 3,
        selectedPart: null,
        hoveredPart: null,
        cellPopover: popover,
        activeCellGrids: [grid],
      });

      const visibleNums = labelLayer.children.filter(
        (el) => el.classList.contains('ov-cell-num') && el.style.display !== 'none',
      );
      const renderedTexts = visibleNums.map((el) => el.textContent);

      // Exactly the 5 revealed cells (0,0), (1,0), (2,0), (0,1), (1,1) are eligible; none of the 4 future 9.9x cells leak
      expect(visibleNums.length).toBeLessThanOrEqual(5);
      for (const forbidden of ['+9.91', '+9.92', '+9.93', '+9.94', '+8.88']) {
        expect(renderedTexts).not.toContain(forbidden);
      }

      // Active frontier cell displays running partial sum +1.23 with .is-accumulating
      const activeNumEl = visibleNums.find((el) => el.classList.contains('is-active'));
      expect(activeNumEl).toBeDefined();
      expect(activeNumEl!.textContent).toBe('+1.23');
      expect(activeNumEl!.classList.contains('is-accumulating')).toBe(true);

      // Popover & value badge carry .is-accumulating
      const popoverEl = labelLayer.children.find((el) => el.classList.contains('ov-cell-popover'));
      expect(popoverEl).toBeDefined();
      expect(popoverEl!.classList.contains('is-accumulating')).toBe(true);
      const popoverBody = popoverEl!.children[1];
      const popoverValEl = popoverBody.children.find((el) => el.classList.contains('ov-cell-popover-val'));
      expect(popoverValEl?.classList.contains('is-accumulating')).toBe(true);

      // Switching to UNCOMPUTED hides all cell numbers and clears .is-accumulating when popover finishes
      overlay.updateCutaway({
        scene,
        snapshot: closeSnap,
        tokens: 3,
        selectedPart: null,
        hoveredPart: null,
        cellPopover: { ...popover, isAccumulating: false, valueText: '= +8.880' },
        activeCellGrids: [{ ...grid, revealMode: SHEET_REVEAL_UNCOMPUTED }],
      });

      const uncomputedNums = labelLayer.children.filter(
        (el) => el.classList.contains('ov-cell-num') && el.style.display !== 'none',
      );
      expect(uncomputedNums).toHaveLength(0);
      expect(popoverEl!.classList.contains('is-accumulating')).toBe(false);
      expect(popoverValEl?.classList.contains('is-accumulating')).toBe(false);
    });
  });
});

function createMockControlsHarness(): {
  els: ControlsElements;
  editorialPanel: MockElement;
  cb: ControlsCallbacks;
} {
  const mkBtn = (id?: string, className?: string, dataset?: Record<string, string>) => {
    const el = new MockElement('button');
    if (id) el.id = id;
    if (className) el.className = className;
    if (dataset) Object.assign(el.dataset, dataset);
    return el;
  };
  const modeValues: ViewMode[] = [
    'tower',
    'cutaway_embedding',
    'cutaway_mamba',
    'cutaway_moe',
    'cutaway_gqa',
    'cutaway_output',
  ];
  const editorialPanel = new MockElement('div');
  const els = {
    sidebarTop: new MockElement('section'),
    sidebarTopBody: new MockElement('div'),
    sidebarTocToggle: mkBtn('sidebar-toc-toggle'),
    sidebarCurrentSection: new MockElement('span'),
    modeButtons: modeValues.map((m) => mkBtn(undefined, 'mode-btn', { mode: m, view: m })),
    tocButtons: [],
    presetButtons: [],
    mambaBayButtons: [],
    mambaViewButtons: [],
    moeBayButtons: [],
    moeViewButtons: [],
    gqaBayButtons: [],
    gqaViewButtons: [],
    miniNodes: [],
    tourToggleBtn: mkBtn('tour-toggle-btn'),
    floatingToolbar: new MockElement('div'),
    towerSubbar: new MockElement('div'),
    mambaSubbar: new MockElement('div'),
    moeSubbar: new MockElement('div'),
    gqaSubbar: new MockElement('div'),
    floatingBadge: new MockElement('div'),
    orbitHint: new MockElement('div'),
    gqaMathDock: new MockElement('section'),
    gqaFormulaTensor: new MockElement('span'),
    gqaFormulaCoord: new MockElement('span'),
    gqaFormulaHead: new MockElement('span'),
    gqaFormulaValue: new MockElement('span'),
    gqaFormulaNoteBtn: mkBtn('gqa-formula-note-btn'),
    gqaFormulaNote: new MockElement('div'),
    gqaCellClear: mkBtn('gqa-cell-clear'),
    gqaFormulaEq: new MockElement('div'),
    gqaFormulaSub: new MockElement('div'),
    gqaStepPrev: mkBtn(),
    gqaStepPlay: mkBtn(),
    gqaStepNext: mkBtn(),
    gqaStepPills: Array.from({ length: 6 }, (_, i) => mkBtn(undefined, 'gqa-step-pill', { gqaStep: String(i) })),
    gqaStepAutoFollow: mkBtn(),
    gqaStepScrub: new MockElement('input'),
    editorialPanel,
    execModeFull: mkBtn(),
    execModeDecode: mkBtn(),
    tokenSlider: new MockElement('input'),
    tokenDec: mkBtn(),
    tokenInc: mkBtn(),
    tokenValue: new MockElement('output'),
    precisionBf16: mkBtn(),
    precisionFp8: mkBtn(),
    memoryReadout: new MockElement('div'),
    memoryReadoutLines: new MockElement('div'),
    memoryChartBtn: mkBtn(),
    bayStepRow: new MockElement('div'),
    bayPrev: mkBtn(),
    bayNext: mkBtn(),
    bayClear: mkBtn(),
    zoomBayBtn: mkBtn(),
    fitBtn: mkBtn(),
    resetBtn: mkBtn(),
    shortcutsBtn: mkBtn(),
    shortcutsPopover: new MockElement('div'),
    shortcutsCloseBtn: mkBtn(),
    spineWaveHud: new MockElement('div'),
  } as unknown as ControlsElements;

  const cb: ControlsCallbacks = {
    onSelectMode: vi.fn(),
    onSelectSection: vi.fn(),
    onSelectMambaBay: vi.fn(),
    onSelectMambaView: vi.fn(),
    onSelectMambaStateToken: vi.fn(),
    onStepMambaStateToken: vi.fn(),
    onToggleMambaTemporalPlay: vi.fn(),
    onSelectMoeBay: vi.fn(),
    onSelectMoeView: vi.fn(),
    onSelectMoeFocusToken: vi.fn(),
    onStepMoeFocusToken: vi.fn(),
    onToggleMoeTokenCycle: vi.fn(),
    onSelectMoeExpert: vi.fn(),
    onSelectGqaBay: vi.fn(),
    onSelectGqaView: vi.fn(),
    onSelectPart: vi.fn(),
    onInspectInternals: vi.fn(),
    onTokensChange: vi.fn(),
    onPrecisionChange: vi.fn(),
    onFitAll: vi.fn(),
    onZoomSelectedBay: vi.fn(),
    onReset: vi.fn(),
    onStepBay: vi.fn(),
    onClearSelection: vi.fn(),
    onTourToggle: vi.fn(),
    onTourPrev: vi.fn(),
    onTourNext: vi.fn(),
    onTourRestart: vi.fn(),
    onTourExit: vi.fn(),
    onToggleMemoryChart: vi.fn(),
    onToggleShortcuts: vi.fn(),
    onCloseShortcuts: vi.fn(),
    onGqaStepPrev: vi.fn(),
    onGqaStepNext: vi.fn(),
    onGqaStepTogglePlay: vi.fn(),
    onGqaStepSelect: vi.fn(),
    onGqaStepScrub: vi.fn(),
    onGqaClearCell: vi.fn(),
    onToggleAutoFollow: vi.fn(),
    onSelectExecutionMode: vi.fn(),
    onDecodeStepToken: vi.fn(),
    onDecodeStepGreedy: vi.fn(),
    onDecodeNavStep: vi.fn(),
    onDecodeSwitchBranch: vi.fn(),
    onDecodeReset: vi.fn(),
  };

  return { els, editorialPanel, cb };
}

function readSidebarMoeWeights(editorialPanel: MockElement): string[] {
  const buttons = editorialPanel.findAll((el) => el.classList.contains('moe-exp-btn'));
  return buttons.map((btn) => {
    const match = /<span class="moe-exp-w">([^<]*)<\/span>/.exec(btn.innerHTML);
    return match ? match[1] : '';
  });
}

describe('RB-2 MoE Running Denominator & Elimination of All 5 Premature-Exposure Paths', () => {
  afterEach(() => {
    delete (globalThis as { document?: unknown }).document;
  });

  it('renders activeValueTextOverride Σ<denom> on 1/6..5/6, reveals weight only at 6/6, and grows 3D denominator highlights progressively', () => {
    const T = 8;
    const focusTok = 6;
    const slot = 2;
    const sim = runMoeSim(T);
    const slotExp = sim.topk_indices[focusTok * sim.topK + slot];
    const exactW = sim.topk_weights[focusTok * sim.topK + slot];
    const exactW2 = exactW.toFixed(2);
    const exactW3 = exactW.toFixed(3);

    const { svg, labelLayer } = installMockDocument();
    const overlay = new MacroOverlay(
      svg as unknown as SVGSVGElement,
      labelLayer as unknown as HTMLDivElement,
    );
    const metrics = makeViewportMetrics(0, 0, 1440, 900, 1);

    let prevFlowVerts = 0;
    for (let m = 0; m < sim.topK; m++) {
      const termProgress = (m + 0.4) / sim.topK;
      const scene = buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: slotExp,
        stepperStep: 2,
        stepperActive: true,
        termProgress,
        activeCell: {
          tensor: 'topk_weights',
          row: 0,
          col: slot,
          tokenIndex: focusTok,
          expertId: slotExp,
          rankSlot: slot,
        },
        viewAngle: 'three_quarter',
      });

      const accum = scene.focusedAccumulation!;
      expect(accum).not.toBeNull();
      const topkGrid = scene.activeCellGrids!.find((g) => g.colDimLabel === 'Top-6 (×2.5)')!;

      // 3D denominator highlight flow geometry grows term-by-term from 1/6 -> 6/6
      const flowLen = scene.flowVertices?.length ?? 0;
      expect(flowLen).toBeGreaterThan(prevFlowVerts);
      prevFlowVerts = flowLen;

      if (m < 5) {
        const expectedOverlayText = `Σ${accum.partialSum.toFixed(2)}`;
        expect(accum.isFinal).toBe(false);
        expect(topkGrid.activeValueTextOverride).toBe(expectedOverlayText);
        expect(topkGrid.activeValueTextOverride).not.toContain(exactW2);
        expect(topkGrid.activeValueTextOverride).not.toContain(exactW3);

        // Verify MacroOverlay renders the exact Σ text override on the active amber cell
        const testGrid: ActiveSheetCellGrid = {
          ...topkGrid,
          originWorld: [0, 0, 0],
          extent: [3.6, 0.06, 0.8],
          axisU: FRAME_XZ_AXES.axisU,
          axisN: FRAME_XZ_AXES.axisN,
        };
        const snap = buildCameraSnapshot(
          createPerspectivePose([0, 0, 0], 4.2, 8, 52, 36),
          { min: [-4, -1, -4], max: [4, 2, 4] },
          metrics,
        );
        overlay.updateCutaway({
          scene,
          snapshot: snap,
          tokens: T,
          selectedPart: null,
          hoveredPart: null,
          cellPopover: null,
          activeCellGrids: [testGrid],
        });
        const activeCellEl = labelLayer.children.find(
          (el) => el.classList.contains('ov-cell-num') && el.classList.contains('is-active') && el.style.display !== 'none',
        );
        expect(activeCellEl).toBeDefined();
        expect(activeCellEl!.textContent).toBe(expectedOverlayText);
        expect(activeCellEl!.classList.contains('is-accumulating')).toBe(true);
      } else {
        expect(accum.isFinal).toBe(true);
        expect(topkGrid.activeValueTextOverride).toBeUndefined();
        expect(topkGrid.activeValueOverride).toBeCloseTo(exactW, 5);
      }
    }
  });

  it('eliminates premature weight exposure in moeSpatial, moeCutaway landmarks, and partCatalog during steps 0..2 while preserving unrestricted & reduced-motion display', () => {
    const T = 8;
    const focusTok = 5;
    const sim = runMoeSim(T);
    const focusExp = sim.topk_indices[focusTok * sim.topK + 2];
    const exactW = sim.topk_weights[focusTok * sim.topK + 2];
    const exactW2 = exactW.toFixed(2);
    const exactW3 = exactW.toFixed(3);

    for (const viewAngle of ['three_quarter', 'near_plan'] as const) {
      // Steps 0, 1, 2 during active playback (reducedMotion=false): ZERO final weight exposure
      for (const step of [0, 1, 2] as const) {
        const scene = buildMoECutaway({
          bayId: 2,
          tokens: T,
          precision: 'bf16',
          selectedPart: null,
          hoveredPart: null,
          focusToken: focusTok,
          activeExpert: focusExp,
          stepperStep: step,
          stepperActive: true,
          reducedMotion: false,
          termProgress: step === 2 ? 0.25 : 1,
          viewAngle,
        });

        const detailLm = scene.landmarks.find((l) => l.id === 'moe-detail-status')!;
        expect(detailLm, `${viewAngle} step ${step} moe-detail-status`).toBeDefined();
        expect(detailLm.label).not.toContain('w=');
        expect(detailLm.label).not.toContain(exactW2);

        const wUpSpec0 = scene.partCatalog['routed-w-up']?.specs[0] ?? '';
        expect(wUpSpec0).toContain('w = pending (Step 3)');
        expect(wUpSpec0).not.toContain(exactW3);

        const expBadge = scene.partCatalog[`moe-expert-${focusExp}`]?.badge ?? '';
        expect(expBadge).toBe('ReLU²');
        expect(expBadge).not.toContain('w =');
        expect(expBadge).not.toContain(exactW3);
      }

      // Step 3 (post-normalization), inactive stepper, and reducedMotion all preserve ordinary unrestricted display
      const unrestrictedCases = [
        { label: 'step3', stepperActive: true, stepperStep: 3, reducedMotion: false },
        { label: 'inactive', stepperActive: false, stepperStep: 1, reducedMotion: false },
        { label: 'reducedMotion', stepperActive: true, stepperStep: 1, reducedMotion: true },
      ] as const;

      for (const c of unrestrictedCases) {
        const scene = buildMoECutaway({
          bayId: 2,
          tokens: T,
          precision: 'bf16',
          selectedPart: null,
          hoveredPart: null,
          focusToken: focusTok,
          activeExpert: focusExp,
          stepperStep: c.stepperStep,
          stepperActive: c.stepperActive,
          reducedMotion: c.reducedMotion,
          viewAngle,
        });

        const detailLm = scene.landmarks.find((l) => l.id === 'moe-detail-status')!;
        expect(detailLm.label, `${viewAngle} ${c.label} landmark`).toContain(`w=${exactW2}`);

        const wUpSpec0 = scene.partCatalog['routed-w-up']?.specs[0] ?? '';
        expect(wUpSpec0, `${viewAngle} ${c.label} routed-w-up spec`).toContain(`w = ${exactW3}`);

        const expBadge = scene.partCatalog[`moe-expert-${focusExp}`]?.badge ?? '';
        expect(expBadge, `${viewAngle} ${c.label} expert badge`).toBe(`w = ${exactW3} · ReLU²`);
      }
    }
  });

  it('masks sidebar Top-6 weights cleanly across steps 0..2, syncs slot & 1/6..6/6 transitions via computeEditorialCacheKey, and preserves unrestricted display', () => {
    installMockDocument();
    const { els, editorialPanel, cb } = createMockControlsHarness();
    const controls = new MacroControls(NEMOTRON_SPEC, els, cb);

    const T = 8;
    const focusTok = 6;
    const sim = runMoeSim(T);
    const expectedWeights = Array.from({ length: sim.topK }, (_, j) =>
      sim.topk_weights[focusTok * sim.topK + j].toFixed(2),
    );

    const state = createInitialState(createPerspectivePose([0, 0, 0], 18), false);
    state.mode = 'cutaway_moe';
    state.moeBayId = 2;
    state.tokens = T;
    state.moeFocusToken = focusTok;
    state.tourActive = false;

    const buildSceneFor = (opts: {
      stepperActive: boolean;
      stepperStep: number;
      reducedMotion?: boolean;
      slot?: number;
      termProgress?: number;
    }) => {
      const slot = opts.slot ?? 0;
      const exp = sim.topk_indices[focusTok * sim.topK + slot];
      return buildMoECutaway({
        bayId: 2,
        tokens: T,
        precision: 'bf16',
        selectedPart: null,
        hoveredPart: null,
        focusToken: focusTok,
        activeExpert: exp,
        stepperActive: opts.stepperActive,
        stepperStep: opts.stepperActive ? opts.stepperStep : null,
        reducedMotion: opts.reducedMotion ?? false,
        termProgress: opts.termProgress,
        activeCell:
          opts.stepperActive && opts.stepperStep === 2
            ? {
                tensor: 'topk_weights',
                row: 0,
                col: slot,
                tokenIndex: focusTok,
                expertId: exp,
                rankSlot: slot,
              }
            : null,
        simOverride: sim,
        viewAngle: 'three_quarter',
      });
    };

    // 1. Steps 0 and 1: ALL 6 sidebar weight badges masked to '—'
    for (const step of [0, 1] as const) {
      state.moeStepperActive = true;
      state.reducedMotion = false;
      state.moeStepperStep = step;
      state.moeStepperSubProgress = 0.2;
      const scene = buildSceneFor({ stepperActive: true, stepperStep: step });
      controls.sync(state, scene, false, null, null);
      expect(readSidebarMoeWeights(editorialPanel), `step ${step}`).toEqual([
        '—',
        '—',
        '—',
        '—',
        '—',
        '—',
      ]);
    }

    // 2. Step 2, slot 2, term 2/6 (!isFinal): slots 0..1 revealed, slot 2 = '…', slots 3..5 = '—'
    state.moeStepperStep = 2;
    state.moeStepperSubProgress = (2 + 0.25) / 6;
    const step2Mid = buildSceneFor({
      stepperActive: true,
      stepperStep: 2,
      slot: 2,
      termProgress: 0.25,
    });
    const keyMid = computeEditorialCacheKey(state, step2Mid);
    controls.sync(state, step2Mid, false, null, null);
    expect(readSidebarMoeWeights(editorialPanel)).toEqual([
      expectedWeights[0],
      expectedWeights[1],
      '…',
      '—',
      '—',
      '—',
    ]);

    // 3. Step 2, slot 2, term 6/6 (isFinal === true): cache key updates and slot 2 reveals weight!
    state.moeStepperSubProgress = (2 + 0.92) / 6;
    const step2Final = buildSceneFor({
      stepperActive: true,
      stepperStep: 2,
      slot: 2,
      termProgress: 0.92,
    });
    const keyFinal = computeEditorialCacheKey(state, step2Final);
    expect(keyFinal).not.toBe(keyMid);
    controls.sync(state, step2Final, false, null, null);
    expect(readSidebarMoeWeights(editorialPanel)).toEqual([
      expectedWeights[0],
      expectedWeights[1],
      expectedWeights[2],
      '—',
      '—',
      '—',
    ]);

    // 4. Advancing to slot 3 (1/6) updates cache key and masks slot 3 to '…', slots 4..5 to '—'
    state.moeStepperSubProgress = (3 + 0.08) / 6;
    const step2Slot3 = buildSceneFor({
      stepperActive: true,
      stepperStep: 2,
      slot: 3,
      termProgress: 0.08,
    });
    const keySlot3 = computeEditorialCacheKey(state, step2Slot3);
    expect(keySlot3).not.toBe(keyFinal);
    controls.sync(state, step2Slot3, false, null, null);
    expect(readSidebarMoeWeights(editorialPanel)).toEqual([
      expectedWeights[0],
      expectedWeights[1],
      expectedWeights[2],
      '…',
      '—',
      '—',
    ]);

    // 5. Step 3: all 6 final weights revealed
    state.moeStepperStep = 3;
    const step3Scene = buildSceneFor({ stepperActive: true, stepperStep: 3 });
    expect(computeEditorialCacheKey(state, step3Scene)).not.toBe(keySlot3);
    controls.sync(state, step3Scene, false, null, null);
    expect(readSidebarMoeWeights(editorialPanel)).toEqual(expectedWeights);

    // 6. Reduced motion on step 1: unrestricted display preserved and cache key updates
    state.moeStepperStep = 1;
    state.reducedMotion = false;
    const keyStep1Masked = computeEditorialCacheKey(state, step2Mid);
    state.reducedMotion = true;
    const reducedScene = buildSceneFor({
      stepperActive: true,
      stepperStep: 1,
      reducedMotion: true,
    });
    expect(computeEditorialCacheKey(state, reducedScene)).not.toBe(keyStep1Masked);
    controls.sync(state, reducedScene, false, null, null);
    expect(readSidebarMoeWeights(editorialPanel)).toEqual(expectedWeights);

    // 7. Outside active playback (moeStepperActive = false): unrestricted display preserved
    state.reducedMotion = false;
    state.moeStepperActive = false;
    state.moeStepperStep = 0;
    const idleScene = buildSceneFor({ stepperActive: false, stepperStep: 0 });
    controls.sync(state, idleScene, false, null, null);
    expect(readSidebarMoeWeights(editorialPanel)).toEqual(expectedWeights);

    controls.dispose();
  });
});
