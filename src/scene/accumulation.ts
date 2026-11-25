/**
 * Presentation-layer Focused-Cell Accumulation Engine (Refinement #1).
 *
 * Derives exact trace-faithful Float32 partial sums directly from the verified
 * GQA, Mamba-2, and MoE simulator tensors without altering simulator arithmetic
 * and without prematurely spoiling completed final cell values while
 * `currentTermIndex < totalTerms - 1`.
 */

import {
  clampCellToSim,
  fmtProb,
  type GqaCellRef,
  type GqaSimResult,
} from '../math/gqaSim';
import {
  formatMambaHeadId,
  mambaGroupForHead,
  readMambaConvTapValue,
  readMambaPriorSsmValue,
  silu,
  type MambaCellRef,
  type MambaSimResult,
} from '../math/mambaSim';
import {
  formatExpertId,
  MOE_SIM_CONFIG,
  type MoeCellRef,
  type MoeSimResult,
} from '../math/moeSim';
import { fmtSigned } from './sheetInspector';

export interface FocusedAccumulationResult {
  /** True ONLY when `termProgress < 1 && totalTerms > 1`. */
  readonly active: boolean;
  /** True ONLY on the final reduction term (`currentTermIndex === totalTerms - 1`). */
  readonly isFinal: boolean;
  /** Zero-based active term index (`0 .. totalTerms - 1`). */
  readonly currentTermIndex: number;
  /** Total number of terms in the focused cell's reduction. */
  readonly totalTerms: number;
  /** Exact running Float32-derived partial sum across terms `0 .. currentTermIndex`. */
  readonly partialSum: number;
  /** Value contributed by `currentTermIndex`. */
  readonly currentTermValue: number;
  /** Concise expression for the active term (e.g. `(+2.04 · +2.34) / 2 = +2.387`). */
  readonly currentTermExpr: string;
  /**
   * Math Dock equation showing ONLY terms `0 .. currentTermIndex` plus `+ …` for
   * unaccumulated terms and `= <partialSum> (running k/N)` — never exposes the
   * completed final value while `currentTermIndex < totalTerms - 1`.
   */
  readonly partialEquation: string;
  /** Secondary Math Dock explanation for the running accumulation state. */
  readonly partialSubEquation: string;
  /**
   * Concise formula for `.ov-cell-popover` showing the active operand product +
   * running partial sum, never spoiling the completed final value.
   */
  readonly popoverEquation: string;
  /** Value badge text (e.g. `≈ +2.546 (2/4)` while running, `= +3.102 (4/4)` at final term). */
  readonly valueBadgeText: string;
  /** Status badge text (e.g. `Term 2/4 · Running`). */
  readonly badgeText: string;
  /** Active operand index (`currentTermIndex`): channel `d`, key `k`, Conv1D `lag`, SSM `0|1`, or MoE rank `0..5`. */
  readonly activeOperandIndex: number;
}

const COMPLETION_THRESHOLD = 0.999;

function clampInt(v: number, lo: number, hi: number): number {
  if (hi < lo) return lo;
  const n = Math.trunc(v);
  return n < lo ? lo : n > hi ? hi : n;
}

function fmtHead(h: number): string {
  return `Q${String(h).padStart(2, '0')}`;
}

function resolveTermIndex(termProgress: number, totalTerms: number): number | null {
  if (!Number.isFinite(termProgress) || termProgress >= COMPLETION_THRESHOLD || totalTerms <= 1) {
    return null;
  }
  const p = Math.max(0, termProgress);
  return Math.min(totalTerms - 1, Math.floor(p * totalTerms));
}

function makeBadgeText(currentTermIndex: number, totalTerms: number): string {
  const stepNum = currentTermIndex + 1;
  return currentTermIndex < totalTerms - 1
    ? `Term ${stepNum}/${totalTerms} · Running`
    : `Term ${stepNum}/${totalTerms} · Final`;
}

function makeSignedValueBadge(
  partialSum: number,
  currentTermIndex: number,
  totalTerms: number,
): string {
  const stepNum = currentTermIndex + 1;
  const prefix = currentTermIndex < totalTerms - 1 ? '≈' : '=';
  return `${prefix} ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`;
}

// ============================================================================
// GQA FOCUSED-CELL ACCUMULATION
// ============================================================================

/**
 * Trace-derived partial-sum evaluator for GQA focused cells during Stepper playback:
 * - Step 2 (`3. QKᵀ/√d`, `scores`, unmasked): accumulates `d = 0..dHead-1` scaled dot products.
 * - Step 4 (`5. Weighted V`, `v_out`): accumulates causal keys `k = 0..prefixTokens+q` of `Softmax · V`.
 *
 * Returns `null` when `termProgress >= 0.999`, when `totalTerms <= 1`, on masked cells,
 * or for steps/tensors without multi-term focused accumulation.
 */
export function computeGqaFocusedAccumulation(
  sim: GqaSimResult,
  cell: GqaCellRef,
  stepperStep: number,
  termProgress: number,
): FocusedAccumulationResult | null {
  if (!Number.isFinite(termProgress) || termProgress >= COMPLETION_THRESHOLD) {
    return null;
  }

  const normalizedCell: GqaCellRef =
    cell.tensor === 'v_out' && cell.kvHeadIndex === undefined && cell.headIndex !== undefined
      ? { ...cell, kvHeadIndex: cell.headIndex < 16 ? 0 : 1 }
      : cell;
  const clamped = clampCellToSim(sim, normalizedCell);
  const T = sim.tokens;
  const K = sim.keyTokens;
  const P = sim.prefixTokens;
  const { dHead } = sim;

  // Step 2 (3. QKᵀ/√d): scaled dot-product accumulation across d = 0..dHead-1
  if (stepperStep === 2 && clamped.tensor === 'scores') {
    const g: 0 | 1 = clamped.kvHeadIndex === 1 ? 1 : 0;
    const h = clamped.headIndex ?? (g === 0 ? 0 : 16);
    const q = clamped.row;
    const k = clamped.col;
    const qAbs = P + q;
    const exactScore = sim.Scores[(h * T + q) * K + k];
    if (k > qAbs || !Number.isFinite(exactScore)) {
      return null;
    }

    const totalTerms = dHead;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    const invSqrtD = 1 / Math.sqrt(dHead);
    const sqrtDVal = Math.sqrt(dHead);
    const sqrtDLabel = Number.isInteger(sqrtDVal) ? String(sqrtDVal) : `√${dHead}`;
    const qHeadOff = h * dHead * T;
    const kHeadOff = g * dHead * K;

    let dotSoFar = 0;
    let curQ = 0;
    let curK = 0;
    let currentTermValue = 0;
    const shownTerms: string[] = [];

    for (let r = 0; r <= currentTermIndex; r++) {
      const qv = sim.Q[qHeadOff + r * T + q];
      const kv = sim.K[kHeadOff + r * K + k];
      dotSoFar += qv * kv;
      shownTerms.push(`(${fmtSigned(qv, 2)}·${fmtSigned(kv, 2)})/${sqrtDLabel}`);
      if (r === currentTermIndex) {
        curQ = qv;
        curK = kv;
        currentTermValue = Math.fround(qv * kv * invSqrtD);
      }
    }

    const partialSum = Math.fround(dotSoFar * invSqrtD);
    const stepNum = currentTermIndex + 1;
    const pending = totalTerms - stepNum;
    const isFinal = pending === 0;
    const currentTermExpr = `(${fmtSigned(curQ, 2)} · ${fmtSigned(curK, 2)}) / ${sqrtDLabel} = ${fmtSigned(currentTermValue, 3)}`;

    const partialEquation = isFinal
      ? `(${fmtHead(h)}[:,${qAbs}] · K${g}[:,${k}]) / √${dHead} = ${shownTerms.join(' + ')} = ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`
      : `(${fmtHead(h)}[:,${qAbs}] · K${g}[:,${k}]) / √${dHead} = ${shownTerms.join(' + ')} + … = ${fmtSigned(partialSum, 3)} (running ${stepNum}/${totalTerms})`;

    const partialSubEquation = isFinal
      ? `Accumulated all ${totalTerms}/${totalTerms} head channels (d=0..${totalTerms - 1}) · exact QKᵀ/√${dHead} score`
      : `Accumulating term ${stepNum}/${totalTerms} (channel d=${currentTermIndex}) · running partial sum (${pending} ${pending === 1 ? 'term' : 'terms'} pending)`;

    const popoverEquation = isFinal
      ? `d${currentTermIndex}: (${fmtSigned(curQ, 2)}·${fmtSigned(curK, 2)})/${sqrtDLabel} → = ${fmtSigned(partialSum, 3)}`
      : `d${currentTermIndex}: (${fmtSigned(curQ, 2)}·${fmtSigned(curK, 2)})/${sqrtDLabel} → Σ = ${fmtSigned(partialSum, 3)}`;

    return {
      active: true,
      isFinal,
      currentTermIndex,
      totalTerms,
      partialSum,
      currentTermValue,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText: makeSignedValueBadge(partialSum, currentTermIndex, totalTerms),
      badgeText: makeBadgeText(currentTermIndex, totalTerms),
      activeOperandIndex: currentTermIndex,
    };
  }

  // Step 4 (5. Weighted V): causal value blend across keys k = 0..P+q
  if (stepperStep === 4 && clamped.tensor === 'v_out') {
    const g: 0 | 1 = clamped.kvHeadIndex === 1 ? 1 : 0;
    const q = clamped.row;
    const keyEnd = P + q;
    const totalTerms = keyEnd + 1;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    const localH = Math.floor(clamped.col / dHead);
    const r = clamped.col % dHead;
    const h = g * 16 + localH;
    const softRowOff = (h * T + q) * K;
    const vHeadOff = g * dHead * K;

    let acc = 0;
    let prevPartial = 0;
    let curW = 0;
    let curV = 0;
    let currentTermValue = 0;
    const shownTerms: string[] = [];

    for (let k = 0; k <= currentTermIndex; k++) {
      if (k === currentTermIndex) {
        prevPartial = Math.fround(acc);
      }
      const w = sim.Softmax[softRowOff + k];
      const v = sim.V[vHeadOff + r * K + k];
      acc += w * v;
      if (currentTermIndex < 4) {
        shownTerms.push(`${fmtProb(w, 2)}·${fmtSigned(v, 2)}`);
      }
      if (k === currentTermIndex) {
        curW = w;
        curV = v;
        currentTermValue = Math.fround(w * v);
      }
    }

    const partialSum = Math.fround(acc);
    const stepNum = currentTermIndex + 1;
    const pending = totalTerms - stepNum;
    const isFinal = pending === 0;
    const currentTermExpr = `${fmtProb(curW, 3)} · ${fmtSigned(curV, 2)} = ${fmtSigned(currentTermValue, 3)}`;

    const termsBody =
      currentTermIndex < 4
        ? shownTerms.join(' + ')
        : `Σ_{k=0..${currentTermIndex - 1}}(${fmtSigned(prevPartial, 3)}) + ${fmtProb(curW, 2)}·${fmtSigned(curV, 2)}`;

    const partialEquation = isFinal
      ? `V_out[${fmtHead(h)}, d=${r}, q=${keyEnd}] = ${termsBody} = ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`
      : `V_out[${fmtHead(h)}, d=${r}, q=${keyEnd}] = ${termsBody} + … = ${fmtSigned(partialSum, 3)} (running ${stepNum}/${totalTerms})`;

    const keyLabel = sim.keyTokenLabels[currentTermIndex] ?? `k${currentTermIndex}`;
    const cacheTag = P > 0 ? (currentTermIndex < P ? ' · cached' : ' · new') : '';
    const partialSubEquation = isFinal
      ? `Accumulated all ${totalTerms}/${totalTerms} causal keys (k=0..${keyEnd}) · exact weighted V_out`
      : `Accumulating term ${stepNum}/${totalTerms} (k=${currentTermIndex} "${keyLabel}"${cacheTag}) · running partial sum (${pending} ${pending === 1 ? 'term' : 'terms'} pending)`;

    const popoverEquation = isFinal
      ? `k${currentTermIndex}: ${fmtProb(curW, 2)}·${fmtSigned(curV, 2)} → = ${fmtSigned(partialSum, 3)}`
      : `k${currentTermIndex}: ${fmtProb(curW, 2)}·${fmtSigned(curV, 2)} → Σ = ${fmtSigned(partialSum, 3)}`;

    return {
      active: true,
      isFinal,
      currentTermIndex,
      totalTerms,
      partialSum,
      currentTermValue,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText: makeSignedValueBadge(partialSum, currentTermIndex, totalTerms),
      badgeText: makeBadgeText(currentTermIndex, totalTerms),
      activeOperandIndex: currentTermIndex,
    };
  }

  return null;
}

// ============================================================================
// MAMBA-2 FOCUSED-CELL ACCUMULATION
// ============================================================================

/**
 * Trace-derived partial-sum evaluator for Mamba-2 focused cells during Stepper playback:
 * - Step 2 (`3. Causal Conv1D`, `conv_buf` / `conv1d` / `x` / `b` / `c`):
 *   accumulates the 4 causal taps newest-to-oldest (`lag = 0..3`, using `readMambaConvTapValue`)
 *   plus `SiLU` on the final stage (`lag = 3`).
 * - Step 3 (`4. SSM Update`, `h_curr`):
 *   accumulates 2 exact terms: Term 0 = `dA · H_{t-1}[p, n]` (using `readMambaPriorSsmValue`),
 *   Term 1 = `+ Δt · x[p] · B[n]` → complete `H_t[p, n]`.
 * - Step 4 (`5. Readout + D`, `y`):
 *   accumulates `n = 0..dState-1` products `H_t[p, n] · C[n]` plus skip `+ D · x[p]`.
 */
export function computeMambaFocusedAccumulation(
  sim: MambaSimResult,
  cell: MambaCellRef,
  stepperStep: number,
  termProgress: number,
): FocusedAccumulationResult | null {
  if (!Number.isFinite(termProgress) || termProgress >= COMPLETION_THRESHOLD) {
    return null;
  }

  const T = sim.tokens;
  const { dHead, dInner, dState, convKernel, convDim, nHeads, nGroups } = sim;
  const tensorName = cell.tensor as string;

  // Step 2 (3. Causal Conv1D): 4 causal taps lag = 0..3 (newest-to-oldest) + SiLU on final stage
  if (
    stepperStep === 2 &&
    (tensorName === 'conv1d' ||
      tensorName === 'conv_buf' ||
      tensorName === 'x' ||
      tensorName === 'b' ||
      tensorName === 'c')
  ) {
    const totalTerms = convKernel;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    let tok: number;
    let convCh: number;
    let targetLabel: string;

    if (tensorName === 'x') {
      tok = clampInt(cell.row, 0, T - 1);
      const h =
        cell.headIndex !== undefined
          ? clampInt(cell.headIndex, 0, nHeads - 1)
          : clampInt(Math.floor(cell.col / dHead), 0, nHeads - 1);
      const p =
        cell.headIndex !== undefined && cell.col >= 0 && cell.col < dHead
          ? cell.col
          : clampInt(cell.col % dHead, 0, dHead - 1);
      convCh = h * dHead + p;
      targetLabel = `x[${formatMambaHeadId(h)},p=${p}]`;
    } else if (tensorName === 'b' || tensorName === 'c') {
      const isB = tensorName === 'b';
      tok = clampInt(cell.row, 0, T - 1);
      const flatCol = clampInt(cell.col, 0, nGroups * dState - 1);
      const isLocalWithGroup =
        cell.groupIndex !== undefined &&
        cell.col >= 0 &&
        cell.col < dState &&
        clampInt(cell.groupIndex, 0, nGroups - 1) > 0;
      const g = isLocalWithGroup
        ? clampInt(cell.groupIndex!, 0, nGroups - 1)
        : Math.floor(flatCol / dState);
      const n = isLocalWithGroup ? cell.col : flatCol % dState;
      const resolvedFlatCol = g * dState + n;
      convCh = dInner + (isB ? 0 : nGroups * dState) + resolvedFlatCol;
      targetLabel = `${isB ? 'B' : 'C'}[G${g},n=${n}]`;
    } else {
      tok = clampInt(cell.tokenIndex ?? T - 1, 0, T - 1);
      convCh = clampInt(cell.col, 0, convDim - 1);
      targetLabel = `xBC[ch=${convCh}]`;
    }

    const bias = sim.b_conv[convCh];
    const wcOff = convCh * convKernel;
    let convAcc = bias;
    let curW = 0;
    let curSample = 0;
    let curSlot = convKernel - 1;
    let curSourceKind: 'current_step' | 'seeded_state' | 'zero_pad' = 'current_step';
    let curAbsSrcT = sim.startTokenOffset + tok;
    let currentTermValue = 0;
    const shownTerms: string[] = [];

    for (let lag = 0; lag <= currentTermIndex; lag++) {
      const slot = convKernel - 1 - lag;
      const tap = readMambaConvTapValue(sim, convCh, tok, slot);
      const wTap = sim.w_conv[wcOff + slot];
      const prod = wTap * tap.value;
      convAcc += prod;
      shownTerms.push(`w[${slot}](${fmtSigned(wTap, 2)})·${fmtSigned(tap.value, 2)}`);
      if (lag === currentTermIndex) {
        curW = wTap;
        curSample = tap.value;
        curSlot = slot;
        curSourceKind = tap.sourceKind;
        curAbsSrcT = tap.absSrcT;
        currentTermValue = Math.fround(prod);
      }
    }

    const stepNum = currentTermIndex + 1;
    const pending = totalTerms - stepNum;
    const isFinal = pending === 0;
    const convPartial = Math.fround(convAcc);
    const partialSum = isFinal ? Math.fround(silu(convPartial)) : convPartial;

    const srcDesc =
      curSourceKind === 'seeded_state'
        ? `carried t${curAbsSrcT}`
        : curSourceKind === 'zero_pad'
          ? '0 pad'
          : `t${curAbsSrcT}`;

    const currentTermExpr = isFinal
      ? `w[${curSlot}](${fmtSigned(curW, 2)}) · ${fmtSigned(curSample, 2)} = ${fmtSigned(currentTermValue, 3)} → SiLU(${fmtSigned(convPartial, 3)}) = ${fmtSigned(partialSum, 3)}`
      : `w[${curSlot}](${fmtSigned(curW, 2)}) · ${fmtSigned(curSample, 2)} = ${fmtSigned(currentTermValue, 3)}`;

    const partialEquation = isFinal
      ? `${targetLabel} = SiLU(b(${fmtSigned(bias, 2)}) + ${shownTerms.join(' + ')}) = SiLU(${fmtSigned(convPartial, 3)}) = ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`
      : `Conv1D(${targetLabel}) = b(${fmtSigned(bias, 2)}) + ${shownTerms.join(' + ')} + … = ${fmtSigned(partialSum, 3)} (running ${stepNum}/${totalTerms})`;

    const partialSubEquation = isFinal
      ? `Accumulated all 4 causal taps (lag 0..3, latest ${srcDesc}) + SiLU · exact post-Conv1D value`
      : `Accumulating term ${stepNum}/${totalTerms} (lag ${currentTermIndex}, slot ${curSlot} · ${srcDesc}) · running pre-SiLU sum (${pending} ${pending === 1 ? 'tap' : 'taps'} + SiLU pending)`;

    const popoverEquation = isFinal
      ? `lag${currentTermIndex}(${srcDesc}): ${fmtSigned(curW, 2)}·${fmtSigned(curSample, 2)} → SiLU = ${fmtSigned(partialSum, 3)}`
      : `lag${currentTermIndex}(${srcDesc}): ${fmtSigned(curW, 2)}·${fmtSigned(curSample, 2)} → Σ = ${fmtSigned(partialSum, 3)}`;

    return {
      active: true,
      isFinal,
      currentTermIndex,
      totalTerms,
      partialSum,
      currentTermValue,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText: makeSignedValueBadge(partialSum, currentTermIndex, totalTerms),
      badgeText: makeBadgeText(currentTermIndex, totalTerms),
      activeOperandIndex: currentTermIndex,
    };
  }

  // Step 3 (4. SSM Update): 2 exact terms — Term 0 = dA·H_{t-1}, Term 1 = + Δt·x·B
  if (stepperStep === 3 && cell.tensor === 'h_curr') {
    const totalTerms = 2;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    const h = clampInt(cell.headIndex ?? 0, 0, nHeads - 1);
    const g = mambaGroupForHead(h);
    const p = clampInt(cell.row, 0, dHead - 1);
    const n = clampInt(cell.col, 0, dState - 1);
    const tok = clampInt(cell.tokenIndex ?? T - 1, 0, T - 1);
    const localIdx = (h * dHead + p) * dState + n;

    const prior = readMambaPriorSsmValue(sim, tok, localIdx);
    const prevVal = prior.value;
    const decay = sim.dA[h * T + tok];
    const step = sim.dt[h * T + tok];
    const xVal = sim.x[(h * dHead + p) * T + tok];
    const bVal = sim.B[(g * dState + n) * T + tok];

    const decayProd = decay * prevVal;
    const inputProd = step * xVal * bVal;
    const headTag = formatMambaHeadId(h);

    if (currentTermIndex === 0) {
      const partialSum = Math.fround(decayProd);
      const currentTermValue = partialSum;
      const currentTermExpr = `${decay.toFixed(3)} · (${fmtSigned(prevVal, 3)}) = ${fmtSigned(currentTermValue, 3)}`;
      const partialEquation = `H_t[${headTag},p=${p},n=${n}] = dA(${decay.toFixed(3)})·H_{t−1}(${fmtSigned(prevVal, 3)}) + … = ${fmtSigned(partialSum, 3)} (running 1/2)`;
      const partialSubEquation = `Accumulating term 1/2 (decay retention${prior.isSeeded ? ' · carried FP32 state' : ''}) · running partial sum (1 term pending)`;
      const popoverEquation = `Decay: ${decay.toFixed(2)}·${fmtSigned(prevVal, 2)} → Σ = ${fmtSigned(partialSum, 3)}`;

      return {
        active: true,
        isFinal: false,
        currentTermIndex: 0,
        totalTerms,
        partialSum,
        currentTermValue,
        currentTermExpr,
        partialEquation,
        partialSubEquation,
        popoverEquation,
        valueBadgeText: makeSignedValueBadge(partialSum, 0, totalTerms),
        badgeText: makeBadgeText(0, totalTerms),
        activeOperandIndex: 0,
      };
    }

    const partialSum = Math.fround(decayProd + inputProd);
    const currentTermValue = Math.fround(inputProd);
    const currentTermExpr = `+ ${step.toFixed(3)} · (${fmtSigned(xVal, 2)}) · (${fmtSigned(bVal, 2)}) = ${fmtSigned(currentTermValue, 3)}`;
    const partialEquation = `H_t[${headTag},p=${p},n=${n}] = ${decay.toFixed(3)}·(${fmtSigned(prevVal, 3)}) + ${step.toFixed(3)}·(${fmtSigned(xVal, 2)})·(${fmtSigned(bVal, 2)}) = ${fmtSigned(partialSum, 3)} (2/2)`;
    const partialSubEquation = `Accumulated 2/2 terms (decay retention + Δt·x·B write) · exact FP32 H_t value`;
    const popoverEquation = `+Δt·x·B: ${step.toFixed(2)}·${fmtSigned(xVal, 2)}·${fmtSigned(bVal, 2)} → = ${fmtSigned(partialSum, 3)}`;

    return {
      active: true,
      isFinal: true,
      currentTermIndex: 1,
      totalTerms,
      partialSum,
      currentTermValue,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText: makeSignedValueBadge(partialSum, 1, totalTerms),
      badgeText: makeBadgeText(1, totalTerms),
      activeOperandIndex: 1,
    };
  }

  // Step 4 (5. Readout + D): n = 0..dState-1 state products H_t[p,n]·C[n] plus skip + D·x[p]
  if (stepperStep === 4 && cell.tensor === 'y') {
    const totalTerms = dState + 1;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    const tok = clampInt(cell.row, 0, T - 1);
    const c = clampInt(cell.col, 0, dInner - 1);
    const h = Math.floor(c / dHead);
    const p = c % dHead;
    const g = mambaGroupForHead(h);
    const stateStride = nHeads * dHead * dState;
    const headTag = formatMambaHeadId(h);

    let readout = 0;
    let curH = 0;
    let curC = 0;
    let currentTermValue = 0;
    const shownTerms: string[] = [];

    const maxStateN = Math.min(currentTermIndex, dState - 1);
    for (let n = 0; n <= maxStateN; n++) {
      const localIdx = (h * dHead + p) * dState + n;
      const hVal = sim.H_history[tok * stateStride + localIdx];
      const cVal = sim.C[(g * dState + n) * T + tok];
      readout += hVal * cVal;
      shownTerms.push(`(${fmtSigned(hVal, 2)}·${fmtSigned(cVal, 2)})`);
      if (n === currentTermIndex) {
        curH = hVal;
        curC = cVal;
        currentTermValue = Math.fround(hVal * cVal);
      }
    }

    const stepNum = currentTermIndex + 1;
    const pending = totalTerms - stepNum;
    const isSkipStage = currentTermIndex === dState;

    if (!isSkipStage) {
      const partialSum = Math.fround(readout);
      const currentTermExpr = `(${fmtSigned(curH, 2)} · ${fmtSigned(curC, 2)}) = ${fmtSigned(currentTermValue, 3)}`;
      const partialEquation = `y[${headTag},p=${p}] = ${shownTerms.join(' + ')} + … = ${fmtSigned(partialSum, 3)} (running ${stepNum}/${totalTerms})`;
      const partialSubEquation = `Accumulating term ${stepNum}/${totalTerms} (state n=${currentTermIndex}: H_t·C[G${g}]) · running partial sum (${pending} ${pending === 1 ? 'term' : 'terms'} pending incl. D·x skip)`;
      const popoverEquation = `n${currentTermIndex}: ${fmtSigned(curH, 2)}·${fmtSigned(curC, 2)} → Σ = ${fmtSigned(partialSum, 3)}`;

      return {
        active: true,
        isFinal: false,
        currentTermIndex,
        totalTerms,
        partialSum,
        currentTermValue,
        currentTermExpr,
        partialEquation,
        partialSubEquation,
        popoverEquation,
        valueBadgeText: makeSignedValueBadge(partialSum, currentTermIndex, totalTerms),
        badgeText: makeBadgeText(currentTermIndex, totalTerms),
        activeOperandIndex: currentTermIndex,
      };
    }

    const ySsmVal = Math.fround(readout);
    const dSkip = sim.D[h];
    const xVal = sim.x[c * T + tok];
    const skipVal = dSkip * xVal;
    const partialSum = Math.fround(ySsmVal + skipVal);
    const skipTermRounded = Math.fround(skipVal);
    const currentTermExpr = `+ D(${dSkip.toFixed(2)}) · x(${fmtSigned(xVal, 2)}) = ${fmtSigned(skipTermRounded, 3)}`;
    const partialEquation = `y[${headTag},p=${p}] = ${shownTerms.join(' + ')} + D(${dSkip.toFixed(2)})·x(${fmtSigned(xVal, 2)}) = ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`;
    const partialSubEquation = `Accumulated all ${totalTerms}/${totalTerms} terms (${dState} state contractions + D·x skip) · exact SSM readout y`;
    const popoverEquation = `+D·x: ${dSkip.toFixed(2)}·${fmtSigned(xVal, 2)} → = ${fmtSigned(partialSum, 3)}`;

    return {
      active: true,
      isFinal: true,
      currentTermIndex,
      totalTerms,
      partialSum,
      currentTermValue: skipTermRounded,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText: makeSignedValueBadge(partialSum, currentTermIndex, totalTerms),
      badgeText: makeBadgeText(currentTermIndex, totalTerms),
      activeOperandIndex: currentTermIndex,
    };
  }

  return null;
}

// ============================================================================
// MOE (128+1) FOCUSED-CELL ACCUMULATION
// ============================================================================

/**
 * Trace-derived partial-sum evaluator for MoE focused cells during Stepper playback:
 * - Step 2 (`3. Top-6 (×2.5)`, `topk_weights`):
 *   accumulates the 6-term unbiased sigmoid denominator `Σ_{m=0..k} σ_m` (`1/6..5/6`)
 *   and reveals the completed normalized `×2.5` expert weight ONLY at `6/6`.
 * - Step 3 / Step 5 (`4. Routed ReLU²` / `6. Merge + Res` on `routed_sum` or `moe_out`):
 *   accumulates the 6 weighted routed expert branches `w_0·y_0 + … + w_k·y_k` (`1/6..6/6`),
 *   merging `+ y_shared` on the final stage for `moe_out`.
 */
export function computeMoeFocusedAccumulation(
  sim: MoeSimResult,
  cell: MoeCellRef,
  stepperStep: number,
  termProgress: number,
  focusToken: number,
  focusExpert: number,
): FocusedAccumulationResult | null {
  if (!Number.isFinite(termProgress) || termProgress >= COMPLETION_THRESHOLD) {
    return null;
  }

  const T = sim.tokens;
  const { d, topK, routedScalingFactor } = sim;
  void focusExpert;

  // Step 2 (3. Top-6 (×2.5)): 6-term unbiased sigmoid denominator accumulation
  if (stepperStep === 2 && cell.tensor === 'topk_weights') {
    const totalTerms = topK;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    const tokIdx = clampInt(cell.tokenIndex ?? focusToken, 0, T - 1);
    const slot = clampInt(cell.rankSlot ?? cell.col, 0, topK - 1);
    const winnerId = sim.topk_indices[tokIdx * topK + slot];
    const winnerTag = formatExpertId(winnerId);
    const targetRawSigma = sim.topk_raw_scores[tokIdx * topK + slot];

    let rawDenom = 0;
    let curSigma = 0;
    let curExpId = winnerId;
    const shownSigmas: string[] = [];

    for (let m = 0; m <= currentTermIndex; m++) {
      const expId = sim.topk_indices[tokIdx * topK + m];
      const sigma = sim.topk_raw_scores[tokIdx * topK + m];
      rawDenom += sigma;
      shownSigmas.push(sigma.toFixed(3));
      if (m === currentTermIndex) {
        curSigma = sigma;
        curExpId = expId;
      }
    }

    const stepNum = currentTermIndex + 1;
    const pending = totalTerms - stepNum;
    const isFinal = pending === 0;
    const curExpTag = formatExpertId(curExpId);

    const runningDenomSum = Math.fround(rawDenom);
    const finalValue = Math.fround(
      (targetRawSigma / (rawDenom + MOE_SIM_CONFIG.probDenomEps)) * routedScalingFactor,
    );
    const partialSum = isFinal ? finalValue : runningDenomSum;

    const currentTermExpr = `+ σ(#${stepNum} ${curExpTag}) = ${curSigma.toFixed(3)} → Σσ = ${rawDenom.toFixed(3)}`;

    const partialEquation = isFinal
      ? `w[${winnerTag}] = (${targetRawSigma.toFixed(3)} / (${shownSigmas.join(' + ')})) × ${routedScalingFactor} = (${targetRawSigma.toFixed(3)} / ${rawDenom.toFixed(3)}) × ${routedScalingFactor} = ${finalValue.toFixed(3)} (${stepNum}/${totalTerms})`
      : `Σσ = ${shownSigmas.join(' + ')} + … = ${rawDenom.toFixed(3)} (${stepNum}/${totalTerms}) · w[${winnerTag}] = (${targetRawSigma.toFixed(3)} / Σ₆σ) × ${routedScalingFactor}`;

    const partialSubEquation = isFinal
      ? `Accumulated all ${totalTerms}/${totalTerms} unbiased σ denominator terms (Σσ = ${rawDenom.toFixed(3)}) · exact ×${routedScalingFactor} weight`
      : `Accumulating denominator term ${stepNum}/${totalTerms} (rank #${stepNum} ${curExpTag} unbiased σ) · running Σσ = ${rawDenom.toFixed(3)} (${pending} ${pending === 1 ? 'term' : 'terms'} pending)`;

    const popoverEquation = isFinal
      ? `#${stepNum} ${curExpTag}: +σ(${curSigma.toFixed(2)}) → Σσ=${rawDenom.toFixed(2)} · w=${finalValue.toFixed(3)}`
      : `#${stepNum} ${curExpTag}: +σ(${curSigma.toFixed(2)}) → Σσ = ${rawDenom.toFixed(3)} (${stepNum}/${totalTerms})`;

    const valueBadgeText = isFinal
      ? `w = ${finalValue.toFixed(3)} (${stepNum}/${totalTerms})`
      : `Σσ = ${rawDenom.toFixed(3)} (${stepNum}/${totalTerms})`;

    return {
      active: true,
      isFinal,
      currentTermIndex,
      totalTerms,
      partialSum,
      currentTermValue: curSigma,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText,
      badgeText: makeBadgeText(currentTermIndex, totalTerms),
      activeOperandIndex: currentTermIndex,
    };
  }

  // Step 3 / Step 5 (4. Routed ReLU² / 6. Merge + Res): 6 weighted routed expert branches
  if (
    (stepperStep === 3 || stepperStep === 5) &&
    (cell.tensor === 'routed_sum' || cell.tensor === 'moe_out')
  ) {
    const totalTerms = topK;
    const currentTermIndex = resolveTermIndex(termProgress, totalTerms);
    if (currentTermIndex === null) return null;

    const isMerge = cell.tensor === 'moe_out';
    const tokIdx = clampInt(cell.row, 0, T - 1);
    const ch = clampInt(cell.col, 0, d - 1);
    const absTok = sim.startTokenOffset + tokIdx;

    let routedAcc = 0;
    let curW = 0;
    let curY = 0;
    let curExpId = sim.topk_indices[tokIdx * topK];
    let currentTermValue = 0;
    const shownTerms: string[] = [];

    for (let j = 0; j <= currentTermIndex; j++) {
      const expId = sim.topk_indices[tokIdx * topK + j];
      const w = sim.topk_weights[tokIdx * topK + j];
      const yv = sim.active_y[(tokIdx * topK + j) * d + ch];
      const contrib = w * yv;
      routedAcc = Math.fround(routedAcc + contrib);
      shownTerms.push(`${w.toFixed(2)}·${formatExpertId(expId)}(${fmtSigned(yv, 2)})`);
      if (j === currentTermIndex) {
        curW = w;
        curY = yv;
        curExpId = expId;
        currentTermValue = Math.fround(contrib);
      }
    }

    const stepNum = currentTermIndex + 1;
    const pending = totalTerms - stepNum;
    const isFinal = pending === 0;
    const curExpTag = formatExpertId(curExpId);

    const yShared = sim.y_shared[ch * T + tokIdx];
    const partialSum = isMerge && isFinal ? Math.fround(routedAcc + yShared) : routedAcc;

    const currentTermExpr =
      isMerge && isFinal
        ? `${curW.toFixed(3)} · ${curExpTag}(${fmtSigned(curY, 2)}) = ${fmtSigned(currentTermValue, 3)} + y_shared(${fmtSigned(yShared, 3)})`
        : `${curW.toFixed(3)} · ${curExpTag}(${fmtSigned(curY, 2)}) = ${fmtSigned(currentTermValue, 3)}`;

    const targetSym = isMerge ? `y_moe[${ch},t${absTok}]` : `y_routed[${ch},t${absTok}]`;
    const partialEquation = isFinal
      ? isMerge
        ? `${targetSym} = ${shownTerms.join(' + ')} + y_shared(${fmtSigned(yShared, 3)}) = ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`
        : `${targetSym} = ${shownTerms.join(' + ')} = ${fmtSigned(partialSum, 3)} (${stepNum}/${totalTerms})`
      : isMerge
        ? `${targetSym} = ${shownTerms.join(' + ')} + … (+ y_shared) = ${fmtSigned(partialSum, 3)} (running ${stepNum}/${totalTerms})`
        : `${targetSym} = ${shownTerms.join(' + ')} + … = ${fmtSigned(partialSum, 3)} (running ${stepNum}/${totalTerms})`;

    const partialSubEquation = isFinal
      ? isMerge
        ? `Accumulated all ${totalTerms}/${totalTerms} active routed branches + always-on shared expert · exact MoE merge`
        : `Accumulated all ${totalTerms}/${totalTerms} active routed experts (122 idle = 0) · exact Routed Sum`
      : `Accumulating term ${stepNum}/${totalTerms} (rank #${stepNum} ${curExpTag}) · running partial sum (${pending} ${pending === 1 ? 'expert' : 'experts'} pending${isMerge ? ' + shared' : ''})`;

    const popoverEquation = isFinal
      ? isMerge
        ? `#${stepNum} ${curExpTag} + shared → = ${fmtSigned(partialSum, 3)}`
        : `#${stepNum} ${curExpTag}: ${curW.toFixed(2)}·${fmtSigned(curY, 2)} → = ${fmtSigned(partialSum, 3)}`
      : `#${stepNum} ${curExpTag}: ${curW.toFixed(2)}·${fmtSigned(curY, 2)} → Σ = ${fmtSigned(partialSum, 3)}`;

    return {
      active: true,
      isFinal,
      currentTermIndex,
      totalTerms,
      partialSum,
      currentTermValue,
      currentTermExpr,
      partialEquation,
      partialSubEquation,
      popoverEquation,
      valueBadgeText: makeSignedValueBadge(partialSum, currentTermIndex, totalTerms),
      badgeText: makeBadgeText(currentTermIndex, totalTerms),
      activeOperandIndex: currentTermIndex,
    };
  }

  return null;
}
