/**
 * Full-Block Mamba-2 Analytical Decomposition & 4-Tap Conv1D Window Replacement Tracker
 * (`src/twotower/scene/mambaDecomposition.ts`)
 *
 * CRITICAL ARCHITECTURAL INVARIANT:
 *   Reads STRICTLY from the `MambaSimResult` of the COMPLETE executed block (`sim.tokens === B`),
 *   reading `sim.H_history`, `sim.initialSsmState`, `sim.initialConvState`, `sim.xBC_pre`,
 *   `sim.dA`, `sim.dt`, `sim.x`, `sim.B`, `sim.C`, `sim.D`, `sim.y_ssm`, `sim.y`, `sim.z`,
 *   `sim.gated`, `sim.rms_gate`, and `sim.y_norm`.
 *   NEVER reruns `runMambaSim` on a shortened prefix and NEVER mutates any simulator buffer.
 *
 * Pure headless TypeScript — zero DOM, zero WebGL2, fully deterministic.
 */

import {
  formatMambaHeadId,
  mambaGroupForHead,
  readMambaConvTapValue,
  readMambaPriorSsmValue,
  type MambaSimResult,
} from '../../math/mambaSim';

/** First bidirectional GQA bay in the 52-layer Nemotron hybrid motif (1-based bay 6, layer index 5). */
export const FIRST_BIDIRECTIONAL_GQA_BAY_ID = 6 as const;

export type MambaConvTapSource =
  | 'inherited_context_history'
  | 'active_block_token'
  | 'zero_pad';

export interface MambaConvTapSlotProvenance {
  /** Buffer slot index `0..3` (`0` = oldest lag 3, `3` = newest lag 0). */
  readonly slotIndex: number;
  /** Causal lag `3, 2, 1, 0` (`lag = 3 - slotIndex`). */
  readonly lag: number;
  readonly source: MambaConvTapSource;
  /** Absolute sequence position of this tap, or `null` if zero-padded prior to position 0. */
  readonly absoluteTokenPosition: number | null;
  /** Local block index `0..B-1` when `source === 'active_block_token'`, else `null`. */
  readonly localBlockPosition: number | null;
  /** Source column slot `1..3` inside the 4-column `initialConvState` cache when inherited, else `null`. */
  readonly inheritedCacheSlot: number | null;
  /** Active head's 4 channel values (`p = 0..3`) entering this Conv1D tap. */
  readonly headChannelValues: Float32Array;
  /** Active head's 4 depthwise Conv1D weights (`p = 0..3`) for this tap. */
  readonly headKernelWeights: Float32Array;
}

export interface MambaConvWindowTracker {
  readonly localPos: number;
  readonly absoluteTokenPosition: number;
  /** NVIDIA allocated Conv1D cache convention: 4 columns (`slots 0..3`, slot 3 = newest prior token). */
  readonly allocatedCacheColumns: number;
  /** Number of preceding raw history columns (`lags 3, 2, 1`, cache slots `1..3`) available at `localPos = 0`. */
  readonly precedingHistoryColumnsAtPos0: number;
  /** Exact count of inherited Context raw history columns still active in the 4-tap window (`3 -> 2 -> 1 -> 0` for `P >= 3`). */
  readonly inheritedHistoryColumnCount: number;
  /** Exact count of active-block token columns in the 4-tap window (`1 -> 2 -> 3 -> 4`). */
  readonly blockColumnCount: number;
  /** Count of causal left zero-pad columns (`0` whenever `P >= 3` and `initialConvState` is seeded). */
  readonly zeroPadColumnCount: number;
  /** Ordered oldest-to-newest (`slotIndex = 0..3`, `lag = 3, 2, 1, 0`). */
  readonly taps: readonly MambaConvTapSlotProvenance[];
  /** Explicit documentation separating 4 allocated cache columns from 3 active preceding history columns. */
  readonly cacheConventionNote: string;
}

export interface MambaHeadDecayProfileEntry {
  readonly headIndex: number;
  readonly headLabel: string;
  readonly groupIndex: number;
  readonly continuousA: number;
  readonly dtAtPos: number;
  readonly dAAtPos: number;
  readonly cumulativeDecay: number;
  readonly hCurrL2Norm: number;
  readonly inheritedRetentionL2Norm: number;
  readonly accumulatedBlockWriteL2Norm: number;
}

export interface MambaPositionDecomposition {
  /** Complete miniature Mamba-2 logical inventory (16 heads, 8 groups, 4×4 per-head SSM state). */
  readonly modelHeadCount: number;
  readonly modelGroupCount: number;
  readonly ssmStateMatrixShape: readonly [number, number];
  /** Selected head slice index (`0..15`) and owning group (`0..7`) shown in the 4×4 decomposition plates. */
  readonly headIndex: number;
  readonly headLabel: string;
  readonly groupIndex: number;
  readonly localPos: number;
  readonly blockLength: number;
  readonly absoluteTokenPosition: number;
  readonly tokenId: number;
  readonly tokenLabel: string;
  readonly hasInitialSsmSeed: boolean;
  readonly hasInitialConvSeed: boolean;

  /** 4-tap Conv1D boundary & active-block column replacement tracker (`3 -> 2 -> 1 -> 0` inherited). */
  readonly convWindow: MambaConvWindowTracker;

  /** Native `4×4` (`dHead × dState`, row-major `[p * 4 + n]`) SSM boundary seed `H_init` before localPos 0. */
  readonly H_init: Float32Array;
  /** Native `4×4` prior SSM state `H_prev` (`initialSsmState` at `localPos = 0`, `H_history[localPos - 1]` at `localPos > 0`). */
  readonly H_prev: Float32Array;
  readonly decayFactor: number;
  /** Native `4×4` decayed prior term `dA * H_prev`. */
  readonly decayedPrev: Float32Array;
  readonly dtValue: number;
  /** Active head `x[h, :, localPos]` (length 4). */
  readonly xVec: Float32Array;
  /** Active group `B[g, :, localPos]` (length 4). */
  readonly bVec: Float32Array;
  /** Active group `C[g, :, localPos]` (length 4). */
  readonly cVec: Float32Array;
  /** Native `4×4` input write outer product `dt * x ⊗ B`. */
  readonly inputWrite: Float32Array;
  /** Exact native `4×4` current SSM state `H_curr = sim.H_history[localPos]` (0 ULP match). */
  readonly H_curr: Float32Array;

  /** Pre-skip SSM readout `y_ssm[h, :, localPos]` (length 4). */
  readonly recurrentReadout: Float32Array;
  /** Per-head skip scalar `D[h]`. */
  readonly skipD: number;
  /** Skip contribution `D[h] * x[h, :, localPos]` (length 4). */
  readonly skipContribution: Float32Array;
  /** Post-skip SSM output `y[h, :, localPos] = y_ssm + D * x` (length 4). */
  readonly ssmOutputY: Float32Array;
  /** Gate bypass input `z[h, :, localPos]` (length 4). */
  readonly zGateInput: Float32Array;
  /** Gated pre-norm activation `gated[h*4..h*4+3, localPos] = y * SiLU(z)` (length 4). */
  readonly gatedActivation: Float32Array;
  /** Per-group RMS denominator `rms_gate[groupIndex, localPos]`. */
  readonly groupRmsDenominator: number;
  /** Final 8-group normalized output `y_norm[h*4..h*4+3, localPos]` (length 4). */
  readonly normalizedOutput: Float32Array;

  /** Analytical cumulative decay `Π_{k=0..localPos} dA[h, k]`. */
  readonly cumulativeDecay: number;
  /** Analytical inherited retention `cumulativeDecay * H_init` (`4×4`). */
  readonly inheritedRetentionMatrix: Float32Array;
  /** Analytical accumulated block write `Σ_{k=0..localPos} (Π_{m=k+1..localPos} dA[h, m]) * (dt[k] * x[k] ⊗ B[k])` (`4×4`). */
  readonly accumulatedBlockWriteMatrix: Float32Array;
  /** Analytical sum `inheritedRetentionMatrix + accumulatedBlockWriteMatrix` (`4×4`). */
  readonly reconstructedSumMatrix: Float32Array;
  /** `max |reconstructedSumMatrix[i] - H_curr[i]|` across all 16 entries. */
  readonly roundingResidualMaxAbs: number;
  /** Explicit IEEE-754 Float32 sequential-vs-analytical rounding explanation. */
  readonly roundingNote: string;

  /** Raw algebraic Frobenius/L2 norms (paired with `decaySemanticsDisclaimer`). */
  readonly hCurrL2Norm: number;
  readonly inheritedRetentionL2Norm: number;
  readonly accumulatedBlockWriteL2Norm: number;

  /** Heterogeneous decay & state norms across ALL 16 heads (`H00..H15`) at `localPos`. */
  readonly perHeadDecayProfile: readonly MambaHeadDecayProfileEntry[];

  /** 1-based bay ID when supplied, else `null`. */
  readonly bayId: number | null;
  /** True strictly for Mamba bays AFTER the first bidirectional GQA bay (Bay 06, layer index 5). */
  readonly isAfterFirstBidirectionalGqa: boolean;
  /** Mandatory epistemic disclaimer: `dA` and state norms do NOT measure semantic information retention. */
  readonly decaySemanticsDisclaimer: string;
  /** Mandatory architectural note on local Mamba causality vs upstream bidirectional GQA mixing. */
  readonly causalAfterGqaNote: string;
}

export interface ExtractMambaDecompositionOptions {
  readonly bayId?: number;
}

function l2Norm(arr: Float32Array): number {
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    sumSq += arr[i] * arr[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Extract exact position-specific Mamba-2 state, 4-tap Conv1D window provenance, grouped RMS state,
 * and closed-form analytical decomposition (`inheritedRetention + accumulatedBlockWrite`) strictly
 * from the ALREADY EXECUTED complete block `MambaSimResult` (never rerunning `runMambaSim`).
 */
export function extractMambaFullBlockPositionState(
  sim: MambaSimResult,
  headIndex: number,
  localPos: number,
  options?: ExtractMambaDecompositionOptions,
): MambaPositionDecomposition {
  if (!Number.isInteger(headIndex) || headIndex < 0 || headIndex >= sim.nHeads) {
    throw new RangeError(
      `Mamba decomposition headIndex (${headIndex}) must be an integer in [0..${sim.nHeads - 1}]`,
    );
  }
  if (!Number.isInteger(localPos) || localPos < 0 || localPos >= sim.tokens) {
    throw new RangeError(
      `Mamba decomposition localPos (${localPos}) must be an integer in [0..${sim.tokens - 1}]`,
    );
  }

  const T = sim.tokens;
  const { nHeads, dHead, dState, convKernel } = sim;
  const stateStride = nHeads * dHead * dState;
  const matSize = dHead * dState; // 4 × 4 = 16
  const g = mambaGroupForHead(headIndex);
  const absPos = sim.startTokenOffset + localPos;

  // ==========================================================================
  // 1. 4-TAP CONV1D BOUNDARY & ACTIVE-BLOCK REPLACEMENT TRACKER
  // ==========================================================================
  const taps: MambaConvTapSlotProvenance[] = [];
  let inheritedHistoryColumnCount = 0;
  let blockColumnCount = 0;
  let zeroPadColumnCount = 0;

  for (let slot = 0; slot < convKernel; slot++) {
    const lag = convKernel - 1 - slot; // slot 0 -> lag 3 ... slot 3 -> lag 0
    const localSrcT = localPos - lag;
    const absSrcT = sim.startTokenOffset + localSrcT;

    let source: MambaConvTapSource;
    let absoluteTokenPosition: number | null;
    let localBlockPosition: number | null;
    let inheritedCacheSlot: number | null;

    if (localSrcT >= 0) {
      source = 'active_block_token';
      absoluteTokenPosition = absSrcT;
      localBlockPosition = localSrcT;
      inheritedCacheSlot = null;
      blockColumnCount++;
    } else {
      const initSlot = convKernel + localSrcT; // localSrcT in {-3,-2,-1} -> initSlot in {1,2,3}
      if (
        sim.initialConvState &&
        initSlot >= 0 &&
        initSlot < convKernel &&
        absSrcT >= 0
      ) {
        source = 'inherited_context_history';
        absoluteTokenPosition = absSrcT;
        localBlockPosition = null;
        inheritedCacheSlot = initSlot;
        inheritedHistoryColumnCount++;
      } else {
        source = 'zero_pad';
        absoluteTokenPosition = null;
        localBlockPosition = null;
        inheritedCacheSlot = null;
        zeroPadColumnCount++;
      }
    }

    const headChannelValues = new Float32Array(dHead);
    const headKernelWeights = new Float32Array(dHead);
    for (let p = 0; p < dHead; p++) {
      const ch = headIndex * dHead + p;
      headChannelValues[p] = readMambaConvTapValue(sim, ch, localPos, slot).value;
      headKernelWeights[p] = sim.w_conv[ch * convKernel + slot];
    }

    taps.push(
      Object.freeze({
        slotIndex: slot,
        lag,
        source,
        absoluteTokenPosition,
        localBlockPosition,
        inheritedCacheSlot,
        headChannelValues,
        headKernelWeights,
      }),
    );
  }

  const precedingHistoryColumnsAtPos0 = convKernel - 1; // 3 preceding raw history columns (lags 3, 2, 1)
  const convWindow: MambaConvWindowTracker = Object.freeze({
    localPos,
    absoluteTokenPosition: absPos,
    allocatedCacheColumns: convKernel,
    precedingHistoryColumnsAtPos0,
    inheritedHistoryColumnCount,
    blockColumnCount,
    zeroPadColumnCount,
    taps: Object.freeze(taps),
    cacheConventionNote:
      `NVIDIA allocates ${convKernel} Conv1D cache columns (slots [0..3], slot 3 = newest prior token); ` +
      `at localPos=0, exactly ${precedingHistoryColumnsAtPos0} preceding raw history columns (lags 3, 2, 1 → cache slots 1, 2, 3) ` +
      `combine with lag 0 (xBC_pre[0]). At localPos=${localPos} (abs t${absPos}): ` +
      `${inheritedHistoryColumnCount} inherited Context history col${inheritedHistoryColumnCount === 1 ? '' : 's'} + ` +
      `${blockColumnCount} active-block col${blockColumnCount === 1 ? '' : 's'}` +
      (zeroPadColumnCount > 0 ? ` + ${zeroPadColumnCount} zero-pad col${zeroPadColumnCount === 1 ? '' : 's'}` : '') +
      '.',
  });

  // ==========================================================================
  // 2. POSITION-SPECIFIC NATIVE 4×4 SSM STATE & GROUPED RMS READOUT
  // ==========================================================================
  const H_init = new Float32Array(matSize);
  const H_prev = new Float32Array(matSize);
  const decayedPrev = new Float32Array(matSize);
  const inputWrite = new Float32Array(matSize);
  const H_curr = new Float32Array(matSize);

  const decayFactor = sim.dA[headIndex * T + localPos];
  const dtValue = sim.dt[headIndex * T + localPos];

  const xVec = new Float32Array(dHead);
  const bVec = new Float32Array(dState);
  const cVec = new Float32Array(dState);
  const recurrentReadout = new Float32Array(dHead);
  const skipD = sim.D[headIndex];
  const skipContribution = new Float32Array(dHead);
  const ssmOutputY = new Float32Array(dHead);
  const zGateInput = new Float32Array(dHead);
  const gatedActivation = new Float32Array(dHead);
  const groupRmsDenominator = sim.rms_gate[g * T + localPos];
  const normalizedOutput = new Float32Array(dHead);

  for (let n = 0; n < dState; n++) {
    bVec[n] = sim.B[(g * dState + n) * T + localPos];
    cVec[n] = sim.C[(g * dState + n) * T + localPos];
  }

  for (let p = 0; p < dHead; p++) {
    const ch = headIndex * dHead + p;
    const xVal = sim.x[ch * T + localPos];
    xVec[p] = xVal;
    recurrentReadout[p] = sim.y_ssm[ch * T + localPos];
    skipContribution[p] = Math.fround(skipD * xVal);
    ssmOutputY[p] = sim.y[ch * T + localPos];
    zGateInput[p] = sim.z[ch * T + localPos];
    gatedActivation[p] = sim.gated[ch * T + localPos];
    normalizedOutput[p] = sim.y_norm[ch * T + localPos];

    for (let n = 0; n < dState; n++) {
      const matIdx = p * dState + n;
      const localIdx = (headIndex * dHead + p) * dState + n;
      H_init[matIdx] = sim.initialSsmState ? sim.initialSsmState[localIdx] : 0;
      const prevVal = readMambaPriorSsmValue(sim, localPos, localIdx).value;
      H_prev[matIdx] = prevVal;
      decayedPrev[matIdx] = Math.fround(decayFactor * prevVal);
      inputWrite[matIdx] = Math.fround(dtValue * xVal * bVec[n]);
      H_curr[matIdx] = sim.H_history[localPos * stateStride + localIdx];
    }
  }

  // ==========================================================================
  // 3. ANALYTICAL DECOMPOSITION (INHERITED RETENTION + ACCUMULATED BLOCK WRITE)
  //    AND 16-HEAD HETEROGENEOUS DECAY PROFILE
  // ==========================================================================
  const computeHeadAnalytical = (h: number) => {
    const grp = mambaGroupForHead(h);
    // Suffix products: suffixDecay[k] = Π_{m=k+1..localPos} dA[h, m]
    const suffixDecay = new Float64Array(localPos + 1);
    suffixDecay[localPos] = 1.0;
    for (let k = localPos - 1; k >= 0; k--) {
      suffixDecay[k] = suffixDecay[k + 1] * sim.dA[h * T + (k + 1)];
    }
    const cumDecay = suffixDecay[0] * sim.dA[h * T + 0];

    const retMat = new Float32Array(matSize);
    const writeMat = new Float32Array(matSize);
    const sumMat = new Float32Array(matSize);
    const currMat = new Float32Array(matSize);
    let maxAbsErr = 0;

    for (let p = 0; p < dHead; p++) {
      for (let n = 0; n < dState; n++) {
        const matIdx = p * dState + n;
        const localIdx = (h * dHead + p) * dState + n;
        const initVal = sim.initialSsmState ? sim.initialSsmState[localIdx] : 0;
        const retVal64 = cumDecay * initVal;

        let writeAcc64 = 0;
        for (let k = 0; k <= localPos; k++) {
          const dtK = sim.dt[h * T + k];
          const xK = sim.x[(h * dHead + p) * T + k];
          const bK = sim.B[(grp * dState + n) * T + k];
          writeAcc64 += suffixDecay[k] * (dtK * xK * bK);
        }

        const retF32 = Math.fround(retVal64);
        const writeF32 = Math.fround(writeAcc64);
        const sumF32 = Math.fround(retVal64 + writeAcc64);
        const hCurrVal = sim.H_history[localPos * stateStride + localIdx];

        retMat[matIdx] = retF32;
        writeMat[matIdx] = writeF32;
        sumMat[matIdx] = sumF32;
        currMat[matIdx] = hCurrVal;

        const diff = Math.abs(sumF32 - hCurrVal);
        if (diff > maxAbsErr) maxAbsErr = diff;
      }
    }

    return {
      cumDecay,
      retMat,
      writeMat,
      sumMat,
      currMat,
      maxAbsErr,
    };
  };

  let activeAnalytical: ReturnType<typeof computeHeadAnalytical> | null = null;
  const perHeadDecayProfile: MambaHeadDecayProfileEntry[] = [];

  for (let h = 0; h < nHeads; h++) {
    const res = computeHeadAnalytical(h);
    if (h === headIndex) {
      activeAnalytical = res;
    }
    perHeadDecayProfile.push(
      Object.freeze({
        headIndex: h,
        headLabel: formatMambaHeadId(h),
        groupIndex: mambaGroupForHead(h),
        continuousA: sim.A[h],
        dtAtPos: sim.dt[h * T + localPos],
        dAAtPos: sim.dA[h * T + localPos],
        cumulativeDecay: res.cumDecay,
        hCurrL2Norm: l2Norm(res.currMat),
        inheritedRetentionL2Norm: l2Norm(res.retMat),
        accumulatedBlockWriteL2Norm: l2Norm(res.writeMat),
      }),
    );
  }

  const active = activeAnalytical!;
  const hCurrL2Norm = l2Norm(H_curr);
  const inheritedRetentionL2Norm = l2Norm(active.retMat);
  const accumulatedBlockWriteL2Norm = l2Norm(active.writeMat);

  const bayId = options?.bayId !== undefined ? Math.trunc(options.bayId) : null;
  const isAfterFirstBidirectionalGqa = bayId !== null && bayId > FIRST_BIDIRECTIONAL_GQA_BAY_ID;

  const roundingNote =
    `Sequential IEEE-754 Float32 recurrence rounds after each step k=0..${localPos}, whereas the closed-form ` +
    `analytical decomposition evaluates (Π dA)·H_init + Σ (Π dA)·Δt·x·B in float64 before final Float32 rounding ` +
    `(max |sum − H_curr| = ${active.maxAbsErr.toExponential(2)}).`;

  const decaySemanticsDisclaimer =
    'Epistemic note: Per-head decay factors dA = exp(Δt·A), cumulative decay Π dA, and SSM matrix norms ' +
    'reflect raw algebraic state magnitudes and do NOT measure retained semantic information or token importance.';

  const causalAfterGqaNote = isAfterFirstBidirectionalGqa
    ? `Bay ${String(bayId).padStart(2, '0')} (layer ${bayId! - 1}) sits AFTER the first bidirectional GQA bay ` +
      `(Bay 06, layer 5): while Mamba-2's Conv1D and SSM recurrence remain strictly locally causal across block ` +
      `positions 0..${T - 1}, its input activations X may already incorporate information from later block positions.`
    : bayId !== null
      ? `Bay ${String(bayId).padStart(2, '0')} (layer ${bayId - 1}) sits BEFORE the first bidirectional GQA bay ` +
        `(Bay 06, layer 5): both its input stream X and Mamba-2's Conv1D/SSM recurrence remain strictly locally causal ` +
        `across block positions 0..${T - 1}.`
      : `Mamba-2's Conv1D and SSM recurrence remain strictly locally causal across block positions 0..${T - 1}; ` +
        `in layers after the first bidirectional GQA bay (Bay 06, layer 5), input activations X may already ` +
        `incorporate information from later block positions.`;

  return Object.freeze({
    modelHeadCount: sim.nHeads,
    modelGroupCount: sim.nGroups,
    ssmStateMatrixShape: Object.freeze([dHead, dState] as const),
    headIndex,
    headLabel: formatMambaHeadId(headIndex),
    groupIndex: g,
    localPos,
    blockLength: T,
    absoluteTokenPosition: absPos,
    tokenId: sim.tokenIds[localPos],
    tokenLabel: sim.tokenLabels[localPos] ?? `t${absPos}`,
    hasInitialSsmSeed: sim.initialSsmState !== undefined,
    hasInitialConvSeed: sim.initialConvState !== undefined,
    convWindow,
    H_init,
    H_prev,
    decayFactor,
    decayedPrev,
    dtValue,
    xVec,
    bVec,
    cVec,
    inputWrite,
    H_curr,
    recurrentReadout,
    skipD,
    skipContribution,
    ssmOutputY,
    zGateInput,
    gatedActivation,
    groupRmsDenominator,
    normalizedOutput,
    cumulativeDecay: active.cumDecay,
    inheritedRetentionMatrix: active.retMat,
    accumulatedBlockWriteMatrix: active.writeMat,
    reconstructedSumMatrix: active.sumMat,
    roundingResidualMaxAbs: active.maxAbsErr,
    roundingNote,
    hCurrL2Norm,
    inheritedRetentionL2Norm,
    accumulatedBlockWriteL2Norm,
    perHeadDecayProfile: Object.freeze(perHeadDecayProfile),
    bayId,
    isAfterFirstBidirectionalGqa,
    decaySemanticsDisclaimer,
    causalAfterGqaNote,
  });
}
