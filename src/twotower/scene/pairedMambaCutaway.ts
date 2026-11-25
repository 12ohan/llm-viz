/**
 * Asymmetric Paired 35/65 TwoTower Mamba-2 Cutaway (`src/twotower/scene/pairedMambaCutaway.ts`)
 *
 * Builds a unified `TwoTowerSpatialScene` for any of the 23 aligned Mamba-2 bays:
 *   - Left Wing (35% width, `x ∈ [-13.5, -3.5]`): Context Tower provenance showing BOTH separate
 *     inherited persistent state channels independently:
 *       1. 4-column Conv1D raw history cache (`convState`, `traceOrigin: 'persistent_cache'`)
 *       2. 16 × [4×4] FP32 SSM boundary state `H_{P-1}` (`traceOrigin: 'persistent_cache'`)
 *     plus actual Context segment execution trace (`context_computed` / `context_reconstructed`) or
 *     explicit pruned notice card when historical Context execution traces were pruned.
 *   - Center Dual-Channel Bridge (`x ∈ [-3.5, -1.8]`): TWO cleanly separated cross-tower transfer
 *     ribbons (Conv1D raw history channel + FP32 SSM boundary seed channel) explicitly annotated:
 *     "Every refinement r seeds from the SAME committed Context boundary (never prior Denoiser scratch)".
 *   - Right Wing (65% width, `x ∈ [-1.8, +13.5]`): Active Denoiser Mamba computation
 *     (`denoiser_scratch`) backed by a dedicated 256×128 live RG32F value atlas packing native
 *     `[4×4]` `H_prev`, `dA·H_prev`, `Δt·x·B`, `H_curr`, `Inherited Retention`, `Block Write`,
 *     4-tap Conv1D window showing active-block column replacement (`3→2→1→0` inherited columns),
 *     `y_ssm`, `D·x`, `gated = y·SiLU(z)`, 8-group `rms_gate`, `y_norm`, `residual`, and the
 *     16-head heterogeneous decay bank (`H00..H15`).
 *   - Supports `'paired'`, `'context_solo'`, and `'denoiser_solo'` focus modes.
 *
 * Pure headless TypeScript — zero DOM, zero WebGL2, fully deterministic.
 */

import type { MambaLayerBoundaryState } from '../../math/cacheContracts';
import {
  describeMambaCellFormula,
  formatMambaHeadId,
  inferMambaCellHead,
  inferMambaCellToken,
  MAMBA_STEPPER_STEPS,
  mambaGroupForHead,
  mambaStepperCellForState,
  readMambaConvTapValue,
  type MambaCellFormula,
  type MambaCellRef,
  type MambaSheetSpan,
  type MambaSimResult,
  type MambaTensorKey,
} from '../../math/mambaSim';
import { MIN_CAMERA_DISTANCE, type CameraPose } from '../../model/state';
import type { AABB, ViewportMetrics } from '../../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  pushBoxFrameXZ,
  pushFreeGuideLine,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetBasisAxes,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import {
  computeMambaFocusedAccumulation,
  type FocusedAccumulationResult,
} from '../../scene/accumulation';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
  type TexRect,
} from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  buildMambaCellPopover,
  computeMambaSheetReveal,
  fitSpatialMambaCamera,
  type MambaRevealParams,
  type SpatialMambaFitOptions,
} from '../../scene/mambaSpatial';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  createSheetFocusSet,
  drawCellHighlight3D,
  drawOperandDependency3D,
  EMPTY_SHEET_FOCUS_SET,
  resolveSheetHighlight,
  RIBBON_RESIDUAL,
  RIBBON_SSM_LOOP,
  spanFromSheetGrid,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type WorldSpan3D,
} from '../../scene/sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../ui/overlay';
import type { TwoTowerTraceOrigin } from '../contracts';
import type { TwoTowerSession } from '../session';
import {
  extractMambaFullBlockPositionState,
  FIRST_BIDIRECTIONAL_GQA_BAY_ID,
  type MambaPositionDecomposition,
} from './mambaDecomposition';
import {
  TWOTOWER_MINI_MODEL_INVENTORY,
  buildTwoTowerSelectionIdentity,
  resolveNearestBayOfKind,
  resolveTwoTowerSessionViewState,
  type TwoTowerPairedFocus,
  type TwoTowerProvenanceBadge,
  type TwoTowerSelectionIdentity,
  type TwoTowerSpatialScene,
  type TwoTowerTowerRole,
} from './provenance';

// ============================================================================
// CONSTANTS, BOUNDS & DEDICATED 256×128 PAIRED MAMBA RG32F ATLAS
// ============================================================================

export const TWOTOWER_MAMBA_ATLAS_WIDTH = 256 as const;
export const TWOTOWER_MAMBA_ATLAS_HEIGHT = 128 as const;

export const PAIRED_MAMBA_FIT_BOUNDS: AABB = {
  min: [-13.85, -0.10, -4.10],
  max: [13.95, 3.35, 4.10],
};

export const CONTEXT_SOLO_MAMBA_FIT_BOUNDS: AABB = {
  min: [-9.25, -0.10, -3.95],
  max: [9.25, 3.20, 3.95],
};

export const DENOISER_SOLO_MAMBA_FIT_BOUNDS: AABB = {
  min: [-9.85, -0.10, -4.05],
  max: [9.85, 3.35, 4.05],
};

export interface TwoTowerMambaValueAtlas extends Rg32fValueAtlas {
  readonly width: typeof TWOTOWER_MAMBA_ATLAS_WIDTH;
  readonly height: typeof TWOTOWER_MAMBA_ATLAS_HEIGHT;
  readonly activeHead: number;
  readonly localPos: number;
  readonly wIn: AtlasSlice;
  readonly wOut: AtlasSlice;
  readonly input: AtlasSlice;
  readonly modulatedInput: AtlasSlice | null;
  readonly rmsnorm: AtlasSlice;
  readonly residual: AtlasSlice;
  readonly xbcPre: AtlasSlice;
  readonly z: AtlasSlice;
  readonly dt: AtlasSlice;
  readonly convWindowTaps: readonly AtlasSlice[]; // 4 taps [0..3], each [128 × 1]
  readonly xHeads: readonly AtlasSlice[]; // 16 heads × [4 × T]
  readonly bGroups: AtlasSlice;
  readonly cGroups: AtlasSlice;
  readonly hPrev: AtlasSlice; // [4 × 4]
  readonly decayedPrev: AtlasSlice; // [4 × 4] dA * H_prev
  readonly inputWrite: AtlasSlice; // [4 × 4] dt * x * B
  readonly hCurr: AtlasSlice; // [4 × 4]
  readonly inheritedRetention: AtlasSlice; // [4 × 4]
  readonly accumulatedBlockWrite: AtlasSlice; // [4 × 4]
  readonly decayBankDa: AtlasSlice; // [16 × 1]
  readonly decayBankCumulative: AtlasSlice; // [16 × 1]
  readonly ssmBankSummary: AtlasSlice; // [16 × 4]
  readonly ySsm: AtlasSlice; // [64 × T]
  readonly skipDx: AtlasSlice; // [4 × 1] active head skip D·x at localPos
  readonly y: AtlasSlice; // [64 × T]
  readonly gated: AtlasSlice; // [64 × T]
  readonly rmsGate: AtlasSlice; // [8 × T]
  readonly yNorm: AtlasSlice; // [64 × T]
  // Context persistent boundary channels (always packed when boundary state available)
  readonly ctxConvCacheTaps: readonly AtlasSlice[] | null; // 4 persistent columns [128 × 1]
  readonly ctxSsmBoundaryHead: AtlasSlice | null; // [4 × 4] active head H_{P-1}
  readonly ctxSsmBoundaryBank: AtlasSlice | null; // [16 × 4] 16-head persistent summary
  // Optional secondary Context segment execution slices
  readonly secondary: {
    readonly input: AtlasSlice;
    readonly rmsnorm: AtlasSlice;
    readonly residual: AtlasSlice;
    readonly dt: AtlasSlice;
    readonly xbcPre: AtlasSlice;
    readonly y: AtlasSlice;
    readonly yNorm: AtlasSlice;
    readonly hCurr: AtlasSlice;
  } | null;
}

export interface BuildPairedMambaAtlasParams {
  readonly primarySim: MambaSimResult;
  readonly decomposition: MambaPositionDecomposition;
  readonly contextBoundaryState?: MambaLayerBoundaryState | null;
  readonly secondaryContextSim?: MambaSimResult | null;
}

export interface TwoTowerMambaAtlasVerification {
  readonly valid: boolean;
  readonly width: number;
  readonly height: number;
  readonly byteLength: number;
  readonly checkedSliceCount: number;
  readonly errors: readonly string[];
}

/**
 * Pack all live Denoiser Mamba-2 tensors, native `[4×4]` analytical decomposition plates,
 * 4-tap Conv1D window, 8-group `rms_gate`, 16-head decay bank, Context persistent Conv1D & SSM
 * boundary channels, and optional Context segment trace into ONE disjoint 256×128 RG32F atlas.
 */
export function buildTwoTowerMambaValueAtlas(
  params: BuildPairedMambaAtlasParams,
): TwoTowerMambaValueAtlas {
  const { primarySim: sim, decomposition: decomp, contextBoundaryState = null, secondaryContextSim = null } = params;
  const W = TWOTOWER_MAMBA_ATLAS_WIDTH;
  const H = TWOTOWER_MAMBA_ATLAS_HEIGHT;
  const { d, nHeads, nGroups, dHead, dInner, dState, convKernel, convDim, inProjDim, tokens: T } = sim;
  const head = decomp.headIndex;
  const tokIdx = decomp.localPos;
  const stateStride = nHeads * dHead * dState;

  const writer = createRg32fAtlas(W, H);
  const { setTexel, makeRect } = writer;

  // y = 0..7: W_in [208 × 8] at x = 0..207
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < inProjDim; c++) {
      setTexel(c, r, sim.W_in[c * d + r]);
    }
  }

  // Native [4×4] SSM state & decomposition plates at y = 0..3, x = 208..231
  const hMax = Math.max(
    computeMaxAbs(decomp.H_prev, 0.25),
    computeMaxAbs(decomp.H_curr, 0.25),
    computeMaxAbs(decomp.inheritedRetentionMatrix, 0.25),
    computeMaxAbs(decomp.accumulatedBlockWriteMatrix, 0.25),
    computeMaxAbs(decomp.inputWrite, 0.25),
    0.35,
  );
  const hScale = 1 / hMax;
  for (let p = 0; p < dHead; p++) {
    for (let n = 0; n < dState; n++) {
      const idx = p * dState + n;
      setTexel(208 + n, p, decomp.H_prev[idx]);
      setTexel(212 + n, p, decomp.decayedPrev[idx]);
      setTexel(216 + n, p, decomp.inputWrite[idx]);
      setTexel(220 + n, p, decomp.H_curr[idx]);
      setTexel(224 + n, p, decomp.inheritedRetentionMatrix[idx]);
      setTexel(228 + n, p, decomp.accumulatedBlockWriteMatrix[idx]);
    }
  }

  // 16-head decay profile strips at x = 232..247, y = 0 (dA) & y = 1 (cumulativeDecay)
  for (let h = 0; h < nHeads; h++) {
    const entry = decomp.perHeadDecayProfile[h];
    setTexel(232 + h, 0, entry.dAAtPos);
    setTexel(232 + h, 1, entry.cumulativeDecay);
  }
  // Active head skip D·x at x = 232..235, y = 2
  for (let p = 0; p < dHead; p++) {
    setTexel(232 + p, 2, decomp.skipContribution[p]);
  }

  // 16-head current SSM bank summary [16 × 4] at x = 228..243, y = 4..7
  for (let h = 0; h < nHeads; h++) {
    for (let p = 0; p < dHead; p++) {
      let meanVal = 0;
      for (let n = 0; n < dState; n++) {
        meanVal += sim.H_history[tokIdx * stateStride + (h * dHead + p) * dState + n];
      }
      setTexel(228 + h, 4 + p, meanVal / dState);
    }
  }

  // Context persistent SSM boundary state at x = 208..227, y = 4..7
  let ctxSsmBoundaryHead: AtlasSlice | null = null;
  let ctxSsmBoundaryBank: AtlasSlice | null = null;
  if (contextBoundaryState) {
    const bScale = 1 / computeMaxAbs(contextBoundaryState.ssmState, 0.35);
    for (let p = 0; p < dHead; p++) {
      for (let n = 0; n < dState; n++) {
        const localIdx = (head * dHead + p) * dState + n;
        setTexel(208 + n, 4 + p, contextBoundaryState.ssmState[localIdx]);
      }
    }
    for (let h = 0; h < nHeads; h++) {
      for (let p = 0; p < dHead; p++) {
        let meanVal = 0;
        for (let n = 0; n < dState; n++) {
          meanVal += contextBoundaryState.ssmState[(h * dHead + p) * dState + n];
        }
        setTexel(212 + h, 4 + p, meanVal / dState);
      }
    }
    ctxSsmBoundaryHead = { rect: makeRect(208, 4, dState, dHead), scale: bScale, mode: 0 };
    ctxSsmBoundaryBank = { rect: makeRect(212, 4, nHeads, dHead), scale: bScale, mode: 0 };
  }

  // y = 8..15: W_out [64 × 8] at x = 0..63, Conv1D window [128 × 4] at (64, 8), Context convState [128 × 4] at (64, 12)
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < dInner; c++) {
      setTexel(c, 8 + i, sim.W_out[i * dInner + c]);
    }
  }

  const convScale =
    1 /
    Math.max(
      computeMaxAbs(sim.xBC_pre, 0.35),
      sim.initialConvState ? computeMaxAbs(sim.initialConvState, 0.35) : 0,
      contextBoundaryState ? computeMaxAbs(contextBoundaryState.convState, 0.35) : 0,
    );

  const convWindowTaps: AtlasSlice[] = [];
  for (let s = 0; s < convKernel; s++) {
    for (let c = 0; c < convDim; c++) {
      setTexel(64 + c, 8 + s, readMambaConvTapValue(sim, c, tokIdx, s).value);
    }
    convWindowTaps.push({ rect: makeRect(64, 8 + s, convDim, 1), scale: convScale, mode: 0 });
  }

  let ctxConvCacheTaps: AtlasSlice[] | null = null;
  if (contextBoundaryState) {
    ctxConvCacheTaps = [];
    for (let s = 0; s < convKernel; s++) {
      for (let c = 0; c < convDim; c++) {
        setTexel(64 + c, 12 + s, contextBoundaryState.convState[c * convKernel + s]);
      }
      ctxConvCacheTaps.push({ rect: makeRect(64, 12 + s, convDim, 1), scale: convScale, mode: 0 });
    }
  }

  // y = 16..31: Sequence band 1 (xBC_pre, z, dt, X, X_mod, U, Y, rms_gate)
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < convDim; c++) setTexel(c, 16 + t, sim.xBC_pre[c * T + t]);
    for (let c = 0; c < dInner; c++) setTexel(128 + c, 16 + t, sim.z[c * T + t]);
    for (let h = 0; h < nHeads; h++) setTexel(192 + h, 16 + t, sim.dt[h * T + t]);
    for (let i = 0; i < d; i++) {
      setTexel(208 + i, 16 + t, sim.X[i * T + t]);
      if (sim.X_mod) setTexel(216 + i, 16 + t, sim.X_mod[i * T + t]);
      setTexel(224 + i, 16 + t, sim.U[i * T + t]);
      setTexel(232 + i, 16 + t, sim.Y[i * T + t]);
    }
    for (let g = 0; g < nGroups; g++) {
      setTexel(240 + g, 16 + t, sim.rms_gate[g * T + t]);
    }
  }

  // y = 32..47: Sequence band 2 (x, B, C, y, y_norm)
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < dInner; c++) {
      setTexel(c, 32 + t, sim.x[c * T + t]);
      setTexel(128 + c, 32 + t, sim.y[c * T + t]);
      setTexel(192 + c, 32 + t, sim.y_norm[c * T + t]);
    }
    for (let k = 0; k < nGroups * dState; k++) {
      setTexel(64 + k, 32 + t, sim.B[k * T + t]);
      setTexel(96 + k, 32 + t, sim.C[k * T + t]);
    }
  }

  // y = 48..63: Primary y_ssm (0..63), Primary gated (64..127), plus Secondary Context scalars (128..239)
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < dInner; c++) {
      setTexel(c, 48 + t, sim.y_ssm[c * T + t]);
      setTexel(64 + c, 48 + t, sim.gated[c * T + t]);
    }
  }

  let secondarySlices: TwoTowerMambaValueAtlas['secondary'] = null;
  if (secondaryContextSim) {
    const T2 = Math.min(16, secondaryContextSim.tokens);
    const lastT2 = Math.max(0, T2 - 1);
    for (let t = 0; t < T2; t++) {
      for (let i = 0; i < d; i++) {
        setTexel(128 + i, 48 + t, secondaryContextSim.X[i * secondaryContextSim.tokens + t]);
        setTexel(136 + i, 48 + t, secondaryContextSim.U[i * secondaryContextSim.tokens + t]);
        setTexel(144 + i, 48 + t, secondaryContextSim.Y[i * secondaryContextSim.tokens + t]);
      }
      for (let h = 0; h < nHeads; h++) {
        setTexel(152 + h, 48 + t, secondaryContextSim.dt[h * secondaryContextSim.tokens + t]);
      }
      for (let c = 0; c < dInner; c++) {
        setTexel(176 + c, 48 + t, secondaryContextSim.y_norm[c * secondaryContextSim.tokens + t]);
        setTexel(128 + c, 64 + t, secondaryContextSim.y[c * secondaryContextSim.tokens + t]);
      }
      for (let c = 0; c < convDim; c++) {
        setTexel(c, 64 + t, secondaryContextSim.xBC_pre[c * secondaryContextSim.tokens + t]);
      }
    }
    for (let p = 0; p < dHead; p++) {
      for (let n = 0; n < dState; n++) {
        const localIdx = (head * dHead + p) * dState + n;
        setTexel(n, 80 + p, secondaryContextSim.H_history[lastT2 * stateStride + localIdx]);
      }
    }
    secondarySlices = {
      input: { rect: makeRect(128, 48, d, T2), scale: 1 / computeMaxAbs(secondaryContextSim.X, 0.4), mode: 0 },
      rmsnorm: { rect: makeRect(136, 48, d, T2), scale: 1 / computeMaxAbs(secondaryContextSim.U, 0.4), mode: 0 },
      residual: { rect: makeRect(144, 48, d, T2), scale: 1 / computeMaxAbs(secondaryContextSim.Y, 0.4), mode: 0 },
      dt: { rect: makeRect(152, 48, nHeads, T2), scale: 1 / computeMaxAbs(secondaryContextSim.dt, 0.5), mode: 1 },
      xbcPre: { rect: makeRect(0, 64, convDim, T2), scale: 1 / computeMaxAbs(secondaryContextSim.xBC_pre, 0.35), mode: 0 },
      y: { rect: makeRect(128, 64, dInner, T2), scale: 1 / computeMaxAbs(secondaryContextSim.y, 0.35), mode: 0 },
      yNorm: { rect: makeRect(176, 48, dInner, T2), scale: 1 / computeMaxAbs(secondaryContextSim.y_norm, 0.4), mode: 0 },
      hCurr: { rect: makeRect(0, 80, dState, dHead), scale: 1 / computeMaxAbs(secondaryContextSim.H_history, 0.35), mode: 0 },
    };
  }

  const xScale = 1 / computeMaxAbs(sim.x, 0.35);
  const xHeads: AtlasSlice[] = [];
  for (let h = 0; h < nHeads; h++) {
    xHeads.push({ rect: makeRect(h * dHead, 32, dHead, T), scale: xScale, mode: 0 });
  }

  return {
    width: W,
    height: H,
    data: writer.data,
    activeHead: head,
    localPos: tokIdx,
    wIn: { rect: makeRect(0, 0, inProjDim, d), scale: 1 / computeMaxAbs(sim.W_in, 0.35), mode: 0 },
    wOut: { rect: makeRect(0, 8, dInner, d), scale: 1 / computeMaxAbs(sim.W_out, 0.25), mode: 0 },
    input: { rect: makeRect(208, 16, d, T), scale: 1 / computeMaxAbs(sim.X, 0.4), mode: 0 },
    modulatedInput: sim.X_mod
      ? { rect: makeRect(216, 16, d, T), scale: 1 / computeMaxAbs(sim.X_mod, 0.4), mode: 0 }
      : null,
    rmsnorm: { rect: makeRect(224, 16, d, T), scale: 1 / computeMaxAbs(sim.U, 0.4), mode: 0 },
    residual: { rect: makeRect(232, 16, d, T), scale: 1 / computeMaxAbs(sim.Y, 0.4), mode: 0 },
    xbcPre: { rect: makeRect(0, 16, convDim, T), scale: convScale, mode: 0 },
    z: { rect: makeRect(128, 16, dInner, T), scale: 1 / computeMaxAbs(sim.z, 0.35), mode: 0 },
    dt: { rect: makeRect(192, 16, nHeads, T), scale: 1 / computeMaxAbs(sim.dt, 0.5), mode: 1 },
    convWindowTaps,
    xHeads,
    bGroups: { rect: makeRect(64, 32, nGroups * dState, T), scale: 1 / computeMaxAbs(sim.B, 0.35), mode: 0 },
    cGroups: { rect: makeRect(96, 32, nGroups * dState, T), scale: 1 / computeMaxAbs(sim.C, 0.35), mode: 0 },
    hPrev: { rect: makeRect(208, 0, dState, dHead), scale: hScale, mode: 0 },
    decayedPrev: { rect: makeRect(212, 0, dState, dHead), scale: hScale, mode: 0 },
    inputWrite: { rect: makeRect(216, 0, dState, dHead), scale: hScale, mode: 0 },
    hCurr: { rect: makeRect(220, 0, dState, dHead), scale: hScale, mode: 0 },
    inheritedRetention: { rect: makeRect(224, 0, dState, dHead), scale: hScale, mode: 0 },
    accumulatedBlockWrite: { rect: makeRect(228, 0, dState, dHead), scale: hScale, mode: 0 },
    decayBankDa: { rect: makeRect(232, 0, nHeads, 1), scale: 1.0, mode: 1 },
    decayBankCumulative: { rect: makeRect(232, 1, nHeads, 1), scale: 1.0, mode: 1 },
    ssmBankSummary: { rect: makeRect(228, 4, nHeads, dHead), scale: hScale, mode: 0 },
    ySsm: { rect: makeRect(0, 48, dInner, T), scale: 1 / computeMaxAbs(sim.y_ssm, 0.35), mode: 0 },
    skipDx: { rect: makeRect(232, 2, dHead, 1), scale: 1 / computeMaxAbs(decomp.skipContribution, 0.35), mode: 0 },
    y: { rect: makeRect(128, 32, dInner, T), scale: 1 / computeMaxAbs(sim.y, 0.35), mode: 0 },
    gated: { rect: makeRect(64, 48, dInner, T), scale: 1 / computeMaxAbs(sim.gated, 0.35), mode: 0 },
    rmsGate: { rect: makeRect(240, 16, nGroups, T), scale: 1 / computeMaxAbs(sim.rms_gate, 0.5), mode: 1 },
    yNorm: { rect: makeRect(192, 32, dInner, T), scale: 1 / computeMaxAbs(sim.y_norm, 0.4), mode: 0 },
    ctxConvCacheTaps,
    ctxSsmBoundaryHead,
    ctxSsmBoundaryBank,
    secondary: secondarySlices,
  };
}

/** Verify every `TexRect` in a `TwoTowerMambaValueAtlas` is in `[0,1]` and strictly non-overlapping. */
export function verifyTwoTowerMambaAtlasLayout(
  atlas: TwoTowerMambaValueAtlas,
): TwoTowerMambaAtlasVerification {
  const errors: string[] = [];
  const { width, height, data } = atlas;
  const boxes: { name: string; x0: number; y0: number; x1: number; y1: number }[] = [];

  const pushSlice = (name: string, slice: AtlasSlice | null | undefined): void => {
    if (!slice) return;
    const [u0, v0, uW, vH]: TexRect = slice.rect;
    if (u0 < -1e-6 || v0 < -1e-6 || uW <= 0 || vH <= 0 || u0 + uW > 1 + 1e-6 || v0 + vH > 1 + 1e-6) {
      errors.push(`${name}: TexRect out of [0,1] bounds`);
    }
    const x0 = Math.round(u0 * width);
    const y0 = Math.round(v0 * height);
    const x1 = Math.round((u0 + uW) * width);
    const y1 = Math.round((v0 + vH) * height);
    if (x0 < 0 || y0 < 0 || x1 > width || y1 > height || x1 <= x0 || y1 <= y0) {
      errors.push(`${name}: texel box out of bounds`);
    }
    boxes.push({ name, x0, y0, x1, y1 });
  };

  pushSlice('wIn', atlas.wIn);
  pushSlice('wOut', atlas.wOut);
  pushSlice('input', atlas.input);
  pushSlice('modulatedInput', atlas.modulatedInput);
  pushSlice('rmsnorm', atlas.rmsnorm);
  pushSlice('residual', atlas.residual);
  pushSlice('xbcPre', atlas.xbcPre);
  pushSlice('z', atlas.z);
  pushSlice('dt', atlas.dt);
  atlas.convWindowTaps.forEach((s, i) => pushSlice(`convWindowTaps[${i}]`, s));
  atlas.xHeads.forEach((s, h) => pushSlice(`xHeads[${h}]`, s));
  pushSlice('bGroups', atlas.bGroups);
  pushSlice('cGroups', atlas.cGroups);
  pushSlice('hPrev', atlas.hPrev);
  pushSlice('decayedPrev', atlas.decayedPrev);
  pushSlice('inputWrite', atlas.inputWrite);
  pushSlice('hCurr', atlas.hCurr);
  pushSlice('inheritedRetention', atlas.inheritedRetention);
  pushSlice('accumulatedBlockWrite', atlas.accumulatedBlockWrite);
  pushSlice('decayBankDa', atlas.decayBankDa);
  pushSlice('decayBankCumulative', atlas.decayBankCumulative);
  pushSlice('ssmBankSummary', atlas.ssmBankSummary);
  pushSlice('ySsm', atlas.ySsm);
  pushSlice('skipDx', atlas.skipDx);
  pushSlice('y', atlas.y);
  pushSlice('gated', atlas.gated);
  pushSlice('rmsGate', atlas.rmsGate);
  pushSlice('yNorm', atlas.yNorm);
  atlas.ctxConvCacheTaps?.forEach((s, i) => pushSlice(`ctxConvCacheTaps[${i}]`, s));
  pushSlice('ctxSsmBoundaryHead', atlas.ctxSsmBoundaryHead);
  pushSlice('ctxSsmBoundaryBank', atlas.ctxSsmBoundaryBank);
  if (atlas.secondary) {
    pushSlice('secondary.input', atlas.secondary.input);
    pushSlice('secondary.rmsnorm', atlas.secondary.rmsnorm);
    pushSlice('secondary.residual', atlas.secondary.residual);
    pushSlice('secondary.dt', atlas.secondary.dt);
    pushSlice('secondary.xbcPre', atlas.secondary.xbcPre);
    pushSlice('secondary.y', atlas.secondary.y);
    pushSlice('secondary.yNorm', atlas.secondary.yNorm);
    pushSlice('secondary.hCurr', atlas.secondary.hCurr);
  }

  for (let i = 0; i < boxes.length; i++) {
    const a = boxes[i];
    for (let j = i + 1; j < boxes.length; j++) {
      const b = boxes[j];
      if (Math.min(a.x1, b.x1) > Math.max(a.x0, b.x0) && Math.min(a.y1, b.y1) > Math.max(a.y0, b.y0)) {
        errors.push(`Overlap between ${a.name} and ${b.name}`);
      }
    }
  }

  return Object.freeze({
    valid: errors.length === 0,
    width,
    height,
    byteLength: data.byteLength,
    checkedSliceCount: boxes.length,
    errors: Object.freeze(errors),
  });
}

// ============================================================================
// OPTIONS, DIAGNOSTICS & CAMERA FITTER
// ============================================================================

export interface BuildPairedMambaOptions {
  readonly bayId?: number;
  readonly towerFocus?: TwoTowerPairedFocus;
  readonly activeHead?: number;
  readonly localTokenIndex?: number;
  readonly selectedCell?: { readonly tower: 'context' | 'denoiser'; readonly cell: MambaCellRef } | null;
  readonly hoveredCell?: { readonly tower: 'context' | 'denoiser'; readonly cell: MambaCellRef } | null;
  readonly stepperStep?: number | null;
  readonly stepProgress?: number;
  readonly termProgress?: number;
  readonly reducedMotion?: boolean;
}

export interface PairedMambaSeedChannelDiagnostics {
  readonly channelId: 'mamba_conv_channel' | 'mamba_ssm_channel';
  readonly label: string;
  readonly ownerRole: 'context_tower';
  readonly consumerRole: 'denoiser_tower';
  readonly traceOrigin: 'persistent_cache';
  readonly storageDtypeMetadata: 'bf16' | 'fp32';
  readonly shapeSummary: string;
  readonly sourceBoundaryDigest: string;
}

export interface PairedMambaDiagnostics {
  readonly bayId: number;
  readonly layerIndex: number;
  readonly towerFocus: TwoTowerPairedFocus;
  readonly modeAtRecord: string;
  readonly activeHead: number;
  readonly activeGroup: number;
  readonly localTokenIndex: number;
  readonly blockLength: number;
  readonly absoluteTokenPosition: number;
  readonly refinementIndex: number | null;
  readonly sourceBoundaryDigest: string;
  readonly contextSnapshotDigest: string;
  readonly currentContextPrefixLen: number;
  readonly currentContextStateDigest: string;
  readonly denoiserConditioningPrefixLen: number | null;
  readonly denoiserSourceRecordIndex: number | null;
  readonly denoiserSourceBlockId: number | null;
  readonly denoiserSourceRefinementIndex: number | null;
  readonly isHistoricalDenoiserInspection: boolean;
  readonly matchesCurrentContextBoundary: boolean;
  readonly historicalProvenanceWarning: string | null;
  readonly conditioningDigest: string | null;
  readonly refinementSeedInvariantNote: string;
  readonly convChannel: PairedMambaSeedChannelDiagnostics;
  readonly ssmChannel: PairedMambaSeedChannelDiagnostics;
  readonly contextTraceStatus: 'available' | 'pruned_unavailable';
  readonly contextTraceOrigin: 'context_computed' | 'context_reconstructed' | 'persistent_cache';
  readonly denoiserPassStatus: 'active_on_record' | 'historical_prior_record' | 'none_yet';
  readonly hasDenoiserPreNormMod: boolean;
  readonly hasDenoiserResidualGate: boolean;
  readonly isAfterFirstBidirectionalGqa: boolean;
  readonly causalAfterGqaNote: string;
  readonly decaySemanticsDisclaimer: string;
  readonly decomposition: MambaPositionDecomposition | null;
  readonly modelMambaInventory: {
    readonly heads: number;
    readonly groups: number;
    readonly headsPerGroup: number;
    readonly dHead: number;
    readonly dState: number;
    readonly stateMatrixShape: readonly [number, number];
    readonly displayedHeadIndex: number;
    readonly displayedGroupIndex: number;
    readonly atlasSelectedDecompHeadCount: 1;
  };
  readonly atlasDimensions: readonly [number, number];
}

/**
 * Calibrated 8-corner projective camera fitter for paired & solo TwoTower Mamba views.
 * Tightens viewing angle and margin so paired 3D sheets fill ~60–75% of the canvas cleanly.
 */
export function fitPairedMambaCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: SpatialMambaFitOptions,
): CameraPose {
  const isNarrow = options?.isNarrow ?? metrics.innerWidth < 760;
  const pose = fitSpatialMambaCamera(bounds, metrics, {
    ...options,
    azimuthDeg: options?.azimuthDeg ?? (isNarrow ? 12 : 14),
    elevationDeg: options?.elevationDeg ?? (isNarrow ? 26 : 25),
  });
  return {
    ...pose,
    distance: Math.max(MIN_CAMERA_DISTANCE, pose.distance * 0.90),
  };
}

// ============================================================================
// INTERNAL BUILDER & PALETTE HELPERS
// ============================================================================

const RIBBON_BRIDGE_CONV: FlowRibbonColor = { r: 0.78, g: 0.50, b: 0.14, a: 0.84, borderAlpha: 0.46 };
const RIBBON_BRIDGE_SSM: FlowRibbonColor = { r: 0.10, g: 0.54, b: 0.72, a: 0.84, borderAlpha: 0.46 };
const RIBBON_GATE_FLOW: FlowRibbonColor = { r: 0.16, g: 0.56, b: 0.45, a: 0.44, borderAlpha: 0.24 };
const RIBBON_CTX_FLOW: FlowRibbonColor = { r: 0.36, g: 0.48, b: 0.62, a: 0.36, borderAlpha: 0.20 };

interface PairedMambaSheetSpec {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly axes: SheetBasisAxes;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly role: number;
  readonly fp8Flag: number;
  readonly highlight: number;
  readonly seed: number;
  readonly partId: string;
  readonly towerRole: TwoTowerTowerRole;
  readonly executionId: string;
  readonly traceOrigin: TwoTowerTraceOrigin;
  readonly semanticTag?: string;
  readonly absoluteTokenOffset?: number;
  readonly headIndex?: number;
  readonly groupIndex?: number;
  readonly tensorKey?: MambaTensorKey;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

class PairedMambaBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly revealFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();
  readonly activeOpPartIds = new Set<string>();

  private readonly bayId: number;
  private readonly revealEnabled: boolean;
  private readonly collectActiveOpParts: boolean;

  constructor(bayId: number, revealEnabled: boolean, collectActiveOpParts: boolean) {
    this.bayId = bayId;
    this.revealEnabled = revealEnabled;
    this.collectActiveOpParts = collectActiveOpParts;
  }

  addSheet(s: PairedMambaSheetSpec, revealParams: MambaRevealParams | null): SheetFrame3D {
    const frame = makeSheetFrame3D(
      [s.cx, s.cy, s.cz],
      [s.ex, s.ey, s.ez],
      s.axes.axisU,
      s.axes.axisN,
    );
    const rect = s.atlasSlice ? s.atlasSlice.rect : ([0, 0, 0, 0] as const);
    const scale = s.atlasSlice ? s.atlasSlice.scale : 1.0;
    const modeFlag = s.atlasSlice ? s.atlasSlice.mode : 0;

    this.sheetFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.gridCols, s.gridRows,
      SHEET_KIND_STANDARD,
      s.role,
      s.fp8Flag,
      s.highlight,
      s.seed,
      rect[0], rect[1], rect[2], rect[3],
      scale,
      modeFlag, 0,
    );

    this.basisFloats.push(
      frame.axisU[0], frame.axisU[1], frame.axisU[2],
      frame.axisN[0], frame.axisN[1], frame.axisN[2],
    );

    if (this.revealEnabled) {
      if (revealParams && s.traceOrigin === 'denoiser_scratch') {
        const [mode, colF, rowF, pulse] = computeMambaSheetReveal(revealParams, {
          tensorKey: s.tensorKey,
          partId: s.partId,
          headIndex: s.headIndex,
        });
        this.revealFloats.push(mode, colF, rowF, pulse);
      } else {
        this.revealFloats.push(SHEET_REVEAL_NONE, 0, 0, 0);
      }
    }

    if (
      this.collectActiveOpParts &&
      (s.highlight === SHEET_HL_ACTIVE ||
        s.highlight === SHEET_HL_OPERAND_A ||
        s.highlight === SHEET_HL_OPERAND_B)
    ) {
      this.activeOpPartIds.add(s.partId);
    }

    this.solidFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.cellFamily,
      s.fp8Flag,
      s.seed,
      this.bayId,
      0.2,
      s.highlight === SHEET_HL_ACTIVE ? 1 : s.highlight === SHEET_HL_BF16_GOLD ? 2 : 0,
    );

    const pickFrame = makeSheetFrame3D(
      frame.center,
      [frame.extent[0], Math.max(frame.extent[1], 0.085), frame.extent[2]],
      frame.axisU,
      frame.axisN,
    );
    const worldAabb = sheetFrameWorldAabb(pickFrame);
    this.pickBoxes.push({
      bayId: this.bayId,
      partId: s.partId,
      headIndex: s.headIndex,
      groupIndex: s.groupIndex,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      tapIndex: s.tapIndex,
      stateTokenIndex: s.stateTokenIndex,
      towerRole: s.towerRole,
      executionId: s.executionId,
      traceOrigin: s.traceOrigin,
      semanticTag: s.semanticTag,
      absoluteTokenOffset: s.absoluteTokenOffset,
      min: worldAabb.min,
      max: worldAabb.max,
      frame: pickFrame,
    });

    if (s.registryKey) {
      this.sheetRegistry.set(s.registryKey, {
        cx: s.cx,
        cy: s.cy,
        cz: s.cz,
        ex: s.ex,
        ey: s.ey,
        ez: s.ez,
        gridCols: s.gridCols,
        gridRows: s.gridRows,
        frame,
      });
    }

    return frame;
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.55,
  ): void {
    pushFreeGuideLine(this.lineFloats, ax, ay, az, bx, by, bz, cssWidth, style, alpha);
  }

  addBoxFrameXZ(
    x0: number, z0: number, x1: number, z1: number, y: number,
    cssWidth = 1.0, style = LINE_STYLE_GUIDE, alpha = 0.45,
  ): void {
    pushBoxFrameXZ(this.lineFloats, x0, z0, x1, z1, y, cssWidth, style, alpha);
  }

  pack() {
    const flowPacked = this.flow.pack();
    const sheetCount = this.sheetFloats.length / SHEET_INSTANCE_FLOATS;
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount,
      sheetBasisData: new Float32Array(this.basisFloats),
      ...(this.revealEnabled && this.revealFloats.length === sheetCount * SHEET_REVEAL_FLOATS
        ? { sheetRevealData: new Float32Array(this.revealFloats) }
        : {}),
      solidData: new Float32Array(this.solidFloats),
      solidCount: this.solidFloats.length / SOLID_INSTANCE_FLOATS,
      lineData: new Float32Array(this.lineFloats),
      lineCount: this.lineFloats.length / LINE_INSTANCE_FLOATS,
      ribbonData: new Float32Array(0),
      ribbonCount: 0,
      flowVertices: flowPacked.vertices,
      flowIndices: flowPacked.indices,
    };
  }
}

function buildMambaFocusSet(formula: MambaCellFormula | null): SheetFocusSet {
  if (!formula) return EMPTY_SHEET_FOCUS_SET;
  const toEntry = (span: MambaSheetSpan): SheetFocusEntry => ({
    tensor: span.tensor,
    headIndex: span.headIndex,
    groupIndex: span.groupIndex,
    tapStart: span.tapStart,
    tapEnd: span.tapEnd,
    rowStart: span.rowStart,
    rowEnd: span.rowEnd,
    colStart: span.colStart,
    colEnd: span.colEnd,
    role: span.role,
  });
  return createSheetFocusSet([
    toEntry(formula.targetSpan),
    ...formula.upstreamSpans.map(toEntry),
  ]);
}

function resolvePairedMambaSpans(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  prefix: 'ctx' | 'den',
  span: MambaSheetSpan,
  displayedHead: number,
): readonly WorldSpan3D[] {
  if (span.tensor === 'conv_buf') {
    const t0 = Math.max(0, Math.min(3, span.tapStart ?? 0));
    const t1 = Math.max(t0, Math.min(3, span.tapEnd ?? t0));
    const res: WorldSpan3D[] = [];
    for (let slot = t0; slot <= t1; slot++) {
      const sheet = registry.get(`${prefix}:conv_buf:${slot}`);
      if (sheet) res.push(spanFromSheetGrid(sheet, 0, 0, span.colStart, span.colEnd));
    }
    return res;
  }
  const key =
    span.tensor === 'x'
      ? `${prefix}:x:${span.headIndex ?? displayedHead}`
      : `${prefix}:${span.tensor}`;
  const sheet = registry.get(key);
  if (!sheet) return [];
  return [spanFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd)];
}

function gridFromSheet4x4(
  rec: SheetGeomRecord | undefined,
  values: Float32Array,
  colDimLabel: string,
  rowDimLabel: string,
  activeCol?: number,
  activeRow?: number,
  activeValueOverride?: number | null,
): ActiveSheetCellGrid | null {
  if (!rec) return null;
  const frame =
    rec.frame ?? makeSheetFrame3D([rec.cx, rec.cy, rec.cz], [rec.ex, rec.ey, rec.ez]);
  return {
    originWorld: frame.center,
    extent: frame.extent,
    axisU: frame.axisU,
    axisN: frame.axisN,
    cols: 4,
    rows: 4,
    values: new Float32Array(values),
    format: 'signed',
    activeCol,
    activeRow,
    ...(activeValueOverride !== undefined && activeValueOverride !== null
      ? { activeValueOverride }
      : {}),
    colLabels: ['n0', 'n1', 'n2', 'n3'],
    rowLabels: ['p0', 'p1', 'p2', 'p3'],
    colDimLabel,
    rowDimLabel,
  };
}

// ============================================================================
// MAIN PAIRED 35/65 MAMBA SCENE BUILDER
// ============================================================================

export function buildTwoTowerPairedMambaScene(
  session: TwoTowerSession,
  options?: BuildPairedMambaOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const bayId = resolveNearestBayOfKind(options?.bayId ?? 1, 'M');
  const layerIndex = bayId - 1;
  const towerFocus: TwoTowerPairedFocus = options?.towerFocus ?? 'paired';

  const ctxLookup = view.resolveContextBayTrace(bayId);
  const contextSim: MambaSimResult | null =
    ctxLookup.status === 'available' && ctxLookup.trace?.kind === 'M' ? ctxLookup.trace.sim : null;

  const denoiserTrace = view.resolveDenoiserBayTrace(bayId);
  const denoiserSim: MambaSimResult | null =
    denoiserTrace?.kind === 'M' ? denoiserTrace.sim : null;

  // Authoritative Context boundary state: strictly distinguish the conditioning seed read by
  // the inspected Denoiser pass (`denoiserSeedBoundaryState`, e.g. H_7 at P_cond=8) from the
  // current record's persistent Context boundary (`currentContextBoundaryState`, e.g. H_11 at P=12).
  const dl = view.denoiserLookup;
  const isHistoricalDenoiser = dl.isHistoricalInspection;
  const matchesCurrentBoundary = dl.matchesCurrentContextBoundary;
  const conditioning = dl.conditioning;
  const currentContextBoundaryState: MambaLayerBoundaryState | null =
    view.contextSnapshot.ctxState.mambaStates.get(bayId) ?? null;
  const denoiserSeedBoundaryState: MambaLayerBoundaryState | null =
    conditioning?.mambaSeedStates.get(bayId) ?? null;
  const contextBoundaryState: MambaLayerBoundaryState | null =
    denoiserSeedBoundaryState ?? currentContextBoundaryState;

  const currentContextPrefixLen = view.committedPrefixLen;
  const currentContextStateDigest = view.contextSnapshot.ctxState.stateDigest;
  const denoiserConditioningPrefixLen = dl.sourceContextPrefixLen;
  const seedPrefixLen =
    denoiserSeedBoundaryState?.processedTokens ??
    denoiserConditioningPrefixLen ??
    currentContextPrefixLen;
  const seedLastPos = Math.max(0, seedPrefixLen - 1);
  const historicalBoundaryMismatch = isHistoricalDenoiser && !matchesCurrentBoundary;
  const sourceBoundaryDigest =
    dl.sourceContextCacheDigest ?? currentContextStateDigest;
  const historicalTag =
    isHistoricalDenoiser && dl.pass !== null
      ? `Historical Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, ${dl.pass.passExecutionId}, P_cond=${seedPrefixLen})`
      : null;

  const requestedCell = options?.hoveredCell ?? options?.selectedCell ?? null;
  let activeTower: 'context' | 'denoiser' =
    requestedCell?.tower ??
    (towerFocus === 'context_solo'
      ? 'context'
      : denoiserSim !== null
        ? 'denoiser'
        : 'context');
  if (activeTower === 'denoiser' && denoiserSim === null && contextSim !== null) {
    activeTower = 'context';
  }
  if (activeTower === 'context' && contextSim === null && denoiserSim !== null) {
    activeTower = 'denoiser';
  }

  const primarySim: MambaSimResult | null = denoiserSim ?? contextSim;
  const activeSim: MambaSimResult | null = activeTower === 'denoiser' ? denoiserSim : contextSim;

  // Resolve active head & local token position strictly from complete executed block (never rerunning)
  const inferredHead = requestedCell ? inferMambaCellHead(requestedCell.cell) : null;
  const activeHead = Math.max(0, Math.min(15, Math.round(options?.activeHead ?? inferredHead ?? 0)));
  const activeGroup = mambaGroupForHead(activeHead);

  const primaryTokens = primarySim?.tokens ?? 1;
  const inferredTok =
    requestedCell && primarySim ? inferMambaCellToken(requestedCell.cell, primaryTokens) : null;
  const localPos = Math.max(
    0,
    Math.min(
      primaryTokens - 1,
      Math.round(options?.localTokenIndex ?? inferredTok ?? Math.max(0, primaryTokens - 1)),
    ),
  );

  // Full-block analytical decomposition strictly from executed primarySim (never reruns runMambaSim!)
  const decomposition: MambaPositionDecomposition | null =
    primarySim !== null
      ? extractMambaFullBlockPositionState(primarySim, activeHead, localPos, { bayId })
      : null;

  // Build dedicated 256×128 live RG32F Mamba atlas
  let mambaAtlas: TwoTowerMambaValueAtlas | null = null;
  let fallbackBoundaryAtlas: Rg32fValueAtlas | null = null;
  if (primarySim !== null && decomposition !== null) {
    mambaAtlas = buildTwoTowerMambaValueAtlas({
      primarySim,
      decomposition,
      contextBoundaryState,
      secondaryContextSim: denoiserSim !== null ? contextSim : null,
    });
  } else if (contextBoundaryState) {
    // Pruned Context-only fail-safe: pack ONLY persistent convState & ssmState
    const writer = createRg32fAtlas(TWOTOWER_MAMBA_ATLAS_WIDTH, TWOTOWER_MAMBA_ATLAS_HEIGHT);
    for (let s = 0; s < 4; s++) {
      for (let c = 0; c < contextBoundaryState.convDim; c++) {
        writer.setTexel(64 + c, 12 + s, contextBoundaryState.convState[c * 4 + s]);
      }
    }
    for (let p = 0; p < 4; p++) {
      for (let n = 0; n < 4; n++) {
        writer.setTexel(208 + n, 4 + p, contextBoundaryState.ssmState[(activeHead * 4 + p) * 4 + n]);
      }
    }
    fallbackBoundaryAtlas = {
      width: TWOTOWER_MAMBA_ATLAS_WIDTH,
      height: TWOTOWER_MAMBA_ATLAS_HEIGHT,
      data: writer.data,
    };
  }

  // Stepper & formula resolution
  const stepperStep =
    options?.stepperStep !== undefined && options.stepperStep !== null
      ? Math.max(0, Math.min(5, Math.round(options.stepperStep)))
      : null;
  const stepProgress = options?.stepProgress ?? 0.25;
  const termProgress = options?.termProgress;
  const reducedMotion = options?.reducedMotion ?? false;
  const revealEnabled = stepperStep !== null && !reducedMotion;

  let activeCell: MambaCellRef | null = null;
  if (requestedCell && requestedCell.tower === activeTower && activeSim !== null) {
    activeCell = requestedCell.cell;
  } else if (stepperStep !== null && activeSim !== null) {
    const baseCell = mambaStepperCellForState(activeSim, activeHead, stepperStep, stepProgress);
    activeCell =
      baseCell.tensor === 'h_curr' || baseCell.tensor === 'h_prev' || baseCell.tensor === 'conv_buf'
        ? { ...baseCell, tokenIndex: Math.min(activeSim.tokens - 1, localPos) }
        : baseCell;
  }

  const rawFormula: MambaCellFormula | null =
    activeSim !== null && activeCell !== null ? describeMambaCellFormula(activeSim, activeCell) : null;

  const activeFormula: MambaCellFormula | null =
    rawFormula !== null
      ? {
          ...rawFormula,
          tensorLabel: `${activeTower === 'denoiser' ? 'Denoiser' : 'Context'} · ${rawFormula.tensorLabel}`,
          subEquation:
            decomposition && activeTower === 'denoiser'
              ? `${rawFormula.subEquation} · Π dA=${decomposition.cumulativeDecay.toFixed(4)} · ` +
                `Conv1D(${decomposition.convWindow.inheritedHistoryColumnCount} ctx + ${decomposition.convWindow.blockColumnCount} blk)`
              : `${activeTower === 'denoiser' ? 'Denoiser' : 'Context'} Tower (Bay ${String(bayId).padStart(2, '0')}) · ${rawFormula.subEquation}`,
        }
      : null;

  const focusedAccumulation: FocusedAccumulationResult | null =
    activeSim !== null &&
    activeCell !== null &&
    stepperStep !== null &&
    !reducedMotion &&
    termProgress !== undefined
      ? computeMambaFocusedAccumulation(activeSim, activeCell, stepperStep, termProgress)
      : null;

  const focusSet = buildMambaFocusSet(activeFormula);
  const denoiserRevealParams: MambaRevealParams | null =
    revealEnabled && activeTower === 'denoiser' && denoiserSim !== null
      ? {
          stepperStep: stepperStep!,
          subProgress: stepProgress,
          clampedT: denoiserSim.tokens,
          activeHead,
          activeGroup,
          stateTokIdx: localPos,
          activeCell,
          sim: denoiserSim,
        }
      : null;

  const b = new PairedMambaBuilder(bayId, revealEnabled, focusSet.active);
  const SHEET_THICK = 0.056;
  const BASE_Y = 0.04;
  const PEDESTAL_Y = 1.08;

  const ctxExecId =
    ctxLookup.segment?.stepExecutionId ?? view.contextSnapshot.ctxState.stateDigest;
  const denExecId =
    view.denoiserLookup.pass?.passExecutionId ?? view.identity.recordId;

  const showContextWing = towerFocus !== 'denoiser_solo';
  const showDenoiserWing = towerFocus !== 'context_solo';

  const ctxXMin = towerFocus === 'context_solo' ? -8.8 : -13.5;
  const ctxXMax = towerFocus === 'context_solo' ? 8.8 : -3.5;
  const ctxX = (u: number): number => ctxXMin + u * (ctxXMax - ctxXMin);
  const ctxSpan = ctxXMax - ctxXMin;

  const denXMin = towerFocus === 'denoiser_solo' ? -9.2 : -1.8;
  const denXMax = towerFocus === 'denoiser_solo' ? 9.2 : 13.5;
  const denX = (u: number): number => denXMin + u * (denXMax - denXMin);
  const denSpan = denXMax - denXMin;

  let ctxConvAnchor: [number, number, number] = [-3.55, 0.54, 1.35];
  let ctxSsmAnchor: [number, number, number] = [-3.55, PEDESTAL_Y + 0.08, -1.25];
  let denConvAnchor: [number, number, number] = [-1.65, 0.54, 1.35];
  let denSsmAnchor: [number, number, number] = [-1.65, PEDESTAL_Y + 0.08, -1.25];

  const isAfterFirstBidirectionalGqa = bayId > FIRST_BIDIRECTIONAL_GQA_BAY_ID;
  const refinementSeedInvariantNote =
    'Every refinement r seeds from the SAME committed Context boundary (never prior Denoiser scratch).';

  // ==========================================================================
  // 1. LEFT WING: CONTEXT TOWER PROVENANCE (2 INDEPENDENT PERSISTENT CHANNELS)
  // ==========================================================================
  if (showContextWing) {
    const cScaleX = ctxSpan / 10.0;
    const ctxAbsOffset = ctxLookup.segment?.tokenRange[0] ?? 0;
    b.addBoxFrameXZ(ctxXMin, -3.65, ctxXMax, 3.65, BASE_Y - 0.03, 1.0, LINE_STYLE_GUIDE, 0.42);

    b.landmarks.push({
      id: 'ctx-mamba-header',
      label: historicalBoundaryMismatch
        ? `Context Wing · Current P=${currentContextPrefixLen} vs Historical Seed P_cond=${seedPrefixLen}`
        : 'Context Tower Provenance (2 Persistent Channels)',
      sublabel:
        historicalBoundaryMismatch
          ? `Right wing shows ${historicalTag}; boundary H_${seedLastPos} below is that prior seed (NOT current Context H_${Math.max(0, currentContextPrefixLen - 1)})`
          : ctxLookup.status === 'available'
            ? `origin=${ctxLookup.traceOrigin} · Channel 1: 4-col Conv1D + Channel 2: 16 heads × [4×4] FP32 SSM (8 groups)`
            : 'Historical trace pruned · Persistent Conv1D & FP32 SSM boundary snapshots remain authoritative',
      world: [ctxX(0.5), 2.85, -3.65],
      bayId,
      partId: 'ctx-ssm-state',
      tone: historicalBoundaryMismatch ? 'gold' : ctxLookup.status === 'available' ? 'cyan' : 'gold',
      align: 'center',
      tier: 'primary',
    });

    // Channel 1: 4-Column Conv1D Raw History Cache (slots 0..3, BF16 prod metadata) at +Z front
    const xConvCtx = ctxX(0.76);
    const zConvCtx = 1.45;
    for (let s = 0; s < 4; s++) {
      b.addSheet(
        {
          cx: xConvCtx + (s - 1.5) * 0.24 * cScaleX,
          cy: 0.34 + s * 0.09,
          cz: zConvCtx,
          ex: 0.20 * cScaleX,
          ey: SHEET_THICK * 0.85,
          ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: 128,
          gridRows: 1,
          role: SHEET_ROLE_KV_K,
          fp8Flag: 0,
          highlight: SHEET_HL_BF16_GOLD,
          seed: 101 + s,
          partId: 'ctx-conv-state',
          towerRole: 'context',
          executionId: ctxExecId,
          traceOrigin: 'persistent_cache',
          semanticTag: 'mamba_conv_channel',
          absoluteTokenOffset: Math.max(0, (contextBoundaryState?.processedTokens ?? 0) - (4 - s)),
          tapIndex: s,
          tensorKey: 'conv_buf',
          atlasSlice: mambaAtlas?.ctxConvCacheTaps?.[s],
          registryKey: `ctx:conv_buf:${s}`,
          cellFamily: CELL_MOE_AMBER,
        },
        null,
      );
    }
    ctxConvAnchor = [xConvCtx + 0.52 * cScaleX, 0.56, zConvCtx];

    // Channel 2: 16 × [4×4] FP32 SSM Boundary State H_{P-1} at -Z rear (cleanly separated from Conv1D!)
    const xSsmCtx = ctxX(0.76);
    const zSsmCtx = -1.35;
    b.addSheet(
      {
        cx: xSsmCtx, cy: PEDESTAL_Y, cz: zSsmCtx,
        ex: 1.10 * cScaleX, ey: SHEET_THICK, ez: 1.10,
        axes: FRAME_XZ_AXES,
        gridCols: 4, gridRows: 4,
        role: SHEET_ROLE_SCORES, fp8Flag: 0,
        highlight: SHEET_HL_BF16_GOLD,
        seed: 108, partId: 'ctx-ssm-state',
        towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
        semanticTag: 'mamba_ssm_channel',
        headIndex: activeHead, groupIndex: activeGroup,
        tensorKey: 'h_prev',
        atlasSlice: mambaAtlas?.ctxSsmBoundaryHead ?? undefined,
        registryKey: 'ctx:h_prev',
        cellFamily: CELL_SSM_CYAN,
      },
      null,
    );
    b.addSheet(
      {
        cx: xSsmCtx, cy: PEDESTAL_Y + 0.56, cz: zSsmCtx - 0.85,
        ex: 1.35 * cScaleX, ey: SHEET_THICK * 0.85, ez: 0.42,
        axes: FRAME_XZ_AXES,
        gridCols: 16, gridRows: 4,
        role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
        highlight: SHEET_HL_NORMAL,
        seed: 109, partId: 'ctx-ssm-bank',
        towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
        semanticTag: 'mamba_ssm_channel',
        atlasSlice: mambaAtlas?.ctxSsmBoundaryBank ?? undefined,
        cellFamily: CELL_SSM_CYAN,
      },
      null,
    );
    ctxSsmAnchor = [xSsmCtx + 0.62 * cScaleX, PEDESTAL_Y + 0.04, zSsmCtx];

    b.landmarks.push(
      {
        id: 'ctx-conv-channel-landmark',
        label: historicalBoundaryMismatch
          ? `Channel 1 · Historical Conv1D Seed (P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen})`
          : 'Channel 1 · Conv1D 4-Col History',
        sublabel: historicalBoundaryMismatch
          ? `${historicalTag} · NOT current Context P=${currentContextPrefixLen} cache · slots [0..3]`
          : 'persistent_cache · BF16 (prod) · slots [0..3] (lags 3,2,1 enter pos 0)',
        world: [xConvCtx, 1.05, zConvCtx + 0.95],
        bayId,
        partId: 'ctx-conv-state',
        tone: 'gold',
        align: 'center',
        tier: 'primary',
      },
      {
        id: 'ctx-ssm-channel-landmark',
        label: historicalBoundaryMismatch
          ? `Channel 2 · Historical FP32 SSM H_${seedLastPos} (P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen}, ${formatMambaHeadId(activeHead)} of 16h)`
          : `Channel 2 · FP32 SSM H_${seedLastPos} (${formatMambaHeadId(activeHead)} slice of 16h / 8g Bank)`,
        sublabel: historicalBoundaryMismatch
          ? `${historicalTag} · Model: 16 heads / 8 groups / 4×4 SSM (1 selected head slice; NOT current H_${Math.max(0, currentContextPrefixLen - 1)})`
          : 'persistent_cache · FP32 (prod) · Model: 16 Mamba heads / 8 groups / 4×4 SSM (64 × [64×128] prod)',
        world: [xSsmCtx, PEDESTAL_Y + 1.05, zSsmCtx - 0.95],
        bayId,
        partId: 'ctx-ssm-state',
        tone: historicalBoundaryMismatch ? 'gold' : 'cyan',
        align: 'center',
        tier: 'primary',
      },
    );

    // Context Segment Execution Trace (or Pruned Notice) on left half of Context wing
    if (contextSim !== null) {
      const cT = contextSim.tokens;
      const usePrimaryForCtx = denoiserSim === null && mambaAtlas !== null;
      const sec = mambaAtlas?.secondary ?? null;

      b.addSheet(
        {
          cx: ctxX(0.14), cy: BASE_Y, cz: 0,
          ex: 0.78 * cScaleX, ey: SHEET_THICK, ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: contextSim.d, gridRows: cT,
          role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 111, partId: 'ctx-input',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
          semanticTag: 'context_mamba_input', absoluteTokenOffset: ctxAbsOffset,
          tensorKey: 'input',
          atlasSlice: usePrimaryForCtx ? mambaAtlas!.input : sec?.input,
          registryKey: 'ctx:input',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        null,
      );
      b.addSheet(
        {
          cx: ctxX(0.30), cy: BASE_Y, cz: 0,
          ex: 0.78 * cScaleX, ey: SHEET_THICK, ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: contextSim.d, gridRows: cT,
          role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 112, partId: 'ctx-rmsnorm',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
          semanticTag: 'context_mamba_rmsnorm', absoluteTokenOffset: ctxAbsOffset,
          tensorKey: 'rmsnorm',
          atlasSlice: usePrimaryForCtx ? mambaAtlas!.rmsnorm : sec?.rmsnorm,
          registryKey: 'ctx:rmsnorm',
          cellFamily: CELL_MOE_AMBER,
        },
        null,
      );
      b.addSheet(
        {
          cx: ctxX(0.46), cy: BASE_Y, cz: 0,
          ex: 0.86 * cScaleX, ey: SHEET_THICK, ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: contextSim.dInner, gridRows: cT,
          role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 113, partId: 'ctx-y-norm',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
          semanticTag: 'context_mamba_y_norm', absoluteTokenOffset: ctxAbsOffset,
          tensorKey: 'y_norm',
          atlasSlice: usePrimaryForCtx ? mambaAtlas!.yNorm : sec?.yNorm,
          registryKey: 'ctx:y_norm',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        null,
      );
      b.flow.addPolyline3D(
        [
          [ctxX(0.18), BASE_Y + 0.05, 0],
          [ctxX(0.42), BASE_Y + 0.05, 0],
        ],
        [0, 1, 0], 0.07, RIBBON_CTX_FLOW, true, 0.14, 0.14, false,
      );
    } else {
      b.addSheet(
        {
          cx: ctxX(0.28), cy: BASE_Y + 0.16, cz: 0,
          ex: 2.8 * cScaleX, ey: SHEET_THICK, ez: 1.9,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 2,
          role: SHEET_ROLE_NEUTRAL, fp8Flag: 0,
          highlight: SHEET_HL_BF16_GOLD,
          seed: 115, partId: 'ctx-pruned-notice',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
          semanticTag: 'context_trace_pruned',
          cellFamily: CELL_NEUTRAL,
        },
        null,
      );
    }
  }

  // ==========================================================================
  // 2. RIGHT WING: ACTIVE DENOISER MAMBA COMPUTATION & ANALYTICAL DECOMPOSITION
  // ==========================================================================
  if (showDenoiserWing) {
    const dScaleX = denSpan / 15.3;
    b.addBoxFrameXZ(denXMin, -3.75, denXMax, 3.75, BASE_Y - 0.03, 1.0, LINE_STYLE_GUIDE, 0.48);

    if (denoiserSim !== null && mambaAtlas !== null && decomposition !== null) {
      const T = denoiserSim.tokens;
      const P = denoiserSim.startTokenOffset;
      const denFocus = activeTower === 'denoiser' ? focusSet : EMPTY_SHEET_FOCUS_SET;

      const xIn = denX(0.04);
      const xMod = denX(0.10);
      const xRms = denX(0.16);
      const xWin = denX(0.23);
      const xXbc = denX(0.31);
      const xConv = denX(0.40);
      const xXHeads = denX(0.50);
      const xHPrev = denX(0.59);
      const xHCurr = denX(0.68);
      const xDecomp = denX(0.76);
      const xReadout = denX(0.84);
      const xNorm = denX(0.90);
      const xWout = denX(0.94);
      const xOut = denX(0.975);

      const hlDen = (tensor: string, fallback: number): number =>
        resolveSheetHighlight(denFocus, { tensor }, fallback);

      // 2a. Input X -> Pre-RMSNorm AdaLN X_mod -> RMSNorm U -> Upright W_in [208×8]
      b.addSheet(
        {
          cx: xIn, cy: BASE_Y, cz: 0,
          ex: 0.68 * dScaleX, ey: SHEET_THICK, ez: 1.45,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.d, gridRows: T,
          role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: hlDen('input', SHEET_HL_NORMAL),
          seed: 201, partId: 'den-input',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_input', absoluteTokenOffset: P,
          tensorKey: 'input', atlasSlice: mambaAtlas.input, registryKey: 'den:input',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      if (denoiserSim.X_mod && mambaAtlas.modulatedInput) {
        b.addSheet(
          {
            cx: xMod, cy: BASE_Y + 0.08, cz: 0,
            ex: 0.62 * dScaleX, ey: SHEET_THICK, ez: 1.45,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.d, gridRows: T,
            role: SHEET_ROLE_NORM, fp8Flag: 0,
            highlight: SHEET_HL_BF16_GOLD,
            seed: 202, partId: 'den-adaln-pre-norm',
            towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
            semanticTag: 'denoiser_pre_norm_adaln', absoluteTokenOffset: P,
            atlasSlice: mambaAtlas.modulatedInput, registryKey: 'den:x_mod',
            cellFamily: CELL_MOE_AMBER,
          },
          denoiserRevealParams,
        );
      }

      b.addSheet(
        {
          cx: xRms, cy: BASE_Y, cz: 0,
          ex: 0.68 * dScaleX, ey: SHEET_THICK, ez: 1.45,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.d, gridRows: T,
          role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: hlDen('rmsnorm', SHEET_HL_NORMAL),
          seed: 203, partId: 'den-rmsnorm',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_rmsnorm', absoluteTokenOffset: P,
          tensorKey: 'rmsnorm', atlasSlice: mambaAtlas.rmsnorm, registryKey: 'den:rmsnorm',
          cellFamily: CELL_MOE_AMBER,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xWin, cy: 1.72, cz: 0,
          ex: 4.80, ey: SHEET_THICK, ez: 0.85,
          axes: FRAME_YZ_AXES,
          gridCols: denoiserSim.inProjDim, gridRows: denoiserSim.d,
          role: SHEET_ROLE_WEIGHT, fp8Flag: 0,
          highlight: hlDen('w_in', SHEET_HL_NORMAL),
          seed: 204, partId: 'den-w-in',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_w_in',
          tensorKey: 'w_in', atlasSlice: mambaAtlas.wIn, registryKey: 'den:w_in',
          cellFamily: CELL_WEIGHT_BLUE,
        },
        denoiserRevealParams,
      );

      // 2b. Branches: Gate z (+Z front), xBC_pre (center), Δt (-Z rear)
      b.addSheet(
        {
          cx: xXbc, cy: BASE_Y, cz: 2.55,
          ex: 0.92 * dScaleX, ey: SHEET_THICK, ez: 1.15,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.dInner, gridRows: T,
          role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: hlDen('z', SHEET_HL_NORMAL),
          seed: 205, partId: 'den-z-gate',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'mamba_internal_silu_gate', absoluteTokenOffset: P,
          tensorKey: 'z', atlasSlice: mambaAtlas.z, registryKey: 'den:z',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xXbc, cy: BASE_Y, cz: 0.25,
          ex: 0.85 * dScaleX, ey: SHEET_THICK, ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.convDim, gridRows: T,
          role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: hlDen('xbc_pre', SHEET_HL_NORMAL),
          seed: 206, partId: 'den-xbc-pre',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_xbc_pre', absoluteTokenOffset: P,
          tensorKey: 'xbc_pre', atlasSlice: mambaAtlas.xbcPre, registryKey: 'den:xbc_pre',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xXbc, cy: BASE_Y, cz: -2.45,
          ex: 0.85 * dScaleX, ey: SHEET_THICK, ez: 0.95,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.nHeads, gridRows: T,
          role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: hlDen('dt', SHEET_HL_NORMAL),
          seed: 207, partId: 'den-dt',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_dt', absoluteTokenOffset: P,
          tensorKey: 'dt', atlasSlice: mambaAtlas.dt, registryKey: 'den:dt',
          cellFamily: CELL_MOE_AMBER,
        },
        denoiserRevealParams,
      );

      // 2c. 4-Tap Conv1D Window showing active-block column replacement (`3→2→1→0` inherited columns!)
      for (let s = 0; s < 4; s++) {
        const tapProv = decomposition.convWindow.taps[s];
        const isInherited = tapProv.source === 'inherited_context_history';
        b.addSheet(
          {
            cx: xConv + (s - 1.5) * 0.25 * dScaleX,
            cy: 0.36 + s * 0.10,
            cz: 1.05,
            ex: 0.21 * dScaleX,
            ey: SHEET_THICK * 0.85,
            ez: 1.48,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.convDim,
            gridRows: 1,
            role: isInherited ? SHEET_ROLE_KV_K : SHEET_ROLE_KV_V,
            fp8Flag: 0,
            highlight: resolveSheetHighlight(
              denFocus,
              { tensor: 'conv_buf', tapIndex: s, isAggregate: false },
              isInherited ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
            ),
            seed: 210 + s,
            partId: 'den-conv1d-window',
            towerRole: 'denoiser',
            executionId: denExecId,
            traceOrigin: isInherited ? 'persistent_cache' : 'denoiser_scratch',
            semanticTag: isInherited ? 'conv_tap_inherited_history' : 'conv_tap_active_block',
            absoluteTokenOffset: tapProv.absoluteTokenPosition ?? undefined,
            tapIndex: s,
            stateTokenIndex: localPos,
            tensorKey: 'conv_buf',
            atlasSlice: mambaAtlas.convWindowTaps[s],
            registryKey: `den:conv_buf:${s}`,
            cellFamily: isInherited ? CELL_MOE_AMBER : CELL_SSM_CYAN,
          },
          denoiserRevealParams,
        );
      }
      denConvAnchor = [xConv - 0.55 * dScaleX, 0.54, 1.05];

      // 2d. 16-Head x Fin Stack + 8-Group B & C
      for (let h = 0; h < 16; h++) {
        const isLifted = h === activeHead;
        b.addSheet(
          {
            cx: xXHeads,
            cy: isLifted ? 1.15 : 0.45,
            cz: -1.15 + (h / 15) * 2.15,
            ex: 0.62 * dScaleX,
            ey: SHEET_THICK * 0.80,
            ez: 0.11,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.dHead,
            gridRows: T,
            role: SHEET_ROLE_ACTIVATION,
            fp8Flag: 0,
            highlight: resolveSheetHighlight(
              denFocus,
              { tensor: 'x', headIndex: h, isAggregate: false },
              isLifted ? SHEET_HL_PATHWAY : SHEET_HL_NORMAL,
            ),
            seed: 220 + h * 0.2,
            partId: `den-x-${h}`,
            towerRole: 'denoiser',
            executionId: denExecId,
            traceOrigin: 'denoiser_scratch',
            semanticTag: 'denoiser_mamba_x_head',
            absoluteTokenOffset: P,
            headIndex: h,
            groupIndex: mambaGroupForHead(h),
            tensorKey: 'x',
            atlasSlice: mambaAtlas.xHeads[h],
            registryKey: `den:x:${h}`,
            cellFamily: CELL_SSM_CYAN,
          },
          denoiserRevealParams,
        );
      }

      b.addSheet(
        {
          cx: xXHeads, cy: BASE_Y, cz: -2.05,
          ex: 0.72 * dScaleX, ey: SHEET_THICK, ez: 0.72,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.nGroups * denoiserSim.dState, gridRows: T,
          role: SHEET_ROLE_KV_K, fp8Flag: 0,
          highlight: hlDen('b', SHEET_HL_NORMAL),
          seed: 240, partId: 'den-b-proj',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_b', absoluteTokenOffset: P,
          tensorKey: 'b', atlasSlice: mambaAtlas.bGroups, registryKey: 'den:b',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xHCurr, cy: BASE_Y, cz: -2.05,
          ex: 0.72 * dScaleX, ey: SHEET_THICK, ez: 0.72,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.nGroups * denoiserSim.dState, gridRows: T,
          role: SHEET_ROLE_KV_V, fp8Flag: 0,
          highlight: hlDen('c', SHEET_HL_NORMAL),
          seed: 241, partId: 'den-c-proj',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_c', absoluteTokenOffset: P,
          tensorKey: 'c', atlasSlice: mambaAtlas.cGroups, registryKey: 'den:c',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealParams,
      );

      // 2e. Raised Fixed FP32 SSM Pedestal: Native [4×4] H_prev, dA·H_prev, Δt·x·B, H_curr,
      //     Inherited Retention, Accumulated Block Write, and 16-Head Heterogeneous Decay Bank!
      const ssmSize = 0.88 * dScaleX;
      b.addSheet(
        {
          cx: xHPrev, cy: PEDESTAL_Y, cz: -0.65,
          ex: ssmSize, ey: SHEET_THICK, ez: 0.88,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 4,
          role: SHEET_ROLE_SCORES, fp8Flag: 0,
          highlight: resolveSheetHighlight(
            denFocus,
            { tensor: 'h_prev', headIndex: activeHead, isAggregate: false },
            localPos === 0 ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
          ),
          seed: 250, partId: 'den-h-prev',
          towerRole: 'denoiser', executionId: denExecId,
          traceOrigin: localPos === 0 ? 'persistent_cache' : 'denoiser_scratch',
          semanticTag: localPos === 0 ? 'ssm_inherited_boundary_h_prev' : 'ssm_block_h_prev',
          absoluteTokenOffset: Math.max(0, P + localPos - 1),
          headIndex: activeHead, groupIndex: activeGroup, tokenIndex: localPos,
          tensorKey: 'h_prev', atlasSlice: mambaAtlas.hPrev, registryKey: 'den:h_prev',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealParams,
      );
      denSsmAnchor = [xHPrev - ssmSize * 0.55, PEDESTAL_Y + 0.04, -0.65];

      // Intermediate decay & write plates below/between H_prev and H_curr
      b.addSheet(
        {
          cx: (xHPrev + xHCurr) * 0.5, cy: PEDESTAL_Y + 0.36, cz: -1.15,
          ex: 0.62 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.62,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 4,
          role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 251, partId: 'den-decayed-prev',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'ssm_decayed_prev', absoluteTokenOffset: P + localPos,
          headIndex: activeHead, groupIndex: activeGroup, tokenIndex: localPos,
          atlasSlice: mambaAtlas.decayedPrev, registryKey: 'den:decayed_prev',
          cellFamily: CELL_MOE_AMBER,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: (xHPrev + xHCurr) * 0.5, cy: PEDESTAL_Y + 0.36, cz: -0.15,
          ex: 0.62 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.62,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 4,
          role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 252, partId: 'den-input-write',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'ssm_input_write', absoluteTokenOffset: P + localPos,
          headIndex: activeHead, groupIndex: activeGroup, tokenIndex: localPos,
          atlasSlice: mambaAtlas.inputWrite, registryKey: 'den:input_write',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xHCurr, cy: PEDESTAL_Y, cz: -0.65,
          ex: ssmSize, ey: SHEET_THICK, ez: 0.88,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 4,
          role: SHEET_ROLE_SCORES, fp8Flag: 0,
          highlight: resolveSheetHighlight(
            denFocus,
            { tensor: 'h_curr', headIndex: activeHead, isAggregate: false },
            SHEET_HL_PATHWAY,
          ),
          seed: 253, partId: 'den-h-curr',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'ssm_curr_state', absoluteTokenOffset: P + localPos,
          headIndex: activeHead, groupIndex: activeGroup, tokenIndex: localPos,
          tensorKey: 'h_curr', atlasSlice: mambaAtlas.hCurr, registryKey: 'den:h_curr',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealParams,
      );

      // Analytical Decomposition Twin Plates: Inherited Retention vs Accumulated Block Write
      b.addSheet(
        {
          cx: xDecomp, cy: PEDESTAL_Y + 0.52, cz: -1.15,
          ex: 0.78 * dScaleX, ey: SHEET_THICK, ez: 0.78,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 4,
          role: SHEET_ROLE_KV_K, fp8Flag: 0,
          highlight: SHEET_HL_BF16_GOLD,
          seed: 254, partId: 'den-inherited-retention',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'persistent_cache',
          semanticTag: 'ssm_analytical_inherited_retention', absoluteTokenOffset: P + localPos,
          headIndex: activeHead, groupIndex: activeGroup, tokenIndex: localPos,
          atlasSlice: mambaAtlas.inheritedRetention, registryKey: 'den:inherited_retention',
          cellFamily: CELL_MOE_AMBER,
        },
        null,
      );

      b.addSheet(
        {
          cx: xDecomp, cy: PEDESTAL_Y + 0.52, cz: 0.05,
          ex: 0.78 * dScaleX, ey: SHEET_THICK, ez: 0.78,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 4,
          role: SHEET_ROLE_SOFTMAX, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 255, partId: 'den-block-write',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'ssm_analytical_block_write', absoluteTokenOffset: P + localPos,
          headIndex: activeHead, groupIndex: activeGroup, tokenIndex: localPos,
          atlasSlice: mambaAtlas.accumulatedBlockWrite, registryKey: 'den:block_write',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      // 16-Head Heterogeneous Decay Bank (dA and cumulative Π dA across H00..H15)
      b.addSheet(
        {
          cx: (xHPrev + xDecomp) * 0.5, cy: PEDESTAL_Y + 1.18, cz: -2.45,
          ex: 2.35 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.34,
          axes: FRAME_XZ_AXES,
          gridCols: 16, gridRows: 1,
          role: SHEET_ROLE_SOFTMAX, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 256, partId: 'den-decay-bank-da',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'mamba_16head_decay_bank', absoluteTokenOffset: P + localPos,
          atlasSlice: mambaAtlas.decayBankDa, registryKey: 'den:decay_bank_da',
          cellFamily: CELL_MOE_AMBER,
        },
        null,
      );

      b.addSheet(
        {
          cx: (xHPrev + xDecomp) * 0.5, cy: PEDESTAL_Y + 1.18, cz: -2.92,
          ex: 2.35 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.34,
          axes: FRAME_XZ_AXES,
          gridCols: 16, gridRows: 1,
          role: SHEET_ROLE_KV_K, fp8Flag: 0,
          highlight: SHEET_HL_BF16_GOLD,
          seed: 257, partId: 'den-decay-bank-cum',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'mamba_16head_cumulative_decay_bank', absoluteTokenOffset: P + localPos,
          atlasSlice: mambaAtlas.decayBankCumulative, registryKey: 'den:decay_bank_cum',
          cellFamily: CELL_SSM_CYAN,
        },
        null,
      );

      // Arched SSM recurrence ribbon H_prev -> H_curr
      b.flow.addCubic3D(
        [xHPrev + ssmSize * 0.52, PEDESTAL_Y + 0.04, -0.65],
        [(xHPrev + xHCurr) * 0.5, PEDESTAL_Y + 0.65, -0.65],
        [(xHPrev + xHCurr) * 0.5, PEDESTAL_Y + 0.65, -0.65],
        [xHCurr - ssmSize * 0.52, PEDESTAL_Y + 0.04, -0.65],
        [0, 1, 0], 0.09, RIBBON_SSM_LOOP, true, 16, true, 0.18, 0.18,
      );

      // 2f. Readout y_ssm, D·x skip, y (y_ssm + D·x), gated = y·SiLU(z), 8-group rms_gate, y_norm, W_out, + Residual Gate Output Y
      b.addSheet(
        {
          cx: xReadout - 0.34 * dScaleX, cy: BASE_Y + 0.62, cz: -1.35,
          ex: 0.56 * dScaleX, ey: SHEET_THICK * 0.85, ez: 1.05,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.dInner, gridRows: T,
          role: SHEET_ROLE_SCORES, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 258, partId: 'den-y-ssm',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_y_ssm', absoluteTokenOffset: P,
          atlasSlice: mambaAtlas.ySsm, registryKey: 'den:y_ssm',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xReadout + 0.24 * dScaleX, cy: BASE_Y + 0.62, cz: -1.35,
          ex: 0.42 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.36,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.dHead, gridRows: 1,
          role: SHEET_ROLE_KV_V, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 259, partId: 'den-skip-dx',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_skip_d_x', absoluteTokenOffset: P + localPos,
          headIndex: activeHead, groupIndex: activeGroup,
          atlasSlice: mambaAtlas.skipDx, registryKey: 'den:skip_dx',
          cellFamily: CELL_MOE_AMBER,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xReadout, cy: BASE_Y + 0.12, cz: -0.25,
          ex: 0.72 * dScaleX, ey: SHEET_THICK, ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.dInner, gridRows: T,
          role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: hlDen('y', SHEET_HL_NORMAL),
          seed: 260, partId: 'den-y',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_ssm_readout_y', absoluteTokenOffset: P,
          tensorKey: 'y', atlasSlice: mambaAtlas.y, registryKey: 'den:y',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xReadout, cy: BASE_Y + 0.12, cz: 1.55,
          ex: 0.72 * dScaleX, ey: SHEET_THICK, ez: 1.15,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.dInner, gridRows: T,
          role: SHEET_ROLE_SOFTMAX, fp8Flag: 0,
          highlight: SHEET_HL_NORMAL,
          seed: 261, partId: 'den-gated',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_gated_y_silu_z', absoluteTokenOffset: P,
          atlasSlice: mambaAtlas.gated, registryKey: 'den:gated',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xNorm, cy: BASE_Y + 0.56, cz: 1.55,
          ex: 0.52 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.92,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.nGroups, gridRows: T,
          role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: SHEET_HL_BF16_GOLD,
          seed: 262, partId: 'den-rms-gate',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_8group_rms_gate', absoluteTokenOffset: P,
          atlasSlice: mambaAtlas.rmsGate, registryKey: 'den:rms_gate',
          cellFamily: CELL_MOE_AMBER,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xNorm, cy: BASE_Y + 0.12, cz: 0,
          ex: 0.68 * dScaleX, ey: SHEET_THICK, ez: 1.55,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.dInner, gridRows: T,
          role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: hlDen('y_norm', SHEET_HL_NORMAL),
          seed: 263, partId: 'den-y-norm',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_y_norm', absoluteTokenOffset: P,
          tensorKey: 'y_norm', atlasSlice: mambaAtlas.yNorm, registryKey: 'den:y_norm',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xWout, cy: 1.48, cz: 0,
          ex: 3.40, ey: SHEET_THICK, ez: 0.72,
          axes: FRAME_YZ_AXES,
          gridCols: denoiserSim.dInner, gridRows: denoiserSim.d,
          role: SHEET_ROLE_WEIGHT, fp8Flag: 0,
          highlight: hlDen('w_out', SHEET_HL_NORMAL),
          seed: 264, partId: 'den-w-out',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_w_out',
          tensorKey: 'w_out', atlasSlice: mambaAtlas.wOut, registryKey: 'den:w_out',
          cellFamily: CELL_WEIGHT_BLUE,
        },
        denoiserRevealParams,
      );

      b.addSheet(
        {
          cx: xOut, cy: BASE_Y, cz: 0,
          ex: 0.68 * dScaleX, ey: SHEET_THICK, ez: 1.45,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.d, gridRows: T,
          role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: hlDen('residual', SHEET_HL_NORMAL),
          seed: 265, partId: 'den-residual',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: denoiserSim.residualGateApplied ? 'denoiser_gated_residual' : 'denoiser_residual',
          absoluteTokenOffset: P,
          tensorKey: 'residual', atlasSlice: mambaAtlas.residual, registryKey: 'den:residual',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealParams,
      );

      // Gate z bypass flow + overhead residual highway
      b.flow.addCubic3D(
        [xXbc + 0.48 * dScaleX, BASE_Y + 0.05, 2.55],
        [xHCurr, BASE_Y + 0.14, 2.55],
        [xDecomp, BASE_Y + 0.14, 2.15],
        [xReadout - 0.36 * dScaleX, BASE_Y + 0.16, 1.55],
        [0, 1, 0], 0.08, RIBBON_GATE_FLOW, true, 16, false, 0.18, 0.18,
      );
      b.flow.addCubic3D(
        [xIn, BASE_Y + 0.05, -0.75],
        [xIn, 2.75, -3.55],
        [xOut, 2.75, -3.55],
        [xOut, BASE_Y + 0.08, -0.75],
        [0, 1, 0], 0.13, RIBBON_RESIDUAL, true, 20, true, 0.24, 0.24,
      );

      // Denoiser Landmarks & Calipers
      b.landmarks.push(
        {
          id: 'den-mamba-header',
          label: historicalBoundaryMismatch
            ? `HISTORICAL Denoiser Mamba-2 · Bay ${String(bayId).padStart(2, '0')} (Rec #${dl.sourceRecordIndex}, P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen})`
            : `Denoiser Mamba-2 · Bay ${String(bayId).padStart(2, '0')} (${formatMambaHeadId(activeHead)} of 16h, localPos=${localPos}/${T - 1})`,
          sublabel:
            (historicalBoundaryMismatch ? `${historicalTag} · ` : '') +
            `Model: 16 heads / 8 groups / 4×4 SSM (1 selected slice) · ` +
            `Conv1D: ${decomposition.convWindow.inheritedHistoryColumnCount} ctx + ${decomposition.convWindow.blockColumnCount} blk cols · ` +
            `dA=${decomposition.decayFactor.toFixed(3)} · Π dA=${decomposition.cumulativeDecay.toFixed(3)} · ` +
            (isAfterFirstBidirectionalGqa ? 'Post-Bay-06 GQA (inputs may mix later block pos)' : 'Pre-Bay-06 GQA (strictly causal)'),
          world: [denX(0.5), 2.95, -3.75],
          bayId,
          partId: 'den-h-curr',
          tone: historicalBoundaryMismatch ? 'gold' : 'default',
          align: 'center',
          tier: 'primary',
        },
        {
          id: 'den-conv-replacement-landmark',
          label: `4-Tap Conv1D Window (${decomposition.convWindow.inheritedHistoryColumnCount} Inherited → ${decomposition.convWindow.blockColumnCount} Block)`,
          sublabel: '4 allocated cache cols vs 3 preceding raw history cols at localPos=0 (3→2→1→0 replacement)',
          world: [xConv, 1.15, 2.05],
          bayId,
          partId: 'den-conv1d-window',
          tone: 'gold',
          align: 'center',
          tier: 'primary',
        },
        {
          id: 'den-decomp-landmark',
          label: `Analytical Decomposition (${formatMambaHeadId(activeHead)})`,
          sublabel: `Inherited Retention + Block Write ≈ H_curr (max residual ${decomposition.roundingResidualMaxAbs.toExponential(1)})`,
          world: [xDecomp, PEDESTAL_Y + 1.08, -0.55],
          bayId,
          partId: 'den-inherited-retention',
          tone: 'cyan',
          align: 'center',
          tier: 'primary',
        },
      );

      b.calipers.push(
        {
          id: 'caliper-conv-window',
          aWorld: [xConv - 0.45 * dScaleX, 0.36, 1.82],
          bWorld: [xConv + 0.45 * dScaleX, 0.66, 1.82],
          label: `Conv1D: ${decomposition.convWindow.inheritedHistoryColumnCount} ctx + ${decomposition.convWindow.blockColumnCount} blk (cap=4)`,
          partIds: ['den-conv1d-window'],
        },
        {
          id: 'caliper-ssm-native-4x4',
          aWorld: [xHCurr - ssmSize * 0.5, PEDESTAL_Y + 0.08, -1.10],
          bWorld: [xHCurr + ssmSize * 0.5, PEDESTAL_Y + 0.08, -1.10],
          label: `Native 4×4 SSM (dHead=4 × dState=4 · G${activeGroup} rms=${decomposition.groupRmsDenominator.toFixed(3)})`,
          partIds: ['den-h-curr'],
        },
      );
    } else {
      b.addSheet(
        {
          cx: denX(0.5), cy: BASE_Y + 0.16, cz: 0,
          ex: 4.2, ey: SHEET_THICK, ez: 2.2,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 2,
          role: SHEET_ROLE_NEUTRAL, fp8Flag: 0,
          highlight: SHEET_HL_DIMMED,
          seed: 290, partId: 'den-standby',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_awaiting_pass',
          cellFamily: CELL_NEUTRAL,
        },
        null,
      );
    }
  }

  // ==========================================================================
  // 3. CENTER DUAL-CHANNEL BRIDGE (CONV1D CHANNEL + FP32 SSM CHANNEL SEPARATED)
  // ==========================================================================
  if (towerFocus === 'paired') {
    b.addGuideLine(-3.5, BASE_Y, -3.75, -3.5, BASE_Y, 3.75, 1.1, LINE_STYLE_GUIDE, 0.60);
    b.addGuideLine(-1.8, BASE_Y, -3.75, -1.8, BASE_Y, 3.75, 1.1, LINE_STYLE_GUIDE, 0.60);

    // Bridge Channel 1 Plaque & Ribbon: Conv1D Raw History (+Z front)
    b.addSheet(
      {
        cx: -2.65, cy: 0.42, cz: 1.25,
        ex: 1.32, ey: 0.036, ez: 0.72,
        axes: FRAME_XZ_AXES,
        gridCols: 4, gridRows: 1,
        role: SHEET_ROLE_KV_K, fp8Flag: 0,
        highlight: SHEET_HL_BF16_GOLD,
        seed: 295, partId: 'bridge-mamba-conv',
        towerRole: 'shared', executionId: ctxExecId, traceOrigin: 'persistent_cache',
        semanticTag: 'mamba_conv_channel',
        cellFamily: CELL_MOE_AMBER,
      },
      null,
    );
    b.flow.addCubic3D(
      ctxConvAnchor,
      [-2.95, 0.64, ctxConvAnchor[2]],
      [-2.35, 0.64, denConvAnchor[2]],
      denConvAnchor,
      [0, 1, 0], 0.11, RIBBON_BRIDGE_CONV, true, 18, true, 0.22, 0.22,
    );

    // Bridge Channel 2 Plaque & Ribbon: FP32 SSM Boundary Seed (-Z rear)
    b.addSheet(
      {
        cx: -2.65, cy: PEDESTAL_Y, cz: -1.00,
        ex: 1.32, ey: 0.036, ez: 0.72,
        axes: FRAME_XZ_AXES,
        gridCols: 4, gridRows: 4,
        role: SHEET_ROLE_SCORES, fp8Flag: 0,
        highlight: SHEET_HL_BF16_GOLD,
        seed: 296, partId: 'bridge-mamba-ssm',
        towerRole: 'shared', executionId: ctxExecId, traceOrigin: 'persistent_cache',
        semanticTag: 'mamba_ssm_channel',
        cellFamily: CELL_SSM_CYAN,
      },
      null,
    );
    b.flow.addCubic3D(
      ctxSsmAnchor,
      [-2.95, PEDESTAL_Y + 0.28, ctxSsmAnchor[2]],
      [-2.35, PEDESTAL_Y + 0.28, denSsmAnchor[2]],
      denSsmAnchor,
      [0, 1, 0], 0.11, RIBBON_BRIDGE_SSM, true, 18, true, 0.22, 0.22,
    );

    b.addGuideLine(-2.65, 0.44, 0.85, -2.65, PEDESTAL_Y - 0.04, -0.60, 0.9, LINE_STYLE_BF16_GOLD, 0.45);

    b.landmarks.push({
      id: 'mamba-dual-bridge-landmark',
      label: historicalBoundaryMismatch
        ? `Historical Boundary Bridge (H_${seedLastPos} @ P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen})`
        : `Dual-Channel Boundary Bridge (Conv1D + FP32 SSM H_${seedLastPos})`,
      sublabel: historicalBoundaryMismatch
        ? `${historicalTag} · Prior seed does NOT belong to current Context boundary P=${currentContextPrefixLen}`
        : refinementSeedInvariantNote,
      world: [-2.65, 1.92, 0.10],
      bayId,
      partId: 'bridge-mamba-ssm',
      tone: 'gold',
      align: 'center',
      tier: 'primary',
    });
  }

  // ==========================================================================
  // 4. CELL HIGHLIGHTS, DEPENDENCY RIBBONS, POPOVER & NATIVE 4×4 CELL GRIDS
  // ==========================================================================
  let cellPopover: CellPopoverSpec | null = null;
  const activeWingPrefix: 'ctx' | 'den' = activeTower === 'denoiser' ? 'den' : 'ctx';

  if (activeFormula && activeSim) {
    const tgtSpans = resolvePairedMambaSpans(
      b.sheetRegistry,
      activeWingPrefix,
      activeFormula.targetSpan,
      activeHead,
    );
    const primaryTgt = tgtSpans[0] ?? null;
    if (primaryTgt) {
      for (const s of tgtSpans) {
        drawCellHighlight3D(b.flow, s, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);
      }
      const sourceWorlds: { readonly world: readonly [number, number, number]; readonly role: 'indigo' | 'teal' }[] = [];
      for (const up of activeFormula.upstreamSpans) {
        const upSpans = resolvePairedMambaSpans(b.sheetRegistry, activeWingPrefix, up, activeHead);
        const isOpA = up.role === 'operand_a';
        for (const u of upSpans) {
          drawOperandDependency3D(b.flow, u, primaryTgt, isOpA);
          sourceWorlds.push({ world: u.frontPoint, role: isOpA ? 'indigo' : 'teal' });
        }
      }
      cellPopover = buildMambaCellPopover(
        activeFormula,
        primaryTgt.frontPoint,
        sourceWorlds,
        focusedAccumulation,
      );
    }
  }

  const activeCellGrids: ActiveSheetCellGrid[] = [];
  if (decomposition) {
    const hTag = formatMambaHeadId(activeHead);
    const tokLbl = `t${decomposition.absoluteTokenPosition} ${decomposition.tokenLabel}`;
    const isHCurrTarget = activeFormula?.targetSpan.tensor === 'h_curr';
    const activeValOverride = focusedAccumulation?.active ? focusedAccumulation.partialSum : null;

    const gCurr = gridFromSheet4x4(
      b.sheetRegistry.get('den:h_curr'),
      decomposition.H_curr,
      `H_curr ${hTag}`,
      tokLbl,
      isHCurrTarget ? activeFormula?.targetSpan.colStart : undefined,
      isHCurrTarget ? activeFormula?.targetSpan.rowStart : undefined,
      isHCurrTarget ? activeValOverride : null,
    );
    const gPrev = gridFromSheet4x4(
      b.sheetRegistry.get('den:h_prev'),
      decomposition.H_prev,
      localPos === 0
        ? historicalBoundaryMismatch
          ? `H_init ${hTag} (Historical P_cond=${seedPrefixLen})`
          : `H_init ${hTag} (Context H_${seedLastPos})`
        : `H_prev ${hTag}`,
      localPos === 0
        ? historicalBoundaryMismatch
          ? `prior H_${seedLastPos} ≠ P=${currentContextPrefixLen}`
          : `boundary t${seedLastPos}`
        : `pos ${localPos - 1}`,
    );
    const gRet = gridFromSheet4x4(
      b.sheetRegistry.get('den:inherited_retention'),
      decomposition.inheritedRetentionMatrix,
      `Inherited Ret (${decomposition.cumulativeDecay.toFixed(3)}·H_init)`,
      tokLbl,
    );
    const gWrite = gridFromSheet4x4(
      b.sheetRegistry.get('den:block_write'),
      decomposition.accumulatedBlockWriteMatrix,
      `Block Write Σ(Π dA·Δt·x·B)`,
      tokLbl,
    );
    if (gCurr) activeCellGrids.push(gCurr);
    if (gPrev) activeCellGrids.push(gPrev);
    if (gRet) activeCellGrids.push(gRet);
    if (gWrite) activeCellGrids.push(gWrite);
  }

  // ==========================================================================
  // 5. PART CATALOG, SELECTION IDENTITY, BADGES, WARNINGS & DIAGNOSTICS
  // ==========================================================================
  const convChannelDiag: PairedMambaSeedChannelDiagnostics = Object.freeze({
    channelId: 'mamba_conv_channel',
    label: historicalBoundaryMismatch
      ? `Historical Conv1D 4-Column Seed (P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen})`
      : 'Conv1D 4-Column Raw History Cache',
    ownerRole: 'context_tower',
    consumerRole: 'denoiser_tower',
    traceOrigin: 'persistent_cache',
    storageDtypeMetadata: 'bf16',
    shapeSummary: '4 allocated cache columns (slots [0..3], 3 preceding raw history cols enter localPos=0)',
    sourceBoundaryDigest,
  });

  const ssmChannelDiag: PairedMambaSeedChannelDiagnostics = Object.freeze({
    channelId: 'mamba_ssm_channel',
    label: historicalBoundaryMismatch
      ? `Historical FP32 SSM Seed H_${seedLastPos} (P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen})`
      : `FP32 SSM Boundary Seed H_${seedLastPos}`,
    ownerRole: 'context_tower',
    consumerRole: 'denoiser_tower',
    traceOrigin: 'persistent_cache',
    storageDtypeMetadata: 'fp32',
    shapeSummary: 'Model inventory: 16 Mamba heads, 8 groups, 4×4 per-head FP32 SSM state (64 × [64×128] prod); 1 selected head displayed in 4×4 decomposition',
    sourceBoundaryDigest,
  });

  const partCatalog: Record<string, CutawayPartInfo> = {
    'ctx-conv-state': {
      id: 'ctx-conv-state',
      title: historicalBoundaryMismatch
        ? `Historical Context Conv1D Seed (P_cond=${seedPrefixLen}, Channel 1)`
        : 'Context Persistent Conv1D Raw History (Channel 1)',
      badge: historicalBoundaryMismatch
        ? `Historical Seed · P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen}`
        : 'persistent_cache · BF16 (prod)',
      summary: historicalBoundaryMismatch
        ? `Historical 4-column Conv1D seed read by ${historicalTag} (P_cond=${seedPrefixLen}); does NOT represent current Context boundary P=${currentContextPrefixLen}.`
        : 'Dedicated 4-column Conv1D rolling cache owned by Context Tower; distinct from FP32 SSM state.',
      specs: [
        '4 allocated columns (slots [0..3], slot 3 = newest prior token)',
        'At localPos=0, 3 preceding raw history columns (lags 3, 2, 1) enter alongside xBC_pre[0]',
        `Seed boundary prefix: P_cond=${seedPrefixLen} (current Context P=${currentContextPrefixLen})`,
        `Boundary digest: ${sourceBoundaryDigest}`,
      ],
    },
    'ctx-ssm-state': {
      id: 'ctx-ssm-state',
      title: historicalBoundaryMismatch
        ? `Historical Context FP32 SSM Seed H_${seedLastPos} (P_cond=${seedPrefixLen}, Channel 2)`
        : `Context Persistent FP32 SSM State H_${seedLastPos} (Channel 2)`,
      badge: historicalBoundaryMismatch
        ? `Historical Seed · P_cond=${seedPrefixLen} ≠ P=${currentContextPrefixLen}`
        : 'persistent_cache · FP32 (prod)',
      summary: historicalBoundaryMismatch
        ? `Historical 16×[4×4] FP32 SSM boundary seed H_${seedLastPos} read by ${historicalTag}; does NOT belong to current Context boundary P=${currentContextPrefixLen} (H_${Math.max(0, currentContextPrefixLen - 1)}).`
        : 'Dedicated 16-head / 8-group [4×4] FP32 SSM boundary seed owned by Context Tower; independent from Conv1D history.',
      specs: [
        `Complete miniature inventory: 16 Mamba heads, 8 groups, 4×4 per-head SSM state (atlas slice displays 1 selected head: ${formatMambaHeadId(activeHead)} / Group G${activeGroup})`,
        `Seed boundary prefix: P_cond=${seedPrefixLen} (current Context P=${currentContextPrefixLen})`,
        `Boundary digest: ${sourceBoundaryDigest}`,
      ],
    },
    'bridge-mamba-conv': {
      id: 'bridge-mamba-conv',
      title: 'Cross-Tower Conv1D History Bridge',
      badge: historicalBoundaryMismatch ? 'Historical Seed · Mismatch' : 'Channel 1 · Read-Only',
      summary: `${refinementSeedInvariantNote} Transfers the 4-column Conv1D raw history cache into Denoiser's Conv1D window.`,
      specs: [
        `sourceBoundaryDigest: ${sourceBoundaryDigest} (P_cond=${seedPrefixLen}, current P=${currentContextPrefixLen})`,
        `Source record/refinement: Rec #${dl.sourceRecordIndex ?? 'n/a'} b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}`,
      ],
    },
    'bridge-mamba-ssm': {
      id: 'bridge-mamba-ssm',
      title: 'Cross-Tower FP32 SSM Boundary Bridge',
      badge: historicalBoundaryMismatch ? 'Historical Seed · Mismatch' : 'Channel 2 · Read-Only',
      summary: `${refinementSeedInvariantNote} Seeds Denoiser H_prev at localPos=0 strictly from committed Context H_${seedLastPos}.`,
      specs: [
        `sourceBoundaryDigest: ${sourceBoundaryDigest} (P_cond=${seedPrefixLen}, current P=${currentContextPrefixLen})`,
        `Source record/refinement: Rec #${dl.sourceRecordIndex ?? 'n/a'} b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}`,
      ],
    },
    'den-conv1d-window': {
      id: 'den-conv1d-window',
      title: 'Denoiser 4-Tap Conv1D Window Replacement',
      badge: decomposition
        ? `${decomposition.convWindow.inheritedHistoryColumnCount} ctx + ${decomposition.convWindow.blockColumnCount} blk`
        : '4-Tap Causal',
      summary: decomposition?.convWindow.cacheConventionNote ?? '4-tap depthwise causal Conv1D window.',
      specs: [
        'Allocated cache columns: 4 · Preceding history columns at pos 0: 3',
        'Active block progressively replaces inherited history columns: 3 → 2 → 1 → 0',
      ],
    },
    'den-inherited-retention': {
      id: 'den-inherited-retention',
      title: 'Analytical Inherited Retention ((Π dA) · H_init)',
      badge: decomposition ? `Π dA = ${decomposition.cumulativeDecay.toFixed(4)}` : 'Analytical',
      summary: decomposition?.decaySemanticsDisclaimer ?? 'Closed-form inherited retention component.',
      specs: [
        `Selected 4×4 decomposition slice: ${formatMambaHeadId(activeHead)} (Group G${activeGroup}) of 16 Mamba heads / 8 groups`,
        decomposition ? `Rounding residual max |sum − H_curr| = ${decomposition.roundingResidualMaxAbs.toExponential(2)}` : '',
        decomposition?.causalAfterGqaNote ?? '',
      ].filter(Boolean),
    },
  };

  const activeSelection: TwoTowerSelectionIdentity = buildTwoTowerSelectionIdentity(view, {
    towerRole: activeTower,
    bayId,
    mixerKind: 'M',
    tensorRole: activeCell?.tensor ?? 'h_curr',
    traceOrigin:
      activeTower === 'denoiser'
        ? activeCell?.tensor === 'h_prev' && localPos === 0
          ? 'persistent_cache'
          : 'denoiser_scratch'
        : ctxLookup.traceOrigin,
    executionId: activeTower === 'denoiser' ? denExecId : ctxExecId,
    absoluteTokenPosition: decomposition?.absoluteTokenPosition ?? null,
    headIndex: activeHead,
    groupIndex: activeGroup,
    provenanceNote:
      `Paired Mamba Bay ${String(bayId).padStart(2, '0')} (layer ${layerIndex}) · ${formatMambaHeadId(activeHead)} (1 of 16 heads, 8 groups) · ` +
      `localPos=${localPos} · boundary=${sourceBoundaryDigest} (P_cond=${seedPrefixLen}, current P=${currentContextPrefixLen})` +
      (historicalTag ? ` · ${historicalTag}` : ''),
  });

  const warnings: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warnings.push(`Context snapshot integrity check failed: ${view.snapshotIntegrity.error}`);
  }
  if (ctxLookup.status === 'pruned_unavailable') {
    warnings.push(ctxLookup.reason);
  }
  if (dl.historicalProvenanceWarning !== null) {
    warnings.push(dl.historicalProvenanceWarning);
  }
  const warningBanner = warnings.length > 0 ? warnings.join(' | ') : null;

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    {
      label: `Bay ${String(bayId).padStart(2, '0')} (Layer ${layerIndex}) · Mamba-2 (16h / 8g / 4×4 SSM · Slice: ${formatMambaHeadId(activeHead)})`,
      tone: 'shared',
    },
    {
      label: `Context: ${ctxLookup.traceOrigin} (P=${currentContextPrefixLen})`,
      tone: ctxLookup.status === 'available' ? 'context' : 'warn',
      origin: ctxLookup.traceOrigin,
    },
    {
      label: isAfterFirstBidirectionalGqa
        ? 'Post-Bay-06 GQA (inputs may mix later block pos)'
        : 'Pre-Bay-06 GQA (strictly causal input & recurrence)',
      tone: isAfterFirstBidirectionalGqa ? 'gold' : 'denoiser',
    },
    {
      label: `Seed H_${seedLastPos} (${sourceBoundaryDigest.slice(0, 8)}, P_cond=${seedPrefixLen}) · r=${dl.sourceRefinementIndex ?? 'ctx'}`,
      tone: historicalBoundaryMismatch ? 'warn' : 'muted',
      origin: 'persistent_cache',
    },
  ];
  if (isHistoricalDenoiser) {
    provenanceBadges.push({
      label: `Historical Denoiser Pass · Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, P_cond=${seedPrefixLen} vs P=${currentContextPrefixLen})`,
      tone: 'warn',
      origin: 'denoiser_scratch',
    });
  }

  const fitBounds: AABB =
    towerFocus === 'context_solo'
      ? CONTEXT_SOLO_MAMBA_FIT_BOUNDS
      : towerFocus === 'denoiser_solo'
        ? DENOISER_SOLO_MAMBA_FIT_BOUNDS
        : PAIRED_MAMBA_FIT_BOUNDS;

  const causalAfterGqaNote =
    decomposition?.causalAfterGqaNote ??
    (isAfterFirstBidirectionalGqa
      ? `Bay ${String(bayId).padStart(2, '0')} (layer ${layerIndex}) sits AFTER Bay 06 (first bidirectional GQA): Mamba stays locally causal while input activations X may already reflect later block positions.`
      : `Bay ${String(bayId).padStart(2, '0')} (layer ${layerIndex}) sits BEFORE Bay 06 (first bidirectional GQA): input activations and Mamba recurrence are strictly locally causal.`);

  const decaySemanticsDisclaimer =
    decomposition?.decaySemanticsDisclaimer ??
    'Epistemic note: Per-head decay factors dA and SSM state norms are algebraic magnitudes and do NOT measure retained semantic information.';

  const miniMamba = TWOTOWER_MINI_MODEL_INVENTORY.mamba;
  const diagnostics: PairedMambaDiagnostics = Object.freeze({
    bayId,
    layerIndex,
    towerFocus,
    modeAtRecord: view.modeAtRecord,
    activeHead,
    activeGroup,
    localTokenIndex: localPos,
    blockLength: primaryTokens,
    absoluteTokenPosition: decomposition?.absoluteTokenPosition ?? view.committedPrefixLen,
    refinementIndex: dl.sourceRefinementIndex,
    sourceBoundaryDigest,
    contextSnapshotDigest: view.contextSnapshot.snapshotDigest,
    currentContextPrefixLen,
    currentContextStateDigest,
    denoiserConditioningPrefixLen,
    denoiserSourceRecordIndex: dl.sourceRecordIndex,
    denoiserSourceBlockId: dl.sourceBlockId,
    denoiserSourceRefinementIndex: dl.sourceRefinementIndex,
    isHistoricalDenoiserInspection: isHistoricalDenoiser,
    matchesCurrentContextBoundary: matchesCurrentBoundary,
    historicalProvenanceWarning: dl.historicalProvenanceWarning,
    conditioningDigest: conditioning?.conditioningDigest ?? null,
    refinementSeedInvariantNote,
    convChannel: convChannelDiag,
    ssmChannel: ssmChannelDiag,
    contextTraceStatus: ctxLookup.status,
    contextTraceOrigin: ctxLookup.traceOrigin,
    denoiserPassStatus: dl.status,
    hasDenoiserPreNormMod: Boolean(denoiserSim?.X_mod),
    hasDenoiserResidualGate: Boolean(denoiserSim?.residualGateApplied),
    isAfterFirstBidirectionalGqa,
    causalAfterGqaNote,
    decaySemanticsDisclaimer,
    decomposition,
    modelMambaInventory: Object.freeze({
      heads: miniMamba.heads,
      groups: miniMamba.groups,
      headsPerGroup: miniMamba.headsPerGroup,
      dHead: miniMamba.dHead,
      dState: miniMamba.dState,
      stateMatrixShape: miniMamba.ssmStateMatrixShape,
      displayedHeadIndex: activeHead,
      displayedGroupIndex: activeGroup,
      atlasSelectedDecompHeadCount: 1 as const,
    }),
    atlasDimensions: [TWOTOWER_MAMBA_ATLAS_WIDTH, TWOTOWER_MAMBA_ATLAS_HEIGHT] as const,
  });

  const packed = b.pack();
  const valueAtlas: Rg32fValueAtlas | undefined = mambaAtlas ?? fallbackBoundaryAtlas ?? undefined;
  const atlasKey = `tt-mamba:${view.identity.recordId}:${ctxExecId}:${denExecId}:b${bayId}:f${towerFocus}:h${activeHead}:p${localPos}`;

  return {
    viewMode: 'paired_mamba',
    title: `Paired Mamba-2 Cutaway · Bay ${String(bayId).padStart(2, '0')} (Layer ${layerIndex})`,
    subtitle:
      (historicalBoundaryMismatch ? `HISTORICAL Denoiser (${historicalTag} ≠ P=${currentContextPrefixLen}) · ` : '') +
      `35/65 Context Dual-Channel Boundary (Conv1D + FP32 SSM H_${seedLastPos}) → Denoiser Analytical Decomposition ` +
      `(${formatMambaHeadId(activeHead)} of 16 heads / 8 groups, localPos=${localPos}/${Math.max(0, primaryTokens - 1)})`,
    bayId,
    layerIndex,
    activeTowerFocus: towerFocus,
    ...packed,
    ...(valueAtlas ? { valueAtlas, atlasKey } : {}),
    fitBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    calipers: b.calipers,
    solidPickBoxes: b.pickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    ...(b.activeOpPartIds.size > 0 ? { activeOpPartIds: b.activeOpPartIds } : {}),
    activeCellGrids,
    cellPopover,
    activeFormula,
    focusedAccumulation,
    activeSelection,
    stepperSteps: MAMBA_STEPPER_STEPS,
    provenanceBadges,
    epistemicNote:
      `${decaySemanticsDisclaimer} ${causalAfterGqaNote} ` +
      `Miniature Mamba-2 inventory: 16 heads, 8 groups, 4×4 per-head FP32 SSM state (256×128 atlas packs all 16 heads/8 groups and shows 1 selected head's 4×4 analytical decomposition slice).`,
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}
