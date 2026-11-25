/**
 * Dual-Tower 52×2 Macro Spatial Composition (`src/twotower/scene/dualMacroScene.ts`)
 *
 * Pure headless TypeScript (ZERO DOM/WebGL2 imports, ZERO `Math.random()`) constructing a
 * distinct editorial drafting-table Dual-Tower Macro scene for Nemotron-Labs-TwoTower:
 *   - Left Tower (`x = -12.0`): Context Tower (`ownerRole: 'context_tower'`) with own Embedding,
 *     Residual Stream (`x = -16.8`), 52 hybrid bays (`23 M · 23 E · 6 *`), unmodulated Final
 *     RMSNorm, untied LM Head, AR Output, and **solid persistent cache geometry** (`6` GQA KV
 *     prefix tracks + `24`-slot ghost capacity rails; `23` dual-compartment Mamba-2 slabs
 *     separating Conv1D 4-tap history from FP32 SSM H state; `23` MoE bays annotated `0 B stateless`).
 *   - Right Tower (`x = +12.0`): Denoiser Tower (`consumerRole: 'denoiser_tower'`) vertically
 *     layer-aligned at identical `bayCenterY(1..52)`, with own independent Embedding, Residual
 *     Stream (`x = +16.8`), unmodulated Final RMSNorm, untied LM Head, unshifted Block Output +
 *     `[MASK]^B` tray, and **dashed/frosted ephemeral scratchpad slabs** visually unmistakable
 *     from Context's solid persistent slabs.
 *   - Central Cross-Tower Alignment Bridge (`x ∈ [-7.5, +7.5]`) with strict hierarchical emphasis:
 *     - 3 separately inspectable linker channels across the exact 29 aligned stateful bays
 *       (`6` `gqa_kv`, `23` `mamba_conv` upper sub-lane, `23` `mamba_ssm` lower sub-lane,
 *       strictly `0` at the `23` MoE bays).
 *     - Shared Timestep-Conditioning Rail (`x = 0.0`) with top hub (`t = nMasked / B → φ(t) →
 *       MLP → 3d tEmb`) and 52 taps tagged pre-norm (`M`/`*`) vs post-norm (`E`), visibly dimmed &
 *       dashed (`BYPASSED (t = null)`) in AR and Mock-AR modes.
 *   - Separate Denoiser Refinement Loop (`x ≈ +18.8`) vs. Completed-Block Context Commit Return
 *     Pathway (distinct lower-to-upper emerald bridge to `x ≈ -18.8` → Context Embedding).
 *
 * Strictly driven by `view.modeAtRecord` (`record.configAtStep.mode`, NEVER `session.pendingConfig`),
 * read-only over all session/snapshot typed arrays, and fail-safe when traces are pruned or
 * snapshots externally mutated.
 */

import { formatKiB, formatMiB } from '../../model/memory';
import {
  formatBayId,
  NEMOTRON_SPEC,
  type Kind,
  type Vec3,
} from '../../model/spec';
import type { CameraPose } from '../../model/state';
import {
  add3,
  cross3,
  dot3,
  fitOpBoundsCamera,
  normalize3,
  scale3,
  sub3,
  type AABB,
  type OpBoundsFitOptions,
  type ViewportMetrics,
} from '../../render/camera';
import {
  LINE_FREE_FACE,
  LINE_FREE_OWNER_ID,
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  BAY_52_Y,
  BAY_PITCH_Y,
  bayCenterY,
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  EMBED_Y,
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  FINAL_NORM_Y,
  LM_HEAD_Y,
  OUTPUT_LOGITS_Y,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_INSTANCE_FLOATS,
  RIBBON_KIND_ARROW_NEG_X,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_ARROW_POS_Y,
  RIBBON_KIND_ARROW_POS_Z,
  RIBBON_KIND_BOX,
  SOLID_INSTANCE_FLOATS,
  SPINE_REPLAY_DISCLAIMER,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from '../../scene/spine';
import type { TwoTowerTraceOrigin } from '../contracts';
import type {
  TwoTowerSession,
  TwoTowerSessionConfig,
  TwoTowerSessionLifecyclePhase,
  TwoTowerSessionTransitionKind,
} from '../session';
import {
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  twoTowerMemoryFor,
  type TwoTowerExecutionMode,
} from '../spec';
import {
  TWOTOWER_MINI_MODEL_INVENTORY,
  buildTwoTowerSelectionIdentity,
  getTwoTowerBayKind,
  resolveTwoTowerSessionViewState,
  type TwoTowerLinkerChannel,
  type TwoTowerMixerRole,
  type TwoTowerPairedFocus,
  type TwoTowerProvenanceBadge,
  type TwoTowerSelectionIdentity,
  type TwoTowerSessionViewState,
  type TwoTowerSpatialScene,
  type TwoTowerTimestepStatus,
  type TwoTowerTowerRole,
} from './provenance';

// ============================================================================
// AUTHORITATIVE DUAL-TOWER MACRO COORDINATE CONSTANTS
// ============================================================================

export const CONTEXT_TOWER_X = -12.0 as const;
export const DENOISER_TOWER_X = 12.0 as const;

export const CONTEXT_RESIDUAL_X = -16.8 as const;
export const DENOISER_RESIDUAL_X = 16.8 as const;

export const BRIDGE_LEFT_X = -7.5 as const;
export const BRIDGE_RIGHT_X = 7.5 as const;
export const TIMESTEP_RAIL_X = 0.0 as const;

export const REFINEMENT_LOOP_X = 18.8 as const;
export const COMMIT_BRIDGE_LEFT_X = -18.8 as const;

export const TIMESTEP_HUB_Y = EMBED_Y + 2.8;
export const BLOCK_TRAY_Y = OUTPUT_LOGITS_Y - 2.35;
export const COMMIT_BRIDGE_BOTTOM_Y = OUTPUT_LOGITS_Y - 4.6;

export const MAMBA_CONV_LANE_DY = 0.18 as const;
export const MAMBA_SSM_LANE_DY = -0.18 as const;

export const MACRO_KV_START_Z = 0.90 as const;
export const MACRO_KV_SLICE_PITCH = 0.32 as const;
export const MACRO_KV_SLICE_DEPTH = 0.22 as const;
export const MACRO_KV_RAIL_OFFSETS_X: readonly number[] = Object.freeze([-1.05, -0.45, 0.45, 1.05]);

/** Exact FlatFlowRibbonsPass interleaved vertex stride (10 floats = 40 bytes). */
export const MACRO_FLOW_VERTEX_FLOATS = 10 as const;

/** Default overview framing bounds encompassing both 52-layer towers, bridge, KV tracks, and return pathways. */
export const TWO_TOWER_MACRO_FIT_BOUNDS: AABB = Object.freeze({
  min: Object.freeze([-20.2, COMMIT_BRIDGE_BOTTOM_Y - 0.8, -1.8] as const),
  max: Object.freeze([
    20.2,
    TIMESTEP_HUB_Y + 2.2,
    MACRO_KV_START_Z + TWOTOWER_MINI_CONFIG.kvCapacity * MACRO_KV_SLICE_PITCH + 0.8,
  ] as const),
});

/** Hero anchor bays elevated in default overview (`activeLinkerChannel === 'all'`, no bay selected) to establish visual hierarchy. */
const OVERVIEW_HERO_LINK_BAYS: ReadonlySet<number> = new Set([1, 6, 42, 43]);

// ============================================================================
// PUBLIC OPTION & DIAGNOSTIC TYPES
// ============================================================================

export type TwoTowerMacroRefinementLoopState = 'dormant' | 'active' | 'paused_ready';

export type TwoTowerMacroCommitPathwayState =
  | 'dormant'
  | 'staged_ready'
  | 'active_block_commit'
  | 'active_single_token';

export interface BuildTwoTowerMacroOptions {
  readonly selectedTower?: TwoTowerTowerRole;
  /** 1-based bay `1..52`, `0` = embed, `53` = output, `-1` = timestep rail. */
  readonly selectedBayId?: number | null;
  readonly hoveredBayId?: number | null;
  readonly hoveredTower?: TwoTowerTowerRole | null;
  readonly activeLinkerChannel?: TwoTowerLinkerChannel;
  /** Explicitly optional `0..1` educational replay phase — strictly separated from measured compute metadata. */
  readonly instructionalReplayPhase?: number | null;
  readonly reducedMotion?: boolean;
}

export interface FitTwoTowerMacroCameraOptions extends OpBoundsFitOptions {
  readonly focusTower?: TwoTowerTowerRole;
}

export interface TwoTowerMacroChannelDiagnostics {
  readonly count: number;
  readonly elevatedCount: number;
  readonly ambientCount: number;
  readonly maxAlpha: number;
  readonly minAlpha: number;
  readonly maxCssWidth: number;
  readonly minCssWidth: number;
  readonly flowRibbonCount: number;
}

export interface TwoTowerMacroDiagnostics extends Readonly<Record<string, unknown>> {
  readonly contextBayCount: 52;
  readonly denoiserBayCount: 52;
  readonly alignedGqaLinkerCount: 6;
  readonly alignedMambaConvLinkerCount: 23;
  readonly alignedMambaSsmLinkerCount: 23;
  readonly moeLinkerCount: 0;
  readonly timestepTapCount: 52;
  readonly timestepPreNormTapCount: 29;
  readonly timestepPostNormTapCount: 23;
  readonly timestepRailActive: boolean;
  readonly timestepValue: number | null;
  readonly timestepStatusLabel: string;
  readonly timestepStatus: TwoTowerTimestepStatus;
  readonly refinementLoopState: TwoTowerMacroRefinementLoopState;
  readonly blockCommitPathwayState: TwoTowerMacroCommitPathwayState;
  readonly contextTraceStatus: 'available' | 'pruned_unavailable';
  readonly denoiserLookupStatus: 'active_on_record' | 'historical_prior_record' | 'none_yet';
  readonly isHistoricalDenoiserInspection: boolean;
  readonly denoiserSourceRecordIndex: number | null;
  readonly denoiserSourceBlockId: number | null;
  readonly denoiserSourceRefinementIndex: number | null;
  readonly denoiserSourceContextPrefixLen: number | null;
  readonly denoiserSourceContextCacheDigest: string | null;
  readonly denoiserMatchesCurrentContextBoundary: boolean;
  readonly historicalProvenanceWarning: string | null;
  readonly modelDimensionalInventory: typeof TWOTOWER_MINI_MODEL_INVENTORY;
  readonly snapshotIntegrityValid: boolean;
  readonly snapshotIntegrityError: string | null;
  readonly instructionalReplayPhase: number | null;
  readonly instructionalReplayDisclaimer: string | null;
  readonly reducedMotion: boolean;
  readonly measuredExecutionMode: TwoTowerExecutionMode;
  readonly historicalConfigAtStep: TwoTowerSessionConfig;
  readonly lifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly recordKind: TwoTowerSessionTransitionKind;
  readonly recordId: string;
  readonly branchId: string;
  readonly cursorIndex: number;
  readonly committedPrefixLen: number;
  readonly kvCapacity: number;
  readonly contextKvConditioningLen: number;
  readonly activeBlockSize: number;
  readonly activeBlockMaskedCount: number;
  readonly contextModelId: string;
  readonly denoiserModelId: string;
  readonly contextParameterFingerprint: string;
  readonly denoiserParameterFingerprint: string;
  readonly activeLinkerChannel: TwoTowerLinkerChannel;
  readonly channelEmphasis: {
    readonly gqa_kv: TwoTowerMacroChannelDiagnostics;
    readonly mamba_conv: TwoTowerMacroChannelDiagnostics;
    readonly mamba_ssm: TwoTowerMacroChannelDiagnostics;
    readonly timestep: TwoTowerMacroChannelDiagnostics;
  };
  readonly flowVertexCount: number;
  readonly flowTriangleCount: number;
}

// ============================================================================
// PUBLIC WORLD ANCHOR & BOUNDS HELPERS
// ============================================================================

export function macroKvSliceCenterZ(slotZeroBased: number): number {
  return MACRO_KV_START_Z + MACRO_KV_SLICE_PITCH * slotZeroBased + MACRO_KV_SLICE_DEPTH * 0.5;
}

export function macroKvSliceFrontZ(slotZeroBased: number): number {
  return MACRO_KV_START_Z + MACRO_KV_SLICE_PITCH * slotZeroBased + MACRO_KV_SLICE_DEPTH;
}

/** Unobstructed front-face world probe on Context Tower bay `1..52` (for ray-picking & camera focus). */
export function contextBayFrontWorld(bayId: number): Vec3 {
  const kind = getTwoTowerBayKind(bayId);
  const cy = bayCenterY(bayId);
  if (kind === '*') return [CONTEXT_TOWER_X - 2.65, cy, 0.44]; // Outer-left O-proj face clear of +Z KV tracks
  if (kind === 'M') return [CONTEXT_TOWER_X - 1.45, cy, 0.42]; // Context Mamba main body plate
  return [CONTEXT_TOWER_X - 0.85, cy, 0.46]; // Context MoE shared/router slab
}

/** Unobstructed front-face world probe on Context Mamba-2 Conv1D 4-tap compartment (`bayId` must be `'M'`). */
export function contextMambaConvFrontWorld(bayId: number): Vec3 {
  return [CONTEXT_TOWER_X + 2.85, bayCenterY(bayId) + 0.24, 0.60];
}

/** Unobstructed front-face world probe on Context Mamba-2 FP32 SSM H compartment (`bayId` must be `'M'`). */
export function contextMambaSsmFrontWorld(bayId: number): Vec3 {
  return [CONTEXT_TOWER_X + 2.85, bayCenterY(bayId) - 0.24, 0.62];
}

/** Unobstructed front-face world probe on Denoiser Tower bay `1..52` (for ray-picking & camera focus). */
export function denoiserBayFrontWorld(bayId: number): Vec3 {
  const kind = getTwoTowerBayKind(bayId);
  const cy = bayCenterY(bayId);
  if (kind === '*') return [DENOISER_TOWER_X + 2.15, cy, 0.44]; // O-proj plate right of ephemeral KV track
  if (kind === 'M') return [DENOISER_TOWER_X + 1.45, cy, 0.42]; // Denoiser Mamba main body plate
  return [DENOISER_TOWER_X + 0.85, cy, 0.46]; // Denoiser MoE shared/router slab
}

/** Focused AABB framing one aligned bay across both towers or one focused tower. */
export function twoTowerMacroBayBounds(
  bayId: number,
  tower: TwoTowerTowerRole = 'shared',
): AABB {
  if (bayId === -1) {
    return {
      min: [-4.0, bayCenterY(8) - 2.0, -1.2],
      max: [4.0, TIMESTEP_HUB_Y + 2.0, 1.2],
    };
  }
  if (bayId === 0) {
    return {
      min: [tower === 'denoiser' ? 6.0 : -18.5, bayCenterY(3) - 1.0, -1.6],
      max: [tower === 'context' ? -6.0 : 18.5, TIMESTEP_HUB_Y + 1.6, 2.2],
    };
  }
  if (bayId === 53) {
    return {
      min: [tower === 'denoiser' ? 6.0 : -19.2, COMMIT_BRIDGE_BOTTOM_Y - 1.0, -1.6],
      max: [tower === 'context' ? -6.0 : 19.2, BAY_52_Y + 2.2, 2.4],
    };
  }
  const cy = bayCenterY(Math.max(1, Math.min(52, Math.trunc(bayId))));
  const minX = tower === 'denoiser' ? 6.2 : -17.6;
  const maxX = tower === 'context' ? -6.2 : 17.6;
  const maxZ =
    bayId >= 1 && bayId <= 52 && getTwoTowerBayKind(bayId) === '*'
      ? MACRO_KV_START_Z + TWOTOWER_MINI_CONFIG.kvCapacity * MACRO_KV_SLICE_PITCH + 0.8
      : 2.2;
  return {
    min: [minX, cy - 2.8, -1.5],
    max: [maxX, cy + 2.8, maxZ],
  };
}

/**
 * Editorial perspective camera fitter for the Dual-Tower 52×2 Macro scene (or any focused sub-bounds).
 * Delegates to Nano's inset-aware `fitOpBoundsCamera` with calibrated dual-tower framing defaults.
 */
export function fitTwoTowerMacroCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: FitTwoTowerMacroCameraOptions,
): CameraPose {
  const defaultAz =
    options?.focusTower === 'context' ? 18 : options?.focusTower === 'denoiser' ? -18 : 12;
  return fitOpBoundsCamera(bounds, metrics, {
    azimuthDeg: options?.azimuthDeg ?? defaultAz,
    elevationDeg: options?.elevationDeg ?? 10,
    fovDeg: options?.fovDeg ?? 36,
    fillFraction: options?.fillFraction ?? 0.86,
    fillX: options?.fillX,
    fillY: options?.fillY,
    topInsetPx: options?.topInsetPx ?? 16,
    bottomInsetPx: options?.bottomInsetPx ?? 16,
    leftInsetPx: options?.leftInsetPx ?? 12,
    rightInsetPx: options?.rightInsetPx ?? 12,
    isNarrow: options?.isNarrow,
    minDistance: options?.minDistance,
  });
}

// ============================================================================
// INTERNAL PURE HEADLESS GEOMETRY PACKERS (ZERO WEBGL / DOM IMPORTS)
// ============================================================================

interface FlowColorRGBA {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
  readonly borderAlpha: number;
}

const FLOW_COLOR_GQA_KV: FlowColorRGBA = { r: 0.14, g: 0.56, b: 0.78, a: 0.88, borderAlpha: 0.42 };
const FLOW_COLOR_MAMBA_CONV: FlowColorRGBA = { r: 0.88, g: 0.57, b: 0.12, a: 0.88, borderAlpha: 0.42 };
const FLOW_COLOR_MAMBA_SSM: FlowColorRGBA = { r: 0.12, g: 0.71, b: 0.88, a: 0.88, borderAlpha: 0.42 };
const FLOW_COLOR_TIMESTEP: FlowColorRGBA = { r: 0.46, g: 0.42, b: 0.86, a: 0.84, borderAlpha: 0.38 };
const FLOW_COLOR_REFINEMENT: FlowColorRGBA = { r: 0.39, g: 0.48, b: 0.92, a: 0.86, borderAlpha: 0.40 };
const FLOW_COLOR_REFINEMENT_PAUSED: FlowColorRGBA = { r: 0.62, g: 0.66, b: 0.76, a: 0.34, borderAlpha: 0.22 };
const FLOW_COLOR_COMMIT_ACTIVE: FlowColorRGBA = { r: 0.15, g: 0.74, b: 0.46, a: 0.90, borderAlpha: 0.44 };
const FLOW_COLOR_COMMIT_STAGED: FlowColorRGBA = { r: 0.19, g: 0.72, b: 0.48, a: 0.44, borderAlpha: 0.30 };

/**
 * Pure headless parallel-transport 3D ribbon builder matching `FlatFlowRibbonsPass`'s exact
 * 10-float interleaved vertex layout (`[x, y, z, edge, along, r, g, b, a, borderAlpha]`).
 */
class PureMacroFlowBuilder {
  private readonly verts: number[] = [];
  private readonly indices: number[] = [];

  private pushVertex(
    x: number,
    y: number,
    z: number,
    edge: number,
    along: number,
    col: FlowColorRGBA,
  ): number {
    const idx = this.verts.length / MACRO_FLOW_VERTEX_FLOATS;
    this.verts.push(x, y, z, edge, along, col.r, col.g, col.b, col.a, col.borderAlpha);
    return idx;
  }

  addPolyline3D(
    rawPoints: readonly Vec3[],
    upHint: Vec3,
    width: number,
    color: FlowColorRGBA,
    options?: {
      readonly arrowHead?: boolean;
      readonly arrowLength?: number;
      readonly arrowWidth?: number;
      readonly crossRibbon?: boolean;
      readonly staticPulse?: boolean;
    },
  ): void {
    const pts: Vec3[] = [];
    for (const p of rawPoints) {
      if (pts.length === 0) {
        pts.push(p);
      } else {
        const prev = pts[pts.length - 1];
        if (Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]) > 1e-4) {
          pts.push(p);
        }
      }
    }
    if (pts.length < 2) return;

    const arrowHead = options?.arrowHead ?? true;
    const arrowLength = options?.arrowLength ?? Math.max(width * 2.1, 0.38);
    const arrowWidth = options?.arrowWidth ?? Math.max(width * 2.3, 0.42);
    const crossRibbon = options?.crossRibbon ?? true;
    const staticPulse = options?.staticPulse ?? false;

    const shaft: Vec3[] = pts.map((p) => [p[0], p[1], p[2]]);
    let tipPoint: Vec3 | null = null;
    let tipDir: Vec3 | null = null;

    if (arrowHead) {
      const last = shaft[shaft.length - 1];
      const prev = shaft[shaft.length - 2];
      const segVec = sub3(last, prev);
      const segLen = Math.hypot(segVec[0], segVec[1], segVec[2]);
      if (segLen > 1e-4) {
        tipDir = [segVec[0] / segLen, segVec[1] / segLen, segVec[2] / segLen];
        tipPoint = [last[0], last[1], last[2]];
        const trim = Math.min(arrowLength * 0.85, segLen * 0.55);
        shaft[shaft.length - 1] = sub3(last, scale3(tipDir, trim));
      }
    }

    const n = shaft.length;
    const tangents: Vec3[] = [];
    for (let i = 0; i < n; i++) {
      let dir: Vec3;
      if (i === 0) {
        dir = sub3(shaft[1], shaft[0]);
      } else if (i === n - 1) {
        dir = sub3(shaft[n - 1], shaft[n - 2]);
      } else {
        const dPrev = normalize3(sub3(shaft[i], shaft[i - 1]));
        const dNext = normalize3(sub3(shaft[i + 1], shaft[i]));
        const avg: Vec3 = [dPrev[0] + dNext[0], dPrev[1] + dNext[1], dPrev[2] + dNext[2]];
        dir = Math.hypot(avg[0], avg[1], avg[2]) > 1e-5 ? avg : dNext;
      }
      tangents.push(normalize3(dir));
    }

    const normals: Vec3[] = [];
    const binormals: Vec3[] = [];
    let n0Cand = sub3(upHint, scale3(tangents[0], dot3(upHint, tangents[0])));
    if (Math.hypot(n0Cand[0], n0Cand[1], n0Cand[2]) < 1e-5) {
      const alt: Vec3 = Math.abs(tangents[0][1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      n0Cand = sub3(alt, scale3(tangents[0], dot3(alt, tangents[0])));
    }
    normals.push(normalize3(n0Cand));
    binormals.push(normalize3(cross3(tangents[0], normals[0])));

    for (let i = 1; i < n; i++) {
      const prevN = normals[i - 1];
      const tCur = tangents[i];
      let nextN = sub3(prevN, scale3(tCur, dot3(prevN, tCur)));
      if (Math.hypot(nextN[0], nextN[1], nextN[2]) < 1e-5) {
        nextN = cross3(binormals[i - 1], tCur);
      }
      const nUnit = normalize3(nextN);
      normals.push(nUnit);
      binormals.push(normalize3(cross3(tCur, nUnit)));
    }

    const emitStrip = (axes: readonly Vec3[], halfW: number, col: FlowColorRGBA) => {
      const baseIndex = this.verts.length / MACRO_FLOW_VERTEX_FLOATS;
      for (let i = 0; i < n; i++) {
        const p = shaft[i];
        const side = axes[i];
        const along = staticPulse ? -1 : n > 1 ? i / (n - 1) : 0;
        const pL = sub3(p, scale3(side, halfW));
        const pR = add3(p, scale3(side, halfW));
        this.pushVertex(pL[0], pL[1], pL[2], -1, along, col);
        this.pushVertex(pR[0], pR[1], pR[2], 1, along, col);
      }
      for (let i = 0; i < n - 1; i++) {
        const i0 = baseIndex + i * 2;
        this.indices.push(i0, i0 + 1, i0 + 2, i0 + 1, i0 + 3, i0 + 2);
      }
    };

    const halfW = width * 0.5;
    emitStrip(binormals, halfW, color);
    if (crossRibbon) {
      emitStrip(normals, halfW * 0.75, {
        ...color,
        a: color.a * 0.78,
        borderAlpha: color.borderAlpha * 0.7,
      });
    }

    if (arrowHead && tipPoint && tipDir) {
      const endB = binormals[n - 1];
      const endN = normals[n - 1];
      const basePt = sub3(tipPoint, scale3(tipDir, arrowLength));
      const ah = arrowWidth * 0.5;
      const alongVal = staticPulse ? -1 : 1;
      const emitHead = (side: Vec3, col: FlowColorRGBA) => {
        const pL = sub3(basePt, scale3(side, ah));
        const pR = add3(basePt, scale3(side, ah));
        const vL = this.pushVertex(pL[0], pL[1], pL[2], -0.9, alongVal, col);
        const vR = this.pushVertex(pR[0], pR[1], pR[2], 0.9, alongVal, col);
        const vT = this.pushVertex(tipPoint![0], tipPoint![1], tipPoint![2], 0, alongVal, col);
        this.indices.push(vL, vR, vT);
      };
      emitHead(endB, color);
      if (crossRibbon) {
        emitHead(endN, { ...color, a: color.a * 0.8 });
      }
    }
  }

  addRoundedOrtho3D(
    waypoints: readonly Vec3[],
    radius: number,
    upHint: Vec3,
    width: number,
    color: FlowColorRGBA,
    options?: {
      readonly arrowHead?: boolean;
      readonly arrowLength?: number;
      readonly arrowWidth?: number;
      readonly crossRibbon?: boolean;
      readonly staticPulse?: boolean;
    },
  ): void {
    if (waypoints.length < 2) return;
    if (waypoints.length === 2) {
      this.addPolyline3D(waypoints, upHint, width, color, options);
      return;
    }

    const pts: Vec3[] = [waypoints[0]];
    const arcSteps = 6;
    for (let i = 1; i < waypoints.length - 1; i++) {
      const prev = waypoints[i - 1];
      const curr = waypoints[i];
      const next = waypoints[i + 1];
      const vA = sub3(prev, curr);
      const vB = sub3(next, curr);
      const lenA = Math.hypot(vA[0], vA[1], vA[2]);
      const lenB = Math.hypot(vB[0], vB[1], vB[2]);
      if (lenA < 1e-4 || lenB < 1e-4) {
        pts.push(curr);
        continue;
      }
      const r = Math.min(radius, lenA * 0.42, lenB * 0.42);
      const start = add3(curr, scale3(vA, r / lenA));
      const end = add3(curr, scale3(vB, r / lenB));
      for (let s = 0; s <= arcSteps; s++) {
        const t = s / arcSteps;
        const u = 1 - t;
        pts.push([
          u * u * start[0] + 2 * u * t * curr[0] + t * t * end[0],
          u * u * start[1] + 2 * u * t * curr[1] + t * t * end[1],
          u * u * start[2] + 2 * u * t * curr[2] + t * t * end[2],
        ]);
      }
    }
    pts.push(waypoints[waypoints.length - 1]);
    this.addPolyline3D(pts, upHint, width, color, options);
  }

  pack(): { vertices: Float32Array; indices: Uint32Array } {
    return {
      vertices: new Float32Array(this.verts),
      indices: new Uint32Array(this.indices),
    };
  }
}

interface SolidSpec {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly cellFamily: number;
  readonly fp8Flag: number;
  readonly phase: number;
  readonly bayId: number;
  readonly cellScale: number;
  readonly highlightFlag: number;
  readonly lineStyle: number;
  readonly cssWidth: number;
  readonly lineAlpha: number;
  readonly emitSolidEdges?: boolean;
  readonly dashedCage?: boolean;
  readonly pick?: {
    readonly towerRole: TwoTowerTowerRole;
    readonly partId: string;
    readonly executionId: string;
    readonly traceOrigin: TwoTowerTraceOrigin;
    readonly semanticTag: string;
    readonly absoluteTokenOffset: number;
    readonly tokenIndex?: number;
    readonly kvHeadIndex?: 0 | 1;
    readonly tensorKey?: string;
  };
}

interface LineSpec {
  readonly ax: number;
  readonly ay: number;
  readonly az: number;
  readonly bx: number;
  readonly by: number;
  readonly bz: number;
  readonly cssWidth: number;
  readonly style: number;
  readonly alpha: number;
  readonly ownerId: number;
  readonly face0: number;
  readonly face1: number;
  readonly ocx: number;
  readonly ocy: number;
  readonly ocz: number;
  readonly oex: number;
  readonly oey: number;
  readonly oez: number;
}

interface RibbonSpec {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly kind: number;
  readonly colorId: number;
  readonly alpha: number;
}

function quant(v: number): number {
  return Math.round(v * 10000);
}

function segmentKey(ax: number, ay: number, az: number, bx: number, by: number, bz: number): string {
  const qax = quant(ax), qay = quant(ay), qaz = quant(az);
  const qbx = quant(bx), qby = quant(by), qbz = quant(bz);
  const aFirst =
    qax < qbx || (qax === qbx && (qay < qby || (qay === qby && qaz <= qbz)));
  return aFirst
    ? `${qax},${qay},${qaz}|${qbx},${qby},${qbz}`
    : `${qbx},${qby},${qbz}|${qax},${qay},${qaz}`;
}

function emitCuboidEdges(s: SolidSpec, dedupMap: Map<string, LineSpec>): void {
  const hx = s.ex * 0.5;
  const hy = s.ey * 0.5;
  const hz = s.ez * 0.5;
  const x0 = s.cx - hx, x1 = s.cx + hx;
  const y0 = s.cy - hy, y1 = s.cy + hy;
  const z0 = s.cz - hz, z1 = s.cz + hz;

  const rawEdges: readonly [number, number, number, number, number, number, number, number][] = [
    [x0, y0, z0, x1, y0, z0, FACE_NEG_Y, FACE_NEG_Z],
    [x0, y1, z0, x1, y1, z0, FACE_POS_Y, FACE_NEG_Z],
    [x0, y0, z1, x1, y0, z1, FACE_NEG_Y, FACE_POS_Z],
    [x0, y1, z1, x1, y1, z1, FACE_POS_Y, FACE_POS_Z],
    [x0, y0, z0, x0, y1, z0, FACE_NEG_X, FACE_NEG_Z],
    [x1, y0, z0, x1, y1, z0, FACE_POS_X, FACE_NEG_Z],
    [x0, y0, z1, x0, y1, z1, FACE_NEG_X, FACE_POS_Z],
    [x1, y0, z1, x1, y1, z1, FACE_POS_X, FACE_POS_Z],
    [x0, y0, z0, x0, y0, z1, FACE_NEG_X, FACE_NEG_Y],
    [x1, y0, z0, x1, y0, z1, FACE_POS_X, FACE_NEG_Y],
    [x0, y1, z0, x0, y1, z1, FACE_NEG_X, FACE_POS_Y],
    [x1, y1, z0, x1, y1, z1, FACE_POS_X, FACE_POS_Y],
  ];

  for (const [ax, ay, az, bx, by, bz, face0, face1] of rawEdges) {
    const key = segmentKey(ax, ay, az, bx, by, bz);
    const existing = dedupMap.get(key);
    const cand: LineSpec = {
      ax, ay, az, bx, by, bz,
      cssWidth: s.cssWidth,
      style: s.lineStyle,
      alpha: s.lineAlpha,
      ownerId: s.bayId,
      face0,
      face1,
      ocx: s.cx, ocy: s.cy, ocz: s.cz,
      oex: s.ex, oey: s.ey, oez: s.ez,
    };
    if (!existing) {
      dedupMap.set(key, cand);
    } else if (cand.style === LINE_STYLE_ACTIVE && existing.style !== LINE_STYLE_ACTIVE) {
      dedupMap.set(key, cand);
    } else if (cand.style === LINE_STYLE_BF16_GOLD && existing.style === LINE_STYLE_NEUTRAL) {
      dedupMap.set(key, cand);
    }
  }
}

function pushDashedLine3D(
  freeLines: LineSpec[],
  a: Vec3,
  b: Vec3,
  cssWidth: number,
  style: number,
  alpha: number,
  dash = 0.28,
  gap = 0.18,
): void {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-4) return;
  const ux = dx / len;
  const uy = dy / len;
  const uz = dz / len;
  const step = dash + gap;
  for (let d = 0; d < len; d += step) {
    const dEnd = Math.min(len, d + dash);
    if (dEnd - d < 1e-4) continue;
    freeLines.push({
      ax: a[0] + ux * d,
      ay: a[1] + uy * d,
      az: a[2] + uz * d,
      bx: a[0] + ux * dEnd,
      by: a[1] + uy * dEnd,
      bz: a[2] + uz * dEnd,
      cssWidth,
      style,
      alpha,
      ownerId: LINE_FREE_OWNER_ID,
      face0: LINE_FREE_FACE,
      face1: LINE_FREE_FACE,
      ocx: 0, ocy: 0, ocz: 0,
      oex: 0, oey: 0, oez: 0,
    });
  }
}

function pushDashedBoxCage3D(
  freeLines: LineSpec[],
  cx: number,
  cy: number,
  cz: number,
  ex: number,
  ey: number,
  ez: number,
  cssWidth: number,
  style: number,
  alpha: number,
  dash = 0.24,
  gap = 0.16,
): void {
  const pad = 0.05;
  const hx = ex * 0.5 + pad;
  const hy = ey * 0.5 + pad;
  const hz = ez * 0.5 + pad;
  const x0 = cx - hx, x1 = cx + hx;
  const y0 = cy - hy, y1 = cy + hy;
  const z0 = cz - hz, z1 = cz + hz;

  // Front & back rectangles + corner depth struts communicate lightweight dashed scratchpad frame
  const segs: readonly [Vec3, Vec3][] = [
    [[x0, y0, z1], [x1, y0, z1]],
    [[x1, y0, z1], [x1, y1, z1]],
    [[x1, y1, z1], [x0, y1, z1]],
    [[x0, y1, z1], [x0, y0, z1]],
    [[x0, y0, z0], [x1, y0, z0]],
    [[x0, y1, z0], [x1, y1, z0]],
    [[x0, y1, z0], [x0, y1, z1]],
    [[x1, y1, z0], [x1, y1, z1]],
  ];
  for (const [p0, p1] of segs) {
    pushDashedLine3D(freeLines, p0, p1, cssWidth, style, alpha, dash, gap);
  }
}

// ============================================================================
// HIERARCHICAL EMPHASIS POLICY (ANTI-CLUTTER MANDATE)
// ============================================================================

type ConcreteLinkerChannel = 'gqa_kv' | 'mamba_conv' | 'mamba_ssm' | 'timestep';

interface ChannelEmphasisDecision {
  readonly elevated: boolean;
  readonly cssWidth: number;
  readonly alpha: number;
  readonly lineStyle: number;
  readonly emitFlowRibbon: boolean;
  readonly dashed: boolean;
}

function resolveChannelEmphasis(params: {
  readonly channel: ConcreteLinkerChannel;
  readonly bayId: number;
  readonly filter: TwoTowerLinkerChannel;
  readonly isBayFocused: boolean;
  readonly hasFocusedBay: boolean;
  readonly modeAtRecord: TwoTowerExecutionMode;
  readonly isOnReplayWave: boolean;
}): ChannelEmphasisDecision {
  const {
    channel,
    bayId,
    filter,
    isBayFocused,
    hasFocusedBay,
    modeAtRecord,
    isOnReplayWave,
  } = params;

  const isTimestep = channel === 'timestep';
  const modeActive = isTimestep
    ? modeAtRecord === 'mask_diffusion'
    : modeAtRecord === 'mask_diffusion' || modeAtRecord === 'mock_ar';

  // Bypassed timestep taps in AR / Mock-AR ALWAYS remain clearly dimmed & dashed
  if (isTimestep && !modeActive) {
    const dimAlpha = filter === 'timestep' ? 0.22 : isBayFocused ? 0.20 : 0.11;
    return {
      elevated: false,
      cssWidth: 0.85,
      alpha: dimAlpha,
      lineStyle: LINE_STYLE_GUIDE,
      emitFlowRibbon: false,
      dashed: true,
    };
  }

  // Dormant cross-tower linkers in AR mode
  if (!isTimestep && !modeActive) {
    if (filter === channel) {
      // User explicitly filtered this channel to inspect topology even in AR mode
      return {
        elevated: true,
        cssWidth: 1.65,
        alpha: isBayFocused ? 0.86 : 0.78,
        lineStyle: LINE_STYLE_ACTIVE,
        emitFlowRibbon: true,
        dashed: false,
      };
    }
    return {
      elevated: false,
      cssWidth: 0.85,
      alpha: filter === 'all' ? (isBayFocused ? 0.24 : 0.12) : 0.10,
      lineStyle: LINE_STYLE_GUIDE,
      emitFlowRibbon: false,
      dashed: false,
    };
  }

  // Specific channel filter active (`gqa_kv`, `mamba_conv`, `mamba_ssm`, or `timestep`)
  if (filter !== 'all') {
    if (filter === channel) {
      const lead = isBayFocused || (!hasFocusedBay && OVERVIEW_HERO_LINK_BAYS.has(bayId));
      return {
        elevated: true,
        cssWidth: lead ? 1.90 : 1.65,
        alpha: lead ? 0.94 : 0.82,
        lineStyle: channel === 'mamba_conv' ? LINE_STYLE_BF16_GOLD : LINE_STYLE_ACTIVE,
        emitFlowRibbon: true,
        dashed: false,
      };
    }
    // Non-matching channels strictly recede (`alpha <= 0.14`)
    return {
      elevated: false,
      cssWidth: 0.85,
      alpha: 0.12,
      lineStyle: LINE_STYLE_GUIDE,
      emitFlowRibbon: false,
      dashed: false,
    };
  }

  // Default `'all'` filter: strict hierarchical prominence
  if (isBayFocused || isOnReplayWave) {
    return {
      elevated: true,
      cssWidth: isBayFocused ? 1.85 : 1.65,
      alpha: isBayFocused ? 0.94 : 0.82,
      lineStyle: channel === 'mamba_conv' ? LINE_STYLE_BF16_GOLD : LINE_STYLE_ACTIVE,
      emitFlowRibbon: true,
      dashed: false,
    };
  }

  // When no bay is explicitly selected, elevate only curated hero conduits (`{1, 6, 42, 43}`)
  // so unselected ambient linkers & taps remain quiet drafting hairlines (`alpha <= 0.24`, `cssWidth = 0.85`).
  if (!hasFocusedBay && OVERVIEW_HERO_LINK_BAYS.has(bayId)) {
    return {
      elevated: true,
      cssWidth: 1.65,
      alpha: isTimestep ? 0.80 : 0.84,
      lineStyle: channel === 'mamba_conv' ? LINE_STYLE_BF16_GOLD : LINE_STYLE_ACTIVE,
      emitFlowRibbon: true,
      dashed: false,
    };
  }

  return {
    elevated: false,
    cssWidth: 0.85,
    alpha: isTimestep ? 0.16 : 0.22,
    lineStyle: LINE_STYLE_GUIDE,
    emitFlowRibbon: false,
    dashed: false,
  };
}

interface ChannelStatsAccumulator {
  count: number;
  elevatedCount: number;
  ambientCount: number;
  maxAlpha: number;
  minAlpha: number;
  maxCssWidth: number;
  minCssWidth: number;
  flowRibbonCount: number;
}

function createChannelStats(): ChannelStatsAccumulator {
  return {
    count: 0,
    elevatedCount: 0,
    ambientCount: 0,
    maxAlpha: 0,
    minAlpha: 1,
    maxCssWidth: 0,
    minCssWidth: Infinity,
    flowRibbonCount: 0,
  };
}

function recordChannelStat(acc: ChannelStatsAccumulator, d: ChannelEmphasisDecision): void {
  acc.count++;
  if (d.elevated) acc.elevatedCount++;
  else acc.ambientCount++;
  if (d.alpha > acc.maxAlpha) acc.maxAlpha = d.alpha;
  if (d.alpha < acc.minAlpha) acc.minAlpha = d.alpha;
  if (d.cssWidth > acc.maxCssWidth) acc.maxCssWidth = d.cssWidth;
  if (d.cssWidth < acc.minCssWidth) acc.minCssWidth = d.cssWidth;
  if (d.emitFlowRibbon) acc.flowRibbonCount++;
}

function freezeChannelStats(acc: ChannelStatsAccumulator): TwoTowerMacroChannelDiagnostics {
  return Object.freeze({
    count: acc.count,
    elevatedCount: acc.elevatedCount,
    ambientCount: acc.ambientCount,
    maxAlpha: acc.count > 0 ? acc.maxAlpha : 0,
    minAlpha: acc.count > 0 ? acc.minAlpha : 0,
    maxCssWidth: acc.count > 0 ? acc.maxCssWidth : 0,
    minCssWidth: acc.count > 0 ? acc.minCssWidth : 0,
    flowRibbonCount: acc.flowRibbonCount,
  });
}

// ============================================================================
// PATHWAY STATE DERIVATION
// ============================================================================

function resolveRefinementLoopState(
  view: TwoTowerSessionViewState,
): TwoTowerMacroRefinementLoopState {
  if (view.modeAtRecord !== 'mask_diffusion') return 'dormant';
  if (
    view.lifecyclePhase === 'block_initialized' ||
    view.lifecyclePhase === 'refinement_denoised' ||
    view.lifecyclePhase === 'block_refining'
  ) {
    return 'active';
  }
  if (view.lifecyclePhase === 'completed_uncommitted') {
    return 'paused_ready';
  }
  return 'dormant';
}

function resolveCommitPathwayState(
  view: TwoTowerSessionViewState,
): TwoTowerMacroCommitPathwayState {
  if (view.record.kind === 'diffusion_block_commit') {
    return 'active_block_commit';
  }
  if (view.record.kind === 'mock_ar_step') {
    return 'active_single_token';
  }
  if (view.lifecyclePhase === 'completed_uncommitted') {
    return 'staged_ready';
  }
  return 'dormant';
}

function resolveContextBayOriginFast(
  view: TwoTowerSessionViewState,
  bayId: number,
): TwoTowerTraceOrigin {
  if (view.latestContextTrace.status !== 'available' || !view.latestContextTrace.execution) {
    return 'persistent_cache';
  }
  const segs = view.latestContextTrace.execution.segments;
  const lastSeg = segs[segs.length - 1];
  const origin = lastSeg.getLayerTraceOrigin(bayId);
  return origin ?? 'persistent_cache';
}

function resolveContextExecutionId(view: TwoTowerSessionViewState): string {
  if (view.latestContextTrace.status === 'available' && view.latestContextTrace.execution) {
    const segs = view.latestContextTrace.execution.segments;
    return segs[segs.length - 1].stepExecutionId;
  }
  return view.identity.resultCacheDigest;
}

function resolveDenoiserExecutionId(view: TwoTowerSessionViewState): string {
  return view.denoiserLookup.pass?.passExecutionId ?? view.identity.recordId;
}

function resolveSharedExecutionId(view: TwoTowerSessionViewState): string {
  return (
    view.record.conditioning?.conditioningDigest ??
    view.denoiserLookup.conditioning?.conditioningDigest ??
    view.identity.recordId
  );
}

function clampReplayPhase(phase: number | null | undefined): number | null {
  if (phase == null || !Number.isFinite(phase)) return null;
  return Math.max(0, Math.min(1, phase));
}

function replayPhaseToBayId(phase: number): number {
  const idx = Math.min(51, Math.max(0, Math.floor(phase * 52)));
  return idx + 1;
}

function replayPhaseToWorldY(phase: number): number {
  const u = phase * 53;
  if (u <= 0) return EMBED_Y;
  if (u >= 53) return OUTPUT_LOGITS_Y;
  if (u < 1) return EMBED_Y + (bayCenterY(1) - EMBED_Y) * u;
  if (u > 52) return bayCenterY(52) + (OUTPUT_LOGITS_Y - bayCenterY(52)) * (u - 52);
  const lo = Math.floor(u);
  const hi = Math.min(52, lo + 1);
  return bayCenterY(lo) + (bayCenterY(hi) - bayCenterY(lo)) * (u - lo);
}

// ============================================================================
// MAIN DUAL-TOWER 52×2 MACRO SCENE BUILDER
// ============================================================================

/**
 * Construct the unified Dual-Tower 52×2 Macro `TwoTowerSpatialScene` from a live or rewound
 * `TwoTowerSession`. Strictly reads `record.configAtStep` (`view.modeAtRecord`) — never
 * `session.pendingConfig` — and never mutates any session or snapshot typed array.
 */
export function buildTwoTowerMacroScene(
  session: TwoTowerSession,
  options?: BuildTwoTowerMacroOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const modeAtRecord = view.modeAtRecord;
  const historicalConfig = view.historicalConfig;
  const committedPrefixLen = view.committedPrefixLen;
  const kvCapacity = view.kvCapacity;
  const activeBlock = view.activeBlock;

  const selectedTower = options?.selectedTower;
  const selectedBayId = options?.selectedBayId ?? null;
  const hoveredBayId = options?.hoveredBayId ?? null;
  const hoveredTower = options?.hoveredTower ?? null;
  const activeLinkerChannel: TwoTowerLinkerChannel = options?.activeLinkerChannel ?? 'all';
  const replayPhase = clampReplayPhase(options?.instructionalReplayPhase);
  const reducedMotion = options?.reducedMotion ?? false;

  const activeTowerFocus: TwoTowerPairedFocus =
    selectedTower === 'context'
      ? 'context_solo'
      : selectedTower === 'denoiser'
        ? 'denoiser_solo'
        : 'paired';

  const contextExecId = resolveContextExecutionId(view);
  const denoiserExecId = resolveDenoiserExecutionId(view);
  const sharedExecId = resolveSharedExecutionId(view);

  const isArMode = modeAtRecord === 'ar';
  const isMockArMode = modeAtRecord === 'mock_ar';
  const isDiffusionMode = modeAtRecord === 'mask_diffusion';
  const timestepRailActive = isDiffusionMode;

  const refinementLoopState = resolveRefinementLoopState(view);
  const blockCommitPathwayState = resolveCommitPathwayState(view);

  // Canonical stage-separated timestep status (`t = nMasked / B`, NEVER `r / K`) and
  // explicit historical Denoiser provenance (`sourceContextPrefixLen` & `sourceContextCacheDigest`).
  const ts = view.timestepStatus;
  const dl = view.denoiserLookup;
  const isHistoricalDenoiser = dl.isHistoricalInspection;

  // Active block / denoiser metrics strictly from historical record
  const activeBlockSize =
    activeBlock?.blockConfig.blockSize ??
    (view.record.kind === 'diffusion_block_commit' ? view.record.blockConfig.blockSize : null) ??
    dl.pass?.blockSize ??
    (isMockArMode ? 1 : historicalConfig.blockSize);
  const activeBlockMaskedCount = ts.currentBlockMask?.currentUnresolvedCount ?? 0;
  // Strictly derived from executed pass input mask fraction — NEVER pairs executed t with another stage's nMasked
  const timestepValue: number | null = ts.executedPass?.tModel ?? null;
  const timestepStatusLabel = ts.summaryLabel;

  // Conditioning KV length exposed by current Context boundary to Denoiser Tower
  const contextKvConditioningLen = isMockArMode
    ? Math.max(0, committedPrefixLen - 1)
    : isDiffusionMode
      ? committedPrefixLen
      : 0;
  const historicalDenoiserLabel =
    isHistoricalDenoiser && dl.pass !== null
      ? `Historical Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, ${dl.pass.passExecutionId}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'})`
      : null;

  const denoiserAbsStart =
    dl.pass?.absoluteTokenStart ??
    activeBlock?.absoluteTokenStart ??
    (isMockArMode ? Math.max(0, committedPrefixLen - 1) : committedPrefixLen);

  // Production BF16 memory reference for crisp architectural captions
  const prodMem = twoTowerMemoryFor(committedPrefixLen, activeBlockSize, modeAtRecord);

  // Fail-safe warning banner assembly (never crashes or invents traces)
  const warnings: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warnings.push(
      `Snapshot integrity violation detected (external mutation): ${view.snapshotIntegrity.error ?? 'digest mismatch'}.`,
    );
  }
  if (view.latestContextTrace.status === 'pruned_unavailable') {
    warnings.push(view.latestContextTrace.reason);
  }
  if (dl.historicalProvenanceWarning !== null) {
    warnings.push(dl.historicalProvenanceWarning);
  }
  const warningBanner = warnings.length > 0 ? warnings.join(' ') : null;

  // Geometry accumulators
  const solids: SolidSpec[] = [];
  const freeLines: LineSpec[] = [];
  const ribbons: RibbonSpec[] = [];
  const flow = new PureMacroFlowBuilder();
  const solidPickBoxes: PickBox[] = [];
  const fasciaPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const partCatalog: Record<string, CutawayPartInfo> = {};

  const channelStats: Record<ConcreteLinkerChannel, ChannelStatsAccumulator> = {
    gqa_kv: createChannelStats(),
    mamba_conv: createChannelStats(),
    mamba_ssm: createChannelStats(),
    timestep: createChannelStats(),
  };

  const pushSolid = (s: SolidSpec) => {
    solids.push(s);
    if (s.dashedCage) {
      pushDashedBoxCage3D(
        freeLines,
        s.cx,
        s.cy,
        s.cz,
        s.ex,
        s.ey,
        s.ez,
        Math.max(0.85, s.cssWidth),
        s.highlightFlag === 1 ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
        s.highlightFlag === 1 ? 0.92 : 0.56,
      );
    }
    if (s.pick) {
      const hx = s.ex * 0.5;
      const hy = s.ey * 0.5;
      const hz = s.ez * 0.5;
      solidPickBoxes.push({
        bayId: s.bayId,
        partId: s.pick.partId,
        towerRole: s.pick.towerRole,
        executionId: s.pick.executionId,
        traceOrigin: s.pick.traceOrigin,
        semanticTag: s.pick.semanticTag,
        absoluteTokenOffset: s.pick.absoluteTokenOffset,
        ...(s.pick.tokenIndex !== undefined ? { tokenIndex: s.pick.tokenIndex } : {}),
        ...(s.pick.kvHeadIndex !== undefined ? { kvHeadIndex: s.pick.kvHeadIndex } : {}),
        ...(s.pick.tensorKey !== undefined ? { tensorKey: s.pick.tensorKey } : {}),
        min: [s.cx - hx, s.cy - hy, s.cz - hz],
        max: [s.cx + hx, s.cy + hy, s.cz + hz],
      });
    }
  };

  const pushFreeLine = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth: number, style: number, alpha: number,
  ) => {
    freeLines.push({
      ax, ay, az, bx, by, bz,
      cssWidth, style, alpha,
      ownerId: LINE_FREE_OWNER_ID,
      face0: LINE_FREE_FACE,
      face1: LINE_FREE_FACE,
      ocx: 0, ocy: 0, ocz: 0,
      oex: 0, oey: 0, oez: 0,
    });
  };

  const hasFocusedBay = selectedBayId !== null || hoveredBayId !== null;
  const replayBayId = replayPhase !== null ? replayPhaseToBayId(replayPhase) : null;
  const replayWorldY = replayPhase !== null ? replayPhaseToWorldY(replayPhase) : null;

  const isTowerDimmed = (tower: 'context' | 'denoiser'): boolean => {
    if (tower === 'denoiser' && isArMode) return true;
    if (selectedTower === 'context' && tower === 'denoiser') return true;
    if (selectedTower === 'denoiser' && tower === 'context') return true;
    return false;
  };
  const ctxDimmed = isTowerDimmed('context');
  const denDimmed = isTowerDimmed('denoiser');

  // ============================================================================
  // 1. TWO RESIDUAL STREAM HIGHWAYS (LEFT x = -16.8, RIGHT x = +16.8)
  // ============================================================================

  const spineHeight = EMBED_Y - OUTPUT_LOGITS_Y;
  const spineMidY = 0.5 * (EMBED_Y + OUTPUT_LOGITS_Y);

  // Context Residual Stream (Left, x = -16.8)
  ribbons.push({
    cx: CONTEXT_RESIDUAL_X,
    cy: spineMidY,
    cz: 0,
    ex: 0.44,
    ey: spineHeight,
    ez: 0.44,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: ctxDimmed ? 0.16 : 0.42,
  });

  // Denoiser Residual Stream (Right, x = +16.8)
  ribbons.push({
    cx: DENOISER_RESIDUAL_X,
    cy: spineMidY,
    cz: 0,
    ex: 0.44,
    ey: spineHeight,
    ez: 0.44,
    kind: RIBBON_KIND_BOX,
    colorId: isDiffusionMode ? RIBBON_COLOR_PERIWINKLE : RIBBON_COLOR_EMERALD,
    alpha: denDimmed ? 0.12 : 0.40,
  });

  // Optional instructional replay wavefront capsule (strictly decoupled from measured metadata)
  if (replayWorldY !== null) {
    const pulseAlpha = reducedMotion ? 0.55 : 0.88;
    ribbons.push({
      cx: CONTEXT_RESIDUAL_X,
      cy: replayWorldY,
      cz: 0,
      ex: 0.78,
      ey: 1.9,
      ez: 0.78,
      kind: RIBBON_KIND_BOX,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: ctxDimmed ? 0.25 : pulseAlpha,
    });
    if (!isArMode) {
      ribbons.push({
        cx: DENOISER_RESIDUAL_X,
        cy: replayWorldY,
        cz: 0,
        ex: 0.78,
        ey: 1.9,
        ez: 0.78,
        kind: RIBBON_KIND_BOX,
        colorId: RIBBON_COLOR_PERIWINKLE,
        alpha: denDimmed ? 0.25 : pulseAlpha,
      });
    }
  }

  // ============================================================================
  // 2. TOP & BOTTOM BOOKENDS FOR BOTH INDEPENDENT TOWERS
  // ============================================================================

  const ctxEmbedActive =
    (selectedBayId === 0 && (selectedTower === 'context' || selectedTower === 'shared' || selectedTower === undefined)) ||
    (hoveredBayId === 0 && (hoveredTower === 'context' || hoveredTower === null));
  const denEmbedActive =
    (selectedBayId === 0 && (selectedTower === 'denoiser' || selectedTower === 'shared' || selectedTower === undefined)) ||
    (hoveredBayId === 0 && (hoveredTower === 'denoiser' || hoveredTower === null));

  // Context Embedding (W_embed_ctx, untied)
  pushSolid({
    cx: CONTEXT_TOWER_X,
    cy: EMBED_Y,
    cz: 0,
    ex: 5.0,
    ey: 1.75,
    ez: 1.05,
    cellFamily: CELL_EMBED_SLATE,
    fp8Flag: ctxDimmed ? 1 : 0,
    phase: 1 + committedPrefixLen * 0.1,
    bayId: 0,
    cellScale: 0.24,
    highlightFlag: ctxEmbedActive ? 1 : 0,
    lineStyle: ctxEmbedActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: ctxEmbedActive ? 1.75 : 1.05,
    lineAlpha: ctxEmbedActive ? 0.98 : ctxDimmed ? 0.30 : 0.62,
    pick: {
      towerRole: 'context',
      partId: 'ctx-embed',
      executionId: contextExecId,
      traceOrigin: 'persistent_cache',
      semanticTag: 'context_embed',
      absoluteTokenOffset: 0,
      tensorKey: 'w_embed_ctx',
    },
  });
  ribbons.push({
    cx: CONTEXT_TOWER_X - 3.65,
    cy: EMBED_Y,
    cz: 0,
    ex: 1.10,
    ey: 0.40,
    ez: 0.36,
    kind: RIBBON_KIND_ARROW_NEG_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: ctxDimmed ? 0.16 : 0.52,
  });

  // Denoiser Embedding (W_embed_den, independent untied weights)
  pushSolid({
    cx: DENOISER_TOWER_X,
    cy: EMBED_Y,
    cz: 0,
    ex: 5.0,
    ey: 1.75,
    ez: 1.05,
    cellFamily: CELL_WEIGHT_BLUE,
    fp8Flag: denDimmed ? 1 : 0,
    phase: 11 + activeBlockSize * 0.2,
    bayId: 0,
    cellScale: 0.24,
    highlightFlag: denEmbedActive ? 1 : 0,
    lineStyle: denEmbedActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: denEmbedActive ? 1.75 : 1.05,
    lineAlpha: denEmbedActive ? 0.98 : denDimmed ? 0.26 : 0.62,
    pick: {
      towerRole: 'denoiser',
      partId: 'den-embed',
      executionId: denoiserExecId,
      traceOrigin: 'denoiser_scratch',
      semanticTag: 'denoiser_embed',
      absoluteTokenOffset: denoiserAbsStart,
      tensorKey: 'w_embed_den',
    },
  });
  ribbons.push({
    cx: DENOISER_TOWER_X + 3.65,
    cy: EMBED_Y,
    cz: 0,
    ex: 1.10,
    ey: 0.40,
    ez: 0.36,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: denDimmed ? 0.14 : 0.52,
  });

  partCatalog['ctx-embed'] = {
    id: 'ctx-embed',
    title: 'Context Tower · W_embed_ctx (Untied)',
    badge: 'Context Tower',
    summary: `Independent Context embedding table (${view.model.contextModel.modelId}); zero weight sharing with Denoiser Tower.`,
    specs: [
      `Production shape: 131,072 × 2,688 BF16 (NoPE)`,
      `Mini runtime: V=${view.model.config.vocabSize} × d=${view.model.config.d}`,
      `Committed prefix length P = ${committedPrefixLen} / ${kvCapacity}`,
      `Fingerprint: ${view.model.contextModel.parameterFingerprint}`,
    ],
  };
  partCatalog['den-embed'] = {
    id: 'den-embed',
    title: 'Denoiser Tower · W_embed_den (Untied)',
    badge: isArMode
      ? 'Idle in AR'
      : isHistoricalDenoiser
        ? 'Historical Prior Pass'
        : 'Denoiser Tower',
    summary: `Independent Denoiser embedding table (${view.model.denoiserModel.modelId}); embeds noisy [MASK]^B block or Mock-AR token.`,
    specs: [
      `Production shape: 131,072 × 2,688 BF16 (NoPE)`,
      `Active query length: S = ${activeBlockSize} at absPos ${denoiserAbsStart}`,
      `Mode at step: ${modeAtRecord}`,
      ...(historicalDenoiserLabel
        ? [`Last Denoiser pass: ${historicalDenoiserLabel} (current Context P=${committedPrefixLen})`]
        : []),
      `Fingerprint: ${view.model.denoiserModel.parameterFingerprint}`,
    ],
  };

  // Bottom Bookends: Context Final RMSNorm, Untied LM Head, AR Output
  const ctxOutActive =
    (selectedBayId === 53 && (selectedTower === 'context' || selectedTower === 'shared' || selectedTower === undefined)) ||
    (hoveredBayId === 53 && (hoveredTower === 'context' || hoveredTower === null));
  const denOutActive =
    (selectedBayId === 53 && (selectedTower === 'denoiser' || selectedTower === 'shared' || selectedTower === undefined)) ||
    (hoveredBayId === 53 && (hoveredTower === 'denoiser' || hoveredTower === null));

  pushSolid({
    cx: CONTEXT_TOWER_X,
    cy: FINAL_NORM_Y,
    cz: 0,
    ex: 3.6,
    ey: 0.65,
    ez: 0.90,
    cellFamily: CELL_MOE_AMBER,
    fp8Flag: ctxDimmed ? 1 : 0,
    phase: 99,
    bayId: 53,
    cellScale: 0.19,
    highlightFlag: ctxOutActive ? 1 : 0,
    lineStyle: ctxOutActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: ctxOutActive ? 1.65 : 1.0,
    lineAlpha: ctxOutActive ? 0.96 : ctxDimmed ? 0.28 : 0.58,
    pick: {
      towerRole: 'context',
      partId: 'ctx-final-norm',
      executionId: contextExecId,
      traceOrigin: 'persistent_cache',
      semanticTag: 'context_final_norm',
      absoluteTokenOffset: Math.max(0, committedPrefixLen - 1),
      tensorKey: 'ctx_norm_f',
    },
  });
  pushSolid({
    cx: CONTEXT_TOWER_X,
    cy: LM_HEAD_Y,
    cz: 0,
    ex: 5.2,
    ey: 1.65,
    ez: 1.02,
    cellFamily: CELL_WEIGHT_BLUE,
    fp8Flag: ctxDimmed ? 1 : 0,
    phase: 101,
    bayId: 53,
    cellScale: 0.24,
    highlightFlag: ctxOutActive ? 1 : 0,
    lineStyle: ctxOutActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: ctxOutActive ? 1.65 : 1.0,
    lineAlpha: ctxOutActive ? 0.96 : ctxDimmed ? 0.28 : 0.58,
    pick: {
      towerRole: 'context',
      partId: 'ctx-lm-head',
      executionId: contextExecId,
      traceOrigin: 'persistent_cache',
      semanticTag: 'context_lm_head',
      absoluteTokenOffset: Math.max(0, committedPrefixLen - 1),
      tensorKey: 'ctx_lm_head',
    },
  });
  pushSolid({
    cx: CONTEXT_TOWER_X,
    cy: OUTPUT_LOGITS_Y,
    cz: 0,
    ex: 4.6,
    ey: 1.15,
    ez: 0.96,
    cellFamily: CELL_ACTIVATION_GREEN,
    fp8Flag: ctxDimmed ? 1 : 0,
    phase: committedPrefixLen + 7,
    bayId: 53,
    cellScale: 0.22,
    highlightFlag: ctxOutActive || view.record.kind === 'ar_step' ? 1 : 0,
    lineStyle: ctxOutActive || view.record.kind === 'ar_step' ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: ctxOutActive || view.record.kind === 'ar_step' ? 1.7 : 1.0,
    lineAlpha: ctxOutActive || view.record.kind === 'ar_step' ? 0.98 : ctxDimmed ? 0.28 : 0.58,
    pick: {
      towerRole: 'context',
      partId: 'ctx-output',
      executionId: contextExecId,
      traceOrigin: 'persistent_cache',
      semanticTag: 'context_ar_output',
      absoluteTokenOffset: Math.max(0, committedPrefixLen - 1),
      tensorKey: 'ctx_ar_logits',
    },
  });
  ribbons.push(
    {
      cx: CONTEXT_TOWER_X - 3.6,
      cy: FINAL_NORM_Y,
      cz: 0,
      ex: 1.1,
      ey: 0.34,
      ez: 0.32,
      kind: RIBBON_KIND_ARROW_POS_X,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: ctxDimmed ? 0.14 : 0.45,
    },
    {
      cx: CONTEXT_TOWER_X,
      cy: 0.5 * (FINAL_NORM_Y + LM_HEAD_Y),
      cz: 0,
      ex: 0.50,
      ey: 0.85,
      ez: 0.32,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: ctxDimmed ? 0.14 : 0.48,
    },
    {
      cx: CONTEXT_TOWER_X,
      cy: 0.5 * (LM_HEAD_Y + OUTPUT_LOGITS_Y),
      cz: 0,
      ex: 0.50,
      ey: 0.85,
      ez: 0.32,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: ctxDimmed ? 0.14 : 0.48,
    },
  );

  // Bottom Bookends: Denoiser Unmodulated Final RMSNorm, Untied LM Head, Unshifted Block Output & [MASK]^B Tray
  pushSolid({
    cx: DENOISER_TOWER_X,
    cy: FINAL_NORM_Y,
    cz: 0,
    ex: 3.6,
    ey: 0.65,
    ez: 0.90,
    cellFamily: CELL_MOE_AMBER,
    fp8Flag: denDimmed ? 1 : 0,
    phase: 199,
    bayId: 53,
    cellScale: 0.19,
    highlightFlag: denOutActive ? 1 : 0,
    lineStyle: denOutActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: denOutActive ? 1.65 : 1.0,
    lineAlpha: denOutActive ? 0.96 : denDimmed ? 0.24 : 0.58,
    pick: {
      towerRole: 'denoiser',
      partId: 'den-final-norm',
      executionId: denoiserExecId,
      traceOrigin: 'denoiser_scratch',
      semanticTag: 'denoiser_final_norm',
      absoluteTokenOffset: denoiserAbsStart,
      tensorKey: 'den_norm_f',
    },
  });
  pushSolid({
    cx: DENOISER_TOWER_X,
    cy: LM_HEAD_Y,
    cz: 0,
    ex: 5.2,
    ey: 1.65,
    ez: 1.02,
    cellFamily: CELL_WEIGHT_BLUE,
    fp8Flag: denDimmed ? 1 : 0,
    phase: 201,
    bayId: 53,
    cellScale: 0.24,
    highlightFlag: denOutActive ? 1 : 0,
    lineStyle: denOutActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: denOutActive ? 1.65 : 1.0,
    lineAlpha: denOutActive ? 0.96 : denDimmed ? 0.24 : 0.58,
    pick: {
      towerRole: 'denoiser',
      partId: 'den-lm-head',
      executionId: denoiserExecId,
      traceOrigin: 'denoiser_scratch',
      semanticTag: 'denoiser_lm_head',
      absoluteTokenOffset: denoiserAbsStart,
      tensorKey: 'den_lm_head',
    },
  });
  const denOutputHot =
    denOutActive ||
    view.record.kind === 'diffusion_refinement' ||
    view.record.kind === 'diffusion_confidence_commit' ||
    view.record.kind === 'mock_ar_step';
  pushSolid({
    cx: DENOISER_TOWER_X,
    cy: OUTPUT_LOGITS_Y,
    cz: 0,
    ex: 4.6,
    ey: 1.15,
    ez: 0.96,
    cellFamily: CELL_ACTIVATION_GREEN,
    fp8Flag: denDimmed ? 1 : 0,
    phase: activeBlockSize + 13,
    bayId: 53,
    cellScale: 0.22,
    highlightFlag: denOutputHot ? 1 : 0,
    lineStyle: denOutputHot ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: denOutputHot ? 1.70 : 1.0,
    lineAlpha: denOutputHot ? 0.98 : denDimmed ? 0.24 : 0.58,
    pick: {
      towerRole: 'denoiser',
      partId: 'den-output',
      executionId: denoiserExecId,
      traceOrigin: 'denoiser_scratch',
      semanticTag: 'denoiser_unshifted_output',
      absoluteTokenOffset: denoiserAbsStart,
      tensorKey: 'den_logits',
    },
  });
  ribbons.push(
    {
      cx: DENOISER_TOWER_X + 3.6,
      cy: FINAL_NORM_Y,
      cz: 0,
      ex: 1.1,
      ey: 0.34,
      ez: 0.32,
      kind: RIBBON_KIND_ARROW_NEG_X,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: denDimmed ? 0.14 : 0.45,
    },
    {
      cx: DENOISER_TOWER_X,
      cy: 0.5 * (FINAL_NORM_Y + LM_HEAD_Y),
      cz: 0,
      ex: 0.50,
      ey: 0.85,
      ez: 0.32,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: denDimmed ? 0.14 : 0.48,
    },
    {
      cx: DENOISER_TOWER_X,
      cy: 0.5 * (LM_HEAD_Y + OUTPUT_LOGITS_Y),
      cz: 0,
      ex: 0.50,
      ey: 0.85,
      ez: 0.32,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: denDimmed ? 0.14 : 0.48,
    },
  );

  // Active Block [MASK]^B Tray below Denoiser Output
  const traySlots = Math.max(1, Math.min(8, activeBlockSize));
  const traySlotPitch = Math.min(1.15, 5.2 / traySlots);
  const traySpanX = traySlots * traySlotPitch;
  const trayStartX = DENOISER_TOWER_X - traySpanX * 0.5 + traySlotPitch * 0.5;
  const maskTokenId = view.model.config.maskTokenId;
  for (let j = 0; j < traySlots; j++) {
    const tokId = activeBlock?.currentTokens[j] ?? (isDiffusionMode ? maskTokenId : -1);
    const isMaskedSlot = isDiffusionMode && activeBlock !== null && tokId === maskTokenId;
    const isUnmaskedBlockSlot = isDiffusionMode && activeBlock !== null && tokId !== maskTokenId;
    const isCompletedUncommitted = view.lifecyclePhase === 'completed_uncommitted';
    pushSolid({
      cx: trayStartX + j * traySlotPitch,
      cy: BLOCK_TRAY_Y,
      cz: 0.18,
      ex: traySlotPitch * 0.78,
      ey: 0.74,
      ez: 0.64,
      cellFamily: isUnmaskedBlockSlot || isCompletedUncommitted
        ? CELL_ACTIVATION_GREEN
        : isMaskedSlot
          ? CELL_MOE_AMBER
          : CELL_NEUTRAL,
      fp8Flag: isArMode || isMaskedSlot ? 1 : 0,
      phase: j + (activeBlock?.completedRefinements ?? 0) * 0.5,
      bayId: 53,
      cellScale: 0.18,
      highlightFlag: isCompletedUncommitted ? 2 : isUnmaskedBlockSlot ? 1 : 0,
      lineStyle: isCompletedUncommitted
        ? LINE_STYLE_BF16_GOLD
        : isUnmaskedBlockSlot
          ? LINE_STYLE_ACTIVE
          : LINE_STYLE_GUIDE,
      cssWidth: isCompletedUncommitted ? 1.55 : isUnmaskedBlockSlot ? 1.35 : 0.95,
      lineAlpha: isArMode ? 0.22 : 0.85,
      dashedCage: isMaskedSlot,
      pick: {
        towerRole: 'denoiser',
        partId: j === 0 ? 'den-block-tray' : `den-block-tray-slot-${j}`,
        executionId: denoiserExecId,
        traceOrigin: 'denoiser_scratch',
        semanticTag: isMaskedSlot ? 'denoiser_mask_slot' : 'denoiser_block_slot',
        absoluteTokenOffset: denoiserAbsStart + j,
        tokenIndex: j,
        tensorKey: 'active_block_tray',
      },
    });
  }

  partCatalog['ctx-final-norm'] = {
    id: 'ctx-final-norm',
    title: 'Context Tower · Unmodulated Final RMSNorm',
    badge: 'Context Tower',
    summary: 'Unmodulated RMSNorm (ε = 1e-5) on Context Tower final residual stream before untied LM head.',
    specs: ['No adaLN modulation on Context Tower', `Hidden size d = 2,688 (mini d = ${view.model.config.d})`],
  };
  partCatalog['ctx-lm-head'] = {
    id: 'ctx-lm-head',
    title: 'Context Tower · Untied LM Head',
    badge: 'Context Tower',
    summary: 'Independent untied LM head projecting Context hidden state to next-token AR logits.',
    specs: ['Shape: 2,688 → 131,072 BF16', `tie_word_embeddings: false`],
  };
  partCatalog['ctx-output'] = {
    id: 'ctx-output',
    title: 'Context Tower · AR Output Logits',
    badge: isArMode ? 'Active AR Output' : 'Context Output',
    summary: 'Standard shifted autoregressive next-token distribution produced by Context Tower.',
    specs: [
      `Committed prefix length P = ${committedPrefixLen}`,
      `Last committed token: "${view.contextSnapshot.lastTokenLabel}" (id=${view.contextSnapshot.lastTokenId})`,
    ],
  };
  partCatalog['den-final-norm'] = {
    id: 'den-final-norm',
    title: 'Denoiser Tower · Unmodulated Final RMSNorm',
    badge: 'Denoiser Tower',
    summary: 'Unmodulated Final RMSNorm (norm_f, ε = 1e-5) — adaLN-single modulates all 52 layers, never norm_f.',
    specs: ['Strictly unmodulated norm_f per official checkpoint', `Mode: ${modeAtRecord}`],
  };
  partCatalog['den-lm-head'] = {
    id: 'den-lm-head',
    title: 'Denoiser Tower · Untied LM Head',
    badge: 'Denoiser Tower',
    summary: 'Independent untied Denoiser LM head producing unshifted block logits [B × V].',
    specs: ['Shape: 2,688 → 131,072 BF16', 'Unshifted per-position prediction (not shifted by +1)'],
  };
  partCatalog['den-output'] = {
    id: 'den-output',
    title: 'Denoiser Tower · Unshifted Block Output',
    badge: isDiffusionMode ? 'Diffusion Block Logits' : isMockArMode ? 'Mock-AR S=1 Logits' : 'Idle in AR',
    summary: 'Unshifted Denoiser vocabulary probabilities feeding confidence-threshold unmasking or Mock-AR decode.',
    specs: [
      `Block size B = ${activeBlockSize}, masked = ${activeBlockMaskedCount}`,
      `Threshold γ = ${historicalConfig.confidenceThreshold.toFixed(2)}, temp = ${historicalConfig.temperature.toFixed(2)}`,
    ],
  };
  partCatalog['den-block-tray'] = {
    id: 'den-block-tray',
    title: `Active Block [MASK]^${activeBlockSize} Tray`,
    badge: view.lifecyclePhase === 'completed_uncommitted' ? 'Completed · Uncommitted' : view.lifecyclePhase,
    summary:
      view.lifecyclePhase === 'completed_uncommitted'
        ? `All ${activeBlockSize} block positions unmasked! Context Tower still sits at P = ${committedPrefixLen} until explicit block commit.`
        : `Holds the ${activeBlockSize}-token denoising block across iterative refinements before single-shot commit to Context Tower.`,
    specs: [
      `Masked positions: ${activeBlockMaskedCount} / ${activeBlockSize}`,
      `Lifecycle phase: ${view.lifecyclePhase}`,
      `Committed Context P = ${committedPrefixLen} (immutable during refinements)`,
    ],
  };

  // ============================================================================
  // 3. SHARED TIMESTEP-CONDITIONING HUB & VERTICAL SPINE (x = 0.0, z = -0.25)
  // ============================================================================

  const timestepHubFocused =
    selectedBayId === -1 || hoveredBayId === -1 || activeLinkerChannel === 'timestep';
  const railZ = -0.25;
  const railTopY = TIMESTEP_HUB_Y - 0.8;
  const railBottomY = BAY_52_Y;

  pushSolid({
    cx: TIMESTEP_RAIL_X,
    cy: TIMESTEP_HUB_Y,
    cz: railZ,
    ex: 6.4,
    ey: 1.45,
    ez: 0.82,
    cellFamily: timestepRailActive ? CELL_WEIGHT_BLUE : CELL_NEUTRAL,
    fp8Flag: timestepRailActive ? 0 : 1,
    phase: timestepValue !== null ? timestepValue * 10 : 0,
    bayId: -1,
    cellScale: 0.22,
    highlightFlag: timestepRailActive && timestepHubFocused ? 1 : 0,
    lineStyle: !timestepRailActive
      ? LINE_STYLE_GUIDE
      : timestepHubFocused
        ? LINE_STYLE_ACTIVE
        : LINE_STYLE_NEUTRAL,
    cssWidth: !timestepRailActive ? 0.95 : timestepHubFocused ? 1.85 : 1.15,
    lineAlpha: !timestepRailActive ? 0.34 : timestepHubFocused ? 0.98 : 0.72,
    dashedCage: !timestepRailActive,
    pick: {
      towerRole: 'shared',
      partId: 'timestep-hub',
      executionId: sharedExecId,
      traceOrigin: 'denoiser_scratch',
      semanticTag: 'timestep',
      absoluteTokenOffset: denoiserAbsStart,
      tensorKey: 't_embedder_mlp',
    },
  });

  // Vertical timestep spine down x = 0.0
  if (timestepRailActive) {
    const spineWidth = activeLinkerChannel === 'timestep' || selectedBayId === -1 ? 1.75 : 1.05;
    const spineAlpha =
      activeLinkerChannel === 'timestep' || selectedBayId === -1
        ? 0.90
        : activeLinkerChannel !== 'all'
          ? 0.14
          : 0.42;
    pushFreeLine(
      TIMESTEP_RAIL_X,
      railTopY,
      railZ,
      TIMESTEP_RAIL_X,
      railBottomY,
      railZ,
      spineWidth,
      activeLinkerChannel === 'timestep' || selectedBayId === -1
        ? LINE_STYLE_ACTIVE
        : LINE_STYLE_NEUTRAL,
      spineAlpha,
    );
    if (activeLinkerChannel === 'timestep' || selectedBayId === -1) {
      flow.addPolyline3D(
        [
          [TIMESTEP_RAIL_X, railTopY, railZ],
          [TIMESTEP_RAIL_X, railBottomY, railZ],
        ],
        [0, 0, 1],
        0.22,
        FLOW_COLOR_TIMESTEP,
        { arrowHead: false, staticPulse: reducedMotion },
      );
    }
  } else {
    // Visibly dimmed & dashed vertical spine in AR and Mock-AR ("BYPASSED (t = null)")
    pushDashedLine3D(
      freeLines,
      [TIMESTEP_RAIL_X, railTopY, railZ],
      [TIMESTEP_RAIL_X, railBottomY, railZ],
      0.85,
      LINE_STYLE_GUIDE,
      0.22,
      0.75,
      0.65,
    );
  }

  // PickBox for vertical timestep rail (strictly behind z = 0 so front cross-tower linkers never collide)
  solidPickBoxes.push({
    bayId: -1,
    partId: 'timestep-rail',
    towerRole: 'shared',
    executionId: sharedExecId,
    traceOrigin: 'denoiser_scratch',
    semanticTag: 'timestep',
    absoluteTokenOffset: denoiserAbsStart,
    tensorKey: 'timestep_rail',
    min: [-0.45, railBottomY, railZ - 0.25],
    max: [0.45, railTopY, 0.0],
  });

  const timestepHubTitle =
    ts.conditioningMode === 'bypassed_ar'
      ? 'Shared Timestep Hub · BYPASSED in AR (t = null)'
      : ts.conditioningMode === 'bypassed_mock_ar'
        ? 'Shared Timestep Hub · BYPASSED in Mock-AR (t = null)'
        : ts.conditioningMode === 'inactive_no_pass_yet'
          ? 'Shared Timestep Hub · INACTIVE (No Diffusion Pass Yet)'
          : ts.conditioningMode === 'historical_prior_pass'
            ? 'Shared Timestep Hub · HISTORICAL Prior Pass'
            : 'Shared Timestep Hub · adaLN-single (ACTIVE)';

  const timestepHubSpecs: string[] = [
    `Status: ${timestepStatusLabel}`,
    `Conditioning mode: ${ts.conditioningMode} · Block state: ${ts.blockCompletionState}`,
    `Block lifecycle: ${ts.blockCompletionNote}`,
  ];
  if (ts.executedPass !== null) {
    timestepHubSpecs.push(
      ts.executedPass.isDiagnosticOverride
        ? `Executed pass: t_override = ${ts.executedPass.overrideValue!.toFixed(4)} vs empirical ${ts.executedPass.nMaskedInput}/${ts.executedPass.blockSize} = ${ts.executedPass.empiricalMaskFraction.toFixed(4)} (Rec #${ts.executedPass.sourceRecordIndex}, b${ts.executedPass.sourceBlockId ?? '-'}, r${ts.executedPass.sourceRefinementIndex ?? '-'}, ${ts.executedPass.passExecutionId})`
        : `Executed pass: t = nMasked/B = ${ts.executedPass.nMaskedInput}/${ts.executedPass.blockSize} = ${ts.executedPass.tModel.toFixed(4)} (Rec #${ts.executedPass.sourceRecordIndex}, b${ts.executedPass.sourceBlockId ?? '-'}, r${ts.executedPass.sourceRefinementIndex ?? '-'}, ${ts.executedPass.passExecutionId})`,
    );
  }
  if (ts.nextEligibleStep !== null) {
    timestepHubSpecs.push(
      `Next eligible step: t = ${ts.nextEligibleStep.nextNMasked}/${ts.nextEligibleStep.blockSize} = ${ts.nextEligibleStep.nextTModel.toFixed(4)} (${ts.nextEligibleStep.remainingSteps} steps left)`,
    );
  }
  timestepHubSpecs.push(
    `Checkpoint params: 30,022,272 (30.02M) across 58 tensors (paper claimed ~1.5M)`,
    `Split ordering: 29 pre-norm (M/* modulate→RMSNorm) · 23 post-norm (E RMSNorm→modulate)`,
  );

  partCatalog['timestep-hub'] = {
    id: 'timestep-hub',
    title: timestepHubTitle,
    badge: timestepStatusLabel,
    summary:
      ts.conditioningMode === 'bypassed_ar' || ts.conditioningMode === 'bypassed_mock_ar'
        ? `Timestep conditioning is explicitly BYPASSED (t = null, tEmb = null) in ${modeAtRecord} mode; zero modulation applied.`
        : ts.conditioningMode === 'inactive_no_pass_yet'
          ? `Mask-diffusion conditioning is INACTIVE (no Denoiser pass has executed yet on this record). Canonical timestep t = nMasked / B (never r / K) activates on the first refinement pass.`
          : 'Computes canonical t = nMasked / B (never r / K) → sinusoidal φ(t) → MLP → 3d global modulation tEmb, broadcast down x = 0 to all 52 Denoiser scale_shift_tables[l].',
    specs: timestepHubSpecs,
  };
  partCatalog['timestep-rail'] = partCatalog['timestep-hub'];

  // ============================================================================
  // 4. 52 HORIZONTALLY PAIRED, VERTICALLY ALIGNED BAYS (1..52) + CENTRAL BRIDGE
  // ============================================================================

  const kvCapacityLength = kvCapacity * MACRO_KV_SLICE_PITCH;
  const kvTrackCenterZ = MACRO_KV_START_Z + kvCapacityLength * 0.5;
  const ephemeralKvCount = isDiffusionMode ? activeBlockSize : isMockArMode ? 1 : 0;

  for (const bay of NEMOTRON_SPEC.bays) {
    const id = bay.id;
    const padId = formatBayId(id);
    const cy = bayCenterY(id);
    const kind: Kind = bay.kind;

    const isSelectedBay = selectedBayId === id;
    const isHoveredBay = hoveredBayId === id;
    const isBayFocused = isSelectedBay || isHoveredBay;
    const isOnWave = replayBayId === id;

    const ctxSelected =
      isSelectedBay && (selectedTower === 'context' || selectedTower === 'shared' || selectedTower === undefined);
    const denSelected =
      isSelectedBay && (selectedTower === 'denoiser' || selectedTower === 'shared' || selectedTower === undefined);
    const ctxHovered = isHoveredBay && (hoveredTower === 'context' || hoveredTower === 'shared' || hoveredTower === null);
    const denHovered = isHoveredBay && (hoveredTower === 'denoiser' || hoveredTower === 'shared' || hoveredTower === null);

    const ctxActive = ctxSelected || ctxHovered || isOnWave;
    const denActive = (denSelected || denHovered || isOnWave) && !isArMode;

    const ctxOrigin = resolveContextBayOriginFast(view, id);
    const ctxLineStyle = ctxActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL;
    const ctxCssWidth = ctxSelected ? 1.80 : ctxHovered ? 1.50 : 0.95;
    const ctxLineAlpha = ctxActive ? 0.98 : ctxDimmed ? 0.28 : 0.54;

    const denLineStyle = denActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL;
    const denCssWidth = denSelected ? 1.80 : denHovered ? 1.50 : 0.95;
    const denLineAlpha = denActive ? 0.98 : denDimmed ? 0.22 : 0.54;

    // --- Context Residual Node + Pre-RMSNorm Wafer ---
    pushSolid({
      cx: CONTEXT_RESIDUAL_X,
      cy,
      cz: 0,
      ex: 0.44,
      ey: 0.88,
      ez: 0.54,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8Flag: ctxDimmed ? 1 : 0,
      phase: id + committedPrefixLen * 0.25,
      bayId: id,
      cellScale: 0.20,
      highlightFlag: ctxActive ? 1 : 0,
      lineStyle: ctxLineStyle,
      cssWidth: ctxActive ? 1.6 : 0.85,
      lineAlpha: ctxActive ? 0.98 : ctxDimmed ? 0.24 : 0.48,
      pick: {
        towerRole: 'context',
        partId: `ctx-residual-bay-${padId}`,
        executionId: contextExecId,
        traceOrigin: ctxOrigin,
        semanticTag: 'context_residual',
        absoluteTokenOffset: 0,
        tensorKey: 'residual',
      },
    });
    pushSolid({
      cx: CONTEXT_TOWER_X - 3.45,
      cy,
      cz: 0,
      ex: 0.26,
      ey: 0.84,
      ez: 0.66,
      cellFamily: CELL_MOE_AMBER,
      fp8Flag: ctxDimmed ? 1 : 0,
      phase: id,
      bayId: id,
      cellScale: 0.18,
      highlightFlag: ctxActive ? 1 : 0,
      lineStyle: ctxLineStyle,
      cssWidth: ctxCssWidth,
      lineAlpha: ctxLineAlpha,
      pick: {
        towerRole: 'context',
        partId: `ctx-norm-bay-${padId}`,
        executionId: contextExecId,
        traceOrigin: ctxOrigin,
        semanticTag: 'context_rmsnorm',
        absoluteTokenOffset: 0,
        tensorKey: 'rmsnorm',
      },
    });
    ribbons.push({
      cx: CONTEXT_TOWER_X - 4.12,
      cy,
      cz: 0,
      ex: 0.92,
      ey: 0.26,
      ez: 0.26,
      kind: RIBBON_KIND_BOX,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: ctxActive ? 0.72 : ctxDimmed ? 0.14 : 0.34,
    });

    // --- Denoiser Residual Node + Split adaLN/RMSNorm Wafer ---
    const modOrderLabel = kind === 'E' ? 'RMSNorm → modulate (post-norm)' : 'modulate → RMSNorm (pre-norm)';
    pushSolid({
      cx: DENOISER_RESIDUAL_X,
      cy,
      cz: 0,
      ex: 0.44,
      ey: 0.88,
      ez: 0.54,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8Flag: denDimmed ? 1 : 0,
      phase: id + activeBlockSize * 0.35 + 3,
      bayId: id,
      cellScale: 0.20,
      highlightFlag: denActive ? 1 : 0,
      lineStyle: denLineStyle,
      cssWidth: denActive ? 1.6 : 0.85,
      lineAlpha: denActive ? 0.98 : denDimmed ? 0.20 : 0.48,
      pick: {
        towerRole: 'denoiser',
        partId: `den-residual-bay-${padId}`,
        executionId: denoiserExecId,
        traceOrigin: 'denoiser_scratch',
        semanticTag: 'denoiser_residual',
        absoluteTokenOffset: denoiserAbsStart,
        tensorKey: 'residual',
      },
    });
    pushSolid({
      cx: DENOISER_TOWER_X + 3.45,
      cy,
      cz: 0,
      ex: 0.26,
      ey: 0.84,
      ez: 0.66,
      cellFamily: CELL_MOE_AMBER,
      fp8Flag: denDimmed ? 1 : 0,
      phase: id + 52,
      bayId: id,
      cellScale: 0.18,
      highlightFlag: denActive ? 1 : 0,
      lineStyle: denLineStyle,
      cssWidth: denCssWidth,
      lineAlpha: denLineAlpha,
      pick: {
        towerRole: 'denoiser',
        partId: `den-norm-bay-${padId}`,
        executionId: denoiserExecId,
        traceOrigin: 'denoiser_scratch',
        semanticTag: kind === 'E' ? 'denoiser_post_norm_adaln' : 'denoiser_pre_norm_adaln',
        absoluteTokenOffset: denoiserAbsStart,
        tensorKey: 'rmsnorm',
      },
    });
    ribbons.push({
      cx: DENOISER_TOWER_X + 4.12,
      cy,
      cz: 0,
      ex: 0.92,
      ey: 0.26,
      ez: 0.26,
      kind: RIBBON_KIND_BOX,
      colorId: isDiffusionMode ? RIBBON_COLOR_PERIWINKLE : RIBBON_COLOR_EMERALD,
      alpha: denActive ? 0.72 : denDimmed ? 0.12 : 0.32,
    });

    // --- Timestep Horizontal Tap (x = 0.0 -> x = +7.5, z = -0.25) into ALL 52 Denoiser bays ---
    const tsEmphasis = resolveChannelEmphasis({
      channel: 'timestep',
      bayId: id,
      filter: activeLinkerChannel,
      isBayFocused: isBayFocused || selectedBayId === -1,
      hasFocusedBay: hasFocusedBay || selectedBayId === -1,
      modeAtRecord,
      isOnReplayWave: isOnWave,
    });
    recordChannelStat(channelStats.timestep, tsEmphasis);

    if (tsEmphasis.dashed) {
      pushDashedLine3D(
        freeLines,
        [TIMESTEP_RAIL_X, cy, railZ],
        [BRIDGE_RIGHT_X, cy, railZ],
        tsEmphasis.cssWidth,
        tsEmphasis.lineStyle,
        tsEmphasis.alpha,
        0.42,
        0.34,
      );
    } else {
      pushFreeLine(
        TIMESTEP_RAIL_X,
        cy,
        railZ,
        BRIDGE_RIGHT_X,
        cy,
        railZ,
        tsEmphasis.cssWidth,
        tsEmphasis.lineStyle,
        tsEmphasis.alpha,
      );
      if (tsEmphasis.emitFlowRibbon) {
        flow.addPolyline3D(
          [
            [TIMESTEP_RAIL_X, cy, railZ],
            [BRIDGE_RIGHT_X, cy, railZ],
          ],
          [0, 1, 0],
          0.14,
          { ...FLOW_COLOR_TIMESTEP, a: tsEmphasis.alpha },
          { arrowHead: true, arrowLength: 0.34, arrowWidth: 0.34, staticPulse: reducedMotion },
        );
      }
    }

    partCatalog[`timestep-tap-bay-${padId}`] = {
      id: `timestep-tap-bay-${padId}`,
      title: `Timestep Tap · Bay ${padId} (${kind === 'E' ? 'Post-Norm MoE' : kind === '*' ? 'Pre-Norm GQA' : 'Pre-Norm Mamba-2'})`,
      badge: timestepRailActive ? 'adaLN-single Tap' : 'BYPASSED (t = null)',
      summary: timestepRailActive
        ? `Injects scale_shift_tables[${id - 1}] + tEmb into Denoiser Bay ${padId} using ${modOrderLabel} + residual gate.`
        : `Timestep tap into Denoiser Bay ${padId} is BYPASSED (t = null) in ${modeAtRecord} mode.`,
      specs: [
        `Layer index: ${id - 1} (Bay ${padId}) · Mixer: ${kind}`,
        `Ordering: ${modOrderLabel}`,
        `Status: ${timestepStatusLabel}`,
      ],
    };

    // --- Mixer-Specific Geometry (Mamba-2 / MoE / GQA) + Cross-Tower Alignment Linkers ---
    if (kind === 'M') {
      // 1. CONTEXT MAMBA-2 BODY + SOLID DUAL-COMPARTMENT FIXED BOUNDARY SLABS (Conv1D + FP32 SSM H)
      pushSolid({
        cx: CONTEXT_TOWER_X - 1.45,
        cy,
        cz: 0,
        ex: 2.85,
        ey: 0.90,
        ez: 0.84,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id * 1.13,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: ctxActive ? 1 : 0,
        lineStyle: ctxLineStyle,
        cssWidth: ctxCssWidth,
        lineAlpha: ctxLineAlpha,
        pick: {
          towerRole: 'context',
          partId: `ctx-mamba-body-${padId}`,
          executionId: contextExecId,
          traceOrigin: ctxOrigin,
          semanticTag: 'context_mamba_body',
          absoluteTokenOffset: 0,
          tensorKey: 'mamba_body',
        },
      });

      // Context Solid Persistent Compartment A: Conv1D 4-tap history (`ctx-conv-bay-XX`)
      const convHot = ctxActive || activeLinkerChannel === 'mamba_conv';
      pushSolid({
        cx: CONTEXT_TOWER_X + 2.85,
        cy: cy + 0.24,
        cz: 0.14,
        ex: 1.70,
        ey: 0.42,
        ez: 0.92,
        cellFamily: CELL_MOE_AMBER,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id * 0.7 + committedPrefixLen,
        bayId: id,
        cellScale: 0.18,
        highlightFlag: convHot ? 1 : 0,
        lineStyle: convHot ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
        cssWidth: convHot ? 1.75 : 1.05,
        lineAlpha: convHot ? 0.98 : ctxDimmed ? 0.30 : 0.72,
        pick: {
          towerRole: 'context',
          partId: `ctx-conv-bay-${padId}`,
          executionId: contextExecId,
          traceOrigin: 'persistent_cache',
          semanticTag: 'mamba_conv',
          absoluteTokenOffset: Math.max(0, committedPrefixLen - 1),
          tensorKey: 'conv_state',
        },
      });

      // Context Solid Persistent Compartment B: FP32 SSM H state (`ctx-ssm-bay-XX`)
      const ssmHot = ctxActive || activeLinkerChannel === 'mamba_ssm';
      pushSolid({
        cx: CONTEXT_TOWER_X + 2.85,
        cy: cy - 0.24,
        cz: 0.14,
        ex: 1.70,
        ey: 0.46,
        ez: 0.96,
        cellFamily: CELL_SSM_CYAN,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id * 1.37 + committedPrefixLen,
        bayId: id,
        cellScale: 0.18,
        highlightFlag: ssmHot ? 1 : 0,
        lineStyle: ssmHot ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        cssWidth: ssmHot ? 1.75 : 1.05,
        lineAlpha: ssmHot ? 0.98 : ctxDimmed ? 0.30 : 0.72,
        pick: {
          towerRole: 'context',
          partId: `ctx-ssm-bay-${padId}`,
          executionId: contextExecId,
          traceOrigin: 'persistent_cache',
          semanticTag: 'mamba_ssm',
          absoluteTokenOffset: Math.max(0, committedPrefixLen - 1),
          tensorKey: 'ssm_state',
        },
      });

      // 2. DENOISER MAMBA-2 BODY + DASHED/FROSTED EPHEMERAL SCRATCHPAD SLABS
      pushSolid({
        cx: DENOISER_TOWER_X + 1.45,
        cy,
        cz: 0,
        ex: 2.85,
        ey: 0.90,
        ez: 0.84,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: denDimmed ? 1 : 0,
        phase: id * 1.13 + 77,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: denActive ? 1 : 0,
        lineStyle: denLineStyle,
        cssWidth: denCssWidth,
        lineAlpha: denLineAlpha,
        pick: {
          towerRole: 'denoiser',
          partId: `den-mamba-body-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_mamba_body',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'mamba_body',
        },
      });

      // Ephemeral Denoiser Mamba Conv1D scratchpad (pale frosted + dashed guide cage)
      pushSolid({
        cx: DENOISER_TOWER_X - 2.85,
        cy: cy + 0.24,
        cz: 0.14,
        ex: 1.50,
        ey: 0.36,
        ez: 0.44,
        cellFamily: CELL_MOE_AMBER,
        fp8Flag: 1, // Frosted ephemeral scratchpad visual language
        phase: id * 0.7 + activeBlockSize + 41,
        bayId: id,
        cellScale: 0.19,
        highlightFlag: denActive || activeLinkerChannel === 'mamba_conv' ? 1 : 0,
        lineStyle: LINE_STYLE_GUIDE,
        cssWidth: 0.85,
        lineAlpha: denDimmed ? 0.18 : 0.46,
        dashedCage: true,
        pick: {
          towerRole: 'denoiser',
          partId: `den-scratch-conv-bay-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_ephemeral_mamba_conv',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'conv_scratch',
        },
      });

      // Ephemeral Denoiser Mamba SSM scratchpad (pale frosted + dashed guide cage)
      pushSolid({
        cx: DENOISER_TOWER_X - 2.85,
        cy: cy - 0.24,
        cz: 0.14,
        ex: 1.50,
        ey: 0.40,
        ez: 0.46,
        cellFamily: CELL_SSM_CYAN,
        fp8Flag: 1, // Frosted ephemeral scratchpad visual language
        phase: id * 1.37 + activeBlockSize + 41,
        bayId: id,
        cellScale: 0.19,
        highlightFlag: denActive || activeLinkerChannel === 'mamba_ssm' ? 1 : 0,
        lineStyle: LINE_STYLE_GUIDE,
        cssWidth: 0.85,
        lineAlpha: denDimmed ? 0.18 : 0.46,
        dashedCage: true,
        pick: {
          towerRole: 'denoiser',
          partId: `den-scratch-ssm-bay-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_ephemeral_mamba_ssm',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'ssm_scratch',
        },
      });

      // 3. TWO SEPARATE MAMBA CROSS-TOWER LINKERS: mamba_conv (cy + 0.18) & mamba_ssm (cy - 0.18)
      const convEmph = resolveChannelEmphasis({
        channel: 'mamba_conv',
        bayId: id,
        filter: activeLinkerChannel,
        isBayFocused,
        hasFocusedBay,
        modeAtRecord,
        isOnReplayWave: isOnWave,
      });
      recordChannelStat(channelStats.mamba_conv, convEmph);

      const convY = cy + MAMBA_CONV_LANE_DY;
      pushFreeLine(
        BRIDGE_LEFT_X,
        convY,
        0.14,
        BRIDGE_RIGHT_X,
        convY,
        0.14,
        convEmph.cssWidth,
        convEmph.lineStyle,
        convEmph.alpha,
      );
      if (convEmph.emitFlowRibbon) {
        flow.addPolyline3D(
          [
            [BRIDGE_LEFT_X, convY, 0.14],
            [BRIDGE_RIGHT_X, convY, 0.14],
          ],
          [0, 1, 0],
          0.16,
          { ...FLOW_COLOR_MAMBA_CONV, a: convEmph.alpha },
          { arrowHead: true, arrowLength: 0.42, arrowWidth: 0.36, staticPulse: reducedMotion },
        );
      }
      solidPickBoxes.push({
        bayId: id,
        partId: `linker-mamba-conv-bay-${padId}`,
        towerRole: 'shared',
        executionId: sharedExecId,
        traceOrigin: 'persistent_cache',
        semanticTag: 'mamba_conv',
        absoluteTokenOffset: Math.max(0, isMockArMode ? committedPrefixLen - 2 : committedPrefixLen - 1),
        tensorKey: 'mamba_conv_linker',
        min: [BRIDGE_LEFT_X, convY - 0.11, 0.02],
        max: [BRIDGE_RIGHT_X, convY + 0.11, 0.26],
      });

      const ssmEmph = resolveChannelEmphasis({
        channel: 'mamba_ssm',
        bayId: id,
        filter: activeLinkerChannel,
        isBayFocused,
        hasFocusedBay,
        modeAtRecord,
        isOnReplayWave: isOnWave,
      });
      recordChannelStat(channelStats.mamba_ssm, ssmEmph);

      const ssmY = cy + MAMBA_SSM_LANE_DY;
      pushFreeLine(
        BRIDGE_LEFT_X,
        ssmY,
        0.14,
        BRIDGE_RIGHT_X,
        ssmY,
        0.14,
        ssmEmph.cssWidth,
        ssmEmph.lineStyle,
        ssmEmph.alpha,
      );
      if (ssmEmph.emitFlowRibbon) {
        flow.addPolyline3D(
          [
            [BRIDGE_LEFT_X, ssmY, 0.14],
            [BRIDGE_RIGHT_X, ssmY, 0.14],
          ],
          [0, 1, 0],
          0.17,
          { ...FLOW_COLOR_MAMBA_SSM, a: ssmEmph.alpha },
          { arrowHead: true, arrowLength: 0.42, arrowWidth: 0.36, staticPulse: reducedMotion },
        );
      }
      solidPickBoxes.push({
        bayId: id,
        partId: `linker-mamba-ssm-bay-${padId}`,
        towerRole: 'shared',
        executionId: sharedExecId,
        traceOrigin: 'persistent_cache',
        semanticTag: 'mamba_ssm',
        absoluteTokenOffset: Math.max(0, isMockArMode ? committedPrefixLen - 2 : committedPrefixLen - 1),
        tensorKey: 'mamba_ssm_linker',
        min: [BRIDGE_LEFT_X, ssmY - 0.11, 0.02],
        max: [BRIDGE_RIGHT_X, ssmY + 0.11, 0.26],
      });

      // Part catalog metadata
      const seedLabel = isMockArMode ? `mambaS2 (P-1 = ${Math.max(0, committedPrefixLen - 1)})` : `S-1 boundary (P = ${committedPrefixLen})`;
      const denMambaCondLabel =
        isHistoricalDenoiser && dl.sourceContextPrefixLen !== null
          ? `Historical pass read H_${Math.max(0, dl.sourceContextPrefixLen - 1)} (P_cond=${dl.sourceContextPrefixLen}, Rec #${dl.sourceRecordIndex}, b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}) ≠ current Context P=${committedPrefixLen}`
          : `Conditioned on current Context ${seedLabel}`;
      const miniMambaInv = TWOTOWER_MINI_MODEL_INVENTORY.mamba;
      partCatalog[`ctx-mamba-body-${padId}`] = {
        id: `ctx-mamba-body-${padId}`,
        title: `Context Bay ${padId} · Mamba-2 (Persistent Boundary Owner)`,
        badge: 'Context Tower · Mamba-2',
        summary: `Context Tower Mamba-2 layer ${id - 1} maintaining fixed-size Conv1D 4-tap + FP32 SSM H boundary slabs (0 B/tok growth).`,
        specs: [
          `Model: ${view.model.contextModel.modelId}`,
          `Miniature Mamba-2 inventory: ${miniMambaInv.heads} heads, ${miniMambaInv.groups} groups, ${miniMambaInv.dHead}×${miniMambaInv.dState} SSM state (prod 64 heads / 8 groups / 64×128 SSM)`,
          `Seeded boundary: ${seedLabel}`,
          `Origin: ${ctxOrigin}`,
        ],
      };
      partCatalog[`den-mamba-body-${padId}`] = {
        id: `den-mamba-body-${padId}`,
        title: `Denoiser Bay ${padId} · Mamba-2 (Ephemeral Scratch)`,
        badge: isArMode
          ? 'Idle in AR'
          : isHistoricalDenoiser
            ? 'Historical Prior Pass'
            : 'Denoiser · Pre-Norm adaLN',
        summary: `Denoiser Tower Mamba-2 layer ${id - 1} using pre-norm adaLN ordering (modulate → RMSNorm) over ephemeral scratch state.`,
        specs: [
          `Model: ${view.model.denoiserModel.modelId}`,
          `Miniature Mamba-2 inventory: ${miniMambaInv.heads} heads, ${miniMambaInv.groups} groups, ${miniMambaInv.dHead}×${miniMambaInv.dState} SSM state`,
          denMambaCondLabel,
          `adaLN ordering: modulate → RMSNorm (pre-norm)`,
          `Origin: denoiser_scratch`,
        ],
      };
      partCatalog[`ctx-conv-bay-${padId}`] = {
        id: `ctx-conv-bay-${padId}`,
        title: `Context Bay ${padId} · Persistent Conv1D 4-Tap History`,
        badge: 'Persistent Cache · BF16',
        summary: 'Fixed-size depthwise Conv1D 4-tap boundary buffer owned exclusively by Context Tower (0 B/tok growth).',
        specs: [
          `Production: 6,144 × 4 BF16 (48 KiB/bay) · Mini: ${miniMambaInv.convDim} × ${miniMambaInv.convKernel} FP32`,
          `Seeded boundary: ${seedLabel}`,
          'Origin: persistent_cache',
        ],
      };
      partCatalog[`ctx-ssm-bay-${padId}`] = {
        id: `ctx-ssm-bay-${padId}`,
        title: `Context Bay ${padId} · Persistent FP32 SSM H State`,
        badge: 'Persistent Cache · FP32',
        summary: 'Fixed-size recurrent FP32 SSM H matrix owned exclusively by Context Tower (0 B/tok growth).',
        specs: [
          `Production: 64 × [64×128] FP32 (2.0 MiB/bay) · Mini: ${miniMambaInv.heads} heads × [${miniMambaInv.dHead}×${miniMambaInv.dState}] across ${miniMambaInv.groups} groups`,
          `Seeded boundary: ${seedLabel}`,
          'Origin: persistent_cache',
        ],
      };
      partCatalog[`den-scratch-conv-bay-${padId}`] = {
        id: `den-scratch-conv-bay-${padId}`,
        title: `Denoiser Bay ${padId} · Ephemeral Conv1D Scratchpad`,
        badge: isHistoricalDenoiser ? 'Historical Scratch · Prior Pass' : 'Ephemeral Scratch · Discarded',
        summary: 'Transient Denoiser Conv1D buffer freshly seeded from Context before each refinement and discarded immediately after.',
        specs: [denMambaCondLabel, 'Origin: denoiser_scratch', 'NEVER overwrites Context persistent boundary during refinements'],
      };
      partCatalog[`den-scratch-ssm-bay-${padId}`] = {
        id: `den-scratch-ssm-bay-${padId}`,
        title: `Denoiser Bay ${padId} · Ephemeral SSM Scratchpad`,
        badge: isHistoricalDenoiser ? 'Historical Scratch · Prior Pass' : 'Ephemeral Scratch · Discarded',
        summary: 'Transient Denoiser FP32 SSM scratch state freshly seeded from Context and discarded after each refinement pass.',
        specs: [denMambaCondLabel, 'Origin: denoiser_scratch', 'Discarded between refinements; Context updates once on block commit'],
      };
      partCatalog[`linker-mamba-conv-bay-${padId}`] = {
        id: `linker-mamba-conv-bay-${padId}`,
        title: `Cross-Tower Linker · Bay ${padId} Mamba Conv1D (4-tap)`,
        badge: isMockArMode ? 'Mock-AR mambaS2' : isDiffusionMode ? 'Diffusion S-1 Seed' : 'Dormant in AR',
        summary: `Read-only Context→Denoiser handoff of raw 4-tap Conv1D history at Bay ${padId} (layer ${id - 1}).`,
        specs: [
          `Upper sub-lane (+0.18 Y)`,
          `Current Context boundary: ${seedLabel}`,
          ...(isHistoricalDenoiser ? [denMambaCondLabel] : []),
          'ownerRole: context_tower → consumerRole: denoiser_tower',
        ],
      };
      partCatalog[`linker-mamba-ssm-bay-${padId}`] = {
        id: `linker-mamba-ssm-bay-${padId}`,
        title: `Cross-Tower Linker · Bay ${padId} Mamba FP32 SSM H`,
        badge: isMockArMode ? 'Mock-AR mambaS2' : isDiffusionMode ? 'Diffusion S-1 Seed' : 'Dormant in AR',
        summary: `Read-only Context→Denoiser handoff of persistent FP32 SSM H state at Bay ${padId} (layer ${id - 1}).`,
        specs: [
          `Lower sub-lane (-0.18 Y)`,
          `Current Context boundary: ${seedLabel}`,
          ...(isHistoricalDenoiser ? [denMambaCondLabel] : []),
          'ownerRole: context_tower → consumerRole: denoiser_tower',
        ],
      };
    } else if (kind === '*') {
      // 1. CONTEXT GQA BODY + SOLID PERSISTENT KV PREFIX CACHE (0..P-1) + 24-SLOT GHOST CAPACITY RAIL
      // O-proj plate sits cleanly left of the +Z KV rails (`x ∈ [-15.32, -13.38]`) for unobstructed front picking
      pushSolid({
        cx: CONTEXT_TOWER_X - 2.35,
        cy,
        cz: -0.02,
        ex: 1.95,
        ey: 1.22,
        ez: 0.92,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id * 1.7,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: ctxActive ? 1 : 0,
        lineStyle: ctxLineStyle,
        cssWidth: ctxActive ? 1.85 : 1.10,
        lineAlpha: ctxActive ? 0.98 : ctxDimmed ? 0.30 : 0.64,
        pick: {
          towerRole: 'context',
          partId: `ctx-gqa-body-${padId}`,
          executionId: contextExecId,
          traceOrigin: ctxOrigin,
          semanticTag: 'context_gqa_body',
          absoluteTokenOffset: 0,
          tensorKey: 'o_proj',
        },
      });
      // Q/K/V projection stack on bridge side of Context GQA bay
      pushSolid({
        cx: CONTEXT_TOWER_X + 1.55,
        cy,
        cz: -0.10,
        ex: 2.35,
        ey: 1.18,
        ez: 0.86,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id * 2.1,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: ctxActive || activeLinkerChannel === 'gqa_kv' ? 1 : 0,
        lineStyle: ctxActive || activeLinkerChannel === 'gqa_kv' ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        cssWidth: ctxActive || activeLinkerChannel === 'gqa_kv' ? 1.80 : 1.05,
        lineAlpha: ctxActive || activeLinkerChannel === 'gqa_kv' ? 0.98 : ctxDimmed ? 0.30 : 0.62,
        pick: {
          towerRole: 'context',
          partId: `ctx-gqa-qkv-${padId}`,
          executionId: contextExecId,
          traceOrigin: ctxOrigin,
          semanticTag: 'context_gqa_qkv',
          absoluteTokenOffset: 0,
          tensorKey: 'qkv_proj',
        },
      });

      // Context Persistent KV Capacity Rails (`kvCapacity = 24`) + Solid Populated Slices (`0..P-1`) + Ghost Capacity Slots (`P..kvCapacity-1`)
      const clampedPrefix = Math.min(kvCapacity, committedPrefixLen);
      for (let r = 0; r < MACRO_KV_RAIL_OFFSETS_X.length; r++) {
        const rx = CONTEXT_TOWER_X + MACRO_KV_RAIL_OFFSETS_X[r];
        const isK = r % 2 === 0;
        const kvHead: 0 | 1 = r < 2 ? 0 : 1;

        // Full 24-slot capacity base rail (`persistent_cache`)
        pushSolid({
          cx: rx,
          cy: cy - 0.34,
          cz: kvTrackCenterZ,
          ex: 0.09,
          ey: 0.08,
          ez: kvCapacityLength,
          cellFamily: CELL_NEUTRAL,
          fp8Flag: 0,
          phase: 0,
          bayId: id,
          cellScale: 0.5,
          highlightFlag: 0,
          lineStyle: LINE_STYLE_GUIDE,
          cssWidth: ctxActive || activeLinkerChannel === 'gqa_kv' ? 1.0 : 0.75,
          lineAlpha: ctxActive || activeLinkerChannel === 'gqa_kv' ? 0.68 : 0.38,
          pick: {
            towerRole: 'context',
            partId: `ctx-kv-capacity-bay-${padId}-r${r}`,
            executionId: contextExecId,
            traceOrigin: 'persistent_cache',
            semanticTag: 'gqa_kv_capacity',
            absoluteTokenOffset: 0,
            kvHeadIndex: kvHead,
            tensorKey: isK ? 'kv_k_capacity' : 'kv_v_capacity',
          },
        });

        // Solid populated persistent KV prefix slices `0..P-1`
        for (let p = 0; p < clampedPrefix; p++) {
          const isMockArExcludedLast = isMockArMode && p === clampedPrefix - 1;
          const isNewest = p === clampedPrefix - 1;
          const hotSlice = (ctxActive || activeLinkerChannel === 'gqa_kv') && isNewest;
          pushSolid({
            cx: rx,
            cy: cy - 0.03,
            cz: macroKvSliceCenterZ(p),
            ex: 0.34,
            ey: 0.50,
            ez: MACRO_KV_SLICE_DEPTH,
            cellFamily: isK ? CELL_ACTIVATION_GREEN : CELL_SSM_CYAN,
            fp8Flag: ctxDimmed || isMockArExcludedLast ? 1 : 0,
            phase: p + id * 0.17 + r * 0.5,
            bayId: id,
            cellScale: 0.16,
            highlightFlag: hotSlice ? 1 : isMockArExcludedLast ? 2 : 0,
            lineStyle: hotSlice
              ? LINE_STYLE_ACTIVE
              : isMockArExcludedLast
                ? LINE_STYLE_BF16_GOLD
                : LINE_STYLE_NEUTRAL,
            cssWidth: hotSlice ? 1.50 : 0.85,
            lineAlpha: hotSlice ? 0.96 : ctxDimmed ? 0.28 : 0.56,
            pick: {
              towerRole: 'context',
              partId: `ctx-kv-slice-bay-${padId}-r${r}-p${p}`,
              executionId: contextExecId,
              traceOrigin: 'persistent_cache',
              semanticTag: isMockArExcludedLast ? 'gqa_kv_excluded_mock_ar_last' : 'gqa_kv',
              absoluteTokenOffset: p,
              tokenIndex: 0,
              kvHeadIndex: kvHead,
              tensorKey: isK ? 'kv_k' : 'kv_v',
            },
          });
        }
      }

      // Ghost capacity slot tick frames along `P .. kvCapacity - 1` so all 24 allocated slots read clearly
      const ghostLeftX = CONTEXT_TOWER_X + MACRO_KV_RAIL_OFFSETS_X[0] - 0.18;
      const ghostRightX = CONTEXT_TOWER_X + MACRO_KV_RAIL_OFFSETS_X[3] + 0.18;
      const ghostY = cy - 0.34;
      for (let slot = clampedPrefix; slot < kvCapacity; slot++) {
        const zFront = macroKvSliceFrontZ(slot);
        const isMajorTick = slot === kvCapacity - 1 || (slot + 1) % 4 === 0;
        pushFreeLine(
          ghostLeftX,
          ghostY,
          zFront,
          ghostRightX,
          ghostY,
          zFront,
          isMajorTick ? 0.95 : 0.70,
          LINE_STYLE_GUIDE,
          isMajorTick ? 0.42 : 0.22,
        );
      }

      // 2. DENOISER GQA BODY + EPHEMERAL BLOCK KV SCRATCHPAD (`B` tokens in diffusion, `1` in mock_ar)
      pushSolid({
        cx: DENOISER_TOWER_X + 2.15,
        cy,
        cz: -0.02,
        ex: 1.85,
        ey: 1.22,
        ez: 0.92,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: denDimmed ? 1 : 0,
        phase: id * 1.7 + 91,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: denActive ? 1 : 0,
        lineStyle: denLineStyle,
        cssWidth: denActive ? 1.85 : 1.10,
        lineAlpha: denActive ? 0.98 : denDimmed ? 0.22 : 0.62,
        pick: {
          towerRole: 'denoiser',
          partId: `den-gqa-body-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_gqa_body',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'o_proj',
        },
      });
      pushSolid({
        cx: DENOISER_TOWER_X - 1.55,
        cy,
        cz: -0.10,
        ex: 2.35,
        ey: 1.18,
        ez: 0.86,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: denDimmed ? 1 : 0,
        phase: id * 2.1 + 91,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: denActive || activeLinkerChannel === 'gqa_kv' ? 1 : 0,
        lineStyle: denActive || activeLinkerChannel === 'gqa_kv' ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        cssWidth: denActive || activeLinkerChannel === 'gqa_kv' ? 1.80 : 1.05,
        lineAlpha: denActive || activeLinkerChannel === 'gqa_kv' ? 0.98 : denDimmed ? 0.22 : 0.60,
        pick: {
          towerRole: 'denoiser',
          partId: `den-gqa-qkv-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_gqa_qkv',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'qkv_proj',
        },
      });

      // Ephemeral Denoiser Block KV Scratchpad (pale frosted + dashed guide cage, length = B or 1, discarded after pass!)
      if (ephemeralKvCount > 0) {
        const scratchDepth = ephemeralKvCount * MACRO_KV_SLICE_PITCH;
        const scratchCenterZ = MACRO_KV_START_Z + scratchDepth * 0.5;
        // Dashed envelope cage around the entire ephemeral block KV scratchpad
        pushDashedBoxCage3D(
          freeLines,
          DENOISER_TOWER_X,
          cy - 0.04,
          scratchCenterZ,
          2.55,
          0.54,
          scratchDepth + 0.08,
          denActive || activeLinkerChannel === 'gqa_kv' ? 1.25 : 0.85,
          denActive || activeLinkerChannel === 'gqa_kv' ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
          denDimmed ? 0.18 : 0.62,
        );
        for (let r = 0; r < 2; r++) {
          const rx = DENOISER_TOWER_X + (r === 0 ? -0.65 : 0.65);
          for (let j = 0; j < ephemeralKvCount; j++) {
            pushSolid({
              cx: rx,
              cy: cy - 0.05,
              cz: macroKvSliceCenterZ(j),
              ex: 0.44,
              ey: 0.40,
              ez: MACRO_KV_SLICE_DEPTH * 0.86,
              cellFamily: r === 0 ? CELL_ACTIVATION_GREEN : CELL_SSM_CYAN,
              fp8Flag: 1, // Frosted ephemeral scratchpad — visually distinct from Context persistent KV!
              phase: j + id * 0.23 + r * 0.7,
              bayId: id,
              cellScale: 0.17,
              highlightFlag: denActive ? 1 : 0,
              lineStyle: LINE_STYLE_GUIDE,
              cssWidth: 0.80,
              lineAlpha: denDimmed ? 0.16 : 0.42,
              pick: {
                towerRole: 'denoiser',
                partId: `den-scratch-kv-bay-${padId}-h${r}-j${j}`,
                executionId: denoiserExecId,
                traceOrigin: 'denoiser_scratch',
                semanticTag: 'denoiser_ephemeral_kv',
                absoluteTokenOffset: denoiserAbsStart + j,
                tokenIndex: j,
                kvHeadIndex: r as 0 | 1,
                tensorKey: r === 0 ? 'scratch_kv_0' : 'scratch_kv_1',
              },
            });
          }
        }
      }

      // 3. GQA KV CROSS-TOWER LOGICAL PREFIX LINKER (`6` bays: {6, 13, 20, 27, 34, 43})
      const gqaEmph = resolveChannelEmphasis({
        channel: 'gqa_kv',
        bayId: id,
        filter: activeLinkerChannel,
        isBayFocused,
        hasFocusedBay,
        modeAtRecord,
        isOnReplayWave: isOnWave,
      });
      recordChannelStat(channelStats.gqa_kv, gqaEmph);

      pushFreeLine(
        BRIDGE_LEFT_X,
        cy,
        0.28,
        BRIDGE_RIGHT_X,
        cy,
        0.28,
        gqaEmph.cssWidth,
        gqaEmph.lineStyle,
        gqaEmph.alpha,
      );
      if (gqaEmph.emitFlowRibbon) {
        flow.addPolyline3D(
          [
            [BRIDGE_LEFT_X, cy, 0.28],
            [BRIDGE_RIGHT_X, cy, 0.28],
          ],
          [0, 1, 0],
          0.22,
          { ...FLOW_COLOR_GQA_KV, a: gqaEmph.alpha },
          { arrowHead: true, arrowLength: 0.48, arrowWidth: 0.44, staticPulse: reducedMotion },
        );
        ribbons.push({
          cx: CONTEXT_TOWER_X,
          cy: cy + 0.24,
          cz: macroKvSliceCenterZ(Math.max(0, clampedPrefix - 1)),
          ex: 0.50,
          ey: 0.36,
          ez: 0.85,
          kind: RIBBON_KIND_ARROW_POS_Z,
          colorId: RIBBON_COLOR_CYAN,
          alpha: 0.72,
        });
      }
      solidPickBoxes.push({
        bayId: id,
        partId: `linker-gqa-kv-bay-${padId}`,
        towerRole: 'shared',
        executionId: sharedExecId,
        traceOrigin: 'persistent_cache',
        semanticTag: 'gqa_kv',
        absoluteTokenOffset: 0,
        tensorKey: 'gqa_kv_linker',
        min: [BRIDGE_LEFT_X, cy - 0.22, 0.12],
        max: [BRIDGE_RIGHT_X, cy + 0.22, 0.44],
      });

      const kvRangeLabel = isMockArMode
        ? `KV[0..${Math.max(0, committedPrefixLen - 2)}] (P-1 = ${contextKvConditioningLen} tokens; excludes last token)`
        : `KV[0..${Math.max(0, committedPrefixLen - 1)}] (P = ${committedPrefixLen} tokens)`;
      const denKvCondLabel =
        isHistoricalDenoiser && dl.sourceContextPrefixLen !== null
          ? `Historical pass read KV[0..${Math.max(0, dl.sourceContextPrefixLen - 1)}] (P_cond=${dl.sourceContextPrefixLen}, Rec #${dl.sourceRecordIndex}, b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}) — does NOT match current Context P=${committedPrefixLen}`
          : `Conditioned prefix: ${kvRangeLabel}`;
      const miniGqaInv = TWOTOWER_MINI_MODEL_INVENTORY.gqa;

      partCatalog[`ctx-gqa-body-${padId}`] = {
        id: `ctx-gqa-body-${padId}`,
        title: `Context Bay ${padId} · GQA Attention (${miniGqaInv.queryHeads}Q / ${miniGqaInv.kvHeads}KV)`,
        badge: 'Persistent KV Owner',
        summary: `Context Tower GQA layer ${id - 1} owning persistent prefix KV slices 0..${committedPrefixLen - 1} within ${kvCapacity}-slot capacity.`,
        specs: [
          `Miniature GQA inventory: ${miniGqaInv.queryHeads} query heads / ${miniGqaInv.kvHeads} KV heads, head width dHead = ${miniGqaInv.headDim} (prod 32Q / 2KV, dHead = 128)`,
          `Populated slots: ${committedPrefixLen} / ${kvCapacity} (${formatKiB((prodMem.kvBytesPerToken / 6) * committedPrefixLen)} at bay)`,
          `Attention: causal over Context prefix`,
        ],
      };
      partCatalog[`den-gqa-body-${padId}`] = {
        id: `den-gqa-body-${padId}`,
        title: `Denoiser Bay ${padId} · GQA Attention (${isDiffusionMode ? 'Bidirectional Block' : isMockArMode ? 'Causal S=1' : 'Idle'})`,
        badge: isHistoricalDenoiser
          ? 'Historical Prior Pass'
          : isDiffusionMode
            ? `Ephemeral B=${activeBlockSize} KV`
            : isMockArMode
              ? 'Mock-AR S=1'
              : 'Idle in AR',
        summary: isDiffusionMode
          ? `Attends bidirectionally over concatenated [Context KV[0..P_cond-1] | Ephemeral Block KV[0..B-1]]; discards block KV after pass.`
          : `Denoiser GQA layer ${id - 1} (${modeAtRecord} mode).`,
        specs: [
          `Miniature GQA inventory: ${miniGqaInv.queryHeads} query heads / ${miniGqaInv.kvHeads} KV heads, head width dHead = ${miniGqaInv.headDim}`,
          denKvCondLabel,
          `Ephemeral block KV: ${ephemeralKvCount} slot${ephemeralKvCount === 1 ? '' : 's'} (discarded after step)`,
          `adaLN ordering: modulate → RMSNorm (pre-norm)`,
        ],
      };
      partCatalog[`linker-gqa-kv-bay-${padId}`] = {
        id: `linker-gqa-kv-bay-${padId}`,
        title: `Cross-Tower Linker · Bay ${padId} GQA Logical KV Prefix`,
        badge: 'Logical Read-Only Access',
        summary: `Logical read-only prefix KV access (${kvRangeLabel}) from Context Tower into Denoiser Bay ${padId} — NOT a mandatory physical inter-device copy (shared-memory zero-copy on single GPU).`,
        specs: [
          `Current Context prefix: ${kvRangeLabel}`,
          ...(isHistoricalDenoiser ? [denKvCondLabel] : []),
          `ownerRole: context_tower → consumerRole: denoiser_tower`,
          `Logical read-only prefix access (never mutates Context KV)`,
        ],
      };
    } else {
      // MoE BAYS ('E', ×23): STRICTLY 0 B STATELESS — ZERO PERSISTENT CACHE & ZERO CROSS-TOWER LINKER!
      // Context MoE: Shared expert + router + compact routed bank
      pushSolid({
        cx: CONTEXT_TOWER_X - 2.05,
        cy,
        cz: 0.06,
        ex: 1.85,
        ey: 0.82,
        ez: 0.84,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id * 1.5,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: ctxActive ? 1 : 0,
        lineStyle: ctxLineStyle,
        cssWidth: ctxCssWidth,
        lineAlpha: ctxLineAlpha,
        pick: {
          towerRole: 'context',
          partId: `ctx-moe-shared-${padId}`,
          executionId: contextExecId,
          traceOrigin: ctxOrigin,
          semanticTag: 'context_moe_stateless',
          absoluteTokenOffset: 0,
          tensorKey: 'shared_expert',
        },
      });
      pushSolid({
        cx: CONTEXT_TOWER_X - 0.85,
        cy,
        cz: 0,
        ex: 0.42,
        ey: 0.88,
        ez: 0.92,
        cellFamily: CELL_MOE_AMBER,
        fp8Flag: ctxDimmed ? 1 : 0,
        phase: id + committedPrefixLen,
        bayId: id,
        cellScale: 0.18,
        highlightFlag: ctxActive ? 1 : 0,
        lineStyle: ctxLineStyle,
        cssWidth: ctxCssWidth,
        lineAlpha: ctxLineAlpha,
        pick: {
          towerRole: 'context',
          partId: `ctx-moe-body-${padId}`,
          executionId: contextExecId,
          traceOrigin: ctxOrigin,
          semanticTag: 'context_moe_stateless',
          absoluteTokenOffset: 0,
          tensorKey: 'moe_router',
        },
      });
      // Compact 3-slab routed expert cluster well inside Context Tower (max X = -9.45 — clean gap from bridge!)
      for (let c = 0; c < 3; c++) {
        const isActiveTile = (id + committedPrefixLen + c) % 3 !== 0;
        pushSolid({
          cx: CONTEXT_TOWER_X + 0.25 + c * 0.92,
          cy: cy + (isActiveTile ? 0.06 : 0),
          cz: (c % 2 === 0 ? -0.18 : 0.18),
          ex: 0.76,
          ey: isActiveTile ? 0.82 : 0.68,
          ez: 0.56,
          cellFamily: isActiveTile ? CELL_MOE_AMBER : CELL_WEIGHT_BLUE,
          fp8Flag: ctxDimmed ? 1 : 0,
          phase: id * 2 + c,
          bayId: id,
          cellScale: 0.20,
          highlightFlag: ctxActive && isActiveTile ? 1 : 0,
          lineStyle: ctxLineStyle,
          cssWidth: ctxActive ? 1.50 : 0.85,
          lineAlpha: ctxActive ? 0.96 : ctxDimmed ? 0.24 : 0.46,
          pick: {
            towerRole: 'context',
            partId: `ctx-moe-routed-${padId}-${c}`,
            executionId: contextExecId,
            traceOrigin: ctxOrigin,
            semanticTag: 'context_moe_stateless',
            absoluteTokenOffset: 0,
            tensorKey: 'routed_experts',
          },
        });
      }

      // Denoiser MoE: Mirror layout well inside Denoiser Tower (min X = +9.45 — ZERO bridge linker!)
      pushSolid({
        cx: DENOISER_TOWER_X + 2.05,
        cy,
        cz: 0.06,
        ex: 1.85,
        ey: 0.82,
        ez: 0.84,
        cellFamily: CELL_WEIGHT_BLUE,
        fp8Flag: denDimmed ? 1 : 0,
        phase: id * 1.5 + 31,
        bayId: id,
        cellScale: 0.22,
        highlightFlag: denActive ? 1 : 0,
        lineStyle: denLineStyle,
        cssWidth: denCssWidth,
        lineAlpha: denLineAlpha,
        pick: {
          towerRole: 'denoiser',
          partId: `den-moe-shared-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_moe_stateless',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'shared_expert',
        },
      });
      pushSolid({
        cx: DENOISER_TOWER_X + 0.85,
        cy,
        cz: 0,
        ex: 0.42,
        ey: 0.88,
        ez: 0.92,
        cellFamily: CELL_MOE_AMBER,
        fp8Flag: denDimmed ? 1 : 0,
        phase: id + activeBlockSize + 31,
        bayId: id,
        cellScale: 0.18,
        highlightFlag: denActive ? 1 : 0,
        lineStyle: denLineStyle,
        cssWidth: denCssWidth,
        lineAlpha: denLineAlpha,
        pick: {
          towerRole: 'denoiser',
          partId: `den-moe-body-${padId}`,
          executionId: denoiserExecId,
          traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_moe_stateless',
          absoluteTokenOffset: denoiserAbsStart,
          tensorKey: 'moe_router',
        },
      });
      for (let c = 0; c < 3; c++) {
        const isActiveTile = (id + activeBlockSize + c) % 3 !== 0;
        pushSolid({
          cx: DENOISER_TOWER_X - 0.25 - c * 0.92,
          cy: cy + (isActiveTile ? 0.06 : 0),
          cz: (c % 2 === 0 ? -0.18 : 0.18),
          ex: 0.76,
          ey: isActiveTile ? 0.82 : 0.68,
          ez: 0.56,
          cellFamily: isActiveTile ? CELL_MOE_AMBER : CELL_WEIGHT_BLUE,
          fp8Flag: denDimmed ? 1 : 0,
          phase: id * 2 + c + 31,
          bayId: id,
          cellScale: 0.20,
          highlightFlag: denActive && isActiveTile ? 1 : 0,
          lineStyle: denLineStyle,
          cssWidth: denActive ? 1.50 : 0.85,
          lineAlpha: denActive ? 0.96 : denDimmed ? 0.20 : 0.46,
          pick: {
            towerRole: 'denoiser',
            partId: `den-moe-routed-${padId}-${c}`,
            executionId: denoiserExecId,
            traceOrigin: 'denoiser_scratch',
            semanticTag: 'denoiser_moe_stateless',
            absoluteTokenOffset: denoiserAbsStart,
            tensorKey: 'routed_experts',
          },
        });
      }

      partCatalog[`ctx-moe-body-${padId}`] = {
        id: `ctx-moe-body-${padId}`,
        title: `Context Bay ${padId} · MoE 128+1 (0 B Stateless)`,
        badge: '0 B Stateless · No Handoff',
        summary: 'Token-independent MoE FFN (top-6 routed + 1 double-wide shared, FP64 router compute). Carries ZERO persistent state and ZERO cross-tower handoff.',
        specs: ['Persistent cache: 0 B', 'Cross-tower handoff: NONE (stateless)', 'FP32 e_score_correction_bias preserved'],
      };
      partCatalog[`den-moe-body-${padId}`] = {
        id: `den-moe-body-${padId}`,
        title: `Denoiser Bay ${padId} · MoE 128+1 (0 B Stateless)`,
        badge: '0 B Stateless · Post-Norm adaLN',
        summary: 'Denoiser MoE layer using post-norm adaLN ordering (RMSNorm → modulate) + residual gate. Carries ZERO cross-tower cache handoff.',
        specs: ['Persistent cache: 0 B', 'Cross-tower handoff: NONE (stateless)', 'adaLN ordering: RMSNorm → modulate (post-norm)'],
      };
    }

    // Tower-isolated fascia PickBoxes for every bay 1..52 on both Context and Denoiser
    const bayHalfY = BAY_PITCH_Y * 0.47;
    const gqaMaxZ = kind === '*' ? MACRO_KV_START_Z + kvCapacityLength : 0.85;
    fasciaPickBoxes.push(
      {
        bayId: id,
        partId: `ctx-fascia-bay-${padId}`,
        towerRole: 'context',
        executionId: contextExecId,
        traceOrigin: ctxOrigin,
        semanticTag: `context_${kind === '*' ? 'gqa' : kind === 'M' ? 'mamba' : 'moe'}_fascia`,
        absoluteTokenOffset: 0,
        min: [CONTEXT_RESIDUAL_X - 0.5, cy - bayHalfY, -0.85],
        max: [BRIDGE_LEFT_X - 0.4, cy + bayHalfY, gqaMaxZ],
      },
      {
        bayId: id,
        partId: `den-fascia-bay-${padId}`,
        towerRole: 'denoiser',
        executionId: denoiserExecId,
        traceOrigin: 'denoiser_scratch',
        semanticTag: `denoiser_${kind === '*' ? 'gqa' : kind === 'M' ? 'mamba' : 'moe'}_fascia`,
        absoluteTokenOffset: denoiserAbsStart,
        min: [BRIDGE_RIGHT_X + 0.4, cy - bayHalfY, -0.85],
        max: [DENOISER_RESIDUAL_X + 0.5, cy + bayHalfY, kind === '*' ? MACRO_KV_START_Z + ephemeralKvCount * MACRO_KV_SLICE_PITCH + 0.5 : 0.85],
      },
    );
  }

  // Subtle horizontal NVIDIA Figure-2 phase divider hairlines across both towers (after bays 35, 41, 44)
  for (const prevBay of [35, 41, 44]) {
    const midY = 0.5 * (bayCenterY(prevBay) + bayCenterY(prevBay + 1));
    pushFreeLine(CONTEXT_RESIDUAL_X - 0.4, midY, -0.5, BRIDGE_LEFT_X - 0.4, midY, -0.5, 0.85, LINE_STYLE_GUIDE, 0.34);
    pushFreeLine(BRIDGE_RIGHT_X + 0.4, midY, -0.5, DENOISER_RESIDUAL_X + 0.4, midY, -0.5, 0.85, LINE_STYLE_GUIDE, 0.34);
  }

  // ============================================================================
  // 5. SEPARATE DENOISER REFINEMENT LOOP (RIGHT x = +18.8) VS. BLOCK COMMIT BRIDGE
  // ============================================================================

  // A. Denoiser Refinement Loop: Denoiser [MASK]^B tray -> outer right arc x = +18.8 -> Denoiser Embedding
  const refWaypoints: readonly Vec3[] = [
    [DENOISER_TOWER_X + 2.8, BLOCK_TRAY_Y, 0.25],
    [REFINEMENT_LOOP_X, BLOCK_TRAY_Y, 0.25],
    [REFINEMENT_LOOP_X, EMBED_Y, 0.25],
    [DENOISER_TOWER_X + 2.8, EMBED_Y, 0.25],
  ];

  if (refinementLoopState === 'active') {
    flow.addRoundedOrtho3D(refWaypoints, 1.4, [0, 0, 1], 0.34, FLOW_COLOR_REFINEMENT, {
      arrowHead: true,
      arrowLength: 0.72,
      arrowWidth: 0.68,
      staticPulse: reducedMotion,
    });
    for (let i = 0; i < refWaypoints.length - 1; i++) {
      const a = refWaypoints[i];
      const b = refWaypoints[i + 1];
      pushFreeLine(a[0], a[1], a[2], b[0], b[1], b[2], 1.75, LINE_STYLE_ACTIVE, 0.90);
    }
    ribbons.push({
      cx: REFINEMENT_LOOP_X,
      cy: spineMidY,
      cz: 0.25,
      ex: 0.52,
      ey: 2.2,
      ez: 0.48,
      kind: RIBBON_KIND_ARROW_POS_Y,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: 0.82,
    });
  } else if (refinementLoopState === 'paused_ready') {
    flow.addRoundedOrtho3D(refWaypoints, 1.4, [0, 0, 1], 0.20, FLOW_COLOR_REFINEMENT_PAUSED, {
      arrowHead: true,
      arrowLength: 0.50,
      arrowWidth: 0.48,
      staticPulse: true,
    });
    for (let i = 0; i < refWaypoints.length - 1; i++) {
      pushDashedLine3D(freeLines, refWaypoints[i], refWaypoints[i + 1], 1.05, LINE_STYLE_GUIDE, 0.42, 0.65, 0.45);
    }
  } else {
    for (let i = 0; i < refWaypoints.length - 1; i++) {
      pushDashedLine3D(freeLines, refWaypoints[i], refWaypoints[i + 1], 0.85, LINE_STYLE_GUIDE, 0.14, 0.60, 0.65);
    }
  }

  solidPickBoxes.push({
    bayId: 53,
    partId: 'pathway-refinement-loop',
    towerRole: 'denoiser',
    executionId: denoiserExecId,
    traceOrigin: 'denoiser_scratch',
    semanticTag: 'refinement_loop',
    absoluteTokenOffset: denoiserAbsStart,
    tensorKey: 'refinement_loop',
    min: [REFINEMENT_LOOP_X - 0.55, BLOCK_TRAY_Y, -0.2],
    max: [REFINEMENT_LOOP_X + 0.55, EMBED_Y, 0.7],
  });

  // B. Completed-Block Context Commit Pathway:
  // Distinct lower-to-upper emerald bridge crossing from Denoiser Output tray -> under towers ->
  // outer left lane x = -18.8 -> Context Embedding input!
  const commitWaypoints: readonly Vec3[] = [
    [DENOISER_TOWER_X, BLOCK_TRAY_Y - 0.42, 0.35],
    [DENOISER_TOWER_X, COMMIT_BRIDGE_BOTTOM_Y, 0.35],
    [COMMIT_BRIDGE_LEFT_X, COMMIT_BRIDGE_BOTTOM_Y, 0.35],
    [COMMIT_BRIDGE_LEFT_X, EMBED_Y + 1.35, 0.35],
    [CONTEXT_TOWER_X, EMBED_Y + 1.35, 0.35],
    [CONTEXT_TOWER_X, EMBED_Y + 0.92, 0.35],
  ];

  if (blockCommitPathwayState === 'active_block_commit' || blockCommitPathwayState === 'active_single_token') {
    const isBlockCommit = blockCommitPathwayState === 'active_block_commit';
    flow.addRoundedOrtho3D(
      commitWaypoints,
      1.6,
      [0, 0, 1],
      isBlockCommit ? 0.40 : 0.28,
      FLOW_COLOR_COMMIT_ACTIVE,
      {
        arrowHead: true,
        arrowLength: 0.76,
        arrowWidth: 0.72,
        staticPulse: reducedMotion,
      },
    );
    for (let i = 0; i < commitWaypoints.length - 1; i++) {
      const a = commitWaypoints[i];
      const b = commitWaypoints[i + 1];
      pushFreeLine(
        a[0], a[1], a[2],
        b[0], b[1], b[2],
        isBlockCommit ? 1.85 : 1.65,
        LINE_STYLE_ACTIVE,
        isBlockCommit ? 0.94 : 0.86,
      );
    }
    ribbons.push({
      cx: COMMIT_BRIDGE_LEFT_X,
      cy: spineMidY,
      cz: 0.35,
      ex: 0.56,
      ey: 2.4,
      ez: 0.48,
      kind: RIBBON_KIND_ARROW_POS_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: 0.88,
    });
  } else if (blockCommitPathwayState === 'staged_ready') {
    // Staged/dashed emerald preview during `completed_uncommitted` ("Ready to Commit Block -> Context")
    flow.addRoundedOrtho3D(commitWaypoints, 1.6, [0, 0, 1], 0.26, FLOW_COLOR_COMMIT_STAGED, {
      arrowHead: true,
      arrowLength: 0.62,
      arrowWidth: 0.58,
      staticPulse: true,
    });
    for (let i = 0; i < commitWaypoints.length - 1; i++) {
      pushDashedLine3D(freeLines, commitWaypoints[i], commitWaypoints[i + 1], 1.45, LINE_STYLE_BF16_GOLD, 0.74, 0.75, 0.45);
    }
    ribbons.push({
      cx: 0,
      cy: COMMIT_BRIDGE_BOTTOM_Y,
      cz: 0.35,
      ex: 2.2,
      ey: 0.46,
      ez: 0.44,
      kind: RIBBON_KIND_ARROW_NEG_X,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: 0.64,
    });
  } else {
    // Dormant during active unmasking / AR
    for (let i = 0; i < commitWaypoints.length - 1; i++) {
      pushDashedLine3D(freeLines, commitWaypoints[i], commitWaypoints[i + 1], 0.85, LINE_STYLE_GUIDE, 0.12, 0.65, 0.75);
    }
  }

  solidPickBoxes.push({
    bayId: 0,
    partId: 'pathway-block-commit',
    towerRole: 'shared',
    executionId: contextExecId,
    traceOrigin: 'persistent_cache',
    semanticTag: 'block_commit',
    absoluteTokenOffset: committedPrefixLen,
    tensorKey: 'block_commit_pathway',
    min: [COMMIT_BRIDGE_LEFT_X - 0.55, COMMIT_BRIDGE_BOTTOM_Y - 0.4, -0.15],
    max: [COMMIT_BRIDGE_LEFT_X + 0.55, EMBED_Y + 1.6, 0.85],
  });

  partCatalog['pathway-refinement-loop'] = {
    id: 'pathway-refinement-loop',
    title: 'Denoiser Refinement Loop (Outer Right Arc)',
    badge:
      refinementLoopState === 'active'
        ? 'Active Unmasking Loop'
        : refinementLoopState === 'paused_ready'
          ? 'Paused · Block Unmasked'
          : 'Dormant',
    summary:
      'Cycles partially unmasked block tokens xt back to Denoiser Embedding for the next refinement pass while Context Tower stays completely frozen at P.',
    specs: [
      `State: ${refinementLoopState}`,
      `Masked remaining: ${activeBlockMaskedCount} / ${activeBlockSize}`,
      `Completed refinements: ${activeBlock?.completedRefinements ?? 0} / ${activeBlock?.blockConfig.stepsPerBlock ?? historicalConfig.stepsPerBlock}`,
    ],
  };
  partCatalog['pathway-block-commit'] = {
    id: 'pathway-block-commit',
    title:
      blockCommitPathwayState === 'staged_ready'
        ? 'Ready to Commit Block → Context'
        : blockCommitPathwayState === 'active_single_token'
          ? 'Single-Token Commit → Context (Mock-AR)'
          : 'Completed-Block Context Commit Pathway',
    badge:
      blockCommitPathwayState === 'active_block_commit'
        ? 'Committed Block → Context'
        : blockCommitPathwayState === 'active_single_token'
          ? 'Mock-AR Token Commit'
          : blockCommitPathwayState === 'staged_ready'
            ? 'Ready to Commit Block -> Context'
            : 'Dormant',
    summary:
      blockCommitPathwayState === 'staged_ready'
        ? `Block is fully unmasked (${activeBlockSize}/${activeBlockSize}) and staged for explicit single-shot causal commit into Context Tower (currently still at P = ${committedPrefixLen}).`
        : 'Advances Context Tower persistent KV + Mamba-2 boundary caches once a denoising block completes (or 1 token on Mock-AR).',
    specs: [
      `State: ${blockCommitPathwayState}`,
      `Committed Context P = ${committedPrefixLen} / ${kvCapacity}`,
      `Mode at step: ${modeAtRecord}`,
    ],
  };

  // ============================================================================
  // 6. LANDMARKS, KV DEPTH TICKS & DIMENSION CALIPERS
  // ============================================================================

  const timestepPillLabel =
    ts.conditioningMode === 'bypassed_ar' || ts.conditioningMode === 'bypassed_mock_ar'
      ? 'BYPASSED (t = null)'
      : ts.conditioningMode === 'inactive_no_pass_yet'
        ? ts.nextEligibleStep
          ? `INACTIVE · Next t=${ts.nextEligibleStep.nextTModel.toFixed(2)} (${ts.nextEligibleStep.nextNMasked}/${ts.nextEligibleStep.blockSize})`
          : 'INACTIVE (no pass yet)'
        : ts.conditioningMode === 'historical_prior_pass' && ts.executedPass
          ? `Prior Rec #${ts.executedPass.sourceRecordIndex} (t=${ts.executedPass.tModel.toFixed(2)}, ${ts.executedPass.nMaskedInput}/${ts.executedPass.blockSize} in)`
          : ts.executedPass
            ? ts.blockCompletionState === 'completed_uncommitted'
              ? `Pass t=${ts.executedPass.tModel.toFixed(2)} (${ts.executedPass.nMaskedInput}/${ts.executedPass.blockSize} in) · Block complete`
              : ts.currentBlockMask && ts.nextEligibleStep
                ? `Pass t=${ts.executedPass.tModel.toFixed(2)} (${ts.executedPass.nMaskedInput}/${ts.executedPass.blockSize} in) · Now ${ts.currentBlockMask.currentUnresolvedCount}/${ts.executedPass.blockSize} [MASK]`
                : `ACTIVE · t=${ts.executedPass.tModel.toFixed(2)} (${ts.executedPass.nMaskedInput}/${ts.executedPass.blockSize} in)`
            : timestepStatusLabel;

  landmarks.push(
    {
      id: 'header-context',
      label: `Context Tower · Persistent Prefix (P = ${committedPrefixLen}/${kvCapacity})`,
      sublabel: `${formatMiB(prodMem.persistentContextMambaBytes, 1)} Mamba + ${formatKiB(prodMem.persistentContextKvBytes)} KV`,
      world: [CONTEXT_TOWER_X - 4.2, EMBED_Y + 2.0, 0],
      bayId: 0,
      partId: 'ctx-embed',
      tone: 'cyan',
      align: 'left',
      tier: 'primary',
    },
    {
      id: 'header-timestep',
      label: `Timestep Rail · ${timestepPillLabel}`,
      sublabel: ts.executedPass?.isDiagnosticOverride
        ? 'Canonical t = nMasked / B (diagnostic override active) → φ(t) → MLP → 3d tEmb'
        : 'Canonical t = nMasked / B (never r / K) → φ(t) → MLP → 3d tEmb',
      world: [TIMESTEP_RAIL_X, TIMESTEP_HUB_Y + 1.8, railZ],
      bayId: -1,
      partId: 'timestep-hub',
      tone: ts.conditioningMode === 'active_diffusion' ? 'gold' : 'muted',
      align: 'center',
      tier: 'primary',
    },
    {
      id: 'header-denoiser',
      label: isArMode
        ? 'Denoiser Tower · Idle (AR Mode)'
        : isMockArMode
          ? 'Denoiser Tower · Mock-AR Causal (S = 1)'
          : isHistoricalDenoiser
            ? `Denoiser · Prior Pass (Rec #${dl.sourceRecordIndex}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'})`
            : `Denoiser Tower · Ephemeral Block (B = ${activeBlockSize})`,
      sublabel: isArMode
        ? isHistoricalDenoiser
          ? `Context-only · Last prior pass Rec #${dl.sourceRecordIndex} (P_cond=${dl.sourceContextPrefixLen ?? 'n/a'}) ≠ P=${committedPrefixLen}`
          : 'Context-only execution'
        : isHistoricalDenoiser
          ? `Prior pass (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}) ≠ current Context P=${committedPrefixLen}`
          : 'Scratchpad discarded every step',
      world: [DENOISER_TOWER_X + 4.2, EMBED_Y + 2.0, 0],
      bayId: 0,
      partId: 'den-embed',
      tone: isArMode || isHistoricalDenoiser ? 'muted' : 'default',
      align: 'right',
      tier: 'primary',
    },
    {
      id: 'moe-stateless-note',
      label: '23 MoE Bays · 0 B Stateless (0 Linkers)',
      world: [0, bayCenterY(2), 0.15],
      bayId: 2,
      partId: 'ctx-moe-body-02',
      tone: 'muted',
      align: 'center',
      tier: 'detail',
    },
    {
      id: 'gqa-bay-06-linker',
      label: isMockArMode ? '06 · GQA Logical KV[:-1]' : '06 · GQA Logical KV[0..P-1]',
      world: [0, bayCenterY(6), 0.35],
      bayId: 6,
      partId: 'linker-gqa-kv-bay-06',
      tone: 'cyan',
      align: 'center',
      tier: 'primary',
    },
    {
      id: 'mamba-bay-01-linker',
      label: '01 · Mamba Conv1D + FP32 SSM H',
      world: [0, bayCenterY(1), 0.25],
      bayId: 1,
      partId: 'linker-mamba-ssm-bay-01',
      tone: 'gold',
      align: 'center',
      tier: 'detail',
    },
    {
      id: 'refinement-loop-label',
      label:
        refinementLoopState === 'active'
          ? 'Refinement Loop (Active)'
          : refinementLoopState === 'paused_ready'
            ? 'Refinement Loop (Paused · Unmasked)'
            : 'Refinement Loop (Dormant)',
      world: [REFINEMENT_LOOP_X + 0.8, bayCenterY(26), 0.25],
      partId: 'pathway-refinement-loop',
      tone: refinementLoopState === 'active' ? 'default' : 'muted',
      align: 'right',
      tier: 'primary',
    },
    {
      id: 'commit-bridge-label',
      label:
        blockCommitPathwayState === 'staged_ready'
          ? 'Ready to Commit Block -> Context'
          : blockCommitPathwayState === 'active_block_commit'
            ? 'Committed Block -> Context'
            : blockCommitPathwayState === 'active_single_token'
              ? 'Single-Token Commit -> Context'
              : 'Context Commit Bridge (Dormant)',
      world: [0, COMMIT_BRIDGE_BOTTOM_Y - 0.55, 0.35],
      partId: 'pathway-block-commit',
      tone:
        blockCommitPathwayState === 'active_block_commit' ||
        blockCommitPathwayState === 'active_single_token'
          ? 'cyan'
          : blockCommitPathwayState === 'staged_ready'
            ? 'gold'
            : 'muted',
      align: 'center',
      tier: 'primary',
    },
  );

  // KV depth ticks beside topmost GQA Bay 06 on Context Tower (`1`, `P`, `kvCapacity = 24`)
  const refGqaY = bayCenterY(6) - 0.34;
  const tickTokens = [1, Math.max(1, Math.min(kvCapacity, committedPrefixLen)), kvCapacity].filter(
    (v, idx, arr) => arr.indexOf(v) === idx,
  );
  for (const token of tickTokens) {
    kvDepthTicks.push({
      token,
      world: [CONTEXT_TOWER_X + 1.65, refGqaY, macroKvSliceFrontZ(token - 1)],
    });
  }

  calipers.push(
    {
      id: 'caliper-dual-bridge',
      aWorld: [BRIDGE_LEFT_X, bayCenterY(6) + 0.9, 0.3],
      bWorld: [BRIDGE_RIGHT_X, bayCenterY(6) + 0.9, 0.3],
      label: '29 Aligned Stateful Bays (6 GQA KV + 23 Mamba Conv/SSM · 0 MoE)',
      partIds: ['linker-gqa-kv-bay-06'],
      minPixels: 64,
    },
    {
      id: 'caliper-kv-capacity',
      aWorld: [CONTEXT_TOWER_X + 1.5, refGqaY, MACRO_KV_START_Z],
      bWorld: [CONTEXT_TOWER_X + 1.5, refGqaY, MACRO_KV_START_Z + kvCapacityLength],
      label: `Persistent Context KV Capacity: ${committedPrefixLen} / ${kvCapacity} slots`,
      partIds: ['ctx-kv-capacity-bay-06-r3'],
      minPixels: 48,
    },
  );

  // ============================================================================
  // 7. PACK GPU INSTANCE BUFFERS (EXACT NANO PASS LAYOUTS)
  // ============================================================================

  const solidData = new Float32Array(solids.length * SOLID_INSTANCE_FLOATS);
  const edgeDedup = new Map<string, LineSpec>();

  for (let i = 0; i < solids.length; i++) {
    const s = solids[i];
    const o = i * SOLID_INSTANCE_FLOATS;
    solidData[o + 0] = s.cx;
    solidData[o + 1] = s.cy;
    solidData[o + 2] = s.cz;
    solidData[o + 3] = s.ex;
    solidData[o + 4] = s.ey;
    solidData[o + 5] = s.ez;
    solidData[o + 6] = s.cellFamily;
    solidData[o + 7] = s.fp8Flag;
    solidData[o + 8] = s.phase;
    solidData[o + 9] = s.bayId;
    solidData[o + 10] = s.cellScale;
    solidData[o + 11] = s.highlightFlag;
    if (s.emitSolidEdges !== false) {
      emitCuboidEdges(s, edgeDedup);
    }
  }

  const neutralLines: LineSpec[] = [];
  const goldLines: LineSpec[] = [];
  const activeLines: LineSpec[] = [];

  for (const l of edgeDedup.values()) {
    if (l.style === LINE_STYLE_ACTIVE) activeLines.push(l);
    else if (l.style === LINE_STYLE_BF16_GOLD) goldLines.push(l);
    else neutralLines.push(l);
  }
  for (const l of freeLines) {
    if (l.style === LINE_STYLE_ACTIVE) activeLines.push(l);
    else if (l.style === LINE_STYLE_BF16_GOLD) goldLines.push(l);
    else neutralLines.push(l);
  }

  const orderedLines = [...neutralLines, ...goldLines, ...activeLines];
  const lineData = new Float32Array(orderedLines.length * LINE_INSTANCE_FLOATS);
  for (let i = 0; i < orderedLines.length; i++) {
    const l = orderedLines[i];
    const o = i * LINE_INSTANCE_FLOATS;
    lineData[o + 0] = l.ax;
    lineData[o + 1] = l.ay;
    lineData[o + 2] = l.az;
    lineData[o + 3] = l.bx;
    lineData[o + 4] = l.by;
    lineData[o + 5] = l.bz;
    lineData[o + 6] = l.cssWidth;
    lineData[o + 7] = l.style;
    lineData[o + 8] = l.alpha;
    lineData[o + 9] = l.ownerId;
    lineData[o + 10] = l.face0;
    lineData[o + 11] = l.face1;
    lineData[o + 12] = l.ocx;
    lineData[o + 13] = l.ocy;
    lineData[o + 14] = l.ocz;
    lineData[o + 15] = l.oex;
    lineData[o + 16] = l.oey;
    lineData[o + 17] = l.oez;
  }

  const ribbonData = new Float32Array(ribbons.length * RIBBON_INSTANCE_FLOATS);
  for (let i = 0; i < ribbons.length; i++) {
    const r = ribbons[i];
    const o = i * RIBBON_INSTANCE_FLOATS;
    ribbonData[o + 0] = r.cx;
    ribbonData[o + 1] = r.cy;
    ribbonData[o + 2] = r.cz;
    ribbonData[o + 3] = r.ex;
    ribbonData[o + 4] = r.ey;
    ribbonData[o + 5] = r.ez;
    ribbonData[o + 6] = r.kind;
    ribbonData[o + 7] = r.colorId;
    ribbonData[o + 8] = r.alpha;
    ribbonData[o + 9] = 0;
    ribbonData[o + 10] = 0;
    ribbonData[o + 11] = 0;
  }

  const { vertices: flowVertices, indices: flowIndices } = flow.pack();

  // ============================================================================
  // 8. ACTIVE SELECTION IDENTITY, PROVENANCE BADGES & STRUCTURED DIAGNOSTICS
  // ============================================================================

  let activeSelection: TwoTowerSelectionIdentity | null = null;
  if (selectedBayId !== null) {
    const selTower: TwoTowerTowerRole =
      selectedBayId === -1 ? 'shared' : (selectedTower ?? (isArMode ? 'context' : 'shared'));
    const selMixer: TwoTowerMixerRole =
      selectedBayId >= 1 && selectedBayId <= 52
        ? getTwoTowerBayKind(selectedBayId)
        : selectedBayId === 0
          ? 'embed'
          : selectedBayId === 53
            ? 'output'
            : 'time_cond';
    const selOrigin: TwoTowerTraceOrigin =
      selTower === 'denoiser'
        ? 'denoiser_scratch'
        : selTower === 'context' && selectedBayId >= 1 && selectedBayId <= 52
          ? resolveContextBayOriginFast(view, selectedBayId)
          : selectedBayId === -1
            ? 'denoiser_scratch'
            : 'persistent_cache';
    const selExecId =
      selTower === 'denoiser'
        ? denoiserExecId
        : selTower === 'context'
          ? contextExecId
          : sharedExecId;
    const selAbsPos =
      selTower === 'denoiser'
        ? denoiserAbsStart
        : selTower === 'context'
          ? Math.max(0, committedPrefixLen - 1)
          : null;

    activeSelection = buildTwoTowerSelectionIdentity(view, {
      towerRole: selTower,
      bayId: selectedBayId,
      mixerKind: selMixer,
      tensorRole:
        selectedBayId === -1
          ? 'timestep_rail'
          : selectedBayId === 0
            ? `${selTower}_embed`
            : selectedBayId === 53
              ? `${selTower}_output`
              : `${selTower}_bay_${formatBayId(selectedBayId)}`,
      traceOrigin: selOrigin,
      executionId: selExecId,
      absoluteTokenPosition: selAbsPos,
      provenanceNote: `${selTower.toUpperCase()} · ${
        selectedBayId >= 1 && selectedBayId <= 52
          ? `Bay ${formatBayId(selectedBayId)} (layer ${selectedBayId - 1}, ${selMixer})`
          : selectedBayId === -1
            ? `Timestep Rail (${timestepStatusLabel})`
            : selectedBayId === 0
              ? 'Embedding'
              : 'Output'
      } · mode=${modeAtRecord} · origin=${selOrigin}`,
    });
  }

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    {
      label: `Mode: ${modeAtRecord} (historical configAtStep)`,
      tone: isDiffusionMode ? 'denoiser' : isMockArMode ? 'gold' : 'context',
    },
    {
      label: `Context P = ${committedPrefixLen}/${kvCapacity}`,
      tone: 'context',
      origin: 'persistent_cache',
    },
    {
      label: `Timestep: ${timestepStatusLabel}`,
      tone:
        ts.conditioningMode === 'active_diffusion'
          ? 'gold'
          : ts.conditioningMode === 'historical_prior_pass'
            ? 'warn'
            : 'muted',
      origin: 'denoiser_scratch',
    },
    {
      label: `Record #${view.cursorIndex} · ${view.record.kind}`,
      tone: 'shared',
    },
  ];
  if (isHistoricalDenoiser) {
    provenanceBadges.push({
      label: `Historical Denoiser Pass · Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} vs P=${committedPrefixLen})`,
      tone: 'warn',
      origin: 'denoiser_scratch',
    });
  }
  if (replayPhase !== null) {
    provenanceBadges.push({
      label: SPINE_REPLAY_DISCLAIMER,
      tone: 'muted',
    });
  }
  if (!view.snapshotIntegrity.valid) {
    provenanceBadges.push({
      label: 'Snapshot Integrity Alert',
      tone: 'warn',
    });
  } else if (view.latestContextTrace.status === 'pruned_unavailable') {
    provenanceBadges.push({
      label: 'Context Trace Pruned',
      tone: 'warn',
    });
  }

  const diagnostics: TwoTowerMacroDiagnostics = Object.freeze({
    contextBayCount: 52,
    denoiserBayCount: 52,
    alignedGqaLinkerCount: 6,
    alignedMambaConvLinkerCount: 23,
    alignedMambaSsmLinkerCount: 23,
    moeLinkerCount: 0,
    timestepTapCount: 52,
    timestepPreNormTapCount: 29,
    timestepPostNormTapCount: 23,
    timestepRailActive,
    timestepValue,
    timestepStatusLabel,
    timestepStatus: ts,
    refinementLoopState,
    blockCommitPathwayState,
    contextTraceStatus: view.latestContextTrace.status,
    denoiserLookupStatus: dl.status,
    isHistoricalDenoiserInspection: dl.isHistoricalInspection,
    denoiserSourceRecordIndex: dl.sourceRecordIndex,
    denoiserSourceBlockId: dl.sourceBlockId,
    denoiserSourceRefinementIndex: dl.sourceRefinementIndex,
    denoiserSourceContextPrefixLen: dl.sourceContextPrefixLen,
    denoiserSourceContextCacheDigest: dl.sourceContextCacheDigest,
    denoiserMatchesCurrentContextBoundary: dl.matchesCurrentContextBoundary,
    historicalProvenanceWarning: dl.historicalProvenanceWarning,
    modelDimensionalInventory: TWOTOWER_MINI_MODEL_INVENTORY,
    snapshotIntegrityValid: view.snapshotIntegrity.valid,
    snapshotIntegrityError: view.snapshotIntegrity.error,
    instructionalReplayPhase: replayPhase,
    instructionalReplayDisclaimer: replayPhase !== null ? SPINE_REPLAY_DISCLAIMER : null,
    reducedMotion,
    measuredExecutionMode: modeAtRecord,
    historicalConfigAtStep: historicalConfig,
    lifecyclePhase: view.lifecyclePhase,
    recordKind: view.record.kind,
    recordId: view.identity.recordId,
    branchId: view.branchId,
    cursorIndex: view.cursorIndex,
    committedPrefixLen,
    kvCapacity,
    contextKvConditioningLen,
    activeBlockSize,
    activeBlockMaskedCount,
    contextModelId: view.model.contextModel.modelId,
    denoiserModelId: view.model.denoiserModel.modelId,
    contextParameterFingerprint: view.model.contextModel.parameterFingerprint,
    denoiserParameterFingerprint: view.model.denoiserModel.parameterFingerprint,
    activeLinkerChannel,
    channelEmphasis: Object.freeze({
      gqa_kv: freezeChannelStats(channelStats.gqa_kv),
      mamba_conv: freezeChannelStats(channelStats.mamba_conv),
      mamba_ssm: freezeChannelStats(channelStats.mamba_ssm),
      timestep: freezeChannelStats(channelStats.timestep),
    }),
    flowVertexCount: flowVertices.length / MACRO_FLOW_VERTEX_FLOATS,
    flowTriangleCount: flowIndices.length / 3,
  });

  const resolvedBayId = selectedBayId ?? 1;
  const resolvedLayerIndex = resolvedBayId >= 1 && resolvedBayId <= 52 ? resolvedBayId - 1 : null;

  const modeSubtitle = isArMode
    ? `AR mode · Context Tower active (P=${committedPrefixLen}/${kvCapacity}) · Denoiser idle · ${timestepStatusLabel}`
    : isMockArMode
      ? `Mock-AR mode · Context KV[:-1] + mambaS2 (${contextKvConditioningLen}) → Denoiser causal S=1 · ${timestepStatusLabel}`
      : `Mask-Diffusion · Context P=${committedPrefixLen} (${ts.blockCompletionState === 'committed_to_context' ? 'committed' : 'immutable'}) → Denoiser B=${activeBlockSize} · ${timestepStatusLabel} · ${view.lifecyclePhase}`;

  return {
    viewMode: 'macro',
    title: 'Dual-Tower 52×2 Macro Architecture',
    subtitle: modeSubtitle,
    bayId: resolvedBayId,
    layerIndex: resolvedLayerIndex,
    activeTowerFocus,
    solidData,
    solidCount: solids.length,
    lineData,
    lineCount: orderedLines.length,
    ribbonData,
    ribbonCount: ribbons.length,
    flowVertices,
    flowIndices,
    fitBounds: TWO_TOWER_MACRO_FIT_BOUNDS,
    landmarks,
    kvDepthTicks,
    calipers,
    solidPickBoxes,
    fasciaPickBoxes,
    partCatalog,
    activeSelection,
    provenanceBadges,
    epistemicNote:
      `${TWOTOWER_SPEC.productionDefaults.epistemicNote} Miniature model inventory: 32 GQA query heads / 2 KV heads (head width 4), 16 Mamba heads / 8 groups (4×4 per-head FP32 SSM state).`,
    warningBanner,
    diagnostics,
  };
}
