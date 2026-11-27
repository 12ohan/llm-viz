import { formatKiB, formatMiB, memoryFor } from '../model/memory';
import {
  bodyPrecision,
  countByKind,
  formatBayId,
  kvCachePrecision,
  type BayDescriptor,
  type NemotronSpec,
  type PrecisionMode,
  type Vec3,
} from '../model/spec';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_INSTANCE_BYTES,
  RIBBON_INSTANCE_FLOATS,
  RIBBON_KIND_ARROW_NEG_X,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_NEG_Z,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_ARROW_POS_Y,
  RIBBON_KIND_ARROW_POS_Z,
  RIBBON_KIND_BOX,
  SOLID_INSTANCE_BYTES,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from '../render/primitives';

export {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_INSTANCE_BYTES,
  RIBBON_INSTANCE_FLOATS,
  RIBBON_KIND_ARROW_NEG_X,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_NEG_Z,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_ARROW_POS_Y,
  RIBBON_KIND_ARROW_POS_Z,
  RIBBON_KIND_BOX,
  SOLID_INSTANCE_BYTES,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
};

/** Line instance layout: 18 floats = 72 bytes per instance. */
export const LINE_INSTANCE_FLOATS = 18;
export const LINE_INSTANCE_BYTES = LINE_INSTANCE_FLOATS * 4;

export const LINE_STYLE_NEUTRAL = 0;
export const LINE_STYLE_ACTIVE = 1;
export const LINE_STYLE_GUIDE = 2;
export const LINE_STYLE_BF16_GOLD = 3;

// Vertical tower geometry constants (top-to-bottom along -Y)
export const RESIDUAL_X = 4.60;
export const BAY_PITCH_Y = 1.65;
const CYCLE_GAP_Y = 0.35;
const SECTION_GAP_Y = 1.45;

export const EMBED_Y = 4.60;
export const KV_TRACK_START_Z = 0.90;
const KV_SLICE_PITCH = 0.42;
const KV_SLICE_DEPTH = 0.30;
export const KV_TRACK_LENGTH = 16 * KV_SLICE_PITCH; // 6.72
const KV_RAIL_OFFSETS_X: readonly number[] = [-1.05, -0.45, 0.45, 1.05]; // K1, V1, K2, V2

/**
 * Compute world center Y for 1-based bay `id` (1..52), descending along -Y,
 * with clean Figure-2 section gaps after bays 35, 41, and 44 and subtle cycle gaps after 7, 14, 21, 28.
 */
export function bayCenterY(id: number): number {
  const idx = id - 1;
  let y = -idx * BAY_PITCH_Y;

  // Subtle visual breathing between the five [M·E·M·E·M·*·E] cycles (after 7, 14, 21, 28)
  const completedCycles = Math.min(4, Math.floor(idx / 7));
  y -= completedCycles * CYCLE_GAP_Y;

  // Distinct Figure-2 phase gaps after bays 35, 41, 44
  if (id > 35) y -= SECTION_GAP_Y;
  if (id > 41) y -= SECTION_GAP_Y;
  if (id > 44) y -= SECTION_GAP_Y;
  return y;
}

export const BAY_52_Y = bayCenterY(52);
export const FINAL_NORM_Y = BAY_52_Y - 3.30;
export const LM_HEAD_Y = BAY_52_Y - 6.10;
export const OUTPUT_LOGITS_Y = BAY_52_Y - 8.90;

/** Center of Mamba-2 SSM state register front face for 1-based M bay (used by picking round-trip test). */
export function mambaRegisterFrontWorld(bayId: number): Vec3 {
  // Register center: (-1.35, bayCenterY(bayId), 0), extent (1.50, 1.10, 1.10) -> front face z = +0.55
  return [-1.35, bayCenterY(bayId), 0.55];
}

function kvSliceCenterZ(tokenIndex1Based: number): number {
  return KV_TRACK_START_Z + KV_SLICE_PITCH * (tokenIndex1Based - 1) + KV_SLICE_DEPTH * 0.5;
}

function kvSliceFrontZ(tokenIndex1Based: number): number {
  return KV_TRACK_START_Z + KV_SLICE_PITCH * (tokenIndex1Based - 1) + KV_SLICE_DEPTH;
}

/**
 * Optically balanced overview bounds for the 52-layer tower + left motif brackets +
 * right-side landmark pills (compensating for 30° yaw +Z KV projection and top toolbar headroom).
 */
export const TOWER_FIT_BOUNDS: AABB = {
  min: [-6.4, OUTPUT_LOGITS_Y - 0.8, -1.6],
  max: [14.4, EMBED_Y + 5.2, KV_TRACK_START_Z + KV_TRACK_LENGTH + 0.6],
};

export interface BayAnchor {
  readonly id: number;
  readonly kind: BayDescriptor['kind'];
  readonly pillWorld: Vec3; // Landmark pill anchor beside the bay
  readonly centerWorld: Vec3;
  readonly bounds: AABB;
}

export interface PhaseSectionAnchor {
  readonly index: 0 | 1 | 2 | 3;
  readonly first: number;
  readonly last: number;
  readonly topY: number;
  readonly bottomY: number;
  readonly midWorld: Vec3;
}

export interface MacroSceneGeometry {
  readonly solidData: Float32Array;
  readonly solidCount: number;
  readonly lineData: Float32Array;
  readonly lineCount: number;
  readonly ribbonData: Float32Array;
  readonly ribbonCount: number;
  readonly fitBounds: AABB;
  readonly bayAnchors: readonly BayAnchor[];
  readonly phaseSections: readonly PhaseSectionAnchor[];
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly solidPickBoxes: readonly PickBox[];
  readonly fasciaPickBoxes: readonly PickBox[];
}

interface SolidSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  cellFamily: number;
  fp8Flag: number;
  phase: number;
  bayId: number;
  cellScale: number; // density of procedural matrix cells
  highlightFlag: number; // 1 = selected/hovered, 2 = BF16 preserved in FP8 mode
  lineStyle: number;
  cssWidth: number;
  lineAlpha: number;
  pickPart?: string;
}

interface LineSpec {
  ax: number;
  ay: number;
  az: number;
  bx: number;
  by: number;
  bz: number;
  cssWidth: number;
  style: number;
  alpha: number;
  ownerId: number;
  face0: number;
  face1: number;
  ocx: number;
  ocy: number;
  ocz: number;
  oex: number;
  oey: number;
  oez: number;
}

interface RibbonSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  kind: number;
  colorId: number;
  alpha: number;
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
    } else if (
      cand.style === LINE_STYLE_BF16_GOLD &&
      existing.style === LINE_STYLE_NEUTRAL
    ) {
      dedupMap.set(key, cand);
    }
  }
}

export type SpinePropagationWaveMode = 'fast' | 'instructional';

/** Tight single-bay neighborhood window (in bay units) used by instructional replay mode. */
export const INSTRUCTIONAL_BAY_WINDOW = 1.2 as const;

/** Normalized stage thresholds shared by instructional replay geometry and the Tower stage HUD. */
export const SPINE_STAGE_EMBED_END = 0.05 as const;
export const SPINE_STAGE_BAYS_END = 0.91 as const;
export const SPINE_STAGE_NORM_END = 0.96 as const;

export const SPINE_REPLAY_DISCLAIMER =
  'Educational replay · not measured compute time' as const;

export type SpineWaveStageKind =
  | 'embedding'
  | 'mamba'
  | 'moe'
  | 'gqa'
  | 'final_norm'
  | 'output';

export interface SpineWaveStageInfo {
  readonly stageKind: SpineWaveStageKind;
  readonly bayId: number | null;
  readonly title: string;
  readonly actionDetail: string;
  readonly rmsText: string | null;
  readonly progressPct: number;
  readonly disclaimer: typeof SPINE_REPLAY_DISCLAIMER;
}

export interface SpineWaveBaySummaryInput {
  readonly bayId: number;
  readonly rmsOut: ArrayLike<number>;
}

export interface SpinePropagationWave {
  /** Normalized progress in [0, 1] descending from Embedding (0.0) -> Bays 01..52 -> Output (1.0). */
  readonly progress: number;
  /** Optional 1-based token index (defaults to clampedTokens) highlighted on crossed GQA KV tracks. */
  readonly activeTokenIndex?: number;
  /** Optional pacing/focus mode: `'fast'` (default macro sweep) vs `'instructional'` (tight single-bay focus). */
  readonly mode?: SpinePropagationWaveMode;
}

export interface BuildTowerOptions {
  readonly spec: NemotronSpec;
  readonly tokens: number;
  readonly precision: PrecisionMode;
  readonly selectedBay: number | null;
  readonly hoveredBay: number | null;
  readonly propagationWave?: SpinePropagationWave | null;
}

function clampProgress(progress: number): number {
  return Number.isFinite(progress) ? Math.max(0, Math.min(1, progress)) : 0;
}

function instructionalBayIndexForProgress(waveP: number, bayCount: number): number {
  const span = SPINE_STAGE_BAYS_END - SPINE_STAGE_EMBED_END;
  const norm = (waveP - SPINE_STAGE_EMBED_END) / span;
  return Math.min(bayCount - 1, Math.max(0, Math.floor(norm * bayCount)));
}

function extractStageRmsText(
  bayId: number | null,
  baySummaries?: readonly SpineWaveBaySummaryInput[] | null,
  activeTokenIndex?: number,
): string | null {
  if (bayId === null || !baySummaries || baySummaries.length === 0) return null;
  const summary = baySummaries.find((entry) => entry.bayId === bayId);
  if (!summary || summary.rmsOut.length === 0) return null;
  const rawIdx =
    activeTokenIndex !== undefined && Number.isFinite(activeTokenIndex)
      ? Math.trunc(activeTokenIndex)
      : summary.rmsOut.length - 1;
  const idx = Math.max(0, Math.min(summary.rmsOut.length - 1, rawIdx));
  const val = summary.rmsOut[idx];
  if (!Number.isFinite(val)) return null;
  return `RMS ${val.toFixed(2)}`;
}

/**
 * Pure stage descriptor for the Tower propagation replay HUD.
 * Maps progress in [0, 1] across Embedding (< 0.05), Bays 01..52 (0.05..0.91),
 * Final RMSNorm (0.91..0.96), and Output LM Head (>= 0.96) with precision-accurate
 * architectural copy and an explicit educational-replay disclaimer.
 */
export function describeSpineWaveStage(
  progress: number,
  spec: NemotronSpec,
  precision: PrecisionMode,
  baySummaries?: readonly SpineWaveBaySummaryInput[] | null,
  activeTokenIndex?: number,
): SpineWaveStageInfo {
  const p = clampProgress(progress);
  const progressPct = Math.round(p * 100);

  if (p < SPINE_STAGE_EMBED_END) {
    return {
      stageKind: 'embedding',
      bayId: null,
      title: `Embedding · Untied ${spec.positionalEncoding}`,
      actionDetail: `Maps token ID into ${spec.hidden.toLocaleString('en-US')}-dim residual stream (${spec.vocabSize.toLocaleString('en-US')} vocab · no positional embedding)`,
      rmsText: null,
      progressPct,
      disclaimer: SPINE_REPLAY_DISCLAIMER,
    };
  }

  if (p >= SPINE_STAGE_NORM_END) {
    return {
      stageKind: 'output',
      bayId: null,
      title: 'Output · Untied LM Head',
      actionDetail: `Projects ${spec.hidden.toLocaleString('en-US')} → ${spec.vocabSize.toLocaleString('en-US')} vocab logits and next-token softmax`,
      rmsText: null,
      progressPct,
      disclaimer: SPINE_REPLAY_DISCLAIMER,
    };
  }

  if (p >= SPINE_STAGE_BAYS_END) {
    return {
      stageKind: 'final_norm',
      bayId: null,
      title: `Final RMSNorm · Post-${spec.bays.length} Stream`,
      actionDetail: `Normalizes final ${spec.hidden.toLocaleString('en-US')}-dim residual stream (ε = 1e-5) before untied LM head`,
      rmsText: null,
      progressPct,
      disclaimer: SPINE_REPLAY_DISCLAIMER,
    };
  }

  const bayIdx = instructionalBayIndexForProgress(p, spec.bays.length);
  const bay = spec.bays[bayIdx];
  const counts = countByKind(spec);
  let kindOrdinal = 0;
  for (let i = 0; i <= bayIdx; i++) {
    if (spec.bays[i].kind === bay.kind) kindOrdinal++;
  }

  const mem = memoryFor(spec, 1, precision);
  const rmsText = extractStageRmsText(bay.id, baySummaries, activeTokenIndex);
  const bayLabel = `Bay ${formatBayId(bay.id)}`;

  if (bay.kind === 'M') {
    const fixedMambaText = formatMiB(mem.fixedMambaBytes, 1);
    const precisionNote =
      precision === 'fp8_selective'
        ? bay.feeder
          ? 'BF16 feeder preserved'
          : 'BF16 Conv1D · FP8 body'
        : '0 B/tok growth';
    return {
      stageKind: 'mamba',
      bayId: bay.id,
      title: `${bayLabel} · Mamba-2${bay.feeder ? ' Feeder' : ''} (${kindOrdinal}/${counts.M})`,
      actionDetail: `Updates fixed ${fixedMambaText} SSM state in place (FP32 · ${precisionNote})`,
      rmsText,
      progressPct,
      disclaimer: SPINE_REPLAY_DISCLAIMER,
    };
  }

  if (bay.kind === 'E') {
    const sharedRatio = Math.round(spec.moe.sharedWidth / spec.moe.width);
    const bodyLabel = precision === 'fp8_selective' ? 'FP8' : 'BF16';
    return {
      stageKind: 'moe',
      bayId: bay.id,
      title: `${bayLabel} · MoE ${spec.moe.routed}+${spec.moe.shared} (${kindOrdinal}/${counts.E})`,
      actionDetail: `Activates ${spec.moe.active}/${spec.moe.routed} routed + ${sharedRatio}× shared (0 B cache · ${bodyLabel})`,
      rmsText,
      progressPct,
      disclaimer: SPINE_REPLAY_DISCLAIMER,
    };
  }

  const kvPerBayBytes = counts['*'] > 0 ? mem.kvBytesPerToken / counts['*'] : 0;
  const kvPrecLabel = precision === 'fp8_selective' ? 'FP8' : 'BF16';
  const bodySuffix = precision === 'fp8_selective' ? ' · BF16 body' : '';
  return {
    stageKind: 'gqa',
    bayId: bay.id,
    title: `${bayLabel} · GQA Attention (${kindOrdinal}/${counts['*']} KV bays)`,
    actionDetail: `Appends +1 KV slice onto +Z (+${formatKiB(kvPerBayBytes)}/bay · +${formatKiB(mem.kvBytesPerToken)}/tok ${kvPrecLabel}${bodySuffix})`,
    rmsText,
    progressPct,
    disclaimer: SPINE_REPLAY_DISCLAIMER,
  };
}

/**
 * Map a normalized propagation progress in [0, 1] smoothly from EMBED_Y through
 * Bays 01..52 down to OUTPUT_LOGITS_Y, staying aligned with `bayIdx ≈ progress * 52`.
 */
function waveProgressToWorldY(waveP: number): number {
  const u = waveP * 53;
  if (u <= 0) return EMBED_Y;
  if (u >= 53) return OUTPUT_LOGITS_Y;
  if (u < 1) {
    return EMBED_Y + (bayCenterY(1) - EMBED_Y) * u;
  }
  if (u > 52) {
    return bayCenterY(52) + (OUTPUT_LOGITS_Y - bayCenterY(52)) * (u - 52);
  }
  const lo = Math.floor(u);
  const hi = Math.min(52, lo + 1);
  const frac = u - lo;
  return bayCenterY(lo) + (bayCenterY(hi) - bayCenterY(lo)) * frac;
}

/**
 * Continuous world-Y mapping for instructional replay mode, synchronized exactly with
 * `describeSpineWaveStage` stage boundaries (Embedding < 0.05, Bays 01..52 in 0.05..0.91,
 * Final RMSNorm in 0.91..0.96, Output LM Head >= 0.96).
 */
function instructionalWaveProgressToWorldY(waveP: number, bayCount: number): number {
  if (waveP <= 0) return EMBED_Y;
  if (waveP >= 1) return OUTPUT_LOGITS_Y;

  const embedExitY = 0.5 * (EMBED_Y + bayCenterY(1));
  if (waveP < SPINE_STAGE_EMBED_END) {
    const t = waveP / SPINE_STAGE_EMBED_END;
    return EMBED_Y + (embedExitY - EMBED_Y) * t;
  }

  const normEntryY = 0.5 * (bayCenterY(bayCount) + FINAL_NORM_Y);
  if (waveP < SPINE_STAGE_BAYS_END) {
    const span = SPINE_STAGE_BAYS_END - SPINE_STAGE_EMBED_END;
    const bayCoord = 0.5 + ((waveP - SPINE_STAGE_EMBED_END) / span) * bayCount;
    if (bayCoord <= 1) {
      return embedExitY + (bayCenterY(1) - embedExitY) * ((bayCoord - 0.5) * 2);
    }
    if (bayCoord >= bayCount) {
      return bayCenterY(bayCount) + (normEntryY - bayCenterY(bayCount)) * ((bayCoord - bayCount) * 2);
    }
    const lo = Math.floor(bayCoord);
    const hi = Math.min(bayCount, lo + 1);
    const frac = bayCoord - lo;
    return bayCenterY(lo) + (bayCenterY(hi) - bayCenterY(lo)) * frac;
  }

  const normExitY = 0.5 * (FINAL_NORM_Y + LM_HEAD_Y);
  if (waveP < SPINE_STAGE_NORM_END) {
    const t = (waveP - SPINE_STAGE_BAYS_END) / (SPINE_STAGE_NORM_END - SPINE_STAGE_BAYS_END);
    if (t <= 0.5) {
      return normEntryY + (FINAL_NORM_Y - normEntryY) * (t * 2);
    }
    return FINAL_NORM_Y + (normExitY - FINAL_NORM_Y) * ((t - 0.5) * 2);
  }

  const t = (waveP - SPINE_STAGE_NORM_END) / (1 - SPINE_STAGE_NORM_END);
  if (t <= 0.5) {
    return normExitY + (LM_HEAD_Y - normExitY) * (t * 2);
  }
  return LM_HEAD_Y + (OUTPUT_LOGITS_Y - LM_HEAD_Y) * ((t - 0.5) * 2);
}

/**
 * Build the vertical top-to-bottom 52-layer Nemotron 3 Nano 3D tower in llm-viz visual language.
 */
export function buildMacroSpine(options: BuildTowerOptions): MacroSceneGeometry {
  const { spec, tokens, precision, selectedBay, hoveredBay, propagationWave } = options;
  const clampedTokens = Math.max(1, Math.min(16, Math.round(tokens)));
  const ssmPhase = clampedTokens; // deterministic per-token cell hash phase
  const isFp8Mode = precision === 'fp8_selective';

  const waveActive = propagationWave != null && Number.isFinite(propagationWave.progress);
  const waveP = waveActive ? clampProgress(propagationWave!.progress) : -1;
  const waveMode: SpinePropagationWaveMode =
    waveActive && propagationWave!.mode === 'instructional' ? 'instructional' : 'fast';
  const isInstructional = waveActive && waveMode === 'instructional';
  const bayWindow = isInstructional ? INSTRUCTIONAL_BAY_WINDOW : 2.2;
  const instructionalActiveBayId =
    isInstructional && waveP >= SPINE_STAGE_EMBED_END && waveP < SPINE_STAGE_BAYS_END
      ? spec.bays[instructionalBayIndexForProgress(waveP, spec.bays.length)].id
      : null;
  const waveBay = !waveActive
    ? -100
    : isInstructional
      ? waveP < SPINE_STAGE_EMBED_END
        ? -10
        : waveP >= SPINE_STAGE_BAYS_END
          ? spec.bays.length + 10
          : 0.5 +
            ((waveP - SPINE_STAGE_EMBED_END) / (SPINE_STAGE_BAYS_END - SPINE_STAGE_EMBED_END)) *
              spec.bays.length
      : waveP * 53;
  const waveY = !waveActive
    ? 0
    : isInstructional
      ? instructionalWaveProgressToWorldY(waveP, spec.bays.length)
      : waveProgressToWorldY(waveP);
  const waveToken = waveActive
    ? Math.max(1, Math.min(clampedTokens, Math.round(propagationWave!.activeTokenIndex ?? clampedTokens)))
    : clampedTokens;

  const solids: SolidSpec[] = [];
  const ribbons: RibbonSpec[] = [];
  const freeLines: LineSpec[] = [];
  const solidPickBoxes: PickBox[] = [];
  const fasciaPickBoxes: PickBox[] = [];
  const bayAnchors: BayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];

  const pushSolid = (s: SolidSpec, pickableBay: boolean) => {
    solids.push(s);
    if (pickableBay && s.bayId > 0) {
      const hx = s.ex * 0.5;
      const hy = s.ey * 0.5;
      const hz = s.ez * 0.5;
      solidPickBoxes.push({
        bayId: s.bayId,
        partId: s.pickPart,
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
      ownerId: 0,
      face0: -1, face1: -1,
      ocx: 0, ocy: 0, ocz: 0,
      oex: 0, oey: 0, oez: 0,
    });
  };

  // --- 1. Translucent Emerald-Green Vertical Residual Spine (Right side of tower, x = +4.6) ---
  const spineTopY = EMBED_Y;
  const spineBottomY = OUTPUT_LOGITS_Y;
  const spineHeight = spineTopY - spineBottomY;
  ribbons.push({
    cx: RESIDUAL_X,
    cy: 0.5 * (spineTopY + spineBottomY),
    cz: 0,
    ex: 0.42,
    ey: spineHeight,
    ez: 0.42,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.34,
  });

  // V4 52-layer residual-spine propagation wavefront ribbons (traversed wake + glowing wavefront pulse)
  if (waveActive) {
    const traversedHeight = spineTopY - waveY;
    if (traversedHeight > 0.35) {
      ribbons.push({
        cx: RESIDUAL_X,
        cy: 0.5 * (spineTopY + waveY),
        cz: 0,
        ex: 0.52,
        ey: traversedHeight,
        ez: 0.52,
        kind: RIBBON_KIND_BOX,
        colorId: RIBBON_COLOR_EMERALD,
        alpha: 0.46,
      });
    }
    // Glowing emerald wavefront pulse capsule centered on waveY
    ribbons.push({
      cx: RESIDUAL_X,
      cy: waveY,
      cz: 0,
      ex: isInstructional ? 0.82 : 0.74,
      ey: isInstructional ? 1.55 : 2.5,
      ez: isInstructional ? 0.82 : 0.74,
      kind: RIBBON_KIND_BOX,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: isInstructional ? 0.94 : 0.86,
    });
    // Leading downward directional emerald arrowhead along the spine
    if (waveY > spineBottomY + 1.2) {
      ribbons.push({
        cx: RESIDUAL_X,
        cy: Math.max(spineBottomY + 0.6, waveY - (isInstructional ? 1.25 : 1.55)),
        cz: 0,
        ex: 0.64,
        ey: isInstructional ? 0.96 : 1.15,
        ez: 0.64,
        kind: RIBBON_KIND_ARROW_NEG_Y,
        colorId: RIBBON_COLOR_EMERALD,
        alpha: 0.90,
      });
    }
  }

  // --- 2. Top Bookend: Embedding Matrix (131,072 × 2,688) ---
  const waveAtEmbed = waveActive && (isInstructional ? waveP < SPINE_STAGE_EMBED_END : waveP <= 0.08);
  pushSolid(
    {
      cx: 1.2, cy: EMBED_Y, cz: 0,
      ex: 4.6, ey: 1.9, ez: 1.1,
      cellFamily: CELL_WEIGHT_BLUE,
      fp8Flag: 0,
      phase: 1,
      bayId: 0,
      cellScale: 0.24,
      highlightFlag: waveAtEmbed ? 1 : 0,
      lineStyle: waveAtEmbed ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: waveAtEmbed ? 1.55 : 1.0,
      lineAlpha: waveAtEmbed ? 0.95 : 0.55,
    },
    false,
  );
  // Embedded token activation vector entering residual spine
  pushSolid(
    {
      cx: RESIDUAL_X, cy: EMBED_Y, cz: 0,
      ex: 0.56, ey: 1.45, ez: 0.72,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8Flag: 0,
      phase: clampedTokens,
      bayId: 0,
      cellScale: 0.20,
      highlightFlag: waveAtEmbed ? 1 : 0,
      lineStyle: waveAtEmbed ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: waveAtEmbed ? 1.65 : 1.0,
      lineAlpha: waveAtEmbed ? 1.0 : 0.55,
    },
    false,
  );
  ribbons.push({
    cx: 3.92, cy: EMBED_Y, cz: 0,
    ex: 0.84, ey: 0.42, ez: 0.38,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: waveAtEmbed ? 0.85 : 0.44,
  });

  landmarks.push({
    id: 'embed',
    label: 'Embedding · 131,072 × 2,688',
    world: [RESIDUAL_X + 0.8, EMBED_Y + 0.2, 0],
  });

  // --- 3. 52 Descending Bays (01 -> 52) ---
  for (const bay of spec.bays) {
    const cy = bayCenterY(bay.id);
    const isSelected = selectedBay === bay.id;
    const isHovered = hoveredBay === bay.id;
    const isActive = isSelected || isHovered;
    const bodyIsFp8 = bodyPrecision(bay, precision) === 'fp8';
    const isPreservedBf16 = isFp8Mode && bay.sensitive;

    const waveDist = waveActive ? Math.abs(bay.id - waveBay) : Infinity;
    const nearWave = waveActive && waveDist <= bayWindow;
    const onWaveCore = isInstructional
      ? bay.id === instructionalActiveBayId
      : waveActive && waveDist <= 1.25;
    const waveCrossed = isInstructional
      ? waveP >= SPINE_STAGE_BAYS_END ||
        (instructionalActiveBayId !== null && bay.id <= instructionalActiveBayId)
      : waveActive && waveBay >= bay.id - 0.35;

    const lineStyle = isActive
      ? LINE_STYLE_ACTIVE
      : isPreservedBf16
        ? LINE_STYLE_BF16_GOLD
        : LINE_STYLE_NEUTRAL;
    const cssWidth = isSelected ? 1.8 : isHovered ? 1.5 : isPreservedBf16 ? 1.35 : 0.9;
    const lineAlpha = isActive ? 1.0 : isPreservedBf16 ? 0.92 : bodyIsFp8 ? 0.32 : 0.52;
    const highlightFlag = isActive ? 1 : isPreservedBf16 ? 2 : 0;
    const fp8Flag = bodyIsFp8 ? 1 : 0;

    // Residual stream activation node on the right spine
    const resNodeActive = isActive || (isInstructional ? onWaveCore : nearWave);
    pushSolid(
      {
        cx: RESIDUAL_X, cy, cz: 0,
        ex: 0.46, ey: 0.92, ez: 0.58,
        cellFamily: CELL_ACTIVATION_GREEN,
        fp8Flag: 0,
        phase: bay.id + clampedTokens * 0.3,
        bayId: bay.id,
        cellScale: 0.20,
        highlightFlag: resNodeActive ? 1 : 0,
        lineStyle: resNodeActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        cssWidth: isActive
          ? 1.6
          : onWaveCore
            ? isInstructional
              ? 1.95
              : 1.8
            : nearWave
              ? isInstructional
                ? 1.2
                : 1.4
              : 0.85,
        lineAlpha: resNodeActive ? 1.0 : isInstructional && nearWave ? 0.72 : 0.45,
        pickPart: 'residual-node',
      },
      true,
    );

    // Pre-RMSNorm amber wafer between residual spine and layer body
    pushSolid(
      {
        cx: 3.25, cy, cz: 0,
        ex: 0.24, ey: 0.86, ez: 0.68,
        cellFamily: CELL_MOE_AMBER,
        fp8Flag,
        phase: bay.id,
        bayId: bay.id,
        cellScale: 0.18,
        highlightFlag: onWaveCore ? 1 : highlightFlag,
        lineStyle: onWaveCore ? LINE_STYLE_ACTIVE : lineStyle,
        cssWidth: onWaveCore ? (isInstructional ? 1.7 : 1.55) : cssWidth,
        lineAlpha: onWaveCore ? (isInstructional ? 0.98 : 0.96) : lineAlpha,
        pickPart: 'rmsnorm',
      },
      true,
    );

    // Emerald tap ribbon connecting residual spine <-> pre-norm
    ribbons.push({
      cx: 3.88, cy, cz: 0,
      ex: 0.98,
      ey: onWaveCore ? (isInstructional ? 0.44 : 0.36) : 0.28,
      ez: onWaveCore ? (isInstructional ? 0.40 : 0.34) : 0.26,
      kind: RIBBON_KIND_BOX,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: onWaveCore
        ? isInstructional
          ? 0.94
          : 0.88
        : nearWave
          ? isInstructional
            ? 0.52
            : 0.68
          : isActive
            ? 0.62
            : bodyIsFp8
              ? 0.18
              : 0.36,
    });

    if (bay.kind === 'M') {
      // Mamba-2 (×23): Input Projection (blue) + Conv1D strip (always BF16!) + Square Cyan SSM State + Output Projection (blue)
      pushSolid(
        {
          cx: 1.85, cy, cz: 0,
          ex: 1.35, ey: 0.86, ez: 0.78,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag,
          phase: bay.id * 1.1,
          bayId: bay.id,
          cellScale: 0.22,
          highlightFlag,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'mamba-in-proj',
        },
        true,
      );

      // Conv1D causal band (6,144 × 4) — ALWAYS BF16 even in FP8 Selective!
      const convGold = isFp8Mode && !isActive && !onWaveCore;
      const convHot = isActive || onWaveCore;
      pushSolid(
        {
          cx: 0.45, cy, cz: 0.08,
          ex: 0.56, ey: 0.96, ez: 0.94,
          cellFamily: CELL_ACTIVATION_GREEN,
          fp8Flag: 0,
          phase: ssmPhase + bay.id * 0.2,
          bayId: bay.id,
          cellScale: 0.20,
          highlightFlag: convHot ? 1 : convGold ? 2 : 0,
          lineStyle: convHot ? LINE_STYLE_ACTIVE : convGold ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
          cssWidth: convHot ? (isInstructional && onWaveCore ? 1.85 : 1.7) : convGold ? 1.2 : 0.95,
          lineAlpha: convHot ? 1.0 : convGold ? 0.82 : 0.55,
          pickPart: 'conv1d',
        },
        true,
      );

      // Square Cyan-Speckled SSM State Register (64 × 64 × 128, FP32 storage, phase steps with T!)
      const ssmPulsed = isActive || (isInstructional ? onWaveCore : nearWave);
      pushSolid(
        {
          cx: -1.35, cy, cz: 0,
          ex: 1.50, ey: 1.10, ez: 1.10,
          cellFamily: CELL_SSM_CYAN,
          fp8Flag: 0, // SSM state is always FP32
          phase: ssmPhase + bay.id * 0.37 + (isInstructional && onWaveCore ? 0.5 : 0),
          bayId: bay.id,
          cellScale: 0.20,
          highlightFlag: ssmPulsed ? 1 : highlightFlag,
          lineStyle: ssmPulsed ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: ssmPulsed ? (isInstructional && onWaveCore ? 2.0 : 1.8) : isPreservedBf16 ? 1.45 : 1.0,
          lineAlpha: ssmPulsed ? 1.0 : isPreservedBf16 ? 0.95 : 0.62,
          pickPart: 'ssm-register',
        },
        true,
      );

      if (onWaveCore) {
        ribbons.push({
          cx: -0.45, cy, cz: 0.04,
          ex: 1.10,
          ey: isInstructional ? 0.56 : 0.44,
          ez: isInstructional ? 0.52 : 0.44,
          kind: RIBBON_KIND_ARROW_NEG_X,
          colorId: RIBBON_COLOR_CYAN,
          alpha: isInstructional ? 0.88 : 0.68,
        });
        if (isInstructional) {
          ribbons.push({
            cx: -2.42, cy, cz: 0.04,
            ex: 0.80, ey: 0.44, ez: 0.40,
            kind: RIBBON_KIND_ARROW_NEG_X,
            colorId: RIBBON_COLOR_CYAN,
            alpha: 0.78,
          });
        }
      }

      // Gated Output Projection (4,096 -> 2,688)
      pushSolid(
        {
          cx: -3.45, cy, cz: 0,
          ex: 1.35, ey: 0.86, ez: 0.78,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag,
          phase: bay.id * 2.3,
          bayId: bay.id,
          cellScale: 0.22,
          highlightFlag,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'mamba-out-proj',
        },
        true,
      );

      // Translucent periwinkle directional arrow from in-proj into Conv1D/SSM
      ribbons.push({
        cx: 2.80, cy, cz: 0,
        ex: 0.55, ey: 0.32, ez: 0.28,
        kind: RIBBON_KIND_ARROW_NEG_X,
        colorId: RIBBON_COLOR_PERIWINKLE,
        alpha: onWaveCore ? 0.78 : bodyIsFp8 ? 0.20 : 0.44,
      });

      bayAnchors.push({
        id: bay.id,
        kind: bay.kind,
        pillWorld: [RESIDUAL_X + 0.95, cy, 0],
        centerWorld: [-0.8, cy, 0],
        bounds: {
          min: [-4.4, cy - 0.8, -0.8],
          max: [5.0, cy + 0.8, 0.9],
        },
      });

      fasciaPickBoxes.push({
        bayId: bay.id,
        partId: 'fascia',
        min: [-4.4, cy - BAY_PITCH_Y * 0.48, -0.7],
        max: [5.0, cy + BAY_PITCH_Y * 0.48, 0.75],
      });
    } else if (bay.kind === 'E') {
      // MoE (×23): Double-width Shared Expert (3,712) + Sigmoid Router (amber) + Tiled Bank of Routed Experts (6 active amber!)
      const moeFlashed = isActive || (isInstructional ? onWaveCore : nearWave);
      // Double-width shared expert slab connected cleanly beside pre-norm
      pushSolid(
        {
          cx: 1.55, cy, cz: 0.12,
          ex: 1.95, ey: 0.84, ez: 0.96,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag,
          phase: bay.id * 1.7,
          bayId: bay.id,
          cellScale: 0.22,
          highlightFlag: onWaveCore ? 1 : highlightFlag,
          lineStyle: onWaveCore ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: onWaveCore ? 1.55 : cssWidth,
          lineAlpha: onWaveCore ? 0.96 : lineAlpha,
          pickPart: 'shared-expert',
        },
        true,
      );

      // 128-score Sigmoid Router (warm amber)
      pushSolid(
        {
          cx: -0.05, cy, cz: 0,
          ex: 0.42, ey: 0.92, ez: 0.84,
          cellFamily: CELL_MOE_AMBER,
          fp8Flag,
          phase: bay.id + clampedTokens,
          bayId: bay.id,
          cellScale: 0.18,
          highlightFlag: moeFlashed ? 1 : highlightFlag,
          lineStyle: moeFlashed ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: moeFlashed ? 1.65 : cssWidth,
          lineAlpha: moeFlashed ? 1.0 : lineAlpha,
          pickPart: 'router',
        },
        true,
      );

      // Tiled bank of 8 visible routed expert slabs (2 rows in Z x 4 columns in X)
      // Highlight 6 active routed experts in warm amber (#F59E0B), alternating deterministically with T!
      const activeOffset = (bay.id + clampedTokens) % 8;
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 4; c++) {
          const tileIdx = r * 4 + c;
          const isRoutedActive = tileIdx !== activeOffset && tileIdx !== (activeOffset + 4) % 8;
          const tileFlashed = moeFlashed && isRoutedActive;
          const tx = -1.10 - c * 1.15;
          const tz = r === 0 ? -0.44 : 0.44;
          pushSolid(
            {
              cx: tx,
              cy: cy + (isRoutedActive ? 0.08 : 0),
              cz: tz,
              ex: 0.92,
              ey: isRoutedActive ? 0.88 : 0.72,
              ez: 0.66,
              cellFamily: isRoutedActive ? CELL_MOE_AMBER : CELL_WEIGHT_BLUE,
              fp8Flag,
              phase: bay.id * 3 + tileIdx + clampedTokens * 0.5,
              bayId: bay.id,
              cellScale: 0.21,
              highlightFlag: tileFlashed ? 1 : highlightFlag,
              lineStyle: tileFlashed ? LINE_STYLE_ACTIVE : lineStyle,
              cssWidth: isActive ? 1.6 : tileFlashed ? 1.45 : 0.85,
              lineAlpha: isActive ? 1.0 : tileFlashed ? 0.95 : bodyIsFp8 ? 0.28 : 0.48,
              pickPart: `expert-tile-${tileIdx}`,
            },
            true,
          );
        }
      }

      if (onWaveCore) {
        ribbons.push({
          cx: -0.58, cy, cz: 0,
          ex: 0.76,
          ey: isInstructional ? 0.48 : 0.36,
          ez: isInstructional ? 0.44 : 0.36,
          kind: RIBBON_KIND_ARROW_NEG_X,
          colorId: RIBBON_COLOR_AMBER,
          alpha: isInstructional ? 0.88 : 0.70,
        });
        if (isInstructional) {
          ribbons.push({
            cx: -2.82, cy: cy + 0.08, cz: 0,
            ex: 4.35, ey: 0.28, ez: 1.50,
            kind: RIBBON_KIND_BOX,
            colorId: RIBBON_COLOR_AMBER,
            alpha: 0.30,
          });
        }
      }

      // Periwinkle block arrow from pre-norm into shared + router
      ribbons.push({
        cx: 2.82, cy, cz: 0.10,
        ex: 0.52, ey: 0.30, ez: 0.28,
        kind: RIBBON_KIND_ARROW_NEG_X,
        colorId: RIBBON_COLOR_PERIWINKLE,
        alpha: onWaveCore ? 0.76 : bodyIsFp8 ? 0.18 : 0.42,
      });

      bayAnchors.push({
        id: bay.id,
        kind: bay.kind,
        pillWorld: [RESIDUAL_X + 0.95, cy, 0],
        centerWorld: [-1.4, cy, 0],
        bounds: {
          min: [-5.6, cy - 0.8, -1.0],
          max: [5.0, cy + 0.8, 1.0],
        },
      });

      fasciaPickBoxes.push({
        bayId: bay.id,
        partId: 'fascia',
        min: [-5.4, cy - BAY_PITCH_Y * 0.48, -0.9],
        max: [5.0, cy + BAY_PITCH_Y * 0.48, 0.9],
      });
    } else {
      // GQA Attention (*, ×6 — bays 06, 13, 20, 27, 34, 43):
      // Multi-plate periwinkle-blue Q/K/V weight stack + O projection + TWO paired KV cache token tracks growing along +Z!
      // Body always stays BF16 (fp8Flag = 0), highlighted in gold in FP8 Selective mode!
      const gqaHot = isActive || onWaveCore;

      // Q projection stack (32 Q heads = 4,096) — tall dramatic periwinkle-blue plate
      pushSolid(
        {
          cx: 1.15, cy, cz: -0.25,
          ex: 2.35, ey: 1.24, ez: 0.88,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag: 0,
          phase: bay.id * 1.3,
          bayId: bay.id,
          cellScale: 0.22,
          highlightFlag: gqaHot ? 1 : highlightFlag,
          lineStyle: gqaHot ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: gqaHot ? 1.85 : isPreservedBf16 ? 1.45 : 1.05,
          lineAlpha: gqaHot ? 1.0 : isPreservedBf16 ? 0.95 : 0.62,
          pickPart: 'q-proj',
        },
        true,
      );

      // K & V projection plates (2 KV heads = 256 each)
      pushSolid(
        {
          cx: -1.10, cy, cz: -0.42,
          ex: 1.35, ey: 1.12, ez: 0.56,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag: 0,
          phase: bay.id * 1.9,
          bayId: bay.id,
          cellScale: 0.21,
          highlightFlag: gqaHot ? 1 : highlightFlag,
          lineStyle: gqaHot ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: gqaHot ? 1.8 : isPreservedBf16 ? 1.4 : 1.0,
          lineAlpha: gqaHot ? 1.0 : isPreservedBf16 ? 0.95 : 0.60,
          pickPart: 'k-proj',
        },
        true,
      );
      pushSolid(
        {
          cx: -1.10, cy, cz: 0.34,
          ex: 1.35, ey: 1.12, ez: 0.56,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag: 0,
          phase: bay.id * 2.1,
          bayId: bay.id,
          cellScale: 0.21,
          highlightFlag: gqaHot ? 1 : highlightFlag,
          lineStyle: gqaHot ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: gqaHot ? 1.8 : isPreservedBf16 ? 1.4 : 1.0,
          lineAlpha: gqaHot ? 1.0 : isPreservedBf16 ? 0.95 : 0.60,
          pickPart: 'v-proj',
        },
        true,
      );

      // Output projection plate O (4,096 -> 2,688) extending prominently left
      pushSolid(
        {
          cx: -4.25, cy, cz: -0.08,
          ex: 2.35, ey: 1.24, ez: 0.88,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag: 0,
          phase: bay.id * 2.7,
          bayId: bay.id,
          cellScale: 0.22,
          highlightFlag: gqaHot ? 1 : highlightFlag,
          lineStyle: gqaHot ? LINE_STYLE_ACTIVE : lineStyle,
          cssWidth: gqaHot ? 1.85 : isPreservedBf16 ? 1.45 : 1.05,
          lineAlpha: gqaHot ? 1.0 : isPreservedBf16 ? 0.95 : 0.62,
          pickPart: 'o-proj',
        },
        true,
      );

      // Translucent periwinkle directional arrows through Q/K/V -> O
      ribbons.push({
        cx: 2.74, cy, cz: -0.15,
        ex: 0.64, ey: 0.36, ez: 0.32,
        kind: RIBBON_KIND_ARROW_NEG_X,
        colorId: RIBBON_COLOR_PERIWINKLE,
        alpha: onWaveCore ? 0.82 : 0.48,
      });
      ribbons.push({
        cx: -2.42, cy, cz: -0.08,
        ex: 0.88, ey: 0.36, ez: 0.32,
        kind: RIBBON_KIND_ARROW_NEG_X,
        colorId: RIBBON_COLOR_PERIWINKLE,
        alpha: onWaveCore ? 0.82 : 0.48,
      });

      if (isInstructional ? onWaveCore : nearWave) {
        ribbons.push({
          cx: -0.30, cy: cy + 0.08, cz: kvSliceCenterZ(waveToken) - 0.55,
          ex: isInstructional ? 0.62 : 0.46,
          ey: isInstructional ? 0.48 : 0.36,
          ez: 0.95,
          kind: RIBBON_KIND_ARROW_POS_Z,
          colorId: RIBBON_COLOR_CYAN,
          alpha: isInstructional ? 0.92 : onWaveCore ? 0.80 : 0.58,
        });
      }

      // TWO paired KV-cache tracks (K1, V1 | K2, V2) growing outward along +Z (tokens 1..T)
      const kvFp8 = kvCachePrecision(bay, precision) === 'fp8' ? 1 : 0;
      const trackCenterZ = KV_TRACK_START_Z + KV_TRACK_LENGTH * 0.5;

      for (let r = 0; r < KV_RAIL_OFFSETS_X.length; r++) {
        const rx = -0.30 + KV_RAIL_OFFSETS_X[r];
        const isKTrack = r % 2 === 0;

        // Fixed 16-token capacity guide rail
        pushSolid(
          {
            cx: rx, cy: cy - 0.32, cz: trackCenterZ,
            ex: 0.09, ey: 0.08, ez: KV_TRACK_LENGTH,
            cellFamily: CELL_NEUTRAL,
            fp8Flag: 0,
            phase: 0,
            bayId: bay.id,
            cellScale: 0.5,
            highlightFlag: 0,
            lineStyle: LINE_STYLE_GUIDE,
            cssWidth: isActive ? 0.95 : 0.75,
            lineAlpha: isActive ? 0.65 : 0.42,
            pickPart: `kv-rail-${r}`,
          },
          true,
        );

        // Active token slices j = 1..T flush atop guide rail (rail top = cy - 0.28, slice halfH = 0.26 -> slice cy = cy - 0.02)
        for (let j = 1; j <= clampedTokens; j++) {
          const isNewest = j === clampedTokens;
          const waveAppendedSlice = waveCrossed && j === waveToken;
          const highlightSlice = (isActive && isNewest) || waveAppendedSlice;
          pushSolid(
            {
              cx: rx,
              cy: cy - 0.02,
              cz: kvSliceCenterZ(j),
              ex: 0.36,
              ey: 0.52,
              ez: KV_SLICE_DEPTH,
              cellFamily: isKTrack ? CELL_ACTIVATION_GREEN : CELL_SSM_CYAN,
              fp8Flag: kvFp8,
              phase: j + bay.id * 0.13 + r * 0.5,
              bayId: bay.id,
              cellScale: 0.17,
              highlightFlag: highlightSlice ? 1 : 0,
              lineStyle: highlightSlice ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
              cssWidth: highlightSlice ? 1.55 : isNewest ? 1.05 : 0.85,
              lineAlpha: highlightSlice ? 0.98 : kvFp8 ? 0.40 : 0.58,
              pickPart: `kv-slice-${r}-${j}`,
            },
            true,
          );
        }
      }

      // Depth scale guide wire alongside the right of KV tracks (at x = +1.35)
      const scaleX = 1.35;
      const scaleY = cy - 0.32;
      pushFreeLine(
        scaleX, scaleY, KV_TRACK_START_Z,
        scaleX, scaleY, KV_TRACK_START_Z + KV_TRACK_LENGTH,
        0.9, LINE_STYLE_GUIDE, 0.55,
      );
      pushFreeLine(
        scaleX, scaleY, KV_TRACK_START_Z,
        scaleX, scaleY, kvSliceFrontZ(clampedTokens),
        1.5, LINE_STYLE_ACTIVE, 0.90,
      );
      for (const tickT of [1, 8, 16, clampedTokens]) {
        const tz = kvSliceFrontZ(tickT);
        const isCur = tickT === clampedTokens;
        pushFreeLine(
          scaleX, scaleY, tz,
          scaleX + (isCur ? 0.45 : 0.30), scaleY, tz,
          isCur ? 1.5 : 0.9,
          isCur ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
          isCur ? 0.95 : 0.65,
        );
      }

      bayAnchors.push({
        id: bay.id,
        kind: bay.kind,
        pillWorld: [RESIDUAL_X + 0.95, cy, 0],
        centerWorld: [-1.1, cy, 2.0],
        bounds: {
          min: [-5.8, cy - 1.1, -1.0],
          max: [5.2, cy + 1.1, KV_TRACK_START_Z + KV_TRACK_LENGTH + 0.5],
        },
      });

      landmarks.push({
        id: `gqa-${bay.id}`,
        label: `${String(bay.id).padStart(2, '0')} · GQA`,
        world: [RESIDUAL_X + 0.85, cy, 0],
        bayId: bay.id,
      });

      fasciaPickBoxes.push({
        bayId: bay.id,
        partId: 'fascia',
        min: [-5.8, cy - BAY_PITCH_Y * 0.48, -0.9],
        max: [5.0, cy + BAY_PITCH_Y * 0.48, KV_TRACK_START_Z + KV_TRACK_LENGTH],
      });
    }
  }

  // --- 4. Subtle Horizontal Figure-2 Phase Divider Lines ---
  const dividerBays = [35, 41, 44];
  for (const prevBay of dividerBays) {
    const midY = 0.5 * (bayCenterY(prevBay) + bayCenterY(prevBay + 1));
    pushFreeLine(-6.2, midY, 0, RESIDUAL_X + 0.6, midY, 0, 1.0, LINE_STYLE_GUIDE, 0.45);
  }

  // --- 5. Bottom Bookends: Final RMSNorm -> Untied LM Head -> Output Logits (131,072) ---
  const waveAtFinalNorm =
    waveActive &&
    (isInstructional ? waveP >= SPINE_STAGE_BAYS_END && waveP < SPINE_STAGE_NORM_END : waveP >= 0.94);
  const waveAtOutput =
    waveActive && (isInstructional ? waveP >= SPINE_STAGE_NORM_END : waveP >= 0.97);
  pushSolid(
    {
      cx: 1.6, cy: FINAL_NORM_Y, cz: 0,
      ex: 3.4, ey: 0.65, ez: 0.92,
      cellFamily: CELL_MOE_AMBER,
      fp8Flag: 0,
      phase: 99,
      bayId: 0,
      cellScale: 0.19,
      highlightFlag: waveAtFinalNorm ? 1 : 0,
      lineStyle: waveAtFinalNorm ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: waveAtFinalNorm ? 1.6 : 1.0,
      lineAlpha: waveAtFinalNorm ? 0.98 : 0.55,
    },
    false,
  );
  ribbons.push({
    cx: 3.85, cy: FINAL_NORM_Y, cz: 0,
    ex: 1.0, ey: 0.34, ez: 0.32,
    kind: RIBBON_KIND_ARROW_NEG_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: waveAtFinalNorm ? 0.85 : 0.42,
  });

  // LM Head (Untied): 2,688 -> 131,072 periwinkle-blue weight slab
  pushSolid(
    {
      cx: 0.4, cy: LM_HEAD_Y, cz: 0,
      ex: 5.4, ey: 1.75, ez: 1.05,
      cellFamily: CELL_WEIGHT_BLUE,
      fp8Flag: 0,
      phase: 101,
      bayId: 0,
      cellScale: 0.24,
      highlightFlag: waveAtOutput ? 1 : 0,
      lineStyle: waveAtOutput ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: waveAtOutput ? 1.6 : 1.0,
      lineAlpha: waveAtOutput ? 0.96 : 0.55,
    },
    false,
  );
  ribbons.push({
    cx: 1.5, cy: 0.5 * (FINAL_NORM_Y + LM_HEAD_Y), cz: 0,
    ex: 0.55, ey: 0.95, ez: 0.34,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: waveAtOutput ? 0.86 : 0.48,
  });

  // Output Logits (131,072) emerald-green activation plate
  pushSolid(
    {
      cx: 1.1, cy: OUTPUT_LOGITS_Y, cz: 0,
      ex: 4.4, ey: 1.15, ez: 0.96,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8Flag: 0,
      phase: clampedTokens + 7,
      bayId: 0,
      cellScale: 0.22,
      highlightFlag: waveAtOutput ? 1 : 0,
      lineStyle: waveAtOutput ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: waveAtOutput ? 1.7 : 1.0,
      lineAlpha: waveAtOutput ? 1.0 : 0.55,
    },
    false,
  );
  ribbons.push({
    cx: 1.5, cy: 0.5 * (LM_HEAD_Y + OUTPUT_LOGITS_Y), cz: 0,
    ex: 0.55, ey: 0.90, ez: 0.34,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: waveAtOutput ? 0.88 : 0.45,
  });

  landmarks.push(
    {
      id: 'final-norm',
      label: 'Final RMSNorm',
      world: [RESIDUAL_X + 0.85, FINAL_NORM_Y, 0],
    },
    {
      id: 'output',
      label: 'Output · 131,072',
      world: [RESIDUAL_X + 0.85, OUTPUT_LOGITS_Y, 0],
    },
  );

  // --- 6. Phase section anchors (4 NVIDIA Figure-2 spans) ---
  const phaseSections: PhaseSectionAnchor[] = spec.brackets.map((b, idx) => {
    const [first, last] = b;
    const topY = bayCenterY(first) + 0.65;
    const bottomY = bayCenterY(last) - 0.65;
    return {
      index: idx as 0 | 1 | 2 | 3,
      first,
      last,
      topY,
      bottomY,
      midWorld: [-6.4, 0.5 * (topY + bottomY), 0],
    };
  });

  // --- 7. Depth tick label anchors beside GQA Bay 06 (topmost attention layer) ---
  const refGqaY = bayCenterY(6) - 0.32;
  const kvDepthTicks: KvDepthTickAnchor[] = [1, 8, 16, clampedTokens]
    .filter((v, i, arr) => arr.indexOf(v) === i)
    .map((token) => ({
      token,
      world: [1.95, refGqaY, kvSliceFrontZ(token)],
    }));

  // --- 8. Pack GPU Instance Buffers ---
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
    emitCuboidEdges(s, edgeDedup);
  }

  // Order line runs: neutral/guide first, BF16 gold second, active selected/hovered last
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

  return {
    solidData,
    solidCount: solids.length,
    lineData,
    lineCount: orderedLines.length,
    ribbonData,
    ribbonCount: ribbons.length,
    fitBounds: TOWER_FIT_BOUNDS,
    bayAnchors,
    phaseSections,
    landmarks,
    kvDepthTicks,
    solidPickBoxes,
    fasciaPickBoxes,
  };
}

/** Named bounds for camera presets / TOC fly-to targets. */
export function boundsForSection(
  section: 'overview' | 'embedding' | 'mamba' | 'moe' | 'gqa' | 'kv_fp8' | 'output',
): AABB {
  switch (section) {
    case 'overview':
      return TOWER_FIT_BOUNDS;
    case 'embedding':
      return {
        min: [-3.5, bayCenterY(3) - 1.0, -1.6],
        max: [6.0, EMBED_Y + 2.0, 2.0],
      };
    case 'mamba': {
      const y = bayCenterY(1);
      return {
        min: [-5.0, y - 2.6, -1.6],
        max: [5.6, y + 2.6, 2.0],
      };
    }
    case 'moe': {
      const y = bayCenterY(2);
      return {
        min: [-6.0, y - 2.6, -1.6],
        max: [5.6, y + 2.6, 2.0],
      };
    }
    case 'gqa': {
      const y = bayCenterY(6);
      return {
        min: [-6.4, y - 2.3, -1.2],
        max: [5.6, y + 2.3, KV_TRACK_START_Z + KV_TRACK_LENGTH + 0.8],
      };
    }
    case 'kv_fp8': {
      // Frame Feeder Bay 42 + Attention Bay 43 + KV tracks together
      const yTop = bayCenterY(41);
      const yBot = bayCenterY(44);
      return {
        min: [-6.4, yBot - 1.5, -1.4],
        max: [5.6, yTop + 1.5, KV_TRACK_START_Z + KV_TRACK_LENGTH + 0.8],
      };
    }
    case 'output':
      return {
        min: [-4.5, OUTPUT_LOGITS_Y - 1.8, -1.6],
        max: [5.8, BAY_52_Y + 2.2, 2.0],
      };
  }
}

export function boundsForBay(bayId: number): AABB {
  const y = bayCenterY(bayId);
  return {
    min: [-6.0, y - 2.6, -1.4],
    max: [5.6, y + 2.6, 6.2],
  };
}
