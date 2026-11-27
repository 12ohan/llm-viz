/**
 * NLD-Uni Spatial Provenance, Pick Adapter, Atlas Auditor & 5-Pass Buffer Builders
 * (`src/uni/scene/provenance.ts`)
 *
 * Shared Step-0 foundation consumed across all 7 NLD-Uni spatial views (`UNI4` & `UNI5`):
 *   - Non-mutating `resolveUniSessionContext` backed by 0-ULP `reconstructLayerTrace`
 *     & `reconstructStepHeadLogits`
 *   - `pickHitToUniCell` decoding `groupIndex` (KV head / CFG row), `headIndex`,
 *     `tapIndex` (rotary plane / tree depth), and `semanticTag` with ZERO edits to `src/render/pick.ts`
 *   - GPU buffer builders (`UniSolidBufferBuilder`, `UniLineBufferBuilder`,
 *     `UniRibbonBufferBuilder`, `UniFlatFlowBufferBuilder`, `UniSheetBufferBuilder`)
 *   - Hard `U25` enforcement: throws if `sheetKind ∈ {1, 2}` lacks a live `RG32F` `TexRect`
 *   - Exact `auditAtlasRegions` (0 overlaps, in-bounds, `<= maxBytes`) and
 *     `validateSceneGeometryFinite` (finite floats, 0 overlaps, `U25` compliance)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import type { Vec3 } from '../../model/spec';
import type { AABB } from '../../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  pushBoxFrameXZ,
  pushDashedFrameXZ,
  pushFreeGuideLine,
} from '../../render/linePacking';
import type { PickBox, PickHit } from '../../render/pick';
import {
  FlowRibbonBuilder,
  type FlowRibbonColor,
  type XZPoint,
} from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_BASIS_FLOATS,
  SHEET_HL_NORMAL,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_ROLE_NEUTRAL,
} from '../../render/sheets';
import type { AtlasSlice, Rg32fValueAtlas, TexRect } from '../../scene/atlasPacker';
import {
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  RIBBON_INSTANCE_FLOATS,
  SOLID_INSTANCE_FLOATS,
} from '../../scene/spine';
import {
  buildUniElasticD5Comparison,
  buildUniViewArchReadouts,
} from '../archReadouts';
import { dualHashHex } from '../prng';
import type { UniReconstructedHeadLogits, UniSession } from '../session';
import {
  UNI_MAX_ATLAS_BYTES,
  UNI_PROJECTION_REVERSAL_BADGE,
  UNI_VISUAL_QUALITY_HONESTY_BADGE,
} from '../spec';
import {
  UNI_MINI_CONFIG,
  type UniExecutionViewId,
  type UniLayerTrace,
  type UniSessionTaskMode,
} from '../types';
import type {
  UniAtlasLayoutAudit,
  UniAtlasRegionBox,
  UniPickIdentity,
  UniProvenanceBadge,
  UniResolvedSessionContext,
  UniSceneValidationResult,
  UniSpatialScene,
  UniSpatialViewMode,
} from './types';

// ============================================================================
// AUTHORITATIVE EPISTEMIC BADGES & EDITORIAL DRAFTING PALETTE
// ============================================================================

export const UNI_SCENE_PROJECTION_REVERSAL_BADGE = UNI_PROJECTION_REVERSAL_BADGE.label;
export const UNI_SCENE_VISUAL_HONESTY_BADGE = UNI_VISUAL_QUALITY_HONESTY_BADGE.label;
export const UNI_SCENE_VISUAL_QUALITY_NOTE = UNI_VISUAL_QUALITY_HONESTY_BADGE.summary;
export const UNI_SCENE_INTERLEAVED_BADGE =
  'Kernel-Supported · No Released Multi-Turn Outer Loop' as const;

/** Mineral drafting palette tokens for FlatFlowRibbonBuilder & overlays (§9). */
export const UNI_FLOW_COLORS = Object.freeze({
  uIndigo: Object.freeze<FlowRibbonColor>({
    r: 0.22,
    g: 0.38,
    b: 0.74,
    a: 0.72,
    borderAlpha: 0.85,
  }),
  gTeal: Object.freeze<FlowRibbonColor>({
    r: 0.102,
    g: 0.541,
    b: 0.478,
    a: 0.74,
    borderAlpha: 0.85,
  }),
  jointCyan: Object.freeze<FlowRibbonColor>({
    r: 0.18,
    g: 0.62,
    b: 0.84,
    a: 0.68,
    borderAlpha: 0.8,
  }),
  wasteAmber: Object.freeze<FlowRibbonColor>({
    r: 0.92,
    g: 0.58,
    b: 0.08,
    a: 0.78,
    borderAlpha: 0.92,
  }),
  ghostSlate: Object.freeze<FlowRibbonColor>({
    r: 0.56,
    g: 0.60,
    b: 0.67,
    a: 0.28,
    borderAlpha: 0.35,
  }),
  activeOxide: Object.freeze<FlowRibbonColor>({
    r: 0.82,
    g: 0.24,
    b: 0.20,
    a: 0.84,
    borderAlpha: 0.95,
  }),
});

export function taskModeToExecutionViewId(taskMode: UniSessionTaskMode): UniExecutionViewId {
  return taskMode;
}

// ============================================================================
// NON-MUTATING SESSION CONTEXT RESOLUTION (0-ULP LAZY RECONSTRUCTION)
// ============================================================================

/**
 * Resolve a strictly read-only spatial view context from `UniSession` (active cursor or explicit
 * `recordId`), never mutating session state and caching 0-ULP lazy `reconstructLayerTrace` and
 * `reconstructStepHeadLogits` calls for the duration of a single scene build.
 */
export function resolveUniSessionContext(
  session: UniSession,
  recordId?: string,
): UniResolvedSessionContext {
  const targetRecordId = recordId ?? session.getActiveRecordId();
  if (targetRecordId === null) {
    throw new RangeError(
      'UniSession has no active branch or record; call startBranch() before building a spatial scene',
    );
  }
  const record = session.getRecord(targetRecordId);
  if (!record) {
    throw new RangeError(`Unknown UniSession recordId: "${targetRecordId}"`);
  }
  const branch = session.getBranch(record.branchId);
  if (!branch) {
    throw new RangeError(`Missing UniSession branch "${record.branchId}" for record "${targetRecordId}"`);
  }

  const executionViewId = taskModeToExecutionViewId(branch.taskMode);
  const recordIndexInBranch = branch.recordIds.indexOf(record.recordId);
  const branchRecordCount = branch.recordIds.length;
  const isHistoricalCursor =
    recordIndexInBranch >= 0 && recordIndexInBranch < branchRecordCount - 1;

  const viewReadouts = buildUniViewArchReadouts();
  const viewReadout =
    viewReadouts.find((v) => v.viewId === executionViewId) ?? viewReadouts[0]!;
  const prefixTokens = Math.max(1, branch.prefixCache.seqLen);
  const d5Comparison = buildUniElasticD5Comparison(executionViewId, prefixTokens);

  const badges: UniProvenanceBadge[] = [
    {
      label: `Task: ${viewReadout.label}`,
      tone:
        branch.taskMode === 'text_to_image' || branch.taskMode === 'image_edit'
          ? 'g_teal'
          : 'u_indigo',
    },
    {
      label:
        branch.prefillMode === 'elastic_pruned'
          ? 'D5: Elastic-Pruned (18L U + 34L G · 0 ULP)'
          : 'D5: HF-Faithful (34L U Prefill)',
      tone: branch.prefillMode === 'elastic_pruned' ? 'g_teal' : 'gold',
    },
    {
      label:
        branch.gainPresetId === 'prod_exact'
          ? 'D3 Gains: Production-Exact'
          : 'D3 Gains: Mini Default',
      tone: 'muted',
    },
    {
      label: UNI_SCENE_PROJECTION_REVERSAL_BADGE,
      tone: 'muted',
    },
  ];
  if (isHistoricalCursor) {
    badges.push({
      label: `Historical Cursor (${record.recordId})`,
      tone: 'gold',
    });
  }

  const layerTraceCache = new Map<string, UniLayerTrace>();
  let headLogitsCache: UniReconstructedHeadLogits | null = null;

  return {
    bundle: session.bundle,
    session,
    branch,
    record,
    executionViewId,
    prefillMode: branch.prefillMode,
    gainPresetId: branch.gainPresetId,
    cfgBatchSize: record.cfgBatchSize,
    isHistoricalCursor,
    recordIndexInBranch,
    branchRecordCount,
    resolveLayerTrace(layerIndex: number, batchRow = 0): UniLayerTrace {
      const clampedLayer = Math.max(
        0,
        Math.min(UNI_MINI_CONFIG.numLayers - 1, Math.trunc(layerIndex)),
      );
      const clampedRow = Math.max(
        0,
        Math.min(record.cfgBatchSize - 1, Math.trunc(batchRow)),
      );
      const key = `${clampedLayer}:${clampedRow}`;
      const cached = layerTraceCache.get(key);
      if (cached) return cached;
      const trace = session.reconstructLayerTrace(record.recordId, clampedLayer, clampedRow);
      layerTraceCache.set(key, trace);
      return trace;
    },
    resolveStepHeadLogits(): UniReconstructedHeadLogits {
      if (headLogitsCache === null) {
        headLogitsCache = session.reconstructStepHeadLogits(record.recordId);
      }
      return headLogitsCache;
    },
    provenanceBadges: Object.freeze(badges),
    d5Comparison,
    viewReadout,
  };
}

// ============================================================================
// PICK ADAPTER (ZERO EDITS TO `src/render/pick.ts`, §7.2 TIER 2 & U25)
// ============================================================================

function parseTagInt(tag: string | undefined, key: string): number | null {
  if (!tag) return null;
  const re = new RegExp(`(?:^|[;:,\\s])${key}=(-?\\d+)`);
  const m = re.exec(tag);
  if (!m) return null;
  const parsed = Number.parseInt(m[1]!, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseBranchFromHit(hit: PickHit): 'u' | 'g' | 'joint' | null {
  const tag = hit.semanticTag ?? '';
  if (/(?:^|[;,\s])branch=u(?:$|[;,\s])/.test(tag) || hit.partId?.startsWith('u_') || hit.partId?.includes('_u_')) {
    return 'u';
  }
  if (/(?:^|[;,\s])branch=g(?:$|[;,\s])/.test(tag) || hit.partId?.startsWith('g_') || hit.partId?.includes('_g_')) {
    return 'g';
  }
  if (/(?:^|[;,\s])branch=joint(?:$|[;,\s])/.test(tag) || hit.partId?.includes('joint') || hit.partId?.includes('coupler')) {
    return 'joint';
  }
  if (hit.towerRole === 'context') return 'u';
  if (hit.towerRole === 'denoiser') return 'g';
  if (hit.towerRole === 'shared') return 'joint';
  return null;
}

/**
 * Decode a `PickHit` from `src/render/pick.ts` into a full `UniPickIdentity` without modifying
 * `src/render/pick.ts`.
 *
 * Conventions:
 *   - `bayId in 1..34` -> zero-based `layerIndex = bayId - 1` (`isSplitLayer = layerIndex >= 18`).
 *   - `headIndex in 0..7` -> `qHeadIndex`.
 *   - `groupIndex`:
 *       * When `semanticTag` includes `cfg=...` or `groupRole=cfg`: decoded as `cfgBatchRow` (`0..2`).
 *       * Otherwise (`0..1`): decoded as `kvHeadIndex` (`0..1`), with optional `cfg=` tag also parsed.
 *   - `tapIndex`:
 *       * When `semanticTag` includes `tapRole=tree` or `viewMode === 'ball_tree'`: decoded as `treeDepth` (`0..8`).
 *       * Otherwise (`0..3`): decoded as `rotaryPlane` (`0..3`), or inferred from `hitCol % 4` on rotary sheets.
 */
export function pickHitToUniCell(
  hit: PickHit | null,
  viewMode: UniSpatialViewMode,
): UniPickIdentity | null {
  if (!hit) return null;

  const layerIndex =
    hit.bayId >= 1 && hit.bayId <= UNI_MINI_CONFIG.numLayers ? hit.bayId - 1 : null;
  const isSplitLayer =
    layerIndex !== null ? layerIndex >= UNI_MINI_CONFIG.dualTowerLayers : null;
  const branch = parseBranchFromHit(hit);
  const tag = hit.semanticTag ?? '';

  let qHeadIndex: number | null =
    hit.headIndex !== undefined &&
    hit.headIndex >= 0 &&
    hit.headIndex < UNI_MINI_CONFIG.numAttentionHeads
      ? hit.headIndex
      : null;

  const isCfgGroupRole = tag.includes('groupRole=cfg');
  let cfgBatchRow: number | null = parseTagInt(tag, 'cfg');
  if (cfgBatchRow === null && isCfgGroupRole && hit.groupIndex !== undefined) {
    if (hit.groupIndex >= 0 && hit.groupIndex <= 2) {
      cfgBatchRow = hit.groupIndex;
    }
  }

  let kvHeadIndex: number | null = parseTagInt(tag, 'kv');
  if (
    kvHeadIndex === null &&
    !isCfgGroupRole &&
    hit.groupIndex !== undefined &&
    hit.groupIndex >= 0 &&
    hit.groupIndex < UNI_MINI_CONFIG.numKeyValueHeads
  ) {
    kvHeadIndex = hit.groupIndex;
  }
  if (kvHeadIndex === null && hit.kvHeadIndex !== undefined) {
    kvHeadIndex = hit.kvHeadIndex;
  }
  if (kvHeadIndex === null && qHeadIndex !== null) {
    kvHeadIndex = Math.floor(qHeadIndex / UNI_MINI_CONFIG.kvGroupSize);
  }

  // Infer Q / KV head from column index on concatenated head sheets
  const tKey = hit.tensorKey ?? '';
  if (
    qHeadIndex === null &&
    hit.hitCol !== undefined &&
    (tKey === 'q_pre' ||
      tKey === 'q_rot' ||
      tKey === 'q_scaled' ||
      tKey === 'attn_concat' ||
      tKey === 'w_q_u' ||
      tKey === 'w_q_g' ||
      tKey === 'w_o_u' ||
      tKey === 'w_o_g')
  ) {
    const inferredQ = Math.floor(hit.hitCol / UNI_MINI_CONFIG.headDim);
    if (inferredQ >= 0 && inferredQ < UNI_MINI_CONFIG.numAttentionHeads) {
      qHeadIndex = inferredQ;
      if (kvHeadIndex === null) {
        kvHeadIndex = Math.floor(inferredQ / UNI_MINI_CONFIG.kvGroupSize);
      }
    }
  }
  if (
    kvHeadIndex === null &&
    hit.hitCol !== undefined &&
    (tKey === 'k_pre' ||
      tKey === 'k_rot' ||
      tKey === 'v_new' ||
      tKey === 'w_k_u' ||
      tKey === 'w_k_g' ||
      tKey === 'w_v_u' ||
      tKey === 'w_v_g')
  ) {
    const inferredKv = Math.floor(hit.hitCol / UNI_MINI_CONFIG.headDim);
    if (inferredKv >= 0 && inferredKv < UNI_MINI_CONFIG.numKeyValueHeads) {
      kvHeadIndex = inferredKv;
    }
  }

  const isTreeTap = tag.includes('tapRole=tree') || viewMode === 'ball_tree';
  let treeDepth: number | null = parseTagInt(tag, 'depth');
  if (
    treeDepth === null &&
    isTreeTap &&
    hit.tapIndex !== undefined &&
    hit.tapIndex >= 0 &&
    hit.tapIndex <= UNI_MINI_CONFIG.vqTreeDepth
  ) {
    treeDepth = hit.tapIndex;
  }

  let rotaryPlane: number | null = parseTagInt(tag, 'plane');
  if (
    rotaryPlane === null &&
    !isTreeTap &&
    hit.tapIndex !== undefined &&
    hit.tapIndex >= 0 &&
    hit.tapIndex < UNI_MINI_CONFIG.headDim / 2
  ) {
    rotaryPlane = hit.tapIndex;
  }
  if (
    rotaryPlane === null &&
    hit.hitCol !== undefined &&
    (tKey === 'q_pre' ||
      tKey === 'q_rot' ||
      tKey === 'q_scaled' ||
      tKey === 'k_pre' ||
      tKey === 'k_rot' ||
      tKey === 'rope_cos' ||
      tKey === 'rope_sin')
  ) {
    rotaryPlane = hit.hitCol % (UNI_MINI_CONFIG.headDim / 2);
  }

  const tokenIndex =
    hit.tokenIndex !== undefined
      ? hit.tokenIndex
      : hit.hitRow !== undefined
        ? hit.hitRow
        : null;

  const absPosition =
    hit.absoluteTokenOffset !== undefined ? hit.absoluteTokenOffset : null;

  return Object.freeze({
    viewMode,
    layerIndex,
    isSplitLayer,
    branch,
    partId: hit.partId ?? null,
    tensorKey: hit.tensorKey ?? null,
    qHeadIndex,
    kvHeadIndex,
    cfgBatchRow,
    rotaryPlane,
    treeDepth,
    tokenIndex,
    absPosition,
    row: hit.hitRow ?? null,
    col: hit.hitCol ?? null,
    semanticTag: hit.semanticTag ?? null,
  });
}

// ============================================================================
// PICK BOX CONVENIENCE BUILDERS
// ============================================================================

export interface UniPickBoxMeta {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly tensorKey?: string;
  readonly gridCols?: number;
  readonly gridRows?: number;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  readonly towerRole?: 'context' | 'denoiser' | 'shared';
  readonly executionId?: string;
  readonly traceOrigin?: string;
  readonly semanticTag?: string;
  readonly absoluteTokenOffset?: number;
}

export function packPickBoxFromAabb(
  cx: number,
  cy: number,
  cz: number,
  ex: number,
  ey: number,
  ez: number,
  meta: UniPickBoxMeta,
): PickBox {
  const hx = ex * 0.5;
  const hy = ey * 0.5;
  const hz = ez * 0.5;
  return {
    ...meta,
    min: [cx - hx, cy - hy, cz - hz],
    max: [cx + hx, cy + hy, cz + hz],
  };
}

export function packPickBoxFromFrame(frame: SheetFrame3D, meta: UniPickBoxMeta): PickBox {
  const aabb = sheetFrameWorldAabb(frame);
  return {
    ...meta,
    min: aabb.min,
    max: aabb.max,
    frame,
  };
}

// ============================================================================
// ATLAS NON-OVERLAP AUDITOR & CONTENT-ADDRESSED CACHE KEY
// ============================================================================

export function auditAtlasRegions(
  width: number,
  height: number,
  regions: readonly UniAtlasRegionBox[],
  maxBudgetBytes: number = UNI_MAX_ATLAS_BYTES,
): UniAtlasLayoutAudit {
  const totalTexels = width * height;
  const byteLength = totalTexels * 8; // RG32F = 2 * 4 B/texel
  const withinBudget = byteLength <= maxBudgetBytes;

  let usedTexels = 0;
  let hasOverlap = false;
  let overlapPair: readonly [string, string] | null = null;
  const outOfBoundsIds: string[] = [];

  for (let i = 0; i < regions.length; i++) {
    const a = regions[i]!;
    usedTexels += Math.max(0, a.cols) * Math.max(0, a.rows);
    if (
      !Number.isInteger(a.x0) ||
      !Number.isInteger(a.y0) ||
      !Number.isInteger(a.cols) ||
      !Number.isInteger(a.rows) ||
      a.x0 < 0 ||
      a.y0 < 0 ||
      a.cols <= 0 ||
      a.rows <= 0 ||
      a.x0 + a.cols > width ||
      a.y0 + a.rows > height
    ) {
      outOfBoundsIds.push(a.id);
    }

    if (!hasOverlap) {
      for (let j = i + 1; j < regions.length; j++) {
        const b = regions[j]!;
        const separated =
          a.x0 + a.cols <= b.x0 ||
          b.x0 + b.cols <= a.x0 ||
          a.y0 + a.rows <= b.y0 ||
          b.y0 + b.rows <= a.y0;
        if (!separated) {
          hasOverlap = true;
          overlapPair = Object.freeze([a.id, b.id] as const);
          break;
        }
      }
    }
  }

  return Object.freeze({
    width,
    height,
    totalTexels,
    usedTexels,
    utilizationPct: totalTexels > 0 ? (usedTexels / totalTexels) * 100 : 0,
    byteLength,
    maxBudgetBytes,
    withinBudget,
    regions: Object.freeze([...regions]),
    hasOverlap,
    overlapPair,
    outOfBoundsIds: Object.freeze(outOfBoundsIds),
  });
}

/** Deterministic dual-hash content key for `RG32F` GPU texture upload deduplication. */
export function computeAtlasContentKey(
  viewMode: UniSpatialViewMode,
  atlas: Rg32fValueAtlas,
  extraTag = '',
): string {
  const bytes = new Uint8Array(atlas.data.buffer, atlas.data.byteOffset, atlas.data.byteLength);
  const digest = dualHashHex(bytes);
  return `${viewMode}:${atlas.width}x${atlas.height}:${extraTag}:${digest}`;
}

// ============================================================================
// SHARED 5-PASS GPU BUFFER BUILDERS (MATCHING `src/render/*` EXACT STRIDES)
// ============================================================================

export class UniSolidBufferBuilder {
  private readonly out: number[] = [];
  private count = 0;

  push(params: {
    readonly cx: number;
    readonly cy: number;
    readonly cz: number;
    readonly ex: number;
    readonly ey: number;
    readonly ez: number;
    readonly cellFamily?: number;
    readonly fp8?: number;
    readonly phase?: number;
    readonly bayId?: number;
    readonly cellScale?: number;
    readonly highlight?: number;
  }): void {
    this.out.push(
      params.cx,
      params.cy,
      params.cz,
      params.ex,
      params.ey,
      params.ez,
      params.cellFamily ?? 0,
      params.fp8 ?? 0,
      params.phase ?? 0,
      params.bayId ?? 0,
      params.cellScale ?? 1.0,
      params.highlight ?? 0,
    );
    this.count += 1;
  }

  pack(): { readonly data: Float32Array; readonly count: number } {
    return {
      data: new Float32Array(this.out),
      count: this.count,
    };
  }
}

export class UniLineBufferBuilder {
  private readonly out: number[] = [];

  addSegment(
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.5,
  ): void {
    pushFreeGuideLine(this.out, ax, ay, az, bx, by, bz, cssWidth, style, alpha);
  }

  addBoxFrameXZ(
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.45,
  ): void {
    pushBoxFrameXZ(this.out, x0, z0, x1, z1, y, cssWidth, style, alpha);
  }

  addDashedFrameXZ(
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
    style = LINE_STYLE_GUIDE,
    alpha = 0.42,
    dash = 0.22,
    gap = 0.14,
    cssWidth = 0.9,
  ): void {
    pushDashedFrameXZ(this.out, x0, z0, x1, z1, y, style, alpha, dash, gap, cssWidth);
  }

  /** Push the 12 crisp front-face-culled edges of an axis-aligned solid box. */
  addBoxEdges(
    cx: number,
    cy: number,
    cz: number,
    ex: number,
    ey: number,
    ez: number,
    ownerId: number,
    cssWidth = 1.0,
    style = LINE_STYLE_NEUTRAL,
    alpha = 0.72,
  ): void {
    const hx = ex * 0.5;
    const hy = ey * 0.5;
    const hz = ez * 0.5;
    const x0 = cx - hx;
    const x1 = cx + hx;
    const y0 = cy - hy;
    const y1 = cy + hy;
    const z0 = cz - hz;
    const z1 = cz + hz;

    const edge = (
      ax: number,
      ay: number,
      az: number,
      bx: number,
      by: number,
      bz: number,
      f0: number,
      f1: number,
    ) => {
      this.out.push(
        ax, ay, az,
        bx, by, bz,
        cssWidth,
        style,
        alpha,
        ownerId,
        f0,
        f1,
        cx, cy, cz,
        ex, ey, ez,
      );
    };

    edge(x0, y0, z0, x1, y0, z0, FACE_NEG_Y, FACE_NEG_Z);
    edge(x1, y0, z0, x1, y0, z1, FACE_NEG_Y, FACE_POS_X);
    edge(x1, y0, z1, x0, y0, z1, FACE_NEG_Y, FACE_POS_Z);
    edge(x0, y0, z1, x0, y0, z0, FACE_NEG_Y, FACE_NEG_X);

    edge(x0, y1, z0, x1, y1, z0, FACE_POS_Y, FACE_NEG_Z);
    edge(x1, y1, z0, x1, y1, z1, FACE_POS_Y, FACE_POS_X);
    edge(x1, y1, z1, x0, y1, z1, FACE_POS_Y, FACE_POS_Z);
    edge(x0, y1, z1, x0, y1, z0, FACE_POS_Y, FACE_NEG_X);

    edge(x0, y0, z0, x0, y1, z0, FACE_NEG_X, FACE_NEG_Z);
    edge(x1, y0, z0, x1, y1, z0, FACE_POS_X, FACE_NEG_Z);
    edge(x1, y0, z1, x1, y1, z1, FACE_POS_X, FACE_POS_Z);
    edge(x0, y0, z1, x0, y1, z1, FACE_NEG_X, FACE_POS_Z);
  }

  pack(): { readonly data: Float32Array; readonly count: number } {
    return {
      data: new Float32Array(this.out),
      count: Math.floor(this.out.length / LINE_INSTANCE_FLOATS),
    };
  }
}

export class UniRibbonBufferBuilder {
  private readonly out: number[] = [];
  private count = 0;

  push(params: {
    readonly cx: number;
    readonly cy: number;
    readonly cz: number;
    readonly ex: number;
    readonly ey: number;
    readonly ez: number;
    readonly kind?: number;
    readonly colorId?: number;
    readonly alpha?: number;
  }): void {
    this.out.push(
      params.cx,
      params.cy,
      params.cz,
      params.ex,
      params.ey,
      params.ez,
      params.kind ?? 0,
      params.colorId ?? 0,
      params.alpha ?? 0.36,
      0,
      0,
      0,
    );
    this.count += 1;
  }

  pack(): { readonly data: Float32Array; readonly count: number } {
    return {
      data: new Float32Array(this.out),
      count: this.count,
    };
  }
}

/** Typed wrapper over shared `FlowRibbonBuilder` from `src/render/ribbons.ts`. */
export class UniFlatFlowBufferBuilder {
  private readonly inner = new FlowRibbonBuilder();

  addPolylineXZ(
    rawPoints: readonly XZPoint[],
    y: number,
    width: number,
    color: FlowRibbonColor,
    arrowHead = true,
    arrowLength = 0.24,
    arrowWidth = 0.22,
  ): void {
    this.inner.addPolylineXZ(rawPoints, y, width, color, arrowHead, arrowLength, arrowWidth);
  }

  addCubicXZ(
    p0: XZPoint,
    c1: XZPoint,
    c2: XZPoint,
    p3: XZPoint,
    y: number,
    width: number,
    color: FlowRibbonColor,
    arrowHead = true,
    steps = 24,
  ): void {
    this.inner.addCubicXZ(p0, c1, c2, p3, y, width, color, arrowHead, steps);
  }

  addRoundedRouteXZ(
    waypoints: readonly XZPoint[],
    y: number,
    width: number,
    color: FlowRibbonColor,
    radius = 0.28,
    arrowHead = true,
  ): void {
    this.inner.addRoundedOrthoXZ(waypoints, radius, y, width, color, arrowHead);
  }

  addPolyline3D(
    rawPoints: readonly Vec3[],
    upHint: Vec3,
    width: number,
    color: FlowRibbonColor,
    arrowHead = false,
    arrowLength = 0.22,
    arrowWidth = 0.22,
    crossRibbon = true,
  ): void {
    this.inner.addPolyline3D(
      rawPoints,
      upHint,
      width,
      color,
      arrowHead,
      arrowLength,
      arrowWidth,
      crossRibbon,
    );
  }

  addCubic3D(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    upHint: Vec3,
    width: number,
    color: FlowRibbonColor,
    arrowHead = false,
    segments = 20,
    crossRibbon = true,
    arrowLength = 0.2,
    arrowWidth = 0.2,
  ): void {
    this.inner.addCubic3D(
      p0,
      p1,
      p2,
      p3,
      upHint,
      width,
      color,
      arrowHead,
      segments,
      crossRibbon,
      arrowLength,
      arrowWidth,
    );
  }

  addCellHighlightXZ(
    cx: number,
    cz: number,
    ex: number,
    ez: number,
    y: number,
    fillColor: FlowRibbonColor,
    frameColor: FlowRibbonColor,
    frameThickness = 0.022,
  ): void {
    this.inner.addCellHighlightXZ(cx, cz, ex, ez, y, fillColor, frameColor, frameThickness);
  }

  addCellHighlightOnFrame(
    frame: SheetFrame3D,
    rowStart: number,
    rowEnd: number,
    colStart: number,
    colEnd: number,
    gridRows: number,
    gridCols: number,
    normalOffset: number,
    fillColor: FlowRibbonColor,
    frameColor: FlowRibbonColor,
    frameThickness = 0.022,
    doubleSided = true,
  ): void {
    this.inner.addCellHighlightOnFrame(
      frame,
      rowStart,
      rowEnd,
      colStart,
      colEnd,
      gridRows,
      gridCols,
      normalOffset,
      fillColor,
      frameColor,
      frameThickness,
      doubleSided,
    );
  }

  pack(): { readonly flowVertices: Float32Array; readonly flowIndices: Uint32Array } {
    const built = this.inner.pack();
    return {
      flowVertices: built.vertices,
      flowIndices: built.indices,
    };
  }
}

export interface UniSheetPushParams {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly sheetKind?: number;
  readonly role?: number;
  readonly fp8?: number;
  readonly highlight?: number;
  readonly seed?: number;
  readonly slice?: AtlasSlice | null;
  readonly texRect?: TexRect;
  readonly valueScale?: number;
  readonly flagMode?: 0 | 1;
  readonly frame?: SheetFrame3D;
  readonly revealMode?: number;
  readonly revealColFrontier?: number;
  readonly revealRowFrontier?: number;
  readonly revealPulsePhase?: number;
}

/**
 * Packed `SheetsPass` instance builder with HARD `U25` enforcement:
 * throws immediately if `SHEET_KIND_CAUSAL_SCORES` (`1`) or `SHEET_KIND_CAUSAL_SOFTMAX` (`2`)
 * is pushed without a valid live `RG32F` `TexRect` (`uWidth > 0 && vHeight > 0`).
 */
export class UniSheetBufferBuilder {
  private readonly inst: number[] = [];
  private readonly basis: number[] = [];
  private readonly reveal: number[] = [];
  private count = 0;

  push(params: UniSheetPushParams): void {
    const sheetKind = params.sheetKind ?? SHEET_KIND_STANDARD;
    const rect: TexRect = params.slice
      ? params.slice.rect
      : (params.texRect ?? [0, 0, 0, 0]);
    const valueScale = params.slice ? params.slice.scale : (params.valueScale ?? 1.0);
    const flagMode = params.slice ? params.slice.mode : (params.flagMode ?? 0);

    // U25 Hard Guard: prevent SheetsPass fallback to geometric `cellCol > cellRow` masking!
    if (
      (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) &&
      (rect[2] <= 0 || rect[3] <= 0)
    ) {
      throw new Error(
        `U25 violation: sheetKind=${sheetKind} requires a live RG32F texRect (got uWidth=${rect[2]}, vHeight=${rect[3]})`,
      );
    }

    this.inst.push(
      params.cx,
      params.cy,
      params.cz,
      params.ex,
      params.ey,
      params.ez,
      Math.max(1, params.gridCols),
      Math.max(1, params.gridRows),
      sheetKind,
      params.role ?? SHEET_ROLE_NEUTRAL,
      params.fp8 ?? 0,
      params.highlight ?? SHEET_HL_NORMAL,
      params.seed ?? this.count * 0.37,
      rect[0],
      rect[1],
      rect[2],
      rect[3],
      valueScale,
      flagMode,
      0,
    );

    const u = params.frame ? params.frame.axisU : FRAME_XZ_AXES.axisU;
    const n = params.frame ? params.frame.axisN : FRAME_XZ_AXES.axisN;
    this.basis.push(u[0], u[1], u[2], n[0], n[1], n[2]);

    this.reveal.push(
      params.revealMode ?? SHEET_REVEAL_NONE,
      params.revealColFrontier ?? 0,
      params.revealRowFrontier ?? 0,
      params.revealPulsePhase ?? 0,
    );

    this.count += 1;
  }

  pack(): {
    readonly sheetData: Float32Array;
    readonly sheetBasisData: Float32Array;
    readonly sheetRevealData: Float32Array;
    readonly sheetCount: number;
  } {
    return {
      sheetData: new Float32Array(this.inst),
      sheetBasisData: new Float32Array(this.basis),
      sheetRevealData: new Float32Array(this.reveal),
      sheetCount: this.count,
    };
  }
}

// ============================================================================
// SCENE GEOMETRY FINITENESS, ATLAS BUDGET & U25 VALIDATOR
// ============================================================================

function checkFiniteArray(
  arr: ArrayLike<number> | undefined | null,
  label: string,
  errors: string[],
): void {
  if (!arr) return;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    if (!Number.isFinite(v)) {
      errors.push(`Non-finite value ${String(v)} in ${label}[${i}]`);
      return;
    }
  }
}

function checkFiniteAabb(bounds: AABB, label: string, errors: string[]): void {
  for (let axis = 0; axis < 3; axis++) {
    if (!Number.isFinite(bounds.min[axis]) || !Number.isFinite(bounds.max[axis])) {
      errors.push(`Non-finite AABB axis ${axis} in ${label}`);
      return;
    }
    if (bounds.max[axis] < bounds.min[axis]) {
      errors.push(`Inverted AABB axis ${axis} in ${label}`);
      return;
    }
  }
}

/**
 * Exhaustive scene geometry & contract validator:
 *   - Every emitted float across all 5 passes, atlas texels, pick boxes, camera/bounds, and RGB previews is finite.
 *   - Buffer lengths match exact `SOLID_INSTANCE_FLOATS` / `LINE_INSTANCE_FLOATS` /
 *     `RIBBON_INSTANCE_FLOATS` / `SHEET_*_FLOATS` strides.
 *   - `U25`: zero `sheetKind ∈ {1, 2}` sheets without a live `RG32F` atlas and positive `TexRect`.
 *   - Atlas audit confirms `0` overlaps, `0` out-of-bounds rects, and `byteLength <= maxBudgetBytes`.
 */
export function validateSceneGeometryFinite(scene: UniSpatialScene): UniSceneValidationResult {
  const errors: string[] = [];

  checkFiniteArray(scene.solidData, `${scene.viewMode}.solidData`, errors);
  checkFiniteArray(scene.lineData, `${scene.viewMode}.lineData`, errors);
  checkFiniteArray(scene.ribbonData, `${scene.viewMode}.ribbonData`, errors);
  checkFiniteArray(scene.sheetData, `${scene.viewMode}.sheetData`, errors);
  checkFiniteArray(scene.sheetBasisData, `${scene.viewMode}.sheetBasisData`, errors);
  checkFiniteArray(scene.sheetRevealData, `${scene.viewMode}.sheetRevealData`, errors);
  checkFiniteArray(scene.flowVertices, `${scene.viewMode}.flowVertices`, errors);
  checkFiniteArray(scene.valueAtlas?.data, `${scene.viewMode}.valueAtlas.data`, errors);
  checkFiniteAabb(scene.fitBounds, `${scene.viewMode}.fitBounds`, errors);
  checkFiniteAabb(scene.cameraPreset.fitBounds, `${scene.viewMode}.cameraPreset.fitBounds`, errors);

  if (!Number.isFinite(scene.cameraPreset.azimuthDeg) || !Number.isFinite(scene.cameraPreset.elevationDeg)) {
    errors.push(`Non-finite cameraPreset angles in ${scene.viewMode}`);
  }

  if (scene.solidData.length < scene.solidCount * SOLID_INSTANCE_FLOATS) {
    errors.push(`${scene.viewMode}.solidData shorter than solidCount * ${SOLID_INSTANCE_FLOATS}`);
  }
  if (scene.lineData.length < scene.lineCount * LINE_INSTANCE_FLOATS) {
    errors.push(`${scene.viewMode}.lineData shorter than lineCount * ${LINE_INSTANCE_FLOATS}`);
  }
  if (scene.ribbonData.length < scene.ribbonCount * RIBBON_INSTANCE_FLOATS) {
    errors.push(`${scene.viewMode}.ribbonData shorter than ribbonCount * ${RIBBON_INSTANCE_FLOATS}`);
  }
  if (scene.sheetCount > 0) {
    if (scene.sheetData.length < scene.sheetCount * SHEET_INSTANCE_FLOATS) {
      errors.push(`${scene.viewMode}.sheetData shorter than sheetCount * ${SHEET_INSTANCE_FLOATS}`);
    }
    if (scene.sheetBasisData.length < scene.sheetCount * SHEET_BASIS_FLOATS) {
      errors.push(`${scene.viewMode}.sheetBasisData shorter than sheetCount * ${SHEET_BASIS_FLOATS}`);
    }
    if (scene.sheetRevealData.length < scene.sheetCount * SHEET_REVEAL_FLOATS) {
      errors.push(`${scene.viewMode}.sheetRevealData shorter than sheetCount * ${SHEET_REVEAL_FLOATS}`);
    }

    // U25 audit: every causal score/softmax sheet MUST bind a live RG32F atlas & positive TexRect
    for (let i = 0; i < scene.sheetCount; i++) {
      const base = i * SHEET_INSTANCE_FLOATS;
      const sheetKind = Math.round(scene.sheetData[base + 8]!);
      const uWidth = scene.sheetData[base + 15]!;
      const vHeight = scene.sheetData[base + 16]!;
      if (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) {
        if (!scene.valueAtlas || uWidth <= 0 || vHeight <= 0) {
          errors.push(
            `U25 violation in ${scene.viewMode}.sheet[${i}]: sheetKind=${sheetKind} without live RG32F atlas/texRect`,
          );
        }
      }
    }
  }

  if (scene.atlasAudit) {
    if (scene.atlasAudit.hasOverlap) {
      errors.push(
        `Atlas texel overlap in ${scene.viewMode}: ${scene.atlasAudit.overlapPair?.join(' vs ') ?? 'unknown'}`,
      );
    }
    if (scene.atlasAudit.outOfBoundsIds.length > 0) {
      errors.push(
        `Atlas out-of-bounds regions in ${scene.viewMode}: ${scene.atlasAudit.outOfBoundsIds.join(', ')}`,
      );
    }
    if (!scene.atlasAudit.withinBudget) {
      errors.push(
        `Atlas byteLength (${scene.atlasAudit.byteLength}) exceeds maxBudgetBytes (${scene.atlasAudit.maxBudgetBytes}) in ${scene.viewMode}`,
      );
    }
  }

  for (let i = 0; i < scene.pickBoxes.length; i++) {
    const pb = scene.pickBoxes[i]!;
    checkFiniteAabb({ min: pb.min, max: pb.max }, `${scene.viewMode}.pickBoxes[${i}]`, errors);
  }

  for (let i = 0; i < scene.rgbPreviews.length; i++) {
    const p = scene.rgbPreviews[i]!;
    if (p.rgbChw.length !== 3 * p.width * p.height) {
      errors.push(
        `RGB preview "${p.id}" length (${p.rgbChw.length}) !== 3 * ${p.width} * ${p.height}`,
      );
    }
    checkFiniteArray(p.rgbChw, `${scene.viewMode}.rgbPreviews[${i}].rgbChw`, errors);
  }

  return Object.freeze({
    valid: errors.length === 0,
    errors: Object.freeze(errors),
  });
}
