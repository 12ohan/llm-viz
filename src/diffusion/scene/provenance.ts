/**
 * Diffusion Spatial Provenance, 8-KV-Head Pick Adapter, Atlas Audit & Buffer Builders
 * (`src/diffusion/scene/provenance.ts`)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, or `src/twotower/` imports.
 * Shared Step-0 foundation consumed across all 7 DF3 spatial views (`3A`, `3B`, `3C`, `3D`):
 *   - Session context resolution with 0-ULP lazy layer & cropped-suffix KV access
 *   - Epistemic & provenance badges (incl. projection-width reversal & `D18`/`D19`/`kvCapacity`)
 *   - 8-KV-head pick adapter (`pickHitToDiffusionCell`) reading `PickBox.groupIndex` (0..7)
 *     without modifying `src/render/pick.ts`
 *   - Exact atlas non-overlap auditor (`auditAtlasRegions`)
 *   - Shared GPU buffer builders matching `SolidsPass`, `LinesPass`, `RibbonsPass`, and `SheetsPass`
 *     with built-in `D24` enforcement (never allows causal/score/softmax sheets without live `RG32F` texRect)
 *   - Finite geometry & `D24` scene validator (`validateSceneGeometryFinite`)
 */

import type { AABB } from '../../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  pushBoxFrameXZ,
  pushDashedFrameXZ,
  pushFreeGuideLine,
} from '../../render/linePacking';
import type { PickHit } from '../../render/pick';
import { FRAME_XZ_AXES, type SheetFrame3D } from '../../render/sheetFrame';
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
import type { AtlasSlice, TexRect } from '../../scene/atlasPacker';
import {
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  RIBBON_INSTANCE_FLOATS,
  SOLID_INSTANCE_FLOATS,
} from '../../render/primitives';
import { verifyKvSnapshotIntegrity } from '../kvCache';
import type { DiffusionSession } from '../session';
import {
  DIFFUSION_MINI_CONFIG,
  type DiffusionLayerTrace,
  type DiffusionSessionRecord,
  type KvPositionSlab,
} from '../types';
import type {
  DiffusionAtlasLayoutAudit,
  DiffusionAtlasRegionBox,
  DiffusionPickIdentity,
  DiffusionProvenanceBadge,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
  DiffusionSpatialViewMode,
} from './types';

// ============================================================================
// AUTHORITATIVE EPISTEMIC & DISCLOSURE STRINGS
// ============================================================================

export const DIFFUSION_PROJECTION_REVERSAL_BADGE =
  'Mini expands 8→256/64 · Prod 14B contracts 5120→4096/1024' as const;

export const DIFFUSION_LORA_RAIL_BADGE =
  'o_proj LoRA (+0.35% prod · r=4, α=16, scale=4.0 mini)' as const;

export const DIFFUSION_EPISTEMIC_NOTE =
  'Educational 40L miniature simulator (d=8, 32Q/8KV, dHead=8, dyadic YaRN) — NOT 14B checkpoint weights.' as const;

// ============================================================================
// SESSION RESOLUTION & PROVENANCE BADGES
// ============================================================================

/** Ensures the session has at least its initial prefill record so spatial views always read real tensors. */
export function ensureSessionHasActiveRecord(
  session: DiffusionSession,
): DiffusionSessionRecord {
  const active = session.getActiveRecord();
  if (active !== null) return active;
  return session.step();
}

export function buildDiffusionProvenanceBadges(
  record: DiffusionSessionRecord,
  kvIntegrityValid: boolean,
  kvIntegrityError: string | null,
): readonly DiffusionProvenanceBadge[] {
  const badges: DiffusionProvenanceBadge[] = [];
  const cfg = record.configAtStep;

  badges.push({
    label:
      cfg.mode === 'ar'
        ? 'Mode: AR'
        : cfg.mode === 'dlm'
          ? cfg.causalContext
            ? 'Mode: DLM (Causal Context)'
            : 'Mode: DLM (Bidir Context)'
          : cfg.mode === 'linear_spec'
            ? 'Mode: Linear Spec'
            : 'Mode: Linear Spec + LoRA',
    tone: cfg.mode === 'linear_spec_lora' ? 'lora' : cfg.causalContext ? 'causal' : 'bidir',
  });

  badges.push({
    label:
      record.kvMaskProvenance === 'token_causal'
        ? 'KV: token_causal'
        : 'KV: block_bidirectional',
    tone: record.kvMaskProvenance === 'token_causal' ? 'causal' : 'bidir',
  });

  const loraDraftActive =
    cfg.mode === 'linear_spec_lora' && record.kind === 'spec_cycle';
  badges.push({
    label: loraDraftActive
      ? 'o_proj LoRA: ON (draft only; verify OFF)'
      : 'o_proj LoRA: OFF',
    tone: loraDraftActive ? 'lora' : 'muted',
  });

  if (record.guardFlags.draftGuardTriggered !== 'none') {
    badges.push({
      label: `D18 Guard: ${record.guardFlags.draftGuardTriggered}`,
      tone: 'warn',
    });
  }
  if (record.guardFlags.residualMasksCommittedCount > 0) {
    badges.push({
      label: `Residual [MASK] committed (${record.guardFlags.residualMasksCommittedCount})`,
      tone: 'warn',
    });
  }
  if (record.guardFlags.trailingMasksAfterEosCount > 0) {
    badges.push({
      label: `Trailing [MASK] after EOS (${record.guardFlags.trailingMasksAfterEosCount})`,
      tone: 'warn',
    });
  }
  if (record.lifecyclePhase === 'capacity_exhausted') {
    badges.push({
      label: `KV Capacity Exhausted (${record.kvSnapshot.seqLen}/${record.kvSnapshot.capacity})`,
      tone: 'warn',
    });
  }
  if (record.cropProvenance !== null && record.cropProvenance.firstEosOffset !== null) {
    badges.push({
      label: `D19 Crop: browser=${record.cropProvenance.browserPostEosCropLen} · upstream=${record.cropProvenance.upstreamPreEosCropLen}`,
      tone: 'gold',
    });
  }
  if (!kvIntegrityValid) {
    badges.push({
      label: `KV Integrity Error: ${kvIntegrityError ?? 'digest mismatch'}`,
      tone: 'warn',
    });
  }

  return Object.freeze(badges);
}

/**
 * Resolve a read-only spatial view context from a `DiffusionSession` (active cursor or explicit `recordId`),
 * never mutating session KV snapshots and providing 0-ULP lazy trace & cropped-suffix KV access.
 */
export function resolveDiffusionSessionContext(
  session: DiffusionSession,
  recordId?: string,
): DiffusionResolvedSessionContext {
  const record =
    recordId !== undefined
      ? session.getRecord(recordId)
      : ensureSessionHasActiveRecord(session);
  if (record === null) {
    throw new RangeError(`Unknown DiffusionSession recordId: "${String(recordId)}"`);
  }

  const state = session.getState();
  const branchRecords = session.getBranchRecords(record.branchId);
  const recordIdxInBranch = branchRecords.findIndex((r) => r.recordId === record.recordId);
  const isHistoricalCursor =
    branchRecords.length > 0 &&
    recordIdxInBranch >= 0 &&
    recordIdxInBranch < branchRecords.length - 1;
  const parentRecord =
    record.parentRecordId !== null ? session.getRecord(record.parentRecordId) : null;

  const integrity = verifyKvSnapshotIntegrity(record.kvSnapshot);
  const provenanceBadges = buildDiffusionProvenanceBadges(
    record,
    integrity.valid,
    integrity.reason,
  );

  const warningParts: string[] = [];
  if (!integrity.valid) {
    warningParts.push(`KV integrity check failed: ${integrity.reason ?? 'unknown'}`);
  }
  if (record.guardFlags.warnings.length > 0) {
    warningParts.push(...record.guardFlags.warnings);
  }

  const resolvedLayerCache = new Map<number, DiffusionLayerTrace>();
  let resolvedSuffixCache: readonly KvPositionSlab[] | null = null;

  return {
    model: session.model,
    session,
    record,
    parentRecord,
    branchId: record.branchId,
    recordIndex: record.recordIndex,
    branchRecordCount: branchRecords.length,
    isHistoricalCursor,
    modeAtRecord: record.configAtStep.mode,
    causalContextAtRecord: record.configAtStep.causalContext,
    loraActiveOnDraft:
      record.configAtStep.mode === 'linear_spec_lora' && record.kind === 'spec_cycle',
    kvSnapshot: record.kvSnapshot,
    kvIntegrityValid: integrity.valid,
    kvIntegrityError: integrity.reason,
    resolveLayerTrace(layerIndex: number): DiffusionLayerTrace {
      const clamped = Math.max(
        0,
        Math.min(session.model.config.numLayers - 1, Math.trunc(layerIndex)),
      );
      const cached = resolvedLayerCache.get(clamped);
      if (cached) return cached;
      const trace = session.reconstructLayerTrace(record.recordId, clamped);
      resolvedLayerCache.set(clamped, trace);
      return trace;
    },
    resolveSpecCroppedSuffixSlabs(): readonly KvPositionSlab[] {
      if (record.kind !== 'spec_cycle') return Object.freeze([]);
      if (resolvedSuffixCache !== null) return resolvedSuffixCache;
      resolvedSuffixCache = session.reconstructSpecCroppedSuffixKvs(record.recordId);
      return resolvedSuffixCache;
    },
    provenanceBadges,
    warningBanner:
      warningParts.length > 0
        ? warningParts.join(' · ')
        : state.activeRecordId !== record.recordId
          ? `Inspecting historical record #${record.recordIndex} (${record.recordId})`
          : null,
  };
}

// ============================================================================
// 8-KV-HEAD PICK ADAPTER (ZERO EDITS TO `src/render/pick.ts`)
// ============================================================================

/**
 * Decode a `PickHit` from `src/render/pick.ts` into a full `DiffusionPickIdentity`.
 * Diffusion's 8 KV heads (`0..7`) are stored in `PickBox.groupIndex` (since Nano's
 * `PickBox.kvHeadIndex` is narrowed to `0 | 1`).
 */
export function pickHitToDiffusionCell(
  hit: PickHit | null,
  viewMode: DiffusionSpatialViewMode,
): DiffusionPickIdentity | null {
  if (!hit) return null;

  const layerIndex =
    hit.bayId >= 1 && hit.bayId <= DIFFUSION_MINI_CONFIG.numLayers
      ? hit.bayId - 1
      : null;

  let qHeadIndex: number | null =
    hit.headIndex !== undefined &&
    hit.headIndex >= 0 &&
    hit.headIndex < DIFFUSION_MINI_CONFIG.numAttentionHeads
      ? hit.headIndex
      : null;

  let kvHeadIndex: number | null =
    hit.groupIndex !== undefined &&
    hit.groupIndex >= 0 &&
    hit.groupIndex < DIFFUSION_MINI_CONFIG.numKeyValueHeads
      ? hit.groupIndex
      : null;

  if (kvHeadIndex === null && qHeadIndex !== null) {
    kvHeadIndex = Math.floor(qHeadIndex / DIFFUSION_MINI_CONFIG.kvGroupSize);
  }
  if (
    qHeadIndex === null &&
    hit.tensorKey !== undefined &&
    (hit.tensorKey === 'q_pre' ||
      hit.tensorKey === 'q_rot' ||
      hit.tensorKey === 'q_scaled' ||
      hit.tensorKey === 'attn_concat' ||
      hit.tensorKey === 'w_q' ||
      hit.tensorKey === 'w_o') &&
    hit.hitCol !== undefined
  ) {
    const inferredQ = Math.floor(hit.hitCol / DIFFUSION_MINI_CONFIG.headDim);
    if (inferredQ >= 0 && inferredQ < DIFFUSION_MINI_CONFIG.numAttentionHeads) {
      qHeadIndex = inferredQ;
      if (kvHeadIndex === null) {
        kvHeadIndex = Math.floor(inferredQ / DIFFUSION_MINI_CONFIG.kvGroupSize);
      }
    }
  }
  if (
    kvHeadIndex === null &&
    hit.tensorKey !== undefined &&
    (hit.tensorKey === 'k_pre' ||
      hit.tensorKey === 'k_rot' ||
      hit.tensorKey === 'k_full' ||
      hit.tensorKey === 'v_full' ||
      hit.tensorKey === 'w_k' ||
      hit.tensorKey === 'w_v') &&
    hit.hitCol !== undefined
  ) {
    const inferredKv = Math.floor(hit.hitCol / DIFFUSION_MINI_CONFIG.headDim);
    if (inferredKv >= 0 && inferredKv < DIFFUSION_MINI_CONFIG.numKeyValueHeads) {
      kvHeadIndex = inferredKv;
    }
  }

  let rotaryPlane: number | null =
    hit.tapIndex !== undefined && hit.tapIndex >= 0 && hit.tapIndex < 4
      ? hit.tapIndex
      : null;
  if (
    rotaryPlane === null &&
    hit.hitCol !== undefined &&
    hit.tensorKey !== undefined &&
    (hit.tensorKey === 'q_pre' ||
      hit.tensorKey === 'q_rot' ||
      hit.tensorKey === 'q_scaled' ||
      hit.tensorKey === 'k_pre' ||
      hit.tensorKey === 'k_rot' ||
      hit.tensorKey === 'rope_cos' ||
      hit.tensorKey === 'rope_sin')
  ) {
    rotaryPlane = hit.hitCol % (DIFFUSION_MINI_CONFIG.headDim / 2);
  }

  const tokenIndex =
    hit.tokenIndex !== undefined
      ? hit.tokenIndex
      : hit.hitRow !== undefined
        ? hit.hitRow
        : null;

  const absPosition =
    hit.absoluteTokenOffset !== undefined
      ? hit.absoluteTokenOffset + (tokenIndex ?? 0)
      : tokenIndex;

  return {
    viewMode,
    layerIndex,
    partId: hit.partId ?? null,
    tensorKey: hit.tensorKey ?? null,
    qHeadIndex,
    kvHeadIndex,
    rotaryPlane,
    tokenIndex,
    absPosition,
    row: hit.hitRow ?? null,
    col: hit.hitCol ?? null,
    semanticTag: hit.semanticTag ?? null,
  };
}

// ============================================================================
// ATLAS NON-OVERLAP & UTILIZATION AUDITOR
// ============================================================================

export function auditAtlasRegions(
  width: number,
  height: number,
  regions: readonly DiffusionAtlasRegionBox[],
): DiffusionAtlasLayoutAudit {
  const outOfBoundsIds: string[] = [];
  let usedTexels = 0;

  for (const r of regions) {
    if (
      r.cols <= 0 ||
      r.rows <= 0 ||
      r.x0 < 0 ||
      r.y0 < 0 ||
      r.x0 + r.cols > width ||
      r.y0 + r.rows > height
    ) {
      outOfBoundsIds.push(r.id);
    }
    usedTexels += Math.max(0, r.cols) * Math.max(0, r.rows);
  }

  let hasOverlap = false;
  let overlapPair: readonly [string, string] | null = null;
  for (let i = 0; i < regions.length && !hasOverlap; i++) {
    const a = regions[i]!;
    const ax1 = a.x0 + a.cols;
    const ay1 = a.y0 + a.rows;
    for (let j = i + 1; j < regions.length; j++) {
      const b = regions[j]!;
      const bx1 = b.x0 + b.cols;
      const by1 = b.y0 + b.rows;
      if (a.x0 < bx1 && ax1 > b.x0 && a.y0 < by1 && ay1 > b.y0) {
        hasOverlap = true;
        overlapPair = [a.id, b.id];
        break;
      }
    }
  }

  const totalTexels = width * height;
  return {
    width,
    height,
    totalTexels,
    usedTexels,
    utilizationPct: totalTexels > 0 ? (usedTexels / totalTexels) * 100 : 0,
    byteLength: totalTexels * 8, // RG32F = 8 bytes/texel
    regions: Object.freeze(regions.slice()),
    hasOverlap,
    overlapPair,
    outOfBoundsIds: Object.freeze(outOfBoundsIds),
  };
}

// ============================================================================
// SHARED WEBGL2 PASS BUFFER BUILDERS (MATCHING `src/render/*` LAYOUTS EXACTLY)
// ============================================================================

export class SolidBufferBuilder {
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
      params.cellScale ?? 0.24,
      params.highlight ?? 0,
    );
    this.count++;
  }

  pack(): { readonly data: Float32Array; readonly count: number } {
    return {
      data: new Float32Array(this.out),
      count: this.count,
    };
  }
}

export class LineBufferBuilder {
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

    // Bottom 4 edges (FACE_NEG_Y)
    edge(x0, y0, z0, x1, y0, z0, FACE_NEG_Y, FACE_NEG_Z);
    edge(x1, y0, z0, x1, y0, z1, FACE_NEG_Y, FACE_POS_X);
    edge(x1, y0, z1, x0, y0, z1, FACE_NEG_Y, FACE_POS_Z);
    edge(x0, y0, z1, x0, y0, z0, FACE_NEG_Y, FACE_NEG_X);
    // Top 4 edges (FACE_POS_Y)
    edge(x0, y1, z0, x1, y1, z0, FACE_POS_Y, FACE_NEG_Z);
    edge(x1, y1, z0, x1, y1, z1, FACE_POS_Y, FACE_POS_X);
    edge(x1, y1, z1, x0, y1, z1, FACE_POS_Y, FACE_POS_Z);
    edge(x0, y1, z1, x0, y1, z0, FACE_POS_Y, FACE_NEG_X);
    // Vertical 4 edges
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

export class RibbonBufferBuilder {
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
    this.count++;
  }

  pack(): { readonly data: Float32Array; readonly count: number } {
    return {
      data: new Float32Array(this.out),
      count: this.count,
    };
  }
}

export interface SheetPushParams {
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
 * Packed `SheetsPass` instance builder with mandatory `D24` enforcement:
 * rejects any `SHEET_KIND_CAUSAL_SCORES` or `SHEET_KIND_CAUSAL_SOFTMAX` sheet
 * lacking a valid live `RG32F` `texRect` (`uWidth > 0 && vHeight > 0`).
 */
export class SheetBufferBuilder {
  private readonly inst: number[] = [];
  private readonly basis: number[] = [];
  private readonly reveal: number[] = [];
  private count = 0;

  push(params: SheetPushParams): void {
    const sheetKind = params.sheetKind ?? SHEET_KIND_STANDARD;
    const rect: TexRect = params.slice
      ? params.slice.rect
      : params.texRect ?? [0, 0, 0, 0];
    const valueScale = params.slice ? params.slice.scale : (params.valueScale ?? 1.0);
    const flagMode = params.slice ? params.slice.mode : (params.flagMode ?? 0);

    // D24 Guard: prevent SheetsPass geometric cellCol > cellRow fallback on causal/score/softmax sheets
    if (
      (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) &&
      (rect[2] <= 0 || rect[3] <= 0)
    ) {
      throw new Error(
        `D24 violation: sheetKind=${sheetKind} requires a live RG32F texRect (got uWidth=${rect[2]}, vHeight=${rect[3]})`,
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

    this.count++;
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
// SCENE GEOMETRY FINITENESS & D24 VALIDATOR
// ============================================================================

function checkFiniteArray(arr: ArrayLike<number> | undefined, label: string, errors: string[]): void {
  if (!arr) return;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
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

export function validateSceneGeometryFinite(scene: DiffusionSpatialScene): {
  readonly valid: boolean;
  readonly errors: readonly string[];
} {
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

  if (scene.solidData.length < scene.solidCount * SOLID_INSTANCE_FLOATS) {
    errors.push(`${scene.viewMode}.solidData shorter than solidCount * ${SOLID_INSTANCE_FLOATS}`);
  }
  if (scene.lineData.length < scene.lineCount * LINE_INSTANCE_FLOATS) {
    errors.push(`${scene.viewMode}.lineData shorter than lineCount * ${LINE_INSTANCE_FLOATS}`);
  }
  if (scene.ribbonData.length < scene.ribbonCount * RIBBON_INSTANCE_FLOATS) {
    errors.push(`${scene.viewMode}.ribbonData shorter than ribbonCount * ${RIBBON_INSTANCE_FLOATS}`);
  }

  const sheetCount = scene.sheetCount ?? 0;
  if (sheetCount > 0 && scene.sheetData) {
    if (scene.sheetData.length < sheetCount * SHEET_INSTANCE_FLOATS) {
      errors.push(`${scene.viewMode}.sheetData shorter than sheetCount * ${SHEET_INSTANCE_FLOATS}`);
    }
    if (scene.sheetBasisData && scene.sheetBasisData.length < sheetCount * SHEET_BASIS_FLOATS) {
      errors.push(`${scene.viewMode}.sheetBasisData shorter than sheetCount * ${SHEET_BASIS_FLOATS}`);
    }
    if (scene.sheetRevealData && scene.sheetRevealData.length < sheetCount * SHEET_REVEAL_FLOATS) {
      errors.push(`${scene.viewMode}.sheetRevealData shorter than sheetCount * ${SHEET_REVEAL_FLOATS}`);
    }

    // D24 audit: every causal score/softmax sheet MUST bind a live RG32F texture rect
    for (let i = 0; i < sheetCount; i++) {
      const base = i * SHEET_INSTANCE_FLOATS;
      const sheetKind = Math.round(scene.sheetData[base + 8]!);
      const uWidth = scene.sheetData[base + 15]!;
      const vHeight = scene.sheetData[base + 16]!;
      if (
        sheetKind === SHEET_KIND_CAUSAL_SCORES ||
        sheetKind === SHEET_KIND_CAUSAL_SOFTMAX
      ) {
        if (!scene.valueAtlas || uWidth <= 0 || vHeight <= 0) {
          errors.push(
            `D24 violation in ${scene.viewMode}.sheet[${i}]: sheetKind=${sheetKind} without live RG32F atlas/texRect`,
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
  }

  for (let i = 0; i < scene.solidPickBoxes.length; i++) {
    const pb = scene.solidPickBoxes[i]!;
    checkFiniteAabb({ min: pb.min, max: pb.max }, `${scene.viewMode}.solidPickBoxes[${i}]`, errors);
  }

  return {
    valid: errors.length === 0,
    errors: Object.freeze(errors),
  };
}
