/**
 * Dory1.0-4B Spatial Provenance, DR20 Causal-Mask Guard, DR21 Pick Adapter,
 * Atlas Packer/Auditor & 5-Pass WebGL2 Buffer Builders (`src/dory/scene/provenance.ts`)
 *
 * Shared Step-0 foundation consumed across all 7 D7 spatial views (`DORY3`, `DORY4`, `DORY5`):
 *   - Non-mutating `resolveDorySessionContext` backed by 0-ULP `reconstructStepTrace`
 *     and read-only `evaluateCounterfactualIntermediateCoda` (`writeKv: false`).
 *   - `DR20` Hard Enforcement (`DoryDr20CausalMaskError`): throws in BOTH
 *     `DorySheetBufferBuilder.push()` AND `validateSceneGeometryFinite()` if any causal
 *     score/softmax sheet (`sheetKind in {1, 2}`) lacks a live `RG32F` `G`-channel mask `TexRect`,
 *     preventing `src/render/sheets.ts:347-368` geometric `cellCol > cellRow` fallback!
 *   - `DR21` Pick Adapter (`encodeDoryPickMetadata` & `pickHitToDoryCell`): encodes and decodes
 *     physical/paired/executed layer, loop iteration, KV head, KV bank slot/kind, rotary plane,
 *     speculative offset, and stage metadata via `groupIndex`, `tapIndex`, `stateTokenIndex`,
 *     and `semanticTag` with ZERO edits to `src/render/pick.ts`!
 *   - `DoryAtlasPacker`, `auditAtlasRegions` (0 overlaps, in-bounds, `<= 512 KiB`), and
 *     deterministic `computeDoryAtlasContentKey` for GPU upload deduplication.
 *   - Exact 5-pass GPU buffer builders (`DorySolidBufferBuilder`, `DoryLineBufferBuilder`,
 *     `DoryRibbonBufferBuilder`, `DoryFlatFlowBufferBuilder`, `DorySheetBufferBuilder`)
 *     and exhaustive `validateSceneGeometryFinite`.
 *
 * `D4` Architectural Boundary Isolation:
 *   - Imports leaf primitives strictly from `src/render/primitives.ts` & `src/scene/atlasPacker.ts`
 *     (plus shared `src/render/*` pass contracts).
 *   - ZERO imports from Gate 4b/4c Nano modules (`src/ui/*`, `src/scene/spine.ts`,
 *     `src/scene/sheetInspector.ts`, `src/model/*`, `src/math/*`, `src/story/*`, `src/main.ts`)
 *     or sibling apps (`twotower`, `diffusion`, `uni`).
 *   - 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins, or ambient RNG.
 */

import type { AABB } from '../../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  pushBoxFrameXZ,
  pushDashedFrameXZ,
  pushFreeGuideLine,
} from '../../render/linePacking';
import type { PickBox, PickHit } from '../../render/pick';
import {
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  RIBBON_INSTANCE_FLOATS,
  SOLID_INSTANCE_FLOATS,
  type Vec3,
} from '../../render/primitives';
import {
  FLOW_RIBBON_VERTEX_FLOATS,
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
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
  type Rg32fValueAtlas,
  type TexRect,
} from '../../scene/atlasPacker';
import {
  computeExecutedDepth,
  computeExecutedLayerIdx,
  computeKvBankCounts,
  getDoryStage,
  getDorySublayerKind,
} from '../spec';
import {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_MAX_ATLAS_BYTES,
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  DORY_SPEC_CONTRACT_A_BADGE,
  type DoryCacheRole,
  type DoryCounterfactualCodaResult,
  type DoryExecutedSublayerRecord,
  type DoryForwardTrace,
  type DoryKvBankKind,
  type DoryKvCacheMode,
  type DoryLayerStage,
  type DoryReconstructedStepResult,
  type DorySession,
} from '../types';
import {
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_SCALING_DISCLOSURE_BADGE,
  DORY_TBPTL_READONLY_BADGE,
  type DoryAtlasLayoutAudit,
  type DoryAtlasRegionBox,
  type DoryPickBoxMeta,
  type DoryPickEncodeInput,
  type DoryPickGroupRole,
  type DoryPickIdentity,
  type DoryPickTapRole,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneValidationOptions,
  type DorySceneValidationResult,
  type DorySpatialScene,
  type DorySpatialViewMode,
} from './types';

// Re-export authoritative editorial plaques for convenient consumption by scene builders
export {
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_NLERP_MARKER,
  DORY_SCALING_DISCLOSURE_BADGE,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_TBPTL_READONLY_BADGE,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
};

// ============================================================================
// DR20 CAUSAL-MASK GUARD ERROR
// ============================================================================

/**
 * Hard `DR20` invariant error thrown when a causal attention score or softmax sheet
 * (`sheetKind === 1` or `sheetKind === 2`) is emitted without a valid live `RG32F` `G`-channel
 * mask `TexRect` (`src/render/sheets.ts:347-368` fallback guard).
 */
export class DoryDr20CausalMaskError extends Error {
  readonly code = 'DR20_MISSING_LIVE_RG32F_MASK' as const;
  readonly sheetKind: number;

  constructor(sheetKind: number, message: string) {
    super(message);
    this.name = 'DoryDr20CausalMaskError';
    this.sheetKind = sheetKind;
  }
}

// ============================================================================
// EDITORIAL #F7F8FA DRAFTING-TABLE PALETTE TOKENS (D7)
// ============================================================================

export const DORY_FLOW_COLORS = Object.freeze({
  preludeIndigo: Object.freeze<FlowRibbonColor>({
    r: 0.22,
    g: 0.38,
    b: 0.74,
    a: 0.74,
    borderAlpha: 0.88,
  }),
  recurrentTeal: Object.freeze<FlowRibbonColor>({
    r: 0.102,
    g: 0.541,
    b: 0.478,
    a: 0.76,
    borderAlpha: 0.88,
  }),
  codaAmber: Object.freeze<FlowRibbonColor>({
    r: 0.82,
    g: 0.53,
    b: 0.12,
    a: 0.76,
    borderAlpha: 0.9,
  }),
  anchorGold: Object.freeze<FlowRibbonColor>({
    r: 0.765,
    g: 0.545,
    b: 0.145,
    a: 0.82,
    borderAlpha: 0.94,
  }),
  swaCyan: Object.freeze<FlowRibbonColor>({
    r: 0.16,
    g: 0.6,
    b: 0.82,
    a: 0.7,
    borderAlpha: 0.84,
  }),
  fullIndigo: Object.freeze<FlowRibbonColor>({
    r: 0.28,
    g: 0.34,
    b: 0.76,
    a: 0.74,
    borderAlpha: 0.88,
  }),
  nlerpEmerald: Object.freeze<FlowRibbonColor>({
    r: 0.12,
    g: 0.58,
    b: 0.38,
    a: 0.76,
    borderAlpha: 0.88,
  }),
  specOxide: Object.freeze<FlowRibbonColor>({
    r: 0.863,
    g: 0.298,
    b: 0.204,
    a: 0.82,
    borderAlpha: 0.94,
  }),
  extrapolationCoral: Object.freeze<FlowRibbonColor>({
    r: 0.88,
    g: 0.26,
    b: 0.28,
    a: 0.86,
    borderAlpha: 0.96,
  }),
  ghostSlate: Object.freeze<FlowRibbonColor>({
    r: 0.56,
    g: 0.6,
    b: 0.67,
    a: 0.28,
    borderAlpha: 0.36,
  }),
  activeOxide: Object.freeze<FlowRibbonColor>({
    r: 0.82,
    g: 0.24,
    b: 0.2,
    a: 0.86,
    borderAlpha: 0.95,
  }),
});

// ============================================================================
// NON-MUTATING SESSION CONTEXT RESOLUTION (0-ULP LAZY RECONSTRUCTION)
// ============================================================================

/**
 * Resolves a strictly read-only spatial context from `DorySession` (active cursor or explicit
 * `recordId`), lazily caching 0-ULP `reconstructStepTrace` and read-only
 * `evaluateCounterfactualIntermediateCoda` (`writeKv: false`) for the duration of a scene build.
 */
export function resolveDorySessionContext(
  session: DorySession,
  recordId?: string,
): DoryResolvedSessionContext {
  const activeRecord = session.getActiveRecord();
  const targetRecordId = recordId ?? activeRecord?.recordId ?? null;
  if (targetRecordId === null) {
    throw new RangeError(
      'DorySession has no active branch or record; call startBranch() before building a spatial scene.',
    );
  }
  const record = session.getRecord(targetRecordId);
  if (record === null) {
    throw new RangeError(`Unknown DorySession recordId='${targetRecordId}'.`);
  }
  const branch = session.getBranch(record.branchId);
  if (branch === null) {
    throw new RangeError(
      `Missing DorySession branch '${record.branchId}' for record '${targetRecordId}'.`,
    );
  }

  const recordIndexInBranch = branch.recordIds.indexOf(record.recordId);
  const branchRecordCount = branch.recordIds.length;
  const isHistoricalCursor =
    recordIndexInBranch >= 0 && recordIndexInBranch < branchRecordCount - 1;

  const kLoops = record.kLoops;
  const kvCacheMode = record.kvCacheMode;
  const executionMode = record.executionMode;
  const executedDepth = computeExecutedDepth(kLoops, session.params.config);
  const bankSummary = computeKvBankCounts(
    kLoops,
    kvCacheMode,
    executionMode === 'self_spec',
  );

  const badges: DoryProvenanceBadge[] = [
    {
      label: `Depth k=${kLoops} · ${executedDepth}L exec (72L phys / ${executedDepth >> 1} paired)`,
      tone: kLoops > 4 ? 'warn' : 'recurrent_teal',
      discrepancyId: 'DR4',
    },
    {
      label:
        kvCacheMode === 'per_loop'
          ? `KV: per_loop (${bankSummary.totalBanks} banks)`
          : `KV: last_loop (${bankSummary.totalBanks} banks · partition-dependent)`,
      tone: kvCacheMode === 'per_loop' ? 'swa_cyan' : 'gold',
      discrepancyId: kvCacheMode === 'last_loop' ? 'DR17' : undefined,
    },
    {
      label: DORY_SCALING_DISCLOSURE_BADGE,
      tone: 'muted',
    },
  ];

  if (executionMode === 'self_spec') {
    badges.push({
      label: DORY_SPEC_CONTRACT_A_BADGE,
      tone: 'spec_oxide',
      discrepancyId: 'DR23',
    });
  }
  if (kLoops > 4) {
    badges.push({
      label: `Base Extrapolation k=${kLoops} (Trained k in {1..4})`,
      tone: 'warn',
      discrepancyId: 'DR11',
    });
  }
  if (isHistoricalCursor) {
    badges.push({
      label: `Historical Cursor (${record.recordId})`,
      tone: 'gold',
    });
  }

  let stepTraceCache: DoryReconstructedStepResult | null = null;
  let forwardTraceCache: DoryForwardTrace | null = null;
  const counterfactualCache = new Map<number, DoryCounterfactualCodaResult>();

  const resolveStepTrace = (): DoryReconstructedStepResult => {
    if (stepTraceCache === null) {
      stepTraceCache = session.reconstructStepTrace(record.recordId);
    }
    return stepTraceCache;
  };

  const resolveForwardTrace = (): DoryForwardTrace => {
    if (forwardTraceCache !== null) {
      return forwardTraceCache;
    }
    const reconstructed = resolveStepTrace();
    const trace =
      reconstructed.forwardResult?.trace ??
      reconstructed.specPrefillResult?.deepTrace ??
      reconstructed.specCycleResult?.verifyTrace ??
      null;
    if (trace === null) {
      throw new Error(
        `Reconstructed step '${record.recordId}' (kind='${record.kind}') produced no DoryForwardTrace.`,
      );
    }
    forwardTraceCache = trace;
    return trace;
  };

  const resolveDefaultTokenOffset = (): number => {
    const trace = resolveForwardTrace();
    const committedLastPos = Math.max(0, record.committedLen - 1);
    const rawOffset = committedLastPos - trace.startPos;
    return Math.max(0, Math.min(trace.seqLen - 1, rawOffset));
  };

  const resolveSublayerRecord = (
    physicalLayerIdx: number,
    loopIteration?: number,
  ): DoryExecutedSublayerRecord => {
    const trace = resolveForwardTrace();
    const clampedPhys = Math.max(
      0,
      Math.min(session.params.config.numHiddenLayers - 1, Math.trunc(physicalLayerIdx)),
    );
    const stage = getDoryStage(clampedPhys);
    const resolvedLoop =
      stage === 'recurrent'
        ? Math.max(1, Math.min(kLoops, Math.trunc(loopIteration ?? 1)))
        : 0;
    const execIdx = computeExecutedLayerIdx(clampedPhys, resolvedLoop, kLoops);
    const direct = trace.executedSublayers[execIdx];
    if (
      direct !== undefined &&
      direct.coord.physicalLayerIdx === clampedPhys &&
      direct.coord.loopIteration === resolvedLoop
    ) {
      return direct;
    }
    const found = trace.executedSublayers.find(
      (r) => r.coord.physicalLayerIdx === clampedPhys && r.coord.loopIteration === resolvedLoop,
    );
    if (found === undefined) {
      throw new RangeError(
        `Missing executed sublayer record for physicalLayerIdx=${clampedPhys}, loopIteration=${resolvedLoop} (kLoops=${kLoops}).`,
      );
    }
    return found;
  };

  const resolveCounterfactualCoda = (loopIteration: number): DoryCounterfactualCodaResult => {
    const clampedLoop = Math.max(1, Math.min(kLoops, Math.trunc(loopIteration)));
    const cached = counterfactualCache.get(clampedLoop);
    if (cached !== undefined) {
      return cached;
    }
    const result = session.evaluateCounterfactualIntermediateCoda(record.recordId, clampedLoop);
    counterfactualCache.set(clampedLoop, result);
    return result;
  };

  return {
    params: session.params,
    ropeTables: session.ropeTables,
    session,
    branch,
    record,
    kLoops,
    kvCacheMode,
    executionMode,
    isHistoricalCursor,
    recordIndexInBranch,
    branchRecordCount,
    resolveStepTrace,
    resolveForwardTrace,
    resolveSublayerRecord,
    resolveDefaultTokenOffset,
    resolveCounterfactualCoda,
    provenanceBadges: Object.freeze(badges),
  };
}

/**
 * Normalizes either a `DorySession` or an already-resolved `DoryResolvedSessionContext`
 * into a context aligned with optional `recordId` (re-resolving cleanly if `recordId` differs).
 */
export function ensureDorySessionContext(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  recordId?: string,
): DoryResolvedSessionContext {
  if ('resolveForwardTrace' in sessionOrCtx) {
    if (recordId !== undefined && recordId !== sessionOrCtx.record.recordId) {
      return resolveDorySessionContext(sessionOrCtx.session, recordId);
    }
    return sessionOrCtx;
  }
  return resolveDorySessionContext(sessionOrCtx, recordId);
}

// ============================================================================
// DR21 PICK ADAPTER (ZERO EDITS TO `src/render/pick.ts`)
// ============================================================================

function parseTagInt(tag: string | undefined, key: string): number | null {
  if (!tag) return null;
  const re = new RegExp(`(?:^|[;:,\\s])${key}=(-?\\d+)(?:$|[;:,\\s])`);
  const m = re.exec(tag);
  if (!m) return null;
  const parsed = Number.parseInt(m[1]!, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseTagToken(tag: string | undefined, key: string): string | null {
  if (!tag) return null;
  const re = new RegExp(`(?:^|[;\\s])${key}=([^;\\s]+)`);
  const m = re.exec(tag);
  return m ? m[1]! : null;
}

function parseStageValue(raw: string | null): DoryLayerStage | 'bookend' | null {
  if (raw === 'input' || raw === 'recurrent' || raw === 'output' || raw === 'bookend') {
    return raw;
  }
  return null;
}

function parseBankKindValue(raw: string | null): DoryKvBankKind | null {
  if (
    raw === 'input' ||
    raw === 'recurrent' ||
    raw === 'output_deep' ||
    raw === 'output_shallow'
  ) {
    return raw;
  }
  return null;
}

function parseKvPolicyValue(raw: string | null): DoryKvCacheMode | null {
  if (raw === 'per_loop' || raw === 'last_loop') {
    return raw;
  }
  return null;
}

function parseCacheRoleValue(raw: string | null): DoryCacheRole | null {
  if (
    raw === 'standard' ||
    raw === 'spec_draft' ||
    raw === 'spec_verify_reused' ||
    raw === 'spec_verify_deep'
  ) {
    return raw;
  }
  return null;
}

function parseGroupRoleValue(raw: string | null): DoryPickGroupRole | null {
  if (raw === 'kv_head' || raw === 'bank' || raw === 'loop') {
    return raw;
  }
  return null;
}

function parseTapRoleValue(raw: string | null): DoryPickTapRole | null {
  if (raw === 'loop' || raw === 'plane' || raw === 'spec') {
    return raw;
  }
  return null;
}

/**
 * Encodes Dory1.0-4B spatial coordinates (physical/paired/executed layer, stage, loop iteration,
 * KV head, KV bank slot/kind, rotary plane, speculative offset, policy, cache role) into shared
 * `PickBox` fields (`bayId`, `groupIndex`, `tapIndex`, `headIndex`, `stateTokenIndex`, `semanticTag`)
 * with ZERO edits to `src/render/pick.ts` (`DR21`).
 */
export function encodeDoryPickMetadata(input: DoryPickEncodeInput): DoryPickBoxMeta {
  const phys =
    input.physicalLayerIdx !== undefined && input.physicalLayerIdx !== null
      ? Math.max(0, Math.min(DORY_MINI_CONFIG.numHiddenLayers - 1, Math.trunc(input.physicalLayerIdx)))
      : null;

  const bayId = phys !== null ? phys + 1 : (input.bayId ?? 0);
  const stage: DoryLayerStage | 'bookend' | undefined =
    input.stage ?? (phys !== null ? getDoryStage(phys) : undefined);

  const towerRole: 'context' | 'denoiser' | 'shared' | undefined =
    stage === 'input'
      ? 'context'
      : stage === 'recurrent'
        ? 'shared'
        : stage === 'output'
          ? 'denoiser'
          : undefined;

  const groupRole: DoryPickGroupRole | undefined =
    input.groupRole ??
    (input.kvBankSlotId !== undefined && input.kvBankSlotId !== null && input.kvHeadIndex == null
      ? 'bank'
      : input.kvHeadIndex !== undefined && input.kvHeadIndex !== null
        ? 'kv_head'
        : input.loopIteration !== undefined && input.loopIteration !== null
          ? 'loop'
          : undefined);

  let groupIndex: number | undefined;
  if (groupRole === 'bank' && input.kvBankSlotId !== undefined && input.kvBankSlotId !== null) {
    groupIndex = Math.trunc(input.kvBankSlotId);
  } else if (groupRole === 'loop' && input.loopIteration !== undefined && input.loopIteration !== null) {
    groupIndex = Math.trunc(input.loopIteration);
  } else if (groupRole === 'kv_head' && input.kvHeadIndex !== undefined && input.kvHeadIndex !== null) {
    groupIndex = Math.trunc(input.kvHeadIndex);
  }

  const tapRole: DoryPickTapRole | undefined =
    input.tapRole ??
    (input.rotaryPlane !== undefined && input.rotaryPlane !== null
      ? 'plane'
      : input.specOffset !== undefined && input.specOffset !== null
        ? 'spec'
        : input.loopIteration !== undefined && input.loopIteration !== null
          ? 'loop'
          : undefined);

  let tapIndex: number | undefined;
  if (tapRole === 'plane' && input.rotaryPlane !== undefined && input.rotaryPlane !== null) {
    tapIndex = Math.trunc(input.rotaryPlane);
  } else if (tapRole === 'spec' && input.specOffset !== undefined && input.specOffset !== null) {
    tapIndex = Math.trunc(input.specOffset);
  } else if (tapRole === 'loop' && input.loopIteration !== undefined && input.loopIteration !== null) {
    tapIndex = Math.trunc(input.loopIteration);
  }

  const tags: string[] = [];
  if (stage !== undefined) tags.push(`stage=${stage}`);
  if (phys !== null) {
    tags.push(`phys=${phys}`);
    tags.push(`paired=${phys >> 1}`);
  }
  if (input.executedLayerIdx !== undefined && input.executedLayerIdx !== null) {
    tags.push(`exec=${Math.trunc(input.executedLayerIdx)}`);
  }
  if (input.loopIteration !== undefined && input.loopIteration !== null) {
    tags.push(`loop=${Math.trunc(input.loopIteration)}`);
  }
  if (input.kvHeadIndex !== undefined && input.kvHeadIndex !== null) {
    tags.push(`kv=${Math.trunc(input.kvHeadIndex)}`);
  }
  if (input.kvBankSlotId !== undefined && input.kvBankSlotId !== null) {
    tags.push(`bank=${Math.trunc(input.kvBankSlotId)}`);
  }
  if (input.kvBankKind !== undefined && input.kvBankKind !== null) {
    tags.push(`bankKind=${input.kvBankKind}`);
  }
  if (input.kvPolicy !== undefined && input.kvPolicy !== null) {
    tags.push(`policy=${input.kvPolicy}`);
  }
  if (input.cacheRole !== undefined && input.cacheRole !== null) {
    tags.push(`role=${input.cacheRole}`);
  }
  if (input.rotaryPlane !== undefined && input.rotaryPlane !== null) {
    tags.push(`plane=${Math.trunc(input.rotaryPlane)}`);
  }
  if (input.specOffset !== undefined && input.specOffset !== null) {
    tags.push(`spec=${Math.trunc(input.specOffset)}`);
  }
  if (groupRole !== undefined) tags.push(`groupRole=${groupRole}`);
  if (tapRole !== undefined) tags.push(`tapRole=${tapRole}`);
  if (input.extraTags !== undefined) {
    const keys = Object.keys(input.extraTags).sort();
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i]!;
      tags.push(`${k}=${String(input.extraTags[k])}`);
    }
  }

  const kvHeadNarrow: 0 | 1 | undefined =
    input.kvHeadIndex === 0 || input.kvHeadIndex === 1 ? input.kvHeadIndex : undefined;

  return {
    bayId,
    ...(input.partId !== undefined ? { partId: input.partId } : {}),
    ...(input.qHeadIndex !== undefined && input.qHeadIndex !== null
      ? { headIndex: Math.trunc(input.qHeadIndex) }
      : {}),
    ...(kvHeadNarrow !== undefined ? { kvHeadIndex: kvHeadNarrow } : {}),
    ...(groupIndex !== undefined ? { groupIndex } : {}),
    ...(input.tensorKey !== undefined ? { tensorKey: input.tensorKey } : {}),
    ...(input.gridCols !== undefined ? { gridCols: input.gridCols } : {}),
    ...(input.gridRows !== undefined ? { gridRows: input.gridRows } : {}),
    ...(input.tokenIndex !== undefined && input.tokenIndex !== null
      ? { tokenIndex: Math.trunc(input.tokenIndex) }
      : {}),
    ...(tapIndex !== undefined ? { tapIndex } : {}),
    ...(input.executedLayerIdx !== undefined && input.executedLayerIdx !== null
      ? { stateTokenIndex: Math.trunc(input.executedLayerIdx) }
      : {}),
    ...(towerRole !== undefined ? { towerRole } : {}),
    ...(input.modelWeightId !== undefined ? { executionId: input.modelWeightId } : {}),
    ...(input.traceOrigin !== undefined || input.cacheRole != null
      ? { traceOrigin: input.traceOrigin ?? input.cacheRole ?? undefined }
      : {}),
    ...(tags.length > 0 ? { semanticTag: tags.join(';') } : {}),
    ...(input.absTokenPos !== undefined && input.absTokenPos !== null
      ? { absoluteTokenOffset: Math.trunc(input.absTokenPos) }
      : {}),
  };
}

const Q_CONCAT_TENSOR_KEYS = new Set<string>([
  'q_pre',
  'q_rot',
  'q_norm',
  'q_scaled',
  'g_pre',
  'gate_values',
  'attn_context',
  'gated_context',
  'attn_sub_norm',
  'q_proj',
  'g_proj',
  'o_proj',
  'sqk',
  's_gate',
]);

const KV_CONCAT_TENSOR_KEYS = new Set<string>([
  'k_pre',
  'k_rot',
  'k_cache',
  'v_new',
  'k_proj',
  'v_proj',
]);

const ROTARY_TENSOR_KEYS = new Set<string>([
  'q_pre',
  'q_rot',
  'q_norm',
  'q_scaled',
  'k_pre',
  'k_rot',
  'k_cache',
  'rope_cos',
  'rope_sin',
]);

/**
 * Decodes a `PickHit` from untouched `src/render/pick.ts` into a full `DoryPickIdentity` (`DR21`),
 * supporting both `encodeDoryPickMetadata`-authored `PickBox`es and raw `PickHit` structs using
 * `bayId`, `groupIndex`, `tapIndex`, `headIndex`, `stateTokenIndex`, and `semanticTag`.
 */
export function pickHitToDoryCell(
  hit: PickHit | null,
  viewMode: DorySpatialViewMode,
): DoryPickIdentity | null {
  if (hit === null) return null;

  const tag = hit.semanticTag ?? '';
  const groupRole = parseGroupRoleValue(parseTagToken(tag, 'groupRole'));
  const tapRole = parseTapRoleValue(parseTagToken(tag, 'tapRole'));

  const tagPhys = parseTagInt(tag, 'phys');
  const physicalLayerIdx =
    tagPhys !== null && tagPhys >= 0 && tagPhys < DORY_MINI_CONFIG.numHiddenLayers
      ? tagPhys
      : hit.bayId >= 1 && hit.bayId <= DORY_MINI_CONFIG.numHiddenLayers
        ? hit.bayId - 1
        : null;

  const pairedLayerIdx = physicalLayerIdx !== null ? physicalLayerIdx >> 1 : null;
  const sublayerKind = physicalLayerIdx !== null ? getDorySublayerKind(physicalLayerIdx) : null;

  let stage: DoryLayerStage | 'bookend' | null = parseStageValue(parseTagToken(tag, 'stage'));
  if (stage === null && physicalLayerIdx !== null) {
    stage = getDoryStage(physicalLayerIdx);
  }
  if (stage === null) {
    if (hit.towerRole === 'context') stage = 'input';
    else if (hit.towerRole === 'shared') stage = 'recurrent';
    else if (hit.towerRole === 'denoiser') stage = 'output';
  }

  let loopIteration: number | null = parseTagInt(tag, 'loop');
  if (loopIteration === null && groupRole === 'loop' && hit.groupIndex !== undefined) {
    loopIteration = hit.groupIndex;
  }
  if (
    loopIteration === null &&
    tapRole !== 'plane' &&
    tapRole !== 'spec' &&
    hit.tapIndex !== undefined &&
    hit.tapIndex >= 0
  ) {
    loopIteration = hit.tapIndex;
  }
  if (loopIteration === null && (stage === 'input' || stage === 'output')) {
    loopIteration = 0;
  }

  let executedLayerIdx: number | null = parseTagInt(tag, 'exec');
  if (
    executedLayerIdx === null &&
    hit.stateTokenIndex !== undefined &&
    hit.stateTokenIndex >= 0
  ) {
    executedLayerIdx = hit.stateTokenIndex;
  }

  let qHeadIndex: number | null =
    hit.headIndex !== undefined &&
    hit.headIndex >= 0 &&
    hit.headIndex < DORY_MINI_CONFIG.numAttentionHeads
      ? hit.headIndex
      : null;

  let kvHeadIndex: number | null = parseTagInt(tag, 'kv');
  if (
    kvHeadIndex === null &&
    groupRole !== 'bank' &&
    groupRole !== 'loop' &&
    hit.groupIndex !== undefined &&
    hit.groupIndex >= 0
  ) {
    kvHeadIndex = hit.groupIndex;
  }
  if (kvHeadIndex === null && hit.kvHeadIndex !== undefined) {
    kvHeadIndex = hit.kvHeadIndex;
  }

  const tKey = hit.tensorKey ?? '';
  if (qHeadIndex === null && hit.hitCol !== undefined && Q_CONCAT_TENSOR_KEYS.has(tKey)) {
    const inferredQ = Math.floor(hit.hitCol / DORY_MINI_CONFIG.headDim);
    if (inferredQ >= 0 && inferredQ < DORY_MINI_CONFIG.numAttentionHeads) {
      qHeadIndex = inferredQ;
    }
  }
  if (kvHeadIndex === null && hit.hitCol !== undefined && KV_CONCAT_TENSOR_KEYS.has(tKey)) {
    const inferredKv = Math.floor(hit.hitCol / DORY_MINI_CONFIG.headDim);
    if (inferredKv >= 0 && inferredKv < DORY_MINI_CONFIG.numKeyValueHeads) {
      kvHeadIndex = inferredKv;
    }
  }
  if (kvHeadIndex === null && qHeadIndex !== null) {
    kvHeadIndex = Math.floor(qHeadIndex / DORY_MINI_CONFIG.kvGroupSize);
  }

  let kvBankSlotId: number | null = parseTagInt(tag, 'bank');
  if (
    kvBankSlotId === null &&
    groupRole === 'bank' &&
    hit.groupIndex !== undefined &&
    hit.groupIndex >= 0
  ) {
    kvBankSlotId = hit.groupIndex;
  }

  let kvBankKind: DoryKvBankKind | null = parseBankKindValue(parseTagToken(tag, 'bankKind'));
  if (kvBankKind === null && kvBankSlotId !== null && stage !== null && stage !== 'bookend') {
    kvBankKind = stage === 'output' ? 'output_deep' : stage;
  }

  const kvPolicy = parseKvPolicyValue(parseTagToken(tag, 'policy'));
  const cacheRole =
    parseCacheRoleValue(parseTagToken(tag, 'role')) ??
    parseCacheRoleValue(hit.traceOrigin ?? null);

  let rotaryPlane: number | null = parseTagInt(tag, 'plane');
  if (
    rotaryPlane === null &&
    tapRole === 'plane' &&
    hit.tapIndex !== undefined &&
    hit.tapIndex >= 0 &&
    hit.tapIndex < DORY_MINI_CONFIG.rotaryPlanes
  ) {
    rotaryPlane = hit.tapIndex;
  }
  if (rotaryPlane === null && hit.hitCol !== undefined && ROTARY_TENSOR_KEYS.has(tKey)) {
    rotaryPlane = hit.hitCol % DORY_MINI_CONFIG.rotaryPlanes;
  }

  let specOffset: number | null = parseTagInt(tag, 'spec');
  if (
    specOffset === null &&
    tapRole === 'spec' &&
    hit.tapIndex !== undefined &&
    hit.tapIndex >= 0
  ) {
    specOffset = hit.tapIndex;
  }

  const tokenIndex =
    hit.tokenIndex !== undefined
      ? hit.tokenIndex
      : hit.hitRow !== undefined
        ? hit.hitRow
        : null;

  const absTokenPos =
    hit.absoluteTokenOffset !== undefined ? hit.absoluteTokenOffset : null;

  return Object.freeze({
    viewMode,
    physicalLayerIdx,
    pairedLayerIdx,
    executedLayerIdx,
    stage,
    sublayerKind,
    loopIteration,
    partId: hit.partId ?? null,
    tensorKey: hit.tensorKey ?? null,
    qHeadIndex,
    kvHeadIndex,
    rotaryPlane,
    kvBankSlotId,
    kvBankKind,
    kvPolicy,
    cacheRole,
    specOffset,
    tokenIndex,
    absTokenPos,
    row: hit.hitRow ?? null,
    col: hit.hitCol ?? null,
    modelWeightId: hit.executionId ?? null,
    semanticTag: hit.semanticTag ?? null,
  });
}

export function packPickBoxFromAabb(
  cx: number,
  cy: number,
  cz: number,
  ex: number,
  ey: number,
  ez: number,
  meta: DoryPickBoxMeta,
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

export function packPickBoxFromFrame(frame: SheetFrame3D, meta: DoryPickBoxMeta): PickBox {
  const aabb = sheetFrameWorldAabb(frame);
  return {
    ...meta,
    min: aabb.min,
    max: aabb.max,
    frame,
  };
}

// ============================================================================
// RG32F ATLAS PACKER, NON-OVERLAP AUDITOR & DETERMINISTIC CONTENT KEY
// ============================================================================

/** Pure-TS deterministic 64-bit (dual 32-bit FNV-1a) hex digest for GPU atlas deduplication. */
function dualFnv1aBytesHex(bytes: Uint8Array): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < bytes.length; i += 1) {
    const b = bytes[i]!;
    h1 ^= b;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= b ^ (i & 0xff);
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 0x7feb352d) >>> 0;
  h2 = Math.imul(h2 ^ (h2 >>> 15), 0x846ca68b) >>> 0;
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

/** Deterministic content-addressed key for `RG32F` WebGL2 texture upload deduplication. */
export function computeDoryAtlasContentKey(
  viewMode: DorySpatialViewMode,
  atlas: Rg32fValueAtlas,
  extraTag = '',
): string {
  const bytes = new Uint8Array(atlas.data.buffer, atlas.data.byteOffset, atlas.data.byteLength);
  const digest = dualFnv1aBytesHex(bytes);
  return `${viewMode}:${atlas.width}x${atlas.height}:${extraTag}:${digest}`;
}

/**
 * Exhaustive `RG32F` atlas layout auditor:
 *   - Verifies every region has positive integer dimensions strictly inside `[0..width) x [0..height)`.
 *   - Verifies pairwise zero-overlap across all allocated `DoryAtlasRegionBox` regions.
 *   - Verifies total `RG32F` payload (`width * height * 8 B`) is `<= maxBudgetBytes` (`<= 512 KiB`).
 */
export function auditAtlasRegions(
  width: number,
  height: number,
  regions: readonly DoryAtlasRegionBox[],
  maxBudgetBytes: number = DORY_MAX_ATLAS_BYTES,
): DoryAtlasLayoutAudit {
  const totalTexels = Math.max(0, width) * Math.max(0, height);
  const byteLength = totalTexels * 8; // RG32F = 2 * 4 bytes/texel
  const withinBudget =
    Number.isInteger(width) &&
    Number.isInteger(height) &&
    width > 0 &&
    height > 0 &&
    byteLength <= maxBudgetBytes;

  let usedTexels = 0;
  let hasOverlap = false;
  let overlapPair: readonly [string, string] | null = null;
  const outOfBoundsIds: string[] = [];

  for (let i = 0; i < regions.length; i += 1) {
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
      for (let j = i + 1; j < regions.length; j += 1) {
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

export interface DoryAtlasPackMatrixOptions {
  readonly mode?: 0 | 1;
  /** Optional binary mask (`1` = masked -> `G = 1, R = 0`, `0` = visible -> `G = 0`). */
  readonly masked?: ArrayLike<number>;
  readonly isMaskOrScore?: boolean;
  readonly minFloor?: number;
  readonly fixedScale?: number;
  readonly physicalLayerIdx?: number;
  readonly loopIteration?: number;
}

/**
 * Deterministic shelf-packing `RG32F` atlas builder backed by `src/scene/atlasPacker.ts`.
 * Guarantees zero overlaps, `<= 512 KiB` budget compliance, and exact `DR20` `G`-channel mask
 * population on every SWA/Full causal score or softmax plate.
 */
export class DoryAtlasPacker {
  private readonly writer: Rg32fAtlasWriter;
  private readonly maxBudgetBytes: number;
  private readonly regions: DoryAtlasRegionBox[] = [];
  private cursorX = 0;
  private cursorY = 0;
  private shelfHeight = 0;

  constructor(
    readonly width: number,
    readonly height: number,
    maxBudgetBytes: number = DORY_MAX_ATLAS_BYTES,
  ) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
      throw new RangeError(`Invalid atlas dimensions ${width}x${height}: expected positive integers.`);
    }
    const byteLength = width * height * 8;
    if (byteLength > maxBudgetBytes) {
      throw new RangeError(
        `Atlas ${width}x${height} (${byteLength} B) exceeds maxBudgetBytes (${maxBudgetBytes} B).`,
      );
    }
    this.maxBudgetBytes = maxBudgetBytes;
    this.writer = createRg32fAtlas(width, height);
  }

  private allocateRect(
    id: string,
    cols: number,
    rows: number,
    mode: 0 | 1,
    isMaskOrScore: boolean,
    physicalLayerIdx?: number,
    loopIteration?: number,
  ): { readonly x0: number; readonly y0: number; readonly rect: TexRect } {
    if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 1 || rows < 1) {
      throw new RangeError(`Invalid atlas slice '${id}' dims ${cols}x${rows}.`);
    }
    if (cols > this.width) {
      throw new RangeError(`Atlas slice '${id}' width ${cols} exceeds atlas width ${this.width}.`);
    }
    if (this.cursorX + cols > this.width) {
      this.cursorX = 0;
      this.cursorY += this.shelfHeight;
      this.shelfHeight = 0;
    }
    if (this.cursorY + rows > this.height) {
      throw new RangeError(
        `Atlas overflow packing '${id}' (${cols}x${rows}) at y=${this.cursorY} in ${this.width}x${this.height} atlas.`,
      );
    }
    const x0 = this.cursorX;
    const y0 = this.cursorY;
    this.cursorX += cols;
    if (rows > this.shelfHeight) {
      this.shelfHeight = rows;
    }

    this.regions.push(
      Object.freeze({
        id,
        x0,
        y0,
        cols,
        rows,
        mode,
        isMaskOrScore,
        ...(physicalLayerIdx !== undefined ? { physicalLayerIdx } : {}),
        ...(loopIteration !== undefined ? { loopIteration } : {}),
      }),
    );

    return {
      x0,
      y0,
      rect: this.writer.makeRect(x0, y0, cols, rows),
    };
  }

  /** Pack a row-major `[rows, cols]` scalar matrix into `RG32F` (`R = value`, `G = masked`). */
  packMatrix(
    id: string,
    data: ArrayLike<number>,
    rows: number,
    cols: number,
    options: DoryAtlasPackMatrixOptions = {},
  ): AtlasSlice {
    const mode: 0 | 1 = options.mode ?? 0;
    const minFloor = options.minFloor ?? 0.25;
    const alloc = this.allocateRect(
      id,
      cols,
      rows,
      mode,
      options.isMaskOrScore ?? false,
      options.physicalLayerIdx,
      options.loopIteration,
    );

    let maxAbs = minFloor;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const idx = r * cols + c;
        const val = idx < data.length ? data[idx]! : 0;
        const isMasked =
          options.masked !== undefined && idx < options.masked.length
            ? options.masked[idx]! > 0.5
            : false;
        this.writer.setTexel(alloc.x0 + c, alloc.y0 + r, val, isMasked);
        if (!isMasked && Number.isFinite(val)) {
          const mag = Math.abs(val);
          if (mag > maxAbs) {
            maxAbs = mag;
          }
        }
      }
    }

    const cellCount = rows * cols;
    const scale =
      options.fixedScale !== undefined
        ? options.fixedScale
        : mode === 1
          ? 1.0
          : 1.0 /
            (data instanceof Float32Array && !options.masked && data.length >= cellCount
              ? computeMaxAbs(data.subarray(0, cellCount), minFloor)
              : maxAbs);

    return Object.freeze({
      rect: alloc.rect,
      scale,
      mode,
    });
  }

  /**
   * Pack an SWA or Full attention score/softmax matrix `[rows, cols]` with explicit binary
   * visibility mask `maskVisible` (`1` = visible -> `G = 0`, `0` = masked -> `G = 1, R = 0`),
   * satisfying `DR20` by construction.
   */
  packCausalMatrix(
    id: string,
    data: ArrayLike<number>,
    maskVisible: ArrayLike<number>,
    rows: number,
    cols: number,
    mode: 0 | 1,
    options: Omit<DoryAtlasPackMatrixOptions, 'mode' | 'masked' | 'isMaskOrScore'> = {},
  ): AtlasSlice {
    const minFloor = options.minFloor ?? 0.25;
    const alloc = this.allocateRect(
      id,
      cols,
      rows,
      mode,
      true,
      options.physicalLayerIdx,
      options.loopIteration,
    );

    let maxAbs = minFloor;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const idx = r * cols + c;
        const visible = idx < maskVisible.length ? maskVisible[idx]! > 0.5 : false;
        const val = idx < data.length ? data[idx]! : 0;
        if (!visible || !Number.isFinite(val)) {
          this.writer.setTexel(alloc.x0 + c, alloc.y0 + r, 0, true);
        } else {
          this.writer.setTexel(alloc.x0 + c, alloc.y0 + r, val, false);
          const mag = Math.abs(val);
          if (mag > maxAbs) {
            maxAbs = mag;
          }
        }
      }
    }

    const scale =
      options.fixedScale !== undefined
        ? options.fixedScale
        : mode === 1
          ? 1.0
          : 1.0 / maxAbs;

    return Object.freeze({
      rect: alloc.rect,
      scale,
      mode,
    });
  }

  build(
    viewMode: DorySpatialViewMode,
    extraTag = '',
  ): {
    readonly valueAtlas: Rg32fValueAtlas;
    readonly atlasKey: string;
    readonly atlasRegions: readonly DoryAtlasRegionBox[];
    readonly atlasAudit: DoryAtlasLayoutAudit;
  } {
    const valueAtlas: Rg32fValueAtlas = {
      width: this.writer.width,
      height: this.writer.height,
      data: this.writer.data,
    };
    const atlasRegions = Object.freeze([...this.regions]);
    const atlasAudit = auditAtlasRegions(
      this.writer.width,
      this.writer.height,
      atlasRegions,
      this.maxBudgetBytes,
    );
    const atlasKey = computeDoryAtlasContentKey(viewMode, valueAtlas, extraTag);
    return {
      valueAtlas,
      atlasKey,
      atlasRegions,
      atlasAudit,
    };
  }
}

// ============================================================================
// SHARED 5-PASS GPU BUFFER BUILDERS (MATCHING `src/render/*` EXACT STRIDES)
// ============================================================================

export class DorySolidBufferBuilder {
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

export class DoryLineBufferBuilder {
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

export class DoryRibbonBufferBuilder {
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

/** Typed wrapper over shared `FlowRibbonBuilder` (`src/render/ribbons.ts`). */
export class DoryFlatFlowBufferBuilder {
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

export interface DorySheetPushParams {
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

function isValidLiveTexRect(rect: TexRect): boolean {
  const [u0, v0, uW, vH] = rect;
  return (
    Number.isFinite(u0) &&
    Number.isFinite(v0) &&
    Number.isFinite(uW) &&
    Number.isFinite(vH) &&
    u0 >= 0 &&
    v0 >= 0 &&
    uW > 0 &&
    vH > 0 &&
    u0 + uW <= 1.0 + 1e-5 &&
    v0 + vH <= 1.0 + 1e-5
  );
}

/**
 * Packed `SheetsPass` instance builder with HARD `DR20` enforcement:
 * Throws `DoryDr20CausalMaskError` immediately if `SHEET_KIND_CAUSAL_SCORES` (`1`) or
 * `SHEET_KIND_CAUSAL_SOFTMAX` (`2`) is pushed without a valid live `RG32F` `G`-channel mask
 * `TexRect` (`uWidth > 0 && vHeight > 0`, finite and within `[0, 1]`), preventing the
 * `src/render/sheets.ts:347-368` fallback to geometric `cellCol > cellRow` masking!
 */
export class DorySheetBufferBuilder {
  private readonly inst: number[] = [];
  private readonly basis: number[] = [];
  private readonly reveal: number[] = [];
  private count = 0;

  push(params: DorySheetPushParams): void {
    const sheetKind = params.sheetKind ?? SHEET_KIND_STANDARD;
    const rect: TexRect = params.slice
      ? params.slice.rect
      : (params.texRect ?? [0, 0, 0, 0]);
    const valueScale = params.slice ? params.slice.scale : (params.valueScale ?? 1.0);
    const flagMode = params.slice ? params.slice.mode : (params.flagMode ?? 0);

    // DR20 Hard Guard: forbid SheetsPass geometric `cellCol > cellRow` fallback on SWA/Full plates!
    if (
      (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) &&
      !isValidLiveTexRect(rect)
    ) {
      throw new DoryDr20CausalMaskError(
        sheetKind,
        `DR20 violation: sheetKind=${sheetKind} requires a live RG32F G-channel mask TexRect (got [${rect.join(', ')}]).`,
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
// WORLD AABB HELPERS
// ============================================================================

export function computeBoundsFromPoints(points: readonly Vec3[], pad = 0.5): AABB {
  if (points.length === 0) {
    return {
      min: [-pad, -pad, -pad],
      max: [pad, pad, pad],
    };
  }
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < points.length; i += 1) {
    const p = points[i]!;
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[2] < minZ) minZ = p[2];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
    if (p[2] > maxZ) maxZ = p[2];
  }
  return {
    min: [minX - pad, minY - pad, minZ - pad],
    max: [maxX + pad, maxY + pad, maxZ + pad],
  };
}

export function mergeAabbs(aabbs: readonly AABB[]): AABB {
  if (aabbs.length === 0) {
    return { min: [-1, -1, -1], max: [1, 1, 1] };
  }
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < aabbs.length; i += 1) {
    const b = aabbs[i]!;
    if (b.min[0] < minX) minX = b.min[0];
    if (b.min[1] < minY) minY = b.min[1];
    if (b.min[2] < minZ) minZ = b.min[2];
    if (b.max[0] > maxX) maxX = b.max[0];
    if (b.max[1] > maxY) maxY = b.max[1];
    if (b.max[2] > maxZ) maxZ = b.max[2];
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
  };
}

// ============================================================================
// EXHAUSTIVE SCENE GEOMETRY FINITENESS, ATLAS BUDGET & DR20 VALIDATOR
// ============================================================================

function checkFiniteArray(
  arr: ArrayLike<number> | undefined | null,
  label: string,
  errors: string[],
): void {
  if (!arr) return;
  for (let i = 0; i < arr.length; i += 1) {
    const v = arr[i]!;
    if (!Number.isFinite(v)) {
      errors.push(`Non-finite value ${String(v)} in ${label}[${i}]`);
      return;
    }
  }
}

function checkFiniteVec3(v: Vec3, label: string, errors: string[]): void {
  if (!Number.isFinite(v[0]) || !Number.isFinite(v[1]) || !Number.isFinite(v[2])) {
    errors.push(`Non-finite Vec3 in ${label}`);
  }
}

function checkFiniteAabb(bounds: AABB, label: string, errors: string[]): void {
  for (let axis = 0; axis < 3; axis += 1) {
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

function verifyRg32fCausalMaskTexels(
  atlas: Rg32fValueAtlas,
  rect: TexRect,
): string | null {
  const { width, height, data } = atlas;
  if (width <= 0 || height <= 0 || data.length !== width * height * 2) {
    return `invalid RG32F atlas storage (${width}x${height}, len=${data.length})`;
  }
  const x0 = Math.max(0, Math.floor(rect[0] * width + 1e-4));
  const y0 = Math.max(0, Math.floor(rect[1] * height + 1e-4));
  const x1 = Math.min(width, Math.ceil((rect[0] + rect[2]) * width - 1e-4));
  const y1 = Math.min(height, Math.ceil((rect[1] + rect[3]) * height - 1e-4));
  if (x1 <= x0 || y1 <= y0) {
    return `empty RG32F mask texel footprint (${x0}..${x1} x ${y0}..${y1})`;
  }
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const idx = (y * width + x) * 2;
      const g = data[idx + 1]!;
      if (!Number.isFinite(g) || (Math.abs(g) > 1e-5 && Math.abs(g - 1.0) > 1e-5)) {
        return `non-binary RG32F G-channel mask value ${String(g)} at texel (${x},${y})`;
      }
    }
  }
  return null;
}

/**
 * Exhaustive scene geometry, stride, atlas-budget & `DR20` validator:
 *   - Every float across all 5 WebGL2 passes, `RG32F` atlas, pick boxes, bounds, camera preset,
 *     overlay anchors, landmarks, calipers, and KV ticks MUST be finite (0 NaN / 0 Inf).
 *   - Buffer lengths match exact `SOLID_INSTANCE_FLOATS` / `LINE_INSTANCE_FLOATS` /
 *     `RIBBON_INSTANCE_FLOATS` / `SHEET_*_FLOATS` / `FLOW_RIBBON_VERTEX_FLOATS` strides.
 *   - `DR20` HARD GUARD: verifies every `sheetKind in {1, 2}` binds a live `RG32F` atlas,
 *     valid positive `TexRect`, and binary `G`-channel mask texels — throwing
 *     `DoryDr20CausalMaskError` by default (`options.throwOnDr20 !== false`).
 *   - Atlas audit confirms `0` overlaps, `0` out-of-bounds rects, and `<= 512 KiB` (`DORY_MAX_ATLAS_BYTES`).
 */
export function validateSceneGeometryFinite(
  scene: DorySpatialScene,
  options: DorySceneValidationOptions = {},
): DorySceneValidationResult {
  const throwOnDr20 = options.throwOnDr20 ?? true;
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

  if (
    !Number.isFinite(scene.cameraPreset.azimuthDeg) ||
    !Number.isFinite(scene.cameraPreset.elevationDeg)
  ) {
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
  if (scene.flowVertices.length % FLOW_RIBBON_VERTEX_FLOATS !== 0) {
    errors.push(
      `${scene.viewMode}.flowVertices length (${scene.flowVertices.length}) not divisible by ${FLOW_RIBBON_VERTEX_FLOATS}`,
    );
  }
  const flowVertCount = Math.floor(scene.flowVertices.length / FLOW_RIBBON_VERTEX_FLOATS);
  for (let i = 0; i < scene.flowIndices.length; i += 1) {
    if (scene.flowIndices[i]! >= flowVertCount) {
      errors.push(
        `${scene.viewMode}.flowIndices[${i}]=${scene.flowIndices[i]} out of bounds for ${flowVertCount} vertices`,
      );
      break;
    }
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

    // DR20 Hard Audit: every causal score/softmax sheet MUST bind a live RG32F G-channel mask!
    for (let i = 0; i < scene.sheetCount; i += 1) {
      const base = i * SHEET_INSTANCE_FLOATS;
      const sheetKind = Math.round(scene.sheetData[base + 8]!);
      if (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) {
        const rect: TexRect = [
          scene.sheetData[base + 13]!,
          scene.sheetData[base + 14]!,
          scene.sheetData[base + 15]!,
          scene.sheetData[base + 16]!,
        ];
        let dr20Failure: string | null = null;
        if (scene.valueAtlas === null) {
          dr20Failure = `DR20 violation in ${scene.viewMode}.sheet[${i}]: sheetKind=${sheetKind} has valueAtlas === null`;
        } else if (!isValidLiveTexRect(rect)) {
          dr20Failure = `DR20 violation in ${scene.viewMode}.sheet[${i}]: sheetKind=${sheetKind} lacks valid live RG32F TexRect ([${rect.join(', ')}])`;
        } else {
          const maskErr = verifyRg32fCausalMaskTexels(scene.valueAtlas, rect);
          if (maskErr !== null) {
            dr20Failure = `DR20 violation in ${scene.viewMode}.sheet[${i}]: sheetKind=${sheetKind} ${maskErr}`;
          }
        }
        if (dr20Failure !== null) {
          if (throwOnDr20) {
            throw new DoryDr20CausalMaskError(sheetKind, dr20Failure);
          }
          errors.push(dr20Failure);
        }
      }
    }
  }

  if (scene.valueAtlas !== null) {
    const atlasBytes = scene.valueAtlas.width * scene.valueAtlas.height * 8;
    if (scene.valueAtlas.data.length !== scene.valueAtlas.width * scene.valueAtlas.height * 2) {
      errors.push(
        `${scene.viewMode}.valueAtlas.data length (${scene.valueAtlas.data.length}) !== 2 * ${scene.valueAtlas.width} * ${scene.valueAtlas.height}`,
      );
    }
    if (atlasBytes > DORY_MAX_ATLAS_BYTES) {
      errors.push(
        `${scene.viewMode}.valueAtlas byteLength (${atlasBytes}) exceeds hard ceiling ${DORY_MAX_ATLAS_BYTES} (512 KiB)`,
      );
    }
  }

  if (scene.atlasAudit !== null) {
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
    if (!scene.atlasAudit.withinBudget || scene.atlasAudit.byteLength > DORY_MAX_ATLAS_BYTES) {
      errors.push(
        `Atlas byteLength (${scene.atlasAudit.byteLength}) exceeds budget (${Math.min(scene.atlasAudit.maxBudgetBytes, DORY_MAX_ATLAS_BYTES)}) in ${scene.viewMode}`,
      );
    }
  }

  for (let i = 0; i < scene.pickBoxes.length; i += 1) {
    const pb = scene.pickBoxes[i]!;
    checkFiniteAabb({ min: pb.min, max: pb.max }, `${scene.viewMode}.pickBoxes[${i}]`, errors);
    if (pb.frame !== undefined) {
      checkFiniteVec3(pb.frame.center, `${scene.viewMode}.pickBoxes[${i}].frame.center`, errors);
      checkFiniteVec3(pb.frame.extent, `${scene.viewMode}.pickBoxes[${i}].frame.extent`, errors);
      checkFiniteVec3(pb.frame.axisU, `${scene.viewMode}.pickBoxes[${i}].frame.axisU`, errors);
      checkFiniteVec3(pb.frame.axisN, `${scene.viewMode}.pickBoxes[${i}].frame.axisN`, errors);
      checkFiniteVec3(pb.frame.axisV, `${scene.viewMode}.pickBoxes[${i}].frame.axisV`, errors);
    }
  }
  for (let i = 0; i < scene.callouts.length; i += 1) {
    const co = scene.callouts[i]!;
    if (co.worldPos !== undefined) {
      checkFiniteVec3(co.worldPos, `${scene.viewMode}.callouts[${i}].worldPos`, errors);
    }
  }
  for (let i = 0; i < scene.overlayAnchors.length; i += 1) {
    checkFiniteVec3(scene.overlayAnchors[i]!.worldPos, `${scene.viewMode}.overlayAnchors[${i}]`, errors);
  }
  for (let i = 0; i < scene.landmarks.length; i += 1) {
    checkFiniteVec3(scene.landmarks[i]!.world, `${scene.viewMode}.landmarks[${i}]`, errors);
  }
  for (let i = 0; i < scene.calipers.length; i += 1) {
    const c = scene.calipers[i]!;
    checkFiniteVec3(c.aWorld, `${scene.viewMode}.calipers[${i}].aWorld`, errors);
    checkFiniteVec3(c.bWorld, `${scene.viewMode}.calipers[${i}].bWorld`, errors);
  }
  for (let i = 0; i < scene.kvDepthTicks.length; i += 1) {
    checkFiniteVec3(scene.kvDepthTicks[i]!.world, `${scene.viewMode}.kvDepthTicks[${i}]`, errors);
  }
  for (let i = 0; i < scene.activeCellGrids.length; i += 1) {
    const grid = scene.activeCellGrids[i]!;
    const gridLabel = `${scene.viewMode}.activeCellGrids[${i}]`;
    checkFiniteVec3(grid.originWorld, `${gridLabel}.originWorld`, errors);
    checkFiniteVec3(grid.extent, `${gridLabel}.extent`, errors);
    checkFiniteVec3(grid.axisU, `${gridLabel}.axisU`, errors);
    checkFiniteVec3(grid.axisN, `${gridLabel}.axisN`, errors);
    if (!Number.isInteger(grid.cols) || grid.cols <= 0) {
      errors.push(`Invalid cols (${String(grid.cols)}) in ${gridLabel}`);
    }
    if (!Number.isInteger(grid.rows) || grid.rows <= 0) {
      errors.push(`Invalid rows (${String(grid.rows)}) in ${gridLabel}`);
    }
    if (grid.activeCol !== undefined && !Number.isFinite(grid.activeCol)) {
      errors.push(`Non-finite activeCol in ${gridLabel}`);
    }
    if (grid.activeRow !== undefined && !Number.isFinite(grid.activeRow)) {
      errors.push(`Non-finite activeRow in ${gridLabel}`);
    }
    if (grid.revealMode !== undefined && !Number.isFinite(grid.revealMode)) {
      errors.push(`Non-finite revealMode in ${gridLabel}`);
    }
    if (grid.revealColFrontier !== undefined && !Number.isFinite(grid.revealColFrontier)) {
      errors.push(`Non-finite revealColFrontier in ${gridLabel}`);
    }
    if (grid.revealRowFrontier !== undefined && !Number.isFinite(grid.revealRowFrontier)) {
      errors.push(`Non-finite revealRowFrontier in ${gridLabel}`);
    }
    if (
      grid.activeValueOverride !== undefined &&
      grid.activeValueOverride !== null &&
      !Number.isFinite(grid.activeValueOverride)
    ) {
      errors.push(`Non-finite activeValueOverride in ${gridLabel}`);
    }
    for (let j = 0; j < grid.values.length; j += 1) {
      const v = grid.values[j]!;
      if (!Number.isFinite(v)) {
        const isExplicitMaskedNegInf =
          v === -Infinity &&
          grid.masked !== undefined &&
          j < grid.masked.length &&
          grid.masked[j]! > 0.5;
        if (!isExplicitMaskedNegInf) {
          errors.push(`Non-finite value ${String(v)} in ${gridLabel}.values[${j}]`);
          break;
        }
      }
    }
  }
  if (scene.activeCellPopover !== null) {
    checkFiniteVec3(
      scene.activeCellPopover.world,
      `${scene.viewMode}.activeCellPopover.world`,
      errors,
    );
    if (scene.activeCellPopover.sourceWorlds !== undefined) {
      for (let i = 0; i < scene.activeCellPopover.sourceWorlds.length; i += 1) {
        checkFiniteVec3(
          scene.activeCellPopover.sourceWorlds[i]!.world,
          `${scene.viewMode}.activeCellPopover.sourceWorlds[${i}].world`,
          errors,
        );
      }
    }
  }

  return Object.freeze({
    valid: errors.length === 0,
    errors: Object.freeze(errors),
  });
}
