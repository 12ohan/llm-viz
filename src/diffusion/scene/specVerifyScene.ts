/**
 * Speculative Verification & Coupled 0..31 KV Cache Lifecycle Scene
 * (`src/diffusion/scene/specVerifyScene.ts`)
 *
 * Driven strictly by real `DiffusionSession` execution tensors:
 *   - If `ctx.record.kind === 'spec_cycle' && ctx.record.specMeta`: inspects the live
 *     `spec_cycle` record directly, including lazy cropped suffix KVs via
 *     `ctx.resolveSpecCroppedSuffixSlabs()` and real LoRA ON vs OFF draft logit deltas.
 *   - Otherwise, if `ctx.kvSnapshot.kvMaskProvenance === 'token_causal'` and
 *     `ctx.kvSnapshot.seqLen + blockLength <= capacity`: executes pure, non-mutating
 *     live preview `runLinearSpecCycle` (both LoRA OFF and LoRA ON) on `ctx.kvSnapshot`
 *     badged `"Live Speculative Preview on Current Causal KV"`.
 *   - Otherwise (bidirectional KV or capacity boundary): displays
 *     `"Bidir KV / Capacity Boundary — Switch to Causal Mode or Reset to Run Spec Verify"`
 *     coupled to the live 0..31 KV cache lifecycle strip.
 *
 * Visual layout:
 *   - Rail A (Draft Block): `B` token tiles with `p = cacheLen + i` headers (`slot 0 = seedToken`,
 *     `slot 1..B-1 = drafted proposals`).
 *   - Rail B (Causal Verify): `B` verify prediction tiles (`row i` at `p + i`, labeled
 *     `predicts p + i + 1 -> arTokens[i]`).
 *   - Diagonal / indexed `FlowRibbonBuilder` connectors linking Verify row `i` (`p+i+1`)
 *     to Draft slot `i+1` (`p+i+1`): emerald for accepted matches, oxide-red scissor at first
 *     mismatch, warm gold badge/ribbon for correction / bonus token `arTokens[matchedDraftCount]`.
 *   - Real LoRA ON vs OFF `[B × 36]` draft logit delta sheet packed in `64×64` `RG32F` atlas
 *     alongside causal verify mask (`D24` live `G ∈ {0, 1}`), verify probs, and 32-slot KV norms.
 *   - Coupled `0..31` KV Cache Lifecycle Strip showing Persistent Prefix KV, Transient Draft KV
 *     (`use_cache=False`), Accepted Verified KV, Cropped/Rejected Suffix KV (real norms), and
 *     Empty Capacity, plus `D19` dual-crop provenance & residual/trailing `[MASK]` badges.
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import { fmtSigned, type LayerFormulaReadout, type StepperStepDef } from '../../math/formulaFormat';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_GHOST_RAIL,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_NORM,
  SHEET_ROLE_SCORES,
} from '../../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
} from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  drawCellHighlight3D,
  spanFromSheetGrid,
  type SheetGeomRecord,
} from '../../scene/sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from '../../scene/spine';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../ui/overlay';
import { runDiffusionForwardPass } from '../forward';
import { cropKvSnapshot, materializePastKvs } from '../kvCache';
import { createXoshiro128ss, mixSeed32 } from '../prng';
import { runLinearSpecCycle } from '../speculative';
import type {
  CroppedKvProvenance,
  KvPositionSlab,
  LinearSpecCycleResult,
  PersistentKvSnapshot,
} from '../types';
import {
  auditAtlasRegions,
  DIFFUSION_EPISTEMIC_NOTE,
  LineBufferBuilder,
  RibbonBufferBuilder,
  SheetBufferBuilder,
  SolidBufferBuilder,
} from './provenance';
import type {
  DiffusionAtlasRegionBox,
  DiffusionFocusState,
  DiffusionPickIdentity,
  DiffusionProvenanceBadge,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
} from './types';

export const DIFFUSION_SPEC_ATLAS_SIZE = 64 as const;

export const DIFFUSION_SPEC_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  Object.freeze({
    index: 0,
    shortLabel: 'Draft',
    title: 'Bidirectional Transient Draft (use_cache=False)',
    summary: 'Fill [seed, MASK..MASK] via Sampler B with optional o_proj LoRA; prefix KV stays 0-ULP immutable.',
  }),
  Object.freeze({
    index: 1,
    shortLabel: 'Verify',
    title: 'Causal Verify Pass (LoRA OFF, use_cache=True)',
    summary: 'Run 1 causal pass over drafted block; row i at position p+i predicts p+i+1 and appends B KVs.',
  }),
  Object.freeze({
    index: 2,
    shortLabel: 'Match / Cut',
    title: 'Shifted Equality Check (D11) & Scissor Cut',
    summary: 'Accept while verifyArTokens[i] === draft[i+1]; cut at first mismatch and emit arTokens[m] correction/bonus.',
  }),
  Object.freeze({
    index: 3,
    shortLabel: 'KV Crop',
    title: 'Rollback Crop & D19 Dual-Crop Provenance',
    summary: 'Crop rejected suffix KVs cleanly (post-EOS in browser vs upstream pre-EOS D19) while keeping prefix slabs shared.',
  }),
]);

const COLOR_ACCEPT_EMERALD: FlowRibbonColor = {
  r: 0.086,
  g: 0.620,
  b: 0.412,
  a: 0.90,
  borderAlpha: 0.44,
};

const COLOR_SCISSOR_RED: FlowRibbonColor = {
  r: 0.863,
  g: 0.298,
  b: 0.204,
  a: 0.96,
  borderAlpha: 0.48,
};

const COLOR_BONUS_GOLD: FlowRibbonColor = {
  r: 0.812,
  g: 0.569,
  b: 0.145,
  a: 0.92,
  borderAlpha: 0.44,
};

const COLOR_DISCARD_MUTED: FlowRibbonColor = {
  r: 0.62,
  g: 0.67,
  b: 0.74,
  a: 0.34,
  borderAlpha: 0.18,
};

const EMPTY_LAYER_SET: ReadonlySet<number> = Object.freeze(new Set<number>());

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

function l2Norm(arr: Float32Array): number {
  let acc = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    acc += v * v;
  }
  return Math.sqrt(acc);
}

function tokenText(vocab: readonly string[], id: number): string {
  return vocab[id] ?? `#${id}`;
}

interface ResolvedSpecData {
  readonly isLiveSpecRecord: boolean;
  readonly hasActiveCycle: boolean;
  readonly statusBanner: string;
  readonly blockLength: number;
  readonly cacheLenBeforeVerify: number;
  readonly draftedBlockTokens: Int32Array;
  readonly verifyArTokens: Int32Array;
  readonly matchedDraftCount: number;
  readonly acceptedCount: number;
  readonly loraEnabled: boolean;
  readonly draftPassCount: number;
  readonly cropProvenance: CroppedKvProvenance | null;
  readonly croppedSuffixSlabs: readonly KvPositionSlab[];
  readonly postCropKvSnapshot: PersistentKvSnapshot;
  readonly draftLogitsOff: Float32Array | null; // [B × 36]
  readonly draftLogitsOn: Float32Array | null; // [B × 36]
  readonly draftLogitsDelta: Float32Array | null; // [B × 36]
  readonly verifyProbs: Float32Array | null; // [B × 36]
  readonly loraOffPreview: LinearSpecCycleResult | null;
  readonly loraOnPreview: LinearSpecCycleResult | null;
}

interface SpecDataCacheEntry {
  readonly key: string;
  readonly data: ResolvedSpecData;
}

/**
 * Lightweight 1-entry memoization cache for `resolveSpecData(ctx)`.
 * Prevents pointer hover / cell selection rebuilds in `spec_verify` from re-running
 * 2–6 full 40-layer forward passes while replacing the single slot on record/config change
 * (zero unbounded memory growth).
 */
let lastResolvedSpecData: SpecDataCacheEntry | null = null;

function buildSpecDataCacheKey(ctx: DiffusionResolvedSessionContext): string {
  const { model, record, kvSnapshot, parentRecord } = ctx;
  const cfg = record.configAtStep;
  return [
    model.combinedFingerprint,
    record.recordId,
    kvSnapshot.snapshotDigest,
    cfg.mode,
    cfg.blockLength,
    cfg.temperature,
    cfg.samplerBThreshold,
    cfg.samplingSeed,
    record.pendingSeedToken ?? 'none',
    parentRecord?.kvSnapshot.snapshotDigest ?? 'root',
  ].join('|');
}

function resolveSpecData(ctx: DiffusionResolvedSessionContext): ResolvedSpecData {
  const cacheKey = buildSpecDataCacheKey(ctx);
  if (lastResolvedSpecData && lastResolvedSpecData.key === cacheKey) {
    return lastResolvedSpecData.data;
  }
  const computed = computeSpecDataUncached(ctx);
  lastResolvedSpecData = { key: cacheKey, data: computed };
  return computed;
}

function computeSpecDataUncached(ctx: DiffusionResolvedSessionContext): ResolvedSpecData {
  const { model, record, kvSnapshot } = ctx;
  const cfg = record.configAtStep;
  const V = model.config.vocabSize;

  // Case 1: Active record is a real `spec_cycle`
  if (record.kind === 'spec_cycle' && record.specMeta) {
    const meta = record.specMeta;
    const B = meta.draftedBlockTokens.length;
    const cropProv = record.cropProvenance;
    const cacheLenBeforeVerify = cropProv
      ? cropProv.cacheLenBeforeVerify
      : Math.max(0, kvSnapshot.seqLen - meta.acceptedCount);
    const croppedSuffixSlabs = ctx.resolveSpecCroppedSuffixSlabs();

    const prefixLen = Math.min(cacheLenBeforeVerify, kvSnapshot.seqLen);
    const prefixSnap =
      ctx.parentRecord && ctx.parentRecord.kvSnapshot.seqLen === prefixLen
        ? ctx.parentRecord.kvSnapshot
        : cropKvSnapshot(kvSnapshot, prefixLen);
    const pastKvs = materializePastKvs(
      prefixSnap,
      model.config.numLayers,
      model.config.kvProjOutDim,
    );

    const initBlock = new Int32Array(B);
    initBlock[0] = meta.draftedBlockTokens[0] ?? model.config.bosTokenId;
    for (let i = 1; i < B; i++) initBlock[i] = model.config.maskTokenId;

    const draftOff = runDiffusionForwardPass(model, {
      tokenIds: initBlock,
      maskMode: 'bidirectional',
      loraEnabled: false,
      useCache: false,
      pastKvs,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    const draftOn = runDiffusionForwardPass(model, {
      tokenIds: initBlock,
      maskMode: 'bidirectional',
      loraEnabled: true,
      useCache: false,
      pastKvs,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    const delta = new Float32Array(B * V);
    for (let i = 0; i < delta.length; i++) {
      delta[i] = Math.fround(draftOn.logits[i]! - draftOff.logits[i]!);
    }

    return {
      isLiveSpecRecord: true,
      hasActiveCycle: true,
      statusBanner: meta.loraEnabled
        ? 'Live Spec Cycle Record (o_proj LoRA ON for Draft · OFF for Verify)'
        : 'Live Spec Cycle Record (o_proj LoRA OFF)',
      blockLength: B,
      cacheLenBeforeVerify,
      draftedBlockTokens: meta.draftedBlockTokens,
      verifyArTokens: meta.verifyArTokens,
      matchedDraftCount: meta.matchedDraftCount,
      acceptedCount: meta.acceptedCount,
      loraEnabled: meta.loraEnabled,
      draftPassCount: meta.draftPassCount,
      cropProvenance: cropProv,
      croppedSuffixSlabs,
      postCropKvSnapshot: kvSnapshot,
      draftLogitsOff: draftOff.logits,
      draftLogitsOn: draftOn.logits,
      draftLogitsDelta: delta,
      verifyProbs: record.primaryPassCheckpoint.probs,
      loraOffPreview: null,
      loraOnPreview: null,
    };
  }

  // Case 2: Non-spec record with compatible causal KV & capacity -> pure non-mutating live preview (LoRA OFF & ON)
  const B = cfg.blockLength;
  const canPreview =
    kvSnapshot.kvMaskProvenance === 'token_causal' &&
    kvSnapshot.seqLen + B <= kvSnapshot.capacity;

  if (canPreview) {
    const seedToken =
      record.pendingSeedToken ??
      (kvSnapshot.seqLen > 0
        ? kvSnapshot.committedTokenIds[kvSnapshot.seqLen - 1]!
        : model.config.bosTokenId);

    const previewSeed = mixSeed32(cfg.samplingSeed, record.recordIndex, kvSnapshot.seqLen);
    const prngOff = createXoshiro128ss(previewSeed);
    const prngOn = prngOff.clone();

    const samplerBConfig = {
      temperature: cfg.temperature,
      threshold: cfg.samplerBThreshold,
      maxDraftPassesPerCycle: cfg.maxDraftPassesPerCycle,
      maskTokenId: model.config.maskTokenId,
    };

    const previewOff = runLinearSpecCycle(model, {
      kvSnapshot,
      seedToken,
      blockLength: B,
      loraEnabled: false,
      samplerBConfig,
      verifyTemperature: cfg.temperature,
      prng: prngOff,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    const previewOn = runLinearSpecCycle(model, {
      kvSnapshot,
      seedToken,
      blockLength: B,
      loraEnabled: true,
      samplerBConfig,
      verifyTemperature: cfg.temperature,
      prng: prngOn,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });

    const chosen = cfg.mode === 'linear_spec_lora' ? previewOn : previewOff;

    const pastKvs = materializePastKvs(
      kvSnapshot,
      model.config.numLayers,
      model.config.kvProjOutDim,
    );
    const initBlock = new Int32Array(B);
    initBlock[0] = seedToken;
    for (let i = 1; i < B; i++) initBlock[i] = model.config.maskTokenId;

    const draftOff = runDiffusionForwardPass(model, {
      tokenIds: initBlock,
      maskMode: 'bidirectional',
      loraEnabled: false,
      useCache: false,
      pastKvs,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    const draftOn = runDiffusionForwardPass(model, {
      tokenIds: initBlock,
      maskMode: 'bidirectional',
      loraEnabled: true,
      useCache: false,
      pastKvs,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    const delta = new Float32Array(B * V);
    for (let i = 0; i < delta.length; i++) {
      delta[i] = Math.fround(draftOn.logits[i]! - draftOff.logits[i]!);
    }

    return {
      isLiveSpecRecord: false,
      hasActiveCycle: true,
      statusBanner: 'Live Speculative Preview on Current Causal KV',
      blockLength: B,
      cacheLenBeforeVerify: kvSnapshot.seqLen,
      draftedBlockTokens: chosen.draftedBlockTokens,
      verifyArTokens: chosen.verifyArTokens,
      matchedDraftCount: chosen.matchedDraftCount,
      acceptedCount: chosen.acceptedCount,
      loraEnabled: chosen.loraEnabled,
      draftPassCount: chosen.draftPassCount,
      cropProvenance: chosen.cropProvenance,
      croppedSuffixSlabs: chosen.croppedSuffixSlabs,
      postCropKvSnapshot: chosen.postCropKvSnapshot,
      draftLogitsOff: draftOff.logits,
      draftLogitsOn: draftOn.logits,
      draftLogitsDelta: delta,
      verifyProbs: chosen.verifyForward?.probs ?? null,
      loraOffPreview: previewOff,
      loraOnPreview: previewOn,
    };
  }

  // Case 3: Bidir KV or Capacity Boundary -> show KV lifecycle strip & clear boundary banner
  return {
    isLiveSpecRecord: false,
    hasActiveCycle: false,
    statusBanner:
      'Bidir KV / Capacity Boundary — Switch to Causal Mode or Reset to Run Spec Verify',
    blockLength: B,
    cacheLenBeforeVerify: kvSnapshot.seqLen,
    draftedBlockTokens: new Int32Array(0),
    verifyArTokens: new Int32Array(0),
    matchedDraftCount: 0,
    acceptedCount: 0,
    loraEnabled: false,
    draftPassCount: 0,
    cropProvenance: null,
    croppedSuffixSlabs: Object.freeze([]),
    postCropKvSnapshot: kvSnapshot,
    draftLogitsOff: null,
    draftLogitsOn: null,
    draftLogitsDelta: null,
    verifyProbs: null,
    loraOffPreview: null,
    loraOnPreview: null,
  };
}

export function buildDiffusionSpecVerifyScene(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const { model, record, kvSnapshot } = ctx;
  const V = model.config.vocabSize; // 36
  const capacity = kvSnapshot.capacity; // 32
  const spec = resolveSpecData(ctx);

  const size = DIFFUSION_SPEC_ATLAS_SIZE;
  const writer = createRg32fAtlas(size, size);
  const regions: DiffusionAtlasRegionBox[] = [];
  const slices: Record<string, AtlasSlice> = {};

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const sheets = new SheetBufferBuilder();
  const flow = new FlowRibbonBuilder();

  const solidPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const partCatalog: Record<string, CutawayPartInfo> = {};

  const badges: DiffusionProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    {
      label: spec.statusBanner,
      tone: spec.isLiveSpecRecord
        ? 'emerald'
        : spec.hasActiveCycle
          ? 'gold'
          : 'warn',
    },
  ];
  if (spec.cropProvenance && spec.cropProvenance.firstEosOffset !== null) {
    badges.push({
      label: `D19 Dual Crop: browser=${spec.cropProvenance.browserPostEosCropLen} · upstream=${spec.cropProvenance.upstreamPreEosCropLen}`,
      tone: 'gold',
    });
  }

  // Pack 32-slot KV L2 norms into atlas `(x=0, y=36, cols=32, rows=2)` (row 0 = K, row 1 = V)
  const kvNorms = new Float32Array(capacity * 2);
  const croppedByPos = new Map<number, KvPositionSlab>();
  for (const slab of spec.croppedSuffixSlabs) {
    croppedByPos.set(slab.position, slab);
  }
  for (let pos = 0; pos < capacity; pos++) {
    const slab = spec.postCropKvSnapshot.slabs[pos] ?? croppedByPos.get(pos) ?? null;
    if (slab) {
      kvNorms[pos] = l2Norm(slab.k);
      kvNorms[capacity + pos] = l2Norm(slab.v);
    }
  }
  const kvNormMax = computeMaxAbs(kvNorms, 0.25);
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < capacity; c++) {
      const val = kvNorms[r * capacity + c]!;
      writer.setTexel(c, 36 + r, val, false);
    }
  }
  regions.push({
    id: 'kv_slot_norms',
    x0: 0,
    y0: 36,
    cols: capacity,
    rows: 2,
    mode: 1,
    isMaskOrScore: false,
  });
  slices.kv_slot_norms = {
    rect: writer.makeRect(0, 36, capacity, 2),
    scale: 1.0 / kvNormMax,
    mode: 1,
  };

  // Pack LoRA ON vs OFF draft logit delta + causal verify mask if cycle active
  let loraDeltaMaxAbs = 0;
  let deltaSheetFrame: SheetFrame3D | null = null;
  let deltaSheetGeom: SheetGeomRecord | null = null;
  let shiftedConnectorsCount = 0;

  // Drafting table outer guide
  lines.addBoxFrameXZ(-8.8, -5.6, 8.8, 5.8, -0.05, 1.0, LINE_STYLE_GUIDE, 0.34);

  if (spec.hasActiveCycle && spec.draftedBlockTokens.length > 0) {
    const B = spec.blockLength;

    // 1. Pack LoRA delta [B × 36] at (0, 0)
    if (spec.draftLogitsDelta) {
      loraDeltaMaxAbs = computeMaxAbs(spec.draftLogitsDelta, 1e-4);
      const scale = 1.0 / Math.max(0.05, loraDeltaMaxAbs);
      for (let r = 0; r < B; r++) {
        for (let c = 0; c < V; c++) {
          writer.setTexel(c, r, spec.draftLogitsDelta[r * V + c]!, false);
        }
      }
      regions.push({
        id: 'draft_lora_delta',
        x0: 0,
        y0: 0,
        cols: V,
        rows: B,
        mode: 0,
        isMaskOrScore: false,
      });
      slices.draft_lora_delta = {
        rect: writer.makeRect(0, 0, V, B),
        scale,
        mode: 0,
      };
    }

    // 2. Pack verify probs [B × 36] at (0, 24)
    if (spec.verifyProbs) {
      for (let r = 0; r < B; r++) {
        for (let c = 0; c < V; c++) {
          writer.setTexel(c, 24 + r, spec.verifyProbs[r * V + c]!, false);
        }
      }
      regions.push({
        id: 'verify_probs',
        x0: 0,
        y0: 24,
        cols: V,
        rows: B,
        mode: 1,
        isMaskOrScore: false,
      });
      slices.verify_probs = {
        rect: writer.makeRect(0, 24, V, B),
        scale: 1.0,
        mode: 1,
      };
    }

    // 3. Pack causal verify mask [B × min(32, cacheLen + B)] at (0, 42) with explicit G ∈ {0, 1} (D24)
    const verifyKvLen = Math.min(capacity, spec.cacheLenBeforeVerify + B);
    for (let r = 0; r < B; r++) {
      const maxK = spec.cacheLenBeforeVerify + r;
      for (let c = 0; c < verifyKvLen; c++) {
        const vis = c <= maxK;
        writer.setTexel(c, 42 + r, vis ? (c < spec.cacheLenBeforeVerify ? 0.56 : 0.96) : 0.0, !vis);
      }
    }
    regions.push({
      id: 'verify_causal_mask',
      x0: 0,
      y0: 42,
      cols: verifyKvLen,
      rows: B,
      mode: 0,
      isMaskOrScore: true,
    });
    slices.verify_causal_mask = {
      rect: writer.makeRect(0, 42, verifyKvLen, B),
      scale: 1.0,
      mode: 0,
    };

    // ------------------------------------------------------------------------
    // RAIL A (Draft Block, z = -3.2) & RAIL B (Causal Verify, z = -0.7)
    // ------------------------------------------------------------------------
    const tileW = 1.42;
    const tileGap = 0.28;
    const pitch = tileW + tileGap;
    const railStartX = -((B - 1) * pitch) * 0.5 - 1.6;
    const draftZ = -3.2;
    const verifyZ = -0.65;
    const tileH = 0.34;
    const tileD = 0.82;

    for (let i = 0; i < B; i++) {
      const x = railStartX + i * pitch;
      const absPos = spec.cacheLenBeforeVerify + i;
      const draftTok = spec.draftedBlockTokens[i]!;
      const verifyTok = spec.verifyArTokens[i]!;

      // Draft tile status: slot 0 = seed (blue), 1..matchedDraftCount = accepted (green),
      // matchedDraftCount+1 = first mismatch (highlight=1), beyond = discarded (neutral)
      const isSeed = i === 0;
      const isAcceptedDraft = i >= 1 && i <= spec.matchedDraftCount;
      const isFirstMismatch = i === spec.matchedDraftCount + 1;
      const draftFamily = isSeed
        ? CELL_WEIGHT_BLUE
        : isAcceptedDraft
          ? CELL_ACTIVATION_GREEN
          : isFirstMismatch
            ? CELL_MOE_AMBER
            : CELL_NEUTRAL;

      solids.push({
        cx: x,
        cy: tileH * 0.5,
        cz: draftZ,
        ex: tileW,
        ey: tileH,
        ez: tileD,
        cellFamily: draftFamily,
        highlight: isFirstMismatch ? 1 : 0,
      });
      lines.addBoxEdges(
        x,
        tileH * 0.5,
        draftZ,
        tileW,
        tileH,
        tileD,
        100 + i,
        1.1,
        isFirstMismatch ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        0.78,
      );

      solidPickBoxes.push({
        bayId: 0,
        partId: `draft_slot_${i}`,
        tensorKey: 'draft_rail',
        tokenIndex: i,
        absoluteTokenOffset: spec.cacheLenBeforeVerify,
        semanticTag: isSeed
          ? 'seed_token'
          : isAcceptedDraft
            ? 'accepted_draft'
            : isFirstMismatch
              ? 'first_mismatch'
              : 'rejected_draft',
        min: [x - tileW * 0.5, 0, draftZ - tileD * 0.5],
        max: [x + tileW * 0.5, tileH, draftZ + tileD * 0.5],
      });

      landmarks.push({
        id: `spec-draft-${i}`,
        label: `p=${absPos}: ${tokenText(model.vocab, draftTok)}`,
        sublabel: isSeed
          ? 'slot 0 (seed)'
          : isAcceptedDraft
            ? `draft[${i}] ✓ match`
            : isFirstMismatch
              ? `draft[${i}] ✕ cut`
              : `draft[${i}] dropped`,
        world: [x, tileH + 0.14, draftZ - tileD * 0.5 - 0.14],
        partId: `draft_slot_${i}`,
        tone: isAcceptedDraft ? 'cyan' : isFirstMismatch ? 'gold' : 'default',
        tier: 'primary',
      });

      // Verify tile at row i (at position p+i, predicting p+i+1 -> arTokens[i])
      const isCorrectionRow = i === spec.matchedDraftCount;
      const isMatchRow = i < spec.matchedDraftCount;
      solids.push({
        cx: x,
        cy: tileH * 0.5,
        cz: verifyZ,
        ex: tileW,
        ey: tileH,
        ez: tileD,
        cellFamily: isCorrectionRow
          ? CELL_MOE_AMBER
          : isMatchRow
            ? CELL_ACTIVATION_GREEN
            : CELL_EMBED_SLATE,
        highlight: isCorrectionRow ? 1 : 0,
      });
      lines.addBoxEdges(
        x,
        tileH * 0.5,
        verifyZ,
        tileW,
        tileH,
        tileD,
        200 + i,
        1.1,
        isCorrectionRow ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
        0.78,
      );

      solidPickBoxes.push({
        bayId: 0,
        partId: `verify_row_${i}`,
        tensorKey: 'verify_rail',
        tokenIndex: i,
        absoluteTokenOffset: spec.cacheLenBeforeVerify,
        semanticTag: isCorrectionRow
          ? 'correction_or_bonus'
          : isMatchRow
            ? 'verify_match'
            : 'verify_unused',
        min: [x - tileW * 0.5, 0, verifyZ - tileD * 0.5],
        max: [x + tileW * 0.5, tileH, verifyZ + tileD * 0.5],
      });

      landmarks.push({
        id: `spec-verify-${i}`,
        label: `row ${i} (p=${absPos}) → p=${absPos + 1}: ${tokenText(model.vocab, verifyTok)}`,
        sublabel: isCorrectionRow
          ? spec.matchedDraftCount === B - 1
            ? 'Bonus token (all matched!)'
            : 'Gold correction token'
          : isMatchRow
            ? `matches draft[${i + 1}]`
            : 'post-cut (unused)',
        world: [x, tileH + 0.14, verifyZ + tileD * 0.5 + 0.16],
        partId: `verify_row_${i}`,
        tone: isCorrectionRow ? 'gold' : isMatchRow ? 'cyan' : 'muted',
        tier: 'primary',
      });

      // Diagonal shifted alignment connectors: Verify row i (predicting p+i+1) -> Draft slot i+1 (at p+i+1)
      if (i < B - 1) {
        const nextDraftX = railStartX + (i + 1) * pitch;
        const connY = tileH * 0.5 + 0.08;
        const startPt = { x, z: verifyZ - tileD * 0.5 - 0.05 };
        const endPt = { x: nextDraftX, z: draftZ + tileD * 0.5 + 0.05 };
        const midZ = (startPt.z + endPt.z) * 0.5;

        const connColor =
          i < spec.matchedDraftCount
            ? COLOR_ACCEPT_EMERALD
            : i === spec.matchedDraftCount
              ? COLOR_SCISSOR_RED
              : COLOR_DISCARD_MUTED;

        flow.addCubicXZ(
          startPt,
          { x: startPt.x, z: midZ + 0.25 },
          { x: endPt.x, z: midZ - 0.25 },
          endPt,
          connY,
          i <= spec.matchedDraftCount ? 0.14 : 0.08,
          connColor,
          true,
          20,
        );
        shiftedConnectorsCount++;

        // Explicit oxide-red scissor X marker at first mismatch (i === matchedDraftCount < B - 1)
        if (i === spec.matchedDraftCount) {
          const sx = (startPt.x + endPt.x) * 0.5;
          const sz = midZ;
          const r = 0.28;
          lines.addSegment(sx - r, connY + 0.04, sz - r, sx + r, connY + 0.04, sz + r, 2.4, LINE_STYLE_ACTIVE, 0.96);
          lines.addSegment(sx - r, connY + 0.04, sz + r, sx + r, connY + 0.04, sz - r, 2.4, LINE_STYLE_ACTIVE, 0.96);
          landmarks.push({
            id: 'spec-scissor-cut',
            label: `✕ First Mismatch Cut (p=${absPos + 1})`,
            sublabel: `ar[${i}]=${tokenText(model.vocab, verifyTok)} ≠ draft[${i + 1}]=${tokenText(model.vocab, spec.draftedBlockTokens[i + 1]!)}`,
            world: [sx, connY + 0.22, sz],
            tone: 'gold',
            tier: 'primary',
          });
        }
      }
    }

    // Warm gold correction / bonus token badge tile to the right of Rail B
    const corrRow = spec.matchedDraftCount;
    const corrTok = spec.verifyArTokens[corrRow]!;
    const corrX = railStartX + B * pitch + 0.55;
    const corrZ = verifyZ;
    solids.push({
      cx: corrX,
      cy: tileH * 0.5,
      cz: corrZ,
      ex: 1.55,
      ey: tileH * 1.15,
      ez: tileD,
      cellFamily: CELL_MOE_AMBER,
      highlight: 1,
    });
    lines.addBoxEdges(corrX, tileH * 0.5, corrZ, 1.55, tileH * 1.15, tileD, 299, 1.4, LINE_STYLE_BF16_GOLD, 0.92);
    flow.addCubicXZ(
      { x: railStartX + corrRow * pitch, z: verifyZ + tileD * 0.5 + 0.06 },
      { x: railStartX + corrRow * pitch + 0.4, z: verifyZ + 1.05 },
      { x: corrX - 0.3, z: verifyZ + 1.05 },
      { x: corrX, z: corrZ + tileD * 0.5 + 0.06 },
      tileH * 0.5 + 0.08,
      0.14,
      COLOR_BONUS_GOLD,
      true,
      20,
    );
    landmarks.push({
      id: 'spec-gold-emitted',
      label:
        spec.matchedDraftCount === B - 1
          ? `★ Bonus Token: ${tokenText(model.vocab, corrTok)}`
          : `★ Correction Token: ${tokenText(model.vocab, corrTok)}`,
      sublabel: `arTokens[${corrRow}] · accepted=${spec.acceptedCount} tok(s)`,
      world: [corrX, tileH + 0.24, corrZ - tileD * 0.5 - 0.14],
      partId: 'spec_correction_bonus',
      tone: 'gold',
      tier: 'primary',
    });
    solidPickBoxes.push({
      bayId: 0,
      partId: 'spec_correction_bonus',
      tensorKey: 'spec_correction_bonus',
      tokenIndex: corrRow,
      min: [corrX - 0.78, 0, corrZ - tileD * 0.5],
      max: [corrX + 0.78, tileH * 1.15, corrZ + tileD * 0.5],
    });

    // 4. LoRA ON vs OFF Draft Logit Diff Sheet [B × 36] & Causal Verify Mask Sheet [B × kvLen]
    const deltaEx = 6.4;
    const deltaEz = Math.max(0.75, B * 0.26);
    const deltaCx = -2.3;
    const deltaCz = 1.65;
    if (slices.draft_lora_delta && spec.draftLogitsDelta) {
      deltaSheetFrame = makeSheetFrame3D(
        [deltaCx, 0.0, deltaCz],
        [deltaEx, 0.068, deltaEz],
        FRAME_XZ_AXES.axisU,
        FRAME_XZ_AXES.axisN,
      );
      deltaSheetGeom = {
        cx: deltaCx,
        cy: 0.0,
        cz: deltaCz,
        ex: deltaEx,
        ey: 0.068,
        ez: deltaEz,
        gridCols: V,
        gridRows: B,
        frame: deltaSheetFrame,
      };
      sheets.push({
        cx: deltaCx,
        cy: 0.0,
        cz: deltaCz,
        ex: deltaEx,
        ey: 0.068,
        ez: deltaEz,
        gridCols: V,
        gridRows: B,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_ACTIVATION,
        highlight: spec.loraEnabled ? SHEET_HL_ACTIVE : SHEET_HL_OPERAND_A,
        slice: slices.draft_lora_delta,
        frame: deltaSheetFrame,
      });
      const dAabb = sheetFrameWorldAabb(deltaSheetFrame);
      solidPickBoxes.push({
        bayId: 0,
        partId: 'draft_lora_delta',
        tensorKey: 'draft_lora_delta',
        gridCols: V,
        gridRows: B,
        min: dAabb.min,
        max: dAabb.max,
        frame: deltaSheetFrame,
      });
      landmarks.push({
        id: 'spec-lm-lora-delta',
        label: `LoRA ON − OFF Draft Logit Δ [${B}×${V}]`,
        sublabel: `max|Δ| = ${loraDeltaMaxAbs.toFixed(4)} · verify LoRA ALWAYS OFF`,
        world: [deltaCx, 0.14, deltaCz - deltaEz * 0.5 - 0.18],
        partId: 'draft_lora_delta',
        tone: 'cyan',
        tier: 'primary',
      });

      activeCellGrids.push({
        originWorld: [deltaCx, 0.0, deltaCz],
        extent: [deltaEx, 0.068, deltaEz],
        axisU: deltaSheetFrame.axisU,
        axisN: deltaSheetFrame.axisN,
        cols: V,
        rows: B,
        values: spec.draftLogitsDelta,
        format: 'signed',
        colLabels: model.vocab,
        rowLabels: Array.from({ length: B }, (_, i) => `p${spec.cacheLenBeforeVerify + i}`),
        colDimLabel: `V=${V}`,
        rowDimLabel: `B=${B}`,
      });
    }

    // Causal Verify Mask Sheet (D24 live RG32F)
    const maskEx = Math.min(5.2, verifyKvLen * 0.26);
    const maskEz = Math.max(0.75, B * 0.26);
    const maskCx = 4.4;
    const maskCz = 1.65;
    const maskFrame = makeSheetFrame3D(
      [maskCx, 0.0, maskCz],
      [maskEx, 0.068, maskEz],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    sheets.push({
      cx: maskCx,
      cy: 0.0,
      cz: maskCz,
      ex: maskEx,
      ey: 0.068,
      ez: maskEz,
      gridCols: verifyKvLen,
      gridRows: B,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      highlight: SHEET_HL_PATHWAY,
      slice: slices.verify_causal_mask!,
      frame: maskFrame,
    });
    const mAabb = sheetFrameWorldAabb(maskFrame);
    solidPickBoxes.push({
      bayId: 0,
      partId: 'verify_causal_mask',
      tensorKey: 'verify_causal_mask',
      gridCols: verifyKvLen,
      gridRows: B,
      min: mAabb.min,
      max: mAabb.max,
      frame: maskFrame,
    });
    landmarks.push({
      id: 'spec-lm-verify-mask',
      label: `Causal Verify Mask [${B}×${verifyKvLen}]`,
      sublabel: 'D24 Live RG32F (G=1 upper masked)',
      world: [maskCx, 0.14, maskCz - maskEz * 0.5 - 0.18],
      partId: 'verify_causal_mask',
      tone: 'default',
      tier: 'detail',
    });
  } else {
    landmarks.push({
      id: 'spec-boundary-banner',
      label: spec.statusBanner,
      sublabel: `KV=${kvSnapshot.seqLen}/${capacity} · provenance=${kvSnapshot.kvMaskProvenance}`,
      world: [0, 0.4, -1.8],
      tone: 'gold',
      tier: 'primary',
    });
  }

  // --------------------------------------------------------------------------
  // COUPLED 0..31 KV CACHE LIFECYCLE STRIP (z = 4.0)
  // --------------------------------------------------------------------------
  const kvStripZ = 4.05;
  const slotPitch = 0.48;
  const slotW = 0.40;
  const slotD = 0.72;
  const kvStartX = -((capacity - 1) * slotPitch) * 0.5;
  const postCropLen = spec.postCropKvSnapshot.seqLen;
  const preCropLen = spec.cropProvenance?.preCropKvLen ?? postCropLen;

  for (let pos = 0; pos < capacity; pos++) {
    const x = kvStartX + pos * slotPitch;
    let stateTag:
      | 'persistent_prefix'
      | 'accepted_verified'
      | 'cropped_suffix'
      | 'empty_capacity' = 'empty_capacity';
    let cellFamily = CELL_NEUTRAL;
    let h = 0.12;
    let hl = 0;

    if (pos < spec.cacheLenBeforeVerify && pos < postCropLen) {
      stateTag = 'persistent_prefix';
      cellFamily =
        spec.postCropKvSnapshot.kvMaskProvenance === 'token_causal'
          ? CELL_WEIGHT_BLUE
          : CELL_SSM_CYAN;
      h = 0.34;
    } else if (pos < postCropLen) {
      stateTag = 'accepted_verified';
      cellFamily = CELL_ACTIVATION_GREEN;
      h = 0.42;
    } else if (pos < preCropLen && croppedByPos.has(pos)) {
      stateTag = 'cropped_suffix';
      cellFamily = CELL_MOE_AMBER;
      h = 0.24;
      hl = 1;
    }

    solids.push({
      cx: x,
      cy: h * 0.5,
      cz: kvStripZ,
      ex: slotW,
      ey: h,
      ez: slotD,
      cellFamily,
      highlight: hl,
    });

    if (pos % 4 === 0 || pos === capacity - 1) {
      kvDepthTicks.push({
        token: pos,
        world: [x, 0.08, kvStripZ + slotD * 0.5 + 0.22],
      });
    }

    solidPickBoxes.push({
      bayId: 0,
      partId: `kv_slot_${pos}`,
      tensorKey: 'kv_lifecycle_strip',
      tokenIndex: pos,
      semanticTag: stateTag,
      min: [x - slotW * 0.5, 0, kvStripZ - slotD * 0.5],
      max: [x + slotW * 0.5, h, kvStripZ + slotD * 0.5],
    });
  }

  // Live RG32F KV L2 norm heatmap sheet tucked neatly directly in front of the 32 solid slots
  const normStripW = capacity * slotPitch;
  const normStripFrame = makeSheetFrame3D(
    [0.0, 0.02, kvStripZ + 0.92],
    [normStripW, 0.056, 0.44],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  sheets.push({
    cx: 0.0,
    cy: 0.02,
    cz: kvStripZ + 0.92,
    ex: normStripW,
    ey: 0.056,
    ez: 0.44,
    gridCols: capacity,
    gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_NORMAL,
    slice: slices.kv_slot_norms!,
    frame: normStripFrame,
  });

  // Floating Transient Draft KV (use_cache=False) ghost rail aligned over active block positions
  if (spec.hasActiveCycle) {
    const B = spec.blockLength;
    const tStart = spec.cacheLenBeforeVerify;
    const tEnd = Math.min(capacity - 1, tStart + B - 1);
    const tCenterX = kvStartX + ((tStart + tEnd) * 0.5) * slotPitch;
    const tWidth = (tEnd - tStart + 1) * slotPitch;
    const ghostFrame = makeSheetFrame3D(
      [tCenterX, 0.68, kvStripZ],
      [tWidth, 0.05, slotD * 0.85],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    sheets.push({
      cx: tCenterX,
      cy: 0.68,
      cz: kvStripZ,
      ex: tWidth,
      ey: 0.05,
      ez: slotD * 0.85,
      gridCols: B,
      gridRows: 1,
      sheetKind: SHEET_KIND_GHOST_RAIL,
      role: SHEET_ROLE_NORM,
      highlight: SHEET_HL_OPERAND_B,
      frame: ghostFrame,
    });
    landmarks.push({
      id: 'spec-lm-transient-kv',
      label: `Transient Draft KV (use_cache=False · p=${tStart}..${tEnd})`,
      sublabel: 'Discarded immediately · 0 ULP prefix mutation',
      world: [tCenterX, 0.94, kvStripZ - 0.15],
      partId: 'kv_transient_draft',
      tone: 'cyan',
      tier: 'detail',
    });
  }

  landmarks.push({
    id: 'spec-lm-kv-strip',
    label: `0..31 KV Cache Lifecycle Strip (${postCropLen}/${capacity} live · ${spec.croppedSuffixSlabs.length} cropped)`,
    sublabel:
      spec.cropProvenance !== null
        ? `Provenance: ${spec.postCropKvSnapshot.kvMaskProvenance} · D19 browser=${spec.cropProvenance.browserPostEosCropLen} / upstream=${spec.cropProvenance.upstreamPreEosCropLen}`
        : `Provenance: ${kvSnapshot.kvMaskProvenance}`,
    world: [-6.8, 0.45, kvStripZ - 0.62],
    partId: 'kv_lifecycle_strip',
    align: 'left',
    tone: 'default',
    tier: 'primary',
  });

  calipers.push({
    id: 'spec-cal-kv-cap',
    aWorld: [kvStartX - slotW * 0.5, 0.12, kvStripZ + 1.32],
    bWorld: [kvStartX + (capacity - 1) * slotPitch + slotW * 0.5, 0.12, kvStripZ + 1.32],
    label: `kvCapacity = ${capacity} (20.00 KiB/tok across 40L · ${postCropLen * 20} KiB active)`,
    partIds: ['kv_lifecycle_strip'],
  });

  // Active cell / formula readout
  const activeCell: DiffusionPickIdentity | null =
    focus.pinnedCell && focus.pinnedCell.viewMode === 'spec_verify'
      ? focus.pinnedCell
      : focus.hoveredCell && focus.hoveredCell.viewMode === 'spec_verify'
        ? focus.hoveredCell
        : null;

  let activeFormula: LayerFormulaReadout | null = null;
  let cellPopover: CellPopoverSpec | null = null;

  if (
    activeCell &&
    activeCell.tensorKey === 'draft_lora_delta' &&
    deltaSheetFrame &&
    deltaSheetGeom &&
    spec.draftLogitsDelta &&
    spec.draftLogitsOn &&
    spec.draftLogitsOff
  ) {
    const r = clampInt(activeCell.row ?? 0, 0, spec.blockLength - 1);
    const c = clampInt(activeCell.col ?? 0, 0, V - 1);
    const idx = r * V + c;
    const dVal = spec.draftLogitsDelta[idx]!;
    const onVal = spec.draftLogitsOn[idx]!;
    const offVal = spec.draftLogitsOff[idx]!;
    const span = spanFromSheetGrid(deltaSheetGeom, r, r, c, c);
    drawCellHighlight3D(flow, span, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.022, true);

    const eq = `Δlogit[slot=${r}, tok=${tokenText(model.vocab, c)}] = LoRA_ON(${fmtSigned(onVal)}) − LoRA_OFF(${fmtSigned(offVal)}) = ${fmtSigned(dVal)}`;
    activeFormula = {
      tensorLabel: 'Draft LoRA ON − OFF Logit Δ',
      coordLabel: `slot=${r} (p=${spec.cacheLenBeforeVerify + r}), vocab=${c} (${tokenText(model.vocab, c)})`,
      headBadge: 'o_proj LoRA r=4, α=16',
      value: dVal,
      isMasked: false,
      valueText: `= ${fmtSigned(dVal)}`,
      equation: eq,
      subEquation: '4.0 · B(A · a) draft-only delta · Causal verifier ALWAYS runs with LoRA OFF (0 ULP invariance)',
    };
    cellPopover = {
      world: localToWorldOnFrame(
        deltaSheetFrame,
        -0.5 + (c + 0.5) / V,
        0.5 + 0.08 / Math.max(0.02, deltaSheetFrame.extent[1]),
        -0.5 + (r + 0.5) / spec.blockLength,
      ),
      tensorLabel: 'LoRA Draft Logit Δ',
      coordLabel: `slot=${r}, ${tokenText(model.vocab, c)}`,
      badge: 'scale = 4.0',
      equation: eq,
      valueText: `= ${fmtSigned(dVal)}`,
      operandColors: ['indigo', 'teal'],
    };
  } else if (spec.hasActiveCycle) {
    const corrIdx = spec.matchedDraftCount;
    const corrTok = spec.verifyArTokens[corrIdx] ?? 0;
    activeFormula = {
      tensorLabel: 'Linear Self-Speculation Verification',
      coordLabel: `cacheLen=${spec.cacheLenBeforeVerify} · B=${spec.blockLength} · matched=${spec.matchedDraftCount}/${spec.blockLength - 1}`,
      headBadge: spec.loraEnabled ? 'LoRA ON (Draft)' : 'LoRA OFF',
      value: spec.acceptedCount,
      isMasked: false,
      valueText: `accepted = ${spec.acceptedCount} tok(s)`,
      equation: `accepted = matched(${spec.matchedDraftCount}) + 1 = ${spec.acceptedCount} · emit ar[${corrIdx}] = ${tokenText(model.vocab, corrTok)}`,
      subEquation:
        spec.cropProvenance !== null
          ? `D19 crop: browser=${spec.cropProvenance.browserPostEosCropLen} · upstream=${spec.cropProvenance.upstreamPreEosCropLen} · croppedSuffix=${spec.croppedSuffixSlabs.length} slab(s)`
          : `Cropped suffix slabs: ${spec.croppedSuffixSlabs.length}`,
    };
  } else {
    activeFormula = {
      tensorLabel: 'KV Lifecycle Boundary',
      coordLabel: `${kvSnapshot.seqLen}/${capacity} slots · ${kvSnapshot.kvMaskProvenance}`,
      headBadge: 'Guard Active',
      value: kvSnapshot.seqLen,
      isMasked: false,
      valueText: `${kvSnapshot.seqLen}/${capacity}`,
      equation: spec.statusBanner,
      subEquation: 'Linear self-speculation requires token_causal KV provenance and seqLen + B <= 32.',
    };
  }

  partCatalog.kv_lifecycle_strip = {
    id: 'kv_lifecycle_strip',
    title: '0..31 Persistent KV Cache Lifecycle Strip',
    badge: `${postCropLen}/${capacity} live · ${spec.postCropKvSnapshot.kvMaskProvenance}`,
    summary:
      'Immutable per-position shared KV slabs (20.00 KiB/tok across 40 layers). Shows prefix, accepted verified, real cropped suffix norms, transient draft ghost, and empty capacity.',
    specs: [
      `Persistent prefix: 0..${Math.max(0, spec.cacheLenBeforeVerify - 1)} (${spec.cacheLenBeforeVerify} tok)`,
      `Accepted verified: ${spec.acceptedCount} tok · Cropped suffix: ${spec.croppedSuffixSlabs.length} slab(s)`,
      spec.cropProvenance
        ? `D19 Dual-Crop: browserPostEos=${spec.cropProvenance.browserPostEosCropLen} vs upstreamPreEos=${spec.cropProvenance.upstreamPreEosCropLen}`
        : 'No speculative crop on current record',
    ],
  };
  partCatalog.draft_lora_delta = {
    id: 'draft_lora_delta',
    title: 'Draft-Only o_proj LoRA Logit Delta (ON − OFF)',
    badge: `max|Δ| = ${loraDeltaMaxAbs.toFixed(4)}`,
    summary:
      'Exact logit shift produced by enabling r=4, α=16 (scale=4.0) o_proj LoRA during bidirectional drafting while keeping causal verify strictly LoRA OFF.',
    specs: [
      '40 layers × (lora_A [4,256] + lora_B [8,4]) = 42,240 mini floats (prod: 47.19M F32 params = +0.35%)',
      spec.loraOffPreview && spec.loraOnPreview
        ? `Live Preview Match: LoRA OFF=${spec.loraOffPreview.matchedDraftCount} · LoRA ON=${spec.loraOnPreview.matchedDraftCount}`
        : 'Real forward pass difference on current prefix KV snapshot',
    ],
  };

  const audit = auditAtlasRegions(size, size, regions);
  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0) {
    throw new Error(
      `Spec Verify 64×64 atlas layout error: overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  const solidPacked = solids.pack();
  const linePacked = lines.pack();
  const ribbonPacked = ribbons.pack();
  const sheetPacked = sheets.pack();
  const flowPacked = flow.pack();

  const fitBounds: AABB = {
    min: [-9.2, -0.4, -5.8],
    max: [9.2, 2.0, 6.2],
  };

  return {
    viewMode: 'spec_verify',
    title: 'Speculative Verification & 0..31 KV Lifecycle Strip',
    subtitle: `${spec.statusBanner} · matched=${spec.matchedDraftCount}, accepted=${spec.acceptedCount}, cropped=${spec.croppedSuffixSlabs.length}`,
    layerIndex: null,
    solidData: solidPacked.data,
    solidCount: solidPacked.count,
    lineData: linePacked.data,
    lineCount: linePacked.count,
    ribbonData: ribbonPacked.data,
    ribbonCount: ribbonPacked.count,
    sheetData: sheetPacked.sheetData,
    sheetCount: sheetPacked.sheetCount,
    sheetBasisData: sheetPacked.sheetBasisData,
    sheetRevealData: sheetPacked.sheetRevealData,
    flowVertices: flowPacked.vertices,
    flowIndices: flowPacked.indices,
    valueAtlas: {
      width: size,
      height: size,
      data: writer.data,
    },
    atlasKey: `spec_verify|rec=${record.recordId}|snap=${kvSnapshot.snapshotDigest}|mode=${record.configAtStep.mode}`,
    atlasAudit: audit,
    fitBounds,
    defaultCameraAzimuthDeg: 20,
    defaultCameraElevationDeg: 44,
    landmarks,
    kvDepthTicks,
    calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeCellGrids,
    cellPopover,
    activeFormula,
    stepperSteps: DIFFUSION_SPEC_STEPPER_STEPS,
    provenanceBadges: Object.freeze(badges),
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner ?? (!spec.hasActiveCycle ? spec.statusBanner : null),
    diagnostics: Object.freeze({
      shiftedConnectorsCount,
      matchedDraftCount: spec.matchedDraftCount,
      acceptedCount: spec.acceptedCount,
      croppedSuffixCount: spec.croppedSuffixSlabs.length,
      kvSeqLen: postCropLen,
      kvMaskProvenance: spec.postCropKvSnapshot.kvMaskProvenance,
      isLiveSpecRecord: spec.isLiveSpecRecord,
      hasPreviewCycle: !spec.isLiveSpecRecord && spec.hasActiveCycle,
      loraDeltaMaxAbs,
      browserPostEosCropLen: spec.cropProvenance?.browserPostEosCropLen ?? null,
      upstreamPreEosCropLen: spec.cropProvenance?.upstreamPreEosCropLen ?? null,
    }),
  };
}
