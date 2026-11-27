/**
 * DORY5 Slice 5C — Headless Readouts, View 7 Bookends, 7-View Dispatcher &
 * 4-Act Guided Walkthrough Test Suite (`tests/dory-readouts-tour.test.ts`)
 *
 * Enforces every `DORY5` Gate 5 exit invariant:
 *   1. `archReadouts.ts` — `0 B` diff 4-checkpoint production constants (`BF16 / MXFP8 /
 *      NVFP4 / Q5_K_M`, exact `8 + header + payload === totalFileBytes` & `219` unquantized
 *      BF16 vs. `288` quantized matrices split), exact `3,975,229,440` BF16 stage parameter
 *      partition (`507` tensors), exact `1 -> 256K` BF16 KV scaling with `S`-invariant
 *      `(4 + 5k) / 9` ratio (`maxAbsDeviation <= 1e-15`) & `4x` marginal slope drop across
 *      `S = 1,024`, `D10` memory bounds (`<= 7.29 MiB` `k=4`, `<= 9.60 MiB` `k=8`, `<= 12 MiB`),
 *      structured `DR1–DR24` filtering, and strictly `READ-ONLY · NO GRADIENTS` TBPTL depth-1
 *      & Paper Results readouts (`0` fake quant toggles, `0` gradients, `0` training loops).
 *   2. View 7 (`src/dory/scene/bookendsScene.ts`) — finite 5-pass geometry (`0` NaN/Inf, exact
 *      GPU buffer strides), compact `128 x 80` (`80.00 KiB <= 128 KiB <= 512 KiB`) live `RG32F`
 *      atlas with `0` overlaps & `0` OOB verified independently via `auditAtlasRegions`, untied
 *      unit-norm bookends (`max|E - W_lm| > 0`), strict 4-stage LM head separation (`z_raw ->
 *      FP32 logit_scale / 0.0177 -> z_scaled -> binary64 softmax`), strictly `READ-ONLY · NO
 *      GRADIENTS` TBPTL pedestal & popovers, real `DR21` 3D raycast picking (`pickMacroBay`),
 *      and zero caller `DorySession` mutation across `standard`, `last_loop`, and Contract A
 *      `self_spec` (`spec_prefill` & `spec_cycle`).
 *   3. Unified 7-View Dispatcher (`src/dory/scene/index.ts`) — exhaustive `DORY_SCENE_BUILDERS`,
 *      `getDorySpatialViewDescriptor`, `resolveDorySpatialFocus`, `focusStateToViewOptions`,
 *      and `buildDorySpatialScene` across ALL 7 `D7` views (`macro`, `hypersphere`, `gqa_rope`,
 *      `swiglu`, `kv_compare`, `self_spec`, `bookends`) with live `DR20` `G`-channel mask audit
 *      and bit-exact zero session mutation.
 *   4. 4-Act Guided Walkthrough (`src/dory/story/walkthrough.ts`) — exact 4-Act, 14-beat
 *      (`3 + 3 + 4 + 4`) catalog covering all 7 views & all 5 isolated tour scenarios, full
 *      8-tuple `DoryExecutionCoord` validation (+ negative controls), live construction of all
 *      14 3D beat scenes passing finite/atlas/`DR20`/memory (`<= 12.00 MiB`)/latency caps,
 *      navigation & restore-state immutability, deterministic repeatability, and ironclad
 *      byte-for-byte isolation proof leaving an external multi-branch `DorySession` completely
 *      untouched (metadata, LRU stats, record typed arrays, KV slabs, and future PRNG steps).
 *   5. `D4` Architectural Boundary Isolation & Clean-Room Static Guard — `0` Gate 4b/4c Nano
 *      imports, `0` sibling-app imports, `0` `node:` imports, `0` `Math.random()`, `0` DOM/WebGL
 *      globals across all headless `src/dory/` modules, and untouched `src/render/pick.ts` (`DR21`).
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { normalize3, sub3 } from '../src/render/camera';
import { LINE_INSTANCE_FLOATS } from '../src/render/linePacking';
import { pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import {
  RIBBON_INSTANCE_FLOATS,
  SOLID_INSTANCE_FLOATS,
  type Vec3,
} from '../src/render/primitives';
import { FLOW_RIBBON_VERTEX_FLOATS } from '../src/render/ribbons';
import { localToWorldOnFrame } from '../src/render/sheetFrame';
import {
  SHEET_BASIS_FLOATS,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_FLOATS,
} from '../src/render/sheets';
import {
  assertBitExactFiniteArray,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';
import {
  DORY_CHECKPOINT_FORMATS,
  DORY_CHECKPOINT_LRU_POOL_BYTES,
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_DESIGN_DECISIONS,
  DORY_DISCREPANCY_LEDGER,
  DORY_MAX_ATLAS_BYTES,
  DORY_MAX_SESSION_RETAINED_BYTES,
  DORY_MINI_CONFIG,
  DORY_MINI_WEIGHT_BYTES,
  DORY_PRODUCTION_SPEC,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  buildDoryDualRopeTables,
  computeProductionKvBytesBf16,
  createDorySession,
  effectiveLogitScale,
  getDefaultDoryModelParameters,
  getDoryBrowserGuards,
  runDoryLmHead,
  softmaxFloat64,
  vectorL2Norm,
  type DoryExecutionCoord,
  type DoryKvCache,
  type DorySession,
  type DorySessionRecord,
} from '../src/dory';
import {
  DORY_ARCH_READOUTS_KV_LENGTHS,
  DORY_PROD_MXFP8_QUANTIZED_BYTES,
  DORY_PROD_NVFP4_QUANTIZED_BYTES,
  DORY_PROD_UNQUANTIZED_BF16_BYTES,
  buildDoryArchReadouts,
  buildDoryBf16StageBreakdown,
  buildDoryCheckpointReadouts,
  buildDoryDepthComputeScalingTable,
  buildDoryMemoryBoundsTable,
  buildDoryPaperResultsReadouts,
  buildDoryProductionKvCurve,
  buildDoryProductionKvScalingTable,
  buildDoryTbptlSchematicReadout,
  filterDoryDiscrepancyLedger,
  formatDoryDualBytes,
  verifyDoryProductionKvSInvariantRatio,
} from '../src/dory/archReadouts';
import {
  DEFAULT_DORY_SPATIAL_FOCUS_STATE,
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_BOOKENDS_ATLAS_BUDGET_BYTES,
  DORY_BOOKENDS_ATLAS_HEIGHT,
  DORY_BOOKENDS_ATLAS_WIDTH,
  DORY_BOOKENDS_TOP_LOGIT_COUNT,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_SCENE_BUILDERS,
  DORY_SPATIAL_VIEW_MODES,
  DORY_TBPTL_READONLY_BADGE,
  auditAtlasRegions,
  buildDoryBookendsScene,
  buildDorySpatialScene,
  computeDoryAtlasContentKey,
  computeUntiedBookendsMaxAbsDiff,
  focusStateToViewOptions,
  getDorySpatialViewDescriptor,
  pickHitToDoryCell,
  resolveDorySessionContext,
  resolveDorySpatialFocus,
  validateSceneGeometryFinite,
  type DoryCheckpointFormatId,
  type DorySpatialFocusState,
  type DorySpatialScene,
  type DorySpatialViewMode,
} from '../src/dory/scene';
import {
  DORY_PAPER_RESULTS_BADGE,
  DORY_TOUR_ACT_DESCRIPTORS,
  DORY_TOUR_MIN_BEATS_PER_ACT,
  DORY_TOUR_SEED,
  DORY_TOUR_TOTAL_ACTS,
  type DoryFreeExploreRestoreState,
  type DoryTourActId,
  type DoryTourScenarioKey,
} from '../src/dory/story/types';
import {
  DORY_TOUR_BEATS,
  DORY_TOUR_PROMPT_TOKENS,
  createDoryGuidedTour,
  validateDoryExecutionCoord,
} from '../src/dory/story/walkthrough';

const PARAMS = getDefaultDoryModelParameters();
const ROPE_TABLES = buildDoryDualRopeTables(PARAMS.config);

/** Canonical 12-token non-EOS prompt crossing SWA window W = 8 (pos 8..11). */
const PROMPT_12 = [12, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38] as const;

// ============================================================================
// SHARED GATE 5 VERIFICATION HELPERS
// ============================================================================

function makeDirectRayAtBox(
  box: PickBox,
  row = 0,
  col = 0,
  distance = 4.0,
): Ray {
  if (box.frame) {
    const cols = Math.max(1, box.gridCols ?? 1);
    const rows = Math.max(1, box.gridRows ?? 1);
    const lx = -0.5 + (col + 0.5) / cols;
    const lz = -0.5 + (row + 0.5) / rows;
    const target = localToWorldOnFrame(box.frame, lx, 0.5, lz);
    const origin: Vec3 = [
      target[0] + box.frame.axisN[0] * distance,
      target[1] + box.frame.axisN[1] * distance,
      target[2] + box.frame.axisN[2] * distance,
    ];
    return {
      origin,
      direction: normalize3(sub3(target, origin)),
    };
  }
  const cx = 0.5 * (box.min[0] + box.max[0]);
  const cz = 0.5 * (box.min[2] + box.max[2]);
  return {
    origin: [cx, box.max[1] + distance, cz],
    direction: [0, -1, 0],
  };
}

function readAtlasRegionR(scene: DorySpatialScene, regionId: string): Float32Array {
  const atlas = scene.valueAtlas;
  expect(atlas, `Missing valueAtlas for region '${regionId}'`).not.toBeNull();
  const reg = scene.atlasRegions.find((r) => r.id === regionId);
  expect(reg, `Missing atlas region '${regionId}' in view '${scene.viewMode}'`).toBeDefined();
  const out = new Float32Array(reg!.rows * reg!.cols);
  for (let r = 0; r < reg!.rows; r += 1) {
    for (let c = 0; c < reg!.cols; c += 1) {
      const idx = ((reg!.y0 + r) * atlas!.width + (reg!.x0 + c)) * 2;
      out[r * reg!.cols + c] = atlas!.data[idx]!;
    }
  }
  return out;
}

/**
 * Exhaustive Gate 5 scene verifier: finite geometry, exact 5-pass GPU buffer strides,
 * independent `auditAtlasRegions` (`0` overlaps, `0` OOB, `<= 512 KiB` & `<= viewBudget`),
 * and `DR20` live `RG32F` `G`-channel mask check on every `sheetKind in {1, 2}` plate.
 */
function assertGate5SceneHealth(
  scene: DorySpatialScene,
  label: string,
  expectedViewBudgetBytes?: number,
): void {
  const viewDesc = getDorySpatialViewDescriptor(scene.viewMode);
  const viewBudget = expectedViewBudgetBytes ?? viewDesc.atlasBudgetBytes;

  const validation = validateSceneGeometryFinite(scene);
  expect(validation.valid, `${label} geometry validation: ${validation.errors.join('; ')}`).toBe(
    true,
  );
  expect(validation.errors).toEqual([]);

  // Exact 5-pass GPU buffer stride parity
  expect(scene.solidData.length).toBe(scene.solidCount * SOLID_INSTANCE_FLOATS);
  expect(scene.lineData.length).toBe(scene.lineCount * LINE_INSTANCE_FLOATS);
  expect(scene.ribbonData.length).toBe(scene.ribbonCount * RIBBON_INSTANCE_FLOATS);
  expect(scene.sheetData.length).toBe(scene.sheetCount * SHEET_INSTANCE_FLOATS);
  expect(scene.sheetBasisData.length).toBe(scene.sheetCount * SHEET_BASIS_FLOATS);
  expect(scene.sheetRevealData.length).toBe(scene.sheetCount * SHEET_REVEAL_FLOATS);
  expect(scene.flowVertices.length % FLOW_RIBBON_VERTEX_FLOATS).toBe(0);

  // Live RG32F atlas, <= 512 KiB cap, 0 overlaps, 0 OOB
  expect(scene.valueAtlas, `${label} must carry live RG32F atlas`).not.toBeNull();
  expect(scene.atlasKey, `${label} must carry content-addressed atlasKey`).not.toBeNull();
  expect(scene.atlasAudit, `${label} must carry atlasAudit`).not.toBeNull();

  const atlas = scene.valueAtlas!;
  const byteLength = atlas.width * atlas.height * 8;
  expect(byteLength, `${label} RG32F <= 512 KiB`).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
  expect(byteLength, `${label} RG32F <= view budget`).toBeLessThanOrEqual(viewBudget);

  const reaudit = auditAtlasRegions(
    atlas.width,
    atlas.height,
    scene.atlasRegions,
    DORY_MAX_ATLAS_BYTES,
  );
  expect(reaudit.hasOverlap, `${label} overlap: ${reaudit.overlapPair?.join(' vs ')}`).toBe(false);
  expect(reaudit.overlapPair).toBeNull();
  expect(reaudit.outOfBoundsIds).toEqual([]);
  expect(reaudit.withinBudget).toBe(true);
  expect(reaudit.byteLength).toBe(byteLength);
  expect(reaudit.usedTexels).toBeGreaterThan(0);
  expect(reaudit.usedTexels).toBeLessThanOrEqual(reaudit.totalTexels);

  expect(scene.atlasAudit!.hasOverlap).toBe(false);
  expect(scene.atlasAudit!.outOfBoundsIds).toEqual([]);
  expect(scene.atlasAudit!.withinBudget).toBe(true);
  expect(scene.atlasAudit!.byteLength).toBe(byteLength);

  // DR20 Direct Instance-Buffer Audit on EVERY causal score/softmax sheet
  for (let i = 0; i < scene.sheetCount; i += 1) {
    const base = i * SHEET_INSTANCE_FLOATS;
    const sheetKind = Math.round(scene.sheetData[base + 8]!);
    if (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) {
      const u0 = scene.sheetData[base + 13]!;
      const v0 = scene.sheetData[base + 14]!;
      const uW = scene.sheetData[base + 15]!;
      const vH = scene.sheetData[base + 16]!;
      expect(uW, `${label}.sheet[${i}] DR20 uW > 0`).toBeGreaterThan(0);
      expect(vH, `${label}.sheet[${i}] DR20 vH > 0`).toBeGreaterThan(0);
      const x0 = Math.floor(u0 * atlas.width + 1e-4);
      const y0 = Math.floor(v0 * atlas.height + 1e-4);
      const x1 = Math.ceil((u0 + uW) * atlas.width - 1e-4);
      const y1 = Math.ceil((v0 + vH) * atlas.height - 1e-4);
      for (let ty = y0; ty < y1; ty += 1) {
        for (let tx = x0; tx < x1; tx += 1) {
          const tIdx = (ty * atlas.width + tx) * 2;
          const rVal = atlas.data[tIdx]!;
          const gVal = atlas.data[tIdx + 1]!;
          expect(Number.isFinite(rVal), `${label}.sheet[${i}] finite R`).toBe(true);
          expect(gVal === 0 || gVal === 1, `${label}.sheet[${i}] binary G`).toBe(true);
          if (gVal === 1) {
            expect(rVal, `${label}.sheet[${i}] masked R === 0`).toBe(0);
          }
        }
      }
    } else {
      expect(sheetKind, `${label}.sheet[${i}] non-causal sheetKind`).toBe(SHEET_KIND_STANDARD);
    }
  }
}

interface RecordByteCopy {
  readonly recordId: string;
  readonly branchId: string;
  readonly kind: DorySessionRecord['kind'];
  readonly committedLen: number;
  readonly pendingToken: number | null;
  readonly chunkSizesBefore: readonly number[];
  readonly chunkSizesAfter: readonly number[];
  readonly inputTokens: Int32Array;
  readonly emittedTokens: Int32Array;
  readonly committedSequence: Int32Array;
  readonly lastTokenScaledLogits: Float32Array;
  readonly lastTokenProbs: Float64Array;
  readonly lastTokenXAnchor: Float32Array;
  readonly lastTokenLoopEntryStates: readonly Float32Array[];
  readonly lastTokenLoopExitStates: readonly Float32Array[];
}

interface ExternalSessionByteSnapshot {
  readonly snapshot: ReturnType<DorySession['getSnapshot']>;
  readonly memory: ReturnType<DorySession['measureMemory']>;
  readonly branchRecords: ReadonlyMap<string, readonly RecordByteCopy[]>;
}

function captureExternalSessionBytes(session: DorySession): ExternalSessionByteSnapshot {
  const snapshot = session.getSnapshot();
  const memory = session.measureMemory();
  const branchRecords = new Map<string, readonly RecordByteCopy[]>();

  for (const branch of snapshot.branches) {
    const recs = session.listBranchRecords(branch.branchId);
    branchRecords.set(
      branch.branchId,
      recs.map((rec) => ({
        recordId: rec.recordId,
        branchId: rec.branchId,
        kind: rec.kind,
        committedLen: rec.committedLen,
        pendingToken: rec.pendingToken,
        chunkSizesBefore: [...rec.chunkSizesBefore],
        chunkSizesAfter: [...rec.chunkSizesAfter],
        inputTokens: new Int32Array(rec.inputTokens),
        emittedTokens: new Int32Array(rec.emittedTokens),
        committedSequence: new Int32Array(rec.committedSequence),
        lastTokenScaledLogits: new Float32Array(rec.lastTokenScaledLogits),
        lastTokenProbs: new Float64Array(rec.lastTokenProbs),
        lastTokenXAnchor: new Float32Array(rec.lastTokenXAnchor),
        lastTokenLoopEntryStates: rec.lastTokenLoopEntryStates.map((a) => new Float32Array(a)),
        lastTokenLoopExitStates: rec.lastTokenLoopExitStates.map((a) => new Float32Array(a)),
      })),
    );
  }

  return { snapshot, memory, branchRecords };
}

function assertExternalSessionUntouched(
  session: DorySession,
  before: ExternalSessionByteSnapshot,
  label: string,
  options: { readonly allowLruReadHits?: boolean } = {},
): void {
  const afterSnap = session.getSnapshot();
  const afterMem = session.measureMemory();

  expect(afterMem, `${label}.memory`).toEqual(before.memory);
  if (options.allowLruReadHits) {
    // Scene builders invoked directly on `session` call `reconstructStepTrace()`, which performs
    // read-only LRU checkpoint lookups (incrementing `hitCount` and `lastUsedSeq`) without
    // mutating branches, records, memory usage, or KV payload/tag buffers.
    expect(afterSnap.modelWeightId, `${label}.modelWeightId`).toBe(before.snapshot.modelWeightId);
    expect(afterSnap.activeBranchId, `${label}.activeBranchId`).toBe(before.snapshot.activeBranchId);
    expect(afterSnap.activeRecordId, `${label}.activeRecordId`).toBe(before.snapshot.activeRecordId);
    expect(afterSnap.bounds, `${label}.bounds`).toEqual(before.snapshot.bounds);
    expect(afterSnap.branches, `${label}.branches`).toEqual(before.snapshot.branches);
    expect(afterSnap.recordCount, `${label}.recordCount`).toBe(before.snapshot.recordCount);
    expect(afterSnap.checkpointStats.usedBytes, `${label}.lruUsedBytes`).toBe(
      before.snapshot.checkpointStats.usedBytes,
    );
    expect(afterSnap.checkpointStats.entryCount, `${label}.lruEntryCount`).toBe(
      before.snapshot.checkpointStats.entryCount,
    );
    expect(afterSnap.checkpointStats.evictionCount, `${label}.lruEvictionCount`).toBe(
      before.snapshot.checkpointStats.evictionCount,
    );
  } else {
    expect(afterSnap, `${label}.snapshot`).toEqual(before.snapshot);
  }

  for (const [branchId, beforeRecs] of before.branchRecords.entries()) {
    const afterRecs = session.listBranchRecords(branchId);
    expect(afterRecs.length, `${label}.${branchId}.recordCount`).toBe(beforeRecs.length);
    for (let i = 0; i < beforeRecs.length; i += 1) {
      const b = beforeRecs[i]!;
      const a = afterRecs[i]!;
      expect(a.recordId).toBe(b.recordId);
      expect(a.branchId).toBe(b.branchId);
      expect(a.kind).toBe(b.kind);
      expect(a.committedLen).toBe(b.committedLen);
      expect(a.pendingToken).toBe(b.pendingToken);
      expect(a.chunkSizesBefore).toEqual(b.chunkSizesBefore);
      expect(a.chunkSizesAfter).toEqual(b.chunkSizesAfter);
      assertBitExactFiniteArray(a.inputTokens, b.inputTokens, `${label}.${b.recordId}.input`);
      assertBitExactFiniteArray(a.emittedTokens, b.emittedTokens, `${label}.${b.recordId}.emitted`);
      assertBitExactFiniteArray(
        a.committedSequence,
        b.committedSequence,
        `${label}.${b.recordId}.committedSeq`,
      );
      assertBitExactFiniteArray(
        a.lastTokenScaledLogits,
        b.lastTokenScaledLogits,
        `${label}.${b.recordId}.logits`,
      );
      assertBitExactFiniteArray(a.lastTokenProbs, b.lastTokenProbs, `${label}.${b.recordId}.probs`);
      assertBitExactFiniteArray(
        a.lastTokenXAnchor,
        b.lastTokenXAnchor,
        `${label}.${b.recordId}.xAnchor`,
      );
      expect(a.lastTokenLoopEntryStates.length).toBe(b.lastTokenLoopEntryStates.length);
      expect(a.lastTokenLoopExitStates.length).toBe(b.lastTokenLoopExitStates.length);
      for (let t = 0; t < b.lastTokenLoopEntryStates.length; t += 1) {
        assertBitExactFiniteArray(
          a.lastTokenLoopEntryStates[t]!,
          b.lastTokenLoopEntryStates[t]!,
          `${label}.${b.recordId}.u[${t}]`,
        );
        assertBitExactFiniteArray(
          a.lastTokenLoopExitStates[t]!,
          b.lastTokenLoopExitStates[t]!,
          `${label}.${b.recordId}.s[${t}]`,
        );
      }
    }
  }
}

function assertKvCachesBitExact(actual: DoryKvCache, expected: DoryKvCache, label: string): void {
  expect(actual.mode).toBe(expected.mode);
  expect(actual.kLoops).toBe(expected.kLoops);
  expect(actual.includeShallowCoda).toBe(expected.includeShallowCoda);
  expect(actual.committedLen).toBe(expected.committedLen);
  expect(actual.chunkSizes).toEqual(expected.chunkSizes);
  expect(actual.allBanks.length).toBe(expected.allBanks.length);
  for (let i = 0; i < expected.allBanks.length; i += 1) {
    const a = actual.allBanks[i]!;
    const e = expected.allBanks[i]!;
    expect(a.bankSlotId).toBe(e.bankSlotId);
    expect(a.logicalLen).toBe(e.logicalLen);
    assertBitExactFiniteArray(a.k, e.k, `${label}.bank[${i}].k`);
    assertBitExactFiniteArray(a.v, e.v, `${label}.bank[${i}].v`);
    assertBitExactFiniteArray(a.absPosTag, e.absPosTag, `${label}.bank[${i}].absPosTag`);
    assertBitExactFiniteArray(a.writerLoopTag, e.writerLoopTag, `${label}.bank[${i}].writerLoopTag`);
  }
}

function collectKvArrayBuffers(kv: DoryKvCache): Set<ArrayBufferLike> {
  const buffers = new Set<ArrayBufferLike>();
  for (const bank of kv.allBanks) {
    buffers.add(bank.k.buffer);
    buffers.add(bank.v.buffer);
    buffers.add(bank.absPosTag.buffer);
    buffers.add(bank.writerLoopTag.buffer);
  }
  return buffers;
}

function listTsFilesRecursive(dir: string): string[] {
  const results: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fullPath = resolve(dir, entry);
    const st = statSync(fullPath);
    if (st.isDirectory()) {
      results.push(...listTsFilesRecursive(fullPath));
    } else if (entry.endsWith('.ts')) {
      results.push(fullPath);
    }
  }
  return results.sort();
}

// ============================================================================
// 1. ARCH READOUTS — 0 B DIFF 4-CHECKPOINT CONSTANTS & BF16 STAGE BREAKDOWN
// ============================================================================

describe('Gate 5 archReadouts.ts — 0 B Diff 4-Checkpoint Constants & BF16 Stage Breakdown', () => {
  it('reconciles all 4 production checkpoints (BF16 / MXFP8 / NVFP4 / Q5_K_M) with exact 0 B safetensors diff and independent byte-level arithmetic', () => {
    const rows = buildDoryCheckpointReadouts();
    expect(rows).toHaveLength(4);
    expect(rows.map((r) => r.id)).toEqual(['bf16', 'mxfp8', 'nvfp4', 'q5_k_m']);

    // Independently derive exact production parameter & byte counts from DORY_PRODUCTION_SPEC
    const d = DORY_PRODUCTION_SPEC.hiddenSize; // 2560
    const dFf = DORY_PRODUCTION_SPEC.intermediateSize; // 7168
    const qDim = DORY_PRODUCTION_SPEC.qProjOutDim; // 4096
    const kvDim = DORY_PRODUCTION_SPEC.kvProjOutDim; // 1024
    const v = DORY_PRODUCTION_SPEC.vocabSize; // 131072
    const pairs = DORY_PRODUCTION_SPEC.numPairedLayers; // 36

    const bookendParams = 2 * v * d + v; // embeddings + lm_head + logit_scale = 671,219,712
    const linearPerPair = (qDim + kvDim + kvDim + qDim + qDim + 3 * dFf) * d; // 91,750,400
    const scalesPerPair = (d + qDim + qDim) + (d + dFf + dFf); // 27,648
    const totalLinearParams = pairs * linearPerPair; // 3,303,014,400
    const totalScaleParams = pairs * scalesPerPair + v; // 1,126,400
    const totalLogicalParams = bookendParams + totalLinearParams + pairs * scalesPerPair; // 3,975,229,440

    expect(totalLogicalParams).toBe(3_975_229_440);
    expect(totalLogicalParams).toBe(DORY_PRODUCTION_SPEC.totalParams);

    const expectedUnquantized219Bytes = (2 * v * d + totalScaleParams) * 2; // 1,344,430,080 B
    const expectedMxfp8QuantBytes = totalLinearParams + totalLinearParams / 32; // 3,406,233,600 B
    const expectedNvfp4QuantBytes = totalLinearParams / 2 + totalLinearParams / 16 + 288 * 4; // 1,857,946,752 B

    expect(DORY_PROD_UNQUANTIZED_BF16_BYTES).toBe(expectedUnquantized219Bytes);
    expect(DORY_PROD_MXFP8_QUANTIZED_BYTES).toBe(expectedMxfp8QuantBytes);
    expect(DORY_PROD_NVFP4_QUANTIZED_BYTES).toBe(expectedNvfp4QuantBytes);

    const [bf16, mxfp8, nvfp4, gguf] = rows;

    // 1. BF16
    expect(bf16!.shortSha).toBe('ee517dd3');
    expect(bf16!.tensorCount).toBe(507);
    expect(bf16!.unquantizedBf16TensorCount).toBe(507);
    expect(bf16!.quantizedLinearMatrixCount).toBe(0);
    expect(bf16!.logicalParams).toBe(3_975_229_440);
    expect(bf16!.headerJsonBytes).toBe(56_584);
    expect(bf16!.payloadBytes).toBe(totalLogicalParams * 2);
    expect(bf16!.payloadBytes).toBe(7_950_458_880);
    expect(bf16!.totalFileBytes).toBe(7_950_515_472);
    expect(8 + bf16!.headerJsonBytes! + bf16!.payloadBytes!).toBe(bf16!.totalFileBytes);
    expect(bf16!.diffBytes).toBe(0);
    expect(bf16!.safetensorsContainerExact).toBe(true);
    expect(bf16!.zeroDiffVerified).toBe(true);
    expect(bf16!.license).toBe('OpenMDW-1.1');

    // 2. MXFP8-PTQ (DR14/DR15)
    expect(mxfp8!.shortSha).toBe('f57fc2d5');
    expect(mxfp8!.tensorCount).toBe(795);
    expect(mxfp8!.unquantizedBf16TensorCount).toBe(219);
    expect(mxfp8!.quantizedLinearMatrixCount).toBe(288);
    expect(mxfp8!.storedElements).toBe(4_078_448_640);
    expect(mxfp8!.headerJsonBytes).toBe(91_616);
    expect(mxfp8!.unquantizedBf16Bytes).toBe(expectedUnquantized219Bytes);
    expect(mxfp8!.quantizedPayloadBytes).toBe(expectedMxfp8QuantBytes);
    expect(mxfp8!.payloadBytes).toBe(expectedUnquantized219Bytes + expectedMxfp8QuantBytes);
    expect(mxfp8!.payloadBytes).toBe(4_750_663_680);
    expect(mxfp8!.totalFileBytes).toBe(4_750_755_304);
    expect(8 + mxfp8!.headerJsonBytes! + mxfp8!.payloadBytes!).toBe(mxfp8!.totalFileBytes);
    expect(mxfp8!.diffBytes).toBe(0);
    expect(mxfp8!.safetensorsContainerExact).toBe(true);
    expect(mxfp8!.zeroDiffVerified).toBe(true);
    expect(mxfp8!.discrepancyRefs).toContain('DR15');
    expect(mxfp8!.dgxSparkReadoutNote).toContain('exact_e8m0_search_with_dominance_bounds');

    // 3. NVFP4-QAD
    expect(nvfp4!.shortSha).toBe('17065a68');
    expect(nvfp4!.tensorCount).toBe(1_083);
    expect(nvfp4!.unquantizedBf16TensorCount).toBe(219);
    expect(nvfp4!.quantizedLinearMatrixCount).toBe(288);
    expect(nvfp4!.storedElements).toBe(2_530_160_928);
    expect(nvfp4!.headerJsonBytes).toBe(121_192);
    expect(nvfp4!.unquantizedBf16Bytes).toBe(expectedUnquantized219Bytes);
    expect(nvfp4!.quantizedPayloadBytes).toBe(expectedNvfp4QuantBytes);
    expect(nvfp4!.payloadBytes).toBe(expectedUnquantized219Bytes + expectedNvfp4QuantBytes);
    expect(nvfp4!.payloadBytes).toBe(3_202_376_832);
    expect(nvfp4!.totalFileBytes).toBe(3_202_498_032);
    expect(8 + nvfp4!.headerJsonBytes! + nvfp4!.payloadBytes!).toBe(nvfp4!.totalFileBytes);
    expect(nvfp4!.diffBytes).toBe(0);
    expect(nvfp4!.safetensorsContainerExact).toBe(true);
    expect(nvfp4!.zeroDiffVerified).toBe(true);

    // 4. GGUF Q5_K_M (QAD)
    expect(gguf!.shortSha).toBe('df53755c');
    expect(gguf!.totalFileBytes).toBe(3_673_288_864);
    expect(gguf!.headerJsonBytes).toBeNull();
    expect(gguf!.payloadBytes).toBeNull();
    expect(gguf!.payloadFormatted).toBeNull();
    expect(gguf!.safetensorsContainerExact).toBeNull();
    expect(gguf!.zeroDiffVerified).toBe(false);
    expect(gguf!.discrepancyRefs).toContain('DR11');
    expect(gguf!.discrepancyRefs).toContain('DR14');
    expect(gguf!.license).toBe('OpenMDW-1.1');

    // Dual SI / IEC byte formatter exactness & error handling (7,950,515,472 / 2^30 = 7.40449... GiB)
    const bf16Fmt = formatDoryDualBytes(7_950_515_472);
    expect(bf16Fmt.siLabel).toBe('7.951 GB');
    expect(bf16Fmt.iecLabel).toBe('7.404 GiB');
    expect(bf16Fmt.dualLabel).toBe('7.404 GiB (7.951 GB)');

    expect(formatDoryDualBytes(0).dualLabel).toBe('0 B');
    expect(formatDoryDualBytes(512).dualLabel).toBe('512 B');
    expect(formatDoryDualBytes(1_024).iecLabel).toBe('1.00 KiB');
    expect(formatDoryDualBytes(1_048_576).iecLabel).toBe('1.00 MiB');

    expect(() => formatDoryDualBytes(-1)).toThrow(RangeError);
    expect(() => formatDoryDualBytes(Number.NaN)).toThrow(RangeError);
    expect(() => formatDoryDualBytes(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });

  it('partitions BF16 stage & component parameters disjointly to exact 507 tensors, 3,975,229,440 params, and 0 norms/biases/adapters', () => {
    const breakdown = buildDoryBf16StageBreakdown();
    const partitionRows = breakdown.filter((r) => !r.isSubtotal);
    const subtotalRows = breakdown.filter((r) => r.isSubtotal === true);

    expect(partitionRows).toHaveLength(6);
    expect(subtotalRows).toHaveLength(3);

    const sumTensors = partitionRows.reduce((acc, r) => acc + r.tensorCount, 0);
    const sumParams = partitionRows.reduce((acc, r) => acc + r.parameters, 0);
    const sumBf16Bytes = partitionRows.reduce((acc, r) => acc + r.bf16Payload.bytes, 0);
    const sumSharePct = partitionRows.reduce((acc, r) => acc + r.shareOfParamsPct, 0);

    expect(sumTensors).toBe(507);
    expect(sumParams).toBe(3_975_229_440);
    expect(sumBf16Bytes).toBe(7_950_458_880);
    expect(sumSharePct).toBeCloseTo(100.0, 10);

    const byId = new Map(breakdown.map((r) => [r.id, r]));
    expect(byId.get('embeddings')!.parameters).toBe(335_544_320);
    expect(byId.get('lm_head_weight')!.parameters).toBe(335_544_320);
    expect(byId.get('lm_head_logit_scale')!.parameters).toBe(131_072);
    expect(byId.get('input_prelude')!.parameters).toBe(734_224_384);
    expect(byId.get('recurrent_core')!.parameters).toBe(1_835_560_960);
    expect(byId.get('output_coda')!.parameters).toBe(734_224_384);

    // Cross-partition subtotals: 288 linear + 217 scales + 2 untied bookend matrices === 507 tensors!
    const linearSub = byId.get('subtotal_linear_weights')!;
    const scaleSub = byId.get('subtotal_ngpt_scales')!;
    const zeroSub = byId.get('subtotal_zero_norms_biases_adapters')!;

    expect(linearSub.tensorCount).toBe(288);
    expect(linearSub.parameters).toBe(3_303_014_400);
    expect(scaleSub.tensorCount).toBe(217);
    expect(scaleSub.parameters).toBe(1_126_400);
    expect(zeroSub.tensorCount).toBe(0);
    expect(zeroSub.parameters).toBe(0);
    expect(linearSub.tensorCount + scaleSub.tensorCount + 2).toBe(507);
    expect(linearSub.parameters + scaleSub.parameters + 2 * 335_544_320).toBe(3_975_229_440);
  });
});

// ============================================================================
// 2. ARCH READOUTS — S-INVARIANT (4+5k)/9 RATIO, 1 -> 256K KV & 4x SWA KNEE
// ============================================================================

describe('Gate 5 archReadouts.ts — Depth Scaling, Exact 1 -> 256K BF16 KV & S-Invariant (4+5k)/9 Ratio', () => {
  it('verifies exact S-invariant KV_per_loop(S, k) / KV_last_loop(S, k) === (4 + 5k) / 9 across all k in {1,2,3,4,6,8} and arbitrary S in [1 .. 262,144]', () => {
    const denseLengths = [
      1, 2, 3, 7, 16, 128, 512, 1_023, 1_024, 1_025, 2_048, 4_096, 8_192, 32_768, 65_536, 131_072,
      262_144,
    ] as const;

    for (const k of DORY_MINI_CONFIG.supportedRecurrentLoops) {
      const check = verifyDoryProductionKvSInvariantRatio(k, denseLengths);
      const expectedRatio = (4 + 5 * k) / 9;
      expect(check.kLoops).toBe(k);
      expect(check.expectedRatio).toBeCloseTo(expectedRatio, 15);
      expect(check.maxAbsDeviation).toBeLessThanOrEqual(1e-15);
      expect(check.invariantVerified).toBe(true);

      const perCurve = buildDoryProductionKvCurve(k, 'per_loop', denseLengths);
      const lastCurve = buildDoryProductionKvCurve(k, 'last_loop', denseLengths);
      expect(perCurve).toHaveLength(denseLengths.length);
      expect(lastCurve).toHaveLength(denseLengths.length);

      for (let i = 0; i < denseLengths.length; i += 1) {
        const s = denseLengths[i]!;
        const pl = perCurve[i]!;
        const ll = lastCurve[i]!;
        expect(pl.seqLen).toBe(s);
        expect(ll.seqLen).toBe(s);
        // Exact S-invariant ratio holds identically for SWA, Full, AND Total bytes!
        expect(pl.totalBytes / ll.totalBytes).toBeCloseTo(expectedRatio, 14);
        expect(pl.swaBytes / ll.swaBytes).toBeCloseTo(expectedRatio, 14);
        expect(pl.fullBytes / ll.fullBytes).toBeCloseTo(expectedRatio, 14);
        // SWA share is identically k-invariant and mode-invariant!
        expect(Math.abs(pl.swaShare - ll.swaShare)).toBeLessThanOrEqual(1e-15);
      }
    }

    // Invalid k or empty / < 1 seqLengths throw RangeError (guards against vacuous true!)
    expect(() => verifyDoryProductionKvSInvariantRatio(0)).toThrow(RangeError);
    expect(() => verifyDoryProductionKvSInvariantRatio(4, [])).toThrow(RangeError);
    expect(() => verifyDoryProductionKvSInvariantRatio(4, [0])).toThrow(RangeError);
    expect(() => verifyDoryProductionKvSInvariantRatio(4, [-4, 0])).toThrow(RangeError);
    expect(() => verifyDoryProductionKvSInvariantRatio(4, [1.5])).toThrow(RangeError);
    expect(() => buildDoryProductionKvCurve(-2, 'per_loop')).toThrow(RangeError);
    expect(() => buildDoryProductionKvCurve(2.5, 'last_loop')).toThrow(RangeError);
  });

  it('reproduces every Audit §7.2 milestone byte count and proves exact 4x marginal slope drop across the S = 1,024 SWA knee', () => {
    const table = buildDoryProductionKvScalingTable();
    expect(table).toHaveLength(DORY_ARCH_READOUTS_KV_LENGTHS.length);

    const kneeRows = table.filter((r) => r.isSwaKnee);
    expect(kneeRows).toHaveLength(1);
    expect(kneeRows[0]!.seqLen).toBe(1_024);

    const bySeq = new Map(table.map((r) => [r.seqLen, r]));
    const MiB = 1_048_576;

    // Audit §7.2 exact milestones
    const r1 = bySeq.get(1)!;
    expect(r1.lastLoopPoint.totalBytes).toBe(36 * 4_096); // 147,456 B (0.14 MiB)
    expect(r1.swaSharePct).toBeCloseTo(75.0, 12);

    const r512 = bySeq.get(512)!;
    expect(r512.lastLoopPoint.totalBytes).toBe(72 * MiB);
    expect(r512.perLoopK2Point.totalBytes).toBe(112 * MiB);
    expect(r512.perLoopK4Point.totalBytes).toBe(192 * MiB);
    expect(r512.swaSharePct).toBeCloseTo(75.0, 12);

    const r1024 = bySeq.get(1_024)!;
    expect(r1024.lastLoopPoint.totalBytes).toBe(144 * MiB);
    expect(r1024.perLoopK2Point.totalBytes).toBe(224 * MiB);
    expect(r1024.perLoopK4Point.totalBytes).toBe(384 * MiB);
    expect(r1024.swaSharePct).toBeCloseTo(75.0, 12);

    const r2048 = bySeq.get(2_048)!;
    expect(r2048.lastLoopPoint.totalBytes).toBe(180 * MiB);
    expect(r2048.perLoopK2Point.totalBytes).toBe(280 * MiB);
    expect(r2048.perLoopK4Point.totalBytes).toBe(480 * MiB);
    expect(r2048.swaSharePct).toBeCloseTo(60.0, 12);

    const r256k = bySeq.get(262_144)!;
    expect(r256k.lastLoopPoint.totalBytes).toBe(9_324 * MiB);
    expect(r256k.perLoopK2Point.totalBytes).toBe(14_504 * MiB);
    expect(r256k.perLoopK4Point.totalBytes).toBe(24_864 * MiB);
    expect(r256k.swaSharePct).toBeCloseTo((3 * 1_024 * 100) / (3 * 1_024 + 262_144), 10);
    expect(r256k.swaSharePct).toBeCloseTo(1.158, 2);

    // Exact 4x marginal bytes/token slope drop across S = 1,024 in BOTH table & finite differences
    for (const row of table) {
      if (row.seqLen <= 1_024) {
        expect(row.marginalBytesPerTokLastLoop).toBe(144 * 1_024); // 144 KiB/tok
        expect(row.marginalBytesPerTokPerLoopK4).toBe(384 * 1_024); // 384 KiB/tok
      } else {
        expect(row.marginalBytesPerTokLastLoop).toBe(36 * 1_024); // 36 KiB/tok (4x drop!)
        expect(row.marginalBytesPerTokPerLoopK4).toBe(96 * 1_024); // 96 KiB/tok (4x drop!)
      }
    }
    const slopePreKneeLl =
      computeProductionKvBytesBf16(1_024, 1, 'last_loop').totalBytes -
      computeProductionKvBytesBf16(1_023, 1, 'last_loop').totalBytes;
    const slopePostKneeLl =
      computeProductionKvBytesBf16(1_025, 1, 'last_loop').totalBytes -
      computeProductionKvBytesBf16(1_024, 1, 'last_loop').totalBytes;
    expect(slopePreKneeLl).toBe(144 * 1_024);
    expect(slopePostKneeLl).toBe(36 * 1_024);
    expect(slopePreKneeLl / slopePostKneeLl).toBe(4);

    // Depth vs. linear compute scaling table (Audit §6.1)
    const depthTable = buildDoryDepthComputeScalingTable();
    expect(depthTable.map((r) => r.kLoops)).toEqual([1, 2, 3, 4, 6, 8]);
    for (const row of depthTable) {
      expect(row.physicalSublayers).toBe(72);
      expect(row.executedSublayers).toBe(32 + 40 * row.kLoops);
      expect(row.effectivePairedDepth).toBe(16 + 20 * row.kLoops);
      expect(row.isTrained).toBe(row.kLoops <= 4);
      expect(row.isDefault).toBe(row.kLoops === 2);
      expect(row.bodyComputeRatioVsK1).toBeCloseTo((4 + 5 * row.kLoops) / 9, 14);
      expect(row.perLoopToLastLoopKvRatio).toBeCloseTo((4 + 5 * row.kLoops) / 9, 14);
      expect(row.perLoopTotalBanks).toBe(16 + 20 * row.kLoops);
      expect(row.lastLoopTotalBanks).toBe(36);
      if (row.kLoops >= 2) {
        // Untied LM head is k-invariant -> total linear ratio is strictly below body ratio
        expect(row.linearWithHeadRatioVsK1).toBeLessThan(row.bodyComputeRatioVsK1);
      }
    }
    const k4Depth = depthTable.find((r) => r.kLoops === 4)!;
    expect(k4Depth.bodyComputeRatioVsK1).toBeCloseTo(8 / 3, 14);
    expect(k4Depth.linearWithHeadRatioVsK1).toBeCloseTo(2.513, 3);
  });
});

// ============================================================================
// 3. ARCH READOUTS — D10 MEMORY BOUNDS, DR1–DR24 FILTER & READ-ONLY TBPTL GUARD
// ============================================================================

describe('Gate 5 archReadouts.ts — D10 Memory Bounds, DR1–DR24 Filter & READ-ONLY · NO GRADIENTS TBPTL', () => {
  it('enforces strictly READ-ONLY · NO GRADIENTS TBPTL depth-1 invariants, Paper Results separation & unified bundle assembly', () => {
    const tbptl = buildDoryTbptlSchematicReadout();
    expect(tbptl.readOnly).toBe(true);
    expect(tbptl.hasGradients).toBe(false);
    expect(tbptl.zeroTrainingLoops).toBe(true);
    expect(tbptl.zeroFakeQuantToggles).toBe(true);
    expect(tbptl.badge).toBe('READ-ONLY · NO GRADIENTS');
    expect(tbptl.badge).toBe(DORY_TBPTL_READONLY_BADGE);
    expect(tbptl.tbptlDepth).toBe(1);
    expect(tbptl.stopGradientBetweenLoops).toBe(true);
    expect(tbptl.earlyExitLosses).toBe(0);
    expect(tbptl.elrRankOverride).toBe(0.1);
    expect(tbptl.rlvrDifficultyRatio).toBe('4:1:1:1');
    expect(tbptl.rlvrAlpha).toBe(0.3);
    expect(tbptl.preludeGradientPath).toContain('+x');
    expect(tbptl.preludeGradientPath).toContain('stop-gradient');
    expect(tbptl.stages).toHaveLength(5);
    expect(tbptl.stages.map((s) => s.id)).toEqual([
      'pretrain_stage1',
      'pretrain_stage2',
      'pretrain_stage3',
      'posttrain_sft',
      'posttrain_rlvr',
    ]);
    expect(tbptl.stages[0]!.tokenBudgetLabel).toContain('4.5T');
    expect(tbptl.stages[1]!.tokenBudgetLabel).toContain('1.5T');
    expect(tbptl.stages[2]!.tokenBudgetLabel).toContain('300B');

    // Paper results strictly separated from synthetic miniature execution
    const paperResults = buildDoryPaperResultsReadouts();
    expect(paperResults.length).toBeGreaterThanOrEqual(5);
    for (const item of paperResults) {
      expect(item.badge).toBe(DORY_PAPER_RESULTS_BADGE);
      expect(item.badge).toContain('NOT MEASURED ON SYNTHETIC MINIATURE');
      expect(item.caveat.length).toBeGreaterThan(20);
    }
    const specResult = paperResults.find((p) => p.id === 'paper_self_spec_speedup')!;
    expect(specResult.reportedValue).toContain('1.37x–1.71x');
    expect(specResult.discrepancyRefs).toContain('DR13');
    expect(specResult.discrepancyRefs).toContain('DR23');

    // Full unified readouts bundle (validates & surfaces activeKLoops and activeKvCacheMode)
    const bundle = buildDoryArchReadouts({
      activeKLoops: 4,
      activeKvCacheMode: 'last_loop',
      selectedCheckpointFormat: 'nvfp4',
      discrepancyFilter: { onlyBrowserGuards: true },
    });
    expect(bundle.readOnly).toBe(true);
    expect(bundle.zeroFakeQuantToggles).toBe(true);
    expect(bundle.allSafetensorsZeroByteDiff).toBe(true);
    expect(bundle.activeKLoops).toBe(4);
    expect(bundle.activeKvCacheMode).toBe('last_loop');
    expect(bundle.selectedCheckpointFormat).toBe('nvfp4');
    expect(bundle.activeSInvariantRatio).toBeCloseTo(24 / 9, 14);
    expect(bundle.swaKneeSlopeDropFactor).toBe(4);
    expect(bundle.designDecisions).toHaveLength(DORY_DESIGN_DECISIONS.length);
    expect(bundle.designDecisions).toHaveLength(10);
    expect(bundle.tbptlSchematic).toEqual(tbptl);
    expect(bundle.discrepancyLedger.totalCount).toBe(24);
    expect(bundle.discrepancyLedger.filteredCount).toBe(getDoryBrowserGuards().length);
    expect(bundle.discrepancyLedger.rows.every((r) => r.isBrowserGuard)).toBe(true);

    expect(() =>
      buildDoryArchReadouts({ selectedCheckpointFormat: 'fp3' as DoryCheckpointFormatId }),
    ).toThrow(RangeError);
    expect(() =>
      buildDoryArchReadouts({ activeKvCacheMode: 'hybrid_invalid' as 'per_loop' }),
    ).toThrow(RangeError);
    expect(() => buildDoryArchReadouts({ activeKLoops: 0 })).toThrow(RangeError);
  });

  it('verifies D10 analytical conditional memory bounds (k=4 <= 7.29 MiB, k=8 <= 9.60 MiB <= 12 MiB), live session integration & DR1–DR24 ledger filtering', () => {
    const staticBounds = buildDoryMemoryBoundsTable();
    expect(staticBounds).toHaveLength(7);
    const byId = new Map(staticBounds.map((r) => [r.id, r]));

    expect(byId.get('singleton_weights')!.bytes).toBe(DORY_MINI_WEIGHT_BYTES);
    expect(byId.get('singleton_weights')!.bytes).toBe(372_992);
    expect(byId.get('kv_k4_spec_104')!.bytes).toBe(260 * 1_024);
    expect(byId.get('kv_k8_extrap_176')!.bytes).toBe(440 * 1_024);
    expect(byId.get('d10_lru_pool')!.bytes).toBe(DORY_CHECKPOINT_LRU_POOL_BYTES);
    expect(byId.get('session_bound_k4')!.bytes).toBeLessThanOrEqual(7.29 * 1_048_576);
    expect(byId.get('session_bound_k8')!.bytes).toBeLessThanOrEqual(9.60 * 1_048_576);
    expect(byId.get('rg32f_atlas_cap')!.bytes).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
    for (const row of staticBounds) {
      expect(row.withinCap, `Bound row '${row.id}' must be withinCap`).toBe(true);
    }

    // Integrate live DorySessionMemoryAudit + activeAtlasBytes
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: { promptTokens: PROMPT_12, kLoops: 2, kvCacheMode: 'per_loop' },
    });
    const liveMem = session.measureMemory();
    const liveBounds = buildDoryMemoryBoundsTable(liveMem, 81_920);
    expect(liveBounds).toHaveLength(8);
    const liveRow = liveBounds.find((r) => r.id === 'live_session_retained')!;
    expect(liveRow.bytes).toBe(liveMem.totalWithWeightsBytes);
    expect(liveRow.withinCap).toBe(true);
    expect(liveBounds.find((r) => r.id === 'rg32f_atlas_cap')!.bytes).toBe(81_920);

    // DR1–DR24 structured ledger filtering
    const allLedger = filterDoryDiscrepancyLedger();
    expect(allLedger.totalCount).toBe(24);
    expect(allLedger.filteredCount).toBe(24);
    expect(allLedger.rows).toEqual(DORY_DISCREPANCY_LEDGER);

    const guardsOnly = filterDoryDiscrepancyLedger({ onlyBrowserGuards: true });
    expect(guardsOnly.rows.map((r) => r.id)).toEqual([
      'DR18',
      'DR19',
      'DR20',
      'DR21',
      'DR22',
      'DR24',
    ]);

    const slerpSearch = filterDoryDiscrepancyLedger({ query: 'slerp' });
    expect(slerpSearch.rows.some((r) => r.id === 'DR7')).toBe(true);

    const mseSearch = filterDoryDiscrepancyLedger({ query: 'e8m0' });
    expect(mseSearch.rows.map((r) => r.id)).toEqual(['DR15']);

    const byIds = filterDoryDiscrepancyLedger({ ids: ['DR14', 'DR23'] });
    expect(byIds.rows.map((r) => r.id)).toEqual(['DR14', 'DR23']);

    const specCategory = filterDoryDiscrepancyLedger({ category: 'speculative' });
    expect(specCategory.rows.map((r) => r.id)).toEqual(['DR23']);
  });
});

// ============================================================================
// 4. VIEW 7 (`bookendsScene.ts`) — FINITE GEOMETRY, <= 512 KiB ATLAS, 0 OVERLAPS & TBPTL
// ============================================================================

describe('Gate 5 View 7 (bookendsScene.ts) — Untied Bookends, 4-Stage Pipeline, Read-Only TBPTL & DR21 Picking', () => {
  it('builds finite, 0-overlap, <= 128 KiB (<= 512 KiB) RG32F scenes across standard, last_loop & Contract A self_spec with exact untied & 4-stage math', () => {
    // 1. Standard k=2 per_loop (prefill + decode step)
    const stdSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const decodeRec = stdSession.step();
    const stdBefore = captureExternalSessionBytes(stdSession);

    const sceneA = buildDoryBookendsScene(stdSession, {
      selectedCheckpointFormat: 'bf16',
    });
    const sceneARepeat = buildDoryBookendsScene(stdSession, {
      selectedCheckpointFormat: 'bf16',
    });
    const sceneNvfp4 = buildDoryBookendsScene(stdSession, {
      selectedCheckpointFormat: 'nvfp4',
      selectedVocabTokenId: 12, // <think>
    });

    assertGate5SceneHealth(sceneA, 'bookends.std.bf16', DORY_BOOKENDS_ATLAS_BUDGET_BYTES);
    assertGate5SceneHealth(sceneNvfp4, 'bookends.std.nvfp4', DORY_BOOKENDS_ATLAS_BUDGET_BYTES);

    // Exact atlas dimensions 128 x 80 = 81,920 B (80.00 KiB <= 128 KiB <= 512 KiB)
    expect(sceneA.valueAtlas!.width).toBe(DORY_BOOKENDS_ATLAS_WIDTH);
    expect(sceneA.valueAtlas!.height).toBe(DORY_BOOKENDS_ATLAS_HEIGHT);
    expect(sceneA.atlasAudit!.byteLength).toBe(81_920);
    expect(sceneA.atlasKey).toBe(sceneARepeat.atlasKey);
    expect(sceneA.atlasKey).not.toBe(sceneNvfp4.atlasKey);
    expect(
      computeDoryAtlasContentKey(
        'bookends',
        sceneA.valueAtlas!,
        `b=${sceneA.branchId}:r=${sceneA.recordId}:pos=${sceneA.bookendsMetadata!.selectedTokenPos}:v=${sceneA.bookendsMetadata!.selectedVocabTokenId}:ckpt=bf16`,
      ),
    ).toBe(sceneA.atlasKey);

    // Zero mutation of session during View 7 build (read-only LRU trace lookups allowed)
    assertExternalSessionUntouched(stdSession, stdBefore, 'bookends.std', {
      allowLruReadHits: true,
    });

    // Verify metadata invariants (Read-Only TBPTL, Zero Fake Quant Toggles, 0 B Diff, Untied)
    const m = sceneA.bookendsMetadata!;
    expect(m.readOnly).toBe(true);
    expect(m.hasGradients).toBe(false);
    expect(m.zeroFakeQuantToggles).toBe(true);
    expect(m.tbptlBadgeLabel).toBe(DORY_TBPTL_READONLY_BADGE);
    expect(m.untiedVerified).toBe(true);
    expect(m.safetensorsZeroByteDiffVerified).toBe(true);
    expect(m.tbptlDepth).toBe(1);
    expect(m.tbptlStopGradientActive).toBe(true);
    expect(m.tbptlEarlyExitLosses).toBe(0);
    expect(m.tbptlPreludeGradientViaFinalAnchorOnly).toBe(true);
    expect(m.checkpointFormats).toEqual(DORY_CHECKPOINT_FORMATS);

    // Constructive untied verification & unit-norm embedding check across all V=64 rows (DR1, DR5)
    const expectedUntiedDiff = computeUntiedBookendsMaxAbsDiff(PARAMS.embeddings, PARAMS.lmHead);
    expect(expectedUntiedDiff).toBeGreaterThan(0.1);
    expect(m.embeddingVsLmHeadMaxAbsDiff).toBe(expectedUntiedDiff);
    expect(m.embeddingRowL2Norm).toBeCloseTo(1.0, 4);
    for (let v = 0; v < PARAMS.config.vocabSize; v += 1) {
      expect(vectorL2Norm(PARAMS.embeddings, v * PARAMS.config.hiddenSize, PARAMS.config.hiddenSize)).toBeCloseTo(
        1.0,
        5,
      );
    }
    expect(m.logitScaleMultiplier).toBeCloseTo(1 / 0.0177, 10);

    // Exact 4-stage LM Head atlas readback verification (Raw -> Eff Scale -> Scaled -> Softmax)
    const ctx = resolveDorySessionContext(stdSession, decodeRec.recordId);
    const fwd = ctx.resolveStepTrace().forwardResult!;
    const expectedEffScale = effectiveLogitScale(PARAMS.logitScale, PARAMS.config);
    const expectedProbs = softmaxFloat64(fwd.scaledLogits, 0, PARAMS.config.vocabSize, 1.0);
    const expectedProbsF32 = new Float32Array(PARAMS.config.vocabSize);
    for (let v = 0; v < expectedProbsF32.length; v += 1) {
      expectedProbsF32[v] = expectedProbs[v]!;
    }

    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_embed_weight'),
      PARAMS.embeddings,
      'bookends.atlas.embeddings',
    );
    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_lm_head_weight'),
      PARAMS.lmHead,
      'bookends.atlas.lmHead',
    );
    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_raw_logits'),
      fwd.rawLogits,
      'bookends.atlas.rawLogits',
    );
    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_logit_scale_raw'),
      PARAMS.logitScale,
      'bookends.atlas.logitScaleRaw',
    );
    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_logit_scale_eff'),
      expectedEffScale,
      'bookends.atlas.logitScaleEff',
    );
    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_scaled_logits'),
      fwd.scaledLogits,
      'bookends.atlas.scaledLogits',
    );
    assertBitExactFiniteArray(
      readAtlasRegionR(sceneA, 'bookend_softmax_probs'),
      expectedProbsF32,
      'bookends.atlas.softmaxProbs',
    );

    // Top-12 stage rows sorted descending by scaledLogit & exact stage multiplication
    expect(m.topLogitRows).toHaveLength(DORY_BOOKENDS_TOP_LOGIT_COUNT);
    expect(m.topLogitRows[0]!.isArgMax).toBe(true);
    expect(m.topLogitRows[0]!.tokenId).toBe(m.argMaxTokenId);
    for (let i = 0; i < m.topLogitRows.length; i += 1) {
      const row = m.topLogitRows[i]!;
      expect(row.effectiveLogitScale).toBeCloseTo(row.rawLogitScale / 0.0177, 4);
      expect(row.scaledLogit).toBeCloseTo(row.rawLogit * row.effectiveLogitScale, 4);
      expect(row.prob).toBeCloseTo(expectedProbs[row.tokenId]!, 10);
      if (i > 0) {
        expect(m.topLogitRows[i - 1]!.scaledLogit).toBeGreaterThanOrEqual(row.scaledLogit);
      }
    }

    // Mandatory READ-ONLY · NO GRADIENTS badges, landmarks & callouts
    expect(sceneA.provenanceBadges.some((b) => b.label === DORY_TBPTL_READONLY_BADGE)).toBe(true);
    expect(sceneA.overlayAnchors.some((a) => a.badge === DORY_TBPTL_READONLY_BADGE)).toBe(true);
    expect(sceneA.callouts.some((c) => c.badge === DORY_TBPTL_READONLY_BADGE)).toBe(true);
    expect(
      sceneA.landmarks.some((l) => l.id === 'lm_bookends_tbptl' && l.label.includes(DORY_TBPTL_READONLY_BADGE)),
    ).toBe(true);

    // 2. All 4 checkpoint formats render cleanly; invalid format throws RangeError
    for (const fmt of ['bf16', 'mxfp8', 'nvfp4', 'q5_k_m'] as const) {
      const sc = buildDoryBookendsScene(stdSession, { selectedCheckpointFormat: fmt });
      assertGate5SceneHealth(sc, `bookends.fmt.${fmt}`, DORY_BOOKENDS_ATLAS_BUDGET_BYTES);
      expect(sc.bookendsMetadata!.selectedCheckpointFormat).toBe(fmt);
    }
    expect(() =>
      buildDoryBookendsScene(stdSession, {
        selectedCheckpointFormat: 'int2' as DoryCheckpointFormatId,
      }),
    ).toThrow(RangeError);

    // 3. Contract A self_spec (BOTH spec_prefill AND spec_cycle paths!) & last_loop chunked
    const specSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'self_spec',
        speculativeGamma: 4,
        sampling: { temperature: 0.2, topK: 0, topP: 0.95 },
        initialSampleUniform: 0.5,
      },
    });
    const specPrefillRec = specSession.getActiveRecord()!;
    const specCycleRec = specSession.step();

    const specPrefillScene = buildDoryBookendsScene(specSession, {
      recordId: specPrefillRec.recordId,
    });
    const specCycleScene = buildDoryBookendsScene(specSession, {
      recordId: specCycleRec.recordId,
    });
    assertGate5SceneHealth(
      specPrefillScene,
      'bookends.spec_prefill',
      DORY_BOOKENDS_ATLAS_BUDGET_BYTES,
    );
    assertGate5SceneHealth(
      specCycleScene,
      'bookends.spec_cycle',
      DORY_BOOKENDS_ATLAS_BUDGET_BYTES,
    );
    expect(specCycleScene.executionMode).toBe('self_spec');
    expect(specCycleScene.bookendsMetadata!.untiedVerified).toBe(true);

    // Verify self_spec LM head logits match direct runDoryLmHead on the reconstructed Coda state
    const specCtx = resolveDorySessionContext(specSession, specCycleRec.recordId);
    const specTrace = specCtx.resolveForwardTrace();
    const lastCodaOut = specTrace.executedSublayers[specTrace.executedSublayers.length - 1]!.trace.nlerp.output;
    const directHead = runDoryLmHead(lastCodaOut, specTrace.seqLen, PARAMS);
    const selOff = specCycleScene.bookendsMetadata!.selectedTokenOffset;
    const expectedScaledSlice = directHead.scaledLogits.subarray(
      selOff * PARAMS.config.vocabSize,
      (selOff + 1) * PARAMS.config.vocabSize,
    );
    expect(
      computeStrictMaxAbsDiff(
        readAtlasRegionR(specCycleScene, 'bookend_scaled_logits'),
        expectedScaledSlice,
        'bookends.spec_cycle.scaledLogits',
      ),
    ).toBeLessThanOrEqual(1e-6);
  });

  it('supports DR21 3D raycast picking (pickMacroBay -> pickHitToDoryCell) and accurate read-only popovers across sheets, bars, 4-format pedestals & TBPTL wall', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: { promptTokens: PROMPT_12, kLoops: 2, kvCacheMode: 'per_loop' },
    });
    const baseScene = buildDoryBookendsScene(session, { selectedCheckpointFormat: 'mxfp8' });

    const pickAndHover = (partId: string, row = 0, col = 0) => {
      const box = baseScene.pickBoxes.find((pb) => pb.partId === partId);
      expect(box, `Missing pickBox '${partId}'`).toBeDefined();
      const hit = pickMacroBay(makeDirectRayAtBox(box!, row, col), baseScene.pickBoxes, []);
      expect(hit, `Raycast missed '${partId}'`).not.toBeNull();
      expect(hit!.partId).toBe(partId);
      const cell = pickHitToDoryCell(hit!, 'bookends');
      expect(cell).not.toBeNull();
      expect(cell!.partId).toBe(partId);
      return buildDoryBookendsScene(session, {
        selectedCheckpointFormat: 'mxfp8',
        hoveredCell: cell,
      });
    };

    // 1. Untied embedding sheet cell
    const embHover = pickAndHover('bookend_embed_weight', 12, 3);
    expect(embHover.activeCellPopover).not.toBeNull();
    expect(embHover.activeCellPopover!.tensorLabel).toContain('backbone.embeddings.weight[12, d3]');
    expect(embHover.activeCellPopover!.badge).toContain('UNTIED');

    // 2. Untied LM head sheet cell
    const lmHover = pickAndHover('bookend_lm_head_weight', 12, 3);
    expect(lmHover.activeCellPopover!.badge).toContain('tie_word_embeddings: false');

    // 3. Scaled logits & softmax strips
    const scaledHover = pickAndHover('bookend_scaled_logits', 0, 5);
    expect(scaledHover.activeCellPopover!.badge).toContain('STAGE 3');
    const probHover = pickAndHover('bookend_softmax_probs', 0, 5);
    expect(probHover.activeCellPopover!.badge).toContain('STAGE 4');
    expect(probHover.activeCellPopover!.valueText).toMatch(/^p = \d+\.\d{4}$/);

    // 4. Top-12 stage matrix & raised bar 0
    const top12Hover = pickAndHover('bookend_top12_stages', 2, 0);
    expect(top12Hover.activeCellPopover!.badge).toBe('4-STAGE UNTIED LM HEAD PIPELINE');
    const bar0Hover = pickAndHover('bookend_top_bar_0');
    expect(bar0Hover.activeCellPopover!.badge).toContain('ARGMAX');

    // 5. Read-only 4-checkpoint pedestal (zero fake quant toggles)
    const ckptHover = pickAndHover('bookend_ckpt_nvfp4');
    expect(ckptHover.activeCellPopover!.badge).toBe('READ-ONLY METADATA · ZERO FAKE QUANT TOGGLES');
    expect(ckptHover.activeCellPopover!.coordLabel).toContain('0 B DIFF VERIFIED');

    // 6. Strictly READ-ONLY · NO GRADIENTS TBPTL pedestals
    const stopGradHover = pickAndHover('bookend_tbptl_stop_grad');
    expect(stopGradHover.activeCellPopover!.badge).toBe(DORY_TBPTL_READONLY_BADGE);
    expect(stopGradHover.activeCellPopover!.valueText).toBe('NO GRADIENTS EXECUTED');
    expect(stopGradHover.activeCellPopover!.equation).toContain('stop_grad');

    const anchorGradHover = pickAndHover('bookend_tbptl_anchor_grad');
    expect(anchorGradHover.activeCellPopover!.badge).toBe(DORY_TBPTL_READONLY_BADGE);
    expect(anchorGradHover.activeCellPopover!.equation).toContain('+x');

    // 7. Backbone bridge prism & exhaustive check across EVERY pickBox in View 7
    const bridgeHover = pickAndHover('bookend_backbone_bridge');
    expect(bridgeHover.activeCellPopover).not.toBeNull();
    expect(bridgeHover.activeCellPopover!.badge).toContain('72 PHYSICAL SUBLAYERS');
    expect(bridgeHover.activeCellPopover!.equation).toContain('112L');

    for (const box of baseScene.pickBoxes) {
      expect(box.partId, 'Every View 7 PickBox must declare a partId').toBeDefined();
      const hoverScene = buildDoryBookendsScene(session, {
        selectedCheckpointFormat: 'mxfp8',
        hoveredCell: {
          viewMode: 'bookends',
          physicalLayerIdx: null,
          pairedLayerIdx: null,
          executedLayerIdx: null,
          stage: 'bookend',
          sublayerKind: null,
          loopIteration: null,
          partId: box.partId!,
          tensorKey: box.tensorKey ?? null,
          qHeadIndex: null,
          kvHeadIndex: null,
          rotaryPlane: null,
          kvBankSlotId: null,
          kvBankKind: null,
          kvPolicy: null,
          cacheRole: null,
          specOffset: null,
          tokenIndex: null,
          absTokenPos: null,
          row: 0,
          col: 0,
          modelWeightId: PARAMS.modelWeightId,
          semanticTag: box.semanticTag ?? null,
        },
      });
      expect(
        hoverScene.activeCellPopover,
        `View 7 PickBox '${box.partId}' must return a non-null hover popover`,
      ).not.toBeNull();
    }
  });
});

// ============================================================================
// 5. UNIFIED 7-VIEW DISPATCHER (`src/dory/scene/index.ts`)
// ============================================================================

describe('Gate 5 Unified 7-View Dispatcher (`src/dory/scene/index.ts`) — buildDorySpatialScene Across All 7 Views', () => {
  it('validates exhaustive 7-view registry, descriptors, focus resolution/clamping & view option projection', () => {
    const expectedIds: readonly DorySpatialViewMode[] = [
      'macro',
      'hypersphere',
      'gqa_rope',
      'swiglu',
      'kv_compare',
      'self_spec',
      'bookends',
    ];
    expect(DORY_SPATIAL_VIEW_MODES.map((v) => v.id)).toEqual(expectedIds);
    expect(Object.keys(DORY_SCENE_BUILDERS)).toEqual(expectedIds);

    for (let i = 0; i < expectedIds.length; i += 1) {
      const id = expectedIds[i]!;
      const desc = getDorySpatialViewDescriptor(id);
      expect(desc.viewNumber).toBe(i + 1);
      expect(desc.atlasBudgetBytes).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
      expect(typeof DORY_SCENE_BUILDERS[id]).toBe('function');
    }
    expect(() => getDorySpatialViewDescriptor('invalid_view' as DorySpatialViewMode)).toThrow(
      RangeError,
    );

    // Default identity & partial focus clamping/null preservation
    expect(resolveDorySpatialFocus()).toBe(DEFAULT_DORY_SPATIAL_FOCUS_STATE);
    const clampedLo = resolveDorySpatialFocus({
      swigluAccumReveal: -0.5,
      selectedTokenOffset: null,
      selectedVocabTokenId: null,
      macroDisplayMode: 'unrolled',
    });
    expect(clampedLo.swigluAccumReveal).toBe(0);
    expect(clampedLo.selectedTokenOffset).toBeNull();
    expect(clampedLo.selectedVocabTokenId).toBeNull();
    expect(clampedLo.macroDisplayMode).toBe('unrolled');

    expect(resolveDorySpatialFocus({ swigluAccumReveal: 3.5 }).swigluAccumReveal).toBe(1);
    expect(resolveDorySpatialFocus({ swigluAccumReveal: Number.NaN }).swigluAccumReveal).toBe(
      DEFAULT_DORY_SPATIAL_FOCUS_STATE.swigluAccumReveal,
    );

    // Pure option projection for all 7 views
    const customFocus: DorySpatialFocusState = resolveDorySpatialFocus({
      macroDisplayMode: 'unrolled',
      selectedPhysicalLayerIdx: 22,
      selectedLoopIteration: 2,
      selectedTokenOffset: 3,
      gqaLayerIdx: 22,
      swigluLayerIdx: 17,
      swigluAccumReveal: 0.5,
      kvComparePartitionMode: 'one_by_one',
      selectedVerifyOffset: 1,
      selectedCheckpointFormat: 'nvfp4',
      selectedVocabTokenId: 10,
      recordId: 'rec-custom',
    });
    expect(focusStateToViewOptions('macro', customFocus).displayMode).toBe('unrolled');
    expect(focusStateToViewOptions('hypersphere', customFocus).tokenOffset).toBe(3);
    expect(focusStateToViewOptions('gqa_rope', customFocus).physicalLayerIdx).toBe(22);
    expect(focusStateToViewOptions('swiglu', customFocus).accumReveal).toBe(0.5);
    expect(focusStateToViewOptions('kv_compare', customFocus).partitionMode).toBe('one_by_one');
    expect(focusStateToViewOptions('self_spec', customFocus).selectedVerifyOffset).toBe(1);
    expect(focusStateToViewOptions('bookends', customFocus).selectedCheckpointFormat).toBe('nvfp4');
    expect(focusStateToViewOptions('bookends', customFocus).selectedTokenOffset).toBe(3);
    expect(focusStateToViewOptions('bookends', customFocus).selectedTokenPos).toBeUndefined();
    expect(() =>
      focusStateToViewOptions('bogus' as DorySpatialViewMode, customFocus),
    ).toThrow(RangeError);
  });

  it('dispatches ALL 7 views cleanly via buildDorySpatialScene across standard k=4, last_loop chunked & Contract A self_spec with zero session mutation', () => {
    const stdSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const decodeRec = stdSession.step();

    const lastLoopSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        prefillChunkSizes: [4, 8],
        kLoops: 4,
        kvCacheMode: 'last_loop',
        executionMode: 'standard',
      },
    });

    const specSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'self_spec',
        speculativeGamma: 4,
        sampling: { temperature: 0.2, topK: 0, topP: 0.95 },
        initialSampleUniform: 0.5,
      },
    });
    specSession.step();

    const stdBefore = captureExternalSessionBytes(stdSession);
    const lastBefore = captureExternalSessionBytes(lastLoopSession);
    const specBefore = captureExternalSessionBytes(specSession);

    const sessions = [
      { name: 'std_k4_decode', session: stdSession },
      { name: 'last_loop_chunked', session: lastLoopSession },
      { name: 'self_spec_cycle', session: specSession },
    ] as const;

    for (const { name, session } of sessions) {
      const sharedCtx = resolveDorySessionContext(session);
      for (const viewDesc of DORY_SPATIAL_VIEW_MODES) {
        const scene = buildDorySpatialScene(sharedCtx, viewDesc.id, {
          macroDisplayMode: 'unrolled',
          selectedLoopIteration: 1,
          selectedCheckpointFormat: 'mxfp8',
        });
        expect(scene.viewMode).toBe(viewDesc.id);
        expect(scene.modelWeightId).toBe(PARAMS.modelWeightId);
        assertGate5SceneHealth(scene, `dispatch.${name}.${viewDesc.id}`, viewDesc.atlasBudgetBytes);

        // Verify view-specific metadata payload is present on every dispatched scene
        switch (viewDesc.id) {
          case 'macro':
            expect(scene.macroMetadata).not.toBeNull();
            break;
          case 'hypersphere':
            expect(scene.hypersphereMetadata).not.toBeNull();
            break;
          case 'gqa_rope':
            expect(scene.gqaRopeMetadata).not.toBeNull();
            break;
          case 'swiglu':
            expect(scene.swigluMetadata).not.toBeNull();
            break;
          case 'kv_compare':
            expect(scene.kvCompareMetadata).not.toBeNull();
            break;
          case 'self_spec':
            expect(scene.selfSpecMetadata).not.toBeNull();
            break;
          case 'bookends':
            expect(scene.bookendsMetadata).not.toBeNull();
            break;
        }
      }
    }

    // Verify bookends selectedTokenOffset matches identically via BOTH buildDorySpatialScene AND
    // direct DORY_SCENE_BUILDERS.bookends(ctx, focusStateToViewOptions('bookends', focus)) on
    // non-zero startPos records (decode_step AND multi-token chunk / spec_cycle!)
    const decodeBookends = buildDorySpatialScene(stdSession, 'bookends', {
      selectedTokenOffset: 0,
      recordId: decodeRec.recordId,
    });
    expect(decodeBookends.bookendsMetadata!.selectedTokenPos).toBe(decodeRec.startPos);
    expect(decodeBookends.bookendsMetadata!.selectedTokenOffset).toBe(0);

    const lastLoopChunkRec = lastLoopSession.getActiveRecord()!; // chunk [4..11], startPos = 4, seqLen = 8
    expect(lastLoopChunkRec.startPos).toBe(4);
    const chunkFocus = resolveDorySpatialFocus({
      selectedTokenOffset: 3,
      recordId: lastLoopChunkRec.recordId,
    });
    const chunkCtx = resolveDorySessionContext(lastLoopSession, lastLoopChunkRec.recordId);
    const viaDispatcher = buildDorySpatialScene(chunkCtx, 'bookends', chunkFocus);
    const viaDirectBuilder = DORY_SCENE_BUILDERS.bookends(
      chunkCtx,
      focusStateToViewOptions('bookends', chunkFocus),
    );
    expect(viaDispatcher.bookendsMetadata!.selectedTokenOffset).toBe(3);
    expect(viaDispatcher.bookendsMetadata!.selectedTokenPos).toBe(7);
    expect(viaDirectBuilder.bookendsMetadata!.selectedTokenOffset).toBe(3);
    expect(viaDirectBuilder.bookendsMetadata!.selectedTokenPos).toBe(7);
    expect(viaDirectBuilder.atlasKey).toBe(viaDispatcher.atlasKey);

    // Invalid viewMode throws RangeError
    expect(() =>
      buildDorySpatialScene(stdSession, 'nonexistent' as DorySpatialViewMode),
    ).toThrow(RangeError);

    // Zero session mutation across ALL 3 sessions after dispatching all 21 scenes
    assertExternalSessionUntouched(stdSession, stdBefore, 'dispatch.std', {
      allowLruReadHits: true,
    });
    assertExternalSessionUntouched(lastLoopSession, lastBefore, 'dispatch.last_loop', {
      allowLruReadHits: true,
    });
    assertExternalSessionUntouched(specSession, specBefore, 'dispatch.self_spec', {
      allowLruReadHits: true,
    });
  });
});

// ============================================================================
// 6. 4-ACT GUIDED WALKTHROUGH — BEATS, 8-TUPLE COORDS, CAPS & ISOLATION PROOF
// ============================================================================

describe('Gate 5 4-Act Guided Walkthrough (`src/dory/story/walkthrough.ts`) — 8-Tuple Coords, Memory/Latency Caps & External Session Isolation', () => {
  it('validates 4-Act, 14-Beat (3+3+4+4) catalog integrity covering all 7 D7 views and all 5 isolated scenarios', () => {
    expect(DORY_TOUR_TOTAL_ACTS).toBe(4);
    expect(DORY_TOUR_ACT_DESCRIPTORS).toHaveLength(4);
    expect(DORY_TOUR_BEATS).toHaveLength(14);
    expect(DORY_TOUR_PROMPT_TOKENS.length).toBeGreaterThan(DORY_MINI_CONFIG.slidingWindow);

    const expectedCounts = [3, 3, 4, 4] as const;
    const seenBeatIds = new Set<string>();
    const seenViews = new Set<DorySpatialViewMode>();
    const seenScenarios = new Set<DoryTourScenarioKey>();

    let cursor = 0;
    for (let a = 0; a < DORY_TOUR_ACT_DESCRIPTORS.length; a += 1) {
      const act = DORY_TOUR_ACT_DESCRIPTORS[a]!;
      expect(act.actIndex).toBe(a);
      expect(act.actNumber).toBe(a + 1);
      expect(act.id).toBe(`act${a + 1}`);

      const actBeats = DORY_TOUR_BEATS.filter((b) => b.actId === act.id);
      expect(actBeats.length).toBe(expectedCounts[a]);
      expect(actBeats.length).toBeGreaterThanOrEqual(DORY_TOUR_MIN_BEATS_PER_ACT);

      const actViewsSeen = new Set<DorySpatialViewMode>();
      for (let j = 0; j < actBeats.length; j += 1) {
        const beat = actBeats[j]!;
        expect(beat.index).toBe(cursor);
        expect(beat.beatInAct).toBe(j + 1);
        expect(beat.actIndex).toBe(a);
        expect(beat.actNumber).toBe(a + 1);
        expect(seenBeatIds.has(beat.id)).toBe(false);
        seenBeatIds.add(beat.id);

        expect(beat.title.length).toBeGreaterThan(5);
        expect(beat.subtitle.length).toBeGreaterThan(5);
        expect(beat.takeaway.length).toBeGreaterThan(10);
        expect(beat.narrative.length).toBeGreaterThan(30);
        expect(beat.epistemicNote.length).toBeGreaterThan(10);
        expect(beat.discrepancyRefs.length).toBeGreaterThan(0);
        expect(beat.designDecisionRefs.length).toBeGreaterThan(0);
        expect(act.views).toContain(beat.viewMode);

        actViewsSeen.add(beat.viewMode);
        seenViews.add(beat.viewMode);
        seenScenarios.add(beat.scenarioKey);
        cursor += 1;
      }

      // Every view listed on the act descriptor is visited in that act
      for (const expectedView of act.views) {
        expect(actViewsSeen.has(expectedView), `${act.id} must visit ${expectedView}`).toBe(true);
      }
    }
    expect(cursor).toBe(DORY_TOUR_BEATS.length);

    // All 7 D7 views and all 5 isolated tour scenarios are covered
    for (const viewDesc of DORY_SPATIAL_VIEW_MODES) {
      expect(seenViews.has(viewDesc.id), `Tour must cover view '${viewDesc.id}'`).toBe(true);
    }
    const allScenarios: readonly DoryTourScenarioKey[] = [
      'standard_k1_per_loop',
      'standard_k2_per_loop',
      'standard_k4_per_loop',
      'standard_k4_last_loop_chunked',
      'self_spec_contract_a',
    ];
    for (const key of allScenarios) {
      expect(seenScenarios.has(key), `Tour must cover scenario '${key}'`).toBe(true);
    }
  });

  it('executes full 4-act walkthrough (all 14 beats + all 14 live 3D scenes) within <= 12 MiB memory & latency caps, verifying 8-tuple coordinates, epistemic badges & determinism', () => {
    const startMs = performance.now();
    const tour = createDoryGuidedTour({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      seed: DORY_TOUR_SEED,
      sceneDispatcher: buildDorySpatialScene,
    });

    const allBeats = tour.getAllBeats();
    expect(allBeats).toHaveLength(14);

    const scenes: DorySpatialScene[] = [];
    for (let i = 0; i < allBeats.length; i += 1) {
      const beat = allBeats[i]!;
      expect(beat.beatIndex).toBe(i);
      expect(beat.totalBeats).toBe(14);
      expect(beat.liveMetrics.length).toBeGreaterThanOrEqual(4);
      expect(beat.provenanceBadges.length).toBeGreaterThanOrEqual(2);

      // Exhaustive 8-tuple DoryExecutionCoord validation on every beat
      const coordCheck = validateDoryExecutionCoord(beat.coord, PARAMS.modelWeightId, beat.kLoops);
      expect(
        coordCheck.valid,
        `Beat #${i} (${beat.beatId}) coord errors: ${coordCheck.errors.join(' | ')}`,
      ).toBe(true);
      expect(beat.coord.modelWeightId).toBe(PARAMS.modelWeightId);
      expect(beat.coord.sessionBranchId).toBe(beat.branchId);

      // Build and audit live 5-pass WebGL2 spatial scene for every beat
      const scene = tour.buildBeatScene(i);
      scenes.push(scene);
      expect(scene.viewMode).toBe(beat.viewMode);
      expect(scene.modelWeightId).toBe(PARAMS.modelWeightId);
      assertGate5SceneHealth(scene, `tour.beat[${i}].${beat.beatId}`);

      // Every beat's highlightPartId (and hoveredPartId) must resolve to an actual PickBox in scene.pickBoxes
      if (beat.highlightPartId !== null) {
        expect(
          scene.pickBoxes.some((pb) => pb.partId === beat.highlightPartId),
          `Beat #${i} (${beat.beatId}) highlightPartId='${beat.highlightPartId}' missing from scene.pickBoxes`,
        ).toBe(true);
      }
      if (beat.focus.hoveredPartId !== null) {
        expect(
          scene.pickBoxes.some((pb) => pb.partId === beat.focus.hoveredPartId),
          `Beat #${i} (${beat.beatId}) focus.hoveredPartId='${beat.focus.hoveredPartId}' missing from scene.pickBoxes`,
        ).toBe(true);
      }
    }

    const elapsedMs = performance.now() - startMs;
    // Generous CI latency cap (typical local run ~100–400 ms; strictly bounded well below Vitest timeout)
    expect(elapsedMs).toBeLessThan(8_000);

    // D10 bounded memory cap on the isolated private tourSession
    const tourMem = tour.measureTourMemory();
    expect(tourMem.branchCount).toBe(5);
    expect(tourMem.branchCount).toBeLessThanOrEqual(12);
    expect(tourMem.recordCount).toBeLessThanOrEqual(128);
    expect(tourMem.singletonWeightBytes).toBe(DORY_MINI_WEIGHT_BYTES);
    expect(tourMem.withinCheckpointPoolBudget).toBe(true);
    expect(tourMem.withinSessionBudget).toBe(true);
    expect(tourMem.withinTotalBudgetWithWeights).toBe(true);
    expect(tourMem.nonAliasedSlabsVerified).toBe(true);
    expect(tourMem.totalSessionRetainedBytes).toBeLessThanOrEqual(DORY_MAX_SESSION_RETAINED_BYTES);
    expect(tourMem.totalWithWeightsBytes).toBeLessThanOrEqual(DORY_MAX_SESSION_RETAINED_BYTES);

    // Key pedagogical, coordinate & epistemic landmark spot-checks across Acts I–IV:
    // Act I: untied bookends (beat 0), folded 72L (beat 1), anti-SLERP NLERP (beat 2)
    expect(allBeats[0]!.coord.physicalLayerIdx).toBe(0);
    expect(allBeats[0]!.coord.cacheOwnership.bankSlotId).toBe(0);
    expect(scenes[0]!.bookendsMetadata!.untiedVerified).toBe(true);
    expect(allBeats[2]!.provenanceBadges.some((b) => b.label === DORY_ANTI_SLERP_PLAQUE)).toBe(true);

    // Act II: GQA SWA (beat 3), Full Dual-RoPE (beat 4), SwiGLU MLP bankSlotId === -1 (beat 5)
    expect(scenes[3]!.gqaRopeMetadata!.attentionKind).toBe('sliding_attention');
    expect(scenes[4]!.gqaRopeMetadata!.attentionKind).toBe('full_attention');
    expect(allBeats[5]!.coord.physicalLayerIdx).toBe(17);
    expect(allBeats[5]!.coord.cacheOwnership.bankSlotId).toBe(-1);

    // Act III: Unrolled k=4 e=136 (beat 6), PCA + 120° caveat + counterfactual Coda (beat 7),
    //          last_loop chunked [4,8] divergence > 1e-2 (beat 8), S-invariant 256K scaling (beat 9)
    expect(allBeats[6]!.coord.executedLayerIdx).toBe(136);
    expect(allBeats[7]!.provenanceBadges.some((b) => b.label === DORY_EXPLORATORY_PCA_BADGE)).toBe(
      true,
    );
    expect(allBeats[7]!.provenanceBadges.some((b) => b.label === DORY_ANCHOR_MAP_CAVEAT_BADGE)).toBe(
      true,
    );
    expect(
      allBeats[7]!.provenanceBadges.some((b) => b.label === DORY_COUNTERFACTUAL_CODA_BADGE),
    ).toBe(true);
    expect(scenes[7]!.hypersphereMetadata!.counterfactualCoda?.readOnlyZeroKvMutation).toBe(true);
    // Beat 7 anchor exit angle theta_4 matches arccos(cos(s_4, x)), while Jacobian labels entry theta_entry4
    const beat7Loop4 = scenes[7]!.hypersphereMetadata!.loopMetrics8d[3]!;
    const expectedExitTheta4Deg =
      (Math.acos(Math.max(-1, Math.min(1, beat7Loop4.anchorCosine))) * 180) / Math.PI;
    const beat7AnchorMetric = allBeats[7]!.liveMetrics.find((m) =>
      m.label.includes('cos(s_4, x)'),
    )!;
    const beat7JacMetric = allBeats[7]!.liveMetrics.find((m) => m.label.includes('||J_N||_2'))!;
    expect(beat7AnchorMetric.value).toContain(`${expectedExitTheta4Deg.toFixed(1)} deg`);
    expect(beat7JacMetric.value).toContain(
      `theta_entry4=${beat7Loop4.entryAnchorJacobian.angleDegrees.toFixed(1)} deg`,
    );
    expect(allBeats[8]!.coord.cacheOwnership.policy).toBe('last_loop');
    expect(allBeats[8]!.coord.loopIteration).toBe(1);
    expect(allBeats[8]!.coord.cacheOwnership.writerLoop).toBe(4);
    expect(scenes[8]!.kvCompareMetadata!.theorem3ChunkedVsPerLoopMaxAbsDiff).toBeGreaterThan(1e-2);

    // Act IV: Contract A 104 banks & binary64 rollback (beats 10–11), 0 B diff & READ-ONLY TBPTL (beats 12–13)
    expect(allBeats[10]!.coord.cacheOwnership.role).toBe('spec_verify_reused');
    expect(allBeats[10]!.provenanceBadges.some((b) => b.label === DORY_SPEC_CONTRACT_A_BADGE)).toBe(
      true,
    );
    expect(scenes[10]!.selfSpecMetadata!.totalBankCount).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
    expect(allBeats[11]!.coord.cacheOwnership.bankKind).toBe('output_deep');
    expect(allBeats[11]!.coord.cacheOwnership.role).toBe('spec_verify_deep');
    expect(allBeats[11]!.focus.selectedVerifyOffset).toBe(1);
    expect(scenes[11]!.selfSpecMetadata!.selectedVerifyOffset).toBe(1);
    expect(scenes[11]!.selfSpecMetadata!.positionBars[1]!.absPos).toBe(allBeats[11]!.coord.absTokenPos);
    expect(scenes[11]!.selfSpecMetadata!.positionBars[1]!.accepted).toBe(false);
    expect(allBeats[0]!.highlightPartId).toBe('bookend_embed_weight');
    expect(allBeats[12]!.highlightPartId).toBe('bookend_ckpt_nvfp4');
    expect(allBeats[13]!.highlightPartId).toBe('bookend_tbptl_stop_grad');
    expect(scenes[12]!.bookendsMetadata!.selectedCheckpointFormat).toBe('nvfp4');
    expect(scenes[12]!.bookendsMetadata!.safetensorsZeroByteDiffVerified).toBe(true);
    expect(scenes[13]!.bookendsMetadata!.readOnly).toBe(true);
    expect(scenes[13]!.bookendsMetadata!.hasGradients).toBe(false);
    expect(allBeats[13]!.provenanceBadges.some((b) => b.label === DORY_TBPTL_READONLY_BADGE)).toBe(
      true,
    );
    expect(allBeats[13]!.provenanceBadges.some((b) => b.label === DORY_PAPER_RESULTS_BADGE)).toBe(
      true,
    );

    // Deterministic repeatability across fresh tour instances with same seed
    const twinTour = createDoryGuidedTour({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      seed: DORY_TOUR_SEED,
      sceneDispatcher: buildDorySpatialScene,
    });
    for (let i = 0; i < allBeats.length; i += 1) {
      expect(twinTour.getBeat(i)).toEqual(allBeats[i]);
      expect(twinTour.buildBeatScene(i).atlasKey).toBe(scenes[i]!.atlasKey);
    }

    // Fallback dispatcher (when sceneDispatcher omitted): Views 1–6 succeed, View 7 throws informative Error
    const fallbackTour = createDoryGuidedTour({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      seed: DORY_TOUR_SEED,
    });
    expect(fallbackTour.buildBeatScene(1).viewMode).toBe('macro');
    expect(fallbackTour.buildBeatScene(5).viewMode).toBe('swiglu');
    expect(fallbackTour.buildBeatScene(10).viewMode).toBe('self_spec');
    expect(() => fallbackTour.buildBeatScene(0)).toThrow(/sceneDispatcher/);
  });

  it('rejects malformed 8-tuple DoryExecutionCoord negative controls across all 8 fields & DR23 guards', () => {
    const validCoord: DoryExecutionCoord = {
      modelWeightId: PARAMS.modelWeightId,
      physicalLayerIdx: 16,
      executedLayerIdx: 16,
      loopIteration: 1,
      absTokenPos: 5,
      forwardStepId: 0,
      sessionBranchId: 'b0',
      cacheOwnership: {
        policy: 'per_loop',
        bankKind: 'recurrent',
        bankSlotId: 8,
        writerLoop: 1,
        role: 'standard',
      },
    };
    expect(validateDoryExecutionCoord(validCoord, PARAMS.modelWeightId, 4).valid).toBe(true);

    // 1. Wrong modelWeightId
    expect(
      validateDoryExecutionCoord({ ...validCoord, modelWeightId: 'wrong' }, PARAMS.modelWeightId, 4)
        .valid,
    ).toBe(false);
    // 2. Out-of-range physicalLayerIdx
    expect(validateDoryExecutionCoord({ ...validCoord, physicalLayerIdx: 72 }).valid).toBe(false);
    // 3. Wrong executedLayerIdx
    expect(validateDoryExecutionCoord({ ...validCoord, executedLayerIdx: 99 }, undefined, 4).valid).toBe(
      false,
    );
    // 4. Recurrent loopIteration = 0 or > kLoops
    expect(validateDoryExecutionCoord({ ...validCoord, loopIteration: 0 }, undefined, 4).valid).toBe(
      false,
    );
    expect(validateDoryExecutionCoord({ ...validCoord, loopIteration: 5 }, undefined, 4).valid).toBe(
      false,
    );
    // 5. Out-of-range absTokenPos
    expect(validateDoryExecutionCoord({ ...validCoord, absTokenPos: 32 }).valid).toBe(false);
    // 6. Negative forwardStepId
    expect(validateDoryExecutionCoord({ ...validCoord, forwardStepId: -1 }).valid).toBe(false);
    // 7. Empty sessionBranchId
    expect(validateDoryExecutionCoord({ ...validCoord, sessionBranchId: '  ' }).valid).toBe(false);
    // 8a. DR23 violation: output_shallow or speculative role under last_loop
    const dr23ShallowViolation = validateDoryExecutionCoord(
      {
        ...validCoord,
        physicalLayerIdx: 56,
        executedLayerIdx: 56,
        loopIteration: 0,
        cacheOwnership: {
          policy: 'last_loop',
          bankKind: 'output_shallow',
          bankSlotId: 96,
          writerLoop: 0,
          role: 'spec_draft',
        },
      },
      PARAMS.modelWeightId,
      4,
    );
    expect(dr23ShallowViolation.valid).toBe(false);
    expect(dr23ShallowViolation.errors.some((e) => e.includes('DR23'))).toBe(true);

    // 8b. MLP with non -1 bankSlotId
    expect(
      validateDoryExecutionCoord(
        {
          ...validCoord,
          physicalLayerIdx: 17,
          executedLayerIdx: 17,
          cacheOwnership: { ...validCoord.cacheOwnership, bankSlotId: 8 },
        },
        PARAMS.modelWeightId,
        4,
      ).valid,
    ).toBe(false);

    // 8c. per_loop recurrent writerLoop !== loopIteration
    expect(
      validateDoryExecutionCoord(
        {
          ...validCoord,
          cacheOwnership: { ...validCoord.cacheOwnership, writerLoop: 4 },
        },
        PARAMS.modelWeightId,
        4,
      ).valid,
    ).toBe(false);

    // 8d. last_loop recurrent attention requires writerLoop === loopIteration || writerLoop === kLoops
    const lastLoopAttnValidT = validateDoryExecutionCoord(
      {
        ...validCoord,
        physicalLayerIdx: 16,
        executedLayerIdx: 16,
        loopIteration: 1,
        cacheOwnership: {
          policy: 'last_loop',
          bankKind: 'recurrent',
          bankSlotId: 8,
          writerLoop: 1,
          role: 'standard',
        },
      },
      PARAMS.modelWeightId,
      4,
    );
    expect(lastLoopAttnValidT.valid).toBe(true);

    const lastLoopAttnValidK = validateDoryExecutionCoord(
      {
        ...validCoord,
        physicalLayerIdx: 16,
        executedLayerIdx: 16,
        loopIteration: 1,
        cacheOwnership: {
          policy: 'last_loop',
          bankKind: 'recurrent',
          bankSlotId: 8,
          writerLoop: 4,
          role: 'standard',
        },
      },
      PARAMS.modelWeightId,
      4,
    );
    expect(lastLoopAttnValidK.valid).toBe(true);

    // Intermediate writerLoop (2 when t=1, k=4) must fail under last_loop!
    expect(
      validateDoryExecutionCoord(
        {
          ...validCoord,
          physicalLayerIdx: 16,
          executedLayerIdx: 16,
          loopIteration: 1,
          cacheOwnership: {
            policy: 'last_loop',
            bankKind: 'recurrent',
            bankSlotId: 8,
            writerLoop: 2,
            role: 'standard',
          },
        },
        PARAMS.modelWeightId,
        4,
      ).valid,
    ).toBe(false);

    // 8e. Recurrent MLP requires writerLoop === loopIteration even under last_loop!
    expect(
      validateDoryExecutionCoord(
        {
          ...validCoord,
          physicalLayerIdx: 17,
          executedLayerIdx: 17,
          loopIteration: 1,
          cacheOwnership: {
            policy: 'last_loop',
            bankKind: 'recurrent',
            bankSlotId: -1,
            writerLoop: 4,
            role: 'standard',
          },
        },
        PARAMS.modelWeightId,
        4,
      ).valid,
    ).toBe(false);
    expect(
      validateDoryExecutionCoord(
        {
          ...validCoord,
          physicalLayerIdx: 17,
          executedLayerIdx: 17,
          loopIteration: 1,
          cacheOwnership: {
            policy: 'last_loop',
            bankKind: 'recurrent',
            bankSlotId: -1,
            writerLoop: 1,
            role: 'standard',
          },
        },
        PARAMS.modelWeightId,
        4,
      ).valid,
    ).toBe(true);
  });

  it('supports start, goToBeat, nextBeat, prevBeat, goToAct, boundary clamping & immutable restoreState on exit', () => {
    const tour = createDoryGuidedTour({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      sceneDispatcher: buildDorySpatialScene,
    });

    expect(tour.getStatus()).toBe('idle');
    expect(tour.isActive()).toBe(false);
    expect(tour.getBeatCount()).toBe(14);
    expect(tour.getCurrentBeatIndex()).toBe(0);
    expect(tour.getRestoreState()).toBeNull();

    const mutablePartition = [4, 8];
    const mutableCameraTarget: [number, number, number] = [1, 2, 3];
    const callerRestore: DoryFreeExploreRestoreState = {
      viewMode: 'kv_compare',
      focus: {
        ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
        kvComparePartitionMode: 'chunked',
        kvCompareChunkPartition: mutablePartition,
        selectedCheckpointFormat: 'nvfp4',
      },
      activeBranchId: 'b_user',
      activeRecordId: 'r_user_3',
      cameraPose: {
        azimuthDeg: -24,
        elevationDeg: 28,
        distance: 14,
        fovDeg: 34,
        target: mutableCameraTarget,
      },
      inspectorTab: 'readouts',
      inspectorOpen: true,
    };

    const s0 = tour.start(callerRestore, 0);
    expect(tour.getStatus()).toBe('active');
    expect(tour.isActive()).toBe(true);
    expect(s0.active).toBe(true);
    expect(s0.beatIndex).toBe(0);
    expect(s0.actIndex).toBe(0);
    expect(s0.currentAct.id).toBe('act1');
    expect(s0.currentAct.isCurrent).toBe(true);

    // Mutate caller's original arrays after start() — saved restore snapshot must be deep-cloned!
    mutablePartition[0] = 999;
    mutableCameraTarget[0] = 999;

    // Clamping at lower bound
    expect(tour.prevBeat().beatIndex).toBe(0);
    expect(tour.goToBeat(-50).beatIndex).toBe(0);

    // Stepping forward & jumping directly to each Act start beat
    expect(tour.nextBeat().beatIndex).toBe(1);
    const expectedActStarts: readonly [DoryTourActId, number][] = [
      ['act1', 0],
      ['act2', 3],
      ['act3', 6],
      ['act4', 10],
    ];
    for (const [actId, startIdx] of expectedActStarts) {
      const st = tour.goToAct(actId);
      expect(st.beatIndex).toBe(startIdx);
      expect(st.currentAct.id).toBe(actId);
      expect(st.currentAct.isCurrent).toBe(true);
      expect(st.currentBeat.beatInAct).toBe(1);
    }

    // Clamping at upper bound
    const sLast = tour.goToBeat(999);
    expect(sLast.beatIndex).toBe(13);
    expect(sLast.actIndex).toBe(3);
    expect(tour.nextBeat().beatIndex).toBe(13);
    expect(tour.prevBeat().beatIndex).toBe(12);

    // Range validation
    expect(() => tour.goToAct('act99' as DoryTourActId)).toThrow(RangeError);
    expect(() => tour.getBeat(-1)).toThrow(RangeError);
    expect(() => tour.getBeat(14)).toThrow(RangeError);
    expect(() => tour.goToBeat(Number.NaN)).toThrow(RangeError);

    // Exit returns exact deep-cloned Free-Explore restore snapshot (unaffected by caller mutation)
    const restored = tour.exit();
    expect(tour.getStatus()).toBe('exited');
    expect(tour.isActive()).toBe(false);
    expect(restored).not.toBeNull();
    expect(restored!.viewMode).toBe('kv_compare');
    expect(restored!.activeBranchId).toBe('b_user');
    expect(restored!.activeRecordId).toBe('r_user_3');
    expect(restored!.focus.kvCompareChunkPartition).toEqual([4, 8]);
    expect(restored!.cameraPose!.target).toEqual([1, 2, 3]);
    expect(restored!.inspectorTab).toBe('readouts');
    expect(restored!.inspectorOpen).toBe(true);

    // Re-entering tour after exit() without passing restoreState must clear stale savedRestoreState
    tour.start(undefined, 2);
    expect(tour.getStatus()).toBe('active');
    expect(tour.getCurrentBeatIndex()).toBe(2);
    expect(tour.getRestoreState()).toBeNull();
    expect(tour.exit()).toBeNull();
  });

  it('leaves a populated multi-branch external Free-Explore DorySession byte-for-byte untouched across the entire 4-act walkthrough', () => {
    const buildRichExternalSession = (): {
      readonly session: DorySession;
      readonly stdBranchId: string;
      readonly lastLoopBranchId: string;
      readonly specBranchId: string;
      readonly parkedRecordId: string;
    } => {
      const session = createDorySession({
        params: PARAMS,
        ropeTables: ROPE_TABLES,
        autoStartInitialBranch: false,
      });

      // Branch 1: standard k=2 per_loop stepped twice across W = 8
      const stdPre = session.startBranch({
        label: 'Free-Explore Std k=2',
        promptTokens: PROMPT_12,
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0.7, topK: 0, topP: 0.95 },
        seed: 0x11223344,
      });
      const stdStep1 = session.step({ branchId: stdPre.branchId });
      session.step({ branchId: stdPre.branchId });

      // Branch 2: last_loop chunked [4, 8] k=4 (DR17/DR24) stepped once
      const llPre = session.startBranch({
        label: 'Free-Explore last_loop [4,8]',
        promptTokens: PROMPT_12,
        prefillChunkSizes: [4, 8],
        kLoops: 4,
        kvCacheMode: 'last_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
        seed: 0x55667788,
      });
      session.step({ branchId: llPre.branchId });

      // Branch 3: Contract A self_spec (104 banks) run through 1 stochastic cycle
      const specPre = session.startBranch({
        label: 'Free-Explore Contract A Spec',
        promptTokens: PROMPT_12,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'self_spec',
        speculativeGamma: 4,
        sampling: { temperature: 0.25, topK: 0, topP: 0.95 },
        seed: 0x99aabbcc,
        initialSampleUniform: 0.5,
      });
      session.step({ branchId: specPre.branchId });

      // Park user cursor on a historical non-tip record
      session.seekRecord(stdStep1.recordId);

      return {
        session,
        stdBranchId: stdPre.branchId,
        lastLoopBranchId: llPre.branchId,
        specBranchId: specPre.branchId,
        parkedRecordId: stdStep1.recordId,
      };
    };

    const external = buildRichExternalSession();
    const untouchedTwin = buildRichExternalSession();

    // Capture full byte-exact external session state BEFORE creating or running the tour
    const beforeExternal = captureExternalSessionBytes(external.session);

    const restoreState: DoryFreeExploreRestoreState = Object.freeze({
      viewMode: 'hypersphere',
      focus: resolveDorySpatialFocus({
        selectedPhysicalLayerIdx: 16,
        selectedLoopIteration: 2,
        recordId: external.parkedRecordId,
      }),
      activeBranchId: external.stdBranchId,
      activeRecordId: external.parkedRecordId,
    });

    // Execute full 4-act guided walkthrough end-to-end: start -> step all 14 beats -> build all 14 scenes -> jump acts -> exit
    const tour = createDoryGuidedTour({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      seed: DORY_TOUR_SEED,
      sceneDispatcher: buildDorySpatialScene,
    });
    expect(tour.tourSession).not.toBe(external.session);

    tour.start(restoreState, 0);
    for (let i = 0; i < tour.totalBeats; i += 1) {
      const state = i === 0 ? tour.getState() : tour.nextBeat();
      expect(state.beatIndex).toBe(i);
      const beatScene = tour.buildBeatScene();
      expect(beatScene.viewMode).toBe(state.viewMode);
    }
    tour.goToAct('act2');
    tour.buildBeatScene();
    tour.goToAct('act4');
    tour.buildBeatScene();
    const exitedRestore = tour.exit();
    expect(exitedRestore).toEqual(restoreState);

    // 1. Metadata, LRU stats (hit/miss/eviction), memory audit, and ALL record typed arrays are byte-for-byte untouched
    assertExternalSessionUntouched(external.session, beforeExternal, 'external_after_full_tour');

    // 2. Every branch's live KV cache slabs match the untouched twin bit-for-bit AND share ZERO ArrayBuffers with tourSession
    const tourAllBuffers = new Set<ArrayBufferLike>();
    for (const tourBranch of tour.tourSession.listBranches()) {
      const tourKv = tour.tourSession.reconstructKvCacheAtRecord(tourBranch.headRecordId);
      for (const buf of collectKvArrayBuffers(tourKv)) {
        tourAllBuffers.add(buf);
      }
    }

    for (const branchId of [
      external.stdBranchId,
      external.lastLoopBranchId,
      external.specBranchId,
    ]) {
      const extHeadId = external.session.getBranch(branchId)!.headRecordId;
      const twinHeadId = untouchedTwin.session.getBranch(branchId)!.headRecordId;
      const extKv = external.session.reconstructKvCacheAtRecord(extHeadId);
      const twinKv = untouchedTwin.session.reconstructKvCacheAtRecord(twinHeadId);
      assertKvCachesBitExact(extKv, twinKv, `external_vs_twin.${branchId}`);

      for (const buf of collectKvArrayBuffers(extKv)) {
        expect(tourAllBuffers.has(buf), `KV ArrayBuffer aliased with tourSession on ${branchId}`).toBe(
          false,
        );
      }
    }

    // 3. Stepping every branch of externalSession after tour yields bit-exact identical output to untouchedTwin
    //    (proves internal private PRNG state and branch head KV buffers were 100% untouched!)
    for (const branchId of [
      external.stdBranchId,
      external.lastLoopBranchId,
      external.specBranchId,
    ]) {
      const extNext = external.session.step({ branchId });
      const twinNext = untouchedTwin.session.step({ branchId });
      assertBitExactFiniteArray(
        extNext.emittedTokens,
        twinNext.emittedTokens,
        `post_tour_step.${branchId}.emitted`,
      );
      assertBitExactFiniteArray(
        extNext.lastTokenScaledLogits,
        twinNext.lastTokenScaledLogits,
        `post_tour_step.${branchId}.logits`,
      );
      assertBitExactFiniteArray(
        extNext.lastTokenProbs,
        twinNext.lastTokenProbs,
        `post_tour_step.${branchId}.probs`,
      );
    }
  });
});

// ============================================================================
// 7. D4 ARCHITECTURAL BOUNDARY ISOLATION & CLEAN-ROOM STATIC GUARD
// ============================================================================

describe('Gate 5 D4 Architectural Boundary Isolation & Clean-Room Static Guard', () => {
  it('enforces zero Gate 4b/4c Nano imports, zero sibling-app imports, zero Node built-ins, zero Math.random(), zero DOM/WebGL globals across all headless src/dory/ modules, and untouched src/render/pick.ts', () => {
    const doryRoot = resolve(process.cwd(), 'src/dory');
    const allDoryTsFiles = listTsFilesRecursive(doryRoot);
    expect(allDoryTsFiles.length).toBeGreaterThanOrEqual(20);

    // Ensure all DORY5 files are explicitly present in the audited file list
    const relPaths = allDoryTsFiles.map((p) => relative(process.cwd(), p).replace(/\\/g, '/'));
    for (const required of [
      'src/dory/archReadouts.ts',
      'src/dory/scene/bookendsScene.ts',
      'src/dory/scene/index.ts',
      'src/dory/story/types.ts',
      'src/dory/story/walkthrough.ts',
    ]) {
      expect(relPaths, `Expected DORY5 module '${required}'`).toContain(required);
    }

    const forbiddenResolvedRelRe =
      /^src\/(?:ui\/(?:controls|overlay|memoryChart)|scene\/(?:spine|sheetInspector|hybridPick|cutaways|embeddingCutaway|outputCutaway|gqaCutaway|mambaCutaway|moeCutaway|gqaSpatial|mambaSpatial|moeSpatial|accumulation)|model\/(?:spec|memory|state|rcPlayback)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)|story\/walkthrough|main|twotower(?:\/|$)|diffusion(?:\/|$)|uni(?:\/|$))(?:$|\.)/;

    const importSpecRe = /\b(?:import|export)\s+(?:type\s+)?(?:[\s\S]*?\s+from\s+)?['"]([^'"]+)['"]/g;

    for (const fullPath of allDoryTsFiles) {
      const relPath = relative(process.cwd(), fullPath).replace(/\\/g, '/');
      const src = readFileSync(fullPath, 'utf-8');

      expect(src, `Math.random() forbidden in ${relPath}`).not.toMatch(/Math\.random\s*\(/);
      expect(src, `node: built-in forbidden in ${relPath}`).not.toMatch(/from\s+['"]node:/);

      for (const match of src.matchAll(importSpecRe)) {
        const spec = match[1]!;
        if (!spec.startsWith('.')) {
          expect(spec, `Forbidden bare/node specifier '${spec}' in ${relPath}`).not.toMatch(
            /^node:|^(?:twotower|diffusion|uni)\b/,
          );
          continue;
        }
        const resolvedRel = relative(process.cwd(), resolve(dirname(fullPath), spec)).replace(
          /\\/g,
          '/',
        );
        expect(
          resolvedRel,
          `D4 forbidden import '${spec}' -> '${resolvedRel}' in ${relPath}`,
        ).not.toMatch(forbiddenResolvedRelRe);
      }

      // Headless modules (outside src/dory/app/) must also contain zero DOM/WebGL globals
      if (!relPath.startsWith('src/dory/app/')) {
        expect(src, `DOM/WebGL globals forbidden in headless ${relPath}`).not.toMatch(
          /\b(?:document\s*\.|window\s*\.|globalThis\s*\.\s*(?:document|window)|typeof\s+(?:document|window)\b|WebGL2RenderingContext|HTMLCanvasElement|HTMLElement)\b/,
        );
      }
    }

    // DR21: src/render/pick.ts remains strictly untouched with Nano's `0 | 1` typing
    const pickSrc = readFileSync(resolve(process.cwd(), 'src/render/pick.ts'), 'utf-8');
    expect(pickSrc).toMatch(/readonly\s+kvHeadIndex\?:\s*0\s*\|\s*1;/);
    expect(pickSrc.toLowerCase()).not.toContain('dory');
  });
});
