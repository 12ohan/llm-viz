/**
 * DORY4 Slice 4C — Views 3–6 Headless Geometry, Atlas, DR20, DR21 & Provenance Test Suite
 * (`tests/dory-scene-mixers-kv-spec.test.ts`)
 *
 * Enforces every `DORY4` Gate 4 exit invariant across all four spatial scenes:
 *   - View 3 (`gqa_rope` — `src/dory/scene/gqaRopeScene.ts`): Gated GQA & Dual-RoPE Cutaway
 *   - View 4 (`swiglu` — `src/dory/scene/swigluScene.ts`): Normalized SwiGLU Mixer Cutaway
 *   - View 5 (`kv_compare` — `src/dory/scene/kvCompareScene.ts`): Dual-Axis KV & `last_loop` Divergence
 *   - View 6 (`self_spec` — `src/dory/scene/selfSpecScene.ts`): Contract A Paper §6.2 Self-Speculation
 *
 * Verified Contracts:
 *   1. Finite geometry (`0` NaN/Inf) and exact 5-pass buffer strides via `validateSceneGeometryFinite`.
 *   2. Every `RG32F` atlas strictly `<= 512 KiB` (`DORY_MAX_ATLAS_BYTES = 524,288 B`) and within
 *      its per-view budget, with `0` overlaps and `0` out-of-bounds slices verified independently
 *      via `auditAtlasRegions` (including heavy `S = 32` & `L > cap_swa = 16` long-chunk `DR24` cases).
 *   3. `DR20` Live `RG32F` `G`-Channel Mask Verification on EVERY SWA & Full score/softmax plate
 *      (`sheetKind in {1, 2}`): verifies exact per-texel binary `G`-channel match against analytical
 *      SWA (`0 <= q - k < 8`) and Full (`0 <= q - k`) visibility rules across all `16` plates in View 3,
 *      all `4` plates in View 5, and both `2` plates in View 6, PLUS `DoryDr20CausalMaskError` tamper
 *      negative controls on live Views 3, 5, and 6 scenes.
 *   4. `DR21` real 3D raycasts (`pickMacroBay` from untouched `src/render/pick.ts`) ->
 *      `pickHitToDoryCell` round-trips & exact `computeSheetCellWorldPos` active cell popovers
 *      across Views 3, 4, 5, and 6.
 *   5. View 3 `q_dim` / `kv_dim` expansion disclosure (`D2`: mini `8.0x`/`2.0x` vs. prod `1.60x`/`0.40x`),
 *      4-plane NeoX dual-RoPE (`θ_S = 16` vs. `θ_F = 1024`, `DR6` RoPE -> L2Norm), Q-only `sqk^2`,
 *      sigmoid gate (`DR3`), full 64-dim concat `attn_sub_norm`, Layer-0 `DR5`, and `last_loop`
 *      non-final `writeKv = false` (`DoryPastReader`, `DR17`).
 *   6. View 4 `d -> d_ff` expansion disclosure (`3.00x` mini vs. `2.80x` prod), linear signed
 *      Family 4 `suv_gate`/`suv_up` (init `1.0`, NO `.abs()`, negative-channel markers), pre-`down_proj`
 *      24-dim `ffn_sub_norm`, progressive `Accum Reveal` (`SHEET_REVEAL_*` + binary64 partial sums `D5`),
 *      and Anti-SLERP (`DR7`).
 *   7. View 5 strictly isolated dual-session execution (0-ULP zero caller mutation `D6`/`DR18`),
 *      Theorems 1–3 (`DR17`, including `> 1e-2` chunked & 1-by-1 divergence at `k >= 2`),
 *      `cap_swa = 16` ring + `absPosTag` (`DR19`), and production `1 -> 256K` BF16 scaling with exact
 *      `S`-invariant `(4 + 5k) / 9` ratio & `S = 1,024` `4x` slope knee.
 *   8. View 6 Contract A Dual-Coda `104`-bank (`8 + 20 + 60 + 8 + 8`, `+20 KiB` `output_shallow`,
 *      `260 KiB` total) provenance badge (`DORY_SPEC_CONTRACT_A_BADGE`, `DR23`), division-free binary64
 *      rejection sampling & residual correction, transaction-local `cap_swa = 16` rollback (`DR19`),
 *      and synthesized read-only preview on standard sessions (0-ULP zero caller mutation).
 *   9. `D4` architectural boundary isolation (`0` Gate 4b/4c/sibling/Node imports, `0` `Math.random()`,
 *      `0` DOM/WebGL globals) and untouched `src/render/pick.ts` (`DR21`).
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
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
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
} from '../src/render/sheets';
import { computeSheetCellWorldPos } from '../src/shared/overlayLayout';
import {
  assertBitExactFiniteArray,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';
import {
  DORY_MAX_ATLAS_BYTES,
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  DORY_PLANTED_EXTRAPOLATION_SPEC,
  DORY_PRODUCTION_SPEC,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  buildDoryDualRopeTables,
  computeProductionKvBytesBf16,
  createDorySession,
  effectiveSGateSquared,
  effectiveSqkSquared,
  effectiveSuvScale,
  getDefaultDoryModelParameters,
  vectorL2Norm,
  type DoryAttentionKind,
  type DoryKvCache,
  type DoryKvCacheMode,
  type DorySession,
} from '../src/dory';
import {
  DORY_ANTI_SLERP_PLAQUE,
  DORY_SPATIAL_VIEW_MODES,
  type DorySpatialScene,
} from '../src/dory/scene/types';
import {
  DoryDr20CausalMaskError,
  auditAtlasRegions,
  computeDoryAtlasContentKey,
  pickHitToDoryCell,
  resolveDorySessionContext,
  validateSceneGeometryFinite,
} from '../src/dory/scene/provenance';
import {
  DORY_GQA_ROPE_ATLAS_BUDGET_BYTES,
  DORY_GQA_ROPE_DEFAULT_FULL_LAYER,
  DORY_GQA_ROPE_DEFAULT_SWA_LAYER,
  buildDoryGqaRopeScene,
  computeDoryRopePhasorPlanes,
  resolveAttentionPhysicalLayerIdx,
} from '../src/dory/scene/gqaRopeScene';
import {
  DORY_SWIGLU_ATLAS_BUDGET_BYTES,
  DORY_SWIGLU_DEFAULT_LAYER,
  buildDorySwigluScene,
  computeSwigluDownProjPartialSum,
  resolveMlpPhysicalLayerIdx,
} from '../src/dory/scene/swigluScene';
import {
  DORY_KV_COMPARE_ATLAS_BUDGET_BYTES,
  DORY_PRODUCTION_KV_SCALING_LENGTHS,
  buildDefaultKvChunkPartition,
  buildDoryKvCompareScene,
  buildProductionKvScalingCurves,
} from '../src/dory/scene/kvCompareScene';
import {
  DORY_DIVISION_FREE_BINARY64_RULE,
  DORY_SELF_SPEC_ATLAS_BUDGET_BYTES,
  buildDorySelfSpecScene,
  computeSpecSwaRingSlotStates,
} from '../src/dory/scene/selfSpecScene';

const PARAMS = getDefaultDoryModelParameters();
const ROPE_TABLES = buildDoryDualRopeTables(PARAMS.config);

/** Canonical 12-token non-EOS prompt (Gate 1 Theorem 3 fixture) crossing SWA window W = 8 (pos 8..11). */
const PROMPT_12 = [12, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38] as const;

/** 8-token non-EOS prompt matching View 1/2 default fixtures. */
const PROMPT_8 = [1, 10, 4, 18, 40, 25, 26, 57] as const;

/** 32-token non-EOS prompt exercising S_max = 32 & L > cap_swa = 16 long-chunk workspace (DR24). */
const PROMPT_32: readonly number[] = Object.freeze(
  Array.from({ length: 32 }, (_, i) => {
    const candidate = 3 + ((i * 7 + 5) % 58);
    // Avoid EOS token IDs 2 and 11
    return candidate === 11 ? 12 : candidate;
  }),
);

// ============================================================================
// SHARED GATE 4 VERIFICATION HELPERS
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
  // Views 3–6 lie on the horizontal XZ drafting table, so aim top-down along -Y to avoid
  // grazing neighboring Z-rows in KV bank matrices, SWA ring slots, and bar charts.
  return {
    origin: [cx, box.max[1] + distance, cz],
    direction: [0, -1, 0],
  };
}

function readAtlasRegionRG(
  scene: DorySpatialScene,
  regionId: string,
): { readonly r: Float32Array; readonly g: Float32Array; readonly rows: number; readonly cols: number } {
  const atlas = scene.valueAtlas;
  expect(atlas, `Missing valueAtlas for region '${regionId}'`).not.toBeNull();
  const reg = scene.atlasRegions.find((rg) => rg.id === regionId);
  expect(reg, `Missing atlas region '${regionId}' in ${scene.viewMode}`).toBeDefined();
  const count = reg!.rows * reg!.cols;
  const rOut = new Float32Array(count);
  const gOut = new Float32Array(count);
  for (let row = 0; row < reg!.rows; row += 1) {
    for (let col = 0; col < reg!.cols; col += 1) {
      const idx = ((reg!.y0 + row) * atlas!.width + (reg!.x0 + col)) * 2;
      rOut[row * reg!.cols + col] = atlas!.data[idx]!;
      gOut[row * reg!.cols + col] = atlas!.data[idx + 1]!;
    }
  }
  return { r: rOut, g: gOut, rows: reg!.rows, cols: reg!.cols };
}

/**
 * Exhaustive Gate 4 health, stride, atlas non-overlap, <= 512 KiB budget, AND DR20 live G-channel
 * mask verifier run across every single generated scene in Views 3–6.
 */
function assertGate4SceneHealth(
  scene: DorySpatialScene,
  label: string,
  expectedViewBudgetBytes: number,
  expectedCausalSheetCount: number,
): void {
  const validation = validateSceneGeometryFinite(scene);
  expect(validation.valid, `${label} validation errors: ${validation.errors.join('; ')}`).toBe(true);
  expect(validation.errors).toEqual([]);

  // Exact 5-pass GPU buffer stride parity
  expect(scene.solidData.length).toBe(scene.solidCount * SOLID_INSTANCE_FLOATS);
  expect(scene.lineData.length).toBe(scene.lineCount * LINE_INSTANCE_FLOATS);
  expect(scene.ribbonData.length).toBe(scene.ribbonCount * RIBBON_INSTANCE_FLOATS);
  expect(scene.sheetData.length).toBe(scene.sheetCount * SHEET_INSTANCE_FLOATS);
  expect(scene.sheetBasisData.length).toBe(scene.sheetCount * SHEET_BASIS_FLOATS);
  expect(scene.sheetRevealData.length).toBe(scene.sheetCount * SHEET_REVEAL_FLOATS);
  expect(scene.flowVertices.length % FLOW_RIBBON_VERTEX_FLOATS).toBe(0);

  // Live RG32F atlas & <= 512 KiB cap
  expect(scene.valueAtlas, `${label} must carry live RG32F atlas`).not.toBeNull();
  expect(scene.atlasKey, `${label} must carry deterministic atlasKey`).not.toBeNull();
  expect(scene.atlasAudit, `${label} must carry atlasAudit`).not.toBeNull();

  const atlas = scene.valueAtlas!;
  const byteLength = atlas.width * atlas.height * 8;
  expect(byteLength, `${label} RG32F <= 512 KiB`).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
  expect(byteLength, `${label} RG32F <= view budget`).toBeLessThanOrEqual(expectedViewBudgetBytes);

  // Independent re-audit via auditAtlasRegions (never blindly trusting scene.atlasAudit)
  const reaudit = auditAtlasRegions(atlas.width, atlas.height, scene.atlasRegions, DORY_MAX_ATLAS_BYTES);
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

  // DR20 Direct Instance-Buffer Audit on EVERY causal score/softmax sheet (sheetKind in {1, 2})
  let causalCount = 0;
  for (let i = 0; i < scene.sheetCount; i += 1) {
    const base = i * SHEET_INSTANCE_FLOATS;
    const gridCols = Math.round(scene.sheetData[base + 6]!);
    const gridRows = Math.round(scene.sheetData[base + 7]!);
    const sheetKind = Math.round(scene.sheetData[base + 8]!);
    if (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) {
      causalCount += 1;
      const u0 = scene.sheetData[base + 13]!;
      const v0 = scene.sheetData[base + 14]!;
      const uW = scene.sheetData[base + 15]!;
      const vH = scene.sheetData[base + 16]!;
      expect(uW, `${label}.sheet[${i}] uW > 0 (DR20)`).toBeGreaterThan(0);
      expect(vH, `${label}.sheet[${i}] vH > 0 (DR20)`).toBeGreaterThan(0);
      expect(u0).toBeGreaterThanOrEqual(0);
      expect(v0).toBeGreaterThanOrEqual(0);
      expect(u0 + uW).toBeLessThanOrEqual(1.0 + 1e-5);
      expect(v0 + vH).toBeLessThanOrEqual(1.0 + 1e-5);

      const x0 = Math.floor(u0 * atlas.width + 1e-4);
      const y0 = Math.floor(v0 * atlas.height + 1e-4);
      const x1 = Math.ceil((u0 + uW) * atlas.width - 1e-4);
      const y1 = Math.ceil((v0 + vH) * atlas.height - 1e-4);
      expect(x1 - x0, `${label}.sheet[${i}] exact texel cols`).toBe(gridCols);
      expect(y1 - y0, `${label}.sheet[${i}] exact texel rows`).toBe(gridRows);

      for (let ty = y0; ty < y1; ty += 1) {
        for (let tx = x0; tx < x1; tx += 1) {
          const tIdx = (ty * atlas.width + tx) * 2;
          const rVal = atlas.data[tIdx]!;
          const gVal = atlas.data[tIdx + 1]!;
          expect(Number.isFinite(rVal), `${label}.sheet[${i}] finite R`).toBe(true);
          expect(gVal === 0 || gVal === 1, `${label}.sheet[${i}] binary G=${gVal}`).toBe(true);
          if (gVal === 1) {
            expect(rVal, `${label}.sheet[${i}] masked cell R === 0`).toBe(0);
          }
        }
      }
    } else {
      expect(sheetKind, `${label}.sheet[${i}] non-causal sheet must be standard`).toBe(
        SHEET_KIND_STANDARD,
      );
    }
  }
  expect(causalCount, `${label} causal sheet count`).toBe(expectedCausalSheetCount);
}

/**
 * Verifies texel-by-texel that an SWA or Full causal atlas region's `G` channel matches the
 * exact analytical Dory causal/sliding-window visibility rule (`Audit §7.1`, `DR20`).
 */
function assertCausalMaskExactInAtlas(params: {
  readonly scene: DorySpatialScene;
  readonly regionId: string;
  readonly seqLen: number;
  readonly kvLen: number;
  readonly startPos: number;
  readonly kvAbsPositions: ArrayLike<number>;
  readonly attentionKind: DoryAttentionKind;
  readonly slidingWindow?: number;
  readonly label: string;
}): { readonly visibleCount: number; readonly maskedCount: number; readonly swaPastEvictedCount: number } {
  const {
    scene,
    regionId,
    seqLen,
    kvLen,
    startPos,
    kvAbsPositions,
    attentionKind,
    slidingWindow = DORY_MINI_CONFIG.slidingWindow,
    label,
  } = params;

  const { r, g, rows, cols } = readAtlasRegionRG(scene, regionId);
  expect(rows, `${label}.rows`).toBe(seqLen);
  expect(cols, `${label}.cols`).toBe(kvLen);

  let visibleCount = 0;
  let maskedCount = 0;
  let swaPastEvictedCount = 0;

  for (let row = 0; row < seqLen; row += 1) {
    const qAbs = startPos + row;
    for (let col = 0; col < kvLen; col += 1) {
      const kAbs = kvAbsPositions[col]!;
      const diff = qAbs - kAbs;
      const expectedVisible =
        attentionKind === 'sliding_attention'
          ? diff >= 0 && diff < slidingWindow
          : diff >= 0;
      const expectedG = expectedVisible ? 0 : 1;
      const idx = row * kvLen + col;
      expect(
        g[idx],
        `${label} G[${row},${col}] (qAbs=${qAbs}, kAbs=${kAbs}, kind=${attentionKind})`,
      ).toBe(expectedG);

      if (expectedVisible) {
        visibleCount += 1;
        expect(Number.isFinite(r[idx]!), `${label} finite visible R[${row},${col}]`).toBe(true);
      } else {
        maskedCount += 1;
        expect(r[idx], `${label} masked R[${row},${col}] === 0`).toBe(0);
        if (attentionKind === 'sliding_attention' && diff >= slidingWindow) {
          swaPastEvictedCount += 1;
        }
      }
    }
  }

  return { visibleCount, maskedCount, swaPastEvictedCount };
}

interface DeepKvSnapshot {
  readonly mode: DoryKvCacheMode;
  readonly kLoops: number;
  readonly includeShallowCoda: boolean;
  readonly committedLen: number;
  readonly chunkSizes: readonly number[];
  readonly bankLogicalLens: readonly number[];
  readonly kCopies: readonly Float32Array[];
  readonly vCopies: readonly Float32Array[];
  readonly posCopies: readonly Int32Array[];
  readonly loopCopies: readonly Int32Array[];
}

function captureDeepKvSnapshot(kv: DoryKvCache): DeepKvSnapshot {
  return {
    mode: kv.mode,
    kLoops: kv.kLoops,
    includeShallowCoda: kv.includeShallowCoda,
    committedLen: kv.committedLen,
    chunkSizes: [...kv.chunkSizes],
    bankLogicalLens: kv.allBanks.map((b) => b.logicalLen),
    kCopies: kv.allBanks.map((b) => new Float32Array(b.k)),
    vCopies: kv.allBanks.map((b) => new Float32Array(b.v)),
    posCopies: kv.allBanks.map((b) => new Int32Array(b.absPosTag)),
    loopCopies: kv.allBanks.map((b) => new Int32Array(b.writerLoopTag)),
  };
}

function assertSessionZeroMutation(
  session: DorySession,
  recordId: string,
  snapBefore: ReturnType<DorySession['getSnapshot']>,
  memBefore: ReturnType<DorySession['measureMemory']>,
  kvBefore: DeepKvSnapshot,
  label: string,
): void {
  const snapAfter = session.getSnapshot();
  const memAfter = session.measureMemory();
  expect(memAfter, `${label}.memory`).toEqual(memBefore);
  expect(snapAfter.recordCount, `${label}.recordCount`).toBe(snapBefore.recordCount);
  expect(snapAfter.activeBranchId, `${label}.activeBranchId`).toBe(snapBefore.activeBranchId);
  expect(snapAfter.activeRecordId, `${label}.activeRecordId`).toBe(snapBefore.activeRecordId);
  expect(snapAfter.branches.length, `${label}.branches`).toBe(snapBefore.branches.length);

  const kvAfter = session.reconstructKvCacheAtRecord(recordId);
  expect(kvAfter.mode).toBe(kvBefore.mode);
  expect(kvAfter.kLoops).toBe(kvBefore.kLoops);
  expect(kvAfter.includeShallowCoda).toBe(kvBefore.includeShallowCoda);
  expect(kvAfter.committedLen).toBe(kvBefore.committedLen);
  expect(kvAfter.chunkSizes).toEqual(kvBefore.chunkSizes);
  expect(kvAfter.allBanks.length).toBe(kvBefore.kCopies.length);
  for (let b = 0; b < kvBefore.kCopies.length; b += 1) {
    const bank = kvAfter.allBanks[b]!;
    expect(bank.logicalLen).toBe(kvBefore.bankLogicalLens[b]);
    assertBitExactFiniteArray(bank.k, kvBefore.kCopies[b]!, `${label}.bank[${b}].k`);
    assertBitExactFiniteArray(bank.v, kvBefore.vCopies[b]!, `${label}.bank[${b}].v`);
    assertBitExactFiniteArray(bank.absPosTag, kvBefore.posCopies[b]!, `${label}.bank[${b}].absPosTag`);
    assertBitExactFiniteArray(
      bank.writerLoopTag,
      kvBefore.loopCopies[b]!,
      `${label}.bank[${b}].writerLoopTag`,
    );
  }
}

// ============================================================================
// 1. VIEW 3 (`gqa_rope`) — GATED GQA, DUAL-ROPE, DR20 LIVE G-MASK & q_dim DISCLOSURE
// ============================================================================

describe('Gate 4 View 3 (gqa_rope) — Gated GQA, Dual-RoPE, DR20 Live G-Channel & q_dim Disclosure', () => {
  it('verifies finite geometry, <= 512 KiB RG32F, 0 overlaps, and exact DR20 live G-channel masks across ALL 16 SWA & Full plates (S=12 eviction, S=32 long-chunk DR24, and incremental decode)', () => {
    // 1. S = 12 crossing SWA window W = 8 (eviction active on pos 8..11)
    const session12 = createDorySession({
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
    const prefillRec = session12.getActiveRecord()!;
    const snapBefore = session12.getSnapshot();
    const memBefore = session12.measureMemory();
    const kvBefore = captureDeepKvSnapshot(session12.reconstructKvCacheAtRecord(prefillRec.recordId));

    const swaScene12 = buildDoryGqaRopeScene(session12, {
      physicalLayerIdx: DORY_GQA_ROPE_DEFAULT_SWA_LAYER, // p = 16 (SWA W=8, theta_S=16)
      loopIteration: 1,
      selectedQHead: 2,
      selectedRotaryPlane: 1,
    });
    const swaScene12Repeat = buildDoryGqaRopeScene(session12, {
      physicalLayerIdx: DORY_GQA_ROPE_DEFAULT_SWA_LAYER,
      loopIteration: 1,
      selectedQHead: 2,
      selectedRotaryPlane: 1,
    });
    const fullScene12 = buildDoryGqaRopeScene(session12, {
      physicalLayerIdx: DORY_GQA_ROPE_DEFAULT_FULL_LAYER, // p = 22 (Full, theta_F=1024)
      loopIteration: 2,
      selectedQHead: 6,
      selectedRotaryPlane: 3,
    });

    assertGate4SceneHealth(swaScene12, 'gqa_rope.S12.swa_p16', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    assertGate4SceneHealth(fullScene12, 'gqa_rope.S12.full_p22', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);

    // Deterministic content-addressed atlasKey repeatability & zero caller session mutation
    expect(swaScene12.atlasKey).toBe(swaScene12Repeat.atlasKey);
    expect(
      computeDoryAtlasContentKey(
        'gqa_rope',
        swaScene12.valueAtlas!,
        `b=${swaScene12.branchId}:r=${swaScene12.recordId}:p=16:t=1:h=2`,
      ),
    ).toBe(swaScene12.atlasKey);
    assertSessionZeroMutation(session12, prefillRec.recordId, snapBefore, memBefore, kvBefore, 'gqa_rope.S12');

    const swaMeta = swaScene12.gqaRopeMetadata!;
    const fullMeta = fullScene12.gqaRopeMetadata!;
    expect(swaMeta.hasLiveCausalMaskGChannel).toBe(true);
    expect(fullMeta.hasLiveCausalMaskGChannel).toBe(true);

    // Exact S = 12 visibility math:
    //   Full causal: 1 + 2 + ... + 12 = 78 visible, 144 - 78 = 66 masked (upper triangle only)
    //   SWA W = 8:   (1+2+3+4+5+6+7) + 5*8 = 68 visible, 144 - 68 = 76 masked (66 upper + 10 evicted lower!)
    expect(fullMeta.visibleCellCount).toBe(78);
    expect(fullMeta.maskedCellCount).toBe(66);
    expect(swaMeta.visibleCellCount).toBe(68);
    expect(swaMeta.maskedCellCount).toBe(76);

    // Verify EVERY SINGLE ONE of the 16 causal score & softmax plates in BOTH SWA and Full scenes
    for (let qh = 0; qh < DORY_MINI_CONFIG.numAttentionHeads; qh += 1) {
      for (const prefix of ['gqa_scores_h', 'gqa_probs_h'] as const) {
        const swaCheck = assertCausalMaskExactInAtlas({
          scene: swaScene12,
          regionId: `${prefix}${qh}`,
          seqLen: swaMeta.seqLen,
          kvLen: swaMeta.kvLen,
          startPos: swaMeta.startPos,
          kvAbsPositions: swaMeta.kvAbsPositions,
          attentionKind: 'sliding_attention',
          label: `gqa_rope.S12.swa.${prefix}${qh}`,
        });
        // Proves genuine SWA lower-band eviction (10 past cells with G=1, NOT geometric cellCol > cellRow!)
        expect(swaCheck.visibleCount).toBe(68);
        expect(swaCheck.maskedCount).toBe(76);
        expect(swaCheck.swaPastEvictedCount).toBe(10);

        const fullCheck = assertCausalMaskExactInAtlas({
          scene: fullScene12,
          regionId: `${prefix}${qh}`,
          seqLen: fullMeta.seqLen,
          kvLen: fullMeta.kvLen,
          startPos: fullMeta.startPos,
          kvAbsPositions: fullMeta.kvAbsPositions,
          attentionKind: 'full_attention',
          label: `gqa_rope.S12.full.${prefix}${qh}`,
        });
        expect(fullCheck.visibleCount).toBe(78);
        expect(fullCheck.maskedCount).toBe(66);
        expect(fullCheck.swaPastEvictedCount).toBe(0);
      }
    }

    // 2. Incremental decode step (pastLen > 0, seqLen = 1) on SWA & Full (testing historical KV & eviction!)
    session12.step();
    const swaDecode = buildDoryGqaRopeScene(session12, {
      physicalLayerIdx: 16,
      loopIteration: 1,
      selectedQHead: 5,
    });
    const fullDecode = buildDoryGqaRopeScene(session12, {
      physicalLayerIdx: 22,
      loopIteration: 2,
      selectedQHead: 5,
    });
    assertGate4SceneHealth(swaDecode, 'gqa_rope.decode.swa', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    assertGate4SceneHealth(fullDecode, 'gqa_rope.decode.full', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    expect(swaDecode.gqaRopeMetadata!.seqLen).toBe(1);
    expect(swaDecode.gqaRopeMetadata!.pastLen).toBe(7); // W - 1 = 7 visible past tokens in SWA
    expect(swaDecode.gqaRopeMetadata!.kvLen).toBe(8);
    expect(fullDecode.gqaRopeMetadata!.seqLen).toBe(1);
    expect(fullDecode.gqaRopeMetadata!.pastLen).toBe(12); // all 12 past tokens in Full
    expect(fullDecode.gqaRopeMetadata!.kvLen).toBe(13);

    // 3. Heavy worst-case atlas stress: S = 32 single-shot prefill (L = 32 > cap_swa = 16, DR24!)
    const session32 = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_32,
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });
    const swaScene32 = buildDoryGqaRopeScene(session32, { physicalLayerIdx: 16, loopIteration: 1 });
    const fullScene32 = buildDoryGqaRopeScene(session32, { physicalLayerIdx: 22, loopIteration: 2 });
    assertGate4SceneHealth(swaScene32, 'gqa_rope.S32.swa', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    assertGate4SceneHealth(fullScene32, 'gqa_rope.S32.full', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    expect(swaScene32.gqaRopeMetadata!.seqLen).toBe(32);
    expect(swaScene32.gqaRopeMetadata!.kvLen).toBe(32);
    expect(swaScene32.atlasAudit!.byteLength).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
    expect(fullScene32.atlasAudit!.byteLength).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);

    // Verify S=32 SWA G-channel mask on head 0 & head 7 (32*32 = 1024 cells; visible = 28 + 25*8 = 228)
    for (const qh of [0, 7]) {
      const check32 = assertCausalMaskExactInAtlas({
        scene: swaScene32,
        regionId: `gqa_probs_h${qh}`,
        seqLen: 32,
        kvLen: 32,
        startPos: 0,
        kvAbsPositions: swaScene32.gqaRopeMetadata!.kvAbsPositions,
        attentionKind: 'sliding_attention',
        label: `gqa_rope.S32.swa.probs_h${qh}`,
      });
      expect(check32.visibleCount).toBe(228);
      expect(check32.maskedCount).toBe(1024 - 228);
      expect(check32.swaPastEvictedCount).toBe(300); // 24*25/2 = 300 evicted lower-triangle cells!
    }
  });

  it('verifies q_dim expansion disclosure (D2), 4-plane dual-RoPE phasors (DR6), Q-only sqk^2, sigmoid gate (DR3), 64-dim concat attn_sub_norm, Layer-0 DR5, and last_loop writeKv=false (DR17)', () => {
    const session = createDorySession({
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

    // Non-final pass t = 1 < k = 4 on SWA Layer 16 (also has planted |attn_alpha| > 1!)
    const swaNonFinal = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 1,
      selectedQHead: 4, // maps to KV1
      selectedQueryRow: 3, // absPos = 4 + 3 = 7
      selectedKeyCol: 2, // cached historical key from chunk 0 (absPos = 2, writerLoop = 4!)
      selectedRotaryPlane: 2,
    });
    const fullCoda70 = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 70, // Coda Full Attention (Pair L35)
      loopIteration: 3, // ignored -> 0 for Coda
      selectedQHead: 7,
    });
    const preludeL0 = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 0, // Prelude SWA Layer 0 (DR5 no pre-norm)
    });

    assertGate4SceneHealth(swaNonFinal, 'gqa_rope.last_loop.t1.p16', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    assertGate4SceneHealth(fullCoda70, 'gqa_rope.coda.p70', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);
    assertGate4SceneHealth(preludeL0, 'gqa_rope.prelude.p0', DORY_GQA_ROPE_ATLAS_BUDGET_BYTES, 16);

    const m = swaNonFinal.gqaRopeMetadata!;

    // 1. q_dim & kv_dim scaling ratio disclosure (D2: 8.0x/2.0x mini vs 1.60x/0.40x prod)
    expect(m.qExpansionRatioMini).toBe(8.0);
    expect(m.qExpansionRatioProd).toBe(1.6);
    expect(m.kvDimRatioMini).toBe(2.0);
    expect(m.kvDimRatioProd).toBe(0.4);
    expect(m.qExpansionDisclosure).toContain('q_dim=64');
    expect(m.qExpansionDisclosure).toContain('8.0x');
    expect(m.qExpansionDisclosure).toContain('4096');
    expect(m.qExpansionDisclosure).toContain('1.60x');
    expect(m.qExpansionDisclosure).toContain('kv_dim=16');
    expect(m.qExpansionDisclosure).toContain('0.40x');
    expect(
      swaNonFinal.provenanceBadges.some(
        (b) => b.label.includes('8.0x') && b.label.includes('1.60x'),
      ),
    ).toBe(true);
    const scaleCallout = swaNonFinal.callouts.find((c) => c.id === 'callout_gqa_scaling_disclosure');
    expect(scaleCallout).toBeDefined();
    expect(scaleCallout!.body).toContain('1.60x');
    expect(swaNonFinal.calipers.some((c) => c.id === 'caliper_gqa_qdim' && c.label.includes('1.60x'))).toBe(
      true,
    );

    // 2. Layer-0 DR5 no pre-norm disclosure
    const l0Callout = preludeL0.callouts.find((c) => c.id === 'callout_gqa_scaling_disclosure');
    expect(l0Callout!.body).toContain('NO pre-norm (DR5)');
    expect(preludeL0.gqaRopeMetadata!.loopIteration).toBe(0);
    expect(fullCoda70.gqaRopeMetadata!.loopIteration).toBe(0);
    expect(fullCoda70.gqaRopeMetadata!.pairedLayerIdx).toBe(35);

    // 3. Dual-RoPE 4-plane phasor accuracy (DR6: RoPE strictly BEFORE per-head Q/K L2Norm)
    expect(m.ropeBeforeQkNorm).toBe(true);
    expect(m.ropeThetaMini).toBe(16);
    expect(m.ropeThetaProd).toBe(10_000);
    expect(m.slidingWindowMini).toBe(8);
    expect(m.slidingWindowProd).toBe(1_024);
    expect(m.selectedQHead).toBe(4);
    expect(m.selectedKvHead).toBe(1); // 4:1 GQA: Q4..Q7 -> KV1
    expect(m.ropePhasors).toHaveLength(4);

    // Exact dyadic SWA inv_freq = [1, 1/2, 1/4, 1/8] and NeoX rotate_half check
    const expectedSwaInvFreq = [1, 0.5, 0.25, 0.125];
    for (let j = 0; j < 4; j += 1) {
      const pl = m.ropePhasors[j]!;
      expect(pl.planeIdx).toBe(j);
      expect(pl.invFreq).toBeCloseTo(expectedSwaInvFreq[j]!, 7);
      expect(pl.periodTokens).toBeCloseTo((2 * Math.PI) / expectedSwaInvFreq[j]!, 5);
      expect(pl.queryAngleRad).toBeCloseTo(m.selectedQueryAbsPos * pl.invFreq, 6);
      expect(pl.keyAngleRad).toBeCloseTo(m.selectedKeyAbsPos * pl.invFreq, 6);
      expect(pl.deltaAngleRad).toBeCloseTo((m.selectedQueryAbsPos - m.selectedKeyAbsPos) * pl.invFreq, 6);

      // Verify NeoX rotate_half(qPrePair, queryAngleRad) === qRotPair
      const cQ = Math.cos(pl.queryAngleRad);
      const sQ = Math.sin(pl.queryAngleRad);
      expect(pl.qPrePair[0] * cQ - pl.qPrePair[1] * sQ).toBeCloseTo(pl.qRotPair[0], 5);
      expect(pl.qPrePair[1] * cQ + pl.qPrePair[0] * sQ).toBeCloseTo(pl.qRotPair[1], 5);

      // Verify historical cached key inverse rotation consistency: R(keyAngleRad) * kPrePair === kRotPair
      const cK = Math.cos(pl.keyAngleRad);
      const sK = Math.sin(pl.keyAngleRad);
      expect(pl.kPrePair[0] * cK - pl.kPrePair[1] * sK).toBeCloseTo(pl.kRotPair[0], 5);
      expect(pl.kPrePair[1] * cK + pl.kPrePair[0] * sK).toBeCloseTo(pl.kRotPair[1], 5);
    }

    // Full Attention RoPE base theta_F = 1,024 vs prod 10,000,000
    const mFull = fullCoda70.gqaRopeMetadata!;
    expect(mFull.ropeThetaMini).toBe(1_024);
    expect(mFull.ropeThetaProd).toBe(10_000_000);
    expect(mFull.slidingWindowMini).toBeNull();
    expect(mFull.slidingWindowProd).toBeNull();
    for (let j = 0; j < 4; j += 1) {
      expect(mFull.ropePhasors[j]!.invFreq).toBeCloseTo(Math.pow(1024, (-2 * j) / 8), 6);
    }

    // 4. Q-only squared sqk^2 & s_gate^2 parity + cached K stays unit-norm per head!
    const layer16Weights = PARAMS.layers[16]!;
    expect(layer16Weights.kind).toBe('attention');
    if (layer16Weights.kind === 'attention') {
      const refSqkSq = effectiveSqkSquared(layer16Weights.sqk, PARAMS.config);
      const refSGateSq = effectiveSGateSquared(layer16Weights.sGate, PARAMS.config);
      const atlasSqk = readAtlasRegionRG(swaNonFinal, 'gqa_sqk_eff_sq').r;
      const atlasSGate = readAtlasRegionRG(swaNonFinal, 'gqa_s_gate_eff_sq').r;
      expect(computeStrictMaxAbsDiff(atlasSqk, refSqkSq, 'gqa.sqk_sq')).toBeLessThanOrEqual(1e-6);
      expect(computeStrictMaxAbsDiff(atlasSGate, refSGateSq, 'gqa.s_gate_sq')).toBeLessThanOrEqual(1e-6);
      expect(m.sqkScaleRange[0]).toBeCloseTo(Math.min(...refSqkSq), 5);
      expect(m.sqkScaleRange[1]).toBeCloseTo(Math.max(...refSqkSq), 5);
      expect(m.sGateScaleRange[0]).toBeCloseTo(Math.min(...refSGateSq), 5);
      expect(m.sGateScaleRange[1]).toBeCloseTo(Math.max(...refSGateSq), 5);
    }

    // Cached K rows in `gqa_k_cache` are unit-norm per head (NOT scaled by sqk^2!)
    const kCacheR = readAtlasRegionRG(swaNonFinal, 'gqa_k_cache').r;
    for (let row = 0; row < m.seqLen; row += 1) {
      for (let kvh = 0; kvh < DORY_MINI_CONFIG.numKeyValueHeads; kvh += 1) {
        const norm = vectorL2Norm(kCacheR, row * 16 + kvh * 8, 8);
        expect(norm).toBeCloseTo(1.0, 4);
      }
    }

    // 5. Full 64-dim concat attn_sub_norm (NOT per-head 8-dim norm!)
    expect(m.attnSubNormAxisDim).toBe(64);
    expect(m.attnSubNormL2Norm).toBeCloseTo(1.0, 4);
    const subNormR = readAtlasRegionRG(swaNonFinal, 'gqa_attn_sub_norm').r;
    const rowOff = m.selectedQueryRow * 64;
    expect(vectorL2Norm(subNormR, rowOff, 64)).toBeCloseTo(1.0, 4);
    // Individual 8-dim head slices inside concat-normalized 64-dim vector must have norm < 0.9 (average ~ 1/sqrt(8) ≈ 0.354)
    const perHeadNorms = Array.from({ length: 8 }, (_, h) => vectorL2Norm(subNormR, rowOff + h * 8, 8));
    expect(Math.max(...perHeadNorms)).toBeLessThan(0.95);

    // 6. Planted |attn_alpha| > 1 extrapolation on Phys L16 & last_loop DoryPastReader writeKv=false
    expect(m.extrapolatedAlphaCount).toBeGreaterThanOrEqual(
      DORY_PLANTED_EXTRAPOLATION_SPEC.channelIndices.length,
    );
    expect(swaNonFinal.provenanceBadges.some((b) => b.discrepancyId === 'DR7')).toBe(true);
    expect(m.writeKv).toBe(false);
    expect(m.pastLen).toBe(4);
    // Past tokens [0..3] were authored by final loop t = 4; current chunk tokens [4..11] by t = 1!
    expect(m.kvWriterLoops.slice(0, 4)).toEqual([4, 4, 4, 4]);
    expect(m.kvWriterLoops.slice(4)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
    expect(
      swaNonFinal.provenanceBadges.some(
        (b) => b.discrepancyId === 'DR17' && b.label.includes('DoryPastReader'),
      ),
    ).toBe(true);

    // 7. Pure helpers: resolveAttentionPhysicalLayerIdx & computeDoryRopePhasorPlanes
    expect(resolveAttentionPhysicalLayerIdx(undefined)).toBe(16);
    expect(resolveAttentionPhysicalLayerIdx(17)).toBe(16);
    expect(resolveAttentionPhysicalLayerIdx(71)).toBe(70);
    expect(resolveAttentionPhysicalLayerIdx(-5)).toBe(0);
    expect(resolveAttentionPhysicalLayerIdx(999)).toBe(70);

    const ctx = resolveDorySessionContext(session);
    const sub16 = ctx.resolveSublayerRecord(16, 1);
    expect(sub16.trace.kind).toBe('attention');
    if (sub16.trace.kind === 'attention') {
      const directPhasors = computeDoryRopePhasorPlanes({
        attnTrace: sub16.trace,
        ropeTable: ROPE_TABLES.sliding,
        postStepKvCache: ctx.resolveStepTrace().postStepKvCache,
        selectedQHead: 4,
        selectedKvHead: 1,
        selectedQueryRow: 3,
        selectedKeyCol: 2,
      });
      expect(directPhasors).toEqual(m.ropePhasors);
    }
  });
});

// ============================================================================
// 2. VIEW 4 (`swiglu`) — NORMALIZED SWIGLU, SIGNED `suv` (NO `.abs()`), `ffn_sub_norm` & ACCUM REVEAL
// ============================================================================

describe('Gate 4 View 4 (swiglu) — Normalized SwiGLU, Linear Signed suv, ffn_sub_norm & Progressive Accum Reveal', () => {
  it('verifies finite geometry, <= 128 KiB RG32F, 0 overlaps, d->d_ff disclosure, signed negative suv (NO .abs()), ffn_sub_norm, and Anti-SLERP across stages & modes', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });
    const rec = session.getActiveRecord()!;
    const snapBefore = session.getSnapshot();
    const memBefore = session.measureMemory();
    const kvBefore = captureDeepKvSnapshot(session.reconstructKvCacheAtRecord(rec.recordId));

    // Check Prelude (p=1), Recurrent (p=17), and Coda (p=71), plus snapping even indices -> odd
    expect(resolveMlpPhysicalLayerIdx(undefined)).toBe(DORY_SWIGLU_DEFAULT_LAYER);
    expect(resolveMlpPhysicalLayerIdx(16)).toBe(17);
    expect(resolveMlpPhysicalLayerIdx(0)).toBe(1);
    expect(resolveMlpPhysicalLayerIdx(70)).toBe(71);
    expect(resolveMlpPhysicalLayerIdx(999)).toBe(71);

    for (const cfg of [
      { p: 1, loop: 0, stage: 'input' as const },
      { p: 17, loop: 3, stage: 'recurrent' as const },
      { p: 71, loop: 0, stage: 'output' as const },
    ]) {
      const scene = buildDorySwigluScene(session, {
        physicalLayerIdx: cfg.p,
        loopIteration: cfg.loop,
        selectedChannel: 5,
        selectedOutputDim: 3,
        accumReveal: 1.0,
      });
      assertGate4SceneHealth(scene, `swiglu.p${cfg.p}`, DORY_SWIGLU_ATLAS_BUDGET_BYTES, 0);

      const meta = scene.swigluMetadata!;
      expect(meta.physicalLayerIdx).toBe(cfg.p);
      expect(meta.pairedLayerIdx).toBe(cfg.p >> 1);
      expect(meta.stage).toBe(cfg.stage);
      expect(meta.loopIteration).toBe(cfg.loop);
      expect(meta.mlpExpansionRatioMini).toBe(3.0);
      expect(meta.mlpExpansionRatioProd).toBe(2.8);
      expect(meta.mlpExpansionDisclosure).toContain('3.00x');
      expect(meta.mlpExpansionDisclosure).toContain('2.80x');
      expect(meta.suvMultiplierMini).toBeCloseTo(Math.sqrt(8), 10);
      expect(meta.suvMultiplierProd).toBeCloseTo(Math.sqrt(2560), 10);

      // Family 4 Linear Signed suv_gate & suv_up (init ~1.0, NO .abs()!)
      const weights = PARAMS.layers[cfg.p]!;
      expect(weights.kind).toBe('mlp');
      if (weights.kind === 'mlp') {
        const refSuvGate = effectiveSuvScale(weights.suvGate, PARAMS.config);
        const refSuvUp = effectiveSuvScale(weights.suvUp, PARAMS.config);
        const atlasSuvGate = readAtlasRegionRG(scene, 'swiglu_suv_gate_eff').r;
        const atlasSuvUp = readAtlasRegionRG(scene, 'swiglu_suv_up_eff').r;
        assertBitExactFiniteArray(atlasSuvGate, refSuvGate, `swiglu.p${cfg.p}.suv_gate`);
        assertBitExactFiniteArray(atlasSuvUp, refSuvUp, `swiglu.p${cfg.p}.suv_up`);

        const negGate = Array.from(refSuvGate).filter((v) => v < 0).length;
        const negUp = Array.from(refSuvUp).filter((v) => v < 0).length;
        expect(negGate).toBe(6); // parameters.ts deterministic negative sign pattern (1/4 of 24)
        expect(negUp).toBe(6);
        expect(meta.hasSignedNegativeSuv).toBe(true);
        expect(meta.negativeSuvGateCount).toBe(negGate);
        expect(meta.negativeSuvUpCount).toBe(negUp);

        // Raised 3D negative signed markers emitted for every negative suv_gate & suv_up channel
        const negGateBoxes = scene.pickBoxes.filter((pb) =>
          /^swiglu_neg_suv_gate_\d+$/.test(pb.partId ?? ''),
        );
        const negUpBoxes = scene.pickBoxes.filter((pb) =>
          /^swiglu_neg_suv_up_\d+$/.test(pb.partId ?? ''),
        );
        expect(negGateBoxes).toHaveLength(negGate);
        expect(negUpBoxes).toHaveLength(negUp);
      }

      // Pre-down_proj 24-dim unit-sphere ffn_sub_norm & Anti-SLERP guard (DR7)
      expect(meta.ffnSubNormAxisDim).toBe(24);
      expect(meta.ffnSubNormL2Norm).toBeCloseTo(1.0, 4);
      expect(meta.antiSlerpMarker).toBe(DORY_NLERP_MARKER);
      expect(
        scene.provenanceBadges.some(
          (b) => b.label === DORY_ANTI_SLERP_PLAQUE && b.discrepancyId === 'DR7',
        ),
      ).toBe(true);
    }

    // Zero mutation of caller session
    assertSessionZeroMutation(session, rec.recordId, snapBefore, memBefore, kvBefore, 'swiglu.stages');

    // Heavy S = 32 stress check: stays <= 128 KiB <= 512 KiB, 0 overlaps
    const session32 = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: { promptTokens: PROMPT_32, kLoops: 2 },
    });
    const scene32 = buildDorySwigluScene(session32, { physicalLayerIdx: 17, loopIteration: 2 });
    assertGate4SceneHealth(scene32, 'swiglu.S32', DORY_SWIGLU_ATLAS_BUDGET_BYTES, 0);
    expect(scene32.swigluMetadata!.seqLen).toBe(32);
  });

  it('verifies progressive Accum Reveal (0%, 25%, 50%, 100%) GPU reveal flags & exact D5 binary64 down_proj partial sums', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: { promptTokens: PROMPT_8, kLoops: 2 },
    });
    session.step();

    const ctx = resolveDorySessionContext(session);
    const sub17 = ctx.resolveSublayerRecord(17, 1);
    expect(sub17.trace.kind).toBe('mlp');
    const mlpTrace = sub17.trace.kind === 'mlp' ? sub17.trace : null;
    expect(mlpTrace).not.toBeNull();
    const w17 = PARAMS.layers[17]!;
    expect(w17.kind).toBe('mlp');
    const downProj = w17.kind === 'mlp' ? w17.downProj : new Float32Array(0);

    const rowIdx = 0;
    const outDim = 4;

    const revealCases = [
      { reveal: 0.0, expectedCount: 0, expectedMode: SHEET_REVEAL_UNCOMPUTED },
      { reveal: 0.25, expectedCount: 6, expectedMode: SHEET_REVEAL_ROW_MAJOR },
      { reveal: 0.5, expectedCount: 12, expectedMode: SHEET_REVEAL_ROW_MAJOR },
      { reveal: 1.0, expectedCount: 24, expectedMode: SHEET_REVEAL_NONE },
    ] as const;

    for (const rc of revealCases) {
      const scene = buildDorySwigluScene(session, {
        physicalLayerIdx: 17,
        loopIteration: 1,
        selectedTokenRow: rowIdx,
        selectedOutputDim: outDim,
        accumReveal: rc.reveal,
      });
      assertGate4SceneHealth(scene, `swiglu.reveal_${rc.expectedCount}`, DORY_SWIGLU_ATLAS_BUDGET_BYTES, 0);

      const meta = scene.swigluMetadata!;
      expect(meta.revealedIntermediateCount).toBe(rc.expectedCount);

      // Verify exact sheetRevealData mode on the swiglu_accum_terms sheet instance
      const accumBoxIdx = scene.pickBoxes.findIndex((pb) => pb.partId === 'swiglu_accum_terms');
      expect(accumBoxIdx).toBeGreaterThanOrEqual(0);
      // Verify via activeCellGrids for swiglu_accum_terms (revealMode + revealColFrontier)
      const accumGrid = scene.activeCellGrids.find((g) => g.colDimLabel?.startsWith('reveal='));
      expect(accumGrid).toBeDefined();
      expect(accumGrid!.revealMode).toBe(rc.expectedMode);
      expect(accumGrid!.revealColFrontier).toBe(Math.max(0, rc.expectedCount - 1));

      // Verify exact binary64 partial sum matches computeSwigluDownProjPartialSum & live downProjOut
      const partial = computeSwigluDownProjPartialSum(
        mlpTrace!.ffnSubNorm,
        rowIdx,
        downProj,
        outDim,
        rc.expectedCount,
      );
      if (rc.expectedCount === 0) {
        expect(partial.partialSum).toBe(0);
      }
      expect(partial.fullSum).toBeCloseTo(mlpTrace!.downProjOut[rowIdx * 8 + outDim]!, 5);

      // Verify activeValueOverride on down_proj_out cell grid (number while accumulating, null when complete)
      const downGrid = scene.activeCellGrids.find(
        (g) => g.cols === 8 && g.activeCol === outDim && g.activeValueOverride !== undefined,
      );
      expect(downGrid).toBeDefined();
      if (rc.expectedCount < 24) {
        expect(downGrid!.activeValueOverride).toBeCloseTo(partial.partialSum, 10);
        expect(downGrid!.activeValueTextOverride).toContain('Σ');
      } else {
        expect(downGrid!.activeValueOverride).toBeNull();
        expect(downGrid!.activeValueTextOverride).toBeNull();
      }

      // Atlas `swiglu_accum_terms` texels equal exact M[row, c] * W_down[outDim, c]
      const accumR = readAtlasRegionRG(scene, 'swiglu_accum_terms').r;
      assertBitExactFiniteArray(accumR, partial.contributions, `swiglu.accum_terms_${rc.expectedCount}`);
    }
  });
});

// ============================================================================
// 3. VIEW 5 (`kv_compare`) — ISOLATED DUAL-SESSION, THEOREMS 1–3, `cap_swa=16` & `1 -> 256K`
// ============================================================================

describe('Gate 4 View 5 (kv_compare) — Isolated Dual-Session, DR17 Theorems 1–3, DR20 Plates & 1->256K Scaling', () => {
  it('enforces isolated zero-mutation dual-session execution (D6/DR18), Theorems 1–3 (DR17), cap_swa=16 ring (DR19), and live DR20 G-channel masks across k in {1, 2, 4, 8} & all 3 partition modes', () => {
    const callerSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });
    const callerRec = callerSession.getActiveRecord()!;
    const snapBefore = callerSession.getSnapshot();
    const memBefore = callerSession.measureMemory();
    const kvBefore = captureDeepKvSnapshot(callerSession.reconstructKvCacheAtRecord(callerRec.recordId));

    // 1. Theorem 2 (`k = 1`): universal equivalence across single_shot, chunked, and one_by_one
    const k1Chunked = buildDoryKvCompareScene(callerSession, {
      kLoopsOverride: 1,
      partitionMode: 'chunked',
      physicalLayerIdx: 16,
      loopIteration: 1,
    });
    assertGate4SceneHealth(k1Chunked, 'kv_compare.k1.chunked', DORY_KV_COMPARE_ATLAS_BUDGET_BYTES, 4);
    const m1 = k1Chunked.kvCompareMetadata!;
    expect(m1.isolatedDualSessionVerified).toBe(true);
    expect(m1.hasLiveCausalMaskGChannel).toBe(true);
    expect(m1.theorem2K1Equivalence).toBe(true);
    expect(m1.theorem1SingleShotMaxAbsDiff).toBeLessThanOrEqual(1e-6);
    expect(m1.theorem3ChunkedVsPerLoopMaxAbsDiff).toBeLessThanOrEqual(1e-6);
    expect(m1.theorem3ChunkedVsOneByOneMaxAbsDiff).toBeLessThanOrEqual(1e-6);
    expect(m1.perLoopBankCount).toBe(36);
    expect(m1.lastLoopBankCount).toBe(36);
    expect(m1.sInvariantRatio).toBeCloseTo(1.0, 12);

    // 2. Theorem 1 & Theorem 3 (`k = 4`, S = 12, `[4, 8]` chunked, single_shot, one_by_one; SWA & Full)
    for (const partitionMode of ['single_shot', 'chunked', 'one_by_one'] as const) {
      for (const layerIdx of [16, 22] as const) {
        const sc = buildDoryKvCompareScene(callerSession, {
          kLoopsOverride: 4,
          partitionMode,
          physicalLayerIdx: layerIdx,
          loopIteration: 1, // non-final loop t = 1 < k = 4 (exposes DoryPastReader!)
        });
        assertGate4SceneHealth(
          sc,
          `kv_compare.k4.${partitionMode}.p${layerIdx}`,
          DORY_KV_COMPARE_ATLAS_BUDGET_BYTES,
          4,
        );

        const meta = sc.kvCompareMetadata!;
        expect(meta.kLoops).toBe(4);
        expect(meta.promptLen).toBe(12);
        expect(meta.perLoopBankCount).toBe(96); // 16 + 20*4
        expect(meta.perLoopSwaBanks).toBe(72); // 12 + 15*4
        expect(meta.perLoopFullBanks).toBe(24); // 4 + 5*4
        expect(meta.lastLoopBankCount).toBe(36);
        expect(meta.lastLoopSwaBanks).toBe(27);
        expect(meta.lastLoopFullBanks).toBe(9);
        expect(meta.sInvariantRatio).toBeCloseTo(24 / 9, 12); // 8/3 = 2.6667x
        expect(meta.swaKneeSlopeDropFactor).toBe(4);

        // Exact bank pickBoxes for per_loop (96) and last_loop (36)
        expect(sc.pickBoxes.filter((pb) => /^per_loop_bank_\d+$/.test(pb.partId ?? ''))).toHaveLength(96);
        expect(sc.pickBoxes.filter((pb) => /^last_loop_bank_\d+$/.test(pb.partId ?? ''))).toHaveLength(36);

        // Theorem 1 (empty-cache single-shot <= 1e-6) & Theorem 3 (k=4 chunked & 1-by-1 > 1e-2!)
        expect(meta.theorem2K1Equivalence).toBe(false);
        expect(meta.theorem1SingleShotMaxAbsDiff).toBeLessThanOrEqual(1e-6);
        expect(meta.theorem3ChunkedVsPerLoopMaxAbsDiff).toBeGreaterThan(1e-2);
        expect(meta.theorem3ChunkedVsOneByOneMaxAbsDiff).toBeGreaterThan(1e-2);

        if (partitionMode === 'single_shot') {
          expect(meta.activePartitionVsPerLoopMaxAbsDiff).toBeLessThanOrEqual(1e-6);
        } else if (partitionMode === 'chunked') {
          expect(meta.chunkPartition).toEqual([4, 8]);
          // Chunk 0 (pos 0..3): empty cache -> zero divergence; Chunk 1 (pos 4..11): > 1e-2 divergence!
          for (let pos = 0; pos < 4; pos += 1) {
            expect(meta.perPositionDivergence[pos]!.maxAbsLogitDiff).toBeLessThanOrEqual(1e-6);
            expect(meta.perPositionDivergence[pos]!.chunkIndex).toBe(0);
          }
          expect(
            Math.max(...meta.perPositionDivergence.slice(4).map((p) => p.maxAbsLogitDiff)),
          ).toBeGreaterThan(1e-2);

          // Verify historical writer-loop provenance in `last_loop_writer_loops` vs `per_loop_writer_loops`
          const plWriters = readAtlasRegionRG(sc, 'per_loop_writer_loops').r;
          const llWriters = readAtlasRegionRG(sc, 'last_loop_writer_loops').r;
          // At t = 1 on chunk 1 ([4..11]): past keys [0..3] have writer = 4 in last_loop vs 1 in per_loop!
          expect(Array.from(plWriters.slice(0, 4))).toEqual([1, 1, 1, 1]);
          expect(Array.from(llWriters.slice(0, 4))).toEqual([4, 4, 4, 4]);
          expect(Array.from(llWriters.slice(4))).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
        } else {
          // one_by_one: pos 0 zero diff, pos >= 1 diverges > 1e-2
          expect(meta.perPositionDivergence[0]!.maxAbsLogitDiff).toBeLessThanOrEqual(1e-6);
          expect(meta.activePartitionVsPerLoopMaxAbsDiff).toBeGreaterThan(1e-2);
        }

        // Verify exact DR20 live G-channel masks on ALL 4 causal plates
        const expectedStartPos =
          partitionMode === 'single_shot' ? 0 : partitionMode === 'chunked' ? 4 : 11;
        const expectedSeqLen =
          partitionMode === 'single_shot' ? 12 : partitionMode === 'chunked' ? 8 : 1;
        const expectedKvAbs =
          partitionMode === 'one_by_one' && layerIdx === 16
            ? [4, 5, 6, 7, 8, 9, 10, 11] // SWA W=8 visible window for query pos 11
            : Array.from({ length: 12 }, (_, idx) => idx);
        const attnKind: DoryAttentionKind =
          layerIdx === 16 ? 'sliding_attention' : 'full_attention';

        for (const regionId of [
          'per_loop_attn_probs',
          'per_loop_attn_scores',
          'last_loop_attn_probs',
          'last_loop_attn_scores',
        ] as const) {
          assertCausalMaskExactInAtlas({
            scene: sc,
            regionId,
            seqLen: expectedSeqLen,
            kvLen: expectedKvAbs.length,
            startPos: expectedStartPos,
            kvAbsPositions: expectedKvAbs,
            attentionKind: attnKind,
            label: `kv_compare.k4.${partitionMode}.p${layerIdx}.${regionId}`,
          });
        }
      }
    }

    // 3. Verify `cap_swa = 16` ring slot tags at S = 12 (`swa_ring_tags` + `swa_ring_slot_0..15`)
    const ringScene = buildDoryKvCompareScene(callerSession, {
      kLoopsOverride: 4,
      partitionMode: 'chunked',
      physicalLayerIdx: 16,
      loopIteration: 1,
    });
    const ringTags = readAtlasRegionRG(ringScene, 'swa_ring_tags');
    expect(ringTags.rows).toBe(2);
    expect(ringTags.cols).toBe(16);
    // Slots 0..11 hold absPos 0..11 (writerLoop = 4); slots 12..15 are empty (-1)
    for (let s = 0; s < 12; s += 1) {
      expect(ringTags.r[s]).toBe(s);
      expect(ringTags.r[16 + s]).toBe(4);
    }
    for (let s = 12; s < 16; s += 1) {
      expect(ringTags.r[s]).toBe(-1);
    }
    const ringSlotBoxes = ringScene.pickBoxes.filter((pb) => /^swa_ring_slot_\d+$/.test(pb.partId ?? ''));
    expect(ringSlotBoxes).toHaveLength(16);
    expect(ringSlotBoxes.filter((pb) => (pb.semanticTag ?? '').includes('ringStatus=window_active'))).toHaveLength(8);
    expect(ringSlotBoxes.filter((pb) => (pb.semanticTag ?? '').includes('ringStatus=rollback_retained'))).toHaveLength(4);
    expect(ringSlotBoxes.filter((pb) => (pb.semanticTag ?? '').includes('ringStatus=empty'))).toHaveLength(4);

    // 4. Extrapolation depth `k = 8` (`176` per_loop banks vs `36` last_loop) + Long Chunk `S = 24 > cap_swa = 16` (DR24)
    const k8Long = buildDoryKvCompareScene(callerSession, {
      kLoopsOverride: 8,
      partitionMode: 'single_shot',
      comparisonTokens: PROMPT_32.slice(0, 24),
      physicalLayerIdx: 16,
      loopIteration: 1,
    });
    assertGate4SceneHealth(k8Long, 'kv_compare.k8.S24', DORY_KV_COMPARE_ATLAS_BUDGET_BYTES, 4);
    expect(k8Long.kvCompareMetadata!.perLoopBankCount).toBe(176);
    expect(k8Long.kvCompareMetadata!.lastLoopBankCount).toBe(36);
    expect(k8Long.kvCompareMetadata!.sInvariantRatio).toBeCloseTo(44 / 9, 12);
    expect(k8Long.kvCompareMetadata!.theorem1SingleShotMaxAbsDiff).toBeLessThanOrEqual(1e-6);

    // 5. Caller DorySession MUST remain bit-for-bit untouched across ALL View 5 builds (D6 / DR18)
    assertSessionZeroMutation(callerSession, callerRec.recordId, snapBefore, memBefore, kvBefore, 'kv_compare.caller');
  });

  it('verifies exact 1 -> 256K BF16 production scaling curve, S-invariant (4+5k)/9 ratio, S=1,024 4x slope knee, and partition helpers', () => {
    for (const k of [1, 2, 3, 4, 6, 8] as const) {
      const { perLoopCurve, lastLoopCurve, sInvariantRatio } = buildProductionKvScalingCurves(k);
      const expectedRatio = (4 + 5 * k) / 9;
      expect(sInvariantRatio).toBeCloseTo(expectedRatio, 12);
      expect(perLoopCurve).toHaveLength(DORY_PRODUCTION_KV_SCALING_LENGTHS.length);
      expect(lastLoopCurve).toHaveLength(DORY_PRODUCTION_KV_SCALING_LENGTHS.length);

      for (let i = 0; i < DORY_PRODUCTION_KV_SCALING_LENGTHS.length; i += 1) {
        const s = DORY_PRODUCTION_KV_SCALING_LENGTHS[i]!;
        const pl = perLoopCurve[i]!;
        const ll = lastLoopCurve[i]!;
        expect(pl.seqLen).toBe(s);
        expect(ll.seqLen).toBe(s);
        expect(pl.totalBytes / ll.totalBytes).toBeCloseTo(expectedRatio, 12);
        expect(pl.totalBytes).toBe(computeProductionKvBytesBf16(s, k, 'per_loop').totalBytes);
        expect(ll.totalBytes).toBe(computeProductionKvBytesBf16(s, k, 'last_loop').totalBytes);
      }

      // Exact 4x marginal bytes/token slope drop across S = 1,024 knee
      const p512 = perLoopCurve.find((pt) => pt.seqLen === 512)!;
      const p1024 = perLoopCurve.find((pt) => pt.seqLen === 1024)!;
      const p2048 = perLoopCurve.find((pt) => pt.seqLen === 2048)!;
      const slopePreKnee = (p1024.totalBytes - p512.totalBytes) / 512;
      const slopePostKnee = (p2048.totalBytes - p1024.totalBytes) / 1024;
      expect(slopePreKnee / slopePostKnee).toBeCloseTo(4.0, 12);
      expect(p1024.swaShare).toBeCloseTo(0.75, 12);
    }

    // Audit §7.2 exact byte milestones at k = 4
    const k4Curves = buildProductionKvScalingCurves(4);
    const knee4 = k4Curves.perLoopCurve.find((pt) => pt.seqLen === DORY_PRODUCTION_SPEC.slidingWindow)!;
    const max4 = k4Curves.perLoopCurve.find((pt) => pt.seqLen === 262_144)!;
    const maxLast = k4Curves.lastLoopCurve.find((pt) => pt.seqLen === 262_144)!;
    expect(knee4.totalBytes).toBe(384 * 1024 * 1024); // 384.00 MiB
    expect(max4.totalBytes).toBe(24_864 * 1024 * 1024); // 24,864.00 MiB
    expect(maxLast.totalBytes).toBe(9_324 * 1024 * 1024); // 9,324.00 MiB
    expect(max4.swaShare).toBeCloseTo((3 * 1024) / (3 * 1024 + 262_144), 6);

    // Partition helpers
    expect(buildDefaultKvChunkPartition(1)).toEqual([1]);
    expect(buildDefaultKvChunkPartition(3)).toEqual([1, 2]);
    expect(buildDefaultKvChunkPartition(4)).toEqual([2, 2]);
    expect(buildDefaultKvChunkPartition(12)).toEqual([4, 8]);
    expect(() => buildDefaultKvChunkPartition(0)).toThrow(RangeError);
  });
});

// ============================================================================
// 4. VIEW 6 (`self_spec`) — CONTRACT A 104-BANK DUAL-CODA PROVENANCE, ROLLBACK & DR20 PLATES
// ============================================================================

describe('Gate 4 View 6 (self_spec) — Contract A 104-Bank Dual-Coda Provenance, Rollback & DR20 Plates', () => {
  it('verifies Contract A 104-bank provenance badge (DR23), exact bank partition (8+20+60+8+8), +20 KiB Shallow Coda accounting, binary64 rejection/residual, SWA rollback (DR19), and live DR20 G-channel plates', () => {
    // 1. Live self_spec session running stochastic T = 1.0 cycles crossing S > W = 8 SWA eviction
    const specSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12, // L = 12 > W = 8 -> exercises eviction-safe rollback!
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: 4,
        sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
        seed: 0xd0415501,
      },
    });
    const cycleRec = specSession.step();
    expect(cycleRec.kind).toBe('spec_cycle');

    const snapBefore = specSession.getSnapshot();
    const memBefore = specSession.measureMemory();
    const kvBefore = captureDeepKvSnapshot(specSession.reconstructKvCacheAtRecord(cycleRec.recordId));

    const swaSpecScene = buildDorySelfSpecScene(specSession, {
      selectedAttentionLayerIdx: 16, // SWA W = 8
      recordId: cycleRec.recordId,
    });
    const fullSpecScene = buildDorySelfSpecScene(specSession, {
      selectedAttentionLayerIdx: 22, // Full causal
      recordId: cycleRec.recordId,
    });

    assertGate4SceneHealth(swaSpecScene, 'self_spec.live.swa_p16', DORY_SELF_SPEC_ATLAS_BUDGET_BYTES, 2);
    assertGate4SceneHealth(fullSpecScene, 'self_spec.live.full_p22', DORY_SELF_SPEC_ATLAS_BUDGET_BYTES, 2);

    // Zero mutation of live self_spec session
    assertSessionZeroMutation(specSession, cycleRec.recordId, snapBefore, memBefore, kvBefore, 'self_spec.live');

    const meta = swaSpecScene.selfSpecMetadata!;

    // Contract A Dual-Coda epistemic disclosure badge (`DR23`)
    expect(meta.contractBadge).toBe(DORY_SPEC_CONTRACT_A_BADGE);
    expect(meta.contractBadge).toContain('Reconstructed from Paper §6.2');
    expect(meta.contractBadge).toContain('Contract A');
    expect(meta.contractBadge).toContain('Not in Upstream Runtimes');
    expect(meta.contractBadge).toContain('per_loop Only');
    expect(swaSpecScene.subtitle).toContain(DORY_SPEC_CONTRACT_A_BADGE);
    expect(
      swaSpecScene.provenanceBadges.some(
        (b) => b.label === DORY_SPEC_CONTRACT_A_BADGE && b.discrepancyId === 'DR23',
      ),
    ).toBe(true);
    const provCallout = swaSpecScene.callouts.find((c) => c.id === 'callout_spec_contract_a_provenance');
    expect(provCallout).toBeDefined();
    expect(provCallout!.title).toBe(DORY_SPEC_CONTRACT_A_BADGE);
    expect(provCallout!.discrepancyIds).toContain('DR23');

    // Exact 104-bank Contract A partition (`8 + 20 + 60 + 8 + 8 = 104`) & byte accounting (`+20 KiB` -> `260 KiB`)
    expect(meta.totalBankCount).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
    expect(meta.totalBankCount).toBe(104);
    expect(meta.inputBankCount).toBe(8);
    expect(meta.loop1SharedBankCount).toBe(20);
    expect(meta.loops2To4VerifierBankCount).toBe(60);
    expect(meta.outputDeepBankCount).toBe(8);
    expect(meta.outputShallowBankCount).toBe(8);
    expect(
      meta.inputBankCount +
        meta.loop1SharedBankCount +
        meta.loops2To4VerifierBankCount +
        meta.outputDeepBankCount +
        meta.outputShallowBankCount,
    ).toBe(104);
    expect(meta.shallowCodaExtraBytes).toBe(20_480); // +20.00 KiB
    expect(meta.totalKvPayloadBytes).toBe(266_240); // 260.00 KiB
    expect(meta.isSynthesizedPreview).toBe(false);
    expect(meta.divisionFreeBinary64Rule).toBe(DORY_DIVISION_FREE_BINARY64_RULE);
    expect(meta.hasLiveCausalMaskGChannel).toBe(true);
    expect(meta.swaRingCapacity).toBe(16);
    expect(meta.slidingWindow).toBe(8);

    // Verify ALL 104 bank PickBoxes exist and carry exact bankKind & cacheRole provenance
    const bankBoxes = swaSpecScene.pickBoxes.filter((pb) => /^spec_bank_\d+$/.test(pb.partId ?? ''));
    expect(bankBoxes).toHaveLength(104);
    const roleCounts = {
      input_reused: 0,
      loop1_reused: 0,
      loops2to4_deep: 0,
      output_deep: 0,
      output_shallow: 0,
    };
    for (const pb of bankBoxes) {
      const dec = pickHitToDoryCell({ ...pb, t: 1, point: [0, 0, 0], isFascia: false }, 'self_spec')!;
      expect(dec.kvPolicy).toBe('per_loop');
      expect(dec.modelWeightId).toBe(PARAMS.modelWeightId);
      if (dec.kvBankKind === 'input') {
        expect(dec.cacheRole).toBe('spec_verify_reused');
        roleCounts.input_reused += 1;
      } else if (dec.kvBankKind === 'recurrent' && dec.loopIteration === 1) {
        expect(dec.cacheRole).toBe('spec_verify_reused');
        roleCounts.loop1_reused += 1;
      } else if (dec.kvBankKind === 'recurrent') {
        expect(dec.cacheRole).toBe('spec_verify_deep');
        roleCounts.loops2to4_deep += 1;
      } else if (dec.kvBankKind === 'output_deep') {
        expect(dec.cacheRole).toBe('spec_verify_deep');
        roleCounts.output_deep += 1;
      } else if (dec.kvBankKind === 'output_shallow') {
        expect(dec.cacheRole).toBe('spec_draft');
        roleCounts.output_shallow += 1;
      }
    }
    expect(roleCounts).toEqual({
      input_reused: 8,
      loop1_reused: 20,
      loops2to4_deep: 60,
      output_deep: 8,
      output_shallow: 8,
    });

    // Verify exact DR20 live G-channel mask on BOTH batched verify plates (SWA & Full)
    const reconstructed = specSession.reconstructStepTrace(cycleRec.recordId);
    const verifyTrace = reconstructed.specCycleResult!.verifyTrace!;
    const swaVerifyAttn = verifyTrace.executedSublayers.find(
      (r) => r.coord.physicalLayerIdx === 16 && r.coord.loopIteration === 4,
    )!.trace;
    const fullVerifyAttn = verifyTrace.executedSublayers.find(
      (r) => r.coord.physicalLayerIdx === 22 && r.coord.loopIteration === 4,
    )!.trace;
    expect(swaVerifyAttn.kind).toBe('attention');
    expect(fullVerifyAttn.kind).toBe('attention');
    if (swaVerifyAttn.kind === 'attention' && fullVerifyAttn.kind === 'attention') {
      for (const regionId of ['spec_verify_attn_probs', 'spec_verify_attn_scores'] as const) {
        assertCausalMaskExactInAtlas({
          scene: swaSpecScene,
          regionId,
          seqLen: swaVerifyAttn.seqLen,
          kvLen: swaVerifyAttn.kvLen,
          startPos: swaVerifyAttn.startPos,
          kvAbsPositions: swaVerifyAttn.kvAbsPositions,
          attentionKind: 'sliding_attention',
          label: `self_spec.live.swa.${regionId}`,
        });
        assertCausalMaskExactInAtlas({
          scene: fullSpecScene,
          regionId,
          seqLen: fullVerifyAttn.seqLen,
          kvLen: fullVerifyAttn.kvLen,
          startPos: fullVerifyAttn.startPos,
          kvAbsPositions: fullVerifyAttn.kvAbsPositions,
          attentionKind: 'full_attention',
          label: `self_spec.live.full.${regionId}`,
        });
      }
    }

    // Verify SWA ring rollback slot states (`cap_swa = 16`, `computeSpecSwaRingSlotStates`)
    expect(meta.swaRingSlots).toHaveLength(16);
    const rolledSlots = meta.swaRingSlots.filter((s) => s.status === 'rolled_back');
    expect(rolledSlots.length).toBe(meta.rolledBackPositions);
    // Also verify pure helper directly on a controlled rejection/rollback state
    const bank16 = reconstructed.postStepKvCache.allBanks.find(
      (b) => b.physicalLayerIdx === 16 && b.loopSlot === 1,
    )!;
    const directRing = computeSpecSwaRingSlotStates(
      bank16,
      reconstructed.specCycleResult!.startCommittedLen,
      reconstructed.specCycleResult!.endCommittedLen,
      reconstructed.specCycleResult!.startCommittedLen + reconstructed.specCycleResult!.verifiedPositions,
    );
    expect(directRing).toEqual(meta.swaRingSlots);

    // Verify BOTH Contract A Coda regimes deterministically across temperature/seed fixtures:
    //   (a) Rejection (`a < gamma'`): binary64 residual distribution non-empty, SWA rollback > 0,
    //       and `shallowCodaCatchupExecuted === false` (0 extra Shallow Coda work!).
    //   (b) Full acceptance (`a === gamma'`, non-EOS): `shallowCodaCatchupExecuted === true`
    //       (1-token Shallow Coda catch-up on cached `s_1^{(L + gamma')}`).
    let observedRejectionWithResidual =
      meta.residualTopTokens.length > 0 &&
      meta.rolledBackPositions > 0 &&
      !meta.shallowCodaCatchupExecuted;
    let observedFullAcceptCatchup =
      meta.cycleSummary.acceptedDraftCount === meta.cycleSummary.draftCount &&
      meta.shallowCodaCatchupExecuted;

    const regimeFixtures = [
      // Sharp temperature amplifies q (k=1) vs p (k=4) differences -> stochastic rejection + residual + rollback
      { prompt: PROMPT_12, temp: 0.15, seed: 0xd0415511 },
      { prompt: PROMPT_12, temp: 0.25, seed: 0xd0415512 },
      { prompt: PROMPT_8, temp: 0.2, seed: 0xd0415513 },
      { prompt: PROMPT_8, temp: 0.35, seed: 0xd0415514 },
      // Higher temperature keeps q ≈ p -> exercises full acceptance & 1-token Shallow Coda catch-up
      { prompt: PROMPT_12, temp: 1.5, seed: 0xd0415521 },
      { prompt: PROMPT_12, temp: 2.0, seed: 0xd0415522 },
      { prompt: PROMPT_8, temp: 2.0, seed: 0xd0415523 },
      { prompt: PROMPT_8, temp: 3.0, seed: 0xd0415524 },
    ] as const;

    for (const fx of regimeFixtures) {
      if (observedRejectionWithResidual && observedFullAcceptCatchup) break;
      const s = createDorySession({
        params: PARAMS,
        ropeTables: ROPE_TABLES,
        initialBranch: {
          promptTokens: fx.prompt,
          executionMode: 'self_spec',
          kvCacheMode: 'per_loop',
          kLoops: 4,
          speculativeGamma: 4,
          sampling: { temperature: fx.temp, topK: 0, topP: 1.0 },
          seed: fx.seed,
        },
      });
      if (s.getActiveBranch()?.haltReason !== null) continue;
      const rec = s.step();
      const sc = buildDorySelfSpecScene(s, { recordId: rec.recordId });
      assertGate4SceneHealth(
        sc,
        `self_spec.T${fx.temp}_${fx.seed.toString(16)}`,
        DORY_SELF_SPEC_ATLAS_BUDGET_BYTES,
        2,
      );
      const sm = sc.selfSpecMetadata!;
      if (sm.residualTopTokens.length > 0 && sm.rolledBackPositions > 0 && !sm.shallowCodaCatchupExecuted) {
        observedRejectionWithResidual = true;
        // Residual top tokens sorted descending and satisfy exact positive residual probability
        for (let i = 1; i < sm.residualTopTokens.length; i += 1) {
          expect(sm.residualTopTokens[i - 1]!.residualProb).toBeGreaterThanOrEqual(
            sm.residualTopTokens[i]!.residualProb,
          );
        }
        expect(sc.pickBoxes.some((pb) => pb.partId === 'spec_residual_dist')).toBe(true);
        const rolledRingSlots = sm.swaRingSlots.filter((sl) => sl.status === 'rolled_back');
        expect(rolledRingSlots.length).toBe(sm.rolledBackPositions);
      }
      if (sm.cycleSummary.acceptedDraftCount === sm.cycleSummary.draftCount && sm.shallowCodaCatchupExecuted) {
        observedFullAcceptCatchup = true;
        expect(sm.rolledBackPositions).toBe(0);
      }
    }
    expect(observedRejectionWithResidual).toBe(true);
    expect(observedFullAcceptCatchup).toBe(true);
  });

  it('synthesizes an isolated Contract A preview cycle on standard & spec_prefill sessions with bit-exact zero caller mutation', () => {
    const stdSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_8,
        kLoops: 2,
        kvCacheMode: 'last_loop', // even when active session is k=2 last_loop, View 6 synthesizes clean Contract A per_loop preview!
        executionMode: 'standard',
      },
    });
    const rec = stdSession.step();
    const snapBefore = stdSession.getSnapshot();
    const memBefore = stdSession.measureMemory();
    const kvBefore = captureDeepKvSnapshot(stdSession.reconstructKvCacheAtRecord(rec.recordId));

    const previewA = buildDorySelfSpecScene(stdSession, {
      selectedAttentionLayerIdx: 16,
      previewSampling: { temperature: 0, topK: 0, topP: 1.0 },
    });
    const previewB = buildDorySelfSpecScene(stdSession, {
      selectedAttentionLayerIdx: 16,
      previewSampling: { temperature: 0, topK: 0, topP: 1.0 },
    });

    assertGate4SceneHealth(previewA, 'self_spec.preview', DORY_SELF_SPEC_ATLAS_BUDGET_BYTES, 2);
    expect(previewA.atlasKey).toBe(previewB.atlasKey);
    expect(previewA.selfSpecMetadata!.isSynthesizedPreview).toBe(true);
    expect(previewA.selfSpecMetadata!.totalBankCount).toBe(104);
    expect(previewA.selfSpecMetadata!.contractBadge).toBe(DORY_SPEC_CONTRACT_A_BADGE);
    expect(previewA.kvCacheMode).toBe('per_loop');
    expect(previewA.kLoops).toBe(4);
    expect(previewA.executionMode).toBe('self_spec');
    expect(
      previewA.provenanceBadges.some((b) => b.label.includes('Isolated Read-Only Preview')),
    ).toBe(true);

    // Caller standard last_loop session remains 0-ULP untouched!
    assertSessionZeroMutation(stdSession, rec.recordId, snapBefore, memBefore, kvBefore, 'self_spec.preview');
  });
});

// ============================================================================
// 5. REAL 3D RAYCAST (`pickMacroBay`) -> `pickHitToDoryCell` ROUND-TRIPS ACROSS VIEWS 3–6 (`DR21`)
// ============================================================================

describe('Gate 4 DR21 Real Raycast Picking — pickMacroBay -> pickHitToDoryCell Round-Trips & Popovers Across Views 3–6', () => {
  const raycastPart = (
    scene: DorySpatialScene,
    partId: string,
    row = 0,
    col = 0,
    isolateBox = false,
  ) => {
    const box = scene.pickBoxes.find((pb) => pb.partId === partId);
    expect(box, `Missing PickBox '${partId}' in ${scene.viewMode}`).toBeDefined();
    const ray = makeDirectRayAtBox(box!, row, col, 4.0);
    const hit = pickMacroBay(ray, isolateBox ? [box!] : scene.pickBoxes, []);
    expect(hit, `Raycast missed '${partId}' in ${scene.viewMode}`).not.toBeNull();
    expect(hit!.partId).toBe(partId);
    const cell = pickHitToDoryCell(hit, scene.viewMode);
    expect(cell, `pickHitToDoryCell returned null for '${partId}'`).not.toBeNull();
    return { box: box!, cell: cell! };
  };

  it('round-trips View 3 (gqa_rope) & View 4 (swiglu) raycasts, cell coordinates, and per-cell popovers', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });

    // --- View 3 (gqa_rope) ---
    const gqaScene = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 2,
      selectedQHead: 5,
    });

    // 1. Causal Score & Softmax plates (both visible AND SWA-masked cells!)
    const scoreH5Vis = raycastPart(gqaScene, 'gqa_scores_h5', 9, 4); // q=9, k=4 -> diff=5 < 8 visible
    expect(scoreH5Vis.cell.physicalLayerIdx).toBe(16);
    expect(scoreH5Vis.cell.pairedLayerIdx).toBe(8);
    expect(scoreH5Vis.cell.loopIteration).toBe(2);
    expect(scoreH5Vis.cell.qHeadIndex).toBe(5);
    expect(scoreH5Vis.cell.kvHeadIndex).toBe(1);
    expect(scoreH5Vis.cell.row).toBe(9);
    expect(scoreH5Vis.cell.col).toBe(4);
    expect(scoreH5Vis.cell.modelWeightId).toBe(PARAMS.modelWeightId);

    const gqaVisPop = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 2,
      selectedQHead: 5,
      hoveredCell: scoreH5Vis.cell,
    });
    expect(gqaVisPop.activeCellPopover).not.toBeNull();
    expect(gqaVisPop.activeCellPopover!.badge).toContain('DR20 LIVE RG32F G=0');
    const expectedVisWorld = computeSheetCellWorldPos(
      {
        originWorld: scoreH5Vis.box.frame!.center,
        extent: scoreH5Vis.box.frame!.extent,
        axisU: scoreH5Vis.box.frame!.axisU,
        axisN: scoreH5Vis.box.frame!.axisN,
        cols: 12,
        rows: 12,
      },
      4,
      9,
      0.5,
    );
    expect(gqaVisPop.activeCellPopover!.world[0]).toBeCloseTo(expectedVisWorld[0], 5);
    expect(gqaVisPop.activeCellPopover!.world[2]).toBeCloseTo(expectedVisWorld[2], 5);

    // SWA-evicted past cell (q=10, k=0 -> diff=10 >= 8 -> G=1 masked!)
    const probH0Evicted = raycastPart(gqaScene, 'gqa_probs_h0', 10, 0);
    expect(probH0Evicted.cell.qHeadIndex).toBe(0);
    expect(probH0Evicted.cell.kvHeadIndex).toBe(0);
    const gqaMaskPop = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 2,
      selectedQHead: 5,
      hoveredCell: probH0Evicted.cell,
    });
    expect(gqaMaskPop.activeCellPopover!.badge).toContain('DR20 LIVE RG32F G=1 MASKED');
    expect(gqaMaskPop.activeCellPopover!.badge).toContain('W=8');

    // 2. Dual-RoPE 3D phasor pedestal (`gqa_rope_plane_2`), sqk^2, s_gate^2, attn_sub_norm, NLERP
    const ropePlane2 = raycastPart(gqaScene, 'gqa_rope_plane_2');
    expect(ropePlane2.cell.rotaryPlane).toBe(2);
    expect(ropePlane2.cell.physicalLayerIdx).toBe(16);
    const ropePop = buildDoryGqaRopeScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 2,
      selectedQHead: 5,
      hoveredCell: ropePlane2.cell,
    });
    expect(ropePop.activeCellPopover!.tensorLabel).toContain('Plane j=2');
    expect(ropePop.activeCellPopover!.badge).toContain('DR6');

    for (const target of [
      { partId: 'gqa_sqk_eff_sq', row: 0, col: 19, badgeSubstr: 'SQUARED ON Q ONLY' },
      { partId: 'gqa_s_gate_eff_sq', row: 0, col: 25, badgeSubstr: 'DR3' },
      { partId: 'gqa_gate_values', row: 2, col: 10, badgeSubstr: 'SIGMOID GATE' },
      { partId: 'gqa_attn_sub_norm', row: 3, col: 41, badgeSubstr: '64-DIM CONCAT' },
      { partId: 'gqa_attn_alpha_eff', row: 0, col: 6, badgeSubstr: 'DR7 UNCLAMPED' },
      { partId: 'gqa_nlerp_output', row: 1, col: 7, badgeSubstr: 'DR7 UNCLAMPED' },
    ] as const) {
      const picked = raycastPart(gqaScene, target.partId, target.row, target.col);
      expect(picked.cell.row).toBe(target.row);
      expect(picked.cell.col).toBe(target.col);
      const withPop = buildDoryGqaRopeScene(session, {
        physicalLayerIdx: 16,
        loopIteration: 2,
        selectedQHead: 5,
        hoveredCell: picked.cell,
      });
      expect(withPop.activeCellPopover, `Missing popover for ${target.partId}`).not.toBeNull();
      expect(withPop.activeCellPopover!.badge).toContain(target.badgeSubstr);
    }

    // --- View 4 (swiglu) ---
    const swigluScene = buildDorySwigluScene(session, {
      physicalLayerIdx: 17,
      loopIteration: 3,
      selectedTokenRow: 2,
      selectedOutputDim: 1,
      accumReveal: 0.5,
    });

    const mlp17 = PARAMS.layers[17]!;
    expect(mlp17.kind).toBe('mlp');
    const negGateCol = mlp17.kind === 'mlp' ? Array.from(mlp17.suvGate).findIndex((v) => v < 0) : 2;
    const posGateCol = mlp17.kind === 'mlp' ? Array.from(mlp17.suvGate).findIndex((v) => v > 0) : 0;
    const negUpCol = mlp17.kind === 'mlp' ? Array.from(mlp17.suvUp).findIndex((v) => v < 0) : 0;
    const posUpCol = mlp17.kind === 'mlp' ? Array.from(mlp17.suvUp).findIndex((v) => v > 0) : 1;

    const swigluSheetTargets = [
      { partId: 'swiglu_suv_gate_eff', row: 0, col: negGateCol, badgeSubstr: 'SIGNED NEGATIVE · NO .abs()', isolate: true },
      { partId: 'swiglu_suv_gate_eff', row: 0, col: posGateCol, badgeSubstr: 'FAMILY 4 · INIT 1.0', isolate: true },
      { partId: 'swiglu_suv_up_eff', row: 0, col: negUpCol, badgeSubstr: 'SIGNED NEGATIVE · NO .abs()', isolate: true },
      { partId: 'swiglu_suv_up_eff', row: 0, col: posUpCol, badgeSubstr: 'FAMILY 4 · INIT 1.0', isolate: true },
      { partId: 'swiglu_product', row: 2, col: 7, badgeSubstr: 'PRE-ffn_sub_norm', isolate: false },
      { partId: 'swiglu_ffn_sub_norm', row: 2, col: 15, badgeSubstr: 'PRE-down_proj', isolate: false },
      { partId: 'swiglu_accum_terms', row: 0, col: 5, badgeSubstr: 'ACCUMULATED TERM', isolate: false },
      { partId: 'swiglu_accum_terms', row: 0, col: 18, badgeSubstr: 'UNREVEALED', isolate: false },
      { partId: 'swiglu_down_proj_out', row: 2, col: 1, badgeSubstr: 'ACCUM REVEAL 12/24', isolate: false },
      { partId: 'swiglu_nlerp_output', row: 2, col: 4, badgeSubstr: DORY_NLERP_MARKER, isolate: false },
    ] as const;

    for (const target of swigluSheetTargets) {
      const picked = raycastPart(swigluScene, target.partId, target.row, target.col, target.isolate);
      expect(picked.cell.physicalLayerIdx).toBe(17);
      expect(picked.cell.pairedLayerIdx).toBe(8);
      expect(picked.cell.loopIteration).toBe(3);
      expect(picked.cell.row).toBe(target.row);
      expect(picked.cell.col).toBe(target.col);
      const withPop = buildDorySwigluScene(session, {
        physicalLayerIdx: 17,
        loopIteration: 3,
        selectedTokenRow: 2,
        selectedOutputDim: 1,
        accumReveal: 0.5,
        hoveredCell: picked.cell,
      });
      expect(withPop.activeCellPopover, `Missing popover for ${target.partId}`).not.toBeNull();
      expect(withPop.activeCellPopover!.badge).toContain(target.badgeSubstr);
    }

    // Hovering a raised 3D negative signed suv bar returns the signed negative popover
    const negSuvBox = swigluScene.pickBoxes.find((pb) =>
      /^swiglu_neg_suv_gate_\d+$/.test(pb.partId ?? ''),
    )!;
    expect(negSuvBox).toBeDefined();
    const negPicked = raycastPart(swigluScene, negSuvBox.partId!);
    const negPop = buildDorySwigluScene(session, {
      physicalLayerIdx: 17,
      loopIteration: 3,
      hoveredCell: negPicked.cell,
    });
    expect(negPop.activeCellPopover!.badge).toContain('FAMILY 4 LINEAR SIGNED');
    expect(negPop.activeCellPopover!.coordLabel).toContain('NEGATIVE SIGNED');
  });

  it('round-trips View 5 (kv_compare) & View 6 (self_spec) raycasts, bank/ring provenance, and popovers', () => {
    const specSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: 4,
        sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
        seed: 0xd0415502,
      },
    });
    const specRec = specSession.step();

    // --- View 5 (kv_compare) ---
    const kvScene = buildDoryKvCompareScene(specSession, {
      kLoopsOverride: 4,
      partitionMode: 'chunked',
      physicalLayerIdx: 16,
      loopIteration: 1,
    });

    // Bank solids carry exact bankSlotId, bankKind, policy, loop
    const plBank8 = raycastPart(kvScene, 'per_loop_bank_8');
    expect(plBank8.cell.kvPolicy).toBe('per_loop');
    expect(plBank8.cell.kvBankSlotId).toBe(8);
    expect(plBank8.cell.kvBankKind).toBe('recurrent');
    expect(plBank8.cell.loopIteration).toBe(1);
    expect(plBank8.cell.physicalLayerIdx).toBe(16);

    const llBank8 = raycastPart(kvScene, 'last_loop_bank_8');
    expect(llBank8.cell.kvPolicy).toBe('last_loop');
    expect(llBank8.cell.kvBankSlotId).toBe(8);
    expect(llBank8.cell.kvBankKind).toBe('recurrent');

    // Locate peak Theorem 3 (> 1e-2) divergent position in chunk 1 (`pos >= 4`)
    const peakDivPt = kvScene.kvCompareMetadata!.perPositionDivergence.reduce((best, pt) =>
      pt.maxAbsLogitDiff > best.maxAbsLogitDiff ? pt : best,
    );
    expect(peakDivPt.absPos).toBeGreaterThanOrEqual(4);
    expect(peakDivPt.maxAbsLogitDiff).toBeGreaterThan(1e-2);

    // Causal sheets, diff sheet, writer strips, SWA ring slots & divergence bars (both chunk-0 & chunk-1!)
    for (const target of [
      { partId: 'per_loop_attn_probs', row: 2, col: 5, badgeSubstr: 'DR20 RG32F G=0' },
      { partId: 'last_loop_attn_probs', row: 2, col: 2, badgeSubstr: 'DoryPastReader' },
      { partId: 'last_loop_attn_scores', row: 0, col: 11, badgeSubstr: 'DR20 RG32F G=1 MASKED' },
      { partId: 'attn_probs_diff', row: 3, col: 4, badgeSubstr: 'DR17 HISTORICAL KV' },
      { partId: 'last_loop_writer_loops', row: 0, col: 1, badgeSubstr: 'FINAL-LOOP HISTORY (t=4)' },
      { partId: 'swa_ring_tags', row: 0, col: 5, badgeSubstr: 'cap_swa = 16' },
      { partId: 'partition_divergence_matrix', row: 1, col: peakDivPt.absPos, badgeSubstr: 'THEOREM 3' },
      { partId: 'swa_ring_slot_11', row: 0, col: 0, badgeSubstr: 'ACTIVE SWA WINDOW' },
      { partId: 'div_bar_0', row: 0, col: 0, badgeSubstr: 'ZERO-DIVERGENCE PREFIX' },
      { partId: `div_bar_${peakDivPt.absPos}`, row: 0, col: 0, badgeSubstr: 'THEOREM 3 DIVERGENCE' },
    ] as const) {
      const picked = raycastPart(kvScene, target.partId, target.row, target.col);
      const withPop = buildDoryKvCompareScene(specSession, {
        kLoopsOverride: 4,
        partitionMode: 'chunked',
        physicalLayerIdx: 16,
        loopIteration: 1,
        hoveredCell: picked.cell,
      });
      expect(withPop.activeCellPopover, `Missing popover for ${target.partId}`).not.toBeNull();
      expect(withPop.activeCellPopover!.badge).toContain(target.badgeSubstr);
    }

    // --- View 6 (self_spec) ---
    const specScene = buildDorySelfSpecScene(specSession, {
      selectedAttentionLayerIdx: 16,
      recordId: specRec.recordId,
    });

    // Contract A 5-group bank picking (`input`, `loop1`, `loops2..4`, `output_deep`, `output_shallow`)
    const b0 = raycastPart(specScene, 'spec_bank_0');
    expect(b0.cell.kvBankKind).toBe('input');
    expect(b0.cell.cacheRole).toBe('spec_verify_reused');
    expect(b0.cell.kvBankSlotId).toBe(0);

    const b8 = raycastPart(specScene, 'spec_bank_8');
    expect(b8.cell.kvBankKind).toBe('recurrent');
    expect(b8.cell.loopIteration).toBe(1);
    expect(b8.cell.cacheRole).toBe('spec_verify_reused');

    const b28 = raycastPart(specScene, 'spec_bank_28');
    expect(b28.cell.kvBankKind).toBe('recurrent');
    expect(b28.cell.loopIteration).toBe(2);
    expect(b28.cell.cacheRole).toBe('spec_verify_deep');

    const b88 = raycastPart(specScene, 'spec_bank_88');
    expect(b88.cell.kvBankKind).toBe('output_deep');
    expect(b88.cell.cacheRole).toBe('spec_verify_deep');

    const b96 = raycastPart(specScene, 'spec_bank_96');
    expect(b96.cell.kvBankKind).toBe('output_shallow');
    expect(b96.cell.cacheRole).toBe('spec_draft');
    expect(b96.cell.kvBankSlotId).toBe(96);

    // Position q/p bars, SWA ring slots, top-8 probs, batched verify DR20 plates & optional residual sheet
    const posQ0 = raycastPart(specScene, 'spec_pos_bar_0_q');
    expect(posQ0.cell.specOffset).toBe(0);
    expect(posQ0.cell.cacheRole).toBe('spec_draft');

    const posP0 = raycastPart(specScene, 'spec_pos_bar_0_p');
    expect(posP0.cell.specOffset).toBe(0);
    expect(posP0.cell.cacheRole).toBe('spec_verify_deep');

    const specTargets: { readonly partId: string; readonly row: number; readonly col: number; readonly badgeSubstr: string }[] = [
      { partId: 'spec_pos_bar_0_q', row: 0, col: 0, badgeSubstr: '' },
      { partId: 'spec_prob_comparison', row: 1, col: 0, badgeSubstr: DORY_DIVISION_FREE_BINARY64_RULE },
      { partId: 'spec_verify_attn_probs', row: 0, col: 7, badgeSubstr: 'DR20 RG32F' },
      { partId: 'spec_verify_attn_scores', row: 0, col: 7, badgeSubstr: 'DR20 RG32F' },
      { partId: 'spec_swa_slot_0', row: 0, col: 0, badgeSubstr: '' },
    ];
    if (specScene.selfSpecMetadata!.residualTopTokens.length > 0) {
      specTargets.push({
        partId: 'spec_residual_dist',
        row: 2,
        col: 0,
        badgeSubstr: 'EXACT BINARY64 RESIDUAL',
      });
    }
    for (const target of specTargets) {
      const picked = raycastPart(specScene, target.partId, target.row, target.col);
      const withPop = buildDorySelfSpecScene(specSession, {
        selectedAttentionLayerIdx: 16,
        recordId: specRec.recordId,
        hoveredCell: picked.cell,
      });
      expect(withPop.activeCellPopover, `Missing popover for ${target.partId}`).not.toBeNull();
      if (target.badgeSubstr.length > 0) {
        expect(withPop.activeCellPopover!.badge).toContain(target.badgeSubstr);
      }
      for (const coord of withPop.activeCellPopover!.world) {
        expect(Number.isFinite(coord)).toBe(true);
      }
    }
  });
});

// ============================================================================
// 6. DR20 TAMPER NEGATIVE CONTROLS ON VIEWS 3/5/6 & D4 ARCHITECTURAL ISOLATION
// ============================================================================

describe('Gate 4 Provenance & Isolation Guards — DR20 Tamper Controls, Untouched pick.ts & D4 Zero Forbidden Imports', () => {
  it('enforces DR20 hard-throw when causal sheets on live Views 3, 5, and 6 scenes are tampered with', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: PROMPT_12,
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });

    const liveCausalScenes: readonly DorySpatialScene[] = [
      buildDoryGqaRopeScene(session, { physicalLayerIdx: 16, loopIteration: 1 }),
      buildDoryKvCompareScene(session, { kLoopsOverride: 2, partitionMode: 'chunked', physicalLayerIdx: 16 }),
      buildDorySelfSpecScene(session, { selectedAttentionLayerIdx: 16 }),
    ];

    for (const scene of liveCausalScenes) {
      // Locate the first causal score/softmax sheet instance in this real scene
      let causalSheetIdx = -1;
      for (let i = 0; i < scene.sheetCount; i += 1) {
        const kind = Math.round(scene.sheetData[i * SHEET_INSTANCE_FLOATS + 8]!);
        if (kind === SHEET_KIND_CAUSAL_SCORES || kind === SHEET_KIND_CAUSAL_SOFTMAX) {
          causalSheetIdx = i;
          break;
        }
      }
      expect(causalSheetIdx, `${scene.viewMode} must have causal sheet`).toBeGreaterThanOrEqual(0);
      const base = causalSheetIdx * SHEET_INSTANCE_FLOATS;

      // (a) Dropping valueAtlas MUST throw DoryDr20CausalMaskError
      expect(() => validateSceneGeometryFinite({ ...scene, valueAtlas: null })).toThrow(
        DoryDr20CausalMaskError,
      );

      // (b) Zeroing uWidth (simulating sheets.ts:351 geometric fallback!) MUST throw DR20
      const zeroedSheetData = new Float32Array(scene.sheetData);
      zeroedSheetData[base + 15] = 0;
      expect(() => validateSceneGeometryFinite({ ...scene, sheetData: zeroedSheetData })).toThrow(
        DoryDr20CausalMaskError,
      );
      const nonThrowing = validateSceneGeometryFinite(
        { ...scene, sheetData: zeroedSheetData },
        { throwOnDr20: false },
      );
      expect(nonThrowing.valid).toBe(false);
      expect(nonThrowing.errors.some((e) => e.includes('DR20'))).toBe(true);

      // (c) Corrupting one G-channel mask texel to 0.5 inside its TexRect MUST throw DR20
      const u0 = scene.sheetData[base + 13]!;
      const v0 = scene.sheetData[base + 14]!;
      const tx = Math.floor(u0 * scene.valueAtlas!.width + 1e-4);
      const ty = Math.floor(v0 * scene.valueAtlas!.height + 1e-4);
      const corruptedData = new Float32Array(scene.valueAtlas!.data);
      corruptedData[(ty * scene.valueAtlas!.width + tx) * 2 + 1] = 0.5;
      expect(() =>
        validateSceneGeometryFinite({
          ...scene,
          valueAtlas: { ...scene.valueAtlas!, data: corruptedData },
        }),
      ).toThrow(DoryDr20CausalMaskError);
    }

    // Also verify DORY_SPATIAL_VIEW_MODES budgets for Views 3–6 all respect <= 512 KiB
    for (const viewId of ['gqa_rope', 'swiglu', 'kv_compare', 'self_spec'] as const) {
      const desc = DORY_SPATIAL_VIEW_MODES.find((v) => v.id === viewId)!;
      expect(desc).toBeDefined();
      expect(desc.atlasBudgetBytes).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
    }
  });

  it('enforces D4 zero forbidden Gate 4b/4c/sibling/Node imports, zero Math.random(), zero DOM/WebGL globals, and untouched src/render/pick.ts across all DORY4 scene modules', () => {
    const sceneFiles = [
      'src/dory/scene/types.ts',
      'src/dory/scene/provenance.ts',
      'src/dory/scene/gqaRopeScene.ts',
      'src/dory/scene/swigluScene.ts',
      'src/dory/scene/kvCompareScene.ts',
      'src/dory/scene/selfSpecScene.ts',
    ];
    const forbiddenImportPatterns = [
      /from\s+['"][^'"]*(?:ui\/(?:controls|overlay|memoryChart)|scene\/(?:spine|sheetInspector|hybridPick|cutaways|embeddingCutaway|outputCutaway|gqaCutaway|mambaCutaway|moeCutaway|gqaSpatial|mambaSpatial|moeSpatial|accumulation)|model\/(?:spec|memory|state|rcPlayback)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)|story\/walkthrough|src\/main|\.\.\/main)['"]/,
      /from\s+['"][^'"]*(?:twotower|diffusion|uni)(?:\/|['"])/,
      /from\s+['"]node:/,
    ];

    for (const relPath of sceneFiles) {
      const src = readFileSync(resolve(process.cwd(), relPath), 'utf-8');
      expect(src, `Math.random forbidden in ${relPath}`).not.toMatch(/Math\.random\s*\(/);
      expect(src, `DOM/WebGL globals forbidden in headless ${relPath}`).not.toMatch(
        /\b(?:document\s*\.|window\s*\.|globalThis\s*\.\s*(?:document|window)|typeof\s+(?:document|window)\b|WebGL2RenderingContext|HTMLCanvasElement|HTMLElement)\b/,
      );
      for (const re of forbiddenImportPatterns) {
        expect(src, `D4 forbidden import in ${relPath}`).not.toMatch(re);
      }
    }

    // DR21: src/render/pick.ts remains strictly untouched with Nano's `0 | 1` typing
    const pickSrc = readFileSync(resolve(process.cwd(), 'src/render/pick.ts'), 'utf-8');
    expect(pickSrc).toMatch(/readonly\s+kvHeadIndex\?:\s*0\s*\|\s*1;/);
    expect(pickSrc.toLowerCase()).not.toContain('dory');
  });
});
