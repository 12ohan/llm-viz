/**
 * DORY3 Slice 3C — Views 1 & 2 Headless Geometry, Atlas, Provenance & Pick Test Suite
 * (`tests/dory-scene-macro-hypersphere.test.ts`)
 *
 * Enforces every `DORY3` Gate 3 exit invariant:
 *   1. 0 NaN/Inf across all 5 WebGL2 passes, `RG32F` atlas, bounds, pick boxes, landmarks,
 *      overlay anchors, calipers, and KV ticks via `validateSceneGeometryFinite`.
 *   2. 0 atlas overlaps & 0 out-of-bounds slices via independent `auditAtlasRegions` audit,
 *      with every `RG32F` atlas strictly `<= 512 KiB` (`DORY_MAX_ATLAS_BYTES = 524,288 B`).
 *   3. `DR20` hard-throw verification in BOTH `DorySheetBufferBuilder.push()` AND
 *      `validateSceneGeometryFinite()` whenever `sheetKind in {1, 2}` omits a live `RG32F`
 *      `G`-channel mask (or binds null atlas / zero-area rect / non-binary `G` texel).
 *   4. `DR21` real 3D raycast (`pickMacroBay` from untouched `src/render/pick.ts`) ->
 *      `pickHitToDoryCell` round-trip picking across View 1 & View 2 (including arbitrary
 *      KV-head indices > 1, live sheet `(row, col)` popovers, and verifying `src/render/pick.ts`
 *      retains untouched `kvHeadIndex?: 0 | 1` with zero Dory edits).
 *   5. View 1 (`macro`) Folded vs. Unrolled topology: exact `72` physical vs. `32 + 40k`
 *      executed sublayers across `k in {1, 2, 3, 4, 6, 8}`, exact 3:1 SWA:Full motif across
 *      all 3 stages, identical singleton `modelWeightId` shared-weight rails across all `k`
 *      recurrent passes vs. distinct live `RG32F` per-loop states `s_1 .. s_k`, and parallel
 *      KV-bank rail (`36..176` `per_loop`, flat `36` `last_loop`, `104` Contract A `self_spec`).
 *   6. View 2 (`hypersphere`):
 *      - Anti-SLERP (`DR7`): `DORY_ANTI_SLERP_PLAQUE`, `dory-nlerp-residual`, planted
 *        unclamped `|alpha_hat| > 1` channels `[6, 7]` at physical layer `16`, Exact 2D
 *        Spanning-Subspace `span(a, b)` geometry, and numerical proof that actual NLERP
 *        differs from both scalar trigonometric SLERP and `[0, 1]`-clamped NLERP.
 *      - Exploratory 2D PCA (`Finding 9`): `DORY_EXPLORATORY_PCA_BADGE`, orthonormal 8D PCA
 *        axes, `projectedNorm <= exact8dNorm <= 1 + 1e-6`, strict 8D recomputation of all
 *        numeric trajectory metrics, and rank-1 `k = 1` fallback safety.
 *      - Local `120-deg` anchor-map Jacobian caveat (`Audit §6.2.1`): `DORY_ANCHOR_MAP_CAVEAT_BADGE`,
 *        callout non-convergence disclosure, and exact `0° / 60° / 120° / 179° / 180°` checks.
 *      - Read-only `Diagnostic Counterfactual` Coda (`writeKv: false`): `DORY_COUNTERFACTUAL_CODA_BADGE`,
 *        binary64 `D_KL >= 0`, `t = k` parity vs. final loop, and strict bit-for-bit (0-ULP)
 *        zero KV / session / LRU / next-step mutation proof.
 *   7. `D4` architectural boundary isolation & zero `Math.random()` across all `src/dory/scene/*.ts`.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { normalize3, sub3 } from '../src/render/camera';
import { pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import type { Vec3 } from '../src/render/primitives';
import { localToWorldOnFrame } from '../src/render/sheetFrame';
import { computeSheetCellWorldPos } from '../src/shared/overlayLayout';
import {
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
} from '../src/render/sheets';
import {
  assertBitExactFiniteArray,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';
import {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_MAX_ATLAS_BYTES,
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  DORY_PLANTED_EXTRAPOLATION_SPEC,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  anchorMapJacobianNorm,
  buildDoryDualRopeTables,
  cosineSimilarity8d,
  createDorySession,
  effectiveResidualAlpha,
  getDefaultDoryModelParameters,
  l2Distance,
  l2NormVector,
  vectorL2Norm,
  type DoryKvCache,
  type DoryKvCacheMode,
} from '../src/dory';
import {
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  type DorySpatialScene,
} from '../src/dory/scene/types';
import {
  DoryAtlasPacker,
  DoryDr20CausalMaskError,
  DorySheetBufferBuilder,
  auditAtlasRegions,
  computeDoryAtlasContentKey,
  encodeDoryPickMetadata,
  packPickBoxFromAabb,
  pickHitToDoryCell,
  resolveDorySessionContext,
  validateSceneGeometryFinite,
} from '../src/dory/scene/provenance';
import {
  DORY_MACRO_LAYOUT,
  buildDoryMacroKvRailSummary,
  buildDoryMacroScene,
  buildDorySharedWeightBindings,
  doryMacroPairRowY,
  doryMacroPhysicalLayerY,
} from '../src/dory/scene/macroScene';
import {
  DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES,
  buildDoryHypersphereScene,
  computeExploratoryPcaTrajectory,
  computeKlDivergenceFloat64,
} from '../src/dory/scene/hypersphereScene';

const PARAMS = getDefaultDoryModelParameters();
const ROPE_TABLES = buildDoryDualRopeTables(PARAMS.config);
const DEFAULT_PROMPT = [1, 10, 4, 18, 40, 25, 26, 57] as const;

// ============================================================================
// TEST HELPERS (RAYCAST, ATLAS READBACK, DEEP KV SNAPSHOT)
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
  const cy = 0.5 * (box.min[1] + box.max[1]);
  return {
    origin: [cx, cy, box.max[2] + distance],
    direction: [0, 0, -1],
  };
}

function readAtlasRegionR(scene: DorySpatialScene, regionId: string): Float32Array {
  const atlas = scene.valueAtlas;
  expect(atlas, `Missing valueAtlas for region '${regionId}'`).not.toBeNull();
  const reg = scene.atlasRegions.find((r) => r.id === regionId);
  expect(reg, `Missing atlas region '${regionId}'`).toBeDefined();
  const out = new Float32Array(reg!.rows * reg!.cols);
  for (let r = 0; r < reg!.rows; r += 1) {
    for (let c = 0; c < reg!.cols; c += 1) {
      const idx = ((reg!.y0 + r) * atlas!.width + (reg!.x0 + c)) * 2;
      out[r * reg!.cols + c] = atlas!.data[idx]!;
    }
  }
  return out;
}

function assertGate3SceneHealth(
  scene: DorySpatialScene,
  label: string,
  expectedViewBudgetBytes: number,
): void {
  const validation = validateSceneGeometryFinite(scene);
  expect(validation.valid, `${label} geometry validation: ${validation.errors.join('; ')}`).toBe(
    true,
  );
  expect(validation.errors).toEqual([]);
  expect(scene.sheetData.length).toBe(scene.sheetCount * SHEET_INSTANCE_FLOATS);

  expect(scene.valueAtlas, `${label} must carry live RG32F atlas`).not.toBeNull();
  expect(scene.atlasKey, `${label} must carry content-addressed atlasKey`).not.toBeNull();
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

function assertKvMatchesSnapshotBitExact(
  kv: DoryKvCache,
  snap: DeepKvSnapshot,
  label: string,
): void {
  expect(kv.mode).toBe(snap.mode);
  expect(kv.kLoops).toBe(snap.kLoops);
  expect(kv.includeShallowCoda).toBe(snap.includeShallowCoda);
  expect(kv.committedLen).toBe(snap.committedLen);
  expect(kv.chunkSizes).toEqual(snap.chunkSizes);
  expect(kv.allBanks.length).toBe(snap.kCopies.length);
  for (let b = 0; b < snap.kCopies.length; b += 1) {
    const bank = kv.allBanks[b]!;
    expect(bank.logicalLen).toBe(snap.bankLogicalLens[b]);
    assertBitExactFiniteArray(bank.k, snap.kCopies[b]!, `${label}.bank[${b}].k`);
    assertBitExactFiniteArray(bank.v, snap.vCopies[b]!, `${label}.bank[${b}].v`);
    assertBitExactFiniteArray(bank.absPosTag, snap.posCopies[b]!, `${label}.bank[${b}].absPosTag`);
    assertBitExactFiniteArray(
      bank.writerLoopTag,
      snap.loopCopies[b]!,
      `${label}.bank[${b}].writerLoopTag`,
    );
  }
}

// ============================================================================
// 1. VIEW 1 (`macro`) — FOLDED vs. UNROLLED, SHARED WEIGHT IDs & KV-BANK RAIL
// ============================================================================

describe('Gate 3 View 1 (macro) — Folded/Unrolled Shared-Weight Topology, KV Rail & Atlas Audit', () => {
  it('validates 0 NaN/Inf, 0 atlas overlaps, <= 512 KiB RG32F, shared-weight IDs, and KV rails across k in {1,2,3,4,6,8} & per_loop/last_loop', () => {
    const matrixCases: readonly {
      readonly kLoops: 1 | 2 | 3 | 4 | 6 | 8;
      readonly kvCacheMode: DoryKvCacheMode;
    }[] = [
      { kLoops: 1, kvCacheMode: 'per_loop' },
      { kLoops: 2, kvCacheMode: 'per_loop' },
      { kLoops: 3, kvCacheMode: 'per_loop' },
      { kLoops: 4, kvCacheMode: 'per_loop' },
      { kLoops: 6, kvCacheMode: 'per_loop' },
      { kLoops: 8, kvCacheMode: 'per_loop' },
      { kLoops: 2, kvCacheMode: 'last_loop' },
      { kLoops: 4, kvCacheMode: 'last_loop' },
      { kLoops: 8, kvCacheMode: 'last_loop' },
    ];

    for (const tc of matrixCases) {
      const session = createDorySession({
        params: PARAMS,
        ropeTables: ROPE_TABLES,
        initialBranch: {
          promptTokens: DEFAULT_PROMPT,
          kLoops: tc.kLoops,
          kvCacheMode: tc.kvCacheMode,
          executionMode: 'standard',
          sampling: { temperature: 0, topK: 0, topP: 1.0 },
        },
      });
      const stepRec = session.step();
      const memBefore = session.measureMemory();

      const folded = buildDoryMacroScene(session, {
        displayMode: 'folded',
        selectedPhysicalLayerIdx: 16,
        selectedLoopIteration: 1,
      });
      const unrolled = buildDoryMacroScene(session, {
        displayMode: 'unrolled',
        selectedPhysicalLayerIdx: 16,
        selectedLoopIteration: tc.kLoops,
      });

      const caseTag = `macro[k=${tc.kLoops},${tc.kvCacheMode}]`;
      assertGate3SceneHealth(folded, `${caseTag}.folded`, 65_536);
      assertGate3SceneHealth(unrolled, `${caseTag}.unrolled`, 65_536);

      // Zero session mutation from scene building
      expect(session.measureMemory()).toEqual(memBefore);

      // Exact physical-72 vs executed (32 + 40k) depth accounting
      const expectedExecDepth = 32 + 40 * tc.kLoops;
      const expectedPairedDepth = 16 + 20 * tc.kLoops;
      for (const sc of [folded, unrolled]) {
        const meta = sc.macroMetadata!;
        expect(meta.physicalSublayerCount).toBe(72);
        expect(meta.pairedLayerCount).toBe(36);
        expect(meta.executedSublayerCount).toBe(expectedExecDepth);
        expect(meta.effectivePairedDepth).toBe(expectedPairedDepth);
        expect(meta.modelWeightId).toBe(PARAMS.modelWeightId);
        expect(meta.kLoops).toBe(tc.kLoops);
        expect(meta.kvCacheMode).toBe(tc.kvCacheMode);
      }

      // Folded: exactly 72 physical sublayer boxes (16 Input + 40 Recurrent + 16 Output)
      const foldedSublayerBoxes = folded.pickBoxes.filter((pb) =>
        /^phys_l\d+$/.test(pb.partId ?? ''),
      );
      expect(foldedSublayerBoxes).toHaveLength(72);
      const foldedPhysIds = new Set(
        foldedSublayerBoxes.map((pb) => pickHitToDoryCell({ ...pb, t: 1, point: [0, 0, 0], isFascia: false }, 'macro')!.physicalLayerIdx),
      );
      expect(foldedPhysIds.size).toBe(72);

      // Verify exact 3:1 SWA:Full motif across all 3 stages inside folded boxes
      const stageKindCounts = {
        input: { sliding_attention: 0, full_attention: 0, mlp: 0 },
        recurrent: { sliding_attention: 0, full_attention: 0, mlp: 0 },
        output: { sliding_attention: 0, full_attention: 0, mlp: 0 },
      };
      for (const pb of foldedSublayerBoxes) {
        const dec = pickHitToDoryCell({ ...pb, t: 1, point: [0, 0, 0], isFascia: false }, 'macro')!;
        expect(dec.modelWeightId).toBe(PARAMS.modelWeightId);
        if (dec.stage === 'input' || dec.stage === 'recurrent' || dec.stage === 'output') {
          stageKindCounts[dec.stage][dec.sublayerKind!] += 1;
        }
      }
      expect(stageKindCounts.input).toEqual({ sliding_attention: 6, full_attention: 2, mlp: 8 });
      expect(stageKindCounts.recurrent).toEqual({ sliding_attention: 15, full_attention: 5, mlp: 20 });
      expect(stageKindCounts.output).toEqual({ sliding_attention: 6, full_attention: 2, mlp: 8 });

      // Unrolled: exactly 32 + 40k executed sublayer boxes covering 0 .. 31 + 40k with ZERO gaps
      const unrolledSublayerBoxes = unrolled.pickBoxes.filter((pb) =>
        /^exec_e\d+_p\d+_t\d+$/.test(pb.partId ?? ''),
      );
      expect(unrolledSublayerBoxes).toHaveLength(expectedExecDepth);

      const execIndicesSeen = new Set<number>();
      const recurrentByPhys = new Map<number, { loops: number[]; weightIds: Set<string>; bankIds: number[] }>();
      for (const pb of unrolledSublayerBoxes) {
        const dec = pickHitToDoryCell({ ...pb, t: 1, point: [0, 0, 0], isFascia: false }, 'macro')!;
        expect(dec.executedLayerIdx).not.toBeNull();
        execIndicesSeen.add(dec.executedLayerIdx!);
        expect(dec.modelWeightId).toBe(PARAMS.modelWeightId);
        if (dec.stage === 'recurrent') {
          const entry = recurrentByPhys.get(dec.physicalLayerIdx!) ?? {
            loops: [],
            weightIds: new Set<string>(),
            bankIds: [],
          };
          entry.loops.push(dec.loopIteration!);
          entry.weightIds.add(dec.modelWeightId!);
          if (dec.kvBankSlotId !== null) entry.bankIds.push(dec.kvBankSlotId);
          recurrentByPhys.set(dec.physicalLayerIdx!, entry);
          expect(pb.semanticTag).toContain('sharedWeight=true');
        }
      }
      expect(execIndicesSeen.size).toBe(expectedExecDepth);
      for (let e = 0; e < expectedExecDepth; e += 1) {
        expect(execIndicesSeen.has(e)).toBe(true);
      }

      // Every recurrent physical layer p in 16..55 binds IDENTICAL singleton modelWeightId across all k passes
      expect(recurrentByPhys.size).toBe(40);
      for (let p = 16; p < 56; p += 1) {
        const info = recurrentByPhys.get(p)!;
        expect(info.loops).toEqual(Array.from({ length: tc.kLoops }, (_, idx) => idx + 1));
        expect(Array.from(info.weightIds)).toEqual([PARAMS.modelWeightId]);
        if (p % 2 === 0) {
          if (tc.kvCacheMode === 'per_loop') {
            expect(new Set(info.bankIds).size).toBe(tc.kLoops);
          } else {
            // last_loop shares ONE overwrite KV bank across all k passes
            expect(new Set(info.bankIds).size).toBe(1);
          }
        }
      }

      // Shared-weight binding ledger matches exact recurrent ranges
      const bindings = unrolled.macroMetadata!.sharedRecurrentWeightBindings;
      expect(bindings).toHaveLength(tc.kLoops);
      for (let t = 1; t <= tc.kLoops; t += 1) {
        const b = bindings[t - 1]!;
        expect(b.loopIteration).toBe(t);
        expect(b.physicalLayerStart).toBe(16);
        expect(b.physicalLayerEnd).toBe(55);
        expect(b.executedLayerStart).toBe(16 + (t - 1) * 40);
        expect(b.executedLayerEnd).toBe(55 + (t - 1) * 40);
        expect(b.modelWeightId).toBe(PARAMS.modelWeightId);
        expect(b.isSharedOverwriteBank).toBe(tc.kvCacheMode === 'last_loop');
        expect(b.kvBankRangeStart).toBe(tc.kvCacheMode === 'last_loop' ? 8 : 8 + (t - 1) * 20);
        expect(b.kvBankRangeEnd).toBe(b.kvBankRangeStart + 19);
      }

      // Parallel KV-bank rail counts & pickBoxes
      const expectedBanks = tc.kvCacheMode === 'per_loop' ? 16 + 20 * tc.kLoops : 36;
      const expectedSwa = tc.kvCacheMode === 'per_loop' ? 12 + 15 * tc.kLoops : 27;
      const expectedFull = tc.kvCacheMode === 'per_loop' ? 4 + 5 * tc.kLoops : 9;
      expect(unrolled.macroMetadata!.kvRail.totalBanks).toBe(expectedBanks);
      expect(unrolled.macroMetadata!.kvRail.totalSwaBanks).toBe(expectedSwa);
      expect(unrolled.macroMetadata!.kvRail.totalFullBanks).toBe(expectedFull);
      expect(folded.macroMetadata!.kvRail).toEqual(unrolled.macroMetadata!.kvRail);

      const kvBoxes = unrolled.pickBoxes.filter((pb) => /^kv_bank_\d+$/.test(pb.partId ?? ''));
      expect(kvBoxes).toHaveLength(expectedBanks);
      if (tc.kvCacheMode === 'last_loop') {
        const recKvBoxes = kvBoxes.filter((pb) => (pb.semanticTag ?? '').includes('bankKind=recurrent'));
        expect(recKvBoxes).toHaveLength(20);
        for (const rb of recKvBoxes) {
          expect(rb.semanticTag).toContain('overwrite=true');
          expect(rb.semanticTag).toContain('policy=last_loop');
        }
      }

      // Live RG32F atlas holds exact per-loop states; for k >= 2, s_1 != s_k (distinct activations, shared weights)
      const atlasS1 = readAtlasRegionR(unrolled, 's_loop_1_8ch');
      assertBitExactFiniteArray(atlasS1, stepRec.lastTokenLoopExitStates[0]!, `${caseTag}.s1`);
      if (tc.kLoops >= 2) {
        const atlasSk = readAtlasRegionR(unrolled, `s_loop_${tc.kLoops}_8ch`);
        assertBitExactFiniteArray(
          atlasSk,
          stepRec.lastTokenLoopExitStates[tc.kLoops - 1]!,
          `${caseTag}.sk`,
        );
        expect(computeStrictMaxAbsDiff(atlasS1, atlasSk, `${caseTag}.s1_vs_sk`)).toBeGreaterThan(1e-4);
      }
    }
  });

  it('renders Contract A 104-bank self_spec KV rail (+8 output_shallow banks), deterministic atlas keys, and monotonic Y layout', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: DEFAULT_PROMPT,
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: 4,
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const cycleRec = session.step();

    const foldedA = buildDoryMacroScene(session, { displayMode: 'folded', recordId: cycleRec.recordId });
    const foldedB = buildDoryMacroScene(session, { displayMode: 'folded', recordId: cycleRec.recordId });
    const unrolled = buildDoryMacroScene(session, { displayMode: 'unrolled', recordId: cycleRec.recordId });

    assertGate3SceneHealth(foldedA, 'macro.self_spec.folded', 65_536);
    assertGate3SceneHealth(unrolled, 'macro.self_spec.unrolled', 65_536);

    // Deterministic content-addressed atlas key & exact geometry repeatability
    expect(foldedA.atlasKey).toBe(foldedB.atlasKey);
    expect(foldedA.solidCount).toBe(foldedB.solidCount);
    expect(foldedA.sheetCount).toBe(foldedB.sheetCount);
    expect(computeDoryAtlasContentKey('macro', foldedA.valueAtlas!, `folded:k4:per_loop:self_spec:${cycleRec.recordId}`)).toBe(
      foldedA.atlasKey,
    );

    // Contract A 104 banks = 8 Input + 80 Recurrent + 8 Deep Coda + 8 Shallow Coda (78 SWA + 26 Full)
    expect(unrolled.macroMetadata!.kvRail).toEqual(
      buildDoryMacroKvRailSummary(4, 'per_loop', true),
    );
    expect(unrolled.macroMetadata!.kvRail.totalBanks).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
    expect(unrolled.macroMetadata!.kvRail.outputDeepBanks).toBe(8);
    expect(unrolled.macroMetadata!.kvRail.outputShallowBanks).toBe(8);
    expect(unrolled.macroMetadata!.kvRail.totalSwaBanks).toBe(78);
    expect(unrolled.macroMetadata!.kvRail.totalFullBanks).toBe(26);

    // Explicit Contract A Reconstructed from Paper §6.2 provenance badge & output_shallow pickBoxes
    expect(unrolled.provenanceBadges.some((b) => b.label === DORY_SPEC_CONTRACT_A_BADGE)).toBe(true);
    const shallowBoxes = unrolled.pickBoxes.filter((pb) =>
      (pb.semanticTag ?? '').includes('bankKind=output_shallow'),
    );
    expect(shallowBoxes).toHaveLength(8);
    for (let i = 0; i < shallowBoxes.length; i += 1) {
      const dec = pickHitToDoryCell(
        { ...shallowBoxes[i]!, t: 1, point: [0, 0, 0], isFascia: false },
        'macro',
      )!;
      expect(dec.kvBankKind).toBe('output_shallow');
      expect(dec.kvBankSlotId).toBe(96 + i);
      expect(dec.cacheRole).toBe('spec_draft');
    }

    // Pure helper checks: monotonic descending pair Y coordinates & clear stage gaps
    for (let pair = 0; pair < 35; pair += 1) {
      expect(doryMacroPairRowY(pair)).toBeGreaterThan(doryMacroPairRowY(pair + 1));
    }
    expect(DORY_MACRO_LAYOUT.preludeBotY - DORY_MACRO_LAYOUT.coreTopY).toBeGreaterThan(
      DORY_MACRO_LAYOUT.stageGapY,
    );
    expect(DORY_MACRO_LAYOUT.coreBotY - DORY_MACRO_LAYOUT.codaTopY).toBeGreaterThan(
      DORY_MACRO_LAYOUT.stageGapY,
    );
    expect(doryMacroPhysicalLayerY(16)).toBeGreaterThan(doryMacroPhysicalLayerY(17));
    expect(buildDorySharedWeightBindings(4, 'per_loop', PARAMS.modelWeightId)).toHaveLength(4);
  });
});

// ============================================================================
// 2. VIEW 2 (`hypersphere`) — ANTI-SLERP, 2D PCA (8D METRICS), 120° CAVEAT & COUNTERFACTUAL
// ============================================================================

describe('Gate 3 View 2 (hypersphere) — Anti-SLERP, Exploratory 2D PCA (8D Metrics), 120-deg Caveat & Read-Only Counterfactual', () => {
  it('enforces Anti-SLERP (DR7), planted |alpha_hat| > 1 extrapolation, Exact 2D span(a, b) & finite <= 512 KiB atlas across layers/depths/modes', () => {
    const k4Session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: DEFAULT_PROMPT,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    k4Session.step();
    k4Session.step();

    // Default p = 16 (first Recurrent Core SWA layer with planted channels [6, 7] in (1.05, 1.25))
    const sceneP16 = buildDoryHypersphereScene(k4Session, {
      physicalLayerIdx: DORY_PLANTED_EXTRAPOLATION_SPEC.physicalLayerIdx,
      loopIteration: 2,
      includeCounterfactualCoda: true,
      counterfactualLoop: 1,
    });
    assertGate3SceneHealth(sceneP16, 'hypersphere.p16.k4', DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES);

    const meta = sceneP16.hypersphereMetadata!;
    expect(meta.antiSlerpPlaque).toBe(DORY_ANTI_SLERP_PLAQUE);
    expect(meta.antiSlerpPlaque).toContain('NLERP');
    expect(meta.antiSlerpPlaque).toContain('NOT Trigonometric SLERP');
    expect(meta.antiSlerpMarker).toBe(DORY_NLERP_MARKER);
    expect(
      sceneP16.provenanceBadges.some(
        (b) => b.label === DORY_ANTI_SLERP_PLAQUE && b.discrepancyId === 'DR7',
      ),
    ).toBe(true);
    const antiSlerpCallout = sceneP16.callouts.find((c) => c.id === 'callout_anti_slerp');
    expect(antiSlerpCallout).toBeDefined();
    expect(antiSlerpCallout!.body).toContain('NEVER runs trigonometric');
    expect(antiSlerpCallout!.body).toContain('NEVER clamps |alpha_hat|');

    // Exact 2D Spanning-Subspace Diagram (a): span(a, b) & unclamped |alpha_hat| > 1
    const span = meta.spanningSubspace;
    expect(span.physicalLayerIdx).toBe(16);
    expect(span.pairedLayerIdx).toBe(8);
    expect(span.loopIteration).toBe(2);
    expect(span.a2d).toEqual([1, 0]);
    expect(Math.hypot(span.b2d[0], span.b2d[1])).toBeCloseTo(1.0, 12);
    expect(span.b2d[0]).toBeCloseTo(Math.cos(span.omegaRadians), 12);
    expect(span.b2d[1]).toBeCloseTo(Math.sin(span.omegaRadians), 12);
    expect(span.omegaDegrees).toBeCloseTo((span.omegaRadians * 180) / Math.PI, 10);

    // Planted unclamped extrapolation channels [6, 7] verified against raw -> effective formula
    const expectedEffAlpha = effectiveResidualAlpha(span.rawAlpha, PARAMS.config);
    expect(computeStrictMaxAbsDiff(span.effectiveAlpha, expectedEffAlpha, 'p16.effAlpha')).toBeLessThanOrEqual(
      1e-6,
    );
    expect(span.extrapolatedChannels).toContain(6);
    expect(span.extrapolatedChannels).toContain(7);
    expect(span.maxEffectiveAlpha).toBeGreaterThan(DORY_PLANTED_EXTRAPOLATION_SPEC.minEffectiveAlpha);
    expect(span.maxEffectiveAlpha).toBeLessThan(DORY_PLANTED_EXTRAPOLATION_SPEC.maxEffectiveAlpha);

    // Raised 3D extrapolation bars exist for planted channels [6, 7]
    expect(sceneP16.pickBoxes.some((pb) => pb.partId === 'nlerp_extrap_ch_6')).toBe(true);
    expect(sceneP16.pickBoxes.some((pb) => pb.partId === 'nlerp_extrap_ch_7')).toBe(true);

    // Uniform-scalar NLERP curve includes both [0, 1] interpolation and > 1 extrapolation points
    const extrapPts = span.uniformAlphaCurve.filter((pt) => pt.isExtrapolation);
    expect(extrapPts.length).toBeGreaterThan(0);
    expect(span.uniformAlphaCurve[0]!.lambda).toBe(0);
    expect(span.uniformAlphaCurve[0]!.x).toBeCloseTo(1, 5);
    expect(span.uniformAlphaCurve[0]!.y).toBeCloseTo(0, 6);
    for (const pt of span.uniformAlphaCurve) {
      const r = Math.hypot(pt.x, pt.y);
      expect(r).toBeGreaterThan(0.999);
      expect(r).toBeLessThanOrEqual(1.0 + 1e-6);
    }

    // Negative mathematical Anti-SLERP proof on the actual reconstructed layer-16 NLERP tensors:
    // actual channel-wise unclamped NLERP differs from BOTH:
    //   (i) classical trigonometric SLERP at scalar t = meanEffectiveAlpha
    //   (ii) [0, 1]-clamped channel-wise NLERP (proving > 1 channels genuinely extrapolate!)
    const ctx = resolveDorySessionContext(k4Session);
    const sub16 = ctx.resolveSublayerRecord(16, 2);
    const off = meta.selectedTokenOffset * DORY_MINI_CONFIG.hiddenSize;
    const aVec = sub16.trace.nlerp.aNorm.subarray(off, off + 8);
    const bVec = sub16.trace.nlerp.bNorm.subarray(off, off + 8);
    const actualOut = sub16.trace.nlerp.output.subarray(off, off + 8);

    const sinOmega = Math.sin(span.omegaRadians);
    const tMean = span.meanEffectiveAlpha;
    const trigSlerp = new Float32Array(8);
    const clampedNlerpPre = new Float32Array(8);
    for (let c = 0; c < 8; c += 1) {
      if (sinOmega > 1e-8) {
        trigSlerp[c] =
          (Math.sin((1 - tMean) * span.omegaRadians) * aVec[c]! +
            Math.sin(tMean * span.omegaRadians) * bVec[c]!) /
          sinOmega;
      } else {
        trigSlerp[c] = aVec[c]!;
      }
      const clampedLam = Math.min(1.0, span.effectiveAlpha[c]!);
      clampedNlerpPre[c] = aVec[c]! + clampedLam * (bVec[c]! - aVec[c]!);
    }
    const clampedNlerpOut = l2NormVector(clampedNlerpPre, PARAMS.config.l2NormEps);

    expect(computeStrictMaxAbsDiff(actualOut, trigSlerp, 'nlerp_vs_trig_slerp')).toBeGreaterThan(1e-2);
    expect(computeStrictMaxAbsDiff(actualOut, clampedNlerpOut, 'nlerp_vs_clamped')).toBeGreaterThan(1e-3);

    // Out-of-plane norm matches Pythagorean identity: ||Pi(out)||^2 + outOfPlaneNorm^2 === ||out||^2
    const out8dNorm = vectorL2Norm(actualOut, 0, 8);
    const projOutNorm = Math.hypot(span.projectedOutput2d[0], span.projectedOutput2d[1]);
    expect(Math.hypot(projOutNorm, span.outOfPlaneNorm)).toBeCloseTo(out8dNorm, 5);
    expect(span.outOfPlaneNorm).toBeGreaterThan(1e-3);

    // Cross-stage & cross-mode coverage: Prelude p=0, Recurrent MLP p=17, Coda p=56, k=1, k=8, last_loop, self_spec
    for (const p of [0, 17, 56]) {
      const sc = buildDoryHypersphereScene(k4Session, {
        physicalLayerIdx: p,
        loopIteration: p === 17 ? 3 : 0,
        includeCounterfactualCoda: false,
      });
      assertGate3SceneHealth(sc, `hypersphere.p${p}`, DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES);
      expect(sc.hypersphereMetadata!.selectedPhysicalLayerIdx).toBe(p);
      expect(sc.hypersphereMetadata!.counterfactualCoda).toBeNull();
    }

    // k = 1 (rank-1 PCA fallback), k = 8 extrapolation, last_loop, and Contract A self_spec
    const extraSessions = [
      { label: 'k1_per_loop', cfg: { kLoops: 1 as const, kvCacheMode: 'per_loop' as const, executionMode: 'standard' as const } },
      { label: 'k8_per_loop', cfg: { kLoops: 8 as const, kvCacheMode: 'per_loop' as const, executionMode: 'standard' as const } },
      { label: 'k4_last_loop', cfg: { kLoops: 4 as const, kvCacheMode: 'last_loop' as const, executionMode: 'standard' as const } },
      { label: 'k4_self_spec', cfg: { kLoops: 4 as const, kvCacheMode: 'per_loop' as const, executionMode: 'self_spec' as const, speculativeGamma: 4 } },
    ];
    for (const es of extraSessions) {
      const s = createDorySession({
        params: PARAMS,
        ropeTables: ROPE_TABLES,
        initialBranch: {
          promptTokens: DEFAULT_PROMPT,
          ...es.cfg,
          sampling: { temperature: 0, topK: 0, topP: 1.0 },
        },
      });
      s.step();
      const sc = buildDoryHypersphereScene(s, {
        physicalLayerIdx: 16,
        loopIteration: 1,
        includeCounterfactualCoda: true,
        counterfactualLoop: 1,
      });
      assertGate3SceneHealth(sc, `hypersphere.${es.label}`, DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES);
      expect(sc.hypersphereMetadata!.exploratoryPca.allProjectedNormsLeqOne).toBe(true);
      expect(sc.hypersphereMetadata!.exploratoryPca.points).toHaveLength(1 + 2 * es.cfg.kLoops);
    }
  });

  it('verifies Exploratory 2D PCA disclosure badge, orthonormal 8D axes, projectedNorm <= exact8dNorm <= 1, and strict 8D metric provenance', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: DEFAULT_PROMPT,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const rec = session.step();
    const scene = buildDoryHypersphereScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 4,
      includeCounterfactualCoda: false,
    });
    const pca = scene.hypersphereMetadata!.exploratoryPca;

    // Authoritative Exploratory 2D PCA disclosure badge
    expect(pca.disclosureBadge).toBe(DORY_EXPLORATORY_PCA_BADGE);
    expect(pca.disclosureBadge).toContain('EXPLORATORY 2D PCA');
    expect(pca.disclosureBadge).toContain('PROJECTED NORMS <= 1');
    expect(pca.disclosureBadge).toContain('8D');
    expect(pca.metricsComputedIn8d).toBe(true);
    expect(pca.allProjectedNormsLeqOne).toBe(true);
    expect(scene.provenanceBadges.some((b) => b.label === DORY_EXPLORATORY_PCA_BADGE)).toBe(true);

    // Orthonormal 8D PCA basis vectors & valid explained variance ratios
    const pc1Norm = vectorL2Norm(pca.pc1Axis8d, 0, 8);
    const pc2Norm = vectorL2Norm(pca.pc2Axis8d, 0, 8);
    let pc12Dot = 0;
    for (let c = 0; c < 8; c += 1) pc12Dot += pca.pc1Axis8d[c]! * pca.pc2Axis8d[c]!;
    expect(pc1Norm).toBeCloseTo(1.0, 5);
    expect(pc2Norm).toBeCloseTo(1.0, 5);
    expect(Math.abs(pc12Dot)).toBeLessThan(1e-5);

    expect(pca.explainedVarianceRatio[0]).toBeGreaterThanOrEqual(pca.explainedVarianceRatio[1] - 1e-12);
    expect(pca.explainedVarianceRatio[0]).toBeGreaterThanOrEqual(0);
    expect(pca.explainedVarianceRatio[1]).toBeGreaterThanOrEqual(0);
    expect(pca.explainedVarianceRatio[0] + pca.explainedVarianceRatio[1]).toBeLessThanOrEqual(1 + 1e-12);

    // Reconstruct raw 8D states and independently recompute every numeric metric in 8D
    const xAnchor = rec.lastTokenXAnchor;
    const rawVectors: { id: string; vec: Float32Array; prev: Float32Array | null }[] = [
      { id: 'x_anchor', vec: xAnchor, prev: null },
    ];
    for (let t = 1; t <= 4; t += 1) {
      const prevExit = t === 1 ? xAnchor : rec.lastTokenLoopExitStates[t - 2]!;
      rawVectors.push(
        { id: `u_${t}`, vec: rec.lastTokenLoopEntryStates[t - 1]!, prev: prevExit },
        { id: `s_${t}`, vec: rec.lastTokenLoopExitStates[t - 1]!, prev: prevExit },
      );
    }
    expect(pca.points).toHaveLength(rawVectors.length); // 1 + 2*4 = 9 points

    let observedStrictForeshortening = false;
    for (let i = 0; i < rawVectors.length; i += 1) {
      const pt = pca.points[i]!;
      const raw = rawVectors[i]!;
      expect(pt.id).toBe(raw.id);

      // Projected 2D norm strictly <= exact 8D norm <= 1 (orthogonal projection onto 2D plane)
      expect(pt.projectedNorm).toBeCloseTo(Math.hypot(pt.x, pt.y), 12);
      expect(pt.projectedNorm).toBeLessThanOrEqual(pt.exact8dNorm + 1e-5);
      expect(pt.projectedNorm).toBeLessThanOrEqual(1.0 + 1e-6);
      if (pt.exact8dNorm - pt.projectedNorm > 1e-3) {
        observedStrictForeshortening = true;
      }

      // Every reported numeric metric MUST equal direct 8D computation (never 2D PCA approximation)
      const refNorm = vectorL2Norm(raw.vec, 0, 8);
      const refAnchorCos = cosineSimilarity8d(raw.vec, xAnchor, 0, 0, 8);
      const refAnchorDeg = (Math.acos(Math.max(-1, Math.min(1, refAnchorCos))) * 180) / Math.PI;
      expect(pt.exact8dNorm).toBeCloseTo(refNorm, 6);
      expect(pt.exact8dNorm).toBeCloseTo(1.0, 4);
      expect(pt.exact8dAnchorCosine).toBeCloseTo(refAnchorCos, 6);
      expect(pt.exact8dAnchorAngleDeg).toBeCloseTo(refAnchorDeg, 5);

      if (raw.prev === null) {
        expect(pt.exact8dConsecutiveCosine).toBeNull();
        expect(pt.exact8dConsecutiveDisplacement).toBeNull();
      } else {
        expect(pt.exact8dConsecutiveCosine).toBeCloseTo(
          cosineSimilarity8d(raw.vec, raw.prev, 0, 0, 8),
          6,
        );
        expect(pt.exact8dConsecutiveDisplacement).toBeCloseTo(
          l2Distance(raw.vec, raw.prev, 0, 0, 8),
          6,
        );
      }
    }
    // Genuine 8D->2D norm collapse occurs on a 9-point trajectory in S^7
    expect(observedStrictForeshortening).toBe(true);

    // Pure rank-1 degenerate trajectory check (k = 1, identical states): finite, orthonormal, <= 1
    const degAnchor = l2NormVector(new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]));
    const degPca = computeExploratoryPcaTrajectory({
      xAnchor: degAnchor,
      loopEntryStates: [degAnchor],
      loopExitStates: [degAnchor],
    });
    expect(degPca.allProjectedNormsLeqOne).toBe(true);
    expect(vectorL2Norm(degPca.pc1Axis8d, 0, 8)).toBeCloseTo(1.0, 5);
    expect(vectorL2Norm(degPca.pc2Axis8d, 0, 8)).toBeCloseTo(1.0, 5);
    for (const pt of degPca.points) {
      expect(Number.isFinite(pt.x)).toBe(true);
      expect(Number.isFinite(pt.y)).toBe(true);
      expect(pt.projectedNorm).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });

  it('enforces the local 120-deg anchor-map Jacobian caveat (Audit §6.2.1, Finding 9)', () => {
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: DEFAULT_PROMPT,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });
    session.step();

    const scene = buildDoryHypersphereScene(session, {
      physicalLayerIdx: 16,
      loopIteration: 2,
    });
    const meta = scene.hypersphereMetadata!;

    // Explicit local-only 120-deg caveat badge and callout (never claiming global convergence!)
    expect(meta.anchorMapCaveatBadge).toBe(DORY_ANCHOR_MAP_CAVEAT_BADGE);
    expect(meta.anchorMapCaveatBadge).toContain('120 deg');
    expect(meta.anchorMapCaveatBadge).toContain('does NOT imply global core contraction or convergence');
    expect(scene.provenanceBadges.some((b) => b.label === DORY_ANCHOR_MAP_CAVEAT_BADGE)).toBe(true);

    const jacCallout = scene.callouts.find((c) => c.id === 'callout_anchor_jacobian_caveat');
    expect(jacCallout).toBeDefined();
    expect(jacCallout!.body).toContain(DORY_ANCHOR_MAP_CAVEAT_BADGE);
    expect(jacCallout!.body).toContain('NO unit-Lipschitz guarantee');

    // Exact ideal vs eps-stabilized boundary-map Jacobian math across critical angles
    const deg = (d: number) => (d * Math.PI) / 180;
    const j0 = anchorMapJacobianNorm(deg(0), 1e-6);
    const j60 = anchorMapJacobianNorm(deg(60), 1e-6);
    const j119 = anchorMapJacobianNorm(deg(119.9), 1e-6);
    const j120 = anchorMapJacobianNorm(deg(120), 1e-6);
    const j179 = anchorMapJacobianNorm(deg(179), 1e-6);
    const j180 = anchorMapJacobianNorm(deg(180), 1e-6);

    expect(j0.epsStabilizedJacobianNorm).toBeCloseTo(0.5, 4);
    expect(j0.isLocalContractiveRegime).toBe(true);
    expect(j60.epsStabilizedJacobianNorm).toBeCloseTo(1 / Math.sqrt(3 + 1e-6), 4);
    expect(j60.isLocalContractiveRegime).toBe(true);
    expect(j119.epsStabilizedJacobianNorm).toBeLessThan(1.0);
    expect(j119.isLocalContractiveRegime).toBe(true);
    expect(j120.idealJacobianNorm).toBeCloseTo(1.0, 10);
    expect(j120.isLocalContractiveRegime).toBe(false); // strictly < 120 deg
    expect(j179.epsStabilizedJacobianNorm).toBeCloseTo(57.2027, 2);
    expect(j179.isLocalContractiveRegime).toBe(false);
    expect(j180.idealJacobianNorm).toBe(Infinity);
    expect(j180.epsStabilizedJacobianNorm).toBeCloseTo(1000.0, 6); // capped at 1 / sqrt(1e-6)
    expect(j180.isLocalContractiveRegime).toBe(false);

    // Loop 1 entry u_1 = N(s_0 + x) = N(2x) has theta = 0 deg -> Jacobian ≈ 0.5000
    expect(meta.loopMetrics8d).toHaveLength(4);
    expect(meta.loopMetrics8d[0]!.entryAnchorJacobian.angleDegrees).toBeCloseTo(0, 3);
    expect(meta.loopMetrics8d[0]!.entryAnchorJacobian.epsStabilizedJacobianNorm).toBeCloseTo(0.5, 4);
    expect(meta.loopMetrics8d[0]!.entryAnchorJacobian.isLocalContractiveRegime).toBe(true);
  });

  it('verifies read-only Diagnostic Counterfactual Coda causes ZERO KV, session, LRU, or next-step mutation (per_loop & last_loop)', () => {
    // Construct twin sessions with identical prompt/seed/config so we prove BOTH:
    //  (a) Bit-for-bit (0-ULP) invariance of every KV slab & metadata before vs. after View 2
    //  (b) Bit-for-bit (0-ULP) identical future decode step compared to an untouched twin session!
    const configs: readonly {
      readonly label: string;
      readonly kvCacheMode: DoryKvCacheMode;
      readonly prefillChunkSizes?: readonly number[];
    }[] = [
      { label: 'per_loop_k4', kvCacheMode: 'per_loop' },
      { label: 'last_loop_k4_chunked', kvCacheMode: 'last_loop', prefillChunkSizes: [4, 4] },
    ];

    for (const cfg of configs) {
      const initOpts = {
        params: PARAMS,
        ropeTables: ROPE_TABLES,
        initialBranch: {
          promptTokens: DEFAULT_PROMPT,
          ...(cfg.prefillChunkSizes ? { prefillChunkSizes: cfg.prefillChunkSizes } : {}),
          kLoops: 4 as const,
          kvCacheMode: cfg.kvCacheMode,
          executionMode: 'standard' as const,
          sampling: { temperature: 0.9, topK: 0, topP: 1.0 },
          seed: 0xd0419901,
        },
      };

      const probedSession = createDorySession(initOpts);
      const controlSession = createDorySession(initOpts);

      // Advance both sessions 3 steps so historical Deep Coda KV holds rich committed context
      let headRec = probedSession.getActiveRecord()!;
      for (let s = 0; s < 3; s += 1) {
        headRec = probedSession.step();
        controlSession.step();
      }

      const snapBefore = probedSession.getSnapshot();
      const memBefore = probedSession.measureMemory();
      const kvBefore = captureDeepKvSnapshot(probedSession.reconstructKvCacheAtRecord(headRec.recordId));

      // Evaluate View 2 Diagnostic Counterfactual Coda across ALL loops t = 1..4
      let cfLoop1Scene: DorySpatialScene | null = null;
      let cfLoop4Scene: DorySpatialScene | null = null;
      for (let loopT = 1; loopT <= 4; loopT += 1) {
        const scene = buildDoryHypersphereScene(probedSession, {
          physicalLayerIdx: 16,
          loopIteration: loopT,
          includeCounterfactualCoda: true,
          counterfactualLoop: loopT,
        });
        assertGate3SceneHealth(
          scene,
          `hypersphere.cf.${cfg.label}.t${loopT}`,
          DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES,
        );

        const cf = scene.hypersphereMetadata!.counterfactualCoda!;
        expect(cf).not.toBeNull();
        expect(cf.badge).toBe(DORY_COUNTERFACTUAL_CODA_BADGE);
        expect(cf.readOnlyZeroKvMutation).toBe(true);
        expect(cf.loopIteration).toBe(loopT);
        expect(cf.kLoops).toBe(4);
        expect(cf.absTokenPos).toBe(headRec.committedLen - 1);
        expect(cf.topTokens).toHaveLength(8);
        expect(cf.klFromFinalLoopNats).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(cf.klFromFinalLoopNats)).toBe(true);

        // Top tokens sorted descending by probability, probabilities sum to 1
        for (let i = 1; i < cf.topTokens.length; i += 1) {
          expect(cf.topTokens[i - 1]!.prob).toBeGreaterThanOrEqual(cf.topTokens[i]!.prob);
        }
        let probSum = 0;
        for (let v = 0; v < cf.rawResult.probs.length; v += 1) probSum += cf.rawResult.probs[v]!;
        expect(probSum).toBeCloseTo(1.0, 12);

        expect(scene.provenanceBadges.some((b) => b.label === DORY_COUNTERFACTUAL_CODA_BADGE)).toBe(true);
        expect(scene.callouts.some((c) => c.id === 'callout_counterfactual_coda')).toBe(true);

        if (loopT === 1) cfLoop1Scene = scene;
        if (loopT === 4) cfLoop4Scene = scene;
      }

      // At final loop t = k = 4, counterfactual Coda on s_4 against committed historical Deep Coda KV
      // MUST match the live final-loop logits within D5 parity (and KL ≈ 0)!
      const cf4 = cfLoop4Scene!.hypersphereMetadata!.counterfactualCoda!;
      expect(
        computeStrictMaxAbsDiff(
          cf4.rawResult.scaledLogits,
          headRec.lastTokenScaledLogits,
          `${cfg.label}.cf_t4_vs_final`,
        ),
      ).toBeLessThanOrEqual(2e-5);
      expect(cf4.klFromFinalLoopNats).toBeLessThan(1e-10);
      expect(cf4.argMaxChanged).toBe(false);

      // Intermediate loop t = 1 genuinely diverges from final loop k = 4
      const cf1 = cfLoop1Scene!.hypersphereMetadata!.counterfactualCoda!;
      expect(cf1.klFromFinalLoopNats).toBeGreaterThan(1e-4);
      expect(computeKlDivergenceFloat64(cf4.rawResult.probs, cf4.rawResult.probs)).toBe(0);

      // (a) Verify ZERO session / memory / KV slab mutation after all 4 counterfactual passes
      const snapAfter = probedSession.getSnapshot();
      const memAfter = probedSession.measureMemory();
      expect(memAfter).toEqual(memBefore);
      expect(snapAfter.recordCount).toBe(snapBefore.recordCount);
      expect(snapAfter.activeBranchId).toBe(snapBefore.activeBranchId);
      expect(snapAfter.activeRecordId).toBe(snapBefore.activeRecordId);
      expect(snapAfter.branches.length).toBe(snapBefore.branches.length);

      const kvAfter = probedSession.reconstructKvCacheAtRecord(headRec.recordId);
      assertKvMatchesSnapshotBitExact(kvAfter, kvBefore, `${cfg.label}.kv_zero_mutation`);

      // (b) Step both probedSession & untouched controlSession -> 0-ULP identical next token & logits!
      const probedNext = probedSession.step();
      const controlNext = controlSession.step();
      expect(probedNext.emittedTokens).toEqual(controlNext.emittedTokens);
      expect(probedNext.pendingToken).toBe(controlNext.pendingToken);
      assertBitExactFiniteArray(
        probedNext.lastTokenScaledLogits,
        controlNext.lastTokenScaledLogits,
        `${cfg.label}.next_step_logits`,
      );
      assertBitExactFiniteArray(
        probedNext.lastTokenProbs,
        controlNext.lastTokenProbs,
        `${cfg.label}.next_step_probs`,
      );
    }
  });
});

// ============================================================================
// 3. DR20 HARD-THROW, DR21 REAL RAYCAST ROUND-TRIP, `pick.ts` UNTOUCHED & D4 ISOLATION
// ============================================================================

describe('Gate 3 Provenance Guards — DR20 Hard Throw, DR21 Raycast Picking, Untouched pick.ts & D4 Isolation', () => {
  it('enforces DR20 hard-throw when sheetKind in {1, 2} omits a live RG32F mask, and catches injected NaN/Inf', () => {
    // 1. DorySheetBufferBuilder.push() hard-throws for both sheetKind = 1 and sheetKind = 2
    const builder = new DorySheetBufferBuilder();
    const invalidRects: readonly (readonly [number, number, number, number] | undefined)[] = [
      undefined,
      [0, 0, 0, 0],
      [0, 0, 0, 0.5],
      [0, 0, 0.5, 0],
      [0.8, 0, 0.5, 0.5], // exceeds u = 1.0
      [0, 0, Number.NaN, 0.25],
    ];
    for (const sheetKind of [SHEET_KIND_CAUSAL_SCORES, SHEET_KIND_CAUSAL_SOFTMAX]) {
      for (const texRect of invalidRects) {
        expect(() =>
          builder.push({
            cx: 0,
            cy: 0,
            cz: 0,
            ex: 1,
            ey: 0.06,
            ez: 1,
            gridCols: 4,
            gridRows: 4,
            sheetKind,
            role: sheetKind === SHEET_KIND_CAUSAL_SCORES ? SHEET_ROLE_SCORES : SHEET_ROLE_SOFTMAX,
            ...(texRect !== undefined ? { texRect } : {}),
          }),
        ).toThrow(DoryDr20CausalMaskError);
      }
    }

    // Standard sheet (sheetKind = 0) without live texture does NOT throw
    expect(() =>
      builder.push({
        cx: 0,
        cy: 0,
        cz: 0,
        ex: 1,
        ey: 0.06,
        ez: 1,
        gridCols: 4,
        gridRows: 4,
        sheetKind: SHEET_KIND_STANDARD,
      }),
    ).not.toThrow();

    // 2. validateSceneGeometryFinite() hard-throws DR20 on tampered real View 2 scene
    const session = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: { promptTokens: DEFAULT_PROMPT, kLoops: 2 },
    });
    const baseScene = buildDoryHypersphereScene(session, { includeCounterfactualCoda: false });
    expect(validateSceneGeometryFinite(baseScene)).toEqual({ valid: true, errors: [] });

    // (a) Promote sheet[0] to sheetKind = 1 while dropping valueAtlas -> MUST throw DR20
    const causalKindData = new Float32Array(baseScene.sheetData);
    causalKindData[8] = SHEET_KIND_CAUSAL_SCORES;
    expect(() =>
      validateSceneGeometryFinite({ ...baseScene, sheetData: causalKindData, valueAtlas: null }),
    ).toThrow(DoryDr20CausalMaskError);

    // (b) Promote sheet[0] to sheetKind = 2 while zeroing uWidth -> MUST throw DR20
    const zeroRectData = new Float32Array(baseScene.sheetData);
    zeroRectData[8] = SHEET_KIND_CAUSAL_SOFTMAX;
    zeroRectData[15] = 0; // uWidth = 0
    expect(() =>
      validateSceneGeometryFinite({ ...baseScene, sheetData: zeroRectData }),
    ).toThrow(DoryDr20CausalMaskError);
    const nonThrowing = validateSceneGeometryFinite(
      { ...baseScene, sheetData: zeroRectData },
      { throwOnDr20: false },
    );
    expect(nonThrowing.valid).toBe(false);
    expect(nonThrowing.errors.some((e) => e.includes('DR20'))).toBe(true);

    // (c) Corrupt G-channel mask texel to non-binary 0.5 on a sheetKind = 1 plate -> MUST throw DR20
    const badMaskAtlasData = new Float32Array(baseScene.valueAtlas!.data);
    const u0 = baseScene.sheetData[13]!;
    const v0 = baseScene.sheetData[14]!;
    const tx = Math.floor(u0 * baseScene.valueAtlas!.width + 1e-4);
    const ty = Math.floor(v0 * baseScene.valueAtlas!.height + 1e-4);
    badMaskAtlasData[(ty * baseScene.valueAtlas!.width + tx) * 2 + 1] = 0.5;
    expect(() =>
      validateSceneGeometryFinite({
        ...baseScene,
        sheetData: causalKindData,
        valueAtlas: { ...baseScene.valueAtlas!, data: badMaskAtlasData },
      }),
    ).toThrow(DoryDr20CausalMaskError);

    // (d) Valid causal plate built via DoryAtlasPacker.packCausalMatrix passes cleanly
    const packer = new DoryAtlasPacker(32, 32);
    const causalSlice = packer.packCausalMatrix(
      'swa_scores',
      new Float32Array([0.4, -Infinity, 0.2, 0.9]),
      new Uint8Array([1, 0, 1, 1]),
      2,
      2,
      0,
    );
    const builtCausalAtlas = packer.build('gqa_rope', 'dr20_valid');
    const validCausalBuilder = new DorySheetBufferBuilder();
    validCausalBuilder.push({
      cx: 0,
      cy: 0,
      cz: 0,
      ex: 1,
      ey: 0.06,
      ez: 1,
      gridCols: 2,
      gridRows: 2,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      slice: causalSlice,
    });
    const packedCausalSheets = validCausalBuilder.pack();
    const validCausalRes = validateSceneGeometryFinite({
      ...baseScene,
      sheetData: packedCausalSheets.sheetData,
      sheetBasisData: packedCausalSheets.sheetBasisData,
      sheetRevealData: packedCausalSheets.sheetRevealData,
      sheetCount: packedCausalSheets.sheetCount,
      valueAtlas: builtCausalAtlas.valueAtlas,
      atlasRegions: builtCausalAtlas.atlasRegions,
      atlasAudit: builtCausalAtlas.atlasAudit,
    });
    expect(validCausalRes).toEqual({ valid: true, errors: [] });

    // (e) Injected NaN / ±Inf across geometry buffers & atlas are caught cleanly
    const nanFlow = new Float32Array(baseScene.flowVertices);
    nanFlow[0] = Number.POSITIVE_INFINITY;
    expect(validateSceneGeometryFinite({ ...baseScene, flowVertices: nanFlow }).valid).toBe(false);

    const nanAtlasData = new Float32Array(baseScene.valueAtlas!.data);
    nanAtlasData[0] = Number.NaN;
    expect(
      validateSceneGeometryFinite({
        ...baseScene,
        valueAtlas: { ...baseScene.valueAtlas!, data: nanAtlasData },
      }).valid,
    ).toBe(false);

    // (f) Negative controls on auditAtlasRegions: overlap, out-of-bounds, and > 512 KiB budget
    const overlapAudit = auditAtlasRegions(64, 64, [
      { id: 'reg_a', x0: 0, y0: 0, cols: 16, rows: 16, mode: 0, isMaskOrScore: false },
      { id: 'reg_b', x0: 8, y0: 8, cols: 16, rows: 16, mode: 0, isMaskOrScore: false },
    ]);
    expect(overlapAudit.hasOverlap).toBe(true);
    expect(overlapAudit.overlapPair).toEqual(['reg_a', 'reg_b']);

    const oobAudit = auditAtlasRegions(32, 32, [
      { id: 'reg_oob', x0: 24, y0: 0, cols: 16, rows: 8, mode: 0, isMaskOrScore: false },
    ]);
    expect(oobAudit.outOfBoundsIds).toEqual(['reg_oob']);

    const overBudgetAudit = auditAtlasRegions(512, 256, [], DORY_MAX_ATLAS_BYTES);
    expect(overBudgetAudit.byteLength).toBe(1_048_576);
    expect(overBudgetAudit.withinBudget).toBe(false);
  });

  it('round-trips real pickMacroBay raycasts through pickHitToDoryCell on Views 1 & 2 (with src/render/pick.ts untouched)', () => {
    // 1. View 1 (macro) Unrolled Contract A self_spec (k=4, 192 exec layers, 104 KV banks)
    const specSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: DEFAULT_PROMPT,
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: 4,
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const specRec = specSession.step();
    const macroUnrolled = buildDoryMacroScene(specSession, {
      displayMode: 'unrolled',
      recordId: specRec.recordId,
    });

    const pickByPartId = (scene: DorySpatialScene, partId: string, row = 0, col = 0) => {
      const box = scene.pickBoxes.find((pb) => pb.partId === partId);
      expect(box, `Missing PickBox '${partId}' in ${scene.viewMode}`).toBeDefined();
      const ray = makeDirectRayAtBox(box!, row, col, 4.0);
      // Raycast against the FULL scene PickBox list so occlusion / overlap bugs are caught too!
      const hit = pickMacroBay(ray, scene.pickBoxes, []);
      expect(hit, `Raycast missed '${partId}'`).not.toBeNull();
      expect(hit!.partId).toBe(partId);
      const cell = pickHitToDoryCell(hit, scene.viewMode);
      expect(cell).not.toBeNull();
      return cell!;
    };

    // Sublayers across Prelude, Recurrent Loops 1 & 4 (same physical layer p=16!), and Coda
    const pre0 = pickByPartId(macroUnrolled, 'exec_e0_p0_t0');
    expect(pre0.physicalLayerIdx).toBe(0);
    expect(pre0.pairedLayerIdx).toBe(0);
    expect(pre0.stage).toBe('input');
    expect(pre0.sublayerKind).toBe('sliding_attention');
    expect(pre0.loopIteration).toBe(0);

    const rec16t1 = pickByPartId(macroUnrolled, 'exec_e16_p16_t1');
    const rec16t4 = pickByPartId(macroUnrolled, 'exec_e136_p16_t4');
    expect(rec16t1.physicalLayerIdx).toBe(16);
    expect(rec16t4.physicalLayerIdx).toBe(16);
    expect(rec16t1.pairedLayerIdx).toBe(8);
    expect(rec16t4.pairedLayerIdx).toBe(8);
    expect(rec16t1.loopIteration).toBe(1);
    expect(rec16t4.loopIteration).toBe(4);
    expect(rec16t1.executedLayerIdx).toBe(16);
    expect(rec16t4.executedLayerIdx).toBe(136);
    expect(rec16t1.modelWeightId).toBe(PARAMS.modelWeightId);
    expect(rec16t4.modelWeightId).toBe(PARAMS.modelWeightId);
    expect(rec16t1.kvBankSlotId).toBe(8);
    expect(rec16t4.kvBankSlotId).toBe(68);

    const coda70 = pickByPartId(macroUnrolled, 'exec_e190_p70_t0');
    expect(coda70.physicalLayerIdx).toBe(70);
    expect(coda70.pairedLayerIdx).toBe(35);
    expect(coda70.stage).toBe('output');
    expect(coda70.sublayerKind).toBe('full_attention');

    // Bookends & Contract A shallow Coda KV bank rail slab
    const embed = pickByPartId(macroUnrolled, 'bookend_embeddings');
    expect(embed.stage).toBe('bookend');
    expect(embed.physicalLayerIdx).toBeNull();

    const shallowBank96 = pickByPartId(macroUnrolled, 'kv_bank_96');
    expect(shallowBank96.kvBankSlotId).toBe(96);
    expect(shallowBank96.kvBankKind).toBe('output_shallow');
    expect(shallowBank96.cacheRole).toBe('spec_draft');
    expect(shallowBank96.kvPolicy).toBe('per_loop');

    // Live XY-oriented 8-channel sheet cell picking + exact per-cell world anchor popover on View 1
    const macroAnchorCell0 = pickByPartId(macroUnrolled, 'anchor_x_s0', 0, 0);
    const macroAnchorCell5 = pickByPartId(macroUnrolled, 'anchor_x_s0', 0, 5);
    expect(macroAnchorCell5.row).toBe(0);
    expect(macroAnchorCell5.col).toBe(5);
    expect(macroAnchorCell5.tensorKey).toBe('x_anchor');
    const macroWithHover0 = buildDoryMacroScene(specSession, {
      displayMode: 'unrolled',
      recordId: specRec.recordId,
      hoveredCell: macroAnchorCell0,
    });
    const macroWithHover5 = buildDoryMacroScene(specSession, {
      displayMode: 'unrolled',
      recordId: specRec.recordId,
      hoveredCell: macroAnchorCell5,
    });
    expect(macroWithHover0.activeCellPopover).not.toBeNull();
    expect(macroWithHover5.activeCellPopover).not.toBeNull();
    expect(macroWithHover5.activeCellPopover!.coordLabel).toContain('ch 5');
    // Popover world MUST anchor to exact hovered cell via computeSheetCellWorldPos (not frame.center!)
    const anchorBox = macroUnrolled.pickBoxes.find((pb) => pb.partId === 'anchor_x_s0')!;
    const expectedAnchorWorld5 = computeSheetCellWorldPos(
      {
        originWorld: anchorBox.frame!.center,
        extent: anchorBox.frame!.extent,
        axisU: anchorBox.frame!.axisU,
        axisN: anchorBox.frame!.axisN,
        cols: 8,
        rows: 1,
      },
      5,
      0,
      0.5,
    );
    expect(macroWithHover5.activeCellPopover!.world[0]).toBeCloseTo(expectedAnchorWorld5[0], 5);
    expect(macroWithHover5.activeCellPopover!.world[1]).toBeCloseTo(expectedAnchorWorld5[1], 5);
    expect(macroWithHover5.activeCellPopover!.world[2]).toBeCloseTo(expectedAnchorWorld5[2], 5);
    expect(macroWithHover5.activeCellPopover!.world[0]).not.toBeCloseTo(
      macroWithHover0.activeCellPopover!.world[0],
      3,
    );
    expect(macroWithHover5.activeCellPopover!.world[0]).not.toBeCloseTo(anchorBox.frame!.center[0], 3);

    // Folded trajectory multi-row sheet cell picking + per-cell popover anchor check
    const macroFolded = buildDoryMacroScene(specSession, {
      displayMode: 'folded',
      recordId: specRec.recordId,
    });
    const trajCell = pickByPartId(macroFolded, 'folded_latent_trajectory', 3, 6);
    expect(trajCell.row).toBe(3);
    expect(trajCell.col).toBe(6);
    const macroFoldedHover = buildDoryMacroScene(specSession, {
      displayMode: 'folded',
      recordId: specRec.recordId,
      hoveredCell: trajCell,
    });
    const trajBox = macroFolded.pickBoxes.find((pb) => pb.partId === 'folded_latent_trajectory')!;
    const expectedTrajWorld = computeSheetCellWorldPos(
      {
        originWorld: trajBox.frame!.center,
        extent: trajBox.frame!.extent,
        axisU: trajBox.frame!.axisU,
        axisN: trajBox.frame!.axisN,
        cols: 8,
        rows: 5,
      },
      6,
      3,
      0.5,
    );
    expect(macroFoldedHover.activeCellPopover).not.toBeNull();
    expect(macroFoldedHover.activeCellPopover!.world[0]).toBeCloseTo(expectedTrajWorld[0], 5);
    expect(macroFoldedHover.activeCellPopover!.world[1]).toBeCloseTo(expectedTrajWorld[1], 5);
    expect(macroFoldedHover.activeCellPopover!.world[2]).toBeCloseTo(expectedTrajWorld[2], 5);

    // 2. View 2 (hypersphere) real raycast across all 3D sheets, PCA nodes, span nodes & popovers
    const stdSession = createDorySession({
      params: PARAMS,
      ropeTables: ROPE_TABLES,
      initialBranch: {
        promptTokens: DEFAULT_PROMPT,
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });
    stdSession.step();
    const sphereScene = buildDoryHypersphereScene(stdSession, {
      physicalLayerIdx: 16,
      loopIteration: 2,
      includeCounterfactualCoda: true,
      counterfactualLoop: 1,
    });

    const sphereSheetTargets: readonly {
      readonly partId: string;
      readonly row: number;
      readonly col: number;
      readonly expectedBadgeSubstr: string;
    }[] = [
      { partId: 'nlerp_a_norm', row: 0, col: 3, expectedBadgeSubstr: 'UNIT-SPHERE RESIDUAL' },
      { partId: 'nlerp_b_norm', row: 0, col: 4, expectedBadgeSubstr: 'UNIT-SPHERE SUBLAYER' },
      { partId: 'nlerp_delta', row: 0, col: 2, expectedBadgeSubstr: 'NLERP CHORD' },
      { partId: 'nlerp_raw_alpha', row: 0, col: 6, expectedBadgeSubstr: 'DR7' },
      { partId: 'nlerp_effective_alpha', row: 0, col: 6, expectedBadgeSubstr: 'DR7' },
      { partId: 'nlerp_blended_pre_norm', row: 0, col: 7, expectedBadgeSubstr: 'EXTRAPOLATED PAST b' },
      { partId: 'nlerp_output', row: 0, col: 7, expectedBadgeSubstr: 'NOT SLERP' },
      { partId: 'latent_trajectory_8d', row: 4, col: 5, expectedBadgeSubstr: '8D' },
      { partId: 'counterfactual_top_probs', row: 1, col: 2, expectedBadgeSubstr: 'DIAGNOSTIC COUNTERFACTUAL' },
    ];

    for (const target of sphereSheetTargets) {
      const box = sphereScene.pickBoxes.find((pb) => pb.partId === target.partId);
      expect(box, `Missing hypersphere PickBox '${target.partId}'`).toBeDefined();
      // For nlerp_effective_alpha ch=6, a 3D extrapolation bar sits directly above the cell,
      // so test isolated box for exact sheet-cell picking when a bar intentionally rises above it!
      const ray = makeDirectRayAtBox(box!, target.row, target.col, 3.0);
      const hit = pickMacroBay(
        ray,
        target.partId === 'nlerp_effective_alpha' ? [box!] : sphereScene.pickBoxes,
        [],
      );
      expect(hit, `Raycast missed '${target.partId}'`).not.toBeNull();
      expect(hit!.partId).toBe(target.partId);

      const cell = pickHitToDoryCell(hit, 'hypersphere')!;
      expect(cell.row).toBe(target.row);
      expect(cell.col).toBe(target.col);
      expect(cell.modelWeightId).toBe(PARAMS.modelWeightId);

      const withPopover = buildDoryHypersphereScene(stdSession, {
        physicalLayerIdx: 16,
        loopIteration: 2,
        includeCounterfactualCoda: true,
        counterfactualLoop: 1,
        hoveredCell: cell,
      });
      expect(withPopover.activeCellPopover, `Missing popover for '${target.partId}'`).not.toBeNull();
      expect(withPopover.activeCellPopover!.badge).toContain(target.expectedBadgeSubstr);
      for (const coord of withPopover.activeCellPopover!.world) {
        expect(Number.isFinite(coord)).toBe(true);
      }
    }

    // Hovering raised 3D extrapolation bars (`nlerp_extrap_ch_6` & `nlerp_extrap_ch_7`) via full-scene
    // raycast MUST return the DR7 unclamped extrapolation popover instead of null!
    for (const ch of DORY_PLANTED_EXTRAPOLATION_SPEC.channelIndices) {
      const barPartId = `nlerp_extrap_ch_${ch}`;
      const barCell = pickByPartId(sphereScene, barPartId);
      expect(barCell.partId).toBe(barPartId);
      expect(barCell.tensorKey).toBe('nlerp_effective_alpha');
      const withBarPopover = buildDoryHypersphereScene(stdSession, {
        physicalLayerIdx: 16,
        loopIteration: 2,
        includeCounterfactualCoda: true,
        counterfactualLoop: 1,
        hoveredCell: barCell,
      });
      expect(withBarPopover.activeCellPopover, `Missing DR7 popover for '${barPartId}'`).not.toBeNull();
      expect(withBarPopover.activeCellPopover!.badge).toContain('DR7 UNCLAMPED EXTRAPOLATION');
      expect(withBarPopover.activeCellPopover!.tensorLabel).toContain(`alpha_hat[${ch}]`);
      expect(withBarPopover.activeCellPopover!.coordLabel).toContain(`ch c=${ch}`);
      expect(withBarPopover.activeCellPopover!.world[1]).toBeGreaterThan(0.2);
      for (const coord of withBarPopover.activeCellPopover!.world) {
        expect(Number.isFinite(coord)).toBe(true);
      }
    }

    // Raycast PCA & exact spanning-subspace pedestal nodes
    const pcaS4 = pickByPartId(sphereScene, 'pca_point_s_4');
    expect(pcaS4.loopIteration).toBe(4);
    expect(pcaS4.tensorKey).toBe('exploratory_pca_2d');

    const spanOut = pickByPartId(sphereScene, 'span_nlerp_out');
    expect(spanOut.physicalLayerIdx).toBe(16);
    expect(spanOut.pairedLayerIdx).toBe(8);
    expect(spanOut.tensorKey).toBe('spanning_subspace_2d');

    // 3. DR21 arbitrary KV-head > 1 round-trip through real pickMacroBay (zero pick.ts edits!)
    const prodKvBox = packPickBoxFromAabb(
      0,
      0,
      0,
      2,
      2,
      2,
      encodeDoryPickMetadata({
        physicalLayerIdx: 22,
        executedLayerIdx: 142,
        loopIteration: 6,
        partId: 'prod_kv_head_3_probe',
        tensorKey: 'q_rot',
        qHeadIndex: 7,
        kvHeadIndex: 3, // exceeds Nano's 0 | 1 typing
        rotaryPlane: 3,
        kvBankSlotId: 135,
        kvBankKind: 'recurrent',
        kvPolicy: 'per_loop',
        modelWeightId: PARAMS.modelWeightId,
      }),
    );
    expect(prodKvBox.kvHeadIndex).toBeUndefined();
    const prodHit = pickMacroBay(makeDirectRayAtBox(prodKvBox), [prodKvBox], []);
    const prodCell = pickHitToDoryCell(prodHit, 'gqa_rope')!;
    expect(prodCell.physicalLayerIdx).toBe(22);
    expect(prodCell.pairedLayerIdx).toBe(11);
    expect(prodCell.loopIteration).toBe(6);
    expect(prodCell.kvHeadIndex).toBe(3);
    expect(prodCell.rotaryPlane).toBe(3);
    expect(prodCell.kvBankSlotId).toBe(135);

    // 4. Static verification that `src/render/pick.ts` remains strictly untouched (DR21)
    const pickSrc = readFileSync(resolve(process.cwd(), 'src/render/pick.ts'), 'utf-8');
    expect(pickSrc).toMatch(/readonly\s+kvHeadIndex\?:\s*0\s*\|\s*1;/);
    expect(pickSrc.toLowerCase()).not.toContain('dory');
  });

  it('enforces D4 zero forbidden Gate 4b/4c/sibling imports and zero Math.random() across ALL src/dory/scene/*.ts', () => {
    const sceneFiles = [
      'src/dory/scene/types.ts',
      'src/dory/scene/provenance.ts',
      'src/dory/scene/macroScene.ts',
      'src/dory/scene/hypersphereScene.ts',
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
        /\b(?:document|window|WebGL2RenderingContext)\b/,
      );
      for (const re of forbiddenImportPatterns) {
        expect(src, `D4 forbidden import in ${relPath}`).not.toMatch(re);
      }
    }
  });
});
