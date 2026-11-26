import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  isAttentionKeyVisible,
  runGqaSim,
  type GqaSimResult,
} from '../src/math/gqaSim';
import { buildCameraSnapshot, type ViewportMetrics } from '../src/render/camera';
import { pickMacroBay, type Ray } from '../src/render/pick';
import {
  SHEET_INSTANCE_FLOATS,
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
} from '../src/render/sheets';
import {
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  getActiveRecord,
  getLatestContextExecution,
  initTwoTowerDiffusionBlock,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionMockAr,
  validateContextSnapshot,
  type TwoTowerSession,
} from '../src/twotower';
import {
  extractMambaFullBlockPositionState,
  FIRST_BIDIRECTIONAL_GQA_BAY_ID,
} from '../src/twotower/scene/mambaDecomposition';
import {
  buildTwoTowerPairedGqaScene,
  CONTEXT_SOLO_GQA_FIT_BOUNDS,
  DENOISER_SOLO_GQA_FIT_BOUNDS,
  fitPairedGqaCamera,
  PAIRED_GQA_FIT_BOUNDS,
  type PairedGqaDiagnostics,
} from '../src/twotower/scene/pairedGqaCutaway';
import {
  buildTwoTowerMambaValueAtlas,
  buildTwoTowerPairedMambaScene,
  CONTEXT_SOLO_MAMBA_FIT_BOUNDS,
  DENOISER_SOLO_MAMBA_FIT_BOUNDS,
  fitPairedMambaCamera,
  PAIRED_MAMBA_FIT_BOUNDS,
  TWOTOWER_MAMBA_ATLAS_HEIGHT,
  TWOTOWER_MAMBA_ATLAS_WIDTH,
  verifyTwoTowerMambaAtlasLayout,
  type PairedMambaDiagnostics,
  type TwoTowerMambaValueAtlas,
} from '../src/twotower/scene/pairedMambaCutaway';
import {
  resolvePickHitTowerIdentity,
  resolveTwoTowerSessionViewState,
} from '../src/twotower/scene/provenance';
import {
  buildTwoTowerGqaValueAtlas,
  TWOTOWER_GQA_ATLAS_HEIGHT,
  TWOTOWER_GQA_ATLAS_WIDTH,
  TWOTOWER_GQA_MAX_KEY,
  TWOTOWER_GQA_MAX_QUERY,
  verifyTwoTowerGqaAtlasLayout,
} from '../src/twotower/scene/twoTowerGqaAtlas';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

const VIEWPORT: ViewportMetrics = {
  hostOriginX: 0,
  hostOriginY: 0,
  hostWidth: 1280,
  hostHeight: 800,
  insetTop: 0,
  insetBottom: 0,
  innerWidth: 1280,
  innerHeight: 800,
  dpr: 1,
};

function makeActiveDiffusionSession(options?: {
  readonly maxRecords?: number;
  readonly refinementCount?: number;
}): TwoTowerSession {
  const model = createDefaultTwoTowerModel();
  let s = createTwoTowerSession(model, {
    mode: 'mask_diffusion',
    blockSize: 4,
    stepsPerBlock: 4,
    confidenceThreshold: 0.9999, // ensure multi-refinement block stays open for r=0, r=1, ...
    eosTokenIds: [],
    ...(options?.maxRecords !== undefined ? { maxRecords: options.maxRecords } : {}),
  });
  s = initTwoTowerDiffusionBlock(s);
  const refinements = options?.refinementCount ?? 1;
  for (let r = 0; r < refinements; r++) {
    s = runTwoTowerDiffusionRefinement(s);
    if (r + 1 < refinements) {
      s = commitTwoTowerRefinementConfidence(s);
    }
  }
  return s;
}

function makeDownwardRayAt(x: number, y: number, z: number): Ray {
  return {
    origin: [x, y + 3.0, z],
    direction: [0, -1, 0],
  };
}

function readAtlasTexel(
  data: Float32Array,
  width: number,
  x: number,
  y: number,
): readonly [number, number] {
  const idx = (y * width + x) * 2;
  return [data[idx], data[idx + 1]];
}

// ============================================================================
// 1. DEDICATED 128×128 TWOTOWER GQA RG32F VALUE ATLAS
// ============================================================================

describe('TwoTower 128×128 GQA RG32F Value Atlas', () => {
  it('verifies exact 128×128 dimensions, 128 KiB byte size, non-overlap, and K=24 / Q=16 max capacity', () => {
    expect(TWOTOWER_GQA_ATLAS_WIDTH).toBe(128);
    expect(TWOTOWER_GQA_ATLAS_HEIGHT).toBe(128);
    expect(TWOTOWER_GQA_MAX_QUERY).toBe(16);
    expect(TWOTOWER_GQA_MAX_KEY).toBe(24);

    // Construct max-capacity K=24, Q=16 (P=8 prefix + S=16 query) for BOTH primary and secondary sims
    const kvCacheMax = {
      seqLen: 8,
      maxTokens: 24,
      K: new Float32Array(8 * 24).fill(0.25),
      V: new Float32Array(8 * 24).fill(-0.15),
    };
    const primaryMax = runGqaSim(16, {
      kvCacheIn: kvCacheMax,
      attentionMaskMode: 'bidirectional_block',
      preNormMod: {
        shift: new Float32Array(8).fill(0.05),
        scale: new Float32Array(8).fill(0.1),
      },
      residualGate: new Float32Array(8).fill(0.8),
    });
    const secondaryMax = runGqaSim(16, {
      kvCacheIn: kvCacheMax,
      attentionMaskMode: 'causal',
    });

    expect(primaryMax.tokens).toBe(16);
    expect(primaryMax.keyTokens).toBe(24);
    expect(secondaryMax.tokens).toBe(16);
    expect(secondaryMax.keyTokens).toBe(24);

    const atlas = buildTwoTowerGqaValueAtlas({
      primarySim: primaryMax,
      secondarySim: secondaryMax,
      group0Head: 5,
      group1Head: 22,
    });

    expect(atlas.width).toBe(128);
    expect(atlas.height).toBe(128);
    expect(atlas.data.length).toBe(128 * 128 * 2);
    expect(atlas.data.byteLength).toBe(128 * 1024);
    expect(atlas.primary.modulatedInput).not.toBeNull();
    expect(atlas.primary.qHeads).toHaveLength(32);
    expect(atlas.primary.kvK[0]).toHaveLength(24);
    expect(atlas.secondary?.kvK[0]).toHaveLength(24);

    const layout = verifyTwoTowerGqaAtlasLayout(atlas);
    expect(layout.errors).toEqual([]);
    expect(layout.valid).toBe(true);
    expect(layout.byteLength).toBe(131_072);
    expect(layout.checkedNonAliasingSliceCount).toBeGreaterThan(200);
    expect(layout.checkedAliasSliceCount).toBe(1); // primary.concat aliasing vOut[0]+vOut[1]
  });

  it('throws RangeError if tokens > 16 or keyTokens > 24 on primary or secondary sim', () => {
    const validSim = runGqaSim(4);
    const badQuerySim = { ...validSim, tokens: 17 } as GqaSimResult;
    const badKeySim = { ...validSim, keyTokens: 25 } as GqaSimResult;

    expect(() => buildTwoTowerGqaValueAtlas({ primarySim: badQuerySim })).toThrow(RangeError);
    expect(() => buildTwoTowerGqaValueAtlas({ primarySim: badKeySim })).toThrow(RangeError);
    expect(() =>
      buildTwoTowerGqaValueAtlas({ primarySim: validSim, secondarySim: badQuerySim }),
    ).toThrow(RangeError);
    expect(() =>
      buildTwoTowerGqaValueAtlas({ primarySim: validSim, secondarySim: badKeySim }),
    ).toThrow(RangeError);
  });

  it('enforces exact channel-G visibility parity: 0 masked in bidirectional Denoiser vs exact causal mask in Context & Mock-AR', () => {
    const s = makeActiveDiffusionSession();
    const view = resolveTwoTowerSessionViewState(s);
    const denTrace = view.resolveDenoiserBayTrace(6);
    const ctxLookup = view.resolveContextBayTrace(6, 0); // Segment 0: [0..6] (tokens=7, prefix=0, causal)
    expect(denTrace?.kind).toBe('*');
    expect(ctxLookup.trace?.kind).toBe('*');
    if (denTrace?.kind !== '*' || ctxLookup.trace?.kind !== '*') throw new Error('expected GQA traces');

    const denSim = denTrace.sim;
    const ctxSim = ctxLookup.trace.sim;
    expect(denSim.attentionMaskMode).toBe('bidirectional_block');
    expect(ctxSim.attentionMaskMode).toBe('causal');
    expect(ctxSim.tokens).toBe(7);

    const atlas = buildTwoTowerGqaValueAtlas({
      primarySim: denSim,
      secondarySim: ctxSim,
      group0Head: 3,
      group1Head: 19,
    });
    expect(verifyTwoTowerGqaAtlasLayout(atlas).valid).toBe(true);

    // Primary (Denoiser bidirectional): 0 masked cells across all 0..P+B-1 keys on BOTH Scores & Softmax
    expect(atlas.primary.maskedScoreCellCount).toBe(0);
    expect(atlas.primary.maskedSoftmaxCellCount).toBe(0);
    for (let grp = 0; grp < 2; grp++) {
      const scoreX0 = Math.round(atlas.primary.scores[grp].rect[0] * 128);
      const scoreY0 = Math.round(atlas.primary.scores[grp].rect[1] * 128);
      const softX0 = Math.round(atlas.primary.softmax[grp].rect[0] * 128);
      const softY0 = Math.round(atlas.primary.softmax[grp].rect[1] * 128);
      const h = grp === 0 ? 3 : 19;
      for (let q = 0; q < denSim.tokens; q++) {
        for (let k = 0; k < denSim.keyTokens; k++) {
          const [sr, sg] = readAtlasTexel(atlas.data, 128, scoreX0 + k, scoreY0 + q);
          const [pr, pg] = readAtlasTexel(atlas.data, 128, softX0 + k, softY0 + q);
          expect(sg).toBe(0);
          expect(pg).toBe(0);
          expect(sr).toBe(denSim.Scores[(h * denSim.tokens + q) * denSim.keyTokens + k]);
          expect(pr).toBe(denSim.Softmax[(h * denSim.tokens + q) * denSim.keyTokens + k]);
        }
      }
    }

    // Secondary (Context causal, T=7, P=0): exact upper triangle k > q strictly masked (G = 1.0) on BOTH Scores & Softmax
    const expectedMaskedPerHead = (ctxSim.tokens * (ctxSim.tokens - 1)) / 2; // 7*6/2 = 21
    expect(atlas.secondary?.maskedScoreCellCount).toBe(expectedMaskedPerHead * 2);
    expect(atlas.secondary?.maskedSoftmaxCellCount).toBe(expectedMaskedPerHead * 2);
    for (let grp = 0; grp < 2; grp++) {
      const scoreX0 = Math.round(atlas.secondary!.scores![grp].rect[0] * 128);
      const scoreY0 = Math.round(atlas.secondary!.scores![grp].rect[1] * 128);
      const softX0 = Math.round(atlas.secondary!.softmax![grp].rect[0] * 128);
      const softY0 = Math.round(atlas.secondary!.softmax![grp].rect[1] * 128);
      for (let q = 0; q < ctxSim.tokens; q++) {
        for (let k = 0; k < ctxSim.keyTokens; k++) {
          const visible = isAttentionKeyVisible('causal', ctxSim.prefixTokens, q, k, ctxSim.keyTokens);
          const [, sg] = readAtlasTexel(atlas.data, 128, scoreX0 + k, scoreY0 + q);
          const [, pg] = readAtlasTexel(atlas.data, 128, softX0 + k, softY0 + q);
          expect(sg).toBe(visible ? 0 : 1);
          expect(pg).toBe(visible ? 0 : 1);
        }
      }
    }

    // Mock-AR causal mode: prefix = P-1 = 7, query = 1 -> K = 8, all k <= 7 visible (0 masked)
    const model = createDefaultTwoTowerModel();
    const sMock = stepTwoTowerSessionMockAr(
      createTwoTowerSession(model, { mode: 'mock_ar', eosTokenIds: [] }),
    );
    const mockView = resolveTwoTowerSessionViewState(sMock);
    const mockDen = mockView.resolveDenoiserBayTrace(6);
    expect(mockDen?.kind).toBe('*');
    if (mockDen?.kind === '*') {
      expect(mockDen.sim.attentionMaskMode).toBe('causal');
      expect(mockDen.sim.prefixTokens).toBe(7);
      expect(mockDen.sim.tokens).toBe(1);
      const mockAtlas = buildTwoTowerGqaValueAtlas({ primarySim: mockDen.sim });
      expect(mockAtlas.primary.maskedScoreCellCount).toBe(0);
      expect(verifyTwoTowerGqaAtlasLayout(mockAtlas).valid).toBe(true);
    }
  });
});

// ============================================================================
// 2. ASYMMETRIC PAIRED 35/65 GQA CUTAWAY
// ============================================================================

describe('Asymmetric Paired 35/65 GQA Cutaway', () => {
  it('exposes explicit 4-length diagnostics, logical read-only wording, and full bidirectional visibility consistency', () => {
    const s = makeActiveDiffusionSession();
    // Select an upper-triangle cell (q = 0, k = P + 3 = 11) in Denoiser Scores — visible in bidirectional_block!
    const scene = buildTwoTowerPairedGqaScene(s, {
      bayId: 6,
      towerFocus: 'paired',
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 0, col: 11 },
      },
      stepperStep: 2,
      termProgress: 0.5,
    });

    const diag = scene.diagnostics as unknown as PairedGqaDiagnostics;
    expect(scene.viewMode).toBe('paired_gqa');
    expect(scene.bayId).toBe(6);
    expect(scene.layerIndex).toBe(5);
    expect(scene.warningBanner).toBeNull();
    expect(scene.fitBounds).toEqual(PAIRED_GQA_FIT_BOUNDS);

    // Explicit 4-length distinction: queryLen=4, contextPrefixKvLen=8, blockKvLen=4, allocatedKvCapacity=24
    expect(diag.queryLen).toBe(4);
    expect(diag.contextPrefixKvLen).toBe(8);
    expect(diag.blockKvLen).toBe(4);
    expect(diag.allocatedKvCapacity).toBe(24);
    expect(diag.lengths.totalAttendedKvLen).toBe(12);
    expect(diag.lengths.unusedKvCapacity).toBe(12);
    expect(scene.calipers?.length).toBeGreaterThanOrEqual(4);

    // Logical read-only prefix conditioning wording (no physical copy)
    expect(diag.conditioningOwnerRole).toBe('context_tower');
    expect(diag.conditioningTransportSemantics).toBe('logical_read_only_no_physical_copy');
    expect(diag.conditioningNote.toLowerCase()).toContain('logical read-only');
    expect(diag.conditioningNote.toLowerCase()).toContain('physical');
    expect(scene.partCatalog['conditioning-bridge'].summary.toLowerCase()).toContain('logical read-only');

    // End-to-end bidirectional visibility agreement for upper-triangle (q=0, k=11)
    expect(scene.activeFormula?.isMasked).toBe(false);
    expect(scene.activeFormula?.subEquation).toContain('Denoiser Scratch Block KV (pos 11)');
    expect(scene.cellPopover?.valueText).not.toContain('−∞');
    expect(scene.focusedAccumulation?.active).toBe(true);
    expect(scene.focusedAccumulation?.totalTerms).toBe(4); // dHead = 4 dot-product terms

    // Active cell grids for Denoiser bidirectional Scores/Softmax have ZERO masked cells
    const scoreGrid = scene.activeCellGrids?.find((g) => g.cols === 12 && g.rows === 4);
    expect(scoreGrid).toBeDefined();
    expect(Array.from(scoreGrid!.masked ?? []).every((v) => v === 0)).toBe(true);

    // Selecting an inherited prefix key (k = 2 < P = 8) reports Inherited Context Prefix KV provenance
    const prefixScene = buildTwoTowerPairedGqaScene(s, {
      bayId: 6,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 1, col: 2 },
      },
    });
    expect(prefixScene.activeFormula?.subEquation).toContain('Inherited Context Prefix KV (pos 2)');
    expect(prefixScene.cellPopover?.coordLabel).toContain('Inherited Context Prefix KV (pos 2)');
  });

  it('NEVER ghosts inherited Context prefix KV during projection reveal (step 1) while block KV reveals progressively', () => {
    const s = makeActiveDiffusionSession();
    const scene = buildTwoTowerPairedGqaScene(s, {
      bayId: 6,
      towerFocus: 'paired',
      stepperStep: 1,
      stepProgress: 0.2, // activeRow = 0 inside B=4 -> block token 0 active, 1..3 uncomputed
    });

    const diag = scene.diagnostics as unknown as PairedGqaDiagnostics;
    expect(scene.sheetRevealData).toBeDefined();
    expect(scene.sheetRevealData!.length).toBe((scene.sheetCount ?? 0) * SHEET_REVEAL_FLOATS);

    // Every single inherited prefix KV slice (0..7 × K/V × 2 groups = 32 slices) MUST be SHEET_REVEAL_NONE!
    expect(diag.prefixKvReveals.length).toBe(8 * 2 * 2);
    for (const rec of diag.prefixKvReveals) {
      expect(rec.traceOrigin).toBe('persistent_cache');
      expect(rec.revealMode).toBe(SHEET_REVEAL_NONE);
      // Verify directly in raw Float32 sheetRevealData GPU buffer too!
      const rawMode = scene.sheetRevealData![rec.sheetIndex * SHEET_REVEAL_FLOATS];
      expect(rawMode).toBe(SHEET_REVEAL_NONE);
    }

    // Newly projected block KV slices (8..11 × K/V × 2 groups = 16 slices) reveal progressively!
    expect(diag.blockKvReveals.length).toBe(4 * 2 * 2);
    const blockModes = new Set(diag.blockKvReveals.map((r) => r.revealMode));
    expect(blockModes.has(SHEET_REVEAL_ROW_MAJOR)).toBe(true);
    expect(blockModes.has(SHEET_REVEAL_UNCOMPUTED)).toBe(true);
    for (const rec of diag.blockKvReveals) {
      expect(rec.traceOrigin).toBe('denoiser_scratch');
      const rawMode = scene.sheetRevealData![rec.sheetIndex * SHEET_REVEAL_FLOATS];
      expect(rawMode).toBe(rec.revealMode);
    }
  });

  it('supports tower-aware picking on both wings, Solo Context / Solo Denoiser framing, and Mock-AR 4-length diagnostics', () => {
    const s = makeActiveDiffusionSession();
    const view = resolveTwoTowerSessionViewState(s);
    const paired = buildTwoTowerPairedGqaScene(s, { bayId: 6, towerFocus: 'paired' });

    const ctxBox = paired.solidPickBoxes.find((b) => b.towerRole === 'context' && b.tensorKey === 'scores');
    const denBox = paired.solidPickBoxes.find((b) => b.towerRole === 'denoiser' && b.tensorKey === 'scores');
    expect(ctxBox).toBeDefined();
    expect(denBox).toBeDefined();
    expect(ctxBox!.frame!.center[0]).toBeLessThan(-3.5);
    expect(denBox!.frame!.center[0]).toBeGreaterThan(-1.8);

    const ctxHit = pickMacroBay(
      makeDownwardRayAt(ctxBox!.frame!.center[0], ctxBox!.frame!.center[1], ctxBox!.frame!.center[2]),
      paired.solidPickBoxes,
      paired.fasciaPickBoxes,
    );
    const denHit = pickMacroBay(
      makeDownwardRayAt(denBox!.frame!.center[0], denBox!.frame!.center[1], denBox!.frame!.center[2]),
      paired.solidPickBoxes,
      paired.fasciaPickBoxes,
    );
    expect(ctxHit?.towerRole).toBe('context');
    expect(denHit?.towerRole).toBe('denoiser');

    const ctxId = resolvePickHitTowerIdentity(view, ctxHit!);
    const denId = resolvePickHitTowerIdentity(view, denHit!);
    expect(ctxId.towerRole).toBe('context');
    expect(denId.towerRole).toBe('denoiser');
    expect(ctxId.parameterFingerprint).not.toBe(denId.parameterFingerprint);

    // Solo Context & Solo Denoiser modes
    const soloCtx = buildTwoTowerPairedGqaScene(s, { bayId: 6, towerFocus: 'context_solo' });
    const soloDen = buildTwoTowerPairedGqaScene(s, { bayId: 6, towerFocus: 'denoiser_solo' });
    expect(soloCtx.fitBounds).toEqual(CONTEXT_SOLO_GQA_FIT_BOUNDS);
    expect(soloDen.fitBounds).toEqual(DENOISER_SOLO_GQA_FIT_BOUNDS);
    expect(soloCtx.solidPickBoxes.every((b) => b.towerRole === 'context')).toBe(true);
    expect(soloDen.solidPickBoxes.some((b) => b.towerRole === 'context')).toBe(false);

    // Camera fitter produces valid finite poses & snapshots
    const pose = fitPairedGqaCamera(paired.fitBounds, VIEWPORT);
    expect(Number.isFinite(pose.distance)).toBe(true);
    expect(pose.distance).toBeGreaterThan(2);
    const snap = buildCameraSnapshot(pose, paired.fitBounds, VIEWPORT);
    expect(snap.viewProj.every(Number.isFinite)).toBe(true);

    // Mock-AR mode 4-length distinction: queryLen=1, contextPrefixKvLen=7 (P-1), blockKvLen=1, cap=24
    const model = createDefaultTwoTowerModel();
    const sMock = stepTwoTowerSessionMockAr(
      createTwoTowerSession(model, { mode: 'mock_ar', eosTokenIds: [] }),
    );
    const mockScene = buildTwoTowerPairedGqaScene(sMock, { bayId: 6 });
    const mockDiag = mockScene.diagnostics as unknown as PairedGqaDiagnostics;
    expect(mockDiag.queryLen).toBe(1);
    expect(mockDiag.contextPrefixKvLen).toBe(7);
    expect(mockDiag.blockKvLen).toBe(1);
    expect(mockDiag.allocatedKvCapacity).toBe(24);
  });

  it('activates pruned-Context fail-safe with explicit warningBanner and never invents synthetic traces', () => {
    // maxRecords = 3: after init + ref0 + conf0 + ref1, context_prefill is pruned from branch records!
    const sPruned = makeActiveDiffusionSession({ maxRecords: 3, refinementCount: 2 });
    expect(getLatestContextExecution(sPruned)).toBeNull();

    const scene = buildTwoTowerPairedGqaScene(sPruned, { bayId: 6, towerFocus: 'paired' });
    const diag = scene.diagnostics as unknown as PairedGqaDiagnostics;
    expect(diag.contextTraceStatus).toBe('pruned_unavailable');
    expect(diag.contextTraceOrigin).toBe('persistent_cache');
    expect(scene.warningBanner).not.toBeNull();
    expect(scene.warningBanner!.toLowerCase()).toContain('prun');

    // Pruned notice plaque and persistent Context KV slices exist, but NO fake Context scores/softmax sheets exist!
    expect(scene.solidPickBoxes.some((b) => b.partId === 'ctx-pruned-notice')).toBe(true);
    expect(scene.solidPickBoxes.some((b) => b.towerRole === 'context' && b.tensorKey === 'kv_k')).toBe(true);
    expect(scene.solidPickBoxes.some((b) => b.towerRole === 'context' && b.tensorKey === 'scores')).toBe(false);
    expect(scene.solidPickBoxes.some((b) => b.towerRole === 'context' && b.tensorKey === 'softmax')).toBe(false);
  });
});

// ============================================================================
// 3. FULL-BLOCK MAMBA ANALYTICAL DECOMPOSITION & CONV1D WINDOW TRACKER
// ============================================================================

describe('Full-Block Mamba Analytical Decomposition & Conv1D Tracker', () => {
  it('matches sim.H_history at 0 ULP across localPos = 0..3, tracks 3→2→1→0 Conv1D replacement, and matches analytical decomposition + all 10 fields', () => {
    const s = makeActiveDiffusionSession();
    const view = resolveTwoTowerSessionViewState(s);
    const denMamba = view.resolveDenoiserBayTrace(1);
    expect(denMamba?.kind).toBe('M');
    if (denMamba?.kind !== 'M') throw new Error('expected Mamba trace');

    const sim = denMamba.sim;
    expect(sim.tokens).toBe(4);
    expect(sim.startTokenOffset).toBe(8);
    expect(sim.initialConvState).toBeDefined();
    expect(sim.initialSsmState).toBeDefined();

    const head = 5;
    const grp = Math.floor(head / 2);
    const stateStride = sim.nHeads * sim.dHead * sim.dState;
    let runningCumDecay = 1.0;

    for (let localPos = 0; localPos < 4; localPos++) {
      const dec = extractMambaFullBlockPositionState(sim, head, localPos, { bayId: 1 });
      runningCumDecay *= sim.dA[head * sim.tokens + localPos];

      // 1. 4-tap Conv1D window replacement: 3 -> 2 -> 1 -> 0 inherited raw history columns vs 4 allocated cache columns!
      expect(dec.convWindow.allocatedCacheColumns).toBe(4);
      expect(dec.convWindow.precedingHistoryColumnsAtPos0).toBe(3);
      expect(dec.convWindow.inheritedHistoryColumnCount).toBe(Math.max(0, 3 - localPos));
      expect(dec.convWindow.blockColumnCount).toBe(Math.min(4, localPos + 1));
      expect(dec.convWindow.zeroPadColumnCount).toBe(0);
      expect(dec.convWindow.taps).toHaveLength(4);
      if (localPos === 0) {
        expect(dec.convWindow.taps.map((t) => t.source)).toEqual([
          'inherited_context_history',
          'inherited_context_history',
          'inherited_context_history',
          'active_block_token',
        ]);
        expect(dec.convWindow.taps.map((t) => t.inheritedCacheSlot)).toEqual([1, 2, 3, null]);
        expect(dec.convWindow.taps.map((t) => t.absoluteTokenPosition)).toEqual([5, 6, 7, 8]);
      } else if (localPos === 3) {
        expect(dec.convWindow.taps.every((t) => t.source === 'active_block_token')).toBe(true);
        expect(dec.convWindow.taps.map((t) => t.absoluteTokenPosition)).toEqual([8, 9, 10, 11]);
      }

      // 2. 0-ULP exact match with sim.H_history[localPos] and prior state without rerunning shortened blocks!
      for (let p = 0; p < 4; p++) {
        for (let n = 0; n < 4; n++) {
          const idx = p * 4 + n;
          const localIdx = (head * 4 + p) * 4 + n;
          const expectedHCurr = sim.H_history[localPos * stateStride + localIdx];
          const expectedHPrev =
            localPos === 0
              ? sim.initialSsmState![localIdx]
              : sim.H_history[(localPos - 1) * stateStride + localIdx];
          expect(Object.is(dec.H_curr[idx], expectedHCurr)).toBe(true);
          expect(Object.is(dec.H_prev[idx], expectedHPrev)).toBe(true);
          expect(Object.is(dec.H_init[idx], sim.initialSsmState![localIdx])).toBe(true);
          expect(Object.is(dec.decayedPrev[idx], Math.fround(dec.decayFactor * expectedHPrev))).toBe(true);
          expect(
            Object.is(
              dec.inputWrite[idx],
              Math.fround(dec.dtValue * dec.xVec[p] * dec.bVec[n]),
            ),
          ).toBe(true);
        }
      }

      // 3. Analytical decomposition: inheritedRetention + accumulatedBlockWrite ≈ H_curr (< 1e-5 Float32 rounding)
      expect(dec.cumulativeDecay).toBeCloseTo(runningCumDecay, 6);
      expect(dec.roundingResidualMaxAbs).toBeLessThan(1e-5);
      expect(dec.roundingNote.toLowerCase()).toContain('float32');
      for (let i = 0; i < 16; i++) {
        const expectedSum = Math.fround(dec.inheritedRetentionMatrix[i] + dec.accumulatedBlockWriteMatrix[i]);
        expect(Math.abs(expectedSum - dec.H_curr[i])).toBeLessThan(1e-5);
      }

      // 4. All position-specific readout, skip, gate, grouped RMS, and normalized output fields match sim exactly!
      expect(dec.groupIndex).toBe(grp);
      expect(Object.is(dec.decayFactor, sim.dA[head * sim.tokens + localPos])).toBe(true);
      expect(Object.is(dec.dtValue, sim.dt[head * sim.tokens + localPos])).toBe(true);
      expect(Object.is(dec.groupRmsDenominator, sim.rms_gate[grp * sim.tokens + localPos])).toBe(true);
      for (let p = 0; p < 4; p++) {
        const ch = head * 4 + p;
        expect(Object.is(dec.recurrentReadout[p], sim.y_ssm[ch * sim.tokens + localPos])).toBe(true);
        expect(Object.is(dec.skipContribution[p], Math.fround(sim.D[head] * sim.x[ch * sim.tokens + localPos]))).toBe(true);
        expect(Object.is(dec.ssmOutputY[p], sim.y[ch * sim.tokens + localPos])).toBe(true);
        expect(Object.is(dec.gatedActivation[p], sim.gated[ch * sim.tokens + localPos])).toBe(true);
        expect(Object.is(dec.normalizedOutput[p], sim.y_norm[ch * sim.tokens + localPos])).toBe(true);
      }

      // 5. Heterogeneous 16-head decay profile & mandatory epistemic notes
      expect(dec.perHeadDecayProfile).toHaveLength(16);
      const uniqueDecays = new Set(dec.perHeadDecayProfile.map((e) => e.dAAtPos.toFixed(4)));
      expect(uniqueDecays.size).toBeGreaterThan(4);
      expect(dec.decaySemanticsDisclaimer.toLowerCase()).toContain('semantic');
      expect(dec.isAfterFirstBidirectionalGqa).toBe(false);
    }

    // Bounds validation
    expect(() => extractMambaFullBlockPositionState(sim, -1, 0)).toThrow(RangeError);
    expect(() => extractMambaFullBlockPositionState(sim, 16, 0)).toThrow(RangeError);
    expect(() => extractMambaFullBlockPositionState(sim, 0, 4)).toThrow(RangeError);
  });
});

// ============================================================================
// 4. ASYMMETRIC PAIRED 35/65 MAMBA CUTAWAY & REFINEMENT BOUNDARY INVARIANT
// ============================================================================

describe('Asymmetric Paired 35/65 Mamba Cutaway', () => {
  it('separates Conv1D and FP32 SSM channels, distinguishes Bay 01 vs Bay 08 causal-after-GQA, and verifies non-overlapping atlas', () => {
    const s = makeActiveDiffusionSession();

    const bay1Scene = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      towerFocus: 'paired',
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'h_curr', headIndex: 2, row: 1, col: 2, tokenIndex: 1 },
      },
      stepperStep: 3,
      termProgress: 0.25,
    });
    const bay8Scene = buildTwoTowerPairedMambaScene(s, {
      bayId: 8,
      towerFocus: 'paired',
      activeHead: 2,
      localTokenIndex: 2,
    });

    const d1 = bay1Scene.diagnostics as unknown as PairedMambaDiagnostics;
    const d8 = bay8Scene.diagnostics as unknown as PairedMambaDiagnostics;

    // Exact atlas layout verification (256×128, zero overlap)
    expect(d1.atlasDimensions).toEqual([TWOTOWER_MAMBA_ATLAS_WIDTH, TWOTOWER_MAMBA_ATLAS_HEIGHT]);
    const atlasCheck = verifyTwoTowerMambaAtlasLayout(bay1Scene.valueAtlas as TwoTowerMambaValueAtlas);
    expect(atlasCheck.errors).toEqual([]);
    expect(atlasCheck.valid).toBe(true);

    // Independent Conv1D & FP32 SSM channels on Context wing AND Center Dual-Channel Bridge
    expect(d1.convChannel.channelId).toBe('mamba_conv_channel');
    expect(d1.ssmChannel.channelId).toBe('mamba_ssm_channel');
    expect(d1.convChannel.storageDtypeMetadata).toBe('bf16');
    expect(d1.ssmChannel.storageDtypeMetadata).toBe('fp32');
    const bridgeConv = bay1Scene.solidPickBoxes.find((b) => b.partId === 'bridge-mamba-conv');
    const bridgeSsm = bay1Scene.solidPickBoxes.find((b) => b.partId === 'bridge-mamba-ssm');
    expect(bridgeConv?.semanticTag).toBe('mamba_conv_channel');
    expect(bridgeSsm?.semanticTag).toBe('mamba_ssm_channel');
    expect(Math.abs(bridgeConv!.frame!.center[2] - bridgeSsm!.frame!.center[2])).toBeGreaterThan(1.0);

    // Bay 01 (before Bay 06) vs Bay 08 (after Bay 06) causal-after-bidirectional-GQA annotation
    expect(FIRST_BIDIRECTIONAL_GQA_BAY_ID).toBe(6);
    expect(d1.isAfterFirstBidirectionalGqa).toBe(false);
    expect(d8.isAfterFirstBidirectionalGqa).toBe(true);
    expect(d1.causalAfterGqaNote.toLowerCase()).toContain('before');
    expect(d8.causalAfterGqaNote.toLowerCase()).toContain('after');
    expect(d1.decomposition?.convWindow.inheritedHistoryColumnCount).toBe(2); // localPos = 1
    expect(d8.decomposition?.convWindow.inheritedHistoryColumnCount).toBe(1); // localPos = 2

    // Native 4×4 grids, focused accumulation, popover, solo modes & camera fit
    expect(bay1Scene.activeCellGrids?.length).toBeGreaterThanOrEqual(4);
    expect(bay1Scene.focusedAccumulation?.active).toBe(true);
    expect(bay1Scene.focusedAccumulation?.totalTerms).toBe(2);
    expect(bay1Scene.cellPopover).not.toBeNull();
    expect(bay1Scene.fitBounds).toEqual(PAIRED_MAMBA_FIT_BOUNDS);

    const soloCtx = buildTwoTowerPairedMambaScene(s, { bayId: 1, towerFocus: 'context_solo' });
    const soloDen = buildTwoTowerPairedMambaScene(s, { bayId: 1, towerFocus: 'denoiser_solo' });
    expect(soloCtx.fitBounds).toEqual(CONTEXT_SOLO_MAMBA_FIT_BOUNDS);
    expect(soloDen.fitBounds).toEqual(DENOISER_SOLO_MAMBA_FIT_BOUNDS);
    expect(fitPairedMambaCamera(bay1Scene.fitBounds, VIEWPORT).distance).toBeGreaterThan(2);
  });

  it('enforces strict partial-accumulation non-exposure in activeFormula & h_curr grid across Mamba Stepper steps 2, 3, and 4', () => {
    const s = makeActiveDiffusionSession();
    const view = resolveTwoTowerSessionViewState(s);
    const denMamba = view.resolveDenoiserBayTrace(1);
    expect(denMamba?.kind).toBe('M');
    if (denMamba?.kind !== 'M') throw new Error('expected Mamba trace');
    const sim = denMamba.sim;

    // 1. Step 2 (Causal Conv1D, `conv_buf` and `x`): partial term 1/4 MUST NOT leak final SiLU or Conv1D output
    const convBufFinal = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'conv_buf', row: 3, col: 9, tapIndex: 3, tokenIndex: 1 },
      },
      stepperStep: 2,
      termProgress: 1.0,
    });
    const convBufPartial = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'conv_buf', row: 3, col: 9, tapIndex: 3, tokenIndex: 1 },
      },
      stepperStep: 2,
      termProgress: 0.1,
    });
    const finalConvSilu = sim.xBC[9 * sim.tokens + 1];
    const finalConvRaw = sim.conv_out[9 * sim.tokens + 1];
    const fmt3 = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`;

    expect(convBufPartial.focusedAccumulation?.active).toBe(true);
    expect(convBufPartial.focusedAccumulation?.isFinal).toBe(false);
    expect(convBufPartial.focusedAccumulation?.totalTerms).toBe(4);
    expect(convBufPartial.activeFormula).not.toBeNull();
    expect(convBufPartial.activeFormula!.value).toBe(convBufPartial.focusedAccumulation!.partialSum);
    expect(convBufPartial.activeFormula!.value).not.toBeCloseTo(finalConvSilu, 3);
    expect(convBufPartial.activeFormula!.valueText).toBe(
      convBufPartial.focusedAccumulation!.valueBadgeText,
    );
    expect(convBufPartial.activeFormula!.equation).toBe(
      convBufPartial.focusedAccumulation!.partialEquation,
    );
    expect(convBufPartial.activeFormula!.headBadge).toBe(
      convBufPartial.focusedAccumulation!.badgeText,
    );
    expect(convBufPartial.activeFormula!.subEquation).toContain(
      convBufPartial.focusedAccumulation!.partialSubEquation,
    );
    // Must NEVER leak final completed Conv1D or SiLU output into equation/valueText/subEquation
    expect(convBufPartial.activeFormula!.equation).not.toContain(fmt3(finalConvSilu));
    expect(convBufPartial.activeFormula!.valueText).not.toContain(fmt3(finalConvSilu));
    expect(convBufPartial.activeFormula!.subEquation).not.toContain(fmt3(finalConvSilu));
    expect(convBufPartial.activeFormula!.subEquation).not.toContain(fmt3(finalConvRaw));
    expect(convBufFinal.focusedAccumulation).toBeNull();

    const xFinal = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'x', headIndex: 2, row: 1, col: 1 },
      },
      stepperStep: 2,
      termProgress: 1.0,
    });
    const xPartial = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'x', headIndex: 2, row: 1, col: 1 },
      },
      stepperStep: 2,
      termProgress: 0.1,
    });
    expect(xPartial.focusedAccumulation?.active).toBe(true);
    expect(xPartial.focusedAccumulation?.isFinal).toBe(false);
    expect(xPartial.activeFormula!.value).toBe(xPartial.focusedAccumulation!.partialSum);
    expect(xPartial.activeFormula!.value).not.toBeCloseTo(xFinal.activeFormula!.value, 3);
    expect(xPartial.activeFormula!.equation).toBe(xPartial.focusedAccumulation!.partialEquation);
    expect(xPartial.activeFormula!.valueText).toBe(xPartial.focusedAccumulation!.valueBadgeText);
    expect(xPartial.activeFormula!.equation).not.toContain(xFinal.activeFormula!.valueText);
    expect(xPartial.activeFormula!.valueText).not.toContain(xFinal.activeFormula!.valueText);
    expect(xPartial.activeFormula!.subEquation).not.toContain(xFinal.activeFormula!.valueText);

    // 2. Step 3 (SSM Update, `h_curr`): partial term 1/2 (decay retention only) MUST NOT leak completed H_curr
    const hFinal = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'h_curr', headIndex: 2, row: 1, col: 2, tokenIndex: 1 },
      },
      stepperStep: 3,
      termProgress: 1.0,
    });
    const hPartial = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'h_curr', headIndex: 2, row: 1, col: 2, tokenIndex: 1 },
      },
      stepperStep: 3,
      termProgress: 0.25,
    });
    const hConverged = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'h_curr', headIndex: 2, row: 1, col: 2, tokenIndex: 1 },
      },
      stepperStep: 3,
      termProgress: 0.75,
    });
    expect(hPartial.focusedAccumulation?.active).toBe(true);
    expect(hPartial.focusedAccumulation?.isFinal).toBe(false);
    expect(hPartial.focusedAccumulation?.totalTerms).toBe(2);
    expect(hPartial.activeFormula!.value).toBe(hPartial.focusedAccumulation!.partialSum);
    expect(hPartial.activeFormula!.value).not.toBeCloseTo(hFinal.activeFormula!.value, 3);
    expect(hPartial.activeFormula!.valueText).toBe(hPartial.focusedAccumulation!.valueBadgeText);
    expect(hPartial.activeFormula!.equation).toBe(hPartial.focusedAccumulation!.partialEquation);
    expect(hPartial.activeFormula!.headBadge).toBe(hPartial.focusedAccumulation!.badgeText);
    expect(hPartial.activeFormula!.equation).not.toContain(hFinal.activeFormula!.valueText);
    expect(hPartial.activeFormula!.valueText).not.toContain(hFinal.activeFormula!.valueText);
    expect(hPartial.activeFormula!.subEquation).not.toContain(hFinal.activeFormula!.valueText);
    // Also verify `den:h_curr` grid masks both override and raw cell buffer at focused (row=1, col=2)
    const hCurrGrid = hPartial.activeCellGrids!.find((g) => g.colDimLabel?.startsWith('H_curr'));
    expect(hCurrGrid).toBeDefined();
    expect(hCurrGrid!.activeValueOverride).toBe(hPartial.focusedAccumulation!.partialSum);
    expect(hCurrGrid!.values[1 * 4 + 2]).toBe(hPartial.focusedAccumulation!.partialSum);
    // Final term 2/2 converges cleanly to completed H_curr
    expect(hConverged.focusedAccumulation?.active).toBe(true);
    expect(hConverged.focusedAccumulation?.isFinal).toBe(true);
    expect(hConverged.activeFormula!.value).toBeCloseTo(hFinal.activeFormula!.value, 5);

    // 3. Step 4 (SSM Readout + D, `y`): partial term 1/5 MUST NOT leak completed readout y
    const yFinal = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'y', headIndex: 2, row: 1, col: 2 * 4 + 1 },
      },
      stepperStep: 4,
      termProgress: 1.0,
    });
    const yPartial = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'y', headIndex: 2, row: 1, col: 2 * 4 + 1 },
      },
      stepperStep: 4,
      termProgress: 0.1,
    });
    const yConverged = buildTwoTowerPairedMambaScene(s, {
      bayId: 1,
      activeHead: 2,
      localTokenIndex: 1,
      selectedCell: {
        tower: 'denoiser',
        cell: { tensor: 'y', headIndex: 2, row: 1, col: 2 * 4 + 1 },
      },
      stepperStep: 4,
      termProgress: 0.9,
    });
    expect(yPartial.focusedAccumulation?.active).toBe(true);
    expect(yPartial.focusedAccumulation?.isFinal).toBe(false);
    expect(yPartial.focusedAccumulation?.totalTerms).toBe(5);
    expect(yPartial.activeFormula!.value).toBe(yPartial.focusedAccumulation!.partialSum);
    expect(yPartial.activeFormula!.value).not.toBeCloseTo(yFinal.activeFormula!.value, 3);
    expect(yPartial.activeFormula!.valueText).toBe(yPartial.focusedAccumulation!.valueBadgeText);
    expect(yPartial.activeFormula!.equation).toBe(yPartial.focusedAccumulation!.partialEquation);
    expect(yPartial.activeFormula!.equation).not.toContain(yFinal.activeFormula!.valueText);
    expect(yPartial.activeFormula!.valueText).not.toContain(yFinal.activeFormula!.valueText);
    expect(yPartial.activeFormula!.subEquation).not.toContain(yFinal.activeFormula!.valueText);
    expect(yConverged.focusedAccumulation?.isFinal).toBe(true);
    expect(yConverged.activeFormula!.value).toBeCloseTo(yFinal.activeFormula!.value, 5);
  });

  it('proves refinement r=0 and r=1 seed from the IDENTICAL committed Context boundary digest & H_init', () => {
    const sR0 = makeActiveDiffusionSession({ refinementCount: 1 });
    const sR1 = runTwoTowerDiffusionRefinement(commitTwoTowerRefinementConfidence(sR0));

    expect(getActiveRecord(sR0).identity.refinementIndex).toBe(0);
    expect(getActiveRecord(sR1).identity.refinementIndex).toBe(1);

    const sceneR0 = buildTwoTowerPairedMambaScene(sR0, { bayId: 1, activeHead: 4, localTokenIndex: 0 });
    const sceneR1 = buildTwoTowerPairedMambaScene(sR1, { bayId: 1, activeHead: 4, localTokenIndex: 0 });

    const dR0 = sceneR0.diagnostics as unknown as PairedMambaDiagnostics;
    const dR1 = sceneR1.diagnostics as unknown as PairedMambaDiagnostics;

    expect(dR0.refinementIndex).toBe(0);
    expect(dR1.refinementIndex).toBe(1);
    expect(dR0.sourceBoundaryDigest).toBe(dR1.sourceBoundaryDigest);
    expect(dR0.contextSnapshotDigest).toBe(dR1.contextSnapshotDigest);
    expect(dR0.convChannel.sourceBoundaryDigest).toBe(dR1.convChannel.sourceBoundaryDigest);
    expect(dR0.ssmChannel.sourceBoundaryDigest).toBe(dR1.ssmChannel.sourceBoundaryDigest);
    expect(dR0.refinementSeedInvariantNote).toContain('SAME committed Context boundary');
    assertBitExactFiniteArray(
      dR0.decomposition!.H_init,
      dR1.decomposition!.H_init,
      'Mamba H_init across r=0 vs r=1',
    );

    // Pruned Context trace fail-safe on Mamba cutaway preserves persistent Conv1D & SSM boundary channels
    const sPruned = makeActiveDiffusionSession({ maxRecords: 3, refinementCount: 2 });
    const prunedMamba = buildTwoTowerPairedMambaScene(sPruned, { bayId: 1 });
    const dPruned = prunedMamba.diagnostics as unknown as PairedMambaDiagnostics;
    expect(dPruned.contextTraceStatus).toBe('pruned_unavailable');
    expect(prunedMamba.warningBanner).not.toBeNull();
    expect(prunedMamba.solidPickBoxes.some((b) => b.partId === 'ctx-pruned-notice')).toBe(true);
    expect(prunedMamba.solidPickBoxes.some((b) => b.partId === 'ctx-conv-state')).toBe(true);
    expect(prunedMamba.solidPickBoxes.some((b) => b.partId === 'ctx-ssm-state')).toBe(true);
  });

  it('guarantees non-mutation of simulation & session buffers and zero DOM / WebGL2 / Math.random()', () => {
    const s = makeActiveDiffusionSession();
    const view = resolveTwoTowerSessionViewState(s);
    const denGqa = view.resolveDenoiserBayTrace(6);
    const denMamba = view.resolveDenoiserBayTrace(1);
    if (denGqa?.kind !== '*' || denMamba?.kind !== 'M') throw new Error('expected traces');

    const gqaScoresBefore = new Float32Array(denGqa.sim.Scores);
    const gqaSoftBefore = new Float32Array(denGqa.sim.Softmax);
    const mambaHistBefore = new Float32Array(denMamba.sim.H_history);
    const ctxSsmBefore = new Float32Array(view.contextSnapshot.ctxState.mambaStates.get(1)!.ssmState);
    const ctxKvBefore = new Float32Array(view.contextSnapshot.ctxState.gqaCaches.get(6)!.K);

    // Build all modes & step states
    for (const towerFocus of ['paired', 'context_solo', 'denoiser_solo'] as const) {
      const gScene = buildTwoTowerPairedGqaScene(s, { bayId: 6, towerFocus, stepperStep: 1, stepProgress: 0.4 });
      const mScene = buildTwoTowerPairedMambaScene(s, { bayId: 8, towerFocus, localTokenIndex: 2, stepperStep: 3, stepProgress: 0.5 });
      expect(gScene.sheetData!.length).toBe(gScene.sheetCount! * SHEET_INSTANCE_FLOATS);
      expect(mScene.sheetData!.length).toBe(mScene.sheetCount! * SHEET_INSTANCE_FLOATS);
    }

    // Verify zero mutation of any simulator or session buffer & intact cryptographic digests
    validateContextSnapshot(view.contextSnapshot, s.model.contextModel);
    assertBitExactFiniteArray(denGqa.sim.Scores, gqaScoresBefore, 'GQA Scores non-mutation');
    assertBitExactFiniteArray(denGqa.sim.Softmax, gqaSoftBefore, 'GQA Softmax non-mutation');
    assertBitExactFiniteArray(denMamba.sim.H_history, mambaHistBefore, 'Mamba H_history non-mutation');
    assertBitExactFiniteArray(view.contextSnapshot.ctxState.mambaStates.get(1)!.ssmState, ctxSsmBefore, 'Context SSM non-mutation');
    assertBitExactFiniteArray(view.contextSnapshot.ctxState.gqaCaches.get(6)!.K, ctxKvBefore, 'Context KV non-mutation');

    // Also verify buildTwoTowerMambaValueAtlas helper directly on bay 42
    const denMamba42 = view.resolveDenoiserBayTrace(42);
    expect(denMamba42?.kind).toBe('M');
    if (denMamba42?.kind === 'M') {
      const dec42 = extractMambaFullBlockPositionState(denMamba42.sim, 11, 3, { bayId: 42 });
      const atlas42 = buildTwoTowerMambaValueAtlas({ primarySim: denMamba42.sim, decomposition: dec42 });
      expect(verifyTwoTowerMambaAtlasLayout(atlas42).valid).toBe(true);
    }

    // Pure TypeScript static audit: ZERO DOM, WebGL2, or Math.random() across all 4 new modules
    const ownedFiles = [
      'src/twotower/scene/twoTowerGqaAtlas.ts',
      'src/twotower/scene/pairedGqaCutaway.ts',
      'src/twotower/scene/mambaDecomposition.ts',
      'src/twotower/scene/pairedMambaCutaway.ts',
    ];
    for (const relPath of ownedFiles) {
      const src = readFileSync(resolve(process.cwd(), relPath), 'utf8');
      expect(src, `${relPath} must not call Math.random()`).not.toMatch(/Math\.random\s*\(/);
      expect(src, `${relPath} must not reference document.*`).not.toMatch(/\bdocument\.[A-Za-z_]/);
      expect(src, `${relPath} must not reference window.*`).not.toMatch(/\bwindow\.[A-Za-z_]/);
      expect(src, `${relPath} must not reference WebGL2RenderingContext`).not.toContain('WebGL2RenderingContext');
      expect(src, `${relPath} must not reference HTMLElement`).not.toContain('HTMLElement');
    }
  });
});
