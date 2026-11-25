import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { runMoeSim, type MoeCellRef } from '../src/math/moeSim';
import { makeViewportMetrics } from '../src/render/camera';
import { SHEET_HL_DIMMED, SHEET_INSTANCE_FLOATS } from '../src/render/sheets';
import {
  assertPresentationSafeNoInfinity,
  commitTwoTowerRefinementConfidence,
  computeGlobalTimestepProjection,
  computeLayerModulationParams,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  getActiveRecord,
  initTwoTowerDiffusionBlock,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  TWOTOWER_MINI_EOS_TOKEN_ID,
  TWOTOWER_MINI_MASK_TOKEN_ID,
  updateTwoTowerSessionConfig,
  validateContextSnapshot,
  type TwoTowerSession,
} from '../src/twotower';
import {
  buildTwoTowerTimestepScene,
  fitTimestepCamera,
  pickHitToTimestepCell,
  type TimestepCutawayDiagnostics,
} from '../src/twotower/scene/timestepCutaway';
import {
  buildTwoTowerMoeScene,
  computeMoeSelectionMargin,
  type TwoTowerMoeDiagnostics,
} from '../src/twotower/scene/twoTowerMoeCutaway';
import {
  buildTwoTowerEmbeddingScene,
  buildTwoTowerOutputScene,
  fitSpatialEmbeddingCamera,
  fitSpatialOutputCamera,
  TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
  type TwoTowerEmbeddingDiagnostics,
  type TwoTowerOutputDiagnostics,
} from '../src/twotower/scene/twoTowerBookendCutaways';

function makeActiveDiffusionSession(options?: {
  readonly blockSize?: number;
  readonly stepsPerBlock?: number;
  readonly initialBlockTokens?: readonly number[];
  readonly maxRecords?: number;
  readonly advanceToSecondRefinement?: boolean;
}): TwoTowerSession {
  const model = createDefaultTwoTowerModel();
  let session = createTwoTowerSession(model, {
    promptTokenIds: [0, 1, 2, 3, 4, 5, 6, 7],
    kvCapacity: 24,
    mode: 'mask_diffusion',
    blockSize: options?.blockSize ?? 4,
    stepsPerBlock: options?.stepsPerBlock ?? 4,
    confidenceThreshold: 0.22,
    temperature: 0,
    ...(options?.maxRecords !== undefined ? { maxRecords: options.maxRecords } : {}),
  });
  session = initTwoTowerDiffusionBlock(session, {
    ...(options?.initialBlockTokens ? { initialBlockTokens: options.initialBlockTokens } : {}),
  });
  session = runTwoTowerDiffusionRefinement(session);
  if (options?.advanceToSecondRefinement) {
    session = commitTwoTowerRefinementConfidence(session);
    const rec = getActiveRecord(session);
    if (rec.activeBlock && rec.activeBlock.nMasked > 0) {
      session = runTwoTowerDiffusionRefinement(session);
    }
  }
  return session;
}

function getSheetHighlightAt(sheetData: Float32Array, sheetIndex: number): number {
  return sheetData[sheetIndex * SHEET_INSTANCE_FLOATS + 11];
}

function readAtlasTexelR(
  atlas: { readonly width: number; readonly height: number; readonly data: Float32Array },
  x: number,
  y: number,
): number {
  return atlas.data[(y * atlas.width + x) * 2];
}

describe('Stage TT3 Wave 3 · Dedicated Timestep Conditioning Cutaway', () => {
  it('renders active diffusion φ(t), tRepr, tEmb, and per-layer shift/scale/gate matching modulation.ts bit-for-bit', () => {
    const session = makeActiveDiffusionSession({ blockSize: 4, stepsPerBlock: 4 });
    const record = getActiveRecord(session);
    expect(record.kind).toBe('diffusion_refinement');
    if (record.kind !== 'diffusion_refinement') throw new Error('expected diffusion_refinement');

    const bayId = 6; // GQA Bay 06 -> layerIndex 5
    const scene = buildTwoTowerTimestepScene(session, {
      bayId,
      localTokenIndex: 1,
      stepperStep: 4,
    });
    const diag = scene.diagnostics as unknown as TimestepCutawayDiagnostics;

    expect(scene.viewMode).toBe('timestep');
    expect(scene.bayId).toBe(6);
    expect(scene.layerIndex).toBe(5);
    expect(diag.bypassed).toBe(false);
    expect(diag.tModel).toBe(1.0);
    expect(diag.nMasked).toBe(4);
    expect(diag.blockSize).toBe(4);
    expect(diag.finalNormModulated).toBe(false);
    expect(diag.residualPreservedUntouched).toBe(true);
    expect(diag.outerGateDim).toBe(8);
    expect(diag.mambaInternalSiluGateDim).toBe(64);
    expect(diag.shiftL2).toBeGreaterThan(0);
    expect(diag.scaleL2).toBeGreaterThan(0);
    expect(diag.gateL2).toBeGreaterThan(0);

    // Exact bit-for-bit numerical parity with modulation.ts & denoiserPass
    const expectedGlobal = computeGlobalTimestepProjection(1.0, session.model.timeConditioning);
    const expectedLayer = computeLayerModulationParams(
      expectedGlobal.tEmb,
      session.model.timeConditioning,
      bayId - 1,
    );
    expect(diag.phi).toEqual(Array.from(expectedGlobal.phi));
    expect(diag.tRepr).toEqual(Array.from(expectedGlobal.tRepr));
    expect(diag.tEmb).toEqual(Array.from(expectedGlobal.tEmb));
    expect(diag.shift).toEqual(Array.from(expectedLayer.shift));
    expect(diag.scale).toEqual(Array.from(expectedLayer.scale));
    expect(diag.gate).toEqual(Array.from(expectedLayer.gate));

    // Geometry, RG32F value atlas, OBB PickBoxes, activeFormula, cellPopover, activeCellGrids
    expect(scene.sheetCount).toBeGreaterThanOrEqual(16);
    expect(scene.sheetData?.length).toBe((scene.sheetCount ?? 0) * SHEET_INSTANCE_FLOATS);
    expect(scene.sheetBasisData?.length).toBe((scene.sheetCount ?? 0) * 6);
    expect(scene.flowVertices?.length ?? 0).toBeGreaterThan(0);
    expect(scene.lineCount).toBeGreaterThan(0);
    expect(scene.valueAtlas).toBeDefined();
    for (let i = 0; i < scene.valueAtlas!.data.length; i++) {
      expect(Number.isFinite(scene.valueAtlas!.data[i])).toBe(true);
    }

    expect(scene.activeFormula).not.toBeNull();
    expect(scene.cellPopover).not.toBeNull();
    expect(scene.activeCellGrids?.length ?? 0).toBeGreaterThanOrEqual(4);
    expect(scene.stepperSteps).toHaveLength(6);

    // Tower-aware OBB PickBoxes & pickHitToTimestepCell round-trip
    expect(scene.solidPickBoxes.length).toBe(scene.sheetCount);
    for (const box of scene.solidPickBoxes) {
      expect(box.towerRole).toBe('denoiser');
      expect(box.executionId).toBe(record.denoiserPass.passExecutionId);
      expect(box.traceOrigin).toBe('denoiser_scratch');
      expect(box.semanticTag).toBe('timestep');
      expect(box.frame).toBeDefined();
    }
    const pickedCell = pickHitToTimestepCell({
      bayId: 6,
      tensorKey: 'layer_mod',
      hitRow: 2,
      hitCol: 3,
      t: 1,
      point: [0, 0, 0],
      isFascia: false,
    });
    expect(pickedCell).toEqual({ tensor: 'layer_mod', row: 2, col: 3 });

    // Camera fitter produces valid finite CameraPose
    const pose = fitTimestepCamera(makeViewportMetrics(0, 0, 1280, 720, 1));
    expect(Number.isFinite(pose.distance)).toBe(true);
    expect(pose.distance).toBeGreaterThan(0);
  });

  it('verifies asymmetric modulation ordering: Bay 01 (M) & Bay 06 (*) use modulate_then_norm while Bay 02 (E) uses norm_then_modulate', () => {
    const session = makeActiveDiffusionSession();

    const bay1Scene = buildTwoTowerTimestepScene(session, { bayId: 1, stepperStep: 5 });
    const bay6Scene = buildTwoTowerTimestepScene(session, { bayId: 6, stepperStep: 5 });
    const bay2Scene = buildTwoTowerTimestepScene(session, { bayId: 2, stepperStep: 5 });

    const d1 = bay1Scene.diagnostics as unknown as TimestepCutawayDiagnostics;
    const d6 = bay6Scene.diagnostics as unknown as TimestepCutawayDiagnostics;
    const d2 = bay2Scene.diagnostics as unknown as TimestepCutawayDiagnostics;

    expect(d1.mixerKind).toBe('M');
    expect(d1.modulationOrder).toBe('modulate_then_norm');
    expect(d1.modulationOrderEquation).toContain('X_mod');
    expect(d1.modulationOrderEquation).toContain('RMSNorm(X_mod)');

    expect(d6.mixerKind).toBe('*');
    expect(d6.modulationOrder).toBe('modulate_then_norm');
    expect(d6.modulationOrderEquation).toContain('X_mod');
    expect(d6.modulationOrderEquation).toContain('RMSNorm(X_mod)');

    expect(d2.mixerKind).toBe('E');
    expect(d2.modulationOrder).toBe('norm_then_modulate');
    expect(d2.modulationOrderEquation).toContain('RMSNorm(X)');
    expect(d2.modulationOrderEquation).toContain('U_mod');

    // Outer residual gate (d=8) explicitly distinguished from Mamba internal SiLU gate (d_inner=64)
    expect(d1.outerGateDim).toBe(8);
    expect(d1.mambaInternalSiluGateDim).toBe(64);
    expect(d1.outerVsInternalGateNote).toContain('d=8');
    expect(d1.outerVsInternalGateNote).toContain('d_inner=64');
    expect(bay1Scene.partCatalog['gate-contrast']?.summary).toContain('d_inner=64');
    expect(bay1Scene.partCatalog['gate-contrast']?.summary).toContain('d=8');

    // Untouched original residual X grid matches real Denoiser trace X bit-for-bit
    const record = getActiveRecord(session);
    if (record.kind !== 'diffusion_refinement') throw new Error('expected diffusion_refinement');
    const bay1Trace = record.denoiserPass.getMambaTrace(1);
    const resXGrid = bay1Scene.activeCellGrids!.find((g) => g.colDimLabel === 'Untouched X')!;
    expect(resXGrid).toBeDefined();
    for (let ch = 0; ch < 8; ch++) {
      expect(resXGrid.values[ch]).toBe(bay1Trace.X[ch * bay1Trace.tokens + 0]);
    }

    // Final RMSNorm strictly unmodulated across all bay kinds
    expect(d1.finalNormModulated).toBe(false);
    expect(d6.finalNormModulated).toBe(false);
    expect(d2.finalNormModulated).toBe(false);
    expect(bay2Scene.partCatalog['final-norm-unmodulated']?.badge).toContain('finalNormModulated = false');
  });

  it('renders BYPASSED status with tModel=null and dims modulation sheets in both Context AR and Mock-AR', () => {
    const model = createDefaultTwoTowerModel();

    // 1. Context AR
    let arSession = createTwoTowerSession(model, {
      promptTokenIds: [0, 1, 2, 3, 4, 5, 6, 7],
      mode: 'ar',
    });
    arSession = stepTwoTowerSessionAr(arSession);
    const arScene = buildTwoTowerTimestepScene(arSession, { bayId: 6 });
    const arDiag = arScene.diagnostics as unknown as TimestepCutawayDiagnostics;

    expect(arDiag.bypassed).toBe(true);
    expect(arDiag.tModel).toBeNull();
    expect(arDiag.shiftL2).toBe(0);
    expect(arDiag.scaleL2).toBe(0);
    expect(arDiag.gateL2).toBe(0);
    expect(arDiag.finalNormModulated).toBe(false);
    expect(arDiag.statusLabel).toContain('BYPASSED in Context AR / Mock-AR (t = null)');
    expect(arScene.warningBanner).toContain('BYPASSED in Context AR / Mock-AR (t = null)');

    // Modulation sheets are explicitly dimmed when bypassed
    const scalarIdx = arScene.solidPickBoxes.findIndex((b) => b.partId === 't-scalar');
    const phiIdx = arScene.solidPickBoxes.findIndex((b) => b.partId === 'phi-sinusoid');
    const layerModIdx = arScene.solidPickBoxes.findIndex((b) => b.partId === 'layer-mod');
    expect(getSheetHighlightAt(arScene.sheetData!, scalarIdx)).toBe(SHEET_HL_DIMMED);
    expect(getSheetHighlightAt(arScene.sheetData!, phiIdx)).toBe(SHEET_HL_DIMMED);
    expect(getSheetHighlightAt(arScene.sheetData!, layerModIdx)).toBe(SHEET_HL_DIMMED);

    // 2. Mock-AR
    let mockArSession = createTwoTowerSession(model, {
      promptTokenIds: [0, 1, 2, 3, 4, 5, 6, 7],
      mode: 'mock_ar',
    });
    mockArSession = stepTwoTowerSessionMockAr(mockArSession);
    const mockScene = buildTwoTowerTimestepScene(mockArSession, { bayId: 2 });
    const mockDiag = mockScene.diagnostics as unknown as TimestepCutawayDiagnostics;

    expect(mockDiag.bypassed).toBe(true);
    expect(mockDiag.tModel).toBeNull();
    expect(mockDiag.modulationOrder).toBe('norm_then_modulate');
    expect(mockDiag.finalNormModulated).toBe(false);
    expect(mockDiag.statusLabel).toContain('BYPASSED in Context AR / Mock-AR (t = null)');
    expect(mockScene.warningBanner).toContain('BYPASSED in Context AR / Mock-AR (t = null)');
  });
});

describe('Stage TT3 Wave 3 · TwoTower MoE Cutaway, FP64 Margin & Partial-Denominator Contract', () => {
  it('computes exact FP64 6th-vs-7th biased-score selection margin and rank ordering', () => {
    const session = makeActiveDiffusionSession();
    const record = getActiveRecord(session);
    if (record.kind !== 'diffusion_refinement') throw new Error('expected diffusion_refinement');
    const moeSim = record.denoiserPass.getMoeTrace(2);

    expect(moeSim.routerPrecisionPolicy).toBe('float64_reference');
    expect(moeSim.scores_for_choice_f64).toBeDefined();

    const margin = computeMoeSelectionMargin(moeSim, 1);
    expect(margin.routerPrecisionPolicy).toBe('float64_reference');
    expect(margin.top8Summary).toHaveLength(8);
    expect(margin.choiceScoreMargin).toBeGreaterThanOrEqual(0);
    expect(margin.choiceScoreMargin).toBe(margin.rank6ChoiceScore - margin.rank7ChoiceScore);
    expect(margin.unbiasedScoreDelta).toBe(margin.rank6UnbiasedScore - margin.rank7UnbiasedScore);
    expect(margin.biasDelta).toBe(margin.rank6Bias - margin.rank7Bias);
    expect(margin.isTightMargin).toBe(margin.choiceScoreMargin < 0.02);

    // Top-6 entries in top8Summary match sim.topk_indices exactly, cutoff strictly after rank 6
    for (let j = 0; j < 6; j++) {
      expect(margin.top8Summary[j].rank).toBe(j + 1);
      expect(margin.top8Summary[j].expertId).toBe(moeSim.topk_indices[1 * 6 + j]);
      expect(margin.top8Summary[j].selected).toBe(true);
      expect(margin.top8Summary[j].cutoffBoundaryAfter).toBe(j === 5);
    }
    expect(margin.top8Summary[6].rank).toBe(7);
    expect(margin.top8Summary[6].expertId).toBe(margin.rank7ExpertId);
    expect(margin.top8Summary[6].selected).toBe(false);
    expect(margin.top8Summary[7].selected).toBe(false);

    // Descending order with lower-index tie-break verified across all 8 entries
    for (let i = 0; i < 7; i++) {
      const a = margin.top8Summary[i];
      const b = margin.top8Summary[i + 1];
      expect(a.choiceScore > b.choiceScore || (a.choiceScore === b.choiceScore && a.expertId < b.expertId)).toBe(true);
    }

    // Also verifies Float32 Nano fallback cleanly when routerPrecisionPolicy === 'float32_nano'
    const nanoSim = runMoeSim(4);
    const nanoMargin = computeMoeSelectionMargin(nanoSim, 0);
    expect(nanoMargin.routerPrecisionPolicy).toBe('float32_nano');
    expect(nanoMargin.choiceScoreMargin).toBeGreaterThanOrEqual(0);
  });

  it('strictly preserves Nano v1.0 partial-denominator non-exposure at stepperStep=2, termProgress=0.4 and reveals final weight at termProgress=0.99', () => {
    const session = makeActiveDiffusionSession();
    const record = getActiveRecord(session);
    if (record.kind !== 'diffusion_refinement') throw new Error('expected diffusion_refinement');
    const sim = record.denoiserPass.getMoeTrace(2);
    const focusToken = 2;
    const slot = 0;
    const winnerId = sim.topk_indices[focusToken * 6 + slot];
    const selectedCell: MoeCellRef = {
      tensor: 'topk_weights',
      row: 0,
      col: slot,
      tokenIndex: focusToken,
      expertId: winnerId,
      rankSlot: slot,
    };

    const exactW = sim.topk_weights[focusToken * 6 + slot];
    const exactW3 = exactW.toFixed(3);
    const exactW2 = exactW.toFixed(2);
    let finalDenom = 0;
    let runningDenom3 = 0;
    for (let m = 0; m < 6; m++) {
      const eid = sim.topk_indices[focusToken * 6 + m];
      const s = sim.router_scores_f64![eid * sim.tokens + focusToken];
      finalDenom += s;
      if (m <= 2) runningDenom3 += s;
    }
    const finalDenom3 = finalDenom.toFixed(3);

    // 1. Mid-accumulation: stepperStep = 2, termProgress = 0.4 -> currentTermIndex = 2 (< 5)
    const midScene = buildTwoTowerMoeScene(session, {
      bayId: 2,
      towerRole: 'denoiser',
      focusToken,
      focusExpert: winnerId,
      selectedCell,
      stepperStep: 2,
      termProgress: 0.4,
    });
    const midDiag = midScene.diagnostics as unknown as TwoTowerMoeDiagnostics;

    expect(midScene.focusedAccumulation).not.toBeNull();
    expect(midScene.focusedAccumulation!.active).toBe(true);
    expect(midScene.focusedAccumulation!.isFinal).toBe(false);
    expect(midScene.focusedAccumulation!.currentTermIndex).toBe(2);
    expect(midScene.focusedAccumulation!.partialSum).toBeCloseTo(Math.fround(runningDenom3), 5);
    expect(midDiag.partialDenominatorActive).toBe(true);
    expect(midDiag.runningDenominator).toBeCloseTo(Math.fround(runningDenom3), 5);

    // activeFormula: ZERO leak of final normalized weight or completed 6-term denominator!
    expect(midScene.activeFormula).not.toBeNull();
    expect(midScene.activeFormula!.value).toBeCloseTo(Math.fround(runningDenom3), 5);
    expect(midScene.activeFormula!.valueText).toContain('Σσ =');
    expect(midScene.activeFormula!.equation).toContain('Σσ =');
    expect(midScene.activeFormula!.equation).toContain('/ Σ₆σ');
    for (const text of [
      midScene.activeFormula!.valueText,
      midScene.activeFormula!.equation,
      midScene.activeFormula!.subEquation,
    ]) {
      expect(text).not.toContain(exactW3);
      expect(text).not.toContain(exactW2);
      expect(text).not.toContain(finalDenom3);
    }

    // cellPopover: ZERO leak of final normalized weight or completed 6-term denominator!
    expect(midScene.cellPopover).not.toBeNull();
    expect(midScene.cellPopover!.isAccumulating).toBe(true);
    expect(midScene.cellPopover!.valueText).toContain('Σσ =');
    for (const text of [
      midScene.cellPopover!.valueText,
      midScene.cellPopover!.equation,
      midScene.cellPopover!.badge ?? '',
    ]) {
      expect(text).not.toContain(exactW3);
      expect(text).not.toContain(exactW2);
      expect(text).not.toContain(finalDenom3);
    }

    // activeCellGrids: Top-6 grid uses running Σ denominator override & sanitized values array!
    const midTopkGrid = midScene.activeCellGrids!.find((g) => g.colDimLabel === 'Top-6 (×2.5)')!;
    expect(midTopkGrid).toBeDefined();
    expect(midTopkGrid.activeValueOverride).toBeCloseTo(Math.fround(runningDenom3), 5);
    expect(midTopkGrid.activeValueTextOverride).toBe(`Σ${midScene.focusedAccumulation!.partialSum.toFixed(2)}`);
    expect(midTopkGrid.activeValueTextOverride).not.toContain(exactW2);
    expect(midTopkGrid.values[slot]).toBeCloseTo(Math.fround(runningDenom3), 5);
    for (let j = slot + 1; j < 6; j++) {
      expect(midTopkGrid.values[j]).toBe(0);
    }

    // diagnostics.sidebarWeights: ZERO final normalized weight leaked!
    expect(midDiag.sidebarWeightTexts).toEqual(['…', '—', '—', '—', '—', '—']);
    for (const entry of midDiag.sidebarWeights) {
      expect(entry.weight).toBeNull();
      expect(entry.displayText).not.toContain(exactW2);
    }

    // 2. Final term: stepperStep = 2, termProgress = 0.99 -> currentTermIndex === 5 (isFinal = true)
    const finalScene = buildTwoTowerMoeScene(session, {
      bayId: 2,
      towerRole: 'denoiser',
      focusToken,
      focusExpert: winnerId,
      selectedCell,
      stepperStep: 2,
      termProgress: 0.99,
    });
    const finalDiag = finalScene.diagnostics as unknown as TwoTowerMoeDiagnostics;

    expect(finalScene.focusedAccumulation).not.toBeNull();
    expect(finalScene.focusedAccumulation!.currentTermIndex).toBe(5);
    expect(finalScene.focusedAccumulation!.isFinal).toBe(true);
    expect(finalDiag.partialDenominatorActive).toBe(false);
    expect(Math.abs(finalScene.activeFormula!.value - exactW)).toBeLessThan(1e-5);
    expect(finalScene.activeFormula!.valueText).toContain(`w = ${exactW3}`);
    expect(finalScene.cellPopover!.valueText).toContain(`w = ${exactW3}`);

    const finalTopkGrid = finalScene.activeCellGrids!.find((g) => g.colDimLabel === 'Top-6 (×2.5)')!;
    expect(finalTopkGrid.activeValueTextOverride).toBeUndefined();
    expect(finalTopkGrid.activeValueOverride).toBeCloseTo(exactW, 5);
    expect(finalDiag.sidebarWeights[slot].state).toBe('revealed');
    expect(finalDiag.sidebarWeights[slot].weight).toBeCloseTo(exactW, 5);
    expect(finalDiag.sidebarWeights[slot].displayText).toBe(exactW2);
  });

  it('verifies 0 B stateless contract, Context vs Denoiser tower selection, and clean pruned_unavailable handling', () => {
    const session = makeActiveDiffusionSession();

    const denScene = buildTwoTowerMoeScene(session, { bayId: 52, towerRole: 'denoiser' });
    const ctxScene = buildTwoTowerMoeScene(session, { bayId: 52, towerRole: 'context' });
    const denDiag = denScene.diagnostics as unknown as TwoTowerMoeDiagnostics;
    const ctxDiag = ctxScene.diagnostics as unknown as TwoTowerMoeDiagnostics;

    expect(denScene.bayId).toBe(52);
    expect(ctxScene.bayId).toBe(52);
    expect(denDiag.cacheBytes).toBe(0);
    expect(denDiag.stateless).toBe(true);
    expect(denDiag.crossTowerCacheTransfer).toBe(false);
    expect(denDiag.modulationStatus).toBe('active');
    expect(denDiag.residualGateApplied).toBe(true);
    expect(denDiag.routerPrecisionPolicy).toBe('float64_reference');

    expect(ctxDiag.cacheBytes).toBe(0);
    expect(ctxDiag.stateless).toBe(true);
    expect(ctxDiag.crossTowerCacheTransfer).toBe(false);
    expect(ctxDiag.modulationStatus).toBe('bypassed');
    expect(ctxDiag.residualGateApplied).toBe(false);
    expect(ctxDiag.routerPrecisionPolicy).toBe('float64_reference');

    // Distinct tower provenance on PickBoxes
    expect(denScene.solidPickBoxes.length).toBeGreaterThan(0);
    expect(ctxScene.solidPickBoxes.length).toBeGreaterThan(0);
    expect(denScene.solidPickBoxes[0].towerRole).toBe('denoiser');
    expect(denScene.solidPickBoxes[0].traceOrigin).toBe('denoiser_scratch');
    expect(denScene.solidPickBoxes[0].absoluteTokenOffset).toBe(8);
    expect(ctxScene.solidPickBoxes[0].towerRole).toBe('context');
    expect(ctxScene.solidPickBoxes[0].executionId).not.toBe(denScene.solidPickBoxes[0].executionId);

    // Pruned history: Context MoE returns clean pruned_unavailable banner without substituting fake trace
    let prunedSession = makeActiveDiffusionSession({ maxRecords: 3 });
    prunedSession = commitTwoTowerRefinementConfidence(prunedSession);
    const prunedScene = buildTwoTowerMoeScene(prunedSession, { bayId: 2, towerRole: 'context' });
    const prunedDiag = prunedScene.diagnostics as unknown as TwoTowerMoeDiagnostics;
    expect(prunedDiag.contextTraceStatus).toBe('pruned_unavailable');
    expect(prunedDiag.selectionMargin).toBeNull();
    expect(prunedDiag.cacheBytes).toBe(0);
    expect(prunedScene.warningBanner).toBeTruthy();
    expect(prunedScene.solidPickBoxes).toHaveLength(0);
  });
});

describe('Stage TT3 Wave 3 · TwoTower Embedding & Output Bookend Cutaways', () => {
  it('exposes independent W_embed fingerprints, handles [MASK] #32 & <eos> #33 at actual block positions, and packs collision-free V=34 atlas', () => {
    // Include BOTH [MASK] (#32) and <eos> (#33) in the Denoiser block at absolute positions [8..11]
    const session = makeActiveDiffusionSession({
      blockSize: 4,
      stepsPerBlock: 4,
      initialBlockTokens: [
        TWOTOWER_MINI_MASK_TOKEN_ID,
        5,
        TWOTOWER_MINI_EOS_TOKEN_ID,
        TWOTOWER_MINI_MASK_TOKEN_ID,
      ],
    });

    const ctxEmb = buildTwoTowerEmbeddingScene(session, { towerRole: 'context' });
    const denEmb = buildTwoTowerEmbeddingScene(session, { towerRole: 'denoiser' });
    const ctxDiag = ctxEmb.diagnostics as unknown as TwoTowerEmbeddingDiagnostics;
    const denDiag = denEmb.diagnostics as unknown as TwoTowerEmbeddingDiagnostics;

    // Strictly distinct independent tower fingerprints
    expect(ctxDiag.weightTableName).toBe('W_embed_ctx');
    expect(denDiag.weightTableName).toBe('W_embed_den');
    expect(ctxDiag.towerParameterFingerprint).toBe(session.model.contextModel.parameterFingerprint);
    expect(denDiag.towerParameterFingerprint).toBe(session.model.denoiserModel.parameterFingerprint);
    expect(ctxDiag.towerParameterFingerprint).not.toBe(denDiag.towerParameterFingerprint);
    expect(ctxDiag.weightsIndependentOfOtherTower).toBe(true);
    expect(denDiag.weightsIndependentOfOtherTower).toBe(true);

    // Exact absolute token offsets & [MASK] / <eos> tracking
    expect(ctxDiag.startTokenOffset).toBe(0);
    expect(ctxDiag.absoluteTokenRange).toEqual([0, 7]);
    expect(ctxDiag.vocabSize).toBe(34);
    expect(denDiag.startTokenOffset).toBe(8);
    expect(denDiag.absoluteTokenRange).toEqual([8, 11]);
    expect(denDiag.maskTokenLocalIndices).toEqual([0, 3]);
    expect(denDiag.maskTokenAbsolutePositions).toEqual([8, 11]);
    expect(denDiag.eosTokenLocalIndices).toEqual([2]);
    expect(denDiag.eosTokenAbsolutePositions).toEqual([10]);
    expect(denDiag.hasMaskTokens).toBe(true);
    expect(denDiag.hasEosTokens).toBe(true);

    // Collision-free V=34 RG32F value atlas: rows #32 ([MASK]) and #33 (<eos>) match W_embed_den bit-exact!
    const denAtlas = denEmb.valueAtlas!;
    expect(denAtlas).toBeDefined();
    for (let ch = 0; ch < 8; ch++) {
      expect(readAtlasTexelR(denAtlas, ch, TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(
        session.model.denoiserModel.W_embed[TWOTOWER_MINI_MASK_TOKEN_ID * 8 + ch],
      );
      expect(readAtlasTexelR(denAtlas, ch, TWOTOWER_MINI_EOS_TOKEN_ID)).toBe(
        session.model.denoiserModel.W_embed[TWOTOWER_MINI_EOS_TOKEN_ID * 8 + ch],
      );
    }

    // Tower-aware PickBox provenance + synthetic-miniature disclosure
    expect(denEmb.solidPickBoxes.length).toBeGreaterThan(0);
    for (const box of denEmb.solidPickBoxes) {
      expect(box.towerRole).toBe('denoiser');
      expect(box.traceOrigin).toBe('denoiser_scratch');
      expect(box.absoluteTokenOffset).toBe(8);
    }
    expect(ctxEmb.solidPickBoxes[0].towerRole).toBe('context');
    expect(ctxEmb.solidPickBoxes[0].absoluteTokenOffset).toBe(0);
    expect(denDiag.syntheticMiniatureDisclosure).toBe(TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE);
    expect(denEmb.epistemicNote).toContain('synthetic miniature');

    // Camera fit sanity check
    const pose = fitSpatialEmbeddingCamera(makeViewportMetrics(0, 0, 1280, 720, 1));
    expect(Number.isFinite(pose.distance)).toBe(true);
  });

  it('distinguishes Context AR shifted vs Denoiser unshifted block logits, verifies unmodulated norm_f, and attaches Infinity-free SamplerStepPresentation', () => {
    // Advance through refinement 0 -> confidence commit -> refinement 1 -> confidence commit so
    // the block contains BOTH locked prior tokens AND newly evaluated masked tokens!
    let session = makeActiveDiffusionSession({
      blockSize: 4,
      stepsPerBlock: 4,
      advanceToSecondRefinement: true,
    });
    const recAtRef1 = getActiveRecord(session);
    if (recAtRef1.kind === 'diffusion_refinement') {
      session = commitTwoTowerRefinementConfidence(session);
    }

    const ctxOut = buildTwoTowerOutputScene(session, { towerRole: 'context' });
    const denOut = buildTwoTowerOutputScene(session, { towerRole: 'denoiser', focusToken: 1 });
    const ctxDiag = ctxOut.diagnostics as unknown as TwoTowerOutputDiagnostics;
    const denDiag = denOut.diagnostics as unknown as TwoTowerOutputDiagnostics;

    // Distinct W_lm_head fingerprints & strictly unmodulated norm_f in BOTH towers
    expect(ctxDiag.weightTableName).toBe('W_lm_head_ctx');
    expect(denDiag.weightTableName).toBe('W_lm_head_den');
    expect(ctxDiag.towerParameterFingerprint).toBe(session.model.contextModel.parameterFingerprint);
    expect(denDiag.towerParameterFingerprint).toBe(session.model.denoiserModel.parameterFingerprint);
    expect(ctxDiag.towerParameterFingerprint).not.toBe(denDiag.towerParameterFingerprint);
    expect(ctxDiag.finalNormModulated).toBe(false);
    expect(denDiag.finalNormModulated).toBe(false);

    // Explicit prediction semantics distinction: Context AR shifted vs Denoiser unshifted block logits
    expect(ctxDiag.predictionSemantics).toBe('causal_next_token_shifted');
    expect(denDiag.predictionSemantics).toBe('unshifted_block_diffusion');
    expect(denDiag.startTokenOffset).toBe(8);
    expect(denDiag.absoluteFocusPosition).toBe(9);
    expect(denOut.subtitle).toContain('Unshifted Block Logits');
    expect(ctxOut.subtitle).toContain('Context AR Next-Token Prediction');

    // Lossless transpose check: Denoiser Output Probs & Logits in atlas match denoiserPass row-major [B, V] bit-exact!
    const activeRec = getActiveRecord(session);
    const denPass = activeRec.denoiserPass!;
    const outAtlas = denOut.valueAtlas!;
    for (const v of [0, 7, TWOTOWER_MINI_MASK_TOKEN_ID, TWOTOWER_MINI_EOS_TOKEN_ID]) {
      expect(readAtlasTexelR(outAtlas, 8, v)).toBe(denPass.logits[1 * 34 + v]);
      expect(readAtlasTexelR(outAtlas, 9, v)).toBe(denPass.probs[1 * 34 + v]);
    }

    // SamplerStepPresentation attached, strictly Infinity-free, locked prior slots say "Locked (prior step)", [MASK] prob === 0
    expect(denDiag.samplerPresentation).not.toBeNull();
    const pres = denDiag.samplerPresentation!;
    expect(() => assertPresentationSafeNoInfinity(pres)).not.toThrow();
    expect(JSON.stringify(pres)).not.toContain('Infinity');

    const lockedSlots = pres.slots.filter((s) => !s.wasMaskedBefore);
    expect(lockedSlots.length).toBeGreaterThanOrEqual(1);
    for (const locked of lockedSlots) {
      expect(locked.confidenceDisplay.kind).toBe('locked_prior');
      expect(locked.confidenceDisplay.unscaledConfidence).toBeNull();
      expect(locked.confidenceDisplay.displayText).toBe('Locked (prior step)');
      expect(locked.constrainedMdlm.maskTokenProb).toBe(0);
      expect(locked.constrainedMdlm.maskTokenExcluded).toBe(true);
    }
    for (const slot of pres.slots) {
      expect(slot.constrainedMdlm.maskTokenProb).toBe(0);
      expect(slot.constrainedMdlm.maskTokenExcluded).toBe(true);
      expect(slot.rawLmHead.topK.length).toBeGreaterThan(0);
      expect(slot.selection.selectedTokenId).toBeGreaterThanOrEqual(0);
    }

    // Also verifies live refinement preview (before commitTwoTowerRefinementConfidence) works & has zero Infinity
    const refOnlySession = makeActiveDiffusionSession({
      blockSize: 4,
      stepsPerBlock: 4,
      initialBlockTokens: [5, TWOTOWER_MINI_MASK_TOKEN_ID, TWOTOWER_MINI_MASK_TOKEN_ID, TWOTOWER_MINI_MASK_TOKEN_ID],
    });
    const refOut = buildTwoTowerOutputScene(refOnlySession, { towerRole: 'denoiser' });
    const refDiag = refOut.diagnostics as unknown as TwoTowerOutputDiagnostics;
    expect(refDiag.samplerPresentationSource).toBe('live_refinement_preview');
    expect(refDiag.samplerPresentation).not.toBeNull();
    expect(() => assertPresentationSafeNoInfinity(refDiag.samplerPresentation)).not.toThrow();
    expect(refDiag.samplerPresentation!.slots[0].confidenceDisplay.displayText).toBe('Locked (prior step)');

    // Synthetic-miniature disclosure, Mock-AR startTokenOffset = P - 1, and pruned Context Output fallback
    expect(denDiag.syntheticMiniatureDisclosure).toBe(TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE);
    let mockSession = updateTwoTowerSessionConfig(
      createTwoTowerSession(session.model, { promptTokenIds: [0, 1, 2, 3, 4, 5, 6, 7], mode: 'mock_ar' }),
      { mode: 'mock_ar' },
    );
    mockSession = stepTwoTowerSessionMockAr(mockSession);
    const mockEmb = buildTwoTowerEmbeddingScene(mockSession, { towerRole: 'denoiser' });
    const mockOut = buildTwoTowerOutputScene(mockSession, { towerRole: 'denoiser' });
    expect((mockEmb.diagnostics as unknown as TwoTowerEmbeddingDiagnostics).startTokenOffset).toBe(7); // P - 1
    expect((mockOut.diagnostics as unknown as TwoTowerOutputDiagnostics).predictionSemantics).toBe('causal_next_token_shifted');
    expect((mockOut.diagnostics as unknown as TwoTowerOutputDiagnostics).samplerPresentation).toBeNull();

    // Pruned Context history cleanly reports pruned_unavailable without faking Bay 52 hidden states
    let prunedSession = makeActiveDiffusionSession({ maxRecords: 3 });
    prunedSession = commitTwoTowerRefinementConfidence(prunedSession);
    const prunedCtxOut = buildTwoTowerOutputScene(prunedSession, { towerRole: 'context' });
    const prunedCtxOutDiag = prunedCtxOut.diagnostics as unknown as TwoTowerOutputDiagnostics;
    expect(prunedCtxOutDiag.contextTraceStatus).toBe('pruned_unavailable');
    expect(prunedCtxOutDiag.finalNormModulated).toBe(false);
    expect(prunedCtxOut.warningBanner).toBeTruthy();

    // Camera fit sanity check
    const pose = fitSpatialOutputCamera(makeViewportMetrics(0, 0, 1280, 720, 1));
    expect(Number.isFinite(pose.distance)).toBe(true);
  });
});

describe('Stage TT3 Wave 3 · Session Non-Mutation & Zero DOM/WebGL/Math.random Guardrails', () => {
  it('never mutates any session, model, snapshot, or Denoiser pass typed-array buffers across all 3 builders', () => {
    const session = makeActiveDiffusionSession({ blockSize: 4, stepsPerBlock: 4 });
    const record = getActiveRecord(session);
    if (record.kind !== 'diffusion_refinement') throw new Error('expected diffusion_refinement');

    // Capture exact byte snapshots before calling all cutaway builders
    const snapDigestBefore = record.contextSnapshot.snapshotDigest;
    const arLogitsBefore = record.contextSnapshot.arLogits.slice();
    const denLogitsBefore = record.denoiserPass.logits.slice();
    const denRes0Before = record.denoiserPass.residualStreams[0].slice();
    const denRes52Before = record.denoiserPass.residualStreams[52].slice();
    const ctxEmbBefore = session.model.contextModel.W_embed.slice();
    const denLmBefore = session.model.denoiserModel.W_lm_head.slice();
    const table5Before = session.model.timeConditioning.scaleShiftTables[5].slice();

    // Execute all spatial cutaway builders across both towers & stepper states
    buildTwoTowerTimestepScene(session, { bayId: 1, stepperStep: 2 });
    buildTwoTowerTimestepScene(session, { bayId: 2, stepperStep: 5 });
    buildTwoTowerTimestepScene(session, { bayId: 6, stepperStep: 4 });
    buildTwoTowerMoeScene(session, { bayId: 2, towerRole: 'denoiser', stepperStep: 2, termProgress: 0.4 });
    buildTwoTowerMoeScene(session, { bayId: 2, towerRole: 'denoiser', stepperStep: 2, termProgress: 0.99 });
    buildTwoTowerMoeScene(session, { bayId: 52, towerRole: 'context' });
    buildTwoTowerEmbeddingScene(session, { towerRole: 'context' });
    buildTwoTowerEmbeddingScene(session, { towerRole: 'denoiser' });
    buildTwoTowerOutputScene(session, { towerRole: 'context' });
    buildTwoTowerOutputScene(session, { towerRole: 'denoiser' });

    // Snapshot remains strictly valid & every buffer is bit-for-bit untouched
    expect(() => validateContextSnapshot(record.contextSnapshot, session.model.contextModel)).not.toThrow();
    expect(record.contextSnapshot.snapshotDigest).toBe(snapDigestBefore);
    expect(record.contextSnapshot.arLogits).toEqual(arLogitsBefore);
    expect(record.denoiserPass.logits).toEqual(denLogitsBefore);
    expect(record.denoiserPass.residualStreams[0]).toEqual(denRes0Before);
    expect(record.denoiserPass.residualStreams[52]).toEqual(denRes52Before);
    expect(session.model.contextModel.W_embed).toEqual(ctxEmbBefore);
    expect(session.model.denoiserModel.W_lm_head).toEqual(denLmBefore);
    expect(session.model.timeConditioning.scaleShiftTables[5]).toEqual(table5Before);
  });

  it('guarantees ZERO DOM, WebGL2, or Math.random() usage across all 3 new spatial cutaway modules', () => {
    const files = [
      'src/twotower/scene/timestepCutaway.ts',
      'src/twotower/scene/twoTowerMoeCutaway.ts',
      'src/twotower/scene/twoTowerBookendCutaways.ts',
    ];
    const forbiddenPatterns: readonly [string, RegExp][] = [
      ['Math.random', /\bMath\.random\b/],
      ['document.', /\bdocument\./],
      ['window.', /\bwindow\./],
      ['WebGL2RenderingContext', /\bWebGL2RenderingContext\b/],
      ['HTMLCanvasElement', /\bHTMLCanvasElement\b/],
      ['HTMLElement', /\bHTMLElement\b/],
      ['SVGSVGElement', /\bSVGSVGElement\b/],
      ['runtime ui import', /^\s*import\s+(?!type\b)[^'"]*\/ui\//m],
      ['render device import', /from\s+['"][^'"]*render\/device['"]/],
    ];

    for (const relPath of files) {
      const rawSrc = readFileSync(resolve(process.cwd(), relPath), 'utf8');
      // Strip block & line comments so JSDoc disclaimers ("zero Math.random()") do not false-positive
      const codeOnly = rawSrc
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/[^\n]*/g, '');
      for (const [label, regex] of forbiddenPatterns) {
        expect(codeOnly, `${relPath} executable code must not contain ${label}`).not.toMatch(regex);
      }
    }
  });
});
