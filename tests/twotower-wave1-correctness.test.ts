import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { GQA_SIM_CONFIG } from '../src/math/gqaSim';
import { MAMBA_SIM_CONFIG } from '../src/math/mambaSim';
import {
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  getActiveRecord,
  getLatestContextExecution,
  initTwoTowerDiffusionBlock,
  navigateTwoTowerSession,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  TWOTOWER_MINI_CONFIG,
  updateTwoTowerSessionConfig,
  type TwoTowerSession,
} from '../src/twotower';
import {
  buildTwoTowerMacroScene,
  type TwoTowerMacroDiagnostics,
} from '../src/twotower/scene/dualMacroScene';
import {
  extractMambaFullBlockPositionState,
} from '../src/twotower/scene/mambaDecomposition';
import {
  buildTwoTowerPairedGqaScene,
  type PairedGqaDiagnostics,
} from '../src/twotower/scene/pairedGqaCutaway';
import {
  buildTwoTowerPairedMambaScene,
  type PairedMambaDiagnostics,
} from '../src/twotower/scene/pairedMambaCutaway';
import {
  resolveTwoTowerSessionViewState,
  TWOTOWER_MINI_MODEL_INVENTORY,
} from '../src/twotower/scene/provenance';
import {
  buildTwoTowerTimestepScene,
  type TimestepCutawayDiagnostics,
} from '../src/twotower/scene/timestepCutaway';
import {
  buildTwoTowerEmbeddingScene,
  buildTwoTowerOutputScene,
  type TwoTowerEmbeddingDiagnostics,
  type TwoTowerOutputDiagnostics,
} from '../src/twotower/scene/twoTowerBookendCutaways';
import { buildTwoTowerGqaValueAtlas } from '../src/twotower/scene/twoTowerGqaAtlas';
import {
  buildTwoTowerMoeScene,
  type TwoTowerMoeDiagnostics,
} from '../src/twotower/scene/twoTowerMoeCutaway';

function completeBlockToUncommitted(base: TwoTowerSession): TwoTowerSession {
  let s = initTwoTowerDiffusionBlock(base);
  while (getActiveRecord(s).lifecyclePhase !== 'completed_uncommitted') {
    s = runTwoTowerDiffusionRefinement(s);
    s = commitTwoTowerRefinementConfidence(s);
  }
  return s;
}

describe('Stage TT4 Wave 1 · Presentation Correctness, Timestep Semantics, Historical Provenance & Dimensional Consistency', () => {
  const sharedModel = createDefaultTwoTowerModel();

  describe('1. Canonical Timestep Semantics (t = nMasked / B, NEVER r / K) & Stage Separation', () => {
    it('strictly separates inactive_no_pass_yet from bypassed_ar and bypassed_mock_ar across prefill, AR, Mock-AR, and Block 0 init', () => {
      // A. context_prefill in mask_diffusion mode -> inactive_no_pass_yet (NOT bypassed_ar/bypassed_mock_ar!)
      const prefillDiff = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        eosTokenIds: [],
      });
      const prefillView = resolveTwoTowerSessionViewState(prefillDiff);
      expect(prefillView.timestepStatus.conditioningMode).toBe('inactive_no_pass_yet');
      expect(prefillView.timestepStatus.executedPass).toBeNull();
      expect(prefillView.timestepStatus.currentBlockMask).toBeNull();
      expect(prefillView.timestepStatus.nextEligibleStep).toBeNull();
      expect(prefillView.timestepStatus.blockCompletionState).toBe('no_active_block');

      const prefillTime = buildTwoTowerTimestepScene(prefillDiff, { bayId: 6, stepperStep: 0 });
      const prefillTimeDiag = prefillTime.diagnostics as unknown as TimestepCutawayDiagnostics;
      expect(prefillTimeDiag.conditioningMode).toBe('inactive_no_pass_yet');
      expect(prefillTimeDiag.bypassed).toBe(false);
      expect(prefillTimeDiag.inactiveNoPassYet).toBe(true);
      expect(prefillTimeDiag.bypassReason).toBeNull();
      expect(prefillTimeDiag.tModel).toBeNull();
      expect(prefillTimeDiag.statusLabel).not.toContain('BYPASSED');
      expect(prefillTime.activeFormula?.equation).toContain('INACTIVE');
      expect(prefillTime.activeFormula?.equation).not.toContain('BYPASSED in Context AR');

      // B. diffusion_block_init (Block 0 before pass 0) -> inactive_no_pass_yet with nextEligibleStep = 4/4 = 1.0
      const initBlock0 = initTwoTowerDiffusionBlock(prefillDiff);
      const initView = resolveTwoTowerSessionViewState(initBlock0);
      expect(initView.timestepStatus.conditioningMode).toBe('inactive_no_pass_yet');
      expect(initView.timestepStatus.executedPass).toBeNull();
      expect(initView.timestepStatus.currentBlockMask).toEqual({
        currentUnresolvedCount: 4,
        blockSize: 4,
        currentMaskFraction: 1.0,
        isBlockComplete: false,
        isCommittedToContext: false,
      });
      expect(initView.timestepStatus.nextEligibleStep).toEqual({
        nextTModel: 1.0,
        nextNMasked: 4,
        blockSize: 4,
        remainingSteps: 4,
      });
      expect(initView.timestepStatus.blockCompletionState).toBe('refining');

      const initMacro = buildTwoTowerMacroScene(initBlock0);
      const initMacroDiag = initMacro.diagnostics as unknown as TwoTowerMacroDiagnostics;
      expect(initMacroDiag.timestepValue).toBeNull();
      expect(initMacroDiag.timestepStatus.conditioningMode).toBe('inactive_no_pass_yet');
      expect(initMacroDiag.timestepStatusLabel).toContain('INACTIVE (no pass yet)');
      expect(initMacroDiag.timestepStatusLabel).toContain('Next t = 1.00 (4/4 masked)');

      const initTime = buildTwoTowerTimestepScene(initBlock0, { bayId: 6, stepperStep: 0 });
      const initTimeDiag = initTime.diagnostics as unknown as TimestepCutawayDiagnostics;
      expect(initTimeDiag.conditioningMode).toBe('inactive_no_pass_yet');
      expect(initTimeDiag.bypassed).toBe(false);
      expect(initTimeDiag.inactiveNoPassYet).toBe(true);
      expect(initTimeDiag.bypassReason).toBeNull();
      expect(initTime.warningBanner).toBeNull();
      expect(initTime.landmarks.find((l) => l.id === 'time-t-scalar')?.label).toContain('INACTIVE');

      // C. Genuine AR bypass -> bypassed_ar (tModel = null, bypassed = true)
      const arSession = stepTwoTowerSessionAr(
        createTwoTowerSession(sharedModel, { mode: 'ar', eosTokenIds: [] }),
      );
      const arTime = buildTwoTowerTimestepScene(arSession, { bayId: 6, stepperStep: 0 });
      const arTimeDiag = arTime.diagnostics as unknown as TimestepCutawayDiagnostics;
      expect(arTimeDiag.conditioningMode).toBe('bypassed_ar');
      expect(arTimeDiag.bypassed).toBe(true);
      expect(arTimeDiag.inactiveNoPassYet).toBe(false);
      expect(arTimeDiag.tModel).toBeNull();
      expect(arTimeDiag.statusLabel).toBe('BYPASSED in Context AR (t = null)');

      // D. Genuine Mock-AR bypass -> bypassed_mock_ar (tModel = null, bypassed = true)
      const mockArSession = stepTwoTowerSessionMockAr(
        createTwoTowerSession(sharedModel, { mode: 'mock_ar', eosTokenIds: [] }),
      );
      const mockTime = buildTwoTowerTimestepScene(mockArSession, { bayId: 6, stepperStep: 0 });
      const mockTimeDiag = mockTime.diagnostics as unknown as TimestepCutawayDiagnostics;
      expect(mockTimeDiag.conditioningMode).toBe('bypassed_mock_ar');
      expect(mockTimeDiag.bypassed).toBe(true);
      expect(mockTimeDiag.inactiveNoPassYet).toBe(false);
      expect(mockTimeDiag.tModel).toBeNull();
      expect(mockTimeDiag.statusLabel).toBe('BYPASSED in Mock-AR (t = null)');
    });

    it('never pairs executedPass tModel with post-commit nMasked, handles diagnostic tOverride separately, and marks completed blocks as requiring no further pass', () => {
      const base = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999, // force 1 unmask per step so post-commit is 3/4, 2/4, 1/4, 0/4
        eosTokenIds: [],
      });
      const init = initTwoTowerDiffusionBlock(base);

      // 1. Diagnostic override tOverride = 0.37 on 4/4 masked input
      const refOverride = runTwoTowerDiffusionRefinement(init, { tOverride: 0.37 });
      const ovView = resolveTwoTowerSessionViewState(refOverride);
      expect(ovView.timestepStatus.conditioningMode).toBe('active_diffusion');
      expect(ovView.timestepStatus.executedPass).toEqual({
        tModel: 0.37,
        nMaskedInput: 4,
        blockSize: 4,
        empiricalMaskFraction: 1.0,
        isDiagnosticOverride: true,
        overrideValue: 0.37,
        passExecutionId: getActiveRecord(refOverride).denoiserPass!.passExecutionId,
        sourceRecordIndex: 2,
        sourceBlockId: 0,
        sourceRefinementIndex: 0,
      });
      expect(ovView.timestepStatus.nextEligibleStep).toBeNull(); // awaiting confidence commit

      const ovTime = buildTwoTowerTimestepScene(refOverride, { bayId: 6, stepperStep: 0 });
      const ovTimeDiag = ovTime.diagnostics as unknown as TimestepCutawayDiagnostics;
      expect(ovTimeDiag.isDiagnosticOverride).toBe(true);
      expect(ovTimeDiag.overrideValue).toBe(0.37);
      expect(ovTimeDiag.empiricalMaskFraction).toBe(1.0);
      expect(ovTimeDiag.nMasked).toBe(4);
      expect(ovTime.activeFormula?.equation).toContain('t_override = 0.3700');
      expect(ovTime.activeFormula?.equation).toContain('4 / 4 = 1.0000');
      // Must NEVER claim false equality "4 / 4 = 0.3700"!
      expect(ovTime.activeFormula?.equation).not.toContain('4 / 4 = 0.37');

      // 2. Canonical refinement 0 (4/4 -> t = 1.0) -> confidence commit (post-commit 3/4)
      const ref0 = runTwoTowerDiffusionRefinement(init);
      const conf0 = commitTwoTowerRefinementConfidence(ref0);
      const conf0View = resolveTwoTowerSessionViewState(conf0);
      expect(conf0View.timestepStatus.conditioningMode).toBe('active_diffusion');
      expect(conf0View.timestepStatus.executedPass!.tModel).toBe(1.0);
      expect(conf0View.timestepStatus.executedPass!.nMaskedInput).toBe(4);
      expect(conf0View.timestepStatus.executedPass!.empiricalMaskFraction).toBe(1.0);
      expect(conf0View.timestepStatus.executedPass!.isDiagnosticOverride).toBe(false);
      expect(conf0View.timestepStatus.executedPass!.overrideValue).toBeNull();
      expect(conf0View.timestepStatus.currentBlockMask).toEqual({
        currentUnresolvedCount: 3,
        blockSize: 4,
        currentMaskFraction: 0.75,
        isBlockComplete: false,
        isCommittedToContext: false,
      });
      expect(conf0View.timestepStatus.nextEligibleStep).toEqual({
        nextTModel: 0.75,
        nextNMasked: 3,
        blockSize: 4,
        remainingSteps: 3,
      });

      const conf0Macro = buildTwoTowerMacroScene(conf0);
      const conf0MacroDiag = conf0Macro.diagnostics as unknown as TwoTowerMacroDiagnostics;
      // Must NEVER pair executed t=1.00 with post-commit 3/4!
      expect(conf0MacroDiag.timestepStatusLabel).not.toContain('t = 1.00 (3/4');
      expect(conf0MacroDiag.timestepStatusLabel).toContain('4/4 input masked');
      expect(conf0MacroDiag.timestepStatusLabel).toContain('Block now 3/4 masked');
      expect(conf0MacroDiag.timestepStatusLabel).toContain('Next t = 0.75');

      // 3. Complete block to `completed_uncommitted` (last refinement had 1/4 input -> 0/4 post-commit)
      const uncomm = completeBlockToUncommitted(base);
      const uncommView = resolveTwoTowerSessionViewState(uncomm);
      expect(uncommView.timestepStatus.blockCompletionState).toBe('completed_uncommitted');
      expect(uncommView.timestepStatus.currentBlockMask).toEqual({
        currentUnresolvedCount: 0,
        blockSize: 4,
        currentMaskFraction: 0,
        isBlockComplete: true,
        isCommittedToContext: false,
      });
      expect(uncommView.timestepStatus.nextEligibleStep).toBeNull();
      expect(uncommView.timestepStatus.blockCompletionNote).toMatch(/no further denoiser evaluation is required/i);
      expect(uncommView.timestepStatus.executedPass!.nMaskedInput).toBe(1);
      expect(uncommView.timestepStatus.executedPass!.tModel).toBe(0.25);

      const uncommMacro = buildTwoTowerMacroScene(uncomm);
      const uncommMacroDiag = uncommMacro.diagnostics as unknown as TwoTowerMacroDiagnostics;
      expect(uncommMacroDiag.timestepStatusLabel).not.toContain('t = 0.25 (0/4');
      expect(uncommMacroDiag.timestepStatusLabel).toContain('1/4 input masked');
      expect(uncommMacroDiag.timestepStatusLabel).toContain('no further pass needed');

      // 4. Commit block to Context -> `committed_to_context`
      const committed = commitTwoTowerActiveBlockToContext(uncomm);
      const commitView = resolveTwoTowerSessionViewState(committed);
      expect(commitView.timestepStatus.blockCompletionState).toBe('committed_to_context');
      expect(commitView.timestepStatus.currentBlockMask?.isCommittedToContext).toBe(true);
      expect(commitView.timestepStatus.nextEligibleStep).toBeNull();
      expect(commitView.timestepStatus.conditioningMode).toBe('historical_prior_pass');
      expect(commitView.timestepStatus.blockCompletionNote).toMatch(/no further denoiser evaluation is required/i);
    });

    it('contains zero r/K timestep definitions across all owned scene files and TwoTower docs', () => {
      const filesToAudit = [
        'src/twotower/scene/provenance.ts',
        'src/twotower/scene/dualMacroScene.ts',
        'src/twotower/scene/timestepCutaway.ts',
        'src/twotower/scene/pairedGqaCutaway.ts',
        'src/twotower/scene/twoTowerGqaAtlas.ts',
        'src/twotower/scene/pairedMambaCutaway.ts',
        'src/twotower/scene/mambaDecomposition.ts',
        'src/twotower/scene/twoTowerMoeCutaway.ts',
        'src/twotower/scene/twoTowerBookendCutaways.ts',
        'docs/TWOTOWER_ARCHITECTURE.md',
        'docs/VERIFICATION.md',
      ];
      for (const rel of filesToAudit) {
        const text = readFileSync(resolve(process.cwd(), rel), 'utf8');
        expect(text, `${rel} must never define t = r / K`).not.toMatch(/\bt\s*=\s*r\s*\/\s*K\b/);
      }
    });
  });

  describe('2. Historical Cache Boundary Provenance Across All 6 Spatial Scene Builders', () => {
    it('enforces explicit historical provenance and mismatch warnings after block commit (P=8 -> 12) across all 6 scene builders', () => {
      const base = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        eosTokenIds: [],
      });
      const uncomm = completeBlockToUncommitted(base);
      const lastRefRec = getActiveRecord(uncomm);
      const p8Digest = lastRefRec.contextSnapshot.ctxState.stateDigest;
      const committed = commitTwoTowerActiveBlockToContext(uncomm); // P = 8 -> 12!

      const view = resolveTwoTowerSessionViewState(committed);
      expect(view.committedPrefixLen).toBe(12);
      expect(view.contextSnapshot.ctxState.stateDigest).not.toBe(p8Digest);

      // Authoritative TwoTowerDenoiserPassLookup provenance
      const dl = view.denoiserLookup;
      expect(dl.status).toBe('historical_prior_record');
      expect(dl.isHistoricalInspection).toBe(true);
      expect(dl.matchesCurrentContextBoundary).toBe(false);
      expect(dl.sourceContextPrefixLen).toBe(8);
      expect(dl.sourceContextCacheDigest).toBe(p8Digest);
      expect(dl.sourceRecordConfig).toEqual(lastRefRec.configAtStep);
      expect(dl.sourceBlockId).toBe(0);
      expect(dl.historicalProvenanceWarning).not.toBeNull();
      expect(dl.historicalProvenanceWarning!).toContain('P=8');
      expect(dl.historicalProvenanceWarning!).toContain('P=12');
      expect(dl.historicalProvenanceWarning!).toContain('does NOT belong to the current Context boundary');

      // 1. Macro scene
      const macro = buildTwoTowerMacroScene(committed);
      const macroDiag = macro.diagnostics as unknown as TwoTowerMacroDiagnostics;
      expect(macroDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(macroDiag.denoiserMatchesCurrentContextBoundary).toBe(false);
      expect(macroDiag.denoiserSourceContextPrefixLen).toBe(8);
      expect(macroDiag.denoiserSourceContextCacheDigest).toBe(p8Digest);
      expect(macro.warningBanner).toContain('P=8');
      expect(macro.provenanceBadges.some((b) => b.tone === 'warn' && b.label.includes('Historical Denoiser Pass'))).toBe(true);
      expect(macro.partCatalog['den-gqa-body-06']?.specs.join(' ')).toContain('KV[0..7]');
      expect(macro.partCatalog['den-mamba-body-01']?.specs.join(' ')).toContain('H_7');

      // 2. Paired GQA cutaway: NEVER relabel prior KV[0..7] as P=12!
      const gqa = buildTwoTowerPairedGqaScene(committed, {
        bayId: 6,
        selectedCell: {
          tower: 'denoiser',
          cell: { tensor: 'scores', headIndex: 0, kvHeadIndex: 0, row: 0, col: 2 },
        },
      });
      const gqaDiag = gqa.diagnostics as unknown as PairedGqaDiagnostics;
      expect(gqaDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(gqaDiag.matchesCurrentContextBoundary).toBe(false);
      expect(gqaDiag.currentContextPrefixLen).toBe(12);
      expect(gqaDiag.denoiserConditioningPrefixLen).toBe(8);
      expect(gqaDiag.contextPrefixKvLen).toBe(8); // actual pass conditioning length, NOT 12!
      expect(gqaDiag.denoiserSourceContextCacheDigest).toBe(p8Digest);
      expect(gqa.warningBanner).toContain('P=8');
      expect(gqa.warningBanner).toContain('P=12');
      expect(gqa.landmarks.find((l) => l.id === 'den-kv-bank-landmark')?.label).toContain('KV[0..7]');
      expect(gqa.landmarks.find((l) => l.id === 'den-kv-bank-landmark')?.label).toContain('P_cond=8 ≠ P=12');
      expect(gqa.cellPopover?.coordLabel).toContain('Historical P_cond=8 ≠ P=12');

      // 3. Paired Mamba cutaway: NEVER relabel prior H_7 as H_11 / current P=12!
      const mamba = buildTwoTowerPairedMambaScene(committed, {
        bayId: 1,
        activeHead: 2,
        localTokenIndex: 0,
      });
      const mambaDiag = mamba.diagnostics as unknown as PairedMambaDiagnostics;
      expect(mambaDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(mambaDiag.matchesCurrentContextBoundary).toBe(false);
      expect(mambaDiag.currentContextPrefixLen).toBe(12);
      expect(mambaDiag.denoiserConditioningPrefixLen).toBe(8);
      expect(mambaDiag.sourceBoundaryDigest).toBe(p8Digest);
      expect(mambaDiag.currentContextStateDigest).toBe(view.contextSnapshot.ctxState.stateDigest);
      expect(mamba.warningBanner).toContain('P=8');
      expect(mamba.warningBanner).toContain('P=12');
      expect(mamba.landmarks.find((l) => l.id === 'ctx-ssm-channel-landmark')?.label).toContain('H_7');
      expect(mamba.landmarks.find((l) => l.id === 'ctx-ssm-channel-landmark')?.label).toContain('P_cond=8 ≠ P=12');
      expect(mamba.activeCellGrids?.some((g) => g.colDimLabel?.includes('Historical P_cond=8'))).toBe(true);

      // 4. Timestep cutaway
      const time = buildTwoTowerTimestepScene(committed, { bayId: 6, stepperStep: 0 });
      const timeDiag = time.diagnostics as unknown as TimestepCutawayDiagnostics;
      expect(timeDiag.conditioningMode).toBe('historical_prior_pass');
      expect(timeDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(timeDiag.matchesCurrentContextBoundary).toBe(false);
      expect(timeDiag.sourceContextPrefixLen).toBe(8);
      expect(timeDiag.sourceContextCacheDigest).toBe(p8Digest);
      expect(time.warningBanner).toContain('Historical Denoiser pass inspection');

      // 5. MoE cutaway (Denoiser)
      const moe = buildTwoTowerMoeScene(committed, { bayId: 2, towerRole: 'denoiser' });
      const moeDiag = moe.diagnostics as unknown as TwoTowerMoeDiagnostics;
      expect(moeDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(moeDiag.matchesCurrentContextBoundary).toBe(false);
      expect(moeDiag.denoiserSourceContextPrefixLen).toBe(8);
      expect(moe.warningBanner).toContain('Historical Denoiser pass inspection');
      expect(moe.provenanceBadges.some((b) => b.tone === 'warn' && b.label.includes('Historical Denoiser Pass'))).toBe(true);

      // 6. Bookend cutaways (Embedding & Output, Denoiser)
      const emb = buildTwoTowerEmbeddingScene(committed, { towerRole: 'denoiser' });
      const embDiag = emb.diagnostics as unknown as TwoTowerEmbeddingDiagnostics;
      expect(embDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(embDiag.matchesCurrentContextBoundary).toBe(false);
      expect(embDiag.denoiserSourceContextPrefixLen).toBe(8);
      expect(emb.warningBanner).toContain('Historical Denoiser pass inspection');

      const out = buildTwoTowerOutputScene(committed, { towerRole: 'denoiser' });
      const outDiag = out.diagnostics as unknown as TwoTowerOutputDiagnostics;
      expect(outDiag.isHistoricalDenoiserInspection).toBe(true);
      expect(outDiag.matchesCurrentContextBoundary).toBe(false);
      expect(outDiag.denoiserSourceContextPrefixLen).toBe(8);
      expect(out.warningBanner).toContain('Historical Denoiser pass inspection');
    });

    it('enforces historical mismatch after mode switch to AR (P=12 -> 13) AND after initializing Block 1 (P=12 before pass 0)', () => {
      const base = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        eosTokenIds: [],
      });
      const committed = commitTwoTowerActiveBlockToContext(completeBlockToUncommitted(base)); // P = 12

      // Scenario B: Switch to AR mode and take an AR step (P = 12 -> 13)
      const arAfterDiff = stepTwoTowerSessionAr(
        updateTwoTowerSessionConfig(committed, { mode: 'ar' }),
      );
      const arView = resolveTwoTowerSessionViewState(arAfterDiff);
      expect(arView.modeAtRecord).toBe('ar');
      expect(arView.committedPrefixLen).toBe(13);
      expect(arView.timestepStatus.conditioningMode).toBe('bypassed_ar');
      expect(arView.denoiserLookup.isHistoricalInspection).toBe(true);
      expect(arView.denoiserLookup.matchesCurrentContextBoundary).toBe(false);
      expect(arView.denoiserLookup.sourceContextPrefixLen).toBe(8);
      expect(arView.denoiserLookup.sourceRecordConfig?.mode).toBe('mask_diffusion');

      const arGqa = buildTwoTowerPairedGqaScene(arAfterDiff, { bayId: 6 });
      expect((arGqa.diagnostics as unknown as PairedGqaDiagnostics).contextPrefixKvLen).toBe(8);
      expect((arGqa.diagnostics as unknown as PairedGqaDiagnostics).currentContextPrefixLen).toBe(13);
      expect(arGqa.warningBanner).toContain('P=8');
      expect(arGqa.warningBanner).toContain('P=13');

      const arMamba = buildTwoTowerPairedMambaScene(arAfterDiff, { bayId: 1 });
      expect((arMamba.diagnostics as unknown as PairedMambaDiagnostics).denoiserConditioningPrefixLen).toBe(8);
      expect((arMamba.diagnostics as unknown as PairedMambaDiagnostics).currentContextPrefixLen).toBe(13);
      expect(arMamba.warningBanner).toContain('P=8');
      expect(arMamba.warningBanner).toContain('P=13');

      // Scenario C: Initialize Block 1 at P = 12 before Block 1 pass 0 executes
      const block1Init = initTwoTowerDiffusionBlock(committed);
      const b1View = resolveTwoTowerSessionViewState(block1Init);
      expect(b1View.committedPrefixLen).toBe(12);
      expect(b1View.activeBlock?.blockId).toBe(1);
      expect(b1View.denoiserLookup.isHistoricalInspection).toBe(true);
      expect(b1View.denoiserLookup.matchesCurrentContextBoundary).toBe(false);
      expect(b1View.denoiserLookup.sourceBlockId).toBe(0);
      expect(b1View.denoiserLookup.sourceContextPrefixLen).toBe(8);
      expect(b1View.denoiserLookup.historicalProvenanceWarning).toContain('Block b1 has not been denoised yet');
      expect(b1View.timestepStatus.conditioningMode).toBe('historical_prior_pass');
      expect(b1View.timestepStatus.executedPass?.sourceBlockId).toBe(0);
      expect(b1View.timestepStatus.nextEligibleStep).toEqual({
        nextTModel: 1.0,
        nextNMasked: 4,
        blockSize: 4,
        remainingSteps: 4,
      });

      const b1Gqa = buildTwoTowerPairedGqaScene(block1Init, { bayId: 6 });
      const b1Mamba = buildTwoTowerPairedMambaScene(block1Init, { bayId: 1 });
      const b1Time = buildTwoTowerTimestepScene(block1Init, { bayId: 6 });
      expect((b1Gqa.diagnostics as unknown as PairedGqaDiagnostics).matchesCurrentContextBoundary).toBe(false);
      expect((b1Mamba.diagnostics as unknown as PairedMambaDiagnostics).matchesCurrentContextBoundary).toBe(false);
      expect((b1Time.diagnostics as unknown as TimestepCutawayDiagnostics).matchesCurrentContextBoundary).toBe(false);
      expect(b1Gqa.warningBanner).toContain('Block b1 has not been denoised yet');
      expect(b1Mamba.warningBanner).toContain('Block b1 has not been denoised yet');
      expect(b1Time.warningBanner).toContain('Block b1 has not been denoised yet');
    });

    it('preserves explicit pruned_unavailable warnings without synthetic traces and strictly preserves historicalConfig over pendingConfig', () => {
      let pruned = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        maxRecords: 2,
        eosTokenIds: [],
      });
      pruned = initTwoTowerDiffusionBlock(pruned);
      pruned = runTwoTowerDiffusionRefinement(pruned);
      pruned = commitTwoTowerRefinementConfidence(pruned);
      expect(getLatestContextExecution(pruned)).toBeNull();

      // Mutate pendingConfig drastically and confirm historicalConfig stays untouched
      const withPending = updateTwoTowerSessionConfig(pruned, {
        mode: 'ar',
        blockSize: 8,
        stepsPerBlock: 2,
        confidenceThreshold: 0.22,
      });
      const rewound = navigateTwoTowerSession(withPending, 0);
      const view = resolveTwoTowerSessionViewState(rewound);
      expect(view.historicalConfig.mode).toBe('mask_diffusion');
      expect(view.historicalConfig.blockSize).toBe(4);
      expect(view.denoiserLookup.sourceRecordConfig?.mode).toBe('mask_diffusion');
      expect(view.denoiserLookup.sourceRecordConfig?.blockSize).toBe(4);

      const gqa = buildTwoTowerPairedGqaScene(withPending, { bayId: 6 });
      const mamba = buildTwoTowerPairedMambaScene(withPending, { bayId: 1 });
      const moe = buildTwoTowerMoeScene(withPending, { bayId: 2, towerRole: 'context' });
      const emb = buildTwoTowerEmbeddingScene(withPending, { towerRole: 'context' });
      const out = buildTwoTowerOutputScene(withPending, { towerRole: 'context' });
      expect(gqa.warningBanner).toMatch(/prun/i);
      expect(mamba.warningBanner).toMatch(/prun/i);
      expect(moe.warningBanner).toMatch(/prun/i);
      expect(emb.warningBanner).toMatch(/prun/i);
      expect(out.warningBanner).toMatch(/prun/i);
    });
  });

  describe('3. Dimensional Consistency (Complete Miniature Inventory vs Selected Atlas Slices) & Headless Purity', () => {
    it('matches exact miniature GQA (32Q/2KV, dHead=4) and Mamba-2 (16h/8g, 4×4 SSM) inventory and distinguishes selected atlas slices', () => {
      // Direct check against authoritative math simulator configs
      expect(TWOTOWER_MINI_MODEL_INVENTORY.d).toBe(TWOTOWER_MINI_CONFIG.d);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.queryHeads).toBe(32);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.queryHeads).toBe(GQA_SIM_CONFIG.nQ);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.kvHeads).toBe(2);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.kvHeads).toBe(GQA_SIM_CONFIG.nKv);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.headDim).toBe(4);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.headDim).toBe(GQA_SIM_CONFIG.dHead);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.gqa.atlasSelectedScoreHeadCount).toBe(2);

      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.heads).toBe(16);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.heads).toBe(MAMBA_SIM_CONFIG.nHeads);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.groups).toBe(8);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.groups).toBe(MAMBA_SIM_CONFIG.nGroups);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.dHead).toBe(4);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.dState).toBe(4);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.ssmStateMatrixShape).toEqual([4, 4]);
      expect(TWOTOWER_MINI_MODEL_INVENTORY.mamba.atlasSelectedDecompHeadCount).toBe(1);

      const s = runTwoTowerDiffusionRefinement(
        initTwoTowerDiffusionBlock(
          createTwoTowerSession(sharedModel, {
            mode: 'mask_diffusion',
            blockSize: 4,
            stepsPerBlock: 4,
            eosTokenIds: [],
          }),
        ),
      );
      const view = resolveTwoTowerSessionViewState(s);
      const denGqa = view.resolveDenoiserBayTrace(6);
      const denMamba = view.resolveDenoiserBayTrace(1);
      if (denGqa?.kind !== '*' || denMamba?.kind !== 'M') throw new Error('expected GQA & Mamba traces');

      // GQA atlas & cutaway distinguish 32 logical Q heads / 2 KV heads / dHead=4 from 2 selected score head slices
      const gqaAtlas = buildTwoTowerGqaValueAtlas({
        primarySim: denGqa.sim,
        group0Head: 3,
        group1Head: 20,
      });
      expect(gqaAtlas.logicalQueryHeadCount).toBe(32);
      expect(gqaAtlas.logicalKvHeadCount).toBe(2);
      expect(gqaAtlas.headDim).toBe(4);
      expect(gqaAtlas.group0Head).toBe(3);
      expect(gqaAtlas.group1Head).toBe(20);

      const gqaScene = buildTwoTowerPairedGqaScene(s, { bayId: 6, group0Head: 3, group1Head: 20 });
      const gqaDiag = gqaScene.diagnostics as unknown as PairedGqaDiagnostics;
      expect(gqaDiag.modelGqaInventory).toEqual({
        queryHeads: 32,
        kvHeads: 2,
        queriesPerKvGroup: 16,
        headDim: 4,
        displayedScoreHeads: [3, 20],
        displayedScoreHeadCount: 2,
      });
      expect(gqaScene.epistemicNote).toContain('32 miniature GQA query heads / 2 KV heads, head width 4');
      expect(gqaScene.epistemicNote).toContain('2 selected Q-head slices');

      // Mamba decomposition & cutaway distinguish 16 Mamba heads / 8 groups / 4×4 SSM from 1 selected head slice
      const mambaDec = extractMambaFullBlockPositionState(denMamba.sim, 5, 0, { bayId: 1 });
      expect(mambaDec.modelHeadCount).toBe(16);
      expect(mambaDec.modelGroupCount).toBe(8);
      expect(mambaDec.ssmStateMatrixShape).toEqual([4, 4]);

      const mambaScene = buildTwoTowerPairedMambaScene(s, { bayId: 1, activeHead: 5, localTokenIndex: 0 });
      const mambaDiag = mambaScene.diagnostics as unknown as PairedMambaDiagnostics;
      expect(mambaDiag.modelMambaInventory).toEqual({
        heads: 16,
        groups: 8,
        headsPerGroup: 2,
        dHead: 4,
        dState: 4,
        stateMatrixShape: [4, 4],
        displayedHeadIndex: 5,
        displayedGroupIndex: 2,
        atlasSelectedDecompHeadCount: 1,
      });
      expect(mambaScene.epistemicNote).toContain('16 heads, 8 groups, 4×4 per-head FP32 SSM state');
      expect(mambaScene.epistemicNote).toContain("1 selected head's 4×4");

      // Macro scene exposes exact inventory
      const macroScene = buildTwoTowerMacroScene(s);
      const macroDiag = macroScene.diagnostics as unknown as TwoTowerMacroDiagnostics;
      expect(macroDiag.modelDimensionalInventory).toBe(TWOTOWER_MINI_MODEL_INVENTORY);
      expect(macroScene.epistemicNote).toContain('32 GQA query heads / 2 KV heads');
      expect(macroScene.epistemicNote).toContain('16 Mamba heads / 8 groups');

      // Docs document exact miniature inventory vs selected slices & TwoTower trace origins
      const archDoc = readFileSync(resolve(process.cwd(), 'docs/TWOTOWER_ARCHITECTURE.md'), 'utf8');
      expect(archDoc).toContain('32 miniature GQA query heads');
      expect(archDoc).toContain('16 miniature Mamba heads');
      expect(archDoc).toContain('8 Mamba groups');
      expect(archDoc).toContain('context_computed');
      expect(archDoc).toContain('context_reconstructed');
      expect(archDoc).not.toContain('computed_this_step');
      expect(archDoc).not.toContain('reconstructed_from_checkpoint');
    });

    it('guarantees zero DOM, WebGL2, or Math.random() across all 10 src/twotower/scene/*.ts modules', () => {
      const sceneFiles = [
        'src/twotower/scene/provenance.ts',
        'src/twotower/scene/dualMacroScene.ts',
        'src/twotower/scene/timestepCutaway.ts',
        'src/twotower/scene/pairedGqaCutaway.ts',
        'src/twotower/scene/twoTowerGqaAtlas.ts',
        'src/twotower/scene/pairedMambaCutaway.ts',
        'src/twotower/scene/mambaDecomposition.ts',
        'src/twotower/scene/twoTowerMoeCutaway.ts',
        'src/twotower/scene/twoTowerBookendCutaways.ts',
        'src/twotower/scene/index.ts',
      ];
      const forbidden: readonly [string, RegExp][] = [
        ['Math.random', /\bMath\.random\b/],
        ['document.*', /\bdocument\.[A-Za-z_]/],
        ['window.*', /\bwindow\.[A-Za-z_]/],
        ['WebGL2RenderingContext', /\bWebGL2RenderingContext\b/],
        ['HTMLCanvasElement', /\bHTMLCanvasElement\b/],
        ['HTMLElement', /\bHTMLElement\b/],
        ['SVGSVGElement', /\bSVGSVGElement\b/],
        ['runtime ui import', /^\s*import\s+(?!type\b)[^'"]*\/ui\//m],
        ['render device import', /from\s+['"][^'"]*render\/device['"]/],
      ];
      for (const rel of sceneFiles) {
        const raw = readFileSync(resolve(process.cwd(), rel), 'utf8');
        const codeOnly = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
        for (const [label, re] of forbidden) {
          expect(codeOnly, `${rel} must not contain ${label}`).not.toMatch(re);
        }
      }
    });
  });
});
