import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  aabbCorners,
  buildCameraSnapshot,
  makeViewportMetrics,
  projectWorldToClientCss,
  projectWorldToInnerCss,
} from '../src/render/camera';
import { LINE_INSTANCE_FLOATS } from '../src/render/linePacking';
import { clientPointToRay, pickMacroBay, pointToRayDistance } from '../src/render/pick';
import {
  bayCenterY,
  RIBBON_INSTANCE_FLOATS,
  SOLID_INSTANCE_FLOATS,
  SPINE_REPLAY_DISCLAIMER,
} from '../src/scene/spine';
import {
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  forkTwoTowerSessionBranch,
  getActiveRecord,
  getLatestContextExecution,
  initTwoTowerDiffusionBlock,
  navigateTwoTowerSession,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  switchTwoTowerSessionBranch,
  TWOTOWER_SPEC,
  updateTwoTowerSessionConfig,
  validateContextSnapshot,
  type TwoTowerSession,
} from '../src/twotower';
import {
  BRIDGE_LEFT_X,
  BRIDGE_RIGHT_X,
  buildTwoTowerMacroScene,
  contextBayFrontWorld,
  contextMambaConvFrontWorld,
  contextMambaSsmFrontWorld,
  denoiserBayFrontWorld,
  fitTwoTowerMacroCamera,
  MACRO_FLOW_VERTEX_FLOATS,
  MAMBA_CONV_LANE_DY,
  MAMBA_SSM_LANE_DY,
  TIMESTEP_HUB_Y,
  TIMESTEP_RAIL_X,
  TWO_TOWER_MACRO_FIT_BOUNDS,
  type TwoTowerMacroDiagnostics,
} from '../src/twotower/scene/dualMacroScene';
import {
  resolvePickHitTowerIdentity,
  resolveTwoTowerSessionViewState,
} from '../src/twotower/scene/provenance';

function getDiag(scene: ReturnType<typeof buildTwoTowerMacroScene>): TwoTowerMacroDiagnostics {
  expect(scene.diagnostics).toBeDefined();
  return scene.diagnostics as TwoTowerMacroDiagnostics;
}

/** Extract all line segments from packed `lineData` matching an exact horizontal bridge channel. */
function extractHorizontalLineMetrics(
  lineData: Float32Array,
  lineCount: number,
  expectedX0: number,
  expectedX1: number,
  expectedY: number,
  expectedZ: number,
): readonly { readonly cssWidth: number; readonly style: number; readonly alpha: number }[] {
  const eps = 1e-3;
  const out: { cssWidth: number; style: number; alpha: number }[] = [];
  for (let i = 0; i < lineCount; i++) {
    const o = i * LINE_INSTANCE_FLOATS;
    const ax = lineData[o + 0];
    const ay = lineData[o + 1];
    const az = lineData[o + 2];
    const bx = lineData[o + 3];
    const by = lineData[o + 4];
    const bz = lineData[o + 5];
    const minX = Math.min(ax, bx);
    const maxX = Math.max(ax, bx);
    if (
      Math.abs(minX - expectedX0) < eps &&
      Math.abs(maxX - expectedX1) < eps &&
      Math.abs(ay - expectedY) < eps &&
      Math.abs(by - expectedY) < eps &&
      Math.abs(az - expectedZ) < eps &&
      Math.abs(bz - expectedZ) < eps
    ) {
      out.push({
        cssWidth: lineData[o + 6],
        style: lineData[o + 7],
        alpha: lineData[o + 8],
      });
    }
  }
  return out;
}

/** Advance a diffusion session to `completed_uncommitted` (and return every intermediate snapshot). */
function stepDiffusionBlockToCompletedUncommitted(baseSession: TwoTowerSession): {
  readonly afterInit: TwoTowerSession;
  readonly afterFirstRefinement: TwoTowerSession;
  readonly afterFirstConfidence: TwoTowerSession;
  readonly atCompletedUncommitted: TwoTowerSession;
  readonly afterBlockCommit: TwoTowerSession;
} {
  const afterInit = initTwoTowerDiffusionBlock(baseSession);
  const afterFirstRefinement = runTwoTowerDiffusionRefinement(afterInit);
  const afterFirstConfidence = commitTwoTowerRefinementConfidence(afterFirstRefinement);

  let s = afterFirstConfidence;
  while (getActiveRecord(s).lifecyclePhase !== 'completed_uncommitted') {
    s = runTwoTowerDiffusionRefinement(s);
    s = commitTwoTowerRefinementConfidence(s);
  }
  const atCompletedUncommitted = s;
  const afterBlockCommit = commitTwoTowerActiveBlockToContext(atCompletedUncommitted);

  return {
    afterInit,
    afterFirstRefinement,
    afterFirstConfidence,
    atCompletedUncommitted,
    afterBlockCommit,
  };
}

describe('Stage TT3 Wave 1: Dual-Tower 52×2 Macro Spatial Composition & Provenance', () => {
  const sharedModel = createDefaultTwoTowerModel();

  describe('1. Dual-Tower Topology, Exact Alignment & Persistent vs. Ephemeral Visual Language', () => {
    it('builds 52 Context bays + 52 Denoiser bays at identical bayCenterY, distinct models, 6 GQA + 23 Conv + 23 SSM + 0 MoE linkers', () => {
      const session = initTwoTowerDiffusionBlock(
        createTwoTowerSession(sharedModel, {
          mode: 'mask_diffusion',
          confidenceThreshold: 0.9999,
          eosTokenIds: [],
        }),
      );
      const scene = buildTwoTowerMacroScene(session);
      const diag = getDiag(scene);

      // Buffer stride integrity (exact Nano SolidsPass, LinesPass, RibbonsPass, FlatFlowRibbonsPass layouts)
      expect(scene.viewMode).toBe('macro');
      expect(scene.solidData.length).toBe(scene.solidCount * SOLID_INSTANCE_FLOATS);
      expect(scene.lineData.length).toBe(scene.lineCount * LINE_INSTANCE_FLOATS);
      expect(scene.ribbonData.length).toBe(scene.ribbonCount * RIBBON_INSTANCE_FLOATS);
      expect(scene.flowVertices).toBeDefined();
      expect(scene.flowIndices).toBeDefined();
      expect(scene.flowVertices!.length % MACRO_FLOW_VERTEX_FLOATS).toBe(0);
      expect(scene.flowIndices!.length % 3).toBe(0);

      for (let i = 0; i < scene.solidData.length; i++) {
        expect(Number.isFinite(scene.solidData[i])).toBe(true);
      }
      for (let i = 0; i < scene.lineData.length; i++) {
        expect(Number.isFinite(scene.lineData[i])).toBe(true);
      }
      for (let i = 0; i < scene.ribbonData.length; i++) {
        expect(Number.isFinite(scene.ribbonData[i])).toBe(true);
      }
      const vertexCount = scene.flowVertices!.length / MACRO_FLOW_VERTEX_FLOATS;
      for (let i = 0; i < scene.flowIndices!.length; i++) {
        expect(scene.flowIndices![i]).toBeLessThan(vertexCount);
      }

      // Exact diagnostic counts & distinct tower parameter identities
      expect(diag.contextBayCount).toBe(52);
      expect(diag.denoiserBayCount).toBe(52);
      expect(diag.alignedGqaLinkerCount).toBe(6);
      expect(diag.alignedMambaConvLinkerCount).toBe(23);
      expect(diag.alignedMambaSsmLinkerCount).toBe(23);
      expect(diag.moeLinkerCount).toBe(0);
      expect(diag.timestepTapCount).toBe(52);
      expect(diag.timestepPreNormTapCount).toBe(29);
      expect(diag.timestepPostNormTapCount).toBe(23);
      expect(diag.contextModelId).not.toBe(diag.denoiserModelId);
      expect(diag.contextParameterFingerprint).not.toBe(diag.denoiserParameterFingerprint);
      expect(diag.snapshotIntegrityValid).toBe(true);
      expect(diag.contextTraceStatus).toBe('available');
      expect(scene.warningBanner).toBeNull();

      // Verify exact 52 Context bays and 52 Denoiser bays vertically aligned at identical bayCenterY(1..52)
      const ctxFasciaByBay = new Map<number, (typeof scene.fasciaPickBoxes)[number]>();
      const denFasciaByBay = new Map<number, (typeof scene.fasciaPickBoxes)[number]>();
      for (const box of scene.fasciaPickBoxes) {
        if (box.towerRole === 'context') ctxFasciaByBay.set(box.bayId, box);
        if (box.towerRole === 'denoiser') denFasciaByBay.set(box.bayId, box);
      }
      expect(ctxFasciaByBay.size).toBe(52);
      expect(denFasciaByBay.size).toBe(52);
      for (let bayId = 1; bayId <= 52; bayId++) {
        const expectedY = bayCenterY(bayId);
        const cf = ctxFasciaByBay.get(bayId)!;
        const df = denFasciaByBay.get(bayId)!;
        expect(0.5 * (cf.min[1] + cf.max[1])).toBeCloseTo(expectedY, 6);
        expect(0.5 * (df.min[1] + df.max[1])).toBeCloseTo(expectedY, 6);
        expect(cf.max[0]).toBeLessThan(BRIDGE_LEFT_X);
        expect(df.min[0]).toBeGreaterThan(BRIDGE_RIGHT_X);
      }

      // Verify exact cross-tower linker PickBoxes: 6 GQA, 23 Mamba Conv1D, 23 Mamba SSM, 0 MoE
      const gqaLinkers = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('linker-gqa-kv-bay-'));
      const convLinkers = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('linker-mamba-conv-bay-'));
      const ssmLinkers = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('linker-mamba-ssm-bay-'));
      expect(gqaLinkers.map((b) => b.bayId)).toEqual([...TWOTOWER_SPEC.layerAlignment.gqaTransferBayIds]);
      expect(convLinkers.map((b) => b.bayId)).toEqual([...TWOTOWER_SPEC.layerAlignment.mambaTransferBayIds]);
      expect(ssmLinkers.map((b) => b.bayId)).toEqual([...TWOTOWER_SPEC.layerAlignment.mambaTransferBayIds]);

      // Zero cross-tower linkers on any MoE bay
      const moeBaySet = new Set(TWOTOWER_SPEC.layerAlignment.statelessMoeBayIds);
      const anyMoeLinker = scene.solidPickBoxes.filter(
        (b) => b.partId?.startsWith('linker-') && moeBaySet.has(b.bayId),
      );
      expect(anyMoeLinker).toHaveLength(0);

      // Verify Context dual-compartment Mamba slabs (`ctx-conv-bay-XX` above `ctx-ssm-bay-XX`) vs Denoiser ephemeral scratchpads
      const ctxConvSlabs = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('ctx-conv-bay-'));
      const ctxSsmSlabs = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('ctx-ssm-bay-'));
      const denConvScratch = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('den-scratch-conv-bay-'));
      const denSsmScratch = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('den-scratch-ssm-bay-'));
      expect(ctxConvSlabs).toHaveLength(23);
      expect(ctxSsmSlabs).toHaveLength(23);
      expect(denConvScratch).toHaveLength(23);
      expect(denSsmScratch).toHaveLength(23);
      for (let i = 0; i < 23; i++) {
        expect(ctxConvSlabs[i].traceOrigin).toBe('persistent_cache');
        expect(ctxSsmSlabs[i].traceOrigin).toBe('persistent_cache');
        expect(ctxConvSlabs[i].min[1]).toBeGreaterThan(ctxSsmSlabs[i].max[1]); // cleanly separated compartments!
        expect(denConvScratch[i].traceOrigin).toBe('denoiser_scratch');
        expect(denSsmScratch[i].traceOrigin).toBe('denoiser_scratch');
      }

      // Context GQA persistent KV slices (P=8 × 4 rails × 6 bays = 192) + 24-slot capacity rails vs Denoiser ephemeral KV (B=4 × 2 × 6 = 48)
      const ctxKvSlices = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('ctx-kv-slice-bay-'));
      const ctxKvCapacityRails = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('ctx-kv-capacity-bay-'));
      const denScratchKv = scene.solidPickBoxes.filter((b) => b.partId?.startsWith('den-scratch-kv-bay-'));
      expect(ctxKvSlices).toHaveLength(6 * 4 * 8);
      expect(ctxKvCapacityRails).toHaveLength(6 * 4);
      expect(denScratchKv).toHaveLength(6 * 2 * 4);
      expect(scene.kvDepthTicks.map((t) => t.token)).toEqual([1, 8, 24]);

      // Logical read-only KV linker copy check (never mandatory physical inter-device copy)
      expect(scene.partCatalog['linker-gqa-kv-bay-06']?.summary).toMatch(/logical read-only/i);
      expect(scene.partCatalog['linker-gqa-kv-bay-06']?.summary).toMatch(/not a mandatory physical/i);
      expect(scene.partCatalog['ctx-moe-body-02']?.badge).toMatch(/0 B Stateless/i);
      expect(scene.partCatalog['den-moe-body-02']?.summary).toMatch(/post-norm/i);
    });
  });

  describe('2. Hierarchical Emphasis (Anti-Clutter Verification on Actual GPU Buffers)', () => {
    it('differentiates elevated vs ambient linkers in default overview and elevates selected bay / filtered channel cleanly', () => {
      const session = initTwoTowerDiffusionBlock(
        createTwoTowerSession(sharedModel, {
          mode: 'mask_diffusion',
          confidenceThreshold: 0.9999,
          eosTokenIds: [],
        }),
      );

      // A. Default overview (`activeLinkerChannel: 'all'`, no bay selected): NOT identical prominence!
      const overview = buildTwoTowerMacroScene(session, { activeLinkerChannel: 'all' });
      const ovDiag = getDiag(overview);
      expect(ovDiag.channelEmphasis.gqa_kv.elevatedCount).toBeGreaterThan(0);
      expect(ovDiag.channelEmphasis.gqa_kv.ambientCount).toBeGreaterThan(0);
      expect(ovDiag.channelEmphasis.gqa_kv.maxAlpha).toBeGreaterThanOrEqual(0.78);
      expect(ovDiag.channelEmphasis.gqa_kv.minAlpha).toBeLessThanOrEqual(0.24);
      expect(ovDiag.channelEmphasis.gqa_kv.minCssWidth).toBeCloseTo(0.85, 5);
      expect(ovDiag.channelEmphasis.gqa_kv.maxCssWidth).toBeGreaterThanOrEqual(1.65);

      expect(ovDiag.channelEmphasis.timestep.elevatedCount).toBeGreaterThan(0);
      expect(ovDiag.channelEmphasis.timestep.ambientCount).toBeGreaterThan(40);
      expect(ovDiag.channelEmphasis.timestep.minAlpha).toBeLessThanOrEqual(0.24);
      expect(ovDiag.channelEmphasis.timestep.maxAlpha).toBeGreaterThanOrEqual(0.78);

      // Verify directly on packed Float32 GPU `lineData`: Bay 06 (hero) vs Bay 20 (ambient unselected)
      const bay06Lines = extractHorizontalLineMetrics(
        overview.lineData,
        overview.lineCount,
        BRIDGE_LEFT_X,
        BRIDGE_RIGHT_X,
        bayCenterY(6),
        0.28,
      );
      const bay20Lines = extractHorizontalLineMetrics(
        overview.lineData,
        overview.lineCount,
        BRIDGE_LEFT_X,
        BRIDGE_RIGHT_X,
        bayCenterY(20),
        0.28,
      );
      expect(bay06Lines).toHaveLength(1);
      expect(bay20Lines).toHaveLength(1);
      expect(bay06Lines[0].alpha).toBeGreaterThanOrEqual(Math.fround(0.78));
      expect(bay06Lines[0].cssWidth).toBeGreaterThanOrEqual(Math.fround(1.65));
      expect(bay20Lines[0].alpha).toBeLessThanOrEqual(Math.fround(0.24));
      expect(bay20Lines[0].cssWidth).toBeCloseTo(0.85, 5);

      // B. Selecting Bay 20 (`*` GQA) elevates Bay 20 while Bay 06 recedes to quiet ambient hairline
      const selBay20 = buildTwoTowerMacroScene(session, {
        selectedBayId: 20,
        selectedTower: 'shared',
        activeLinkerChannel: 'all',
      });
      const bay20Selected = extractHorizontalLineMetrics(
        selBay20.lineData,
        selBay20.lineCount,
        BRIDGE_LEFT_X,
        BRIDGE_RIGHT_X,
        bayCenterY(20),
        0.28,
      );
      const bay06When20Selected = extractHorizontalLineMetrics(
        selBay20.lineData,
        selBay20.lineCount,
        BRIDGE_LEFT_X,
        BRIDGE_RIGHT_X,
        bayCenterY(6),
        0.28,
      );
      expect(bay20Selected[0].alpha).toBeGreaterThanOrEqual(Math.fround(0.90));
      expect(bay20Selected[0].cssWidth).toBeGreaterThanOrEqual(Math.fround(1.65));
      expect(bay06When20Selected[0].alpha).toBeLessThanOrEqual(Math.fround(0.24));
      expect(bay06When20Selected[0].cssWidth).toBeCloseTo(0.85, 5);

      // C. Channel filter `gqa_kv`: all 6 GQA elevated (>=0.78, >=1.65), Conv/SSM/Timestep recede (<=0.14)
      const gqaFilter = buildTwoTowerMacroScene(session, { activeLinkerChannel: 'gqa_kv' });
      const gqaDiag = getDiag(gqaFilter);
      expect(gqaDiag.channelEmphasis.gqa_kv.elevatedCount).toBe(6);
      expect(gqaDiag.channelEmphasis.gqa_kv.minAlpha).toBeGreaterThanOrEqual(0.78);
      expect(gqaDiag.channelEmphasis.gqa_kv.flowRibbonCount).toBe(6);
      expect(gqaDiag.channelEmphasis.mamba_conv.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(gqaDiag.channelEmphasis.mamba_conv.flowRibbonCount).toBe(0);
      expect(gqaDiag.channelEmphasis.mamba_ssm.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(gqaDiag.channelEmphasis.timestep.maxAlpha).toBeLessThanOrEqual(0.14);

      // D. Channel filter `mamba_conv`: all 23 Conv elevated, SSM/GQA/Timestep recede (<=0.14)
      const convFilter = buildTwoTowerMacroScene(session, { activeLinkerChannel: 'mamba_conv' });
      const convDiag = getDiag(convFilter);
      expect(convDiag.channelEmphasis.mamba_conv.elevatedCount).toBe(23);
      expect(convDiag.channelEmphasis.mamba_conv.minAlpha).toBeGreaterThanOrEqual(0.78);
      expect(convDiag.channelEmphasis.mamba_ssm.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(convDiag.channelEmphasis.gqa_kv.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(convDiag.channelEmphasis.timestep.maxAlpha).toBeLessThanOrEqual(0.14);

      // Verify directly on packed Float32 GPU lineData at Bay 01 upper Conv lane vs lower SSM lane
      const bay01ConvLine = extractHorizontalLineMetrics(
        convFilter.lineData,
        convFilter.lineCount,
        BRIDGE_LEFT_X,
        BRIDGE_RIGHT_X,
        bayCenterY(1) + MAMBA_CONV_LANE_DY,
        0.14,
      );
      const bay01SsmLine = extractHorizontalLineMetrics(
        convFilter.lineData,
        convFilter.lineCount,
        BRIDGE_LEFT_X,
        BRIDGE_RIGHT_X,
        bayCenterY(1) + MAMBA_SSM_LANE_DY,
        0.14,
      );
      expect(bay01ConvLine[0].alpha).toBeGreaterThanOrEqual(Math.fround(0.78));
      expect(bay01ConvLine[0].cssWidth).toBeGreaterThanOrEqual(Math.fround(1.65));
      expect(bay01SsmLine[0].alpha).toBeLessThanOrEqual(Math.fround(0.14));

      // E. Channel filter `mamba_ssm`: all 23 SSM elevated, Conv/GQA/Timestep recede (<=0.14)
      const ssmFilter = buildTwoTowerMacroScene(session, { activeLinkerChannel: 'mamba_ssm' });
      const ssmDiag = getDiag(ssmFilter);
      expect(ssmDiag.channelEmphasis.mamba_ssm.elevatedCount).toBe(23);
      expect(ssmDiag.channelEmphasis.mamba_ssm.minAlpha).toBeGreaterThanOrEqual(0.78);
      expect(ssmDiag.channelEmphasis.mamba_conv.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(ssmDiag.channelEmphasis.gqa_kv.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(ssmDiag.channelEmphasis.timestep.maxAlpha).toBeLessThanOrEqual(0.14);

      // F. Channel filter `timestep`: all 52 timestep taps elevated in diffusion, cache linkers recede (<=0.14)
      const tsFilter = buildTwoTowerMacroScene(session, { activeLinkerChannel: 'timestep' });
      const tsDiag = getDiag(tsFilter);
      expect(tsDiag.channelEmphasis.timestep.elevatedCount).toBe(52);
      expect(tsDiag.channelEmphasis.timestep.minAlpha).toBeGreaterThanOrEqual(0.78);
      expect(tsDiag.channelEmphasis.gqa_kv.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(tsDiag.channelEmphasis.mamba_conv.maxAlpha).toBeLessThanOrEqual(0.14);
      expect(tsDiag.channelEmphasis.mamba_ssm.maxAlpha).toBeLessThanOrEqual(0.14);
    });
  });

  describe('3. Distinct Mode Transitions (AR, Mock-AR, Mask-Diffusion Full Lifecycle)', () => {
    it('renders AR, Mock-AR (bypassed timestep + P-1 conditioning), and all Mask-Diffusion states distinctly', () => {
      // 1. AR mode (`ar_step`)
      const arBase = createTwoTowerSession(sharedModel, { mode: 'ar', eosTokenIds: [] });
      const arStepped = stepTwoTowerSessionAr(arBase);
      const arScene = buildTwoTowerMacroScene(arStepped);
      const arDiag = getDiag(arScene);

      expect(arDiag.measuredExecutionMode).toBe('ar');
      expect(arDiag.recordKind).toBe('ar_step');
      expect(arDiag.committedPrefixLen).toBe(9);
      expect(arDiag.timestepRailActive).toBe(false);
      expect(arDiag.timestepValue).toBeNull();
      expect(arDiag.timestepStatusLabel).toBe('BYPASSED (t = null)');
      expect(arDiag.refinementLoopState).toBe('dormant');
      expect(arDiag.blockCommitPathwayState).toBe('dormant');
      expect(arDiag.channelEmphasis.gqa_kv.elevatedCount).toBe(0);
      expect(arDiag.channelEmphasis.timestep.elevatedCount).toBe(0);
      expect(arScene.landmarks.some((l) => l.label.includes('BYPASSED (t = null)'))).toBe(true);
      expect(arScene.landmarks.some((l) => l.label.includes('Idle (AR Mode)'))).toBe(true);

      // 2. Mock-AR mode (`mock_ar_step`) — bypassed timestep rail, P-1 conditioning, single-token commit active
      const mockArBase = createTwoTowerSession(sharedModel, { mode: 'mock_ar', eosTokenIds: [] });
      const mockArStepped = stepTwoTowerSessionMockAr(mockArBase);
      const mockArScene = buildTwoTowerMacroScene(mockArStepped);
      const mockArDiag = getDiag(mockArScene);

      expect(mockArDiag.measuredExecutionMode).toBe('mock_ar');
      expect(mockArDiag.recordKind).toBe('mock_ar_step');
      expect(mockArDiag.committedPrefixLen).toBe(9);
      expect(mockArDiag.contextKvConditioningLen).toBe(8); // P - 1 conditioning (KV[:-1] + mambaS2)!
      expect(mockArDiag.timestepRailActive).toBe(false);
      expect(mockArDiag.timestepValue).toBeNull();
      expect(mockArDiag.timestepStatusLabel).toBe('BYPASSED (t = null)');
      expect(mockArDiag.refinementLoopState).toBe('dormant');
      expect(mockArDiag.blockCommitPathwayState).toBe('active_single_token');
      // Even when filtering `timestep` in Mock-AR, timestep rail stays bypassed (not elevated!)
      const mockArTsFiltered = buildTwoTowerMacroScene(mockArStepped, { activeLinkerChannel: 'timestep' });
      expect(getDiag(mockArTsFiltered).timestepRailActive).toBe(false);
      expect(getDiag(mockArTsFiltered).channelEmphasis.timestep.elevatedCount).toBe(0);

      // 3. Mask-Diffusion full lifecycle: init -> refinement -> confidence_commit -> completed_uncommitted -> block_commit
      const diffBase = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999, // force 1 token/step so we hit partial refining + completed_uncommitted cleanly
        eosTokenIds: [],
      });
      const {
        afterInit,
        afterFirstRefinement,
        afterFirstConfidence,
        atCompletedUncommitted,
        afterBlockCommit,
      } = stepDiffusionBlockToCompletedUncommitted(diffBase);

      // 3a. diffusion_block_init
      const initDiag = getDiag(buildTwoTowerMacroScene(afterInit));
      expect(initDiag.measuredExecutionMode).toBe('mask_diffusion');
      expect(initDiag.recordKind).toBe('diffusion_block_init');
      expect(initDiag.lifecyclePhase).toBe('block_initialized');
      expect(initDiag.committedPrefixLen).toBe(8);
      expect(initDiag.activeBlockSize).toBe(4);
      expect(initDiag.activeBlockMaskedCount).toBe(4);
      expect(initDiag.timestepRailActive).toBe(true);
      expect(initDiag.timestepValue).toBeCloseTo(1.0, 6);
      expect(initDiag.refinementLoopState).toBe('active');
      expect(initDiag.blockCommitPathwayState).toBe('dormant');

      // 3b. diffusion_refinement
      const refDiag = getDiag(buildTwoTowerMacroScene(afterFirstRefinement));
      expect(refDiag.recordKind).toBe('diffusion_refinement');
      expect(refDiag.lifecyclePhase).toBe('refinement_denoised');
      expect(refDiag.committedPrefixLen).toBe(8);
      expect(refDiag.timestepRailActive).toBe(true);
      expect(refDiag.timestepValue).toBeCloseTo(1.0, 6);
      expect(refDiag.refinementLoopState).toBe('active');
      expect(refDiag.blockCommitPathwayState).toBe('dormant');

      // 3c. diffusion_confidence_commit (partial unmasking -> block_refining)
      const confDiag = getDiag(buildTwoTowerMacroScene(afterFirstConfidence));
      expect(confDiag.recordKind).toBe('diffusion_confidence_commit');
      expect(confDiag.lifecyclePhase).toBe('block_refining');
      expect(confDiag.committedPrefixLen).toBe(8); // Context immutable during refinements!
      expect(confDiag.activeBlockMaskedCount).toBe(3);
      expect(confDiag.refinementLoopState).toBe('active');
      expect(confDiag.blockCommitPathwayState).toBe('dormant');

      // 3d. completed_uncommitted: Context STILL at P=8, commit bridge STAGED!
      const uncommScene = buildTwoTowerMacroScene(atCompletedUncommitted);
      const uncommDiag = getDiag(uncommScene);
      expect(uncommDiag.lifecyclePhase).toBe('completed_uncommitted');
      expect(uncommDiag.committedPrefixLen).toBe(8); // STILL at P!
      expect(uncommDiag.activeBlockMaskedCount).toBe(0);
      expect(uncommDiag.refinementLoopState).toBe('paused_ready');
      expect(uncommDiag.blockCommitPathwayState).toBe('staged_ready');
      expect(
        uncommScene.landmarks.some((l) => l.label.includes('Ready to Commit Block -> Context')),
      ).toBe(true);

      // 3e. diffusion_block_commit: Context advances to P+B = 12, commit bridge ACTIVE!
      const commitScene = buildTwoTowerMacroScene(afterBlockCommit);
      const commitDiag = getDiag(commitScene);
      expect(commitDiag.recordKind).toBe('diffusion_block_commit');
      expect(commitDiag.committedPrefixLen).toBe(12); // P + B = 8 + 4!
      expect(commitDiag.refinementLoopState).toBe('dormant');
      expect(commitDiag.blockCommitPathwayState).toBe('active_block_commit');
      expect(commitScene.landmarks.some((l) => l.label.includes('Committed Block -> Context'))).toBe(true);

      // Optional instructional replay phase is strictly separated from measured metadata
      const replayScene = buildTwoTowerMacroScene(afterFirstRefinement, {
        instructionalReplayPhase: 0.42,
        reducedMotion: true,
      });
      const replayDiag = getDiag(replayScene);
      expect(replayDiag.instructionalReplayPhase).toBeCloseTo(0.42, 6);
      expect(replayDiag.instructionalReplayDisclaimer).toBe(SPINE_REPLAY_DISCLAIMER);
      expect(replayDiag.reducedMotion).toBe(true);
      expect(replayDiag.measuredExecutionMode).toBe('mask_diffusion');
      expect(replayScene.provenanceBadges.some((b) => b.label === SPINE_REPLAY_DISCLAIMER)).toBe(true);
    });
  });

  describe('4. Tower-Aware Ray-Picking, Camera Fitting & 10-Axis Selection Identity', () => {
    it('accurately ray-picks Context vs Denoiser bays, Mamba dual compartments, linkers, and timestep hub across DPR 1 & 2', () => {
      const session = runTwoTowerDiffusionRefinement(
        initTwoTowerDiffusionBlock(
          createTwoTowerSession(sharedModel, {
            mode: 'mask_diffusion',
            confidenceThreshold: 0.9999,
            eosTokenIds: [],
          }),
        ),
      );
      const view = resolveTwoTowerSessionViewState(session);
      const scene = buildTwoTowerMacroScene(session, {
        selectedBayId: 6,
        selectedTower: 'context',
      });

      // Active selection populated on scene
      expect(scene.activeSelection).not.toBeNull();
      expect(scene.activeSelection!.towerRole).toBe('context');
      expect(scene.activeSelection!.bayId).toBe(6);
      expect(scene.activeSelection!.layerIndex).toBe(5);
      expect(scene.activeSelection!.mixerKind).toBe('*');
      expect(scene.activeSelection!.modelId).toBe(sharedModel.contextModel.modelId);
      expect(scene.activeSelection!.parameterFingerprint).toBe(
        sharedModel.contextModel.parameterFingerprint,
      );

      // Fit camera across desktop & compact viewports and verify all 8 fitBounds corners project cleanly inside viewport
      for (const [w, h, dpr] of [
        [1280, 800, 1],
        [768, 600, 2],
      ] as const) {
        const metrics = makeViewportMetrics(24, 40, w, h, dpr);
        const pose = fitTwoTowerMacroCamera(scene.fitBounds, metrics);
        const snap = buildCameraSnapshot(pose, scene.fitBounds, metrics);
        for (const corner of aabbCorners(TWO_TOWER_MACRO_FIT_BOUNDS)) {
          const proj = projectWorldToInnerCss(corner, snap);
          expect(proj.visible).toBe(true);
          expect(proj.x).toBeGreaterThanOrEqual(-1);
          expect(proj.x).toBeLessThanOrEqual(metrics.innerWidth + 1);
          expect(proj.y).toBeGreaterThanOrEqual(-1);
          expect(proj.y).toBeLessThanOrEqual(metrics.innerHeight + 1);
        }
      }

      const metrics = makeViewportMetrics(37, 53, 1280, 800, 2);
      const pose = fitTwoTowerMacroCamera(scene.fitBounds, metrics);
      const snap = buildCameraSnapshot(pose, scene.fitBounds, metrics);

      const castAt = (worldPt: readonly [number, number, number]) => {
        const clientPt = projectWorldToClientCss(worldPt, snap);
        expect(clientPt.visible).toBe(true);
        const ray = clientPointToRay(clientPt.x, clientPt.y, snap);
        expect(pointToRayDistance(worldPt, ray)).toBeLessThan(1e-3);
        const hit = pickMacroBay(ray, scene.solidPickBoxes, scene.fasciaPickBoxes);
        expect(hit).not.toBeNull();
        return { hit: hit!, identity: resolvePickHitTowerIdentity(view, hit!) };
      };

      // Context vs Denoiser across Mamba (01, 42), MoE (02), and GQA (06)
      for (const [bayId, expectedKind] of [
        [1, 'M'],
        [2, 'E'],
        [6, '*'],
        [42, 'M'],
      ] as const) {
        const ctxPick = castAt(contextBayFrontWorld(bayId));
        expect(ctxPick.hit.isFascia).toBe(false);
        expect(ctxPick.identity.towerRole).toBe('context');
        expect(ctxPick.identity.bayId).toBe(bayId);
        expect(ctxPick.identity.layerIndex).toBe(bayId - 1);
        expect(ctxPick.identity.mixerKind).toBe(expectedKind);
        expect(ctxPick.identity.modelId).toBe(sharedModel.contextModel.modelId);
        expect(ctxPick.identity.parameterFingerprint).toBe(
          sharedModel.contextModel.parameterFingerprint,
        );
        expect(ctxPick.identity.historicalConfigAtStep).toEqual(view.historicalConfig);
        expect(['context_computed', 'context_reconstructed', 'persistent_cache']).toContain(
          ctxPick.identity.traceOrigin,
        );

        const denPick = castAt(denoiserBayFrontWorld(bayId));
        expect(denPick.hit.isFascia).toBe(false);
        expect(denPick.identity.towerRole).toBe('denoiser');
        expect(denPick.identity.bayId).toBe(bayId);
        expect(denPick.identity.layerIndex).toBe(bayId - 1);
        expect(denPick.identity.mixerKind).toBe(expectedKind);
        expect(denPick.identity.modelId).toBe(sharedModel.denoiserModel.modelId);
        expect(denPick.identity.parameterFingerprint).toBe(
          sharedModel.denoiserModel.parameterFingerprint,
        );
        expect(denPick.identity.traceOrigin).toBe('denoiser_scratch');
        expect(denPick.identity.executionId).toBe(view.denoiserLookup.pass!.passExecutionId);
      }

      // Context Mamba-2 dual compartments (Conv1D vs FP32 SSM H) pick cleanly and distinctly
      const convPick = castAt(contextMambaConvFrontWorld(1));
      expect(convPick.hit.partId).toBe('ctx-conv-bay-01');
      expect(convPick.identity.towerRole).toBe('context');
      expect(convPick.identity.traceOrigin).toBe('persistent_cache');

      const ssmPick = castAt(contextMambaSsmFrontWorld(1));
      expect(ssmPick.hit.partId).toBe('ctx-ssm-bay-01');
      expect(ssmPick.identity.towerRole).toBe('context');
      expect(ssmPick.identity.traceOrigin).toBe('persistent_cache');

      // Cross-tower GQA KV linker, Mamba Conv1D & SSM sub-lane linkers, & Timestep Hub pick cleanly
      const gqaLinkerPick = castAt([0, bayCenterY(6), 0.44]);
      expect(gqaLinkerPick.hit.partId).toBe('linker-gqa-kv-bay-06');
      expect(gqaLinkerPick.identity.towerRole).toBe('shared');
      expect(gqaLinkerPick.identity.bayId).toBe(6);
      expect(gqaLinkerPick.identity.layerIndex).toBe(5);
      expect(gqaLinkerPick.identity.traceOrigin).toBe('persistent_cache');

      const convLinkerPick = castAt([0, bayCenterY(1) + MAMBA_CONV_LANE_DY, 0.26]);
      expect(convLinkerPick.hit.partId).toBe('linker-mamba-conv-bay-01');
      expect(convLinkerPick.identity.towerRole).toBe('shared');
      expect(convLinkerPick.identity.bayId).toBe(1);
      expect(convLinkerPick.identity.layerIndex).toBe(0);

      const ssmLinkerPick = castAt([0, bayCenterY(1) + MAMBA_SSM_LANE_DY, 0.26]);
      expect(ssmLinkerPick.hit.partId).toBe('linker-mamba-ssm-bay-01');
      expect(ssmLinkerPick.identity.towerRole).toBe('shared');
      expect(ssmLinkerPick.identity.bayId).toBe(1);
      expect(ssmLinkerPick.identity.layerIndex).toBe(0);

      const tsHubPick = castAt([TIMESTEP_RAIL_X, TIMESTEP_HUB_Y, 0.16]);
      expect(tsHubPick.hit.partId).toBe('timestep-hub');
      expect(tsHubPick.identity.towerRole).toBe('shared');
      expect(tsHubPick.identity.bayId).toBe(-1);
      expect(tsHubPick.identity.layerIndex).toBeNull();
      expect(tsHubPick.identity.mixerKind).toBe('time_cond');
    });
  });

  describe('5. Historical configAtStep Immutability (Never Substitutes session.pendingConfig)', () => {
    it('renders historical record configAtStep even after mutating session.pendingConfig to a different mode/B/K/γ', () => {
      const s0 = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        eosTokenIds: [],
      });
      const s1 = initTwoTowerDiffusionBlock(s0);
      const s2 = runTwoTowerDiffusionRefinement(s1); // cursor = 2 (mask_diffusion, B=4, K=4, γ=0.9999)
      const s3 = commitTwoTowerRefinementConfidence(s2); // cursor = 3

      // Mutate pendingConfig to completely different settings: AR mode, B=8, K=2, γ=0.22
      const withChangedPending = updateTwoTowerSessionConfig(s3, {
        mode: 'ar',
        blockSize: 8,
        stepsPerBlock: 2,
        confidenceThreshold: 0.22,
      });
      expect(withChangedPending.pendingConfig.mode).toBe('ar');
      expect(withChangedPending.pendingConfig.blockSize).toBe(8);

      // Rewind to historical diffusion_refinement record #2
      const rewound = navigateTwoTowerSession(withChangedPending, 2);
      const scene = buildTwoTowerMacroScene(rewound, { selectedBayId: 6, selectedTower: 'denoiser' });
      const diag = getDiag(scene);

      expect(diag.measuredExecutionMode).toBe('mask_diffusion');
      expect(diag.historicalConfigAtStep.mode).toBe('mask_diffusion');
      expect(diag.historicalConfigAtStep.blockSize).toBe(4);
      expect(diag.historicalConfigAtStep.stepsPerBlock).toBe(4);
      expect(diag.historicalConfigAtStep.confidenceThreshold).toBeCloseTo(0.9999, 6);
      expect(diag.timestepRailActive).toBe(true);
      expect(diag.activeBlockSize).toBe(4);
      expect(scene.activeSelection!.historicalConfigAtStep.mode).toBe('mask_diffusion');
      expect(scene.activeSelection!.historicalConfigAtStep.blockSize).toBe(4);

      // Conversely: AR record rewound after changing pendingConfig to mask_diffusion still renders AR!
      const arSession = stepTwoTowerSessionAr(
        createTwoTowerSession(sharedModel, { mode: 'ar', eosTokenIds: [] }),
      );
      const arPendingDiff = updateTwoTowerSessionConfig(arSession, {
        mode: 'mask_diffusion',
        blockSize: 8,
        stepsPerBlock: 6,
      });
      const arScene = buildTwoTowerMacroScene(arPendingDiff);
      expect(getDiag(arScene).measuredExecutionMode).toBe('ar');
      expect(getDiag(arScene).timestepRailActive).toBe(false);
    });
  });

  describe('6. Forked-Branch Shared Snapshots, Zero-Mutation Guarantee & Fail-Safe Guardrails', () => {
    it('shares ancestral snapshots across forked branches and never mutates any snapshot typed arrays', () => {
      const s0 = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        eosTokenIds: [],
      });
      const s1 = initTwoTowerDiffusionBlock(s0);
      const s2 = runTwoTowerDiffusionRefinement(s1);

      // Fork branch B at cursor 1 (after init) while main stays on refinement
      const rewoundMain = navigateTwoTowerSession(s2, 1);
      const forked = forkTwoTowerSessionBranch(rewoundMain, { label: 'alt-branch' });
      const forkedRec = getActiveRecord(forked);
      const mainBranchSession = switchTwoTowerSessionBranch(forked, forked.branches[0].branchId);
      const mainRecAt1 = mainBranchSession.branches[0].records[1];

      // Ancestral Context snapshot is shared by reference across branches
      expect(forkedRec.contextSnapshot).toBe(mainRecAt1.contextSnapshot);
      const digestBefore = forkedRec.contextSnapshot.snapshotDigest;
      const kvBefore = new Float32Array(forkedRec.contextSnapshot.ctxState.gqaCaches.get(6)!.K);
      const ssmBefore = new Float32Array(forkedRec.contextSnapshot.ctxState.mambaStates.get(1)!.ssmState);
      const logitsBefore = new Float32Array(forkedRec.contextSnapshot.arLogits);

      // Build macro scenes across multiple channels, selections, and replay phases
      for (const channel of ['all', 'gqa_kv', 'mamba_conv', 'mamba_ssm', 'timestep'] as const) {
        buildTwoTowerMacroScene(forked, {
          activeLinkerChannel: channel,
          selectedBayId: 6,
          selectedTower: 'context',
          instructionalReplayPhase: 0.33,
        });
      }
      buildTwoTowerMacroScene(mainBranchSession, { selectedBayId: 1, selectedTower: 'denoiser' });

      // Bit-exact non-mutation verification
      expect(() => validateContextSnapshot(forkedRec.contextSnapshot, sharedModel.contextModel)).not.toThrow();
      expect(forkedRec.contextSnapshot.snapshotDigest).toBe(digestBefore);
      expect(Array.from(forkedRec.contextSnapshot.ctxState.gqaCaches.get(6)!.K)).toEqual(
        Array.from(kvBefore),
      );
      expect(Array.from(forkedRec.contextSnapshot.ctxState.mambaStates.get(1)!.ssmState)).toEqual(
        Array.from(ssmBefore),
      );
      expect(Array.from(forkedRec.contextSnapshot.arLogits)).toEqual(Array.from(logitsBefore));
    });

    it('handles externally tampered Context snapshots fail-safely with explicit warningBanner', () => {
      const session = initTwoTowerDiffusionBlock(
        createTwoTowerSession(sharedModel, {
          mode: 'mask_diffusion',
          eosTokenIds: [],
        }),
      );
      const snap = getActiveRecord(session).contextSnapshot;
      const gqa6 = snap.ctxState.gqaCaches.get(6)!;
      const originalVal = gqa6.K[0];

      try {
        // Simulate unauthorized external typed-array tampering
        gqa6.K[0] = originalVal + 1.25;
        const scene = buildTwoTowerMacroScene(session);
        const diag = getDiag(scene);
        expect(diag.snapshotIntegrityValid).toBe(false);
        expect(diag.snapshotIntegrityError).toBeTruthy();
        expect(scene.warningBanner).not.toBeNull();
        expect(scene.warningBanner!).toMatch(/integrity|mutation|digest/i);
        expect(scene.solidCount).toBeGreaterThan(0);
      } finally {
        // Restore cleanly
        gqa6.K[0] = originalVal;
      }
    });

    it('handles pruned historical Context execution fail-safely without crashing or inventing a fake trace', () => {
      // maxRecords = 2 ensures context_prefill is evicted after init -> refinement -> confidence_commit
      let session = createTwoTowerSession(sharedModel, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        maxRecords: 2,
        eosTokenIds: [],
      });
      session = initTwoTowerDiffusionBlock(session);
      session = runTwoTowerDiffusionRefinement(session);
      session = commitTwoTowerRefinementConfidence(session);

      expect(getLatestContextExecution(session)).toBeNull();

      const scene = buildTwoTowerMacroScene(session, { selectedBayId: 1, selectedTower: 'context' });
      const diag = getDiag(scene);

      expect(diag.contextTraceStatus).toBe('pruned_unavailable');
      expect(diag.snapshotIntegrityValid).toBe(true);
      expect(scene.warningBanner).not.toBeNull();
      expect(scene.warningBanner!).toMatch(/prun|unavailable/i);
      expect(scene.solidCount).toBeGreaterThan(0);
      expect(scene.activeSelection!.traceOrigin).toBe('persistent_cache');
    });
  });

  describe('7. Strict Headless & Zero-Math.random Guardrails', () => {
    it('enforces zero DOM/WebGL imports and zero Math.random() in dualMacroScene.ts and provenance.ts', () => {
      const files = [
        'src/twotower/scene/dualMacroScene.ts',
        'src/twotower/scene/provenance.ts',
      ];
      for (const rel of files) {
        const src = readFileSync(resolve(__dirname, '..', rel), 'utf-8');
        const codeOnly = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
        expect(codeOnly, `${rel} must not reference DOM or WebGL APIs`).not.toMatch(
          /\b(?:window|document|HTMLElement|WebGL2RenderingContext|HTMLCanvasElement)\b/,
        );
        expect(codeOnly, `${rel} must not import WebGL pass modules`).not.toMatch(
          /from\s+['"][^'"]*render\/(?:device|solids|lines|ribbons|sheets)['"]/,
        );
        expect(codeOnly, `${rel} must not call Math.random()`).not.toMatch(/Math\.random\s*\(/);
      }
    });
  });
});
