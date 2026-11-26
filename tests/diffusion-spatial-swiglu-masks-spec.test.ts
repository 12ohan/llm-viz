import { afterAll, describe, expect, it } from 'vitest';
import {
  createDefaultDiffusionModel,
  createDiffusionSession,
  DIFFUSION_MINI_MASK_TOKEN_ID,
  type DiffusionExplorerMaskMode,
} from '../src/diffusion';
import {
  buildDiffusionSpatialScene,
  DEFAULT_DIFFUSION_FOCUS_STATE,
  DIFFUSION_BOOKEND_ATLAS_SIZE,
  DIFFUSION_MASK_ATLAS_SIZE,
  DIFFUSION_SPATIAL_VIEW_MODES,
  DIFFUSION_SPEC_ATLAS_SIZE,
  DIFFUSION_SWIGLU_ATLAS_SIZE,
  pickHitToDiffusionCell,
  resolveDiffusionSessionContext,
  validateSceneGeometryFinite,
  type DiffusionFocusState,
  type DiffusionSpatialScene,
  type DiffusionSpatialViewMode,
} from '../src/diffusion/scene';
import type { Vec3 } from '../src/model/spec';
import { normalize3, sub3 } from '../src/render/camera';
import { pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import { localToWorldOnFrame } from '../src/render/sheetFrame';
import {
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_ROW_MAJOR,
} from '../src/render/sheets';

function readAtlasTexel(
  data: Float32Array,
  width: number,
  x: number,
  y: number,
): readonly [number, number] {
  const idx = (y * width + x) * 2;
  return [data[idx]!, data[idx + 1]!];
}

function makeDirectRayAtBoxCell(
  box: PickBox,
  row: number,
  col: number,
  distance = 3.5,
): Ray {
  const frame = box.frame!;
  const cols = Math.max(1, box.gridCols ?? 1);
  const rows = Math.max(1, box.gridRows ?? 1);
  const lx = -0.5 + (col + 0.5) / cols;
  const lz = -0.5 + (row + 0.5) / rows;
  const target = localToWorldOnFrame(frame, lx, 0.5, lz);
  const origin: Vec3 = [
    target[0] + frame.axisN[0] * distance,
    target[1] + frame.axisN[1] * distance,
    target[2] + frame.axisN[2] * distance,
  ];
  return {
    origin,
    direction: normalize3(sub3(target, origin)),
  };
}

describe('DF3 Spatial Views: SwiGLU, 5-Tab Mask Explorer, Spec Verify, Bookends & 7-View Router', () => {
  const model = createDefaultDiffusionModel();
  const prompt = [0, 1, 2, 3, 4, 5, 6, 7];

  const sweptAtlasStats = new Map<
    DiffusionSpatialViewMode,
    { width: number; height: number; utilizationPct: number } | null
  >();

  afterAll(() => {
    const summary: string[] = [];
    for (const desc of DIFFUSION_SPATIAL_VIEW_MODES) {
      const stat = sweptAtlasStats.get(desc.id);
      if (stat) {
        summary.push(`${desc.id}=${stat.width}x${stat.height}(${stat.utilizationPct.toFixed(2)}%)`);
      } else {
        summary.push(`${desc.id}=no-atlas`);
      }
    }
    console.info('[DF3-All-7-Views-Atlas-Summary]', summary.join(' | '));
  });

  describe('1. Dense SwiGLU Cutaway (`swiglu`)', () => {
    it('verifies 64×64 non-overlapping atlas, live cell-hover formula, progressive accumulation reveal, OBB picking round-trip, and finite geometry', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 4, temperature: 0 },
      });
      session.step();
      const ctx = resolveDiffusionSessionContext(session);

      // Baseline full-reveal scene
      const baseScene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'swiglu',
        selectedLayerIndex: 14,
      });

      expect(baseScene.viewMode).toBe('swiglu');
      expect(baseScene.valueAtlas?.width).toBe(DIFFUSION_SWIGLU_ATLAS_SIZE);
      expect(baseScene.valueAtlas?.height).toBe(DIFFUSION_SWIGLU_ATLAS_SIZE);
      expect(baseScene.valueAtlas?.data.length).toBe(64 * 64 * 2);
      expect(baseScene.atlasAudit?.hasOverlap).toBe(false);
      expect(baseScene.atlasAudit?.outOfBoundsIds.length).toBe(0);
      expect(validateSceneGeometryFinite(baseScene).valid).toBe(true);

      // OBB picking round-trip on `gated`, `gate_silu`, and `layer_out`
      const targets: Array<{ partId: string; row: number; col: number }> = [
        { partId: 'gated', row: 3, col: 11 },
        { partId: 'gate_silu', row: 2, col: 5 },
        { partId: 'layer_out', row: 4, col: 6 },
      ];
      for (const t of targets) {
        const box = baseScene.solidPickBoxes.find((b) => b.partId === t.partId);
        expect(box, `SwiGLU PickBox ${t.partId}`).toBeDefined();
        const ray = makeDirectRayAtBoxCell(box!, t.row, t.col);
        const hit = pickMacroBay(ray, baseScene.solidPickBoxes, baseScene.fasciaPickBoxes);
        expect(hit?.partId).toBe(t.partId);
        const identity = pickHitToDiffusionCell(hit, 'swiglu');
        expect(identity).not.toBeNull();
        expect(identity!.layerIndex).toBe(14);
        expect(identity!.tensorKey).toBe(t.partId);
        expect(identity!.row).toBe(t.row);
        expect(identity!.col).toBe(t.col);

        // Feeding picked identity into hoveredCell drives live formula + cellPopover
        const hoveredScene = buildDiffusionSpatialScene(ctx, {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'swiglu',
          selectedLayerIndex: 14,
          hoveredCell: identity,
        });
        expect(hoveredScene.activeFormula).toBeTruthy();
        expect(hoveredScene.activeFormula!.equation.length).toBeGreaterThan(0);
        expect(hoveredScene.cellPopover).not.toBeNull();
        expect(hoveredScene.cellPopover!.equation).toBe(
          hoveredScene.activeFormula!.equation,
        );
        expect(validateSceneGeometryFinite(hoveredScene).valid).toBe(true);
      }

      // Progressive accumulation reveal at swigluAccumReveal = 0.5
      const revealScene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'swiglu',
        selectedLayerIndex: 14,
        swigluAccumReveal: 0.5,
        hoveredCell: {
          viewMode: 'swiglu',
          layerIndex: 14,
          partId: 'gated',
          tensorKey: 'gated',
          qHeadIndex: null,
          kvHeadIndex: null,
          rotaryPlane: null,
          tokenIndex: 2,
          absPosition: 2,
          row: 2,
          col: 7,
          semanticTag: 'gated',
        },
      });

      const sheetCount = revealScene.sheetCount ?? 0;
      expect(sheetCount).toBeGreaterThan(0);
      expect(revealScene.sheetRevealData).toBeDefined();
      let rowMajorCount = 0;
      for (let i = 0; i < sheetCount; i++) {
        const mode = Math.round(revealScene.sheetRevealData![i * SHEET_REVEAL_FLOATS]!);
        if (mode === SHEET_REVEAL_ROW_MAJOR) rowMajorCount++;
      }
      // 5 progressive sheets: gate_pre, gate_silu, up_val, gated, swiglu_out
      expect(rowMajorCount).toBe(5);
      expect(revealScene.cellPopover?.isAccumulating).toBe(true);
      expect(validateSceneGeometryFinite(revealScene).valid).toBe(true);
    });
  });

  describe('2. 5-Tab Mask Explorer (`mask_explorer`)', () => {
    it('iterates all 5 tabs verifying 64×64 non-overlap, D24 live RG32F G-channel mask predicates, and joint_training OBB + 3-coordinate readout', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 4 },
      });
      session.step();
      const ctx = resolveDiffusionSessionContext(session);

      const expectedStats: Record<
        DiffusionExplorerMaskMode,
        { readonly visible: number; readonly masked: number; readonly plates: number }
      > = {
        joint_training: { visible: 45, masked: 99, plates: 1 },
        ar_causal: { visible: 15, masked: 6, plates: 2 },
        block_bidir: { visible: 32, masked: 0, plates: 1 },
        causal_commit_verify: { visible: 26, masked: 6, plates: 1 },
        vlm_quadratic_ss: { visible: 29, masked: 19, plates: 1 },
      };

      const tabs: readonly DiffusionExplorerMaskMode[] = [
        'joint_training',
        'ar_causal',
        'block_bidir',
        'causal_commit_verify',
        'vlm_quadratic_ss',
      ];

      for (const tab of tabs) {
        const scene = buildDiffusionSpatialScene(ctx, {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'mask_explorer',
          selectedMaskTab: tab,
        });

        expect(scene.viewMode).toBe('mask_explorer');
        expect(scene.valueAtlas?.width).toBe(DIFFUSION_MASK_ATLAS_SIZE);
        expect(scene.valueAtlas?.height).toBe(DIFFUSION_MASK_ATLAS_SIZE);
        expect(scene.atlasAudit?.hasOverlap, `${tab}.overlap`).toBe(false);
        expect(scene.atlasAudit?.outOfBoundsIds.length, `${tab}.oob`).toBe(0);
        expect(validateSceneGeometryFinite(scene).valid, `${tab}.finite`).toBe(true);

        // Verify every SHEET_KIND_CAUSAL_SCORES sheet binds a live RG32F slice and count G=0 vs G=1
        const sheetCount = scene.sheetCount ?? 0;
        const sheetData = scene.sheetData!;
        const atlas = scene.valueAtlas!;
        let causalScoreSheets = 0;
        let gZeroCount = 0;
        let gOneCount = 0;

        for (let i = 0; i < sheetCount; i++) {
          const base = i * SHEET_INSTANCE_FLOATS;
          const cols = Math.round(sheetData[base + 6]!);
          const rows = Math.round(sheetData[base + 7]!);
          const kind = Math.round(sheetData[base + 8]!);
          const u0 = sheetData[base + 13]!;
          const v0 = sheetData[base + 14]!;
          const uW = sheetData[base + 15]!;
          const vH = sheetData[base + 16]!;
          expect(kind).not.toBe(SHEET_KIND_CAUSAL_SOFTMAX);

          if (kind === SHEET_KIND_CAUSAL_SCORES) {
            causalScoreSheets++;
            expect(uW, `${tab}.sheet[${i}].uWidth`).toBeGreaterThan(0);
            expect(vH, `${tab}.sheet[${i}].vHeight`).toBeGreaterThan(0);

            const x0 = Math.round(u0 * atlas.width);
            const y0 = Math.round(v0 * atlas.height);
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) {
                const [, g] = readAtlasTexel(atlas.data, atlas.width, x0 + c, y0 + r);
                expect(g === 0.0 || g === 1.0, `${tab} G in {0,1}`).toBe(true);
                if (g === 0.0) gZeroCount++;
                else gOneCount++;
              }
            }
          }
        }

        const exp = expectedStats[tab];
        expect(causalScoreSheets, `${tab}.plates`).toBe(exp.plates);
        expect(gZeroCount, `${tab}.visibleCount`).toBe(exp.visible);
        expect(gOneCount, `${tab}.maskedCount`).toBe(exp.masked);

        if (tab === 'joint_training') {
          const mobc = scene.diagnostics?.mobcCheck as
            | { valid: boolean; minDeltaP: number | null; zeroTwinViolations: number }
            | undefined;
          expect(mobc?.valid).toBe(true);
          expect(mobc?.minDeltaP).toBeGreaterThanOrEqual(1);
          expect(mobc?.zeroTwinViolations).toBe(0);
        }

        if (tab === 'vlm_quadratic_ss') {
          expect(
            scene.provenanceBadges.some((b) =>
              b.label.includes('Quadratic SS · VLM/Paper Ref'),
            ),
          ).toBe(true);
          expect(scene.diagnostics?.isReadOnlyReference).toBe(true);
        }
      }

      // OBB cell picking on joint_training + 3-coordinate formula readout
      const jtScene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'mask_explorer',
        selectedMaskTab: 'joint_training',
      });
      const jtBox = jtScene.solidPickBoxes.find((b) => b.partId === 'mask_joint_training');
      expect(jtBox).toBeDefined();

      // Row 3 (noisy x_t, p=3, block=1) x Col 6 (clean x_0, p=0, block=0) -> M_OBC visible
      const ray = makeDirectRayAtBoxCell(jtBox!, 3, 6);
      const hit = pickMacroBay(ray, jtScene.solidPickBoxes, jtScene.fasciaPickBoxes);
      expect(hit?.partId).toBe('mask_joint_training');
      const identity = pickHitToDiffusionCell(hit, 'mask_explorer');
      expect(identity?.row).toBe(3);
      expect(identity?.col).toBe(6);

      const inspectedScene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'mask_explorer',
        selectedMaskTab: 'joint_training',
        hoveredCell: identity,
      });
      expect(inspectedScene.activeFormula?.equation).toContain('M_OBC');
      expect(inspectedScene.activeFormula?.subEquation).toContain('seq_idx=');
      expect(inspectedScene.activeFormula?.subEquation).toContain('RoPE pos');
      expect(inspectedScene.activeFormula?.subEquation).toContain('cache_pos');
      expect(inspectedScene.cellPopover).not.toBeNull();
    });
  });

  describe('3. Speculative Verify & 0..31 KV Lifecycle Strip (`spec_verify`)', () => {
    it('inspects live & historical linear_spec_lora spec_cycle records (with lazy cropped suffix KV reconstruction) and dlm(causalContext=false)', () => {
      // Advance linear_spec_lora until we have at least one spec_cycle with cropped suffix slabs
      // followed by another step so the first spec_cycle becomes historical & evicted.
      const specSession = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: {
          mode: 'linear_spec_lora',
          blockLength: 4,
          maxNewTokens: 20,
          temperature: 0.7,
          samplerBThreshold: 0.0,
        },
      });
      specSession.step(); // prefill

      let targetSpecRecId: string | null = null;
      let targetExpectedCropCount = 0;
      for (let i = 0; i < 6; i++) {
        const rec = specSession.step();
        if (rec.kind === 'spec_cycle' && rec.cropProvenance && rec.cropProvenance.croppedSuffixLen > 0) {
          if (targetSpecRecId === null) {
            targetSpecRecId = rec.recordId;
            targetExpectedCropCount = rec.cropProvenance.croppedSuffixLen;
          } else {
            // Subsequent step evicts activeCroppedSuffixSlabs on earlier records
            break;
          }
        }
      }

      // Live tip spec_cycle scene
      const liveTipRec = specSession.getActiveRecord()!;
      expect(liveTipRec.kind).toBe('spec_cycle');
      const liveScene = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(specSession, liveTipRec.recordId),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'spec_verify' },
      );
      expect(liveScene.viewMode).toBe('spec_verify');
      expect(liveScene.valueAtlas?.width).toBe(DIFFUSION_SPEC_ATLAS_SIZE);
      expect(liveScene.valueAtlas?.height).toBe(DIFFUSION_SPEC_ATLAS_SIZE);
      expect(liveScene.atlasAudit?.hasOverlap).toBe(false);
      expect(liveScene.atlasAudit?.outOfBoundsIds.length).toBe(0);
      expect(liveScene.flowIndices?.length).toBeGreaterThan(0);
      expect(Number(liveScene.diagnostics?.shiftedConnectorsCount)).toBeGreaterThan(0);
      expect(Number(liveScene.diagnostics?.loraDeltaMaxAbs)).toBeGreaterThan(1e-4);
      expect(liveScene.diagnostics?.kvMaskProvenance).toBe('token_causal');
      expect(validateSceneGeometryFinite(liveScene).valid).toBe(true);

      // Historical spec_cycle with evicted activeCroppedSuffixSlabs -> lazy reconstructSpecCroppedSuffixKvs
      expect(targetSpecRecId).not.toBeNull();
      const evictedRec = specSession.getRecord(targetSpecRecId!)!;
      expect(evictedRec.activeCroppedSuffixSlabs).toBeNull();
      const historicalCtx = resolveDiffusionSessionContext(specSession, targetSpecRecId!);
      const reconstructedSlabs = historicalCtx.resolveSpecCroppedSuffixSlabs();
      expect(reconstructedSlabs.length).toBe(targetExpectedCropCount);

      const histScene = buildDiffusionSpatialScene(historicalCtx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'spec_verify',
      });
      expect(histScene.diagnostics?.croppedSuffixCount).toBe(targetExpectedCropCount);
      expect(histScene.diagnostics?.isLiveSpecRecord).toBe(true);
      expect(histScene.atlasAudit?.hasOverlap).toBe(false);
      expect(validateSceneGeometryFinite(histScene).valid).toBe(true);

      // Also test hovering a cell in `draft_lora_delta` to verify live LoRA ON vs OFF delta formula
      const deltaBox = histScene.solidPickBoxes.find((b) => b.partId === 'draft_lora_delta');
      expect(deltaBox).toBeDefined();
      const deltaRay = makeDirectRayAtBoxCell(deltaBox!, 0, 5);
      const deltaHit = pickMacroBay(deltaRay, histScene.solidPickBoxes, histScene.fasciaPickBoxes);
      expect(deltaHit?.partId).toBe('draft_lora_delta');
      const deltaCell = pickHitToDiffusionCell(deltaHit, 'spec_verify');
      const deltaHoverScene = buildDiffusionSpatialScene(historicalCtx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'spec_verify',
        hoveredCell: deltaCell,
      });
      expect(deltaHoverScene.activeFormula?.equation).toContain('LoRA_ON');
      expect(deltaHoverScene.activeFormula?.equation).toContain('LoRA_OFF');
      expect(deltaHoverScene.cellPopover).not.toBeNull();

      // Verify linear_spec (LoRA OFF) vs linear_spec_lora (LoRA ON) both produce valid 64x64 scenes
      const specNoLora = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: { mode: 'linear_spec', blockLength: 4, maxNewTokens: 8, temperature: 0 },
      });
      specNoLora.step();
      specNoLora.step();
      const noLoraScene = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(specNoLora),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'spec_verify' },
      );
      expect(noLoraScene.atlasAudit?.hasOverlap).toBe(false);
      expect(noLoraScene.activeFormula?.headBadge).toBe('LoRA OFF');
      expect(validateSceneGeometryFinite(noLoraScene).valid).toBe(true);

      // DLM causalContext=false ('block_bidirectional') record
      const dlmBidirSession = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: {
          mode: 'dlm',
          causalContext: false,
          blockLength: 4,
          maxNewTokens: 8,
          temperature: 0,
        },
      });
      dlmBidirSession.step(); // prefill
      const dlmRec = dlmBidirSession.step(); // dlm block
      expect(dlmRec.kvMaskProvenance).toBe('block_bidirectional');
      const dlmScene = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(dlmBidirSession, dlmRec.recordId),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'spec_verify' },
      );
      expect(dlmScene.atlasAudit?.hasOverlap).toBe(false);
      expect(dlmScene.diagnostics?.kvMaskProvenance).toBe('block_bidirectional');
      expect(dlmScene.diagnostics?.croppedSuffixCount).toBe(0);
      expect(dlmScene.warningBanner).toContain('Bidir KV');
      expect(validateSceneGeometryFinite(dlmScene).valid).toBe(true);
    });
  });

  describe('4. Bookend Cutaways (`embedding` & `output`)', () => {
    it('verifies `embedding` 64×64 atlas, special tokens 32..35 highlight, RoPE per-layer disclosure, and finite geometry', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 4 },
      });
      session.step();
      const ctx = resolveDiffusionSessionContext(session);
      const scene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'embedding',
      });

      expect(scene.viewMode).toBe('embedding');
      expect(scene.valueAtlas?.width).toBe(DIFFUSION_BOOKEND_ATLAS_SIZE);
      expect(scene.valueAtlas?.height).toBe(DIFFUSION_BOOKEND_ATLAS_SIZE);
      expect(scene.atlasAudit?.hasOverlap).toBe(false);
      expect(scene.atlasAudit?.outOfBoundsIds.length).toBe(0);
      expect(scene.diagnostics?.specialTokenIds).toEqual([32, 33, 34, 35]);
      expect(scene.partCatalog.special_embeddings).toBeDefined();
      expect(
        scene.landmarks.some((l) => l.id === 'embed-lm-specials' && l.label.includes('[MASK]')),
      ).toBe(true);
      expect(
        scene.provenanceBadges.some((b) => b.label.includes('Zero Learned Pos Emb')),
      ).toBe(true);
      expect(scene.activeFormula?.subEquation).toContain('YaRN RoPE');
      expect(validateSceneGeometryFinite(scene).valid).toBe(true);
    });

    it('verifies `output` 64×64 atlas, +1 vs +0 alignment contrast, un-suppressed [MASK] softmax, and Sampler A vs B threshold planes', () => {
      // Causal (+1) vs DLM Same-Pos (+0)
      const arSession = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 4, temperature: 0 },
      });
      arSession.step();
      const arScene = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(arSession),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'output' },
      );

      expect(arScene.viewMode).toBe('output');
      expect(arScene.valueAtlas?.width).toBe(DIFFUSION_BOOKEND_ATLAS_SIZE);
      expect(arScene.valueAtlas?.height).toBe(DIFFUSION_BOOKEND_ATLAS_SIZE);
      expect(arScene.atlasAudit?.hasOverlap).toBe(false);
      expect(arScene.atlasAudit?.outOfBoundsIds.length).toBe(0);
      expect(arScene.diagnostics?.alignmentMode).toBe('causal_shifted_plus_1');
      expect(Number(arScene.diagnostics?.maskTokenLiveProb)).toBeGreaterThan(0);

      // Verify raw un-suppressed [MASK] (col 32) in the `probs` sheet slice inside the 64×64 atlas
      const [maskProbInAtlas] = readAtlasTexel(
        arScene.valueAtlas!.data,
        arScene.valueAtlas!.width,
        20 + DIFFUSION_MINI_MASK_TOKEN_ID,
        20,
      );
      expect(maskProbInAtlas).toBeGreaterThan(0);

      // Shared diffusion_head +1 vs +0 contrast + Sampler A vs B threshold planes
      expect(arScene.partCatalog.diffusion_head?.badge).toContain('(+1) & DLM (+0)');
      expect(
        arScene.landmarks.some((l) => l.label.includes('Sampler A:')),
      ).toBe(true);
      expect(
        arScene.landmarks.some((l) => l.label.includes('Sampler B:')),
      ).toBe(true);
      expect(
        arScene.landmarks.some((l) => l.id === 'out-lm-thr-planes'),
      ).toBe(true);
      expect(validateSceneGeometryFinite(arScene).valid).toBe(true);

      // DLM Same-Position (+0) alignment on bidirectional block pass (`causalContext: false`)
      const dlmSession = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: { mode: 'dlm', causalContext: false, blockLength: 4, maxNewTokens: 8, temperature: 0 },
      });
      dlmSession.step(); // prefill
      dlmSession.step(); // dlm block
      const dlmOutScene = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(dlmSession),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'output' },
      );
      expect(dlmOutScene.diagnostics?.alignmentMode).toBe('diffusion_same_pos_plus_0');
      expect(dlmOutScene.provenanceBadges.some((b) => b.label.includes('(+0)'))).toBe(true);
      expect(dlmOutScene.atlasAudit?.hasOverlap).toBe(false);
      expect(validateSceneGeometryFinite(dlmOutScene).valid).toBe(true);
    });
  });

  describe('5. Unified Router & All-7-View Sweep', () => {
    it('sweeps buildDiffusionSpatialScene across ALL 7 DIFFUSION_SPATIAL_VIEW_MODES and records atlas dimensions & utilization', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: {
          mode: 'linear_spec_lora',
          blockLength: 4,
          maxNewTokens: 12,
          temperature: 0.7,
        },
      });
      session.step(); // prefill
      const specRec = session.step(); // spec_cycle
      const ctx = resolveDiffusionSessionContext(session, specRec.recordId);

      const scenes: DiffusionSpatialScene[] = [];
      for (const viewDesc of DIFFUSION_SPATIAL_VIEW_MODES) {
        const focus: DiffusionFocusState = {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: viewDesc.id,
          selectedLayerIndex: 7,
          selectedQHead: 4,
          selectedKvHead: 1,
          selectedRotaryPlane: 2,
          selectedMaskTab: 'joint_training',
        };
        const scene = buildDiffusionSpatialScene(ctx, focus);
        scenes.push(scene);

        expect(scene.viewMode).toBe(viewDesc.id);
        expect(validateSceneGeometryFinite(scene).valid, `${viewDesc.id}.finite`).toBe(true);
        if (scene.atlasAudit) {
          expect(scene.atlasAudit.hasOverlap, `${viewDesc.id}.overlap`).toBe(false);
          expect(scene.atlasAudit.outOfBoundsIds.length, `${viewDesc.id}.oob`).toBe(0);
          sweptAtlasStats.set(viewDesc.id, {
            width: scene.atlasAudit.width,
            height: scene.atlasAudit.height,
            utilizationPct: scene.atlasAudit.utilizationPct,
          });
        } else {
          sweptAtlasStats.set(viewDesc.id, null);
        }
      }

      expect(scenes.length).toBe(7);
      expect(sweptAtlasStats.size).toBe(7);
    });
  });
});
