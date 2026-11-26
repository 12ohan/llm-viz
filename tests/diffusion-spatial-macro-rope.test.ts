import { describe, expect, it } from 'vitest';
import {
  createDefaultDiffusionModel,
  createDiffusionSession,
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_MINI_CONFIG,
  materializePastKvs,
  runDiffusionForwardPass,
  type DiffusionExecutionMode,
} from '../src/diffusion';
import {
  buildDiffusionGqaRopeAtlas,
  buildDiffusionSpatialScene,
  computeRopePhasorTelemetry,
  DEFAULT_DIFFUSION_FOCUS_STATE,
  DIFFUSION_GQA_ATLAS_HEIGHT,
  DIFFUSION_GQA_ATLAS_WIDTH,
  DIFFUSION_GQA_MAX_KEY,
  DIFFUSION_GQA_MAX_QUERY,
  DIFFUSION_LORA_RAIL_BADGE,
  DIFFUSION_PROJECTION_REVERSAL_BADGE,
  pickHitToDiffusionCell,
  resolveDiffusionSessionContext,
  validateSceneGeometryFinite,
  type DiffusionFocusState,
  type DiffusionSpatialScene,
} from '../src/diffusion/scene';
import type { Vec3 } from '../src/model/spec';
import {
  buildCameraSnapshot,
  fitOpBoundsCamera,
  makeViewportMetrics,
  normalize3,
  projectWorldToClientCss,
  sub3,
  type ViewportMetrics,
} from '../src/render/camera';
import { clientPointToRay, pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import { localToWorldOnFrame } from '../src/render/sheetFrame';
import {
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
} from '../src/render/sheets';

const VIEWPORT: ViewportMetrics = makeViewportMetrics(0, 0, 1440, 900, 1, 0, 0);

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
  distance = 4.0,
): Ray {
  if (!box.frame) {
    const cx = 0.5 * (box.min[0] + box.max[0]);
    const cy = 0.5 * (box.min[1] + box.max[1]);
    const cz = 0.5 * (box.min[2] + box.max[2]);
    return {
      origin: [cx, cy, cz + distance],
      direction: [0, 0, -1],
    };
  }
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

function assertGqaD24SheetCompliance(
  scene: DiffusionSpatialScene,
  expectedPastLen: number,
  expectedMaskMode: 'causal' | 'bidirectional' | 'custom',
  label: string,
): void {
  const sheetCount = scene.sheetCount ?? 0;
  expect(sheetCount, `${label}.sheetCount`).toBeGreaterThan(0);
  expect(scene.sheetData, `${label}.sheetData`).toBeDefined();
  expect(scene.valueAtlas, `${label}.valueAtlas`).toBeDefined();

  const data = scene.sheetData!;
  const atlas = scene.valueAtlas!;
  let checkedMaskSheets = 0;

  for (let i = 0; i < sheetCount; i++) {
    const base = i * SHEET_INSTANCE_FLOATS;
    const gridCols = Math.round(data[base + 6]!);
    const gridRows = Math.round(data[base + 7]!);
    const sheetKind = Math.round(data[base + 8]!);
    const u0 = data[base + 13]!;
    const v0 = data[base + 14]!;
    const uWidth = data[base + 15]!;
    const vHeight = data[base + 16]!;

    if (
      sheetKind === SHEET_KIND_CAUSAL_SCORES ||
      sheetKind === SHEET_KIND_CAUSAL_SOFTMAX
    ) {
      checkedMaskSheets++;
      expect(uWidth, `${label}.sheet[${i}].uWidth`).toBeGreaterThan(0);
      expect(vHeight, `${label}.sheet[${i}].vHeight`).toBeGreaterThan(0);

      const x0 = Math.round(u0 * atlas.width);
      const y0 = Math.round(v0 * atlas.height);
      const texCols = Math.round(uWidth * atlas.width);
      const texRows = Math.round(vHeight * atlas.height);
      expect(texCols, `${label}.sheet[${i}].texCols`).toBe(gridCols);
      expect(texRows, `${label}.sheet[${i}].texRows`).toBe(gridRows);

      for (let q = 0; q < gridRows; q++) {
        for (let k = 0; k < gridCols; k++) {
          const [, gFlag] = readAtlasTexel(atlas.data, atlas.width, x0 + k, y0 + q);
          const shouldBeMasked =
            expectedMaskMode === 'causal' ? k > expectedPastLen + q : false;
          expect(gFlag, `${label}.sheet[${i}][q=${q},k=${k}].G`).toBe(
            shouldBeMasked ? 1.0 : 0.0,
          );
        }
      }
    }
  }

  // 8 KV-group Scores + 8 KV-group Softmax = 16 causal/mask sheets
  expect(checkedMaskSheets, `${label}.checkedMaskSheets`).toBe(16);
}

describe('DF3 Spatial Macro (40L) & RoPE/YaRN/GQA Cutaway (256×256 Atlas)', () => {
  const model = createDefaultDiffusionModel();
  const defaultPrompt = [0, 1, 2, 3, 4, 5, 6, 7];

  describe('1. 40L Backbone Macro (`macro`)', () => {
    it('renders 40 physical layers (zero duplicated towers), bookends, 0..31 KV ticks, and badges', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 8, temperature: 0 },
      });
      session.step(); // prefill (8 tokens)
      session.step(); // ar_step (9 tokens)

      const ctx = resolveDiffusionSessionContext(session);
      const focus: DiffusionFocusState = {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'macro',
        selectedLayerIndex: 17,
      };
      const scene = buildDiffusionSpatialScene(ctx, focus);

      expect(scene.viewMode).toBe('macro');
      expect(scene.diagnostics?.layerCount).toBe(40);
      expect(validateSceneGeometryFinite(scene).valid).toBe(true);

      // Verify exactly 40 physical layer pick boxes (`bayId = 1..40`, zero duplicated towers)
      const layerPickBoxes = scene.solidPickBoxes.filter(
        (b) => b.semanticTag === 'layer_block',
      );
      expect(layerPickBoxes.length).toBe(40);
      for (let l = 0; l < 40; l++) {
        const box = layerPickBoxes[l]!;
        expect(box.bayId).toBe(l + 1);
        expect(box.partId).toBe(`layer-${String(l + 1).padStart(2, '0')}`);
        expect(scene.partCatalog[box.partId!]).toBeDefined();
      }

      // Bookends: embed_tokens + diffusion_head
      const embedBox = scene.solidPickBoxes.find((b) => b.partId === 'embed_tokens');
      const headBox = scene.solidPickBoxes.find((b) => b.partId === 'diffusion_head');
      expect(embedBox).toBeDefined();
      expect(embedBox?.bayId).toBe(0);
      expect(headBox).toBeDefined();
      expect(headBox?.bayId).toBe(41);
      expect(scene.partCatalog.embed_tokens).toBeDefined();
      expect(scene.partCatalog.diffusion_head).toBeDefined();

      // 0..31 KV rail ticks (capacity = 32)
      expect(scene.kvDepthTicks.length).toBeGreaterThanOrEqual(5);
      expect(scene.kvDepthTicks.map((t) => t.token)).toContain(1);
      expect(scene.kvDepthTicks.map((t) => t.token)).toContain(16);
      expect(scene.kvDepthTicks.map((t) => t.token)).toContain(32);
      expect(scene.diagnostics?.kvCapacity).toBe(32);
      expect(scene.diagnostics?.kvSeqLen).toBe(9);

      // Provenance badges include mode, KV provenance, projection-width reversal, and LoRA rail
      const badgeLabels = scene.provenanceBadges.map((b) => b.label);
      expect(badgeLabels.some((l) => l.startsWith('Mode:'))).toBe(true);
      expect(badgeLabels.some((l) => l.includes('token_causal'))).toBe(true);
      expect(badgeLabels).toContain(DIFFUSION_PROJECTION_REVERSAL_BADGE);
      expect(badgeLabels).toContain(DIFFUSION_LORA_RAIL_BADGE);
    });

    it('asserts loraBridged === true ONLY on linear_spec_lora spec_cycle records and false on prefill/ar/dlm/linear_spec', () => {
      // 1. linear_spec_lora: prefill -> false, spec_cycle -> true
      const specLoraSession = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: {
          mode: 'linear_spec_lora',
          blockLength: 4,
          maxNewTokens: 8,
          temperature: 0.7,
        },
      });
      const prefillRec = specLoraSession.step();
      expect(prefillRec.kind).toBe('prefill');
      const prefillMacro = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(specLoraSession, prefillRec.recordId),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'macro' },
      );
      expect(prefillMacro.diagnostics?.loraBridged).toBe(false);
      expect(validateSceneGeometryFinite(prefillMacro).valid).toBe(true);

      const specCycleRec = specLoraSession.step();
      expect(specCycleRec.kind).toBe('spec_cycle');
      const specLoraMacro = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(specLoraSession, specCycleRec.recordId),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'macro' },
      );
      expect(specLoraMacro.diagnostics?.loraBridged).toBe(true);
      expect(validateSceneGeometryFinite(specLoraMacro).valid).toBe(true);

      // 2. ar, dlm (causal & bidir), and linear_spec -> false
      const nonLoraModes: Array<{
        readonly label: string;
        readonly mode: DiffusionExecutionMode;
        readonly causalContext: boolean;
      }> = [
        { label: 'ar', mode: 'ar', causalContext: true },
        { label: 'dlm-causal', mode: 'dlm', causalContext: true },
        { label: 'dlm-bidir', mode: 'dlm', causalContext: false },
        { label: 'linear_spec', mode: 'linear_spec', causalContext: true },
      ];

      for (const cfg of nonLoraModes) {
        const s = createDiffusionSession(model, {
          promptTokenIds: defaultPrompt,
          config: {
            mode: cfg.mode,
            causalContext: cfg.causalContext,
            blockLength: 4,
            maxNewTokens: 8,
            temperature: 0,
          },
        });
        s.step(); // prefill
        const stepRec = s.step(); // active step
        const macro = buildDiffusionSpatialScene(
          resolveDiffusionSessionContext(s, stepRec.recordId),
          { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'macro' },
        );
        expect(macro.diagnostics?.loraBridged, `${cfg.label}.loraBridged`).toBe(false);
        expect(validateSceneGeometryFinite(macro).valid, `${cfg.label}.finite`).toBe(true);
        // Detached rail must sit farther out on -X than bridged rail
        expect(Number(macro.diagnostics?.loraRailX)).toBeLessThan(
          Number(specLoraMacro.diagnostics?.loraRailX),
        );
      }
    });

    it('ray-picks every layer pick box (bayId = 1..40) and resolves exact layerIndex = 0..39', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 4 },
      });
      session.step();
      const ctx = resolveDiffusionSessionContext(session);
      const scene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'macro',
      });

      for (let l = 0; l < 40; l++) {
        const box = scene.solidPickBoxes.find(
          (b) => b.bayId === l + 1 && b.semanticTag === 'layer_block',
        );
        expect(box, `layer box ${l + 1}`).toBeDefined();

        // Aim ray from -Z toward SwiGLU right-side center so KV rail (+Z) never occludes
        const targetX = box!.max[0] - 0.5;
        const targetY = 0.5 * (box!.min[1] + box!.max[1]);
        const targetZ = 0.5 * (box!.min[2] + box!.max[2]);
        const ray: Ray = {
          origin: [targetX, targetY, targetZ - 5.0],
          direction: [0, 0, 1],
        };
        const hit = pickMacroBay(ray, scene.solidPickBoxes, scene.fasciaPickBoxes);
        expect(hit, `hit layer ${l + 1}`).not.toBeNull();
        expect(hit!.bayId).toBe(l + 1);
        const identity = pickHitToDiffusionCell(hit, 'macro');
        expect(identity?.layerIndex).toBe(l);
        expect(identity?.partId).toBe(`layer-${String(l + 1).padStart(2, '0')}`);
      }
    });
  });

  describe('2. RoPE + YaRN + GQA Cutaway (`gqa_rope`) & 256×256 Atlas', () => {
    it('passes 256×256 non-overlap audit, D24 RG32F mask check, and geometry finiteness across ar, dlm, linear_spec, linear_spec_lora, and max-stress (S=16, K=32)', () => {
      const modeCases: Array<{
        readonly label: string;
        readonly mode: DiffusionExecutionMode;
        readonly causalContext: boolean;
        readonly expectedMaskMode: 'causal' | 'bidirectional';
      }> = [
        { label: 'ar-prefill', mode: 'ar', causalContext: true, expectedMaskMode: 'causal' },
        { label: 'dlm-bidir', mode: 'dlm', causalContext: false, expectedMaskMode: 'bidirectional' },
        { label: 'linear_spec', mode: 'linear_spec', causalContext: true, expectedMaskMode: 'causal' },
        { label: 'linear_spec_lora', mode: 'linear_spec_lora', causalContext: true, expectedMaskMode: 'causal' },
      ];

      for (const c of modeCases) {
        const session = createDiffusionSession(model, {
          promptTokenIds: defaultPrompt,
          config: {
            mode: c.mode,
            causalContext: c.causalContext,
            blockLength: 4,
            maxNewTokens: 8,
            temperature: c.mode === 'linear_spec_lora' ? 0.7 : 0,
          },
        });
        const r0 = session.step();
        const rActive = c.label === 'ar-prefill' ? r0 : session.step();

        const ctx = resolveDiffusionSessionContext(session, rActive.recordId);
        const scene = buildDiffusionSpatialScene(ctx, {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'gqa_rope',
          selectedLayerIndex: 12,
          selectedQHead: 9,
          selectedKvHead: 2,
          selectedRotaryPlane: 1,
        });

        expect(scene.valueAtlas?.width).toBe(DIFFUSION_GQA_ATLAS_WIDTH);
        expect(scene.valueAtlas?.height).toBe(DIFFUSION_GQA_ATLAS_HEIGHT);
        expect(scene.atlasAudit?.hasOverlap, `${c.label}.overlap`).toBe(false);
        expect(scene.atlasAudit?.outOfBoundsIds.length, `${c.label}.oob`).toBe(0);
        expect(validateSceneGeometryFinite(scene).valid, `${c.label}.finite`).toBe(true);

        const trace = ctx.resolveLayerTrace(12);
        assertGqaD24SheetCompliance(
          scene,
          trace.attention.pastLen,
          c.expectedMaskMode,
          c.label,
        );

        // Projection-width reversal badge in both provenanceBadges and calipers
        expect(scene.provenanceBadges.map((b) => b.label)).toContain(
          DIFFUSION_PROJECTION_REVERSAL_BADGE,
        );
        expect(
          scene.calipers?.some((cal) => cal.label === DIFFUSION_PROJECTION_REVERSAL_BADGE),
        ).toBe(true);
      }

      // Max-stress check: S = 16, K = 32 (pastLen = 16 + query = 16)
      const prompt16 = new Int32Array(DIFFUSION_GQA_MAX_QUERY);
      for (let i = 0; i < DIFFUSION_GQA_MAX_QUERY; i++) prompt16[i] = i % 32;
      const pastForward = runDiffusionForwardPass(model, {
        tokenIds: prompt16,
        maskMode: 'causal',
        loraEnabled: false,
        useCache: true,
      });
      const stressSession = createDiffusionSession(model, {
        promptTokenIds: prompt16,
        config: { mode: 'dlm', causalContext: true, blockLength: 4, maxNewTokens: 8 },
      });
      const stressPrefill = stressSession.step();
      const pastKvs16 = materializePastKvs(stressPrefill.kvSnapshot);

      const query16 = new Int32Array(DIFFUSION_GQA_MAX_QUERY);
      for (let i = 0; i < DIFFUSION_GQA_MAX_QUERY; i++) query16[i] = (i + 16) % 32;
      const stressForward = runDiffusionForwardPass(model, {
        tokenIds: query16,
        maskMode: 'causal',
        loraEnabled: true,
        useCache: false,
        pastKvs: pastKvs16,
        retainedTraceLayers: new Set([0]),
      });
      const stressTrace = stressForward.layerTraces.get(0)!;
      expect(stressTrace.attention.seqLen).toBe(DIFFUSION_GQA_MAX_QUERY);
      expect(stressTrace.attention.kvLen).toBe(DIFFUSION_GQA_MAX_KEY);

      const keyPositionIds = new Int32Array(DIFFUSION_GQA_MAX_KEY);
      for (let k = 0; k < DIFFUSION_GQA_MAX_KEY; k++) keyPositionIds[k] = k;

      const stressAtlas = buildDiffusionGqaRopeAtlas(
        stressTrace,
        model.layers[0]!,
        DIFFUSION_MINI_CONFIG,
        {
          layerIndex: 0,
          selectedQHead: 0,
          selectedKvHead: 0,
          positionIds: stressForward.positionIds,
          cachePositions: stressForward.cachePositions,
          keyPositionIds,
          maskMode: 'causal',
          loraEnabled: true,
          snapshotDigest: pastForward.logits.byteLength.toString(),
          inputHiddenStates: stressForward.residualCheckpoints[0],
        },
      );

      expect(stressAtlas.queryTokens).toBe(16);
      expect(stressAtlas.keyTokens).toBe(32);
      expect(stressAtlas.audit.width).toBe(256);
      expect(stressAtlas.audit.height).toBe(256);
      expect(stressAtlas.audit.hasOverlap).toBe(false);
      expect(stressAtlas.audit.outOfBoundsIds.length).toBe(0);

      // Also verify full cutaway scene at S=16 prefill
      const stressScene = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(stressSession, stressPrefill.recordId),
        { ...DEFAULT_DIFFUSION_FOCUS_STATE, viewMode: 'gqa_rope' },
      );
      expect(stressScene.atlasAudit?.hasOverlap).toBe(false);
      expect(validateSceneGeometryFinite(stressScene).valid).toBe(true);
    });

    it('performs 8-KV-head OBB picking round-trip across ALL KV heads g = 0..7 resolving exact (row, col) and kvHeadIndex', () => {
      const session = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 4, temperature: 0 },
      });
      session.step(); // S = 8, K = 8
      const ctx = resolveDiffusionSessionContext(session);
      const scene = buildDiffusionSpatialScene(ctx, {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'gqa_rope',
        selectedLayerIndex: 5,
      });

      const pose = fitOpBoundsCamera(scene.fitBounds, VIEWPORT, {
        azimuthDeg: scene.defaultCameraAzimuthDeg,
        elevationDeg: scene.defaultCameraElevationDeg,
      });
      const camSnapshot = buildCameraSnapshot(pose, scene.fitBounds, VIEWPORT);

      for (let g = 0; g < DIFFUSION_MINI_CONFIG.numKeyValueHeads; g++) {
        const targetRow = g % 8;
        const targetCol = (g * 3 + 1) % 8;

        // 1. Camera clientPointToRay round-trip on scores_g{g}
        const scoreBox = scene.solidPickBoxes.find((b) => b.partId === `scores_g${g}`);
        expect(scoreBox, `scores_g${g} PickBox`).toBeDefined();
        expect(scoreBox!.groupIndex).toBe(g);
        expect(scoreBox!.frame).toBeDefined();

        const worldPt = localToWorldOnFrame(
          scoreBox!.frame!,
          -0.5 + (targetCol + 0.5) / scoreBox!.gridCols!,
          0.5,
          -0.5 + (targetRow + 0.5) / scoreBox!.gridRows!,
        );
        const projected = projectWorldToClientCss(worldPt, camSnapshot);
        expect(projected.visible, `scores_g${g} visible`).toBe(true);

        const camRay = clientPointToRay(projected.x, projected.y, camSnapshot);
        const camHit = pickMacroBay(camRay, scene.solidPickBoxes, scene.fasciaPickBoxes);
        expect(camHit?.partId).toBe(`scores_g${g}`);
        const camCell = pickHitToDiffusionCell(camHit, 'gqa_rope');
        expect(camCell?.kvHeadIndex).toBe(g);
        expect(camCell?.row).toBe(targetRow);
        expect(camCell?.col).toBe(targetCol);
        expect(camCell?.layerIndex).toBe(5);

        // 2. Direct OBB ray pick on softmax_g{g}
        const softmaxBox = scene.solidPickBoxes.find((b) => b.partId === `softmax_g${g}`);
        expect(softmaxBox, `softmax_g${g} PickBox`).toBeDefined();
        const directRay = makeDirectRayAtBoxCell(softmaxBox!, targetRow, targetCol);
        const smHit = pickMacroBay(directRay, scene.solidPickBoxes, scene.fasciaPickBoxes);
        expect(smHit?.partId).toBe(`softmax_g${g}`);
        const smCell = pickHitToDiffusionCell(smHit, 'gqa_rope');
        expect(smCell?.kvHeadIndex).toBe(g);
        expect(smCell?.row).toBe(targetRow);
        expect(smCell?.col).toBe(targetCol);

        // 3. Direct OBB ray pick on k_full (column within KV head g's 8-dim slice)
        const kFullBox = scene.solidPickBoxes.find((b) => b.partId === 'k_full');
        expect(kFullBox).toBeDefined();
        const kCol = g * DIFFUSION_MINI_CONFIG.headDim + 2;
        const kRay = makeDirectRayAtBoxCell(kFullBox!, targetRow, kCol);
        const kHit = pickMacroBay(kRay, scene.solidPickBoxes, scene.fasciaPickBoxes);
        expect(kHit?.partId).toBe('k_full');
        const kCell = pickHitToDiffusionCell(kHit, 'gqa_rope');
        expect(kCell?.row).toBe(targetRow);
        expect(kCell?.col).toBe(kCol);
      }
    });

    it('verifies RoPE phasor & live Llama-4 scaling at pos < 16 vs pos >= 16 across all 4 dyadic rotary planes 0..3', () => {
      // Advance AR session past pos >= 16 (prompt = 8 + 10 decode steps -> pos 17)
      const session = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: { mode: 'ar', blockLength: 4, maxNewTokens: 12, temperature: 0 },
      });
      const prefillRec = session.step(); // pos 0..7 (< 16)
      for (let i = 0; i < 10; i++) {
        session.step();
      }
      const post16Rec = session.getActiveRecord()!;
      expect(post16Rec.primaryPassCheckpoint.positionIds[0]).toBeGreaterThanOrEqual(16);

      const expectedInvFreq = Array.from(DIFFUSION_DYADIC_YARN_INV_FREQ);
      expect(expectedInvFreq).toEqual([1.0, 0.3125, 0.0625, 0.03125]);

      // Case A: pos < 16 (row 5 in prefill)
      const preCtx = resolveDiffusionSessionContext(session, prefillRec.recordId);
      for (let plane = 0; plane < 4; plane++) {
        const preScene = buildDiffusionSpatialScene(preCtx, {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'gqa_rope',
          selectedLayerIndex: 3,
          selectedQHead: 5,
          selectedKvHead: 1,
          selectedRotaryPlane: plane,
          selectedTokenRow: 5,
        });
        const phasor = preScene.ropePhasor!;
        expect(phasor).toBeDefined();
        expect(phasor.queryPosition).toBe(5);
        expect(phasor.rotaryPlane).toBe(plane);
        expect(phasor.invFreq).toBeCloseTo(expectedInvFreq[plane]!, 7);
        expect(phasor.llama4Scale).toBe(1.0);
        expect(phasor.qScaledPair[0]).toBe(phasor.qRotPair[0]);
        expect(phasor.qScaledPair[1]).toBe(phasor.qRotPair[1]);

        const kPreNorm = Math.hypot(phasor.kPrePair[0], phasor.kPrePair[1]);
        const kRotNorm = Math.hypot(phasor.kRotPair[0], phasor.kRotPair[1]);
        expect(kRotNorm).toBeCloseTo(kPreNorm, 4);
        expect(validateSceneGeometryFinite(preScene).valid).toBe(true);
      }

      // Case B: pos >= 16 (AR decode at pos 17)
      const expectedL4 = 1 + 0.1 * Math.log(2); // ~1.0693147
      const postCtx = resolveDiffusionSessionContext(session, post16Rec.recordId);
      for (let plane = 0; plane < 4; plane++) {
        const postScene = buildDiffusionSpatialScene(postCtx, {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'gqa_rope',
          selectedLayerIndex: 3,
          selectedQHead: 5,
          selectedKvHead: 1,
          selectedRotaryPlane: plane,
          selectedTokenRow: 0,
        });
        const phasor = postScene.ropePhasor!;
        expect(phasor.queryPosition).toBeGreaterThanOrEqual(16);
        expect(phasor.rotaryPlane).toBe(plane);
        expect(phasor.invFreq).toBeCloseTo(expectedInvFreq[plane]!, 7);
        expect(phasor.llama4Scale).toBeCloseTo(expectedL4, 5);

        const qRotNorm = Math.hypot(phasor.qRotPair[0], phasor.qRotPair[1]);
        const qScaledNorm = Math.hypot(phasor.qScaledPair[0], phasor.qScaledPair[1]);
        if (qRotNorm > 1e-5) {
          expect(qScaledNorm / qRotNorm).toBeCloseTo(expectedL4, 4);
        }

        // Keys remain strictly unscaled by Llama-4!
        const kPreNorm = Math.hypot(phasor.kPrePair[0], phasor.kPrePair[1]);
        const kRotNorm = Math.hypot(phasor.kRotPair[0], phasor.kRotPair[1]);
        expect(kRotNorm).toBeCloseTo(kPreNorm, 4);
        expect(validateSceneGeometryFinite(postScene).valid).toBe(true);
      }

      // Also verify standalone computeRopePhasorTelemetry agrees on all 4 planes
      const trace = postCtx.resolveLayerTrace(3);
      const ckpt = post16Rec.primaryPassCheckpoint;
      const keyPosIds = Int32Array.from({ length: trace.attention.kvLen }, (_, k) => k);
      const t0 = computeRopePhasorTelemetry(trace, DIFFUSION_MINI_CONFIG, {
        layerIndex: 3,
        qHeadIndex: 5,
        kvHeadIndex: 1,
        rotaryPlane: 2,
        queryRow: 0,
        keyCol: trace.attention.kvLen - 1,
        positionIds: ckpt.positionIds,
        cachePositions: ckpt.cachePositions,
        keyPositionIds: keyPosIds,
      });
      expect(t0.allPlaneInvFreqs).toEqual([1.0, 0.3125, 0.0625, 0.03125]);
      expect(t0.allPlaneZones).toEqual([
        'original',
        'blended',
        'interpolated',
        'interpolated',
      ]);
    });

    it('reflects nonzero o_proj LoRA delta when loraEnabled=true and exact zero/bypassed when loraEnabled=false', () => {
      // LoRA ON: linear_spec_lora spec_cycle
      const sOn = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: { mode: 'linear_spec_lora', blockLength: 4, maxNewTokens: 8, temperature: 0.7 },
      });
      sOn.step(); // prefill
      const specOnRec = sOn.step(); // spec_cycle
      const sceneOn = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(sOn, specOnRec.recordId),
        {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'gqa_rope',
          selectedLayerIndex: 0,
          activeStepperStep: 5,
        },
      );
      expect(sceneOn.diagnostics?.loraActive).toBe(true);
      expect(sceneOn.partCatalog.lora_delta?.badge).toContain('ACTIVE');
      expect(sceneOn.partCatalog.lora_a?.badge).toContain('ACTIVE');

      // Inspect actual lora_delta texels in 256×256 atlas (band3Y at x=24..31)
      const atlasOn = sceneOn.valueAtlas!;
      const SOn = Number(sceneOn.diagnostics?.queryTokens);
      // band3Y for S=4, K=12: yCursor = 20 + 4*4 = 36; band2Height = max(12, 8, 16) = 16 -> band3Y = 52
      let maxAbsDeltaOn = 0;
      const band3YOn = 20 + 4 * SOn + Math.max(Number(sceneOn.diagnostics?.keyTokens), 2 * SOn, 16);
      for (let r = 0; r < SOn; r++) {
        for (let c = 0; c < 8; c++) {
          const [v] = readAtlasTexel(atlasOn.data, atlasOn.width, 24 + c, band3YOn + r);
          maxAbsDeltaOn = Math.max(maxAbsDeltaOn, Math.abs(v));
        }
      }
      expect(maxAbsDeltaOn).toBeGreaterThan(1e-4);
      expect(validateSceneGeometryFinite(sceneOn).valid).toBe(true);

      // LoRA OFF: linear_spec spec_cycle
      const sOff = createDiffusionSession(model, {
        promptTokenIds: defaultPrompt,
        config: { mode: 'linear_spec', blockLength: 4, maxNewTokens: 8, temperature: 0 },
      });
      sOff.step();
      const specOffRec = sOff.step();
      const sceneOff = buildDiffusionSpatialScene(
        resolveDiffusionSessionContext(sOff, specOffRec.recordId),
        {
          ...DEFAULT_DIFFUSION_FOCUS_STATE,
          viewMode: 'gqa_rope',
          selectedLayerIndex: 0,
          activeStepperStep: 5,
        },
      );
      expect(sceneOff.diagnostics?.loraActive).toBe(false);
      expect(sceneOff.partCatalog.lora_delta?.badge).toContain('BYPASSED');

      const atlasOff = sceneOff.valueAtlas!;
      const SOff = Number(sceneOff.diagnostics?.queryTokens);
      const band3YOff = 20 + 4 * SOff + Math.max(Number(sceneOff.diagnostics?.keyTokens), 2 * SOff, 16);
      let maxAbsDeltaOff = 0;
      for (let r = 0; r < SOff; r++) {
        for (let c = 0; c < 8; c++) {
          const [v] = readAtlasTexel(atlasOff.data, atlasOff.width, 24 + c, band3YOff + r);
          maxAbsDeltaOff = Math.max(maxAbsDeltaOff, Math.abs(v));
        }
      }
      expect(maxAbsDeltaOff).toBe(0);
      expect(validateSceneGeometryFinite(sceneOff).valid).toBe(true);
    });
  });
});
