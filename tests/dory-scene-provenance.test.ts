/**
 * DORY3 Step-0 Spatial Foundation & Provenance Guard Contract Suite
 * (`tests/dory-scene-provenance.test.ts`)
 *
 * Verifies the frozen `src/dory/scene/types.ts` and `src/dory/scene/provenance.ts` contracts:
 *   1. All 7 `D7` view modes & authoritative epistemic badges (`DR7`, `DR23`, `Finding 9`).
 *   2. `DR20` hard throw in BOTH `DorySheetBufferBuilder.push()` AND `validateSceneGeometryFinite()`
 *      when `sheetKind in {1, 2}` lacks a live `RG32F` `G`-channel mask `TexRect`.
 *   3. `DR21` `encodeDoryPickMetadata` + `pickHitToDoryCell` round-trip encoding loop, KV head
 *      (including arbitrary KV head indices > 1 without touching `src/render/pick.ts`), KV bank,
 *      rotary plane, speculative offset, and stage metadata via `groupIndex`/`tapIndex`/`semanticTag`.
 *   4. `DoryAtlasPacker`, `auditAtlasRegions` (0 overlaps, out-of-bounds, `<= 512 KiB` cap),
 *      and deterministic `computeDoryAtlasContentKey`.
 *   5. Non-mutating `resolveDorySessionContext` across both standard and `self_spec` sessions,
 *      verifying harmonized `verifyTrace.executedSublayers[0..191]` lookup and zero KV mutation.
 *   6. `D4` static boundary isolation & zero `Math.random()` across `src/dory/scene/**`.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import type { PickHit } from '../src/render/pick';
import { makeSheetFrame3D } from '../src/render/sheetFrame';
import {
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
} from '../src/render/sheets';
import {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_MAX_ATLAS_BYTES,
  DORY_NLERP_MARKER,
  DORY_SPEC_CONTRACT_A_BADGE,
  createDorySession,
} from '../src/dory';
import {
  DEFAULT_DORY_SPATIAL_FOCUS_STATE,
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_SPATIAL_VIEW_MODES,
  DORY_TBPTL_READONLY_BADGE,
  type DorySpatialScene,
} from '../src/dory/scene/types';
import {
  DoryAtlasPacker,
  DoryDr20CausalMaskError,
  DoryFlatFlowBufferBuilder,
  DoryLineBufferBuilder,
  DoryRibbonBufferBuilder,
  DorySheetBufferBuilder,
  DorySolidBufferBuilder,
  auditAtlasRegions,
  computeDoryAtlasContentKey,
  encodeDoryPickMetadata,
  ensureDorySessionContext,
  packPickBoxFromAabb,
  packPickBoxFromFrame,
  pickHitToDoryCell,
  resolveDorySessionContext,
  validateSceneGeometryFinite,
} from '../src/dory/scene/provenance';

function makeMinimalValidScene(overrides: Partial<DorySpatialScene> = {}): DorySpatialScene {
  const solids = new DorySolidBufferBuilder();
  solids.push({ cx: 0, cy: 0, cz: 0, ex: 1, ey: 1, ez: 1, bayId: 1 });
  const solidPack = solids.pack();

  const lines = new DoryLineBufferBuilder();
  lines.addBoxEdges(0, 0, 0, 1, 1, 1, 1);
  const linePack = lines.pack();

  const ribbons = new DoryRibbonBufferBuilder();
  ribbons.push({ cx: 0, cy: 0.6, cz: 0, ex: 0.4, ey: 0.1, ez: 0.4 });
  const ribbonPack = ribbons.pack();

  const flow = new DoryFlatFlowBufferBuilder();
  flow.addPolylineXZ(
    [
      { x: 0, z: 0 },
      { x: 2, z: 0 },
    ],
    0.05,
    0.12,
    { r: 0.2, g: 0.4, b: 0.7, a: 0.8 },
  );
  const flowPack = flow.pack();

  const packer = new DoryAtlasPacker(32, 32);
  const scores = new Float32Array([1.0, -Infinity, 0.5, 0.8]);
  const maskVisible = new Uint8Array([1, 0, 1, 1]);
  const scoreSlice = packer.packCausalMatrix('scores_h0', scores, maskVisible, 2, 2, 0);
  const atlasBuilt = packer.build('gqa_rope', 'unit');

  const sheets = new DorySheetBufferBuilder();
  sheets.push({
    cx: 0,
    cy: 0.2,
    cz: 0,
    ex: 1,
    ey: 0.06,
    ez: 1,
    gridCols: 2,
    gridRows: 2,
    sheetKind: SHEET_KIND_CAUSAL_SCORES,
    role: SHEET_ROLE_SCORES,
    slice: scoreSlice,
  });
  const sheetPack = sheets.pack();

  const fitBounds = { min: [-2, -1, -2] as const, max: [3, 2, 2] as const };

  return {
    viewMode: 'gqa_rope',
    title: 'Test Scene',
    subtitle: 'Unit test',
    modelWeightId: 'test-weight-id',
    branchId: 'b0',
    recordId: 'r0',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    physicalLayerIdx: 16,
    loopIteration: 1,
    solidData: solidPack.data,
    solidCount: solidPack.count,
    lineData: linePack.data,
    lineCount: linePack.count,
    ribbonData: ribbonPack.data,
    ribbonCount: ribbonPack.count,
    sheetData: sheetPack.sheetData,
    sheetBasisData: sheetPack.sheetBasisData,
    sheetRevealData: sheetPack.sheetRevealData,
    sheetCount: sheetPack.sheetCount,
    flowVertices: flowPack.flowVertices,
    flowIndices: flowPack.flowIndices,
    pickBoxes: [
      packPickBoxFromAabb(
        0,
        0.2,
        0,
        1,
        0.06,
        1,
        encodeDoryPickMetadata({ physicalLayerIdx: 16, loopIteration: 1, kvHeadIndex: 0 }),
      ),
    ],
    cameraPreset: { azimuthDeg: -24, elevationDeg: 30, fitBounds },
    fitBounds,
    valueAtlas: atlasBuilt.valueAtlas,
    atlasKey: atlasBuilt.atlasKey,
    atlasRegions: atlasBuilt.atlasRegions,
    atlasAudit: atlasBuilt.atlasAudit,
    provenanceBadges: [],
    callouts: [],
    overlayAnchors: [],
    landmarks: [],
    kvDepthTicks: [],
    calipers: [],
    activeCellGrids: [],
    activeCellPopover: null,
    ...overrides,
  };
}

describe('DORY3 Step 0 — Frozen Spatial Types & Provenance Guards (DR20, DR21, D4)', () => {
  it('freezes all 7 D7 view modes, budgets <= 512 KiB, and authoritative epistemic badges', () => {
    expect(DORY_SPATIAL_VIEW_MODES.map((v) => v.id)).toEqual([
      'macro',
      'hypersphere',
      'gqa_rope',
      'swiglu',
      'kv_compare',
      'self_spec',
      'bookends',
    ]);
    for (let i = 0; i < DORY_SPATIAL_VIEW_MODES.length; i += 1) {
      const desc = DORY_SPATIAL_VIEW_MODES[i];
      expect(desc.viewNumber).toBe(i + 1);
      expect(desc.atlasBudgetBytes).toBeLessThanOrEqual(DORY_MAX_ATLAS_BYTES);
    }
    expect(DEFAULT_DORY_SPATIAL_FOCUS_STATE.selectedPhysicalLayerIdx).toBe(16);
    expect(DEFAULT_DORY_SPATIAL_FOCUS_STATE.selectedKeyCol).toBeNull();
    expect(DEFAULT_DORY_SPATIAL_FOCUS_STATE.selectedRotaryPlane).toBe(0);
    expect(DEFAULT_DORY_SPATIAL_FOCUS_STATE.swigluOutputDim).toBe(0);
    expect(DEFAULT_DORY_SPATIAL_FOCUS_STATE.kvCompareLayerIdx).toBe(16);
    expect(DORY_ANTI_SLERP_PLAQUE).toContain('NOT Trigonometric SLERP');
    expect(DORY_NLERP_MARKER).toBe('dory-nlerp-residual');
    expect(DORY_EXPLORATORY_PCA_BADGE).toContain('8D');
    expect(DORY_ANCHOR_MAP_CAVEAT_BADGE).toContain('120 deg');
    expect(DORY_TBPTL_READONLY_BADGE).toBe('READ-ONLY · NO GRADIENTS');
    expect(DORY_SPEC_CONTRACT_A_BADGE).toContain('Contract A');
    expect(DORY_COUNTERFACTUAL_CODA_BADGE).toContain('DIAGNOSTIC COUNTERFACTUAL');
  });

  it('enforces DR20 hard throw in BOTH DorySheetBufferBuilder.push() and validateSceneGeometryFinite()', () => {
    const builder = new DorySheetBufferBuilder();

    // Standard sheet without live texture is permitted (structural calm fill)
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

    // sheetKind = 1 (CAUSAL_SCORES) without live TexRect MUST hard-throw DR20 error
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
        sheetKind: SHEET_KIND_CAUSAL_SCORES,
        role: SHEET_ROLE_SCORES,
      }),
    ).toThrow(DoryDr20CausalMaskError);

    // sheetKind = 2 (CAUSAL_SOFTMAX) with zero-area TexRect MUST hard-throw DR20 error
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
        sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
        role: SHEET_ROLE_SOFTMAX,
        texRect: [0, 0, 0, 0.5],
      }),
    ).toThrow(/DR20/);

    // Valid scene with live RG32F G-channel causal mask passes cleanly
    const validScene = makeMinimalValidScene();
    expect(validateSceneGeometryFinite(validScene)).toEqual({ valid: true, errors: [] });

    // Tampered scene with valueAtlas = null while sheetKind = 1 MUST hard-throw DR20 in validateSceneGeometryFinite
    const nullAtlasScene = makeMinimalValidScene({ valueAtlas: null });
    expect(() => validateSceneGeometryFinite(nullAtlasScene)).toThrow(DoryDr20CausalMaskError);
    expect(validateSceneGeometryFinite(nullAtlasScene, { throwOnDr20: false }).valid).toBe(false);

    // Tampered scene with zeroed TexRect inside sheetData MUST hard-throw DR20
    const badSheetData = new Float32Array(validScene.sheetData);
    badSheetData[15] = 0; // uWidth = 0
    const badRectScene = makeMinimalValidScene({ sheetData: badSheetData });
    expect(() => validateSceneGeometryFinite(badRectScene)).toThrow(DoryDr20CausalMaskError);

    // Non-finite geometry float is reported cleanly
    const nanSolidData = new Float32Array(validScene.solidData);
    nanSolidData[0] = Number.NaN;
    const nanValidation = validateSceneGeometryFinite(
      makeMinimalValidScene({ solidData: nanSolidData }),
    );
    expect(nanValidation.valid).toBe(false);
    expect(nanValidation.errors.some((e) => e.includes('Non-finite'))).toBe(true);
  });

  it('validates finiteness of pickBoxes[i].frame, activeCellGrids[i], activeCellPopover, and callouts[i].worldPos', () => {
    // 1. Valid oriented PickBox frame + masked causal score grid (-Infinity only where masked=1) + valid popover
    const goodFrame = makeSheetFrame3D([0, 0.2, 0], [1, 0.06, 1]);
    const validRichScene = makeMinimalValidScene({
      pickBoxes: [
        packPickBoxFromFrame(
          goodFrame,
          encodeDoryPickMetadata({ physicalLayerIdx: 16, loopIteration: 1, kvHeadIndex: 0 }),
        ),
      ],
      activeCellGrids: [
        {
          originWorld: goodFrame.center,
          extent: goodFrame.extent,
          axisU: goodFrame.axisU,
          axisN: goodFrame.axisN,
          cols: 2,
          rows: 2,
          values: new Float32Array([1.0, -Infinity, 0.5, 0.8]),
          masked: new Uint8Array([0, 1, 0, 0]),
          format: 'signed',
          activeCol: 0,
          activeRow: 0,
        },
      ],
      activeCellPopover: {
        world: [0.1, 0.25, 0.1],
        tensorLabel: 'Scores',
        coordLabel: 'q=0 · k=0',
        equation: 'q·k / sqrt(d_h)',
        valueText: '= +1.000',
        sourceWorlds: [{ world: [-0.2, 0.2, 0.0], role: 'indigo' }],
      },
      callouts: [
        {
          id: 'c0',
          title: 'Causal Plate',
          body: 'Live G-channel mask',
          tone: 'swa_cyan',
          worldPos: [0, 0.5, 0],
        },
      ],
    });
    expect(validateSceneGeometryFinite(validRichScene)).toEqual({ valid: true, errors: [] });

    // 2. Non-finite PickBox frame vector is rejected
    const badFrameRes = validateSceneGeometryFinite(
      makeMinimalValidScene({
        pickBoxes: [
          {
            ...packPickBoxFromFrame(
              goodFrame,
              encodeDoryPickMetadata({ physicalLayerIdx: 16, loopIteration: 1 }),
            ),
            frame: { ...goodFrame, axisN: [0, Number.NaN, 0] },
          },
        ],
      }),
    );
    expect(badFrameRes.valid).toBe(false);
    expect(badFrameRes.errors.some((e) => e.includes('pickBoxes[0].frame.axisN'))).toBe(true);

    // 3. Unmasked -Infinity, masked NaN, masked +Infinity, and non-finite grid coordinates are rejected
    const unmaskedNegInf = validateSceneGeometryFinite(
      makeMinimalValidScene({
        activeCellGrids: [
          {
            originWorld: [0, 0, 0],
            extent: [1, 0.06, 1],
            axisU: [1, 0, 0],
            axisN: [0, 1, 0],
            cols: 2,
            rows: 1,
            values: new Float32Array([1.0, -Infinity]),
          },
        ],
      }),
    );
    expect(unmaskedNegInf.valid).toBe(false);
    expect(unmaskedNegInf.errors.some((e) => e.includes('activeCellGrids[0].values[1]'))).toBe(true);

    const maskedNan = validateSceneGeometryFinite(
      makeMinimalValidScene({
        activeCellGrids: [
          {
            originWorld: [0, 0, 0],
            extent: [1, 0.06, 1],
            axisU: [1, 0, 0],
            axisN: [0, 1, 0],
            cols: 2,
            rows: 1,
            values: new Float32Array([1.0, Number.NaN]),
            masked: new Uint8Array([0, 1]),
          },
        ],
      }),
    );
    expect(maskedNan.valid).toBe(false);

    const badGridCoord = validateSceneGeometryFinite(
      makeMinimalValidScene({
        activeCellGrids: [
          {
            originWorld: [Number.POSITIVE_INFINITY, 0, 0],
            extent: [1, 0.06, 1],
            axisU: [1, 0, 0],
            axisN: [0, 1, 0],
            cols: 1,
            rows: 1,
            values: new Float32Array([0.5]),
          },
        ],
      }),
    );
    expect(badGridCoord.valid).toBe(false);
    expect(badGridCoord.errors.some((e) => e.includes('activeCellGrids[0].originWorld'))).toBe(true);

    // 4. Non-finite activeCellPopover.world, sourceWorlds, and callout.worldPos are rejected
    const badPopoverWorld = validateSceneGeometryFinite(
      makeMinimalValidScene({
        activeCellPopover: {
          world: [0, Number.NaN, 0],
          tensorLabel: 'Bad',
          coordLabel: 'r0',
          equation: 'x',
          valueText: '0',
        },
      }),
    );
    expect(badPopoverWorld.valid).toBe(false);
    expect(badPopoverWorld.errors.some((e) => e.includes('activeCellPopover.world'))).toBe(true);

    const badPopoverSource = validateSceneGeometryFinite(
      makeMinimalValidScene({
        activeCellPopover: {
          world: [0, 0, 0],
          tensorLabel: 'BadSource',
          coordLabel: 'r0',
          equation: 'x',
          valueText: '0',
          sourceWorlds: [{ world: [0, Number.NEGATIVE_INFINITY, 0], role: 'teal' }],
        },
      }),
    );
    expect(badPopoverSource.valid).toBe(false);
    expect(badPopoverSource.errors.some((e) => e.includes('sourceWorlds[0]'))).toBe(true);

    const badCalloutWorld = validateSceneGeometryFinite(
      makeMinimalValidScene({
        callouts: [
          {
            id: 'bad_co',
            title: 'Bad',
            body: 'Bad',
            tone: 'warn',
            worldPos: [Number.NaN, 0, 0],
          },
        ],
      }),
    );
    expect(badCalloutWorld.valid).toBe(false);
    expect(badCalloutWorld.errors.some((e) => e.includes('callouts[0].worldPos'))).toBe(true);
  });

  it('round-trips DR21 metadata via groupIndex/tapIndex/semanticTag without touching src/render/pick.ts', () => {
    // 1. Recurrent attention sheet with arbitrary KV head (e.g. prod KV head 3), loop 4, rotary plane 2
    const encoded = encodeDoryPickMetadata({
      physicalLayerIdx: 22, // Full attention in recurrent core (paired 11)
      executedLayerIdx: 142,
      loopIteration: 4,
      partId: 'rec_full_l22_t4',
      tensorKey: 'q_rot',
      qHeadIndex: 6,
      kvHeadIndex: 3, // exceeds Nano's 0 | 1 typing — carried cleanly via groupIndex + semanticTag!
      rotaryPlane: 2,
      kvBankSlotId: 79,
      kvBankKind: 'recurrent',
      kvPolicy: 'per_loop',
      cacheRole: 'spec_verify_deep',
      tokenIndex: 3,
      absTokenPos: 11,
      modelWeightId: 'sha256-deadbeef',
    });

    expect(encoded.bayId).toBe(23);
    expect(encoded.kvHeadIndex).toBeUndefined(); // never violates 0 | 1 narrowing!
    expect(encoded.groupIndex).toBe(3);
    expect(encoded.tapIndex).toBe(2);
    expect(encoded.semanticTag).toContain('stage=recurrent');
    expect(encoded.semanticTag).toContain('loop=4');
    expect(encoded.semanticTag).toContain('kv=3');
    expect(encoded.semanticTag).toContain('bank=79');

    const hit: PickHit = {
      ...encoded,
      hitRow: 3,
      hitCol: 50,
      t: 4.5,
      point: [0, 0, 0],
      isFascia: false,
    };
    const decoded = pickHitToDoryCell(hit, 'gqa_rope');
    expect(decoded).not.toBeNull();
    expect(decoded!.physicalLayerIdx).toBe(22);
    expect(decoded!.pairedLayerIdx).toBe(11);
    expect(decoded!.executedLayerIdx).toBe(142);
    expect(decoded!.stage).toBe('recurrent');
    expect(decoded!.sublayerKind).toBe('full_attention');
    expect(decoded!.loopIteration).toBe(4);
    expect(decoded!.qHeadIndex).toBe(6);
    expect(decoded!.kvHeadIndex).toBe(3);
    expect(decoded!.rotaryPlane).toBe(2);
    expect(decoded!.kvBankSlotId).toBe(79);
    expect(decoded!.kvBankKind).toBe('recurrent');
    expect(decoded!.kvPolicy).toBe('per_loop');
    expect(decoded!.cacheRole).toBe('spec_verify_deep');
    expect(decoded!.absTokenPos).toBe(11);
    expect(decoded!.modelWeightId).toBe('sha256-deadbeef');

    // 2. Contract A output_shallow KV bank rail slab via groupRole='bank' & tapRole='spec'
    const bankMeta = encodeDoryPickMetadata({
      physicalLayerIdx: 56,
      loopIteration: 0,
      kvBankSlotId: 96,
      kvBankKind: 'output_shallow',
      kvPolicy: 'per_loop',
      cacheRole: 'spec_draft',
      specOffset: 2,
      groupRole: 'bank',
      tapRole: 'spec',
    });
    const bankDecoded = pickHitToDoryCell(
      { ...bankMeta, t: 1.0, point: [0, 0, 0], isFascia: false },
      'self_spec',
    )!;
    expect(bankDecoded.stage).toBe('output');
    expect(bankDecoded.kvBankSlotId).toBe(96);
    expect(bankDecoded.kvBankKind).toBe('output_shallow');
    expect(bankDecoded.specOffset).toBe(2);
    expect(bankDecoded.cacheRole).toBe('spec_draft');

    // 3. Raw manual PickHit (no encoder) decodes bayId, groupIndex, tapIndex cleanly
    const rawDecoded = pickHitToDoryCell(
      {
        bayId: 17, // physical 16 (first recurrent SWA)
        groupIndex: 1,
        tapIndex: 3,
        tensorKey: 'scores',
        t: 2.0,
        point: [0, 0, 0],
        isFascia: false,
      },
      'gqa_rope',
    )!;
    expect(rawDecoded.physicalLayerIdx).toBe(16);
    expect(rawDecoded.stage).toBe('recurrent');
    expect(rawDecoded.kvHeadIndex).toBe(1);
    expect(rawDecoded.loopIteration).toBe(3);
    expect(pickHitToDoryCell(null, 'macro')).toBeNull();
  });

  it('audits atlas overlaps, bounds, <= 512 KiB budget, and deterministic content keys', () => {
    const overlapAudit = auditAtlasRegions(64, 64, [
      { id: 'a', x0: 0, y0: 0, cols: 16, rows: 16, mode: 0, isMaskOrScore: false },
      { id: 'b', x0: 8, y0: 8, cols: 16, rows: 16, mode: 0, isMaskOrScore: false },
    ]);
    expect(overlapAudit.hasOverlap).toBe(true);
    expect(overlapAudit.overlapPair).toEqual(['a', 'b']);

    const oobAudit = auditAtlasRegions(32, 32, [
      { id: 'oob', x0: 24, y0: 0, cols: 16, rows: 8, mode: 0, isMaskOrScore: false },
    ]);
    expect(oobAudit.outOfBoundsIds).toEqual(['oob']);

    // Exceeding 512 KiB RG32F cap throws at DoryAtlasPacker construction
    expect(() => new DoryAtlasPacker(512, 256)).toThrow(RangeError); // 512*256*8 = 1 MiB > 512 KiB

    const p1 = new DoryAtlasPacker(64, 32);
    p1.packMatrix('x', new Float32Array([0.25, -0.5, 0.75, 1.0]), 2, 2);
    const b1 = p1.build('hypersphere', 'k2');
    const p2 = new DoryAtlasPacker(64, 32);
    p2.packMatrix('x', new Float32Array([0.25, -0.5, 0.75, 1.0]), 2, 2);
    const b2 = p2.build('hypersphere', 'k2');
    expect(b1.atlasAudit.hasOverlap).toBe(false);
    expect(b1.atlasAudit.withinBudget).toBe(true);
    expect(b1.atlasKey).toBe(b2.atlasKey);
    expect(computeDoryAtlasContentKey('hypersphere', b1.valueAtlas, 'k2')).toBe(b1.atlasKey);
  });

  it('resolves non-mutating DorySession contexts across standard and Contract A self_spec records', () => {
    const session = createDorySession({
      initialBranch: {
        promptTokens: [1, 10, 4, 18, 40, 25, 26, 57],
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: 4,
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const cycRec = session.step();
    const memBefore = session.measureMemory();

    const ctx = resolveDorySessionContext(session, cycRec.recordId);
    expect(ensureDorySessionContext(ctx, cycRec.recordId)).toBe(ctx);
    expect(ensureDorySessionContext(session, cycRec.recordId).record.recordId).toBe(cycRec.recordId);
    const firstRecId = session.getActiveBranch()!.recordIds[0]!;
    expect(ensureDorySessionContext(ctx, firstRecId).record.recordId).toBe(firstRecId);
    expect(ctx.executionMode).toBe('self_spec');
    expect(ctx.kLoops).toBe(4);
    expect(ctx.provenanceBadges.some((b) => b.label === DORY_SPEC_CONTRACT_A_BADGE)).toBe(true);

    const trace = ctx.resolveForwardTrace();
    expect(trace.executedSublayers.length).toBe(192);
    // Harmonized Prelude (p=0), Loop-1 (p=16, t=1), Loop-4 (p=16, t=4), and Coda (p=56) share identical seqLen!
    const pre0 = ctx.resolveSublayerRecord(0, 0);
    const rec16L1 = ctx.resolveSublayerRecord(16, 1);
    const rec16L4 = ctx.resolveSublayerRecord(16, 4);
    const coda56 = ctx.resolveSublayerRecord(56, 0);
    expect(pre0.trace.seqLen).toBe(trace.seqLen);
    expect(rec16L1.trace.seqLen).toBe(trace.seqLen);
    expect(rec16L4.trace.seqLen).toBe(trace.seqLen);
    expect(coda56.trace.seqLen).toBe(trace.seqLen);

    const defaultOff = ctx.resolveDefaultTokenOffset();
    expect(defaultOff).toBeGreaterThanOrEqual(0);
    expect(defaultOff).toBeLessThan(trace.seqLen);

    const cf1 = ctx.resolveCounterfactualCoda(1);
    expect(cf1.badge).toBe(DORY_COUNTERFACTUAL_CODA_BADGE);
    expect(session.measureMemory()).toEqual(memBefore);
  });

  it('enforces D4 zero Gate 4b/4c/sibling imports and zero Math.random() in src/dory/scene/*', () => {
    const forbiddenImportPatterns = [
      /from\s+['"][^'"]*(?:ui\/(?:controls|overlay|memoryChart)|scene\/(?:spine|sheetInspector|hybridPick|cutaways|embeddingCutaway|outputCutaway|gqaCutaway|mambaCutaway|moeCutaway|gqaSpatial|mambaSpatial|moeSpatial|accumulation)|model\/(?:spec|memory|state|rcPlayback)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)|story\/walkthrough|src\/main|\.\.\/main)['"]/,
      /from\s+['"][^'"]*(?:twotower|diffusion|uni)(?:\/|['"])/,
    ];
    for (const relPath of ['src/dory/scene/types.ts', 'src/dory/scene/provenance.ts']) {
      const src = readFileSync(resolve(process.cwd(), relPath), 'utf-8');
      expect(src, `Math.random forbidden in ${relPath}`).not.toMatch(/Math\.random\s*\(/);
      for (const re of forbiddenImportPatterns) {
        expect(src, `D4 forbidden import in ${relPath}`).not.toMatch(re);
      }
    }
  });
});
