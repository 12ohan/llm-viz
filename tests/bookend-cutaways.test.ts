import { describe, expect, it } from 'vitest';

import {
  createDefaultHybridModel,
  defaultTokenIdsForLength,
  runHybridExecution,
} from '../src/math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeStep,
  runIncrementalDecodeStep,
  stepDecodeSessionGreedy,
} from '../src/math/incrementalDecode';
import {
  aabbCorners,
  buildCameraSnapshot,
  makeViewportMetrics,
  projectWorldToInnerCss,
  type AABB,
} from '../src/render/camera';
import { pickMacroBay, type Ray } from '../src/render/pick';
import { sheetFrameWorldAabb } from '../src/render/sheetFrame';
import {
  SHEET_BASIS_FLOATS,
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_GHOST_RAIL,
} from '../src/render/sheets';
import {
  buildEmbeddingCutaway,
  describeEmbeddingCellFormula,
  EMBEDDING_SPATIAL_FIT_BOUNDS,
  fitSpatialEmbeddingCamera,
  pickHitToEmbeddingCell,
  type EmbeddingCellRef,
} from '../src/scene/embeddingCutaway';
import {
  buildOutputCutaway,
  describeOutputCellFormula,
  fitSpatialOutputCamera,
  OUTPUT_SPATIAL_FIT_BOUNDS,
  parseOutputCandidatePartId,
  pickHitToOutputCell,
  type OutputCellRef,
} from '../src/scene/outputCutaway';

function sampleAtlasTexel(
  atlas: { width: number; height: number; data: Float32Array },
  sheetData: Float32Array,
  sheetIdx: number,
  col: number,
  row: number,
): { r: number; g: number } {
  const off = sheetIdx * SHEET_INSTANCE_FLOATS;
  const cols = sheetData[off + 6];
  const rows = sheetData[off + 7];
  const u0 = sheetData[off + 13];
  const v0 = sheetData[off + 14];
  const uw = sheetData[off + 15];
  const vh = sheetData[off + 16];
  const u = u0 + ((col + 0.5) / cols) * uw;
  const v = v0 + ((row + 0.5) / rows) * vh;
  const tx = Math.floor(u * atlas.width);
  const ty = Math.floor(v * atlas.height);
  const idx = (ty * atlas.width + tx) * 2;
  return { r: atlas.data[idx], g: atlas.data[idx + 1] };
}

function assertOrthonormalRightHandedBases(
  sheetBasisData: Float32Array,
  sheetCount: number,
): { hasXZ: boolean; hasYZ: boolean; hasXY: boolean } {
  expect(sheetBasisData.length).toBe(sheetCount * SHEET_BASIS_FLOATS);
  let hasXZ = false;
  let hasYZ = false;
  let hasXY = false;

  for (let i = 0; i < sheetCount; i++) {
    const o = i * SHEET_BASIS_FLOATS;
    const ux = sheetBasisData[o], uy = sheetBasisData[o + 1], uz = sheetBasisData[o + 2];
    const nx = sheetBasisData[o + 3], ny = sheetBasisData[o + 4], nz = sheetBasisData[o + 5];

    const uLen = Math.hypot(ux, uy, uz);
    const nLen = Math.hypot(nx, ny, nz);
    const dot = ux * nx + uy * ny + uz * nz;
    expect(uLen).toBeCloseTo(1, 5);
    expect(nLen).toBeCloseTo(1, 5);
    expect(Math.abs(dot)).toBeLessThan(1e-5);

    // axisV = cross(axisU, axisN), det([U, N, V]) = +1 (right-handed orthonormal)
    const vx = uy * nz - uz * ny;
    const vy = uz * nx - ux * nz;
    const vz = ux * ny - uy * nx;
    const vLen = Math.hypot(vx, vy, vz);
    expect(vLen).toBeCloseTo(1, 5);
    const det =
      ux * (ny * vz - nz * vy) -
      uy * (nx * vz - nz * vx) +
      uz * (nx * vy - ny * vx);
    expect(det).toBeCloseTo(1, 5);

    if (Math.abs(ny) > 0.9) hasXZ = true;
    if (Math.abs(nx) > 0.9) hasYZ = true;
    if (Math.abs(nz) > 0.9) hasXY = true;
  }

  return { hasXZ, hasYZ, hasXY };
}

function aabbVolume(b: AABB): number {
  return (b.max[0] - b.min[0]) * (b.max[1] - b.min[1]) * (b.max[2] - b.min[2]);
}

function assertOpFocusBoundsValid(
  opFocusBounds: Readonly<Record<number, AABB>> | undefined,
  expectedKeys: readonly number[],
  outerBounds: AABB,
): void {
  expect(opFocusBounds).toBeDefined();
  const outerVol = aabbVolume(outerBounds);

  for (const k of expectedKeys) {
    const b = opFocusBounds![k];
    expect(b).toBeDefined();
    for (let axis = 0; axis < 3; axis++) {
      expect(Number.isFinite(b.min[axis])).toBe(true);
      expect(Number.isFinite(b.max[axis])).toBe(true);
      expect(b.max[axis]).toBeGreaterThan(b.min[axis]);
      expect(b.min[axis]).toBeGreaterThanOrEqual(outerBounds.min[axis] - 1e-4);
      expect(b.max[axis]).toBeLessThanOrEqual(outerBounds.max[axis] + 1e-4);
    }
    expect(aabbVolume(b)).toBeLessThan(outerVol);
  }
}

describe('V4 Bookend Cutaways — Dedicated 3D Thin-Sheet Embedding (NoPE) & Output / Untied LM Head', () => {
  const model = createDefaultHybridModel();
  const fullExec = runHybridExecution(model, {
    tokenIds: defaultTokenIdsForLength(8),
  });

  // Prefill 4 tokens [0, 1, 2, 3] then take 1 cached AR decode step (S=1, startTokenOffset=4)
  const prefillSession = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
  const prefillStep = getActiveDecodeStep(prefillSession);
  const arStep = runIncrementalDecodeStep(model, {
    newTokenIds: [prefillStep.greedyNextToken.tokenId],
    prevState: prefillStep.nextState,
    branchId: 'branch-0',
    parentStepExecutionId: prefillStep.stepExecutionId,
    initialStateDigest: prefillStep.provenance.initialStateDigest,
    stepIndex: 1,
  });

  it('builds Embedding (NoPE) cutaway for Full Seq (T=8) with orthonormal 3D frames, NoPE ghost rail, and bit-exact RG32F Float32 values', () => {
    const scene = buildEmbeddingCutaway({
      model,
      embedding: { X: fullExec.X_embed },
      tokenIds: fullExec.tokenIds,
      tokenLabels: fullExec.tokenLabels,
      startTokenOffset: 0,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 7,
    });

    expect(scene.kind).toBe('cutaway_embedding');
    expect(scene.fitBounds).toEqual(EMBEDDING_SPATIAL_FIT_BOUNDS);
    expect(scene.sheetCount).toBeGreaterThanOrEqual(5);
    expect(scene.solidPickBoxes.length).toBe(scene.sheetCount);
    expect(scene.valueAtlas).toBeDefined();
    expect((scene.flowVertices?.length ?? 0)).toBeGreaterThan(100);
    expect((scene.flowIndices?.length ?? 0)).toBeGreaterThan(60);

    // Orthonormal right-handed 3D sheet frames (det = +1) spanning both XZ decks and upright YZ W_embed wall
    const planes = assertOrthonormalRightHandedBases(scene.sheetBasisData!, scene.sheetCount!);
    expect(planes.hasXZ).toBe(true);
    expect(planes.hasYZ).toBe(true);

    // Verify NoPE Ghost Rail exists with SHEET_KIND_GHOST_RAIL and explicit NoPE landmark
    const sd = scene.sheetData!;
    const nopeIdx = scene.solidPickBoxes.findIndex((b) => b.partId === 'nope-slot');
    expect(nopeIdx).toBeGreaterThanOrEqual(0);
    expect(sd[nopeIdx * SHEET_INSTANCE_FLOATS + 8]).toBe(SHEET_KIND_GHOST_RAIL);
    const nopeLandmark = scene.landmarks.find((l) => l.id === 'emb-nope');
    expect(nopeLandmark?.label).toContain('No Pos Embed (NoPE)');
    expect(nopeLandmark?.label).toContain('Bay 01 Mamba-2');

    // Verify bit-exact RG32F Float32 values for W_embed [32×8] and X₀ [8×8]
    const wIdx = scene.solidPickBoxes.findIndex((b) => b.tensorKey === 'w_embed');
    const xIdx = scene.solidPickBoxes.findIndex((b) => b.tensorKey === 'x_embed');
    const tokIdx = scene.solidPickBoxes.findIndex((b) => b.tensorKey === 'token_strip');
    expect(wIdx).toBeGreaterThanOrEqual(0);
    expect(xIdx).toBeGreaterThanOrEqual(0);
    expect(tokIdx).toBeGreaterThanOrEqual(0);

    expect(scene.solidPickBoxes[wIdx].gridCols).toBe(8);
    expect(scene.solidPickBoxes[wIdx].gridRows).toBe(32);
    expect(scene.solidPickBoxes[xIdx].gridCols).toBe(8);
    expect(scene.solidPickBoxes[xIdx].gridRows).toBe(8);
    expect(scene.solidPickBoxes[tokIdx].gridCols).toBe(1);
    expect(scene.solidPickBoxes[tokIdx].gridRows).toBe(8);

    const atlas = scene.valueAtlas!;
    for (let v = 0; v < 32; v++) {
      for (let ch = 0; ch < 8; ch++) {
        const tex = sampleAtlasTexel(atlas, sd, wIdx, ch, v);
        expect(Object.is(tex.r, model.W_embed[v * 8 + ch])).toBe(true);
        expect(tex.g).toBe(0);
      }
    }

    // X₀ in atlas matches embedding.X AND Bay 01 Mamba-2 input X bit-for-bit!
    const bay1X = fullExec.getMambaTrace(1).X;
    for (let ch = 0; ch < 8; ch++) {
      for (let t = 0; t < 8; t++) {
        const tex = sampleAtlasTexel(atlas, sd, xIdx, t, ch);
        expect(Object.is(tex.r, fullExec.X_embed[ch * 8 + t])).toBe(true);
        expect(Object.is(tex.r, bay1X[ch * 8 + t])).toBe(true);
        expect(tex.g).toBe(0);
      }
    }

    // All pick boxes fit cleanly inside invariant fitBounds
    for (const box of scene.solidPickBoxes) {
      const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
      expect(aabb.min[0]).toBeGreaterThanOrEqual(EMBEDDING_SPATIAL_FIT_BOUNDS.min[0] - 1e-4);
      expect(aabb.min[1]).toBeGreaterThanOrEqual(EMBEDDING_SPATIAL_FIT_BOUNDS.min[1] - 1e-4);
      expect(aabb.min[2]).toBeGreaterThanOrEqual(EMBEDDING_SPATIAL_FIT_BOUNDS.min[2] - 1e-4);
      expect(aabb.max[0]).toBeLessThanOrEqual(EMBEDDING_SPATIAL_FIT_BOUNDS.max[0] + 1e-4);
      expect(aabb.max[1]).toBeLessThanOrEqual(EMBEDDING_SPATIAL_FIT_BOUNDS.max[1] + 1e-4);
      expect(aabb.max[2]).toBeLessThanOrEqual(EMBEDDING_SPATIAL_FIT_BOUNDS.max[2] + 1e-4);
    }

    // Valid per-operation focus bounds
    assertOpFocusBoundsValid(scene.opFocusBounds, [0, 1, 2, 3], EMBEDDING_SPATIAL_FIT_BOUNDS);
  });

  it('supports AR Decode (S=1, startTokenOffset=4) and exact cell formula + operand highlighting in Embedding cutaway', () => {
    expect(arStep.tokens).toBe(1);
    expect(arStep.prefixTokens).toBe(4);

    const activeCell: EmbeddingCellRef = {
      tensor: 'x_embed',
      row: 3,
      col: 0,
      tokenIndex: 0,
      vocabId: arStep.newTokenIds[0],
    };
    const opts = {
      model,
      embedding: { X: arStep.X_embed },
      tokenIds: arStep.newTokenIds,
      tokenLabels: arStep.newTokenLabels,
      startTokenOffset: arStep.prefixTokens,
      precision: 'fp8_selective' as const,
      selectedPart: null,
      hoveredPart: null,
      activeCell,
      focusToken: 0,
    };
    const scene = buildEmbeddingCutaway(opts);

    // Exact AR position t4 in formula & landmarks
    const formula = describeEmbeddingCellFormula(opts, activeCell);
    expect(formula.isMasked).toBe(false);
    expect(formula.coordLabel).toContain('t4');
    expect(formula.equation).toContain('W_embed');
    expect(formula.subEquation).toContain('NoPE');
    expect(Object.is(formula.value, arStep.X_embed[3])).toBe(true);
    expect(Object.is(formula.value, model.W_embed[arStep.newTokenIds[0] * 8 + 3])).toBe(true);
    expect(scene.activeFormula).toEqual(formula);

    // Operand-aware sheet highlighting: x_embed=ACTIVE, token_strip=OPERAND_A, w_embed=OPERAND_B, nope-slot=DIMMED
    const sd = scene.sheetData!;
    const idxOf = (part: string) => scene.solidPickBoxes.findIndex((b) => b.partId === part);
    expect(sd[idxOf('x-embed') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_ACTIVE);
    expect(sd[idxOf('token-strip') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_OPERAND_A);
    expect(sd[idxOf('w-embed') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_OPERAND_B);
    expect(sd[idxOf('nope-slot') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_DIMMED);

    // Formula checks for token_strip and w_embed cells
    const fTok = describeEmbeddingCellFormula(opts, { tensor: 'token_strip', row: 0, col: 0 });
    expect(fTok.value).toBe(arStep.newTokenIds[0]);
    expect(fTok.coordLabel).toContain('t4');
    expect(fTok.isMasked).toBe(false);

    const fW = describeEmbeddingCellFormula(opts, { tensor: 'w_embed', row: arStep.newTokenIds[0], col: 5 });
    expect(Object.is(fW.value, model.W_embed[arStep.newTokenIds[0] * 8 + 5])).toBe(true);
    expect(fW.headBadge).toContain('Untied');

    // Ray-pick round-trip onto X₀ sheet recovers exact EmbeddingCellRef
    const xBox = scene.solidPickBoxes.find((b) => b.tensorKey === 'x_embed')!;
    const f = xBox.frame!;
    const downRay: Ray = {
      origin: [f.center[0], f.center[1] + 2.0, f.center[2]],
      direction: [0, -1, 0],
    };
    const hit = pickMacroBay(downRay, [xBox], []);
    expect(hit).not.toBeNull();
    const picked = pickHitToEmbeddingCell(hit, arStep.newTokenIds);
    expect(picked?.tensor).toBe('x_embed');
    expect(picked?.tokenIndex).toBe(0);
    expect(picked?.vocabId).toBe(arStep.newTokenIds[0]);
  });

  it('builds Output & Untied LM Head cutaway for Full Seq (T=8) with orthonormal 3D frames, bit-exact RG32F values, probability palette, and elevated clickable Top-5 bars', () => {
    const bay52Y = fullExec.residualStreams[52];
    const focusToken = 7;
    const scene = buildOutputCutaway({
      model,
      residualIn: bay52Y,
      finalNorm: fullExec.finalNorm,
      lmHead: fullExec.lmHead,
      tokenLabels: fullExec.tokenLabels,
      startTokenOffset: 0,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken,
      isArDecode: false,
    });

    expect(scene.kind).toBe('cutaway_output');
    expect(scene.fitBounds).toEqual(OUTPUT_SPATIAL_FIT_BOUNDS);
    expect(scene.sheetCount).toBeGreaterThanOrEqual(13); // 8 pipeline sheets + 5 Top-5 bars
    expect(scene.solidPickBoxes.length).toBe(scene.sheetCount);
    expect(scene.valueAtlas).toBeDefined();
    expect((scene.flowVertices?.length ?? 0)).toBeGreaterThan(200);
    expect((scene.flowIndices?.length ?? 0)).toBeGreaterThan(100);

    // Orthonormal 3D bases spanning horizontal XZ decks, standing YZ W_lm_head wall, and upright XY Logits/Probs/Top-5
    const planes = assertOrthonormalRightHandedBases(scene.sheetBasisData!, scene.sheetCount!);
    expect(planes.hasXZ).toBe(true);
    expect(planes.hasYZ).toBe(true);
    expect(planes.hasXY).toBe(true);

    // Verify bit-exact Float32 values in RG32F atlas for residualIn, finalNorm.U_final, lmHead.logits, lmHead.probs
    const sd = scene.sheetData!;
    const atlas = scene.valueAtlas!;
    const resIdx = scene.solidPickBoxes.findIndex((b) => b.partId === 'residual-in');
    const normIdx = scene.solidPickBoxes.findIndex((b) => b.partId === 'final-norm');
    const wLmIdx = scene.solidPickBoxes.findIndex((b) => b.partId === 'w-lm-head');
    const logitsIdx = scene.solidPickBoxes.findIndex((b) => b.partId === 'logits');
    const probsIdx = scene.solidPickBoxes.findIndex((b) => b.partId === 'probs');
    expect(resIdx).toBeGreaterThanOrEqual(0);
    expect(normIdx).toBeGreaterThanOrEqual(0);
    expect(wLmIdx).toBeGreaterThanOrEqual(0);
    expect(logitsIdx).toBeGreaterThanOrEqual(0);
    expect(probsIdx).toBeGreaterThanOrEqual(0);

    // Probs uses probability palette (flags.x = 1)
    expect(sd[probsIdx * SHEET_INSTANCE_FLOATS + 18]).toBe(1);

    for (let t = 0; t < 8; t++) {
      for (let ch = 0; ch < 8; ch++) {
        expect(Object.is(sampleAtlasTexel(atlas, sd, resIdx, ch, t).r, bay52Y[ch * 8 + t])).toBe(true);
        expect(Object.is(sampleAtlasTexel(atlas, sd, normIdx, ch, t).r, fullExec.finalNorm.U_final[ch * 8 + t])).toBe(true);
      }
    }
    for (let v = 0; v < 32; v++) {
      expect(Object.is(sampleAtlasTexel(atlas, sd, logitsIdx, 0, v).r, fullExec.lmHead.logits[v * 8 + focusToken])).toBe(true);
      expect(Object.is(sampleAtlasTexel(atlas, sd, probsIdx, 0, v).r, fullExec.lmHead.probs[v * 8 + focusToken])).toBe(true);
      for (let ch = 0; ch < 8; ch++) {
        expect(Object.is(sampleAtlasTexel(atlas, sd, wLmIdx, ch, v).r, fullExec.lmHead.W_lm_head[v * 8 + ch])).toBe(true);
      }
    }

    // Top-5 candidate bars carry pickPart 'output-cand-<vocabId>', are visibly elevated in +Z, and use prob palette
    const top5 = fullExec.lmHead.topKByToken[focusToken];
    expect(top5).toHaveLength(5);
    const probsZ = scene.solidPickBoxes[probsIdx].frame!.center[2];
    for (const cand of top5) {
      const expectedPart = `output-cand-${cand.tokenId}`;
      expect(parseOutputCandidatePartId(expectedPart)).toBe(cand.tokenId);
      const cIdx = scene.solidPickBoxes.findIndex((b) => b.partId === expectedPart);
      expect(cIdx).toBeGreaterThanOrEqual(0);
      expect(scene.solidPickBoxes[cIdx].frame!.center[2]).toBeGreaterThan(probsZ + 0.10);
      expect(sd[cIdx * SHEET_INSTANCE_FLOATS + 18]).toBe(1);
      expect(Object.is(sampleAtlasTexel(atlas, sd, cIdx, 0, 0).r, fullExec.lmHead.probs[cand.tokenId * 8 + focusToken])).toBe(true);
    }

    // Explicit untied weight badge & synthetic miniature metadata landmarks
    expect(scene.landmarks.find((l) => l.id === 'out-w-lm')?.sublabel).toContain('Untied Weights ≠ W_embed');
    expect(scene.landmarks.find((l) => l.id === 'out-top5-header')?.sublabel).toContain('V=32 mini');
    expect(scene.landmarks.find((l) => l.id === 'out-top5-header')?.sublabel).toContain('131,072 prod');
    expect(scene.landmarks.find((l) => l.id === 'out-ar-loop')).toBeUndefined();

    // All pick boxes fit cleanly inside OUTPUT_SPATIAL_FIT_BOUNDS
    for (const box of scene.solidPickBoxes) {
      const aabb = box.frame ? sheetFrameWorldAabb(box.frame) : { min: box.min, max: box.max };
      expect(aabb.min[0]).toBeGreaterThanOrEqual(OUTPUT_SPATIAL_FIT_BOUNDS.min[0] - 1e-4);
      expect(aabb.min[1]).toBeGreaterThanOrEqual(OUTPUT_SPATIAL_FIT_BOUNDS.min[1] - 1e-4);
      expect(aabb.min[2]).toBeGreaterThanOrEqual(OUTPUT_SPATIAL_FIT_BOUNDS.min[2] - 1e-4);
      expect(aabb.max[0]).toBeLessThanOrEqual(OUTPUT_SPATIAL_FIT_BOUNDS.max[0] + 1e-4);
      expect(aabb.max[1]).toBeLessThanOrEqual(OUTPUT_SPATIAL_FIT_BOUNDS.max[1] + 1e-4);
      expect(aabb.max[2]).toBeLessThanOrEqual(OUTPUT_SPATIAL_FIT_BOUNDS.max[2] + 1e-4);
    }

    assertOpFocusBoundsValid(scene.opFocusBounds, [0, 1, 2, 3, 4, 5], OUTPUT_SPATIAL_FIT_BOUNDS);
  });

  it('supports AR Decode (S=1, startTokenOffset=4), AR feedback loop arrow, candidate staging, and all 5 Output cell formula tensors', () => {
    const arSession2 = stepDecodeSessionGreedy(prefillSession);
    const step = getActiveDecodeStep(arSession2);
    expect(step.tokens).toBe(1);
    expect(step.prefixTokens).toBe(4);

    const altCand = step.predictedNextTokens[1]; // rank #2 alternative branch candidate
    const baseOpts = {
      model,
      residualIn: step.residualStreams[52],
      finalNorm: step.finalNorm,
      lmHead: step.lmHead,
      tokenLabels: step.newTokenLabels,
      startTokenOffset: step.prefixTokens,
      precision: 'fp8_selective' as const,
      selectedPart: null,
      hoveredPart: null,
      focusToken: 0,
      isArDecode: true,
      selectedCandidateTokenId: altCand.tokenId,
    };

    const scene = buildOutputCutaway(baseOpts);

    // AR Decode feedback loop landmark & selected candidate highlight
    const arLoop = scene.landmarks.find((l) => l.id === 'out-ar-loop');
    expect(arLoop).toBeDefined();
    expect(arLoop!.label).toContain('→ AR Next Step');
    expect(arLoop!.label).toContain(altCand.label);
    expect(arLoop!.label).toContain('t5');

    const altIdx = scene.solidPickBoxes.findIndex((b) => b.partId === `output-cand-${altCand.tokenId}`);
    expect(altIdx).toBeGreaterThanOrEqual(0);
    expect(scene.sheetData![altIdx * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_ACTIVE);

    // Ray-pick onto the elevated #2 candidate bar recovers probs cell with exact vocabId
    const altBox = scene.solidPickBoxes[altIdx];
    const ray: Ray = {
      origin: [altBox.frame!.center[0], altBox.frame!.center[1], altBox.frame!.center[2] + 2.0],
      direction: [0, 0, -1],
    };
    const hit = pickMacroBay(ray, [altBox], []);
    expect(hit?.partId).toBe(`output-cand-${altCand.tokenId}`);
    const cell = pickHitToOutputCell(hit, 0);
    expect(cell).toEqual({
      tensor: 'probs',
      row: altCand.tokenId,
      col: 0,
      tokenIndex: 0,
      vocabId: altCand.tokenId,
    });

    // Verify exact formulas for all 5 OutputCellRef tensors
    const cases: readonly { ref: OutputCellRef; expectedVal: number; eqContains: string }[] = [
      {
        ref: { tensor: 'residual_in', row: 0, col: 2, tokenIndex: 0 },
        expectedVal: step.residualStreams[52][2],
        eqContains: 'Y₅₂',
      },
      {
        ref: { tensor: 'final_norm', row: 0, col: 2, tokenIndex: 0 },
        expectedVal: step.finalNorm.U_final[2],
        eqContains: 'U_final',
      },
      {
        ref: { tensor: 'w_lm_head', row: altCand.tokenId, col: 4, vocabId: altCand.tokenId },
        expectedVal: step.lmHead.W_lm_head[altCand.tokenId * 8 + 4],
        eqContains: '≠ W_embed',
      },
      {
        ref: { tensor: 'logits', row: altCand.tokenId, col: 0, tokenIndex: 0, vocabId: altCand.tokenId },
        expectedVal: step.lmHead.logits[altCand.tokenId],
        eqContains: 'W_lm_head',
      },
      {
        ref: { tensor: 'probs', row: altCand.tokenId, col: 0, tokenIndex: 0, vocabId: altCand.tokenId },
        expectedVal: step.lmHead.probs[altCand.tokenId],
        eqContains: 'exp(',
      },
    ];

    for (const c of cases) {
      const f = describeOutputCellFormula(baseOpts, c.ref);
      expect(f.isMasked).toBe(false);
      expect(Object.is(f.value, c.expectedVal)).toBe(true);
      expect(f.equation).toContain(c.eqContains);
      expect(f.subEquation.length).toBeGreaterThan(10);
    }

    // Verify operand highlighting when inspecting logits cell: logits=ACTIVE, final_norm=OPERAND_A, w_lm_head=OPERAND_B, residual_in=DIMMED
    const traceScene = buildOutputCutaway({
      ...baseOpts,
      activeCell: { tensor: 'logits', row: altCand.tokenId, col: 0, tokenIndex: 0, vocabId: altCand.tokenId },
    });
    const sd = traceScene.sheetData!;
    const idxOf = (part: string) => traceScene.solidPickBoxes.findIndex((b) => b.partId === part);
    expect(sd[idxOf('logits') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_ACTIVE);
    expect(sd[idxOf('final-norm') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_OPERAND_A);
    expect(sd[idxOf('w-lm-head') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_OPERAND_B);
    expect(sd[idxOf('residual-in') * SHEET_INSTANCE_FLOATS + 11]).toBe(SHEET_HL_DIMMED);
  });

  it('frames all 8 fitBounds corners cleanly inside 1440×900 and 1024×800 viewports for both bookend cutaways', () => {
    const viewports = [
      // Full desktop 1440×900 (both full viewport and canvas-stage minus 328px sidebar)
      { name: '1440x900-full', w: 1440, h: 900, top: 84, bottom: 118 },
      { name: '1440x900-stage', w: 1112, h: 900, top: 84, bottom: 118 },
      // Compact desktop 1024×800 (both full viewport and canvas-stage minus 328px sidebar)
      { name: '1024x800-full', w: 1024, h: 800, top: 84, bottom: 118 },
      { name: '1024x800-stage', w: 696, h: 800, top: 84, bottom: 118 },
    ];

    const cutaways = [
      {
        name: 'embedding',
        bounds: EMBEDDING_SPATIAL_FIT_BOUNDS,
        fit: (m: ReturnType<typeof makeViewportMetrics>) => fitSpatialEmbeddingCamera(m),
      },
      {
        name: 'output',
        bounds: OUTPUT_SPATIAL_FIT_BOUNDS,
        fit: (m: ReturnType<typeof makeViewportMetrics>) => fitSpatialOutputCamera(m),
      },
    ];

    for (const c of cutaways) {
      const corners = aabbCorners(c.bounds);
      for (const vp of viewports) {
        const metrics = makeViewportMetrics(0, 0, vp.w, vp.h, 1);
        const pose = c.fit(metrics);
        const snap = buildCameraSnapshot(pose, c.bounds, metrics);

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (const corner of corners) {
          const pt = projectWorldToInnerCss(corner, snap);
          expect(pt.visible, `${c.name} @ ${vp.name}`).toBe(true);
          expect(pt.x, `${c.name} @ ${vp.name}`).toBeGreaterThanOrEqual(0);
          expect(pt.x, `${c.name} @ ${vp.name}`).toBeLessThanOrEqual(vp.w);
          expect(pt.y, `${c.name} @ ${vp.name}`).toBeGreaterThanOrEqual(vp.top - 1);
          expect(pt.y, `${c.name} @ ${vp.name}`).toBeLessThanOrEqual(vp.h - vp.bottom + 1);
          if (pt.x < minX) minX = pt.x;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.y > maxY) maxY = pt.y;
        }

        const spanFracX = (maxX - minX) / vp.w;
        const spanFracY = (maxY - minY) / (vp.h - vp.top - vp.bottom);
        expect(Math.max(spanFracX, spanFracY), `${c.name} @ ${vp.name}`).toBeGreaterThan(0.72);
      }
    }
  });

  it('emits V3 cellPopover, activeOpPartIds, embeddingFormula/outputFormula, and bit-exact activeCellGrids for both bookends', () => {
    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });

    const embScene = buildEmbeddingCutaway({
      model,
      embedding: { X: exec.X_embed },
      tokenIds: exec.tokenIds,
      tokenLabels: exec.tokenLabels,
      focusToken: 3,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'x_embed', row: 2, col: 3, tokenIndex: 3, vocabId: exec.tokenIds[3] },
    });
    expect(embScene.embeddingFormula).toBeTruthy();
    expect(embScene.activeOpPartIds?.has('x-embed')).toBe(true);
    expect(embScene.activeOpPartIds?.has('w-embed')).toBe(true);
    expect(embScene.activeOpPartIds?.has('token-strip')).toBe(true);
    expect(embScene.cellPopover?.tensorLabel).toContain('Embedding');
    expect(embScene.cellPopover?.sourceWorlds?.length).toBe(2);
    expect(embScene.activeCellGrids?.length).toBe(2);
    const x0Grid = embScene.activeCellGrids![0];
    expect(x0Grid.rows).toBe(8);
    expect(x0Grid.cols).toBe(8);
    expect(Object.is(x0Grid.values[2 * 8 + 3], exec.X_embed[2 * 8 + 3])).toBe(true);

    const top0 = exec.lmHead.topKByToken[7][0];
    const outScene = buildOutputCutaway({
      model,
      residualIn: exec.residualStreams[exec.residualStreams.length - 1],
      finalNorm: exec.finalNorm,
      lmHead: exec.lmHead,
      tokenLabels: exec.tokenLabels,
      focusToken: 7,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeCell: { tensor: 'logits', row: top0.tokenId, col: 0, tokenIndex: 7, vocabId: top0.tokenId },
      isArDecode: false,
    });
    expect(outScene.outputFormula).toBeTruthy();
    expect(outScene.activeOpPartIds?.has('logits')).toBe(true);
    expect(outScene.activeOpPartIds?.has('final-norm')).toBe(true);
    expect(outScene.activeOpPartIds?.has('w-lm-head')).toBe(true);
    expect(outScene.cellPopover?.tensorLabel).toContain('Logits');
    expect(outScene.cellPopover?.sourceWorlds?.length).toBe(2);
    expect(outScene.activeCellGrids?.length).toBeGreaterThanOrEqual(3);
    const probsGrid = outScene.activeCellGrids![0];
    expect(probsGrid.format).toBe('prob');
    expect(Object.is(probsGrid.values[top0.tokenId], exec.lmHead.probs[top0.tokenId * 8 + 7])).toBe(true);
    const logitsGrid = outScene.activeCellGrids![1];
    expect(Object.is(logitsGrid.values[top0.tokenId], exec.lmHead.logits[top0.tokenId * 8 + 7])).toBe(true);
  });
});
