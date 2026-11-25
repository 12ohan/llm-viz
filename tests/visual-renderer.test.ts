import { describe, expect, it } from 'vitest';
import { NEMOTRON_SPEC } from '../src/model/spec';
import {
  aabbCorners,
  buildCameraSnapshot,
  fitPerspectiveToBounds,
  makeViewportMetrics,
  projectWorldToInnerCss,
} from '../src/render/camera';
import {
  FLOW_RIBBON_VERTEX_FLOATS,
  FlatFlowRibbonsPass,
  FlowRibbonBuilder,
} from '../src/render/ribbons';
import { makeSheetFrame3D } from '../src/render/sheetFrame';
import {
  SHEET_BASIS_BYTES,
  SHEET_BASIS_FLOATS,
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_BYTES,
  SHEET_INSTANCE_FLOATS,
  SHEET_REVEAL_BYTES,
  SHEET_REVEAL_COL_MAJOR,
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
  SheetsPass,
} from '../src/render/sheets';
import {
  bayCenterY,
  buildMacroSpine,
  describeSpineWaveStage,
  EMBED_Y,
  FINAL_NORM_Y,
  INSTRUCTIONAL_BAY_WINDOW,
  KV_TRACK_LENGTH,
  KV_TRACK_START_Z,
  LM_HEAD_Y,
  OUTPUT_LOGITS_Y,
  RESIDUAL_X,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_INSTANCE_FLOATS,
  SOLID_INSTANCE_FLOATS,
  SPINE_REPLAY_DISCLAIMER,
  TOWER_FIT_BOUNDS,
} from '../src/scene/spine';

/** Minimal WebGL2 mock that records shader source, attrib locations, and uniform calls. */
function createMockWebGL2() {
  const shaderSources: string[] = [];
  const enabledAttribs = new Set<number>();
  const attribLocationsUsed = new Set<number>();
  const attrib4fCalls: Array<{ loc: number; v: [number, number, number, number] }> = [];
  const uniform1fCalls: Array<{ loc: string; val: number }> = [];
  let nextObjId = 1;

  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    TEXTURE_2D: 0x0de1,
    TEXTURE0: 0x84c0,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    NEAREST: 0x2600,
    CLAMP_TO_EDGE: 0x812f,
    RGBA: 0x1908,
    RG32F: 0x8230,
    RG: 0x8227,
    UNSIGNED_BYTE: 0x1401,
    UNSIGNED_SHORT: 0x1403,
    UNSIGNED_INT: 0x1405,
    FLOAT: 0x1406,
    ARRAY_BUFFER: 0x8892,
    ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88e4,
    DYNAMIC_DRAW: 0x88e8,
    UNPACK_ALIGNMENT: 0x0cf5,
    DEPTH_TEST: 0x0b71,
    LEQUAL: 0x0203,
    CULL_FACE: 0x0b44,
    BACK: 0x0405,
    BLEND: 0x0be2,
    SRC_ALPHA: 0x0302,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    POLYGON_OFFSET_FILL: 0x8037,
    TRIANGLES: 0x0004,

    createShader: () => ({ id: nextObjId++ }),
    shaderSource: (_s: unknown, src: string) => {
      shaderSources.push(src);
    },
    compileShader: () => {},
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    deleteShader: () => {},
    createProgram: () => ({ id: nextObjId++ }),
    attachShader: () => {},
    linkProgram: () => {},
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    deleteProgram: () => {},
    getUniformLocation: (_p: unknown, name: string) => name,
    createVertexArray: () => ({ id: nextObjId++ }),
    deleteVertexArray: () => {},
    bindVertexArray: () => {},
    createBuffer: () => ({ id: nextObjId++ }),
    deleteBuffer: () => {},
    bindBuffer: () => {},
    bufferData: () => {},
    createTexture: () => ({ id: nextObjId++ }),
    deleteTexture: () => {},
    bindTexture: () => {},
    texParameteri: () => {},
    texImage2D: () => {},
    pixelStorei: () => {},
    enableVertexAttribArray: (loc: number) => {
      enabledAttribs.add(loc);
      attribLocationsUsed.add(loc);
    },
    disableVertexAttribArray: (loc: number) => {
      enabledAttribs.delete(loc);
    },
    vertexAttribPointer: (loc: number) => {
      attribLocationsUsed.add(loc);
    },
    vertexAttribDivisor: (loc: number) => {
      attribLocationsUsed.add(loc);
    },
    vertexAttrib3f: (loc: number) => {
      attribLocationsUsed.add(loc);
    },
    vertexAttrib4f: (loc: number, x: number, y: number, z: number, w: number) => {
      attribLocationsUsed.add(loc);
      attrib4fCalls.push({ loc, v: [x, y, z, w] });
    },
    useProgram: () => {},
    uniformMatrix4fv: () => {},
    uniform1i: () => {},
    uniform1f: (loc: string | null, val: number) => {
      if (loc) uniform1fCalls.push({ loc, val });
    },
    activeTexture: () => {},
    enable: () => {},
    disable: () => {},
    depthFunc: () => {},
    depthMask: () => {},
    cullFace: () => {},
    blendFunc: () => {},
    polygonOffset: () => {},
    drawElementsInstanced: () => {},
    drawElements: () => {},
  };

  return {
    gl: gl as unknown as WebGL2RenderingContext,
    shaderSources,
    enabledAttribs,
    attribLocationsUsed,
    attrib4fCalls,
    uniform1fCalls,
  };
}

describe('V1 · Tower Optical Centering & Perspective Framing', () => {
  it('encloses the full 52-layer tower, brackets, and KV tracks while optically centering overview framing', () => {
    // 1. TOWER_FIT_BOUNDS strictly encloses all physical tower bounds
    expect(TOWER_FIT_BOUNDS.min[0]).toBeLessThanOrEqual(-6.4);
    expect(TOWER_FIT_BOUNDS.max[0]).toBeGreaterThanOrEqual(RESIDUAL_X + 1.0);
    expect(TOWER_FIT_BOUNDS.min[1]).toBeLessThanOrEqual(OUTPUT_LOGITS_Y - 0.58);
    expect(TOWER_FIT_BOUNDS.max[1]).toBeGreaterThanOrEqual(EMBED_Y + 0.95);
    expect(TOWER_FIT_BOUNDS.max[2]).toBeGreaterThanOrEqual(KV_TRACK_START_Z + KV_TRACK_LENGTH);

    // 2. Evaluate visual balance at 1440x900 canvas (1076x858) and 1024x800 canvas (696x758)
    for (const [w, h] of [
      [1076, 858],
      [696, 758],
    ] as const) {
      const metrics = makeViewportMetrics(0, 0, w, h, 1);
      const pose = fitPerspectiveToBounds(TOWER_FIT_BOUNDS, 30, 14, w, h, 34, 0.84);
      const snap = buildCameraSnapshot(pose, TOWER_FIT_BOUNDS, metrics);

      // Every AABB corner stays strictly inside viewport
      for (const c of aabbCorners(TOWER_FIT_BOUNDS)) {
        const pt = projectWorldToInnerCss(c, snap);
        expect(pt.visible).toBe(true);
        expect(pt.x).toBeGreaterThanOrEqual(0);
        expect(pt.x).toBeLessThanOrEqual(w);
        expect(pt.y).toBeGreaterThanOrEqual(0);
        expect(pt.y).toBeLessThanOrEqual(h);
      }

      // Measure optical horizontal balance including left motif brackets (~176px) and right Embedding pill (~194px)
      const leftBracketAnchor = projectWorldToInnerCss([-6.4, bayCenterY(18), 0], snap);
      const rightEmbedAnchor = projectWorldToInnerCss([RESIDUAL_X + 0.8, EMBED_Y + 0.2, 0], snap);
      const leftMarginPct = ((leftBracketAnchor.x - 176) / w) * 100;
      const rightMarginPct = ((w - (rightEmbedAnchor.x + 194)) / w) * 100;
      const visualCenterPct = (0.5 * (leftBracketAnchor.x - 176 + (rightEmbedAnchor.x + 194)) / w) * 100;

      // Left/right whitespace balanced well within ±6% (target ±12%) and visual center near 50%
      expect(Math.abs(leftMarginPct - rightMarginPct)).toBeLessThan(6);
      expect(visualCenterPct).toBeGreaterThan(46);
      expect(visualCenterPct).toBeLessThan(54);

      // Tower top sits comfortably below the floating two-row toolbar (~84px) on desktop
      const embedTop = projectWorldToInnerCss([1.2, EMBED_Y + 0.95, 0], snap);
      expect(embedTop.y).toBeGreaterThan(84);
    }
  });

  it('supports optional opticalOffsetNdc in fitPerspectiveToBounds cleanly without NaN', () => {
    const metrics = makeViewportMetrics(0, 0, 1076, 858, 1);
    const basePose = fitPerspectiveToBounds(TOWER_FIT_BOUNDS, 30, 14, 1076, 858, 34, 0.84);
    const shiftedPose = fitPerspectiveToBounds(TOWER_FIT_BOUNDS, 30, 14, 1076, 858, 34, 0.84, {
      x: -0.05,
      y: 0.02,
    });
    const baseSnap = buildCameraSnapshot(basePose, TOWER_FIT_BOUNDS, metrics);
    const shiftedSnap = buildCameraSnapshot(shiftedPose, TOWER_FIT_BOUNDS, metrics);

    const pBase = projectWorldToInnerCss([0, bayCenterY(26), 0], baseSnap);
    const pShifted = projectWorldToInnerCss([0, bayCenterY(26), 0], shiftedSnap);

    expect(Number.isFinite(pShifted.x)).toBe(true);
    expect(Number.isFinite(pShifted.y)).toBe(true);
    // Negative NDC X offset shifts projected tower leftward in CSS X
    expect(pShifted.x).toBeLessThan(pBase.x - 20);
  });
});

describe('V1/V2 · SheetsPass Contrast & Progressive Accumulation Reveal Mask', () => {
  it('preserves 20-float sheet layout and exports exact progressive reveal constants', () => {
    expect(SHEET_INSTANCE_FLOATS).toBe(20);
    expect(SHEET_INSTANCE_BYTES).toBe(80);
    expect(SHEET_BASIS_FLOATS).toBe(6);
    expect(SHEET_BASIS_BYTES).toBe(24);
    expect(SHEET_REVEAL_FLOATS).toBe(4);
    expect(SHEET_REVEAL_BYTES).toBe(16);

    expect(SHEET_REVEAL_NONE).toBe(0);
    expect(SHEET_REVEAL_ROW_MAJOR).toBe(1);
    expect(SHEET_REVEAL_COL_MAJOR).toBe(2);
    expect(SHEET_REVEAL_UNCOMPUTED).toBe(3);

    // Existing highlight hierarchy constants preserved untouched
    expect(SHEET_HL_NORMAL).toBe(0);
    expect(SHEET_HL_ACTIVE).toBe(1);
    expect(SHEET_HL_PATHWAY).toBe(2);
    expect(SHEET_HL_DIMMED).toBe(3);
    expect(SHEET_HL_BF16_GOLD).toBe(4);
    expect(SHEET_HL_OPERAND_A).toBe(5);
    expect(SHEET_HL_OPERAND_B).toBe(6);
  });

  it('keeps all WebGL2 vertex attribute locations < 16 and defaults iReveal to (0,0,0,0) when omitted', () => {
    const mock = createMockWebGL2();
    const pass = new SheetsPass(mock.gl);

    // Critical WebGL2 portability invariant: all bound attrib locations strictly in 0..15
    for (const loc of mock.attribLocationsUsed) {
      expect(loc).toBeGreaterThanOrEqual(0);
      expect(loc).toBeLessThan(16);
    }

    // Fragment shader contains progressive reveal branches and uncomputed ghost state
    const fs = mock.shaderSources[1];
    expect(fs).toContain('UNCOMPUTED_GHOST');
    expect(fs).toContain('isFrontierCell');
    expect(fs).toContain('isFullUncomputed');

    // Upload without revealData keeps reveal attribute disabled with default (0,0,0,0)
    const sheetData = new Float32Array(2 * SHEET_INSTANCE_FLOATS);
    pass.upload(sheetData, 2);
    expect(mock.enabledAttribs.has(12)).toBe(false);
    expect(mock.attrib4fCalls.at(-1)).toEqual({ loc: 12, v: [0, 0, 0, 0] });

    const identity = new Float32Array(16);
    pass.draw(identity);

    // Upload with parallel revealData enables location 12
    const revealData = new Float32Array([
      SHEET_REVEAL_ROW_MAJOR, 3, 4, 0.5,
      SHEET_REVEAL_UNCOMPUTED, 0, 0, 0,
    ]);
    pass.upload(sheetData, 2, undefined, revealData);
    expect(mock.enabledAttribs.has(12)).toBe(true);
    pass.draw(identity);

    // Subsequent upload without revealData cleanly disables location 12 again
    pass.upload(sheetData, 2);
    expect(mock.enabledAttribs.has(12)).toBe(false);

    pass.dispose();
  });
});

describe('V2 · FlatFlowRibbonsPass Animated Directional Pulses', () => {
  it('defaults uPulseTime to -1 (static) and forwards explicit pulseTime when animated', () => {
    const mock = createMockWebGL2();
    const pass = new FlatFlowRibbonsPass(mock.gl);

    // Fragment shader guards pulse on uPulseTime >= 0.0 and vColor.a >= 0.65 with directional fract()
    const fs = mock.shaderSources[1];
    expect(fs).toContain('uniform float uPulseTime');
    expect(fs).toContain('vColor.a >= 0.65');
    expect(fs).toContain('fract(vAlong * 1.4 - uPulseTime * 0.85)');

    const builder = new FlowRibbonBuilder();
    builder.addCubic3D(
      [0, 0, 0],
      [1, 0.5, 0],
      [2, 0.5, 0],
      [3, 0, 0],
      [0, 1, 0],
      0.06,
      { r: 0.22, g: 0.38, b: 0.74, a: 0.92, borderAlpha: 0.42 },
      true,
    );
    const { vertices, indices } = builder.pack();
    pass.upload(vertices, indices);

    const viewProj = new Float32Array(16);
    pass.draw(viewProj);
    expect(mock.uniform1fCalls.at(-1)).toEqual({ loc: 'uPulseTime', val: -1 });

    pass.draw(viewProj, 1.75);
    expect(mock.uniform1fCalls.at(-1)).toEqual({ loc: 'uPulseTime', val: 1.75 });

    pass.dispose();
  });

  it('keeps directional threads in aAlong [0..1] while marking static cell highlight frames with aAlong = -1', () => {
    const threadBuilder = new FlowRibbonBuilder();
    threadBuilder.addPolyline3D(
      [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
      ],
      [0, 1, 0],
      0.05,
      { r: 0.22, g: 0.38, b: 0.74, a: 0.92 },
    );
    const threadVerts = threadBuilder.pack().vertices;
    for (let i = 0; i < threadVerts.length; i += FLOW_RIBBON_VERTEX_FLOATS) {
      const along = threadVerts[i + 4];
      expect(along).toBeGreaterThanOrEqual(0);
      expect(along).toBeLessThanOrEqual(1);
    }

    const frameBuilder = new FlowRibbonBuilder();
    const frame = makeSheetFrame3D([0, 0, 0], [2, 0.08, 2]);
    frameBuilder.addCellHighlightOnFrame(
      frame,
      0,
      0,
      0,
      0,
      4,
      4,
      0.01,
      { r: 0.86, g: 0.30, b: 0.20, a: 0.28 },
      { r: 0.86, g: 0.30, b: 0.20, a: 1.0 },
    );
    const frameVerts = frameBuilder.pack().vertices;
    expect(frameVerts.length).toBeGreaterThan(0);
    for (let i = 0; i < frameVerts.length; i += FLOW_RIBBON_VERTEX_FLOATS) {
      expect(frameVerts[i + 4]).toBe(-1);
    }
  });
});

describe('V4 · 52-Layer Residual-Spine Propagation Wave', () => {
  function extractSolidsByBayAndPart(scene: ReturnType<typeof buildMacroSpine>) {
    // Helper to query solid instances by bayId + coordinates/extents
    const list: Array<{
      cx: number;
      cy: number;
      cz: number;
      ex: number;
      ey: number;
      ez: number;
      cellFamily: number;
      fp8: number;
      bayId: number;
      hl: number;
    }> = [];
    for (let i = 0; i < scene.solidCount; i++) {
      const o = i * SOLID_INSTANCE_FLOATS;
      list.push({
        cx: scene.solidData[o + 0],
        cy: scene.solidData[o + 1],
        cz: scene.solidData[o + 2],
        ex: scene.solidData[o + 3],
        ey: scene.solidData[o + 4],
        ez: scene.solidData[o + 5],
        cellFamily: scene.solidData[o + 6],
        fp8: scene.solidData[o + 7],
        bayId: scene.solidData[o + 9],
        hl: scene.solidData[o + 11],
      });
    }
    return list;
  }

  function extractRibbons(scene: ReturnType<typeof buildMacroSpine>) {
    const list: Array<{
      cx: number;
      cy: number;
      ex: number;
      ey: number;
      kind: number;
      colorId: number;
      alpha: number;
    }> = [];
    for (let i = 0; i < scene.ribbonCount; i++) {
      const o = i * RIBBON_INSTANCE_FLOATS;
      list.push({
        cx: scene.ribbonData[o + 0],
        cy: scene.ribbonData[o + 1],
        ex: scene.ribbonData[o + 3],
        ey: scene.ribbonData[o + 4],
        kind: scene.ribbonData[o + 6],
        colorId: scene.ribbonData[o + 7],
        alpha: scene.ribbonData[o + 8],
      });
    }
    return list;
  }

  it('produces bit-for-bit identical geometry when propagationWave is omitted vs null', () => {
    const omitted = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
    });
    const nulled = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: null,
    });

    expect(nulled.solidCount).toBe(omitted.solidCount);
    expect(nulled.ribbonCount).toBe(omitted.ribbonCount);
    expect(nulled.lineCount).toBe(omitted.lineCount);
    expect(Array.from(nulled.solidData)).toEqual(Array.from(omitted.solidData));
    expect(Array.from(nulled.ribbonData)).toEqual(Array.from(omitted.ribbonData));
  });

  it('sweeps cleanly from Embedding (0.0) through Bays 01..52 (0.5) to Output (1.0), highlighting Mamba SSM, MoE, and crossed GQA KV slices', () => {
    const base = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'fp8_selective',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: null,
    });
    const wave0 = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'fp8_selective',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.0 },
    });
    const waveMid = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'fp8_selective',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.5 },
    });
    const waveEnd = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'fp8_selective',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 1.0 },
    });

    // Helper: identify highlighted KV slices (ex=0.36, ey=0.52, ez=0.30)
    const highlightedKvSlices = (sc: ReturnType<typeof buildMacroSpine>) =>
      extractSolidsByBayAndPart(sc).filter(
        (s) =>
          Math.abs(s.ex - 0.36) < 1e-4 &&
          Math.abs(s.ey - 0.52) < 1e-4 &&
          Math.abs(s.ez - 0.30) < 1e-4 &&
          s.hl === 1,
      );

    // Baseline (null): 0 highlighted KV slices when no bay selected/hovered
    expect(highlightedKvSlices(base)).toHaveLength(0);

    // At progress = 0.0: Embedding glows, glowing wavefront emerald capsule emitted at EMBED_Y, 0 GQA KV slices crossed yet
    const s0 = extractSolidsByBayAndPart(wave0);
    const embedAct0 = s0.find((s) => s.bayId === 0 && Math.abs(s.cy - EMBED_Y) < 1e-4 && Math.abs(s.cx - RESIDUAL_X) < 1e-4);
    expect(embedAct0?.hl).toBe(1);
    expect(highlightedKvSlices(wave0)).toHaveLength(0);
    const pulseRibbons0 = extractRibbons(wave0).filter(
      (r) => Math.abs(r.cx - RESIDUAL_X) < 1e-4 && r.colorId === RIBBON_COLOR_EMERALD && r.alpha >= 0.85,
    );
    expect(pulseRibbons0.length).toBeGreaterThanOrEqual(1);
    expect(Math.abs(pulseRibbons0[0].cy - EMBED_Y)).toBeLessThan(1e-3);

    // At progress = 0.5 (waveBay = 26.5):
    // - Glowing emerald pulse sits midway down the spine (~bayCenterY(26..27))
    // - Bay 26 (Mamba-2) cyan SSM register (cx = -1.35) pulses (hl = 1)
    // - Bay 25 (MoE) router (cx = -0.05) flashes (hl = 1)
    // - Residual nodes near bays 25..28 are highlighted while far-away bay 01 & bay 52 are not
    // - Crossed GQA bays [06, 13, 20] have their 8th KV slice highlighted (3 bays * 4 rails = 12); uncrossed [27, 34, 43] do NOT!
    const sMid = extractSolidsByBayAndPart(waveMid);
    const mamba26Ssm = sMid.find((s) => s.bayId === 26 && Math.abs(s.cx - -1.35) < 1e-4);
    expect(mamba26Ssm?.hl).toBe(1);
    const moe25Router = sMid.find((s) => s.bayId === 25 && Math.abs(s.cx - -0.05) < 1e-4);
    expect(moe25Router?.hl).toBe(1);
    const res26 = sMid.find((s) => s.bayId === 26 && Math.abs(s.cx - RESIDUAL_X) < 1e-4);
    const res01 = sMid.find((s) => s.bayId === 1 && Math.abs(s.cx - RESIDUAL_X) < 1e-4);
    expect(res26?.hl).toBe(1);
    expect(res01?.hl).toBe(0);

    const midKvSlices = highlightedKvSlices(waveMid);
    const midCrossedGqaBays = Array.from(new Set(midKvSlices.map((s) => s.bayId))).sort((a, b) => a - b);
    expect(midCrossedGqaBays).toEqual([6, 13, 20]);
    expect(midKvSlices).toHaveLength(3 * 4); // 3 crossed GQA bays × 4 rails × 1 newest token slice

    // At progress = 1.0:
    // - Output Logits glows (hl = 1)
    // - ALL 6 GQA bays [6, 13, 20, 27, 34, 43] have their newest KV slice highlighted (6 * 4 = 24)
    // - FP8 metadata on KV slices stays strictly intact (fp8 === 1 in fp8_selective mode)
    const sEnd = extractSolidsByBayAndPart(waveEnd);
    const outLogits = sEnd.find((s) => s.bayId === 0 && Math.abs(s.cy - OUTPUT_LOGITS_Y) < 1e-4);
    expect(outLogits?.hl).toBe(1);
    const endKvSlices = highlightedKvSlices(waveEnd);
    const endCrossedGqaBays = Array.from(new Set(endKvSlices.map((s) => s.bayId))).sort((a, b) => a - b);
    expect(endCrossedGqaBays).toEqual([6, 13, 20, 27, 34, 43]);
    expect(endKvSlices).toHaveLength(6 * 4);
    expect(endKvSlices.every((s) => s.fp8 === 1)).toBe(true);
    expect(waveEnd.ribbonCount).toBeGreaterThan(base.ribbonCount);
  });

  it('respects custom activeTokenIndex on propagationWave when highlighting crossed GQA KV slices', () => {
    const waveCustom = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 1.0, activeTokenIndex: 5 },
    });

    const solids = extractSolidsByBayAndPart(waveCustom);
    const hlSlices = solids.filter(
      (s) =>
        Math.abs(s.ex - 0.36) < 1e-4 &&
        Math.abs(s.ey - 0.52) < 1e-4 &&
        Math.abs(s.ez - 0.30) < 1e-4 &&
        s.hl === 1,
    );
    // Expected z of 5th token slice: KV_TRACK_START_Z + 0.42 * 4 + 0.15 = 0.90 + 1.68 + 0.15 = 2.73
    const expectedZ = KV_TRACK_START_Z + 0.42 * 4 + 0.15;
    expect(hlSlices).toHaveLength(6 * 4);
    for (const slice of hlSlices) {
      expect(slice.cz).toBeCloseTo(expectedZ, 5);
    }
  });
});

describe('RC3 · Spine Propagation Pacing & Instructional Replay Stage Model', () => {
  function extractSolids(scene: ReturnType<typeof buildMacroSpine>) {
    const list: Array<{
      cx: number;
      cy: number;
      cz: number;
      ex: number;
      ey: number;
      ez: number;
      cellFamily: number;
      fp8: number;
      bayId: number;
      hl: number;
    }> = [];
    for (let i = 0; i < scene.solidCount; i++) {
      const o = i * SOLID_INSTANCE_FLOATS;
      list.push({
        cx: scene.solidData[o + 0],
        cy: scene.solidData[o + 1],
        cz: scene.solidData[o + 2],
        ex: scene.solidData[o + 3],
        ey: scene.solidData[o + 4],
        ez: scene.solidData[o + 5],
        cellFamily: scene.solidData[o + 6],
        fp8: scene.solidData[o + 7],
        bayId: scene.solidData[o + 9],
        hl: scene.solidData[o + 11],
      });
    }
    return list;
  }

  function extractRibbons(scene: ReturnType<typeof buildMacroSpine>) {
    const list: Array<{
      cx: number;
      cy: number;
      ex: number;
      ey: number;
      kind: number;
      colorId: number;
      alpha: number;
    }> = [];
    for (let i = 0; i < scene.ribbonCount; i++) {
      const o = i * RIBBON_INSTANCE_FLOATS;
      list.push({
        cx: scene.ribbonData[o + 0],
        cy: scene.ribbonData[o + 1],
        ex: scene.ribbonData[o + 3],
        ey: scene.ribbonData[o + 4],
        kind: scene.ribbonData[o + 6],
        colorId: scene.ribbonData[o + 7],
        alpha: scene.ribbonData[o + 8],
      });
    }
    return list;
  }

  it('describes all spine replay stages with precision-accurate KV copy, optional RMS, and explicit educational disclaimer', () => {
    expect(SPINE_REPLAY_DISCLAIMER).toBe('Educational replay · not measured compute time');
    expect(INSTRUCTIONAL_BAY_WINDOW).toBe(1.2);

    const summaries = [
      { bayId: 1, rmsOut: [0.84, 1.12] },
      { bayId: 2, rmsOut: new Float32Array([0.92, 1.37]) },
      { bayId: 6, rmsOut: [1.05, 1.48] },
      { bayId: 43, rmsOut: [1.21, 1.64] },
    ];

    // 1. Embedding (< 0.05)
    const embedStage = describeSpineWaveStage(0.01, NEMOTRON_SPEC, 'bf16', summaries, 1);
    expect(embedStage.stageKind).toBe('embedding');
    expect(embedStage.bayId).toBeNull();
    expect(embedStage.title).toContain('Embedding');
    expect(embedStage.title).toContain('NoPE');
    expect(embedStage.rmsText).toBeNull();
    expect(embedStage.progressPct).toBe(1);
    expect(embedStage.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    // 2. Bay 01 Mamba-2 (0.06)
    const mambaStage = describeSpineWaveStage(0.06, NEMOTRON_SPEC, 'bf16', summaries, 1);
    expect(mambaStage.stageKind).toBe('mamba');
    expect(mambaStage.bayId).toBe(1);
    expect(mambaStage.title).toContain('Bay 01');
    expect(mambaStage.title).toContain('Mamba-2');
    expect(mambaStage.actionDetail).toContain('47.1 MiB');
    expect(mambaStage.actionDetail).toContain('in place');
    expect(mambaStage.rmsText).toBe('RMS 1.12');
    expect(mambaStage.progressPct).toBe(6);
    expect(mambaStage.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    // Active token index selects exact RMS slot (0 vs default last)
    const mambaTok0 = describeSpineWaveStage(0.06, NEMOTRON_SPEC, 'bf16', summaries, 0);
    expect(mambaTok0.rmsText).toBe('RMS 0.84');
    const mambaNoSummaries = describeSpineWaveStage(0.06, NEMOTRON_SPEC, 'bf16', null);
    expect(mambaNoSummaries.rmsText).toBeNull();

    // 3. Bay 02 MoE (progress = 0.075 -> bay 2)
    const moeStage = describeSpineWaveStage(0.075, NEMOTRON_SPEC, 'fp8_selective', summaries);
    expect(moeStage.stageKind).toBe('moe');
    expect(moeStage.bayId).toBe(2);
    expect(moeStage.title).toContain('Bay 02');
    expect(moeStage.title).toContain('MoE');
    expect(moeStage.actionDetail).toContain('6/128');
    expect(moeStage.actionDetail).toContain('0 B cache');
    expect(moeStage.rmsText).toBe('RMS 1.37');
    expect(moeStage.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    // 4. Bay 06 GQA (progress = 0.14 -> bay 6) in BF16 vs FP8 Selective
    const gqaBf16 = describeSpineWaveStage(0.14, NEMOTRON_SPEC, 'bf16', summaries, 1);
    expect(gqaBf16.stageKind).toBe('gqa');
    expect(gqaBf16.bayId).toBe(6);
    expect(gqaBf16.title).toContain('Bay 06');
    expect(gqaBf16.title).toContain('GQA Attention');
    expect(gqaBf16.title).toContain('1/6 KV bays');
    expect(gqaBf16.actionDetail).toContain('+6 KiB/tok');
    expect(gqaBf16.actionDetail).toContain('BF16');
    expect(gqaBf16.actionDetail).not.toContain('+3 KiB/tok');
    expect(gqaBf16.rmsText).toBe('RMS 1.48');
    expect(gqaBf16.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    const gqaFp8 = describeSpineWaveStage(0.14, NEMOTRON_SPEC, 'fp8_selective', summaries, 1);
    expect(gqaFp8.stageKind).toBe('gqa');
    expect(gqaFp8.bayId).toBe(6);
    expect(gqaFp8.actionDetail).toContain('+3 KiB/tok');
    expect(gqaFp8.actionDetail).toContain('FP8');
    expect(gqaFp8.actionDetail).not.toContain('+6 KiB/tok');
    expect(gqaFp8.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    // 5. Final RMSNorm (0.93)
    const normStage = describeSpineWaveStage(0.93, NEMOTRON_SPEC, 'bf16', summaries);
    expect(normStage.stageKind).toBe('final_norm');
    expect(normStage.bayId).toBeNull();
    expect(normStage.title).toContain('Final RMSNorm');
    expect(normStage.rmsText).toBeNull();
    expect(normStage.progressPct).toBe(93);
    expect(normStage.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    // 6. Output LM Head (0.99)
    const outStage = describeSpineWaveStage(0.99, NEMOTRON_SPEC, 'fp8_selective', summaries);
    expect(outStage.stageKind).toBe('output');
    expect(outStage.bayId).toBeNull();
    expect(outStage.title).toContain('Output');
    expect(outStage.actionDetail).toContain('131,072');
    expect(outStage.rmsText).toBeNull();
    expect(outStage.progressPct).toBe(99);
    expect(outStage.disclaimer).toBe(SPINE_REPLAY_DISCLAIMER);

    // Out-of-range clamping safety
    expect(describeSpineWaveStage(-0.5, NEMOTRON_SPEC, 'bf16').stageKind).toBe('embedding');
    expect(describeSpineWaveStage(1.5, NEMOTRON_SPEC, 'bf16').stageKind).toBe('output');
    expect(describeSpineWaveStage(Number.NaN, NEMOTRON_SPEC, 'bf16').disclaimer).toBe(
      SPINE_REPLAY_DISCLAIMER,
    );
  });

  it('preserves bit-for-bit fast wave defaults and sharpens single-bay focus in instructional mode', () => {
    const none = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: null,
    });
    const fastDefault = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.5 },
    });
    const fastExplicit = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.5, mode: 'fast' },
    });
    const instructionalMid = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.5, mode: 'instructional' },
    });

    // Omitted mode === explicit 'fast' bit-for-bit
    expect(Array.from(fastExplicit.solidData)).toEqual(Array.from(fastDefault.solidData));
    expect(Array.from(fastExplicit.ribbonData)).toEqual(Array.from(fastDefault.ribbonData));
    expect(Array.from(fastExplicit.lineData)).toEqual(Array.from(fastDefault.lineData));

    const highlightedResidualBays = (sc: ReturnType<typeof buildMacroSpine>) =>
      extractSolids(sc)
        .filter((s) => s.bayId > 0 && Math.abs(s.cx - RESIDUAL_X) < 1e-4 && s.hl === 1)
        .map((s) => s.bayId)
        .sort((a, b) => a - b);

    // Null wave highlights 0 residual bay nodes
    expect(highlightedResidualBays(none)).toEqual([]);

    // Fast wave at 0.5 spans a wider multi-bay blur (4 residual nodes: bays 25..28)
    expect(highlightedResidualBays(fastExplicit)).toEqual([25, 26, 27, 28]);

    // Instructional wave at 0.5 focuses strictly on the single active bay matching describeSpineWaveStage (Bay 28 MoE)
    const midStage = describeSpineWaveStage(0.5, NEMOTRON_SPEC, 'bf16');
    expect(midStage.bayId).toBe(28);
    expect(midStage.stageKind).toBe('moe');
    expect(highlightedResidualBays(instructionalMid)).toEqual([28]);

    const midSolids = extractSolids(instructionalMid);
    const moe28Router = midSolids.find((s) => s.bayId === 28 && Math.abs(s.cx - -0.05) < 1e-4);
    const mamba26Ssm = midSolids.find((s) => s.bayId === 26 && Math.abs(s.cx - -1.35) < 1e-4);
    const moe25Router = midSolids.find((s) => s.bayId === 25 && Math.abs(s.cx - -0.05) < 1e-4);
    expect(moe28Router?.hl).toBe(1);
    expect(mamba26Ssm?.hl).toBe(0);
    expect(moe25Router?.hl).toBe(0);

    // Instructional MoE bay emits amber focus ribbons while spine beacon stays tighter (ey = 1.55 vs 2.5)
    const midRibbons = extractRibbons(instructionalMid);
    const fastBeacon = extractRibbons(fastExplicit).find(
      (r) => Math.abs(r.cx - RESIDUAL_X) < 1e-4 && r.colorId === RIBBON_COLOR_EMERALD && r.alpha >= 0.85,
    );
    const instrBeacon = midRibbons.find(
      (r) => Math.abs(r.cx - RESIDUAL_X) < 1e-4 && r.colorId === RIBBON_COLOR_EMERALD && r.alpha >= 0.90,
    );
    expect(fastBeacon?.ey).toBeCloseTo(2.5, 4);
    expect(instrBeacon?.ey).toBeCloseTo(1.55, 4);
    expect(midRibbons.some((r) => r.colorId === RIBBON_COLOR_AMBER && r.alpha >= 0.85)).toBe(true);

    // Verify instructional single-bay focus on Bay 01 Mamba (0.06), Bay 06 GQA (0.14), Final Norm (0.93), and Output (0.99)
    const instrMamba = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.06, mode: 'instructional' },
    });
    expect(highlightedResidualBays(instrMamba)).toEqual([1]);
    expect(
      extractSolids(instrMamba).find((s) => s.bayId === 1 && Math.abs(s.cx - -1.35) < 1e-4)?.hl,
    ).toBe(1);
    expect(
      extractRibbons(instrMamba).some((r) => r.colorId === RIBBON_COLOR_CYAN && r.alpha >= 0.85),
    ).toBe(true);

    const instrGqa = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'fp8_selective',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.14, mode: 'instructional' },
    });
    expect(highlightedResidualBays(instrGqa)).toEqual([6]);
    const gqa06KvSlices = extractSolids(instrGqa).filter(
      (s) =>
        Math.abs(s.ex - 0.36) < 1e-4 &&
        Math.abs(s.ey - 0.52) < 1e-4 &&
        Math.abs(s.ez - 0.30) < 1e-4 &&
        s.hl === 1,
    );
    expect(gqa06KvSlices).toHaveLength(4);
    expect(gqa06KvSlices.every((s) => s.bayId === 6 && s.fp8 === 1)).toBe(true);

    const instrNorm = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.93, mode: 'instructional' },
    });
    expect(highlightedResidualBays(instrNorm)).toEqual([]);
    const normSolids = extractSolids(instrNorm);
    expect(normSolids.find((s) => s.bayId === 0 && Math.abs(s.cy - FINAL_NORM_Y) < 1e-4)?.hl).toBe(1);
    expect(normSolids.find((s) => s.bayId === 0 && Math.abs(s.cy - LM_HEAD_Y) < 1e-4)?.hl).toBe(0);
    expect(normSolids.find((s) => s.bayId === 0 && Math.abs(s.cy - OUTPUT_LOGITS_Y) < 1e-4)?.hl).toBe(0);

    const instrOut = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
      propagationWave: { progress: 0.99, mode: 'instructional' },
    });
    const outSolids = extractSolids(instrOut);
    expect(outSolids.find((s) => s.bayId === 0 && Math.abs(s.cy - FINAL_NORM_Y) < 1e-4)?.hl).toBe(0);
    expect(outSolids.find((s) => s.bayId === 0 && Math.abs(s.cy - LM_HEAD_Y) < 1e-4)?.hl).toBe(1);
    expect(outSolids.find((s) => s.bayId === 0 && Math.abs(s.cy - OUTPUT_LOGITS_Y) < 1e-4)?.hl).toBe(1);
  });
});
