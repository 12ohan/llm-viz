import { describe, expect, it } from 'vitest';

import {
  createDefaultHybridModel,
  defaultTokenIdsForLength,
  runHybridExecution,
} from '../src/math/hybridFlow';
import {
  LINE_FREE_FACE,
  LINE_FREE_OWNER_ID,
  LINE_INSTANCE_ATTRIBS,
  LINE_INSTANCE_BYTES,
  LINE_INSTANCE_FLOATS,
  LINE_OFFSETS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  pushBoxFrameXZ,
  pushDashedFrameXZ,
  pushFreeGuideLine,
} from '../src/render/linePacking';
import { LinesPass } from '../src/render/lines';
import type { CutawaySceneGeometry } from '../src/scene/cutaways';
import { buildEmbeddingCutaway } from '../src/scene/embeddingCutaway';
import { buildGqaCutaway } from '../src/scene/gqaCutaway';
import { buildMambaCutaway } from '../src/scene/mambaCutaway';
import { buildMoECutaway } from '../src/scene/moeCutaway';
import { buildOutputCutaway } from '../src/scene/outputCutaway';
import {
  LINE_INSTANCE_BYTES as SPINE_LINE_INSTANCE_BYTES,
  LINE_INSTANCE_FLOATS as SPINE_LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE as SPINE_LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD as SPINE_LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE as SPINE_LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL as SPINE_LINE_STYLE_NEUTRAL,
} from '../src/scene/spine';

const VALID_STYLES = new Set<number>([
  LINE_STYLE_NEUTRAL,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  LINE_STYLE_BF16_GOLD,
]);

interface RecordedAttribPointer {
  readonly loc: number;
  readonly size: number;
  readonly type: number;
  readonly normalized: boolean;
  readonly stride: number;
  readonly offset: number;
}

function createLinesPassMockGl() {
  const pointers: RecordedAttribPointer[] = [];
  const divisors = new Map<number, number>();
  const shaderSources: string[] = [];

  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: 0x8892,
    ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88e4,
    DYNAMIC_DRAW: 0x88e8,
    FLOAT: 0x1406,

    createShader: () => ({}),
    shaderSource: (_shader: unknown, src: string) => {
      shaderSources.push(src);
    },
    compileShader: () => {},
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    deleteShader: () => {},

    createProgram: () => ({}),
    attachShader: () => {},
    linkProgram: () => {},
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    deleteProgram: () => {},
    getUniformLocation: () => ({}),

    createVertexArray: () => ({}),
    bindVertexArray: () => {},
    deleteVertexArray: () => {},
    createBuffer: () => ({}),
    bindBuffer: () => {},
    bufferData: () => {},
    deleteBuffer: () => {},

    enableVertexAttribArray: () => {},
    vertexAttribPointer: (
      loc: number,
      size: number,
      type: number,
      normalized: boolean,
      stride: number,
      offset: number,
    ) => {
      pointers.push({ loc, size, type, normalized, stride, offset });
    },
    vertexAttribDivisor: (loc: number, divisor: number) => {
      divisors.set(loc, divisor);
    },
  };

  return {
    gl: gl as unknown as WebGL2RenderingContext,
    pointers,
    divisors,
    shaderSources,
  };
}

function assertAllGuideLinesValid(
  scene: Pick<CutawaySceneGeometry, 'lineData' | 'lineCount'>,
  label: string,
): void {
  expect(scene.lineCount, `${label}: expected > 0 guide lines`).toBeGreaterThan(0);
  expect(scene.lineData.length, `${label}: lineData float length mismatch`).toBe(
    scene.lineCount * LINE_INSTANCE_FLOATS,
  );

  const d = scene.lineData;
  for (let i = 0; i < scene.lineCount; i++) {
    const o = i * LINE_INSTANCE_FLOATS;
    const ax = d[o + LINE_OFFSETS.pointA];
    const ay = d[o + LINE_OFFSETS.pointA + 1];
    const az = d[o + LINE_OFFSETS.pointA + 2];
    const bx = d[o + LINE_OFFSETS.pointB];
    const by = d[o + LINE_OFFSETS.pointB + 1];
    const bz = d[o + LINE_OFFSETS.pointB + 2];
    const cssWidth = d[o + LINE_OFFSETS.cssWidth];
    const style = d[o + LINE_OFFSETS.style];
    const alpha = d[o + LINE_OFFSETS.alpha];
    const ownerId = d[o + LINE_OFFSETS.ownerId];
    const face0 = d[o + LINE_OFFSETS.face0];
    const face1 = d[o + LINE_OFFSETS.face1];

    expect(Number.isFinite(ax), `${label}[${i}] ax finite`).toBe(true);
    expect(Number.isFinite(ay), `${label}[${i}] ay finite`).toBe(true);
    expect(Number.isFinite(az), `${label}[${i}] az finite`).toBe(true);
    expect(Number.isFinite(bx), `${label}[${i}] bx finite`).toBe(true);
    expect(Number.isFinite(by), `${label}[${i}] by finite`).toBe(true);
    expect(Number.isFinite(bz), `${label}[${i}] bz finite`).toBe(true);

    const segLen = Math.hypot(bx - ax, by - ay, bz - az);
    expect(segLen, `${label}[${i}] non-degenerate segment length`).toBeGreaterThan(1e-4);

    expect(Number.isFinite(cssWidth) && cssWidth > 0, `${label}[${i}] cssWidth > 0 (got ${cssWidth})`).toBe(true);
    expect(Number.isFinite(alpha) && alpha > 0 && alpha <= 1, `${label}[${i}] alpha in (0, 1] (got ${alpha})`).toBe(true);
    expect(VALID_STYLES.has(style), `${label}[${i}] valid style (got ${style})`).toBe(true);

    expect(ownerId, `${label}[${i}] ownerId sentinel`).toBe(LINE_FREE_OWNER_ID);
    expect(face0, `${label}[${i}] face0 sentinel (disables backface culling)`).toBe(LINE_FREE_FACE);
    expect(face1, `${label}[${i}] face1 sentinel (disables backface culling)`).toBe(LINE_FREE_FACE);

    expect(d[o + LINE_OFFSETS.ownerCenter]).toBe(0);
    expect(d[o + LINE_OFFSETS.ownerCenter + 1]).toBe(0);
    expect(d[o + LINE_OFFSETS.ownerCenter + 2]).toBe(0);
    expect(d[o + LINE_OFFSETS.ownerExtent]).toBe(0);
    expect(d[o + LINE_OFFSETS.ownerExtent + 1]).toBe(0);
    expect(d[o + LINE_OFFSETS.ownerExtent + 2]).toBe(0);
  }
}

describe('LinesPass 18-float packing contract (src/render/linePacking.ts)', () => {
  it('matches spine constants and covers the 18-float (72-byte) stride contiguously with zero gaps', () => {
    expect(LINE_INSTANCE_FLOATS).toBe(18);
    expect(LINE_INSTANCE_BYTES).toBe(72);
    expect(LINE_INSTANCE_FLOATS).toBe(SPINE_LINE_INSTANCE_FLOATS);
    expect(LINE_INSTANCE_BYTES).toBe(SPINE_LINE_INSTANCE_BYTES);
    expect(LINE_STYLE_NEUTRAL).toBe(SPINE_LINE_STYLE_NEUTRAL);
    expect(LINE_STYLE_ACTIVE).toBe(SPINE_LINE_STYLE_ACTIVE);
    expect(LINE_STYLE_GUIDE).toBe(SPINE_LINE_STYLE_GUIDE);
    expect(LINE_STYLE_BF16_GOLD).toBe(SPINE_LINE_STYLE_BF16_GOLD);

    expect(LINE_OFFSETS).toEqual({
      pointA: 0,
      pointB: 3,
      cssWidth: 6,
      style: 7,
      alpha: 8,
      ownerId: 9,
      face0: 10,
      face1: 11,
      ownerCenter: 12,
      ownerExtent: 15,
    });

    let cursor = 0;
    for (const attr of LINE_INSTANCE_ATTRIBS) {
      expect(attr.floatOffset).toBe(cursor);
      expect(attr.byteOffset).toBe(cursor * 4);
      cursor += attr.size;
    }
    expect(cursor).toBe(LINE_INSTANCE_FLOATS);
  });

  it('aligns LINE_OFFSETS and LINE_INSTANCE_ATTRIBS 1-to-1 with live LinesPass WebGL2 attribute bindings and LINES_VS layout locations', () => {
    const mock = createLinesPassMockGl();
    const pass = new LinesPass(mock.gl);
    pass.dispose();

    const instancePointers = mock.pointers.filter((p) => p.loc >= 2);
    expect(instancePointers).toHaveLength(LINE_INSTANCE_ATTRIBS.length);

    for (let i = 0; i < LINE_INSTANCE_ATTRIBS.length; i++) {
      const expected = LINE_INSTANCE_ATTRIBS[i];
      const actual = instancePointers[i];
      expect(actual.loc).toBe(expected.location);
      expect(actual.size).toBe(expected.size);
      expect(actual.stride).toBe(LINE_INSTANCE_BYTES);
      expect(actual.offset).toBe(expected.byteOffset);
      expect(actual.normalized).toBe(false);
      expect(mock.divisors.get(expected.location)).toBe(1);
    }

    // Verify LINES_VS GLSL layout(location = N) declarations match attribute names & types
    const vsSrc = mock.shaderSources[0];
    expect(vsSrc).toBeDefined();
    for (const attr of LINE_INSTANCE_ATTRIBS) {
      const glslType = attr.size === 3 ? 'vec3' : 'float';
      const pattern = new RegExp(
        `layout\\(location\\s*=\\s*${attr.location}\\)\\s+in\\s+${glslType}\\s+${attr.name};`,
      );
      expect(vsSrc).toMatch(pattern);
    }
  });

  it('packs pushFreeGuideLine records with exact offsets and -1 sentinels', () => {
    const out: number[] = [];
    pushFreeGuideLine(out, 1.5, 2.5, -3.5, 4.25, -5.75, 6.0, 1.25, LINE_STYLE_ACTIVE, 0.85);
    expect(out).toHaveLength(LINE_INSTANCE_FLOATS);

    expect(out.slice(LINE_OFFSETS.pointA, LINE_OFFSETS.pointA + 3)).toEqual([1.5, 2.5, -3.5]);
    expect(out.slice(LINE_OFFSETS.pointB, LINE_OFFSETS.pointB + 3)).toEqual([4.25, -5.75, 6.0]);
    expect(out[LINE_OFFSETS.cssWidth]).toBe(1.25);
    expect(out[LINE_OFFSETS.style]).toBe(LINE_STYLE_ACTIVE);
    expect(out[LINE_OFFSETS.alpha]).toBe(0.85);
    expect(out[LINE_OFFSETS.ownerId]).toBe(-1);
    expect(out[LINE_OFFSETS.face0]).toBe(-1);
    expect(out[LINE_OFFSETS.face1]).toBe(-1);
    expect(out.slice(LINE_OFFSETS.ownerCenter, LINE_OFFSETS.ownerCenter + 3)).toEqual([0, 0, 0]);
    expect(out.slice(LINE_OFFSETS.ownerExtent, LINE_OFFSETS.ownerExtent + 3)).toEqual([0, 0, 0]);

    // Defaults check on a second appended segment
    pushFreeGuideLine(out, 0, 0, 0, 1, 0, 0);
    expect(out).toHaveLength(LINE_INSTANCE_FLOATS * 2);
    const o = LINE_INSTANCE_FLOATS;
    expect(out[o + LINE_OFFSETS.cssWidth]).toBe(1.0);
    expect(out[o + LINE_OFFSETS.style]).toBe(LINE_STYLE_GUIDE);
    expect(out[o + LINE_OFFSETS.alpha]).toBe(0.5);
    expect(out[o + LINE_OFFSETS.ownerId]).toBe(-1);
    expect(out[o + LINE_OFFSETS.face0]).toBe(-1);
    expect(out[o + LINE_OFFSETS.face1]).toBe(-1);
  });

  it('packs pushBoxFrameXZ as 4 closed horizontal XZ perimeter edges', () => {
    const out: number[] = [];
    pushBoxFrameXZ(out, -2, -1, 3, 4, 0.25, 1.15, LINE_STYLE_BF16_GOLD, 0.62);

    expect(out).toHaveLength(4 * LINE_INSTANCE_FLOATS);
    const expectedEdges: readonly [number, number, number, number][] = [
      [-2, -1, 3, -1],
      [3, -1, 3, 4],
      [3, 4, -2, 4],
      [-2, 4, -2, -1],
    ];
    for (let i = 0; i < 4; i++) {
      const o = i * LINE_INSTANCE_FLOATS;
      const [ax, az, bx, bz] = expectedEdges[i];
      expect(out[o + LINE_OFFSETS.pointA]).toBe(ax);
      expect(out[o + LINE_OFFSETS.pointA + 1]).toBe(0.25);
      expect(out[o + LINE_OFFSETS.pointA + 2]).toBe(az);
      expect(out[o + LINE_OFFSETS.pointB]).toBe(bx);
      expect(out[o + LINE_OFFSETS.pointB + 1]).toBe(0.25);
      expect(out[o + LINE_OFFSETS.pointB + 2]).toBe(bz);
      expect(out[o + LINE_OFFSETS.cssWidth]).toBe(1.15);
      expect(out[o + LINE_OFFSETS.style]).toBe(LINE_STYLE_BF16_GOLD);
      expect(out[o + LINE_OFFSETS.alpha]).toBe(0.62);
      expect(out[o + LINE_OFFSETS.ownerId]).toBe(-1);
      expect(out[o + LINE_OFFSETS.face0]).toBe(-1);
      expect(out[o + LINE_OFFSETS.face1]).toBe(-1);
    }
  });

  it('packs pushDashedFrameXZ into clean finite dash segments and guards degenerate inputs', () => {
    const out: number[] = [];
    // Rectangle 1.0 x 0.5 with dash=0.22, gap=0.14 (step=0.36):
    //   length 1.0 -> ceil(1.0 / 0.36) = 3 dashes per horizontal edge (6)
    //   length 0.5 -> ceil(0.5 / 0.36) = 2 dashes per vertical edge (4)
    //   total = 10 dashes
    pushDashedFrameXZ(out, 0, 0, 1.0, 0.5, 0.02, LINE_STYLE_GUIDE, 0.40, 0.22, 0.14, 0.9);
    expect(out).toHaveLength(10 * LINE_INSTANCE_FLOATS);

    const packed = {
      lineData: new Float32Array(out),
      lineCount: out.length / LINE_INSTANCE_FLOATS,
    };
    assertAllGuideLinesValid(packed, 'pushDashedFrameXZ');

    for (let i = 0; i < packed.lineCount; i++) {
      const o = i * LINE_INSTANCE_FLOATS;
      const len = Math.hypot(
        packed.lineData[o + LINE_OFFSETS.pointB] - packed.lineData[o + LINE_OFFSETS.pointA],
        packed.lineData[o + LINE_OFFSETS.pointB + 2] - packed.lineData[o + LINE_OFFSETS.pointA + 2],
      );
      expect(len).toBeGreaterThan(1e-4);
      expect(len).toBeLessThanOrEqual(0.22 + 1e-5);
      expect(packed.lineData[o + LINE_OFFSETS.pointA + 1]).toBeCloseTo(0.02, 5);
      expect(packed.lineData[o + LINE_OFFSETS.pointB + 1]).toBeCloseTo(0.02, 5);
      expect(packed.lineData[o + LINE_OFFSETS.cssWidth]).toBeCloseTo(0.9, 5);
      expect(packed.lineData[o + LINE_OFFSETS.alpha]).toBeCloseTo(0.40, 5);
    }

    // Degenerate zero-area and non-positive dash guards emit 0 segments without throwing or hanging
    const empty: number[] = [];
    pushDashedFrameXZ(empty, 1, 1, 1, 1, 0);
    pushDashedFrameXZ(empty, 0, 0, 2, 2, 0, LINE_STYLE_GUIDE, 0.4, 0, 0.1);
    pushDashedFrameXZ(empty, 0, 0, 2, 2, 0, LINE_STYLE_GUIDE, 0.4, -0.2, 0.1);
    expect(empty).toHaveLength(0);
  });
});

describe('All 8 cutaway builders emit valid non-zero LinesPass guide lines (338/338)', () => {
  const model = createDefaultHybridModel();
  const fullExec = runHybridExecution(model, {
    tokenIds: defaultTokenIdsForLength(8),
  });

  it('emits properly packed, visible guide lines across all 5 cutaways (8 builders)', () => {
    const embedding = buildEmbeddingCutaway({
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

    const output = buildOutputCutaway({
      model,
      residualIn: fullExec.residualStreams[52],
      finalNorm: fullExec.finalNorm,
      lmHead: fullExec.lmHead,
      tokenLabels: fullExec.tokenLabels,
      startTokenOffset: 0,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 7,
      isArDecode: false,
    });

    const gqaPlan = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'near_plan',
    });

    const gqaSpatial = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    const mambaPlan = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'near_plan',
    });

    const mambaSpatial = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    const moePlan = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'near_plan',
    });

    const moeSpatial = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });

    const scenes: readonly [string, CutawaySceneGeometry, number][] = [
      ['embeddingCutaway', embedding, 4],
      ['outputCutaway', output, 2],
      ['gqaCutaway (near_plan)', gqaPlan, 6],
      ['gqaSpatial (three_quarter)', gqaSpatial, 6],
      ['mambaCutaway (near_plan)', mambaPlan, 9],
      ['mambaSpatial (three_quarter)', mambaSpatial, 25],
      ['moeCutaway (near_plan)', moePlan, 164],
      ['moeSpatial (three_quarter)', moeSpatial, 122],
    ];

    let totalLines = 0;
    for (const [name, scene, expectedCount] of scenes) {
      assertAllGuideLinesValid(scene, name);
      expect(scene.lineCount, `${name} expected lineCount`).toBe(expectedCount);
      totalLines += scene.lineCount;
    }
    expect(totalLines).toBe(338);
  });

  it('preserves valid packing across FP8 Selective (BF16 gold frames) and active-part/cell highlights', () => {
    const mambaSpatialFp8 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: 'mamba-head-3',
      hoveredPart: null,
      activeHead: 3,
      viewAngle: 'three_quarter',
    });
    assertAllGuideLinesValid(mambaSpatialFp8, 'mambaSpatial (fp8_selective)');

    let mambaGoldCount = 0;
    let mambaActiveCount = 0;
    for (let i = 0; i < mambaSpatialFp8.lineCount; i++) {
      const style = mambaSpatialFp8.lineData[i * LINE_INSTANCE_FLOATS + LINE_OFFSETS.style];
      if (style === LINE_STYLE_BF16_GOLD) mambaGoldCount++;
      if (style === LINE_STYLE_ACTIVE) mambaActiveCount++;
    }
    expect(mambaGoldCount).toBe(8); // 4 Conv1D base frame + 4 Conv1D vertical stanchions
    expect(mambaActiveCount).toBeGreaterThanOrEqual(1); // active lifted x-head anchor

    const moeSpatialFp8 = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: 'routed-bank',
      hoveredPart: null,
      viewAngle: 'three_quarter',
    });
    assertAllGuideLinesValid(moeSpatialFp8, 'moeSpatial (fp8_selective)');

    let moeGoldCount = 0;
    let moeActiveCount = 0;
    for (let i = 0; i < moeSpatialFp8.lineCount; i++) {
      const style = moeSpatialFp8.lineData[i * LINE_INSTANCE_FLOATS + LINE_OFFSETS.style];
      if (style === LINE_STYLE_BF16_GOLD) moeGoldCount++;
      if (style === LINE_STYLE_ACTIVE) moeActiveCount++;
    }
    expect(moeGoldCount).toBe(8); // 4 Top-6 bridge box frame + 4 Top-6 bridge vertical stanchions
    expect(moeActiveCount).toBeGreaterThanOrEqual(8); // 6 active pins + workbench frame + 2 callout leaders

    const gqaPlanT5 = buildGqaCutaway({
      bayId: 43,
      tokens: 5,
      precision: 'fp8_selective',
      selectedPart: 'kv-head-1',
      hoveredPart: null,
      viewAngle: 'near_plan',
    });
    assertAllGuideLinesValid(gqaPlanT5, 'gqaCutaway (T=5, kv-head-1)');
    expect(gqaPlanT5.lineCount).toBe(7); // 2 brackets + 1 spine + 4 ticks {1, 5, 8, 16}
  });
});
