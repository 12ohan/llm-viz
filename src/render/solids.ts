import { SOLID_INSTANCE_BYTES } from './primitives';
import { linkProgram, PALETTE } from './device';

const SOLIDS_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aFaceId;

// Per-instance (48 bytes stride)
layout(location = 3) in vec3 iCenter;
layout(location = 4) in vec3 iExtent;
layout(location = 5) in float iCellFamily;
layout(location = 6) in float iFp8;
layout(location = 7) in float iPhase;
layout(location = 8) in float iBayId;
layout(location = 9) in float iCellScale;
layout(location = 10) in float iHighlight;

uniform mat4 uViewProj;

out vec3 vLocalWorld;
flat out vec3 vExtent;
flat out vec3 vNormal;
flat out int vFaceId;
flat out int vCellFamily;
flat out float vFp8;
flat out float vPhase;
flat out float vBayId;
flat out float vCellScale;
flat out int vHighlight;

void main() {
  vec3 localWorld = aPos * iExtent;
  vec3 worldPos = iCenter + localWorld;
  gl_Position = uViewProj * vec4(worldPos, 1.0);

  vLocalWorld = localWorld;
  vExtent = iExtent;
  vNormal = aNormal;
  vFaceId = int(floor(aFaceId + 0.5));
  vCellFamily = int(floor(iCellFamily + 0.5));
  vFp8 = iFp8;
  vPhase = iPhase;
  vBayId = iBayId;
  vCellScale = iCellScale;
  vHighlight = int(floor(iHighlight + 0.5));
}
`;

const SOLIDS_FS = `#version 300 es
precision highp float;

in vec3 vLocalWorld;
flat in vec3 vExtent;
flat in vec3 vNormal;
flat in int vFaceId;
flat in int vCellFamily;
flat in float vFp8;
flat in float vPhase;
flat in float vBayId;
flat in float vCellScale;
flat in int vHighlight;

uniform vec3 uSlabTop;
uniform vec3 uSlabFront;
uniform vec3 uSlabSide;
uniform vec3 uCellDark;
uniform vec3 uCellBlue;
uniform vec3 uCellGreen;
uniform vec3 uCellCyan;
uniform vec3 uCellAmber;
uniform vec3 uFp8Frosted;
uniform vec3 uBf16Gold;

out vec4 fragColor;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  // Base dark charcoal-slate directional face shading (#4B5262 top / #414756 front / #363C49 side)
  vec3 slabBase;
  float shadeMult;
  if (abs(vNormal.y) > 0.5) {
    slabBase = uSlabTop;
    shadeMult = 1.08;
  } else if (abs(vNormal.z) > 0.5) {
    slabBase = uSlabFront;
    shadeMult = 1.00;
  } else {
    slabBase = uSlabSide;
    shadeMult = 0.88;
  }

  // Face-local coordinates (u, v) in world units, shifted to start at 0 on each face corner
  float u;
  float v;
  float spanU;
  float spanV;
  if (vFaceId == 4 || vFaceId == 5) {
    u = vLocalWorld.x + vExtent.x * 0.5;
    v = vLocalWorld.y + vExtent.y * 0.5;
    spanU = vExtent.x;
    spanV = vExtent.y;
  } else if (vFaceId == 2 || vFaceId == 3) {
    u = vLocalWorld.x + vExtent.x * 0.5;
    v = -vLocalWorld.z + vExtent.z * 0.5;
    spanU = vExtent.x;
    spanV = vExtent.z;
  } else {
    u = -vLocalWorld.z + vExtent.z * 0.5;
    v = vLocalWorld.y + vExtent.y * 0.5;
    spanU = vExtent.z;
    spanV = vExtent.y;
  }

  vec3 color = slabBase;

  if (vCellFamily > 0) {
    vec3 accentColor = uCellBlue;
    if (vCellFamily == 2) accentColor = uCellGreen;
    else if (vCellFamily == 3) accentColor = uCellCyan;
    else if (vCellFamily == 4) accentColor = uCellAmber;

    float cellSize = max(0.08, vCellScale);
    vec2 gridCoord = vec2(u, v) / cellSize;
    vec2 cellId = floor(gridCoord);
    vec2 cellUv = fract(gridCoord);

    // Margin inside outer edge of slab so a crisp charcoal frame surrounds the matrix cells
    float border = cellSize * 0.22;
    float inFrame =
      step(border, u) * step(u, spanU - border) *
      step(border, v) * step(v, spanV - border);

    // fwidth()-antialiased square cell interior vs slate mortar gap
    vec2 fw = max(fwidth(gridCoord), vec2(0.0005));
    float maxFw = max(fw.x, fw.y);

    // Deterministic per-cell value hash (updates cleanly with phase for SSM registers / tokens)
    float h = hash21(cellId + vec2(vBayId * 7.13 + float(vFaceId) * 3.17, vPhase * 11.41));
    // Bi-modal activation/weight distribution: dark near-black cells vs bright accent cells
    float val = smoothstep(0.18, 0.88, h);
    vec3 cellCol = mix(uCellDark, accentColor, val) * shadeMult;

    // Crisp individual cell square with mortar inset = 0.14
    float m = 0.14;
    vec2 edgeLo = smoothstep(vec2(m) - fw * 0.6, vec2(m) + fw * 0.6, cellUv);
    vec2 edgeHi = 1.0 - smoothstep(vec2(1.0 - m) - fw * 0.6, vec2(1.0 - m) + fw * 0.6, cellUv);
    float cellMask = edgeLo.x * edgeLo.y * edgeHi.x * edgeHi.y * inFrame;

    // Macro average tint when cells shrink below Nyquist sampling resolution (zero moire!)
    float detailFade = 1.0 - smoothstep(0.32, 0.72, maxFw);
    vec3 macroAvg = mix(slabBase, mix(uCellDark, accentColor, 0.56) * shadeMult, 0.68);
    vec3 detailed = mix(slabBase, cellCol, cellMask);

    color = mix(macroAvg, detailed, detailFade);
  }

  // Unmistakable FP8 Selective visual contrast (§4.2):
  // Quantized 40 layers fade into cool pale frosted slate so 12 BF16-preserved layers pop!
  if (vFp8 > 0.5) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 desat = mix(color, vec3(luma), 0.72);
    color = mix(desat, uFp8Frosted * shadeMult, 0.56);
  } else if (vHighlight == 2) {
    // Subtle warm gold luminance lift for BF16-preserved layers in FP8 Selective mode
    color = mix(color, color + uBf16Gold * 0.12, 0.55);
  }

  // Subtle brightness lift on actively selected / hovered bay
  if (vHighlight == 1) {
    color = min(vec3(1.0), color * 1.12 + vec3(0.03));
  }

  fragColor = vec4(color, 1.0);
}
`;

function buildUnitCubeGeometry(): { vertices: Float32Array; indices: Uint16Array } {
  const verts: number[] = [];
  const indices: number[] = [];

  const faces: readonly {
    id: number;
    n: readonly [number, number, number];
    corners: readonly (readonly [number, number, number])[];
  }[] = [
    {
      id: 0, n: [1, 0, 0],
      corners: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]],
    },
    {
      id: 1, n: [-1, 0, 0],
      corners: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]],
    },
    {
      id: 2, n: [0, 1, 0],
      corners: [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]],
    },
    {
      id: 3, n: [0, -1, 0],
      corners: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]],
    },
    {
      id: 4, n: [0, 0, 1],
      corners: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
    },
    {
      id: 5, n: [0, 0, -1],
      corners: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]],
    },
  ];

  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    const base = f * 4;
    for (const c of face.corners) {
      verts.push(c[0], c[1], c[2], face.n[0], face.n[1], face.n[2], face.id);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  return {
    vertices: new Float32Array(verts),
    indices: new Uint16Array(indices),
  };
}

/**
 * Pass 1: Opaque instanced tensor slabs with procedural fwidth()-antialiased matrix cells.
 */
export class SolidsPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private cubeVbo!: WebGLBuffer;
  private cubeIbo!: WebGLBuffer;
  private instanceVbo!: WebGLBuffer;
  private uViewProj!: WebGLUniformLocation | null;
  private uSlabTop!: WebGLUniformLocation | null;
  private uSlabFront!: WebGLUniformLocation | null;
  private uSlabSide!: WebGLUniformLocation | null;
  private uCellDark!: WebGLUniformLocation | null;
  private uCellBlue!: WebGLUniformLocation | null;
  private uCellGreen!: WebGLUniformLocation | null;
  private uCellCyan!: WebGLUniformLocation | null;
  private uCellAmber!: WebGLUniformLocation | null;
  private uFp8Frosted!: WebGLUniformLocation | null;
  private uBf16Gold!: WebGLUniformLocation | null;
  private instanceCount = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  init(): void {
    const gl = this.gl;
    this.program = linkProgram(gl, SOLIDS_VS, SOLIDS_FS);
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj');
    this.uSlabTop = gl.getUniformLocation(this.program, 'uSlabTop');
    this.uSlabFront = gl.getUniformLocation(this.program, 'uSlabFront');
    this.uSlabSide = gl.getUniformLocation(this.program, 'uSlabSide');
    this.uCellDark = gl.getUniformLocation(this.program, 'uCellDark');
    this.uCellBlue = gl.getUniformLocation(this.program, 'uCellBlue');
    this.uCellGreen = gl.getUniformLocation(this.program, 'uCellGreen');
    this.uCellCyan = gl.getUniformLocation(this.program, 'uCellCyan');
    this.uCellAmber = gl.getUniformLocation(this.program, 'uCellAmber');
    this.uFp8Frosted = gl.getUniformLocation(this.program, 'uFp8Frosted');
    this.uBf16Gold = gl.getUniformLocation(this.program, 'uBf16Gold');

    const { vertices, indices } = buildUnitCubeGeometry();
    const vao = gl.createVertexArray();
    const cubeVbo = gl.createBuffer();
    const cubeIbo = gl.createBuffer();
    const instanceVbo = gl.createBuffer();
    if (!vao || !cubeVbo || !cubeIbo || !instanceVbo) {
      throw new Error('Failed to allocate SolidsPass WebGL resources');
    }
    this.vao = vao;
    this.cubeVbo = cubeVbo;
    this.cubeIbo = cubeIbo;
    this.instanceVbo = instanceVbo;

    gl.bindVertexArray(this.vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeVbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const vStride = 7 * 4;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, vStride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, vStride, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, vStride, 24);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cubeIbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    const bindInst = (loc: number, size: number, offset: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, SOLID_INSTANCE_BYTES, offset);
      gl.vertexAttribDivisor(loc, 1);
    };
    bindInst(3, 3, 0);   // iCenter
    bindInst(4, 3, 12);  // iExtent
    bindInst(5, 1, 24);  // iCellFamily
    bindInst(6, 1, 28);  // iFp8
    bindInst(7, 1, 32);  // iPhase
    bindInst(8, 1, 36);  // iBayId
    bindInst(9, 1, 40);  // iCellScale
    bindInst(10, 1, 44); // iHighlight

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  upload(data: Float32Array, count: number): void {
    const gl = this.gl;
    this.instanceCount = count;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  draw(viewProj: Float32Array): void {
    if (this.instanceCount <= 0) return;
    const gl = this.gl;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);
    gl.uniform3fv(this.uSlabTop, PALETTE.slabTop);
    gl.uniform3fv(this.uSlabFront, PALETTE.slabFront);
    gl.uniform3fv(this.uSlabSide, PALETTE.slabSide);
    gl.uniform3fv(this.uCellDark, PALETTE.cellDark);
    gl.uniform3fv(this.uCellBlue, PALETTE.cellBlue);
    gl.uniform3fv(this.uCellGreen, PALETTE.cellGreen);
    gl.uniform3fv(this.uCellCyan, PALETTE.cellCyan);
    gl.uniform3fv(this.uCellAmber, PALETTE.cellAmber);
    gl.uniform3fv(this.uFp8Frosted, PALETTE.fp8Frosted);
    gl.uniform3fv(this.uBf16Gold, PALETTE.bf16Gold);

    gl.frontFace(gl.CCW);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.BLEND);

    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1.0, 1.0);

    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    gl.bindVertexArray(null);

    gl.disable(gl.POLYGON_OFFSET_FILL);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.cubeVbo);
    gl.deleteBuffer(this.cubeIbo);
    gl.deleteBuffer(this.instanceVbo);
  }
}
