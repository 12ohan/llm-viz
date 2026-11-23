import { SOLID_INSTANCE_BYTES } from '../scene/spine';
import { linkProgram, PALETTE } from './device';

const SOLIDS_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aFaceId;

// Per-instance (48 bytes stride)
layout(location = 3) in vec3 iCenter;
layout(location = 4) in vec3 iExtent;
layout(location = 5) in float iPattern;
layout(location = 6) in float iFp8;
layout(location = 7) in float iPhase;
layout(location = 8) in float iBayId;
layout(location = 9) in float iFaceMask;
layout(location = 10) in float iTone;

uniform mat4 uViewProj;

out vec3 vLocalWorld;
flat out vec3 vNormal;
flat out int vFaceId;
flat out int vPattern;
flat out float vFp8;
flat out float vPhase;
flat out int vFaceMask;
flat out int vTone;

void main() {
  vec3 localWorld = aPos * iExtent;
  vec3 worldPos = iCenter + localWorld;
  gl_Position = uViewProj * vec4(worldPos, 1.0);

  vLocalWorld = localWorld;
  vNormal = aNormal;
  vFaceId = int(floor(aFaceId + 0.5));
  vPattern = int(floor(iPattern + 0.5));
  vFp8 = iFp8;
  vPhase = iPhase;
  vFaceMask = int(floor(iFaceMask + 0.5));
  vTone = int(floor(iTone + 0.5));
}
`;

const SOLIDS_FS = `#version 300 es
precision highp float;

in vec3 vLocalWorld;
flat in vec3 vNormal;
flat in int vFaceId;
flat in int vPattern;
flat in float vFp8;
flat in float vPhase;
flat in int vFaceMask;
flat in int vTone;

uniform vec3 uTopColor;
uniform vec3 uFrontColor;
uniform vec3 uRightColor;
uniform vec3 uGuideColor;
uniform vec3 uFp8Stone;
uniform vec3 uInkColor;

out vec4 fragColor;

float stripeCoverage(float q, float h) {
  float d = abs(fract(q + 0.5) - 0.5);
  float a = max(fwidth(q), 0.0001);
  float line = 1.0 - smoothstep(max(0.0, h - 0.5 * a), h + 0.5 * a, d);
  float fade = 1.0 - smoothstep(0.25, 0.50, a);
  return line * fade;
}

void main() {
  vec3 baseColor;
  if (abs(vNormal.y) > 0.5) {
    baseColor = uTopColor;
  } else if (abs(vNormal.z) > 0.5) {
    baseColor = uFrontColor;
  } else {
    baseColor = uRightColor;
  }

  // Subtle neutral shift for residual strip / guide rails / bookends
  if (vTone == 2) {
    baseColor = mix(baseColor, uGuideColor, 0.45);
  } else if (vTone == 1) {
    baseColor = mix(baseColor, uTopColor, 0.20);
  }

  // FP8 stone tint mixed 35% into base face (§2)
  if (vFp8 > 0.5) {
    baseColor = mix(baseColor, uFp8Stone, 0.35);
  }

  // Face-local world coordinates (u, v) per §2:
  // Front/back (+Z/-Z): +u -> +X, +v -> +Y
  // Top/bottom (+Y/-Y): +u -> +X, +v -> -Z
  // Right/left (+X/-X): +u -> -Z, +v -> +Y
  float u;
  float v;
  if (vFaceId == 4 || vFaceId == 5) {
    u = vLocalWorld.x;
    v = vLocalWorld.y;
  } else if (vFaceId == 2 || vFaceId == 3) {
    u = vLocalWorld.x;
    v = -vLocalWorld.z;
  } else {
    u = -vLocalWorld.z;
    v = vLocalWorld.y;
  }

  vec3 color = baseColor;

  // Family pattern (applied when this face bit is set in faceMask)
  bool hasFacePattern = (vFaceMask & (1 << vFaceId)) != 0;
  if (hasFacePattern) {
    if (vPattern == 1) {
      // Register front/top diagonal hatch
      float q = (u + v) / (1.41421356 * 0.32) + vPhase;
      float cov = stripeCoverage(q, 0.09);
      color = mix(color, uInkColor, 0.30 * cov);
    } else if (vPattern == 2) {
      // MoE cabinet front grid
      float cu = stripeCoverage(u / 0.24, 0.055);
      float cv = stripeCoverage(v / 0.24, 0.055);
      float cov = max(cu, cv);
      color = mix(color, uInkColor, 0.24 * cov);
    }
  }

  // Pattern 3: FP8 overlay applied after family pattern so precision & block identity stay independent (§2)
  if (vFp8 > 0.5) {
    float q3 = (u - 2.0 * v) / (2.23606798 * 0.55);
    float cov3 = stripeCoverage(q3, 0.07);
    color = mix(color, uInkColor, 0.18 * cov3);
  }

  fragColor = vec4(color, 1.0);
}
`;

/** Build static 24-vertex unit cube [-0.5, +0.5]^3 with CCW faces, outward normals, and face IDs 0..5. */
function buildUnitCubeGeometry(): { vertices: Float32Array; indices: Uint16Array } {
  // Each vertex: pos.xyz, normal.xyz, faceId (7 floats)
  const verts: number[] = [];
  const indices: number[] = [];

  const faces: readonly {
    id: number;
    n: readonly [number, number, number];
    corners: readonly (readonly [number, number, number])[];
  }[] = [
    // 0: +X
    {
      id: 0,
      n: [1, 0, 0],
      corners: [
        [0.5, -0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
      ],
    },
    // 1: -X
    {
      id: 1,
      n: [-1, 0, 0],
      corners: [
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
      ],
    },
    // 2: +Y
    {
      id: 2,
      n: [0, 1, 0],
      corners: [
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
      ],
    },
    // 3: -Y
    {
      id: 3,
      n: [0, -1, 0],
      corners: [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
      ],
    },
    // 4: +Z
    {
      id: 4,
      n: [0, 0, 1],
      corners: [
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
      ],
    },
    // 5: -Z
    {
      id: 5,
      n: [0, 0, -1],
      corners: [
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
      ],
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
 * Pass 1: Opaque instanced unit cuboids with slope-scaled polygonOffset(1.0, 1.0) (Patch 2).
 */
export class SolidsPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private cubeVbo!: WebGLBuffer;
  private cubeIbo!: WebGLBuffer;
  private instanceVbo!: WebGLBuffer;
  private uViewProj!: WebGLUniformLocation | null;
  private uTopColor!: WebGLUniformLocation | null;
  private uFrontColor!: WebGLUniformLocation | null;
  private uRightColor!: WebGLUniformLocation | null;
  private uGuideColor!: WebGLUniformLocation | null;
  private uFp8Stone!: WebGLUniformLocation | null;
  private uInkColor!: WebGLUniformLocation | null;
  private instanceCount = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  init(): void {
    const gl = this.gl;
    this.program = linkProgram(gl, SOLIDS_VS, SOLIDS_FS);
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj');
    this.uTopColor = gl.getUniformLocation(this.program, 'uTopColor');
    this.uFrontColor = gl.getUniformLocation(this.program, 'uFrontColor');
    this.uRightColor = gl.getUniformLocation(this.program, 'uRightColor');
    this.uGuideColor = gl.getUniformLocation(this.program, 'uGuideColor');
    this.uFp8Stone = gl.getUniformLocation(this.program, 'uFp8Stone');
    this.uInkColor = gl.getUniformLocation(this.program, 'uInkColor');

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

    // Static unit cube buffer: 7 floats (28 bytes) per vertex
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

    // Instance buffer: 48 bytes stride
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    const bindInst = (loc: number, size: number, offset: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, SOLID_INSTANCE_BYTES, offset);
      gl.vertexAttribDivisor(loc, 1);
    };
    bindInst(3, 3, 0);   // iCenter
    bindInst(4, 3, 12);  // iExtent
    bindInst(5, 1, 24);  // iPattern
    bindInst(6, 1, 28);  // iFp8
    bindInst(7, 1, 32);  // iPhase
    bindInst(8, 1, 36);  // iBayId
    bindInst(9, 1, 40);  // iFaceMask
    bindInst(10, 1, 44); // iTone

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
    gl.uniform3fv(this.uTopColor, PALETTE.topFace);
    gl.uniform3fv(this.uFrontColor, PALETTE.frontFace);
    gl.uniform3fv(this.uRightColor, PALETTE.rightFace);
    gl.uniform3fv(this.uGuideColor, PALETTE.guideTint);
    gl.uniform3fv(this.uFp8Stone, PALETTE.fp8Stone);
    gl.uniform3fv(this.uInkColor, PALETTE.ink);

    gl.frontFace(gl.CCW);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.BLEND);

    // Patch 2: slope-scaled polygon offset (1.0, 1.0)
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
