import { RIBBON_INSTANCE_BYTES } from '../scene/spine';
import { linkProgram, PALETTE } from './device';

const RIBBONS_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

// Per-instance (48 bytes stride)
layout(location = 2) in vec3 iCenter;
layout(location = 3) in vec3 iExtent;
layout(location = 4) in float iKind;    // 0=box, 1=arrow -X, 2=arrow +X, 3=arrow -Y
layout(location = 5) in float iColorId; // 0=emerald, 1=periwinkle, 2=cyan, 3=amber
layout(location = 6) in float iAlpha;

uniform mat4 uViewProj;

flat out vec3 vNormal;
flat out int vColorId;
flat out float vAlpha;

void main() {
  int kind = int(floor(iKind + 0.5));
  vec3 pos = aPos;

  // Taper leading half of box into a clean directional 3D block arrowhead
  if (kind == 1) {
    // Arrow pointing -X
    float t = clamp((-pos.x + 0.5), 0.0, 1.0); // 0 at tail (+0.5), 1 at tip (-0.5)
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.y *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.y *= 0.62;
    }
  } else if (kind == 2) {
    // Arrow pointing +X
    float t = clamp((pos.x + 0.5), 0.0, 1.0); // 0 at tail (-0.5), 1 at tip (+0.5)
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.y *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.y *= 0.62;
    }
  } else if (kind == 3) {
    // Arrow pointing -Y (downward flow)
    float t = clamp((-pos.y + 0.5), 0.0, 1.0); // 0 at top (+0.5), 1 at bottom (-0.5)
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.x *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.x *= 0.62;
    }
  }

  vec3 worldPos = iCenter + pos * iExtent;
  gl_Position = uViewProj * vec4(worldPos, 1.0);
  vNormal = aNormal;
  vColorId = int(floor(iColorId + 0.5));
  vAlpha = iAlpha;
}
`;

const RIBBONS_FS = `#version 300 es
precision highp float;

flat in vec3 vNormal;
flat in int vColorId;
flat in float vAlpha;

uniform vec3 uEmerald;
uniform vec3 uPeriwinkle;
uniform vec3 uCyan;
uniform vec3 uAmber;

out vec4 fragColor;

void main() {
  vec3 col = uEmerald;
  if (vColorId == 1) col = uPeriwinkle;
  else if (vColorId == 2) col = uCyan;
  else if (vColorId == 3) col = uAmber;

  float shade = abs(vNormal.y) > 0.5 ? 1.05 : (abs(vNormal.z) > 0.5 ? 0.98 : 0.90);
  fragColor = vec4(col * shade, vAlpha);
}
`;

function buildCubeGeometry(): { vertices: Float32Array; indices: Uint16Array } {
  const verts: number[] = [];
  const indices: number[] = [];
  const faces: readonly {
    n: readonly [number, number, number];
    corners: readonly (readonly [number, number, number])[];
  }[] = [
    { n: [1, 0, 0], corners: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]] },
    { n: [-1, 0, 0], corners: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]] },
    { n: [0, 1, 0], corners: [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]] },
    { n: [0, -1, 0], corners: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]] },
    { n: [0, 0, 1], corners: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
    { n: [0, 0, -1], corners: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]] },
  ];
  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    const base = f * 4;
    for (const c of face.corners) {
      verts.push(c[0], c[1], c[2], face.n[0], face.n[1], face.n[2]);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { vertices: new Float32Array(verts), indices: new Uint16Array(indices) };
}

/**
 * Pass 3: Translucent emerald residual ribbons & periwinkle directional 3D block arrows
 * (depthMask(false), alpha blended).
 */
export class RibbonsPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private cubeVbo!: WebGLBuffer;
  private cubeIbo!: WebGLBuffer;
  private instanceVbo!: WebGLBuffer;
  private uViewProj!: WebGLUniformLocation | null;
  private uEmerald!: WebGLUniformLocation | null;
  private uPeriwinkle!: WebGLUniformLocation | null;
  private uCyan!: WebGLUniformLocation | null;
  private uAmber!: WebGLUniformLocation | null;
  private instanceCount = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  init(): void {
    const gl = this.gl;
    this.program = linkProgram(gl, RIBBONS_VS, RIBBONS_FS);
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj');
    this.uEmerald = gl.getUniformLocation(this.program, 'uEmerald');
    this.uPeriwinkle = gl.getUniformLocation(this.program, 'uPeriwinkle');
    this.uCyan = gl.getUniformLocation(this.program, 'uCyan');
    this.uAmber = gl.getUniformLocation(this.program, 'uAmber');

    const { vertices, indices } = buildCubeGeometry();
    const vao = gl.createVertexArray();
    const cubeVbo = gl.createBuffer();
    const cubeIbo = gl.createBuffer();
    const instanceVbo = gl.createBuffer();
    if (!vao || !cubeVbo || !cubeIbo || !instanceVbo) {
      throw new Error('Failed to allocate RibbonsPass WebGL resources');
    }
    this.vao = vao;
    this.cubeVbo = cubeVbo;
    this.cubeIbo = cubeIbo;
    this.instanceVbo = instanceVbo;

    gl.bindVertexArray(this.vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeVbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.cubeIbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    const bindInst = (loc: number, size: number, offset: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, RIBBON_INSTANCE_BYTES, offset);
      gl.vertexAttribDivisor(loc, 1);
    };
    bindInst(2, 3, 0);  // iCenter
    bindInst(3, 3, 12); // iExtent
    bindInst(4, 1, 24); // iKind
    bindInst(5, 1, 28); // iColorId
    bindInst(6, 1, 32); // iAlpha

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
    gl.uniform3fv(this.uEmerald, PALETTE.ribbonEmerald);
    gl.uniform3fv(this.uPeriwinkle, PALETTE.ribbonPeriwinkle);
    gl.uniform3fv(this.uCyan, PALETTE.ribbonCyan);
    gl.uniform3fv(this.uAmber, PALETTE.ribbonAmber);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    gl.bindVertexArray(null);

    gl.depthMask(true);
    gl.disable(gl.BLEND);
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
