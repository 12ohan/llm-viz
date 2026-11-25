import type { Vec3 } from '../model/spec';
import { linkProgram, PALETTE } from './device';
import {
  LINE_INSTANCE_ATTRIBS,
  LINE_INSTANCE_BYTES,
  LINE_INSTANCE_FLOATS,
  LINE_OFFSETS,
} from './linePacking';

export {
  LINE_INSTANCE_ATTRIBS,
  LINE_INSTANCE_BYTES,
  LINE_INSTANCE_FLOATS,
  LINE_OFFSETS,
};

/**
 * Pass 2: Crisp depth-tested screen-space hairlines for perspective camera (no gl_FragDepth).
 * Supports charcoal edges, subtle guides, warm gold BF16-preserved highlights, and vivid active strokes.
 */
const LINES_VS = `#version 300 es
precision highp float;

layout(location = 0) in float aEnd;
layout(location = 1) in float aSide;

// Per-instance (72 bytes stride)
layout(location = 2) in vec3 iPointA;
layout(location = 3) in vec3 iPointB;
layout(location = 4) in float iCssWidth;
layout(location = 5) in float iStyle;
layout(location = 6) in float iAlpha;
layout(location = 7) in float iOwnerId;
layout(location = 8) in float iFace0;
layout(location = 9) in float iFace1;
layout(location = 10) in vec3 iOwnerCenter;
layout(location = 11) in vec3 iOwnerExtent;

uniform mat4 uViewProj;
uniform vec2 uViewportSize;
uniform float uDpr;
uniform vec3 uCameraPos;

out float vTransverse;
flat out float vHalfWidth;
flat out float vAlpha;
flat out int vStyle;

vec3 faceNormal(int faceId) {
  if (faceId == 0) return vec3(1.0, 0.0, 0.0);
  if (faceId == 1) return vec3(-1.0, 0.0, 0.0);
  if (faceId == 2) return vec3(0.0, 1.0, 0.0);
  if (faceId == 3) return vec3(0.0, -1.0, 0.0);
  if (faceId == 4) return vec3(0.0, 0.0, 1.0);
  return vec3(0.0, 0.0, -1.0);
}

// Homogeneous Liang-Barsky clip against perspective/ortho frustum (-w <= x,y,z <= w)
bool clipSegmentToFrustum(inout vec4 ca, inout vec4 cb) {
  float t0 = 0.0;
  float t1 = 1.0;

  for (int i = 0; i < 6; i++) {
    float da;
    float db;
    if (i == 0) { da = ca.w + ca.x; db = cb.w + cb.x; }
    else if (i == 1) { da = ca.w - ca.x; db = cb.w - cb.x; }
    else if (i == 2) { da = ca.w + ca.y; db = cb.w + cb.y; }
    else if (i == 3) { da = ca.w - ca.y; db = cb.w - cb.y; }
    else if (i == 4) { da = ca.w + ca.z; db = cb.w + cb.z; }
    else { da = ca.w - ca.z; db = cb.w - cb.z; }

    if (da < 0.0 && db < 0.0) return false;
    if (da < 0.0 || db < 0.0) {
      float t = da / (da - db);
      if (da < 0.0) t0 = max(t0, t);
      else t1 = min(t1, t);
      if (t0 > t1) return false;
    }
  }

  vec4 origA = ca;
  vec4 origB = cb;
  ca = mix(origA, origB, t0);
  cb = mix(origA, origB, t1);
  return true;
}

void main() {
  float _unused = iOwnerId * 0.0 + (iOwnerCenter.x + iOwnerExtent.x) * 0.0;

  // Perspective front-face edge culling using view vector from edge midpoint to camera position
  int f0 = int(floor(iFace0 + 0.5));
  int f1 = int(floor(iFace1 + 0.5));
  if (f0 >= 0 && f1 >= 0) {
    vec3 mid = 0.5 * (iPointA + iPointB);
    vec3 toCam = normalize(uCameraPos - mid);
    float d0 = dot(faceNormal(f0), toCam);
    float d1 = dot(faceNormal(f1), toCam);
    if (d0 <= 1e-4 && d1 <= 1e-4) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      vTransverse = 0.0;
      vHalfWidth = 0.0;
      vAlpha = 0.0;
      vStyle = 0;
      return;
    }
  }

  vec4 ca = uViewProj * vec4(iPointA, 1.0);
  vec4 cb = uViewProj * vec4(iPointB, 1.0);

  if (!clipSegmentToFrustum(ca, cb) || ca.w <= 1e-5 || cb.w <= 1e-5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vTransverse = 0.0;
    vHalfWidth = 0.0;
    vAlpha = 0.0;
    vStyle = 0;
    return;
  }

  vec2 pa = (ca.xy / ca.w * 0.5 + 0.5) * uViewportSize;
  vec2 pb = (cb.xy / cb.w * 0.5 + 0.5) * uViewportSize;
  vec2 dp = pb - pa;
  float segLen = length(dp);

  if (segLen < 0.25) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vTransverse = 0.0;
    vHalfWidth = 0.0;
    vAlpha = 0.0;
    vStyle = 0;
    return;
  }

  vec2 normalPx = vec2(-dp.y, dp.x) / segLen;
  float halfW = iCssWidth * uDpr * 0.5;
  float feather = 0.5;
  float expand = halfW + feather;

  vec4 c = (aEnd < 0.5) ? ca : cb;
  vec2 p = (aEnd < 0.5) ? pa : pb;
  vec2 pOffset = p + aSide * normalPx * expand;

  vec4 outClip = c;
  outClip.xy = (2.0 * pOffset / uViewportSize - 1.0) * c.w;
  // Subtle clip-z pull without gl_FragDepth
  outClip.z -= (4.0 * c.w) / 16777215.0 + _unused;

  gl_Position = outClip;
  vTransverse = aSide * expand;
  vHalfWidth = halfW;
  vAlpha = iAlpha;
  vStyle = int(floor(iStyle + 0.5));
}
`;

const LINES_FS = `#version 300 es
precision highp float;

in float vTransverse;
flat in float vHalfWidth;
flat in float vAlpha;
flat in int vStyle;

uniform vec3 uHairlineColor;
uniform vec3 uActiveColor;
uniform vec3 uGuideColor;
uniform vec3 uGoldColor;

out vec4 fragColor;

void main() {
  float feather = 0.5;
  float cov = 1.0 - smoothstep(max(0.0, vHalfWidth - feather), vHalfWidth + feather, abs(vTransverse));
  if (cov <= 0.001) {
    discard;
  }

  vec3 col = uHairlineColor;
  if (vStyle == 1) col = uActiveColor;
  else if (vStyle == 2) col = uGuideColor;
  else if (vStyle == 3) col = uGoldColor;

  fragColor = vec4(col, vAlpha * cov);
}
`;

export class LinesPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private quadVbo!: WebGLBuffer;
  private quadIbo!: WebGLBuffer;
  private instanceVbo!: WebGLBuffer;
  private uViewProj!: WebGLUniformLocation | null;
  private uViewportSize!: WebGLUniformLocation | null;
  private uDpr!: WebGLUniformLocation | null;
  private uCameraPos!: WebGLUniformLocation | null;
  private uHairlineColor!: WebGLUniformLocation | null;
  private uActiveColor!: WebGLUniformLocation | null;
  private uGuideColor!: WebGLUniformLocation | null;
  private uGoldColor!: WebGLUniformLocation | null;
  private instanceCount = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  init(): void {
    const gl = this.gl;
    this.program = linkProgram(gl, LINES_VS, LINES_FS);
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj');
    this.uViewportSize = gl.getUniformLocation(this.program, 'uViewportSize');
    this.uDpr = gl.getUniformLocation(this.program, 'uDpr');
    this.uCameraPos = gl.getUniformLocation(this.program, 'uCameraPos');
    this.uHairlineColor = gl.getUniformLocation(this.program, 'uHairlineColor');
    this.uActiveColor = gl.getUniformLocation(this.program, 'uActiveColor');
    this.uGuideColor = gl.getUniformLocation(this.program, 'uGuideColor');
    this.uGoldColor = gl.getUniformLocation(this.program, 'uGoldColor');

    const quadVerts = new Float32Array([
      0, -1,
      0,  1,
      1,  1,
      1, -1,
    ]);
    const quadIndices = new Uint16Array([0, 1, 2, 0, 2, 3]);

    const vao = gl.createVertexArray();
    const quadVbo = gl.createBuffer();
    const quadIbo = gl.createBuffer();
    const instanceVbo = gl.createBuffer();
    if (!vao || !quadVbo || !quadIbo || !instanceVbo) {
      throw new Error('Failed to allocate LinesPass WebGL resources');
    }
    this.vao = vao;
    this.quadVbo = quadVbo;
    this.quadIbo = quadIbo;
    this.instanceVbo = instanceVbo;

    gl.bindVertexArray(this.vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 8, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 8, 4);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.quadIbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, quadIndices, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    const bindInst = (loc: number, size: number, offset: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, LINE_INSTANCE_BYTES, offset);
      gl.vertexAttribDivisor(loc, 1);
    };
    for (const attr of LINE_INSTANCE_ATTRIBS) {
      bindInst(attr.location, attr.size, attr.byteOffset);
    }

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

  draw(
    viewProj: Float32Array,
    vpWidth: number,
    vpHeight: number,
    dpr: number,
    cameraPos: Vec3,
  ): void {
    if (this.instanceCount <= 0) return;
    const gl = this.gl;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);
    gl.uniform2f(this.uViewportSize, vpWidth, vpHeight);
    gl.uniform1f(this.uDpr, dpr);
    gl.uniform3f(this.uCameraPos, cameraPos[0], cameraPos[1], cameraPos[2]);
    gl.uniform3fv(this.uHairlineColor, PALETTE.hairline);
    gl.uniform3fv(this.uActiveColor, PALETTE.activeStroke);
    gl.uniform3fv(this.uGuideColor, PALETTE.guideLine);
    gl.uniform3fv(this.uGoldColor, PALETTE.bf16Gold);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    gl.bindVertexArray(null);

    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.quadVbo);
    gl.deleteBuffer(this.quadIbo);
    gl.deleteBuffer(this.instanceVbo);
  }
}
