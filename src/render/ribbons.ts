import { add3, cross3, dot3, normalize3, scale3, sub3 } from './camera';
import { linkProgram, PALETTE } from './device';
import { RIBBON_INSTANCE_BYTES, type Vec3 } from './primitives';
import { localToWorldOnFrame, type SheetFrame3D } from './sheetFrame';

const RIBBONS_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

// Per-instance (48 bytes stride)
layout(location = 2) in vec3 iCenter;
layout(location = 3) in vec3 iExtent;
layout(location = 4) in float iKind;    // 0=box, 1=-X, 2=+X, 3=-Y, 4=+Y, 5=+Z, 6=-Z
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
    float t = clamp((-pos.x + 0.5), 0.0, 1.0);
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.y *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.y *= 0.62;
    }
  } else if (kind == 2) {
    // Arrow pointing +X
    float t = clamp((pos.x + 0.5), 0.0, 1.0);
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.y *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.y *= 0.62;
    }
  } else if (kind == 3) {
    // Arrow pointing -Y (downward flow)
    float t = clamp((-pos.y + 0.5), 0.0, 1.0);
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.x *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.x *= 0.62;
    }
  } else if (kind == 4) {
    // Arrow pointing +Y (upward feedback return)
    float t = clamp((pos.y + 0.5), 0.0, 1.0);
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.x *= tip * 1.45;
      pos.z *= tip * 1.15;
    } else {
      pos.x *= 0.62;
    }
  } else if (kind == 5) {
    // Arrow pointing +Z (outward toward KV cache / viewer)
    float t = clamp((pos.z + 0.5), 0.0, 1.0);
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.x *= tip * 1.45;
      pos.y *= tip * 1.15;
    } else {
      pos.x *= 0.62;
    }
  } else if (kind == 6) {
    // Arrow pointing -Z (inward from KV cache)
    float t = clamp((-pos.z + 0.5), 0.0, 1.0);
    if (t > 0.55) {
      float tip = (1.0 - t) / 0.45;
      pos.x *= tip * 1.45;
      pos.y *= tip * 1.15;
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

// ============================================================================
// SMOOTH CONTINUOUS FLAT EXTRUDED QUAD FLOW RIBBONS (XZ Drafting-Table Plane)
// ============================================================================

/**
 * Interleaved vertex layout for FlatFlowRibbonsPass: 10 floats = 40 bytes per vertex.
 *   [0..2] aPos         vec3  world position (x, y, z)
 *   [3]    aEdge        float -1..+1 across ribbon width (0 at arrowhead apex)
 *   [4]    aAlong       float 0..1 normalized arc parameter
 *   [5..8] aColor       vec4  RGBA fill color
 *   [9]    aBorderAlpha float subtle edge stroke intensity (0..1)
 */
export const FLOW_RIBBON_VERTEX_FLOATS = 10;
const FLOW_RIBBON_VERTEX_BYTES = FLOW_RIBBON_VERTEX_FLOATS * 4;

export interface FlowRibbonColor {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
  readonly borderAlpha?: number;
}

export interface XZPoint {
  readonly x: number;
  readonly z: number;
}

/**
 * Builds smooth, continuous flat quad ribbons with optional rounded orthogonal corners,
 * cubic Bezier curves, and crisp flat triangular arrowheads — zero voxel beads or stair-steps.
 */
export class FlowRibbonBuilder {
  private readonly verts: number[] = [];
  private readonly indices: number[] = [];

  private pushVertex(
    x: number,
    y: number,
    z: number,
    edge: number,
    along: number,
    col: FlowRibbonColor,
  ): number {
    const idx = this.verts.length / FLOW_RIBBON_VERTEX_FLOATS;
    this.verts.push(
      x,
      y,
      z,
      edge,
      along,
      col.r,
      col.g,
      col.b,
      col.a,
      col.borderAlpha ?? 0.35,
    );
    return idx;
  }

  /** Extrude a smooth flat ribbon along an XZ polyline at constant elevation `y`. */
  addPolylineXZ(
    rawPoints: readonly XZPoint[],
    y: number,
    width: number,
    color: FlowRibbonColor,
    arrowHead = true,
    arrowLength = Math.max(width * 2.1, 0.22),
    arrowWidth = Math.max(width * 2.25, 0.24),
  ): void {
    if (rawPoints.length < 2) return;

    // Filter consecutive duplicate points
    const pts: XZPoint[] = [rawPoints[0]];
    for (let i = 1; i < rawPoints.length; i++) {
      const prev = pts[pts.length - 1];
      const cur = rawPoints[i];
      if (Math.hypot(cur.x - prev.x, cur.z - prev.z) > 1e-4) {
        pts.push(cur);
      }
    }
    if (pts.length < 2) return;

    // Compute cumulative arc length
    const cumLen: number[] = [0];
    let totalLen = 0;
    for (let i = 1; i < pts.length; i++) {
      totalLen += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z);
      cumLen.push(totalLen);
    }
    if (totalLen < 1e-4) return;

    // Trim shaft slightly before tip so flat triangular arrowhead meets cleanly with zero overlap
    const tip = pts[pts.length - 1];
    const effArrowLen = arrowHead ? Math.min(arrowLength, totalLen * 0.45) : 0;
    const shaftTargetLen = totalLen - effArrowLen;

    const shaftPts: XZPoint[] = [];
    const shaftCum: number[] = [];
    if (arrowHead && shaftTargetLen > 1e-4) {
      for (let i = 0; i < pts.length; i++) {
        if (cumLen[i] < shaftTargetLen) {
          shaftPts.push(pts[i]);
          shaftCum.push(cumLen[i]);
        } else {
          const prevI = Math.max(0, i - 1);
          const segLen = Math.max(1e-6, cumLen[i] - cumLen[prevI]);
          const f = Math.max(0, Math.min(1, (shaftTargetLen - cumLen[prevI]) / segLen));
          shaftPts.push({
            x: pts[prevI].x + (pts[i].x - pts[prevI].x) * f,
            z: pts[prevI].z + (pts[i].z - pts[prevI].z) * f,
          });
          shaftCum.push(shaftTargetLen);
          break;
        }
      }
    } else {
      for (let i = 0; i < pts.length; i++) {
        shaftPts.push(pts[i]);
        shaftCum.push(cumLen[i]);
      }
    }

    const halfW = width * 0.5;
    if (shaftPts.length >= 2) {
      const n = shaftPts.length;
      const baseIndices: number[] = [];

      for (let i = 0; i < n; i++) {
        const p = shaftPts[i];
        let tx = 0;
        let tz = 0;
        if (i === 0) {
          tx = shaftPts[1].x - p.x;
          tz = shaftPts[1].z - p.z;
        } else if (i === n - 1) {
          tx = p.x - shaftPts[n - 2].x;
          tz = p.z - shaftPts[n - 2].z;
        } else {
          // Average incoming and outgoing unit tangents for smooth miter joins
          const ax = p.x - shaftPts[i - 1].x;
          const az = p.z - shaftPts[i - 1].z;
          const al = Math.hypot(ax, az) || 1;
          const bx = shaftPts[i + 1].x - p.x;
          const bz = shaftPts[i + 1].z - p.z;
          const bl = Math.hypot(bx, bz) || 1;
          tx = ax / al + bx / bl;
          tz = az / al + bz / bl;
          if (Math.hypot(tx, tz) < 1e-4) {
            tx = bx / bl;
            tz = bz / bl;
          }
        }
        const tl = Math.hypot(tx, tz) || 1;
        tx /= tl;
        tz /= tl;

        // Perpendicular in XZ plane
        const nx = -tz;
        const nz = tx;
        const along = shaftCum[i] / totalLen;

        const vL = this.pushVertex(p.x + nx * halfW, y, p.z + nz * halfW, -1, along, color);
        const vR = this.pushVertex(p.x - nx * halfW, y, p.z - nz * halfW, 1, along, color);
        baseIndices.push(vL, vR);
      }

      for (let i = 0; i < n - 1; i++) {
        const l0 = baseIndices[i * 2];
        const r0 = baseIndices[i * 2 + 1];
        const l1 = baseIndices[(i + 1) * 2];
        const r1 = baseIndices[(i + 1) * 2 + 1];
        this.indices.push(l0, r0, l1, r0, r1, l1);
      }
    }

    if (arrowHead && effArrowLen > 1e-4) {
      const base = shaftPts[shaftPts.length - 1];
      let dx = tip.x - base.x;
      let dz = tip.z - base.z;
      const dl = Math.hypot(dx, dz) || 1;
      dx /= dl;
      dz /= dl;
      const nx = -dz;
      const nz = dx;
      const ahw = arrowWidth * 0.5;

      const vTip = this.pushVertex(tip.x, y, tip.z, 0, 1, color);
      const vLeft = this.pushVertex(base.x + nx * ahw, y, base.z + nz * ahw, -1, 0.96, color);
      const vRight = this.pushVertex(base.x - nx * ahw, y, base.z - nz * ahw, 1, 0.96, color);
      this.indices.push(vLeft, vRight, vTip);
    }
  }

  /** Smooth cubic Bezier curve in the XZ plane. */
  addCubicXZ(
    p0: XZPoint,
    c1: XZPoint,
    c2: XZPoint,
    p3: XZPoint,
    y: number,
    width: number,
    color: FlowRibbonColor,
    arrowHead = true,
    steps = 24,
  ): void {
    const pts: XZPoint[] = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const u = 1 - t;
      const u2 = u * u;
      const u3 = u2 * u;
      const t2 = t * t;
      const t3 = t2 * t;
      pts.push({
        x: u3 * p0.x + 3 * u2 * t * c1.x + 3 * u * t2 * c2.x + t3 * p3.x,
        z: u3 * p0.z + 3 * u2 * t * c1.z + 3 * u * t2 * c2.z + t3 * p3.z,
      });
    }
    this.addPolylineXZ(pts, y, width, color, arrowHead);
  }

  /** Smoothly rounded orthogonal (Manhattan-with-fillets) connector in XZ. */
  addRoundedOrthoXZ(
    waypoints: readonly XZPoint[],
    radius: number,
    y: number,
    width: number,
    color: FlowRibbonColor,
    arrowHead = true,
  ): void {
    if (waypoints.length < 2) return;
    if (waypoints.length === 2) {
      this.addPolylineXZ(waypoints, y, width, color, arrowHead);
      return;
    }

    const pts: XZPoint[] = [waypoints[0]];
    const arcSteps = 8;

    for (let i = 1; i < waypoints.length - 1; i++) {
      const prev = waypoints[i - 1];
      const curr = waypoints[i];
      const next = waypoints[i + 1];

      const ax = prev.x - curr.x;
      const az = prev.z - curr.z;
      const al = Math.hypot(ax, az);
      const bx = next.x - curr.x;
      const bz = next.z - curr.z;
      const bl = Math.hypot(bx, bz);

      if (al < 1e-4 || bl < 1e-4) {
        pts.push(curr);
        continue;
      }

      const r = Math.min(radius, al * 0.45, bl * 0.45);
      const start: XZPoint = {
        x: curr.x + (ax / al) * r,
        z: curr.z + (az / al) * r,
      };
      const end: XZPoint = {
        x: curr.x + (bx / bl) * r,
        z: curr.z + (bz / bl) * r,
      };

      // Quadratic fillet around corner `curr`
      for (let s = 0; s <= arcSteps; s++) {
        const t = s / arcSteps;
        const u = 1 - t;
        pts.push({
          x: u * u * start.x + 2 * u * t * curr.x + t * t * end.x,
          z: u * u * start.z + 2 * u * t * curr.z + t * t * end.z,
        });
      }
    }

    pts.push(waypoints[waypoints.length - 1]);
    this.addPolylineXZ(pts, y, width, color, arrowHead);
  }

  /**
   * Draw a crisp flat cell or row/column span highlight frame on top of a thin sheet:
   * a delicate translucent fill quad plus 4 sharp outer frame edges.
   */
  addCellHighlightXZ(
    cx: number,
    cz: number,
    ex: number,
    ez: number,
    y: number,
    fillColor: FlowRibbonColor,
    frameColor: FlowRibbonColor,
    frameThickness = 0.022,
  ): void {
    const hx = ex * 0.5;
    const hz = ez * 0.5;
    const x0 = cx - hx;
    const x1 = cx + hx;
    const z0 = cz - hz;
    const z1 = cz + hz;

    // 1. Subtle interior tint quad (along = -1 marks static cell highlight geometry)
    if (fillColor.a > 0.004) {
      const v0 = this.pushVertex(x0, y, z0, 0, -1, fillColor);
      const v1 = this.pushVertex(x1, y, z0, 0, -1, fillColor);
      const v2 = this.pushVertex(x1, y, z1, 0, -1, fillColor);
      const v3 = this.pushVertex(x0, y, z1, 0, -1, fillColor);
      this.indices.push(v0, v1, v2, v0, v2, v3);
    }

    // 2. Crisp 4-sided perimeter frame slightly elevated above the fill (static, non-pulsed)
    const fy = y + 0.002;
    const t = Math.min(frameThickness, Math.min(ex, ez) * 0.45 + 0.014);
    const halfT = t * 0.5;
    const emitStaticEdgeXZ = (ax: number, az: number, bx: number, bz: number, sideX: number, sideZ: number) => {
      const i0 = this.pushVertex(ax - sideX * halfT, fy, az - sideZ * halfT, -1, -1, frameColor);
      const i1 = this.pushVertex(ax + sideX * halfT, fy, az + sideZ * halfT, 1, -1, frameColor);
      const i2 = this.pushVertex(bx - sideX * halfT, fy, bz - sideZ * halfT, -1, -1, frameColor);
      const i3 = this.pushVertex(bx + sideX * halfT, fy, bz + sideZ * halfT, 1, -1, frameColor);
      this.indices.push(i0, i1, i2, i1, i3, i2);
    };
    emitStaticEdgeXZ(x0 - halfT, z0, x1 + halfT, z0, 0, 1);
    emitStaticEdgeXZ(x0 - halfT, z1, x1 + halfT, z1, 0, 1);
    emitStaticEdgeXZ(x0, z0 - halfT, x0, z1 + halfT, 1, 0);
    emitStaticEdgeXZ(x1, z0 - halfT, x1, z1 + halfT, 1, 0);
  }

  /**
   * Append a smooth 3D extruded ribbon along arbitrary 3D waypoints using a parallel-transport
   * (Bishop) frame so vertical climbs and cross-plane bends never twist or kink.
   * When `crossRibbon` is true, also adds a subtle perpendicular fin so 3D ribbons remain crisp
   * and legible from every orbit angle without vanishing edge-on.
   */
  addPolyline3D(
    rawPoints: readonly Vec3[],
    upHint: Vec3,
    width: number,
    color: FlowRibbonColor,
    arrowHead = false,
    arrowLength = 0.22,
    arrowWidth = 0.22,
    crossRibbon = true,
  ): void {
    const pts: Vec3[] = [];
    for (const p of rawPoints) {
      if (pts.length === 0) {
        pts.push(p);
      } else {
        const prev = pts[pts.length - 1];
        if (Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]) > 1e-4) {
          pts.push(p);
        }
      }
    }
    if (pts.length < 2) return;

    // Trim shaft slightly before tip for crisp arrowhead
    let tipPoint: Vec3 | null = null;
    let tipDir: Vec3 | null = null;
    const shaft: Vec3[] = pts.map((p) => [p[0], p[1], p[2]]);

    if (arrowHead) {
      const last = shaft[shaft.length - 1];
      const prev = shaft[shaft.length - 2];
      const segVec = sub3(last, prev);
      const segLen = Math.hypot(segVec[0], segVec[1], segVec[2]);
      if (segLen > 1e-4) {
        tipDir = [segVec[0] / segLen, segVec[1] / segLen, segVec[2] / segLen];
        tipPoint = [last[0], last[1], last[2]];
        const trim = Math.min(arrowLength * 0.84, segLen * 0.55);
        shaft[shaft.length - 1] = sub3(last, scale3(tipDir, trim));
      }
    }

    // Build per-vertex tangent T, normal N, binormal B via parallel transport
    const n = shaft.length;
    const tangents: Vec3[] = [];
    for (let i = 0; i < n; i++) {
      let dir: Vec3;
      if (i === 0) {
        dir = sub3(shaft[1], shaft[0]);
      } else if (i === n - 1) {
        dir = sub3(shaft[n - 1], shaft[n - 2]);
      } else {
        const dPrev = normalize3(sub3(shaft[i], shaft[i - 1]));
        const dNext = normalize3(sub3(shaft[i + 1], shaft[i]));
        const avg: Vec3 = [dPrev[0] + dNext[0], dPrev[1] + dNext[1], dPrev[2] + dNext[2]];
        dir = Math.hypot(avg[0], avg[1], avg[2]) > 1e-5 ? avg : dNext;
      }
      tangents.push(normalize3(dir));
    }

    const normals: Vec3[] = [];
    const binormals: Vec3[] = [];
    // Initial normal orthogonal to tangents[0]
    let n0Cand = sub3(upHint, scale3(tangents[0], dot3(upHint, tangents[0])));
    if (Math.hypot(n0Cand[0], n0Cand[1], n0Cand[2]) < 1e-5) {
      const alt: Vec3 = Math.abs(tangents[0][1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      n0Cand = sub3(alt, scale3(tangents[0], dot3(alt, tangents[0])));
    }
    normals.push(normalize3(n0Cand));
    binormals.push(normalize3(cross3(tangents[0], normals[0])));

    for (let i = 1; i < n; i++) {
      const prevN = normals[i - 1];
      const tCur = tangents[i];
      let nextN = sub3(prevN, scale3(tCur, dot3(prevN, tCur)));
      if (Math.hypot(nextN[0], nextN[1], nextN[2]) < 1e-5) {
        const prevB = binormals[i - 1];
        nextN = cross3(prevB, tCur);
      }
      const nUnit = normalize3(nextN);
      normals.push(nUnit);
      binormals.push(normalize3(cross3(tCur, nUnit)));
    }

    const emitStrip = (axes: readonly Vec3[], halfW: number, col: FlowRibbonColor) => {
      const baseIndex = this.verts.length / FLOW_RIBBON_VERTEX_FLOATS;
      for (let i = 0; i < n; i++) {
        const p = shaft[i];
        const side = axes[i];
        const along = n > 1 ? i / (n - 1) : 0;
        const pL = sub3(p, scale3(side, halfW));
        const pR = add3(p, scale3(side, halfW));
        this.pushVertex(pL[0], pL[1], pL[2], -1, along, col);
        this.pushVertex(pR[0], pR[1], pR[2], 1, along, col);
      }
      for (let i = 0; i < n - 1; i++) {
        const i0 = baseIndex + i * 2;
        const i1 = i0 + 1;
        const i2 = i0 + 2;
        const i3 = i0 + 3;
        this.indices.push(i0, i1, i2, i1, i3, i2);
      }
    };

    const halfW = width * 0.5;
    emitStrip(binormals, halfW, color);
    if (crossRibbon) {
      const finColor: FlowRibbonColor = {
        r: color.r,
        g: color.g,
        b: color.b,
        a: color.a * 0.78,
        borderAlpha: (color.borderAlpha ?? 0) * 0.7,
      };
      emitStrip(normals, halfW * 0.78, finColor);
    }

    if (arrowHead && tipPoint && tipDir) {
      const endB = binormals[n - 1];
      const endN = normals[n - 1];
      const basePt = sub3(tipPoint, scale3(tipDir, arrowLength));
      const ah = arrowWidth * 0.5;
      const emitHead = (side: Vec3, col: FlowRibbonColor) => {
        const pL = sub3(basePt, scale3(side, ah));
        const pR = add3(basePt, scale3(side, ah));
        const vL = this.pushVertex(pL[0], pL[1], pL[2], -0.9, 1, col);
        const vR = this.pushVertex(pR[0], pR[1], pR[2], 0.9, 1, col);
        const vT = this.pushVertex(tipPoint![0], tipPoint![1], tipPoint![2], 0.0, 1, col);
        this.indices.push(vL, vR, vT);
      };
      emitHead(endB, color);
      if (crossRibbon) {
        emitHead(endN, { ...color, a: color.a * 0.8 });
      }
    }
  }

  /** Append a smooth 3D cubic Bezier ribbon between arbitrary 3D points. */
  addCubic3D(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    upHint: Vec3,
    width: number,
    color: FlowRibbonColor,
    arrowHead = false,
    segments = 20,
    crossRibbon = true,
    arrowLength = 0.2,
    arrowWidth = 0.2,
  ): void {
    const pts: Vec3[] = [];
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const mt = 1 - t;
      const b0 = mt * mt * mt;
      const b1 = 3 * mt * mt * t;
      const b2 = 3 * mt * t * t;
      const b3 = t * t * t;
      pts.push([
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
        b0 * p0[2] + b1 * p1[2] + b2 * p2[2] + b3 * p3[2],
      ]);
    }
    this.addPolyline3D(pts, upHint, width, color, arrowHead, arrowLength, arrowWidth, crossRibbon);
  }

  /**
   * Draw a crisp cell or row/column span highlight flush on an oriented `SheetFrame3D` plane.
   * When `doubleSided` is true, draws the identical local-coordinate highlight slightly proud
   * of BOTH the `+N` and `-N` faces so inspecting from either side shows the exact same cell.
   */
  addCellHighlightOnFrame(
    frame: SheetFrame3D,
    rowStart: number,
    rowEnd: number,
    colStart: number,
    colEnd: number,
    gridRows: number,
    gridCols: number,
    normalOffset: number,
    fillColor: FlowRibbonColor,
    frameColor: FlowRibbonColor,
    frameThickness = 0.022,
    doubleSided = true,
  ): void {
    const c0 = Math.max(0, Math.min(gridCols - 1, colStart));
    const c1 = Math.max(c0, Math.min(gridCols - 1, colEnd));
    const r0 = Math.max(0, Math.min(gridRows - 1, rowStart));
    const r1 = Math.max(r0, Math.min(gridRows - 1, rowEnd));

    const lx0 = -0.5 + c0 / gridCols;
    const lx1 = -0.5 + (c1 + 1) / gridCols;
    const lz0 = -0.5 + r0 / gridRows;
    const lz1 = -0.5 + (r1 + 1) / gridRows;

    const spanU = (lx1 - lx0) * frame.extent[0];
    const spanV = (lz1 - lz0) * frame.extent[2];
    const t = Math.min(frameThickness, Math.min(spanU, spanV) * 0.45 + 0.014);
    const halfT = t * 0.5;
    const halfTuLocal = halfT / Math.max(1e-5, frame.extent[0]);
    const halfTvLocal = halfT / Math.max(1e-5, frame.extent[2]);

    const emitFace = (sign: 1 | -1) => {
      const lyFill = sign * (0.5 + normalOffset / Math.max(1e-5, frame.extent[1]));
      const lyEdge = sign * (0.5 + (normalOffset + 0.002) / Math.max(1e-5, frame.extent[1]));

      const p00 = localToWorldOnFrame(frame, lx0, lyFill, lz0);
      const p10 = localToWorldOnFrame(frame, lx1, lyFill, lz0);
      const p11 = localToWorldOnFrame(frame, lx1, lyFill, lz1);
      const p01 = localToWorldOnFrame(frame, lx0, lyFill, lz1);

      if (fillColor.a > 0.004) {
        const v0 = this.pushVertex(p00[0], p00[1], p00[2], 0, -1, fillColor);
        const v1 = this.pushVertex(p10[0], p10[1], p10[2], 0, -1, fillColor);
        const v2 = this.pushVertex(p11[0], p11[1], p11[2], 0, -1, fillColor);
        const v3 = this.pushVertex(p01[0], p01[1], p01[2], 0, -1, fillColor);
        this.indices.push(v0, v1, v2, v0, v2, v3);
      }

      const emitEdgeQuad = (a: Vec3, b: Vec3, sideAxis: Vec3) => {
        const aL = sub3(a, scale3(sideAxis, halfT));
        const aR = add3(a, scale3(sideAxis, halfT));
        const bL = sub3(b, scale3(sideAxis, halfT));
        const bR = add3(b, scale3(sideAxis, halfT));
        const i0 = this.pushVertex(aL[0], aL[1], aL[2], -1, -1, frameColor);
        const i1 = this.pushVertex(aR[0], aR[1], aR[2], 1, -1, frameColor);
        const i2 = this.pushVertex(bL[0], bL[1], bL[2], -1, -1, frameColor);
        const i3 = this.pushVertex(bR[0], bR[1], bR[2], 1, -1, frameColor);
        this.indices.push(i0, i1, i2, i1, i3, i2);
      };

      // Top & bottom edges (along axisU, width along axisV)
      emitEdgeQuad(
        localToWorldOnFrame(frame, lx0 - halfTuLocal, lyEdge, lz0),
        localToWorldOnFrame(frame, lx1 + halfTuLocal, lyEdge, lz0),
        frame.axisV,
      );
      emitEdgeQuad(
        localToWorldOnFrame(frame, lx0 - halfTuLocal, lyEdge, lz1),
        localToWorldOnFrame(frame, lx1 + halfTuLocal, lyEdge, lz1),
        frame.axisV,
      );
      // Left & right edges (along axisV, width along axisU)
      emitEdgeQuad(
        localToWorldOnFrame(frame, lx0, lyEdge, lz0 - halfTvLocal),
        localToWorldOnFrame(frame, lx0, lyEdge, lz1 + halfTvLocal),
        frame.axisU,
      );
      emitEdgeQuad(
        localToWorldOnFrame(frame, lx1, lyEdge, lz0 - halfTvLocal),
        localToWorldOnFrame(frame, lx1, lyEdge, lz1 + halfTvLocal),
        frame.axisU,
      );
    };

    emitFace(1);
    if (doubleSided) {
      emitFace(-1);
    }
  }

  pack(): { vertices: Float32Array; indices: Uint32Array } {
    return {
      vertices: new Float32Array(this.verts),
      indices: new Uint32Array(this.indices),
    };
  }
}

const FLAT_FLOW_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aEdge;
layout(location = 2) in float aAlong;
layout(location = 3) in vec4 aColor;
layout(location = 4) in float aBorderAlpha;

uniform mat4 uViewProj;

out float vEdge;
out float vAlong;
out vec4 vColor;
out float vBorderAlpha;

void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  vEdge = aEdge;
  vAlong = aAlong;
  vColor = aColor;
  vBorderAlpha = aBorderAlpha;
}
`;

const FLAT_FLOW_FS = `#version 300 es
precision highp float;

in float vEdge;
in float vAlong;
in vec4 vColor;
in float vBorderAlpha;

uniform float uPulseTime;

out vec4 fragColor;

void main() {
  float dist = abs(vEdge);
  float fw = max(fwidth(dist), 0.015);
  // Crisp antialiased outer ribbon boundary
  float cov = 1.0 - smoothstep(1.0 - fw * 1.15, 1.0 + fw * 0.35, dist);
  // Subtle darker border along the ribbon edges for drafting-table precision
  float border = smoothstep(1.0 - fw * 2.6, 1.0 - fw * 0.9, dist) * vBorderAlpha;
  vec3 col = mix(vColor.rgb, vColor.rgb * 0.68, border);
  float alpha = vColor.a * cov;

  // V2 animated directional dataflow pulse along active high-alpha dependency threads
  // (vAlong < 0 marks static cell-highlight frames; uPulseTime < 0 renders completely static)
  if (uPulseTime >= 0.0 && vColor.a >= 0.65 && vAlong >= -0.001) {
    float phase = fract(vAlong * 1.4 - uPulseTime * 0.85);
    float band = smoothstep(0.18, 0.54, phase) * (1.0 - smoothstep(0.54, 0.72, phase));
    float core = 1.0 - smoothstep(0.25, 0.92, dist);
    float pulse = band * core;
    col = mix(col, mix(col, vec3(1.0, 0.99, 0.97), 0.78), pulse * 0.58);
    alpha = min(1.0, alpha + pulse * 0.14 * cov);
  }

  if (alpha < 0.004) discard;
  fragColor = vec4(col, alpha);
}
`;

/**
 * Smooth continuous flat extruded quad ribbon pass (no voxel cubes or stair-stepping).
 * Alpha-blended, depth-tested with depthMask(false), double-sided for both 3/4 and near-plan views.
 * Supports optional V2 directional dataflow pulse animation via `draw(viewProj, pulseTime)`.
 */
export class FlatFlowRibbonsPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private vbo!: WebGLBuffer;
  private ibo!: WebGLBuffer;
  private uViewProj!: WebGLUniformLocation | null;
  private uPulseTime!: WebGLUniformLocation | null;
  private indexCount = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  init(): void {
    const gl = this.gl;
    this.program = linkProgram(gl, FLAT_FLOW_VS, FLAT_FLOW_FS);
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj');
    this.uPulseTime = gl.getUniformLocation(this.program, 'uPulseTime');

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    const ibo = gl.createBuffer();
    if (!vao || !vbo || !ibo) {
      throw new Error('Failed to allocate FlatFlowRibbonsPass WebGL resources');
    }
    this.vao = vao;
    this.vbo = vbo;
    this.ibo = ibo;

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, FLOW_RIBBON_VERTEX_BYTES, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, FLOW_RIBBON_VERTEX_BYTES, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, FLOW_RIBBON_VERTEX_BYTES, 16);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, FLOW_RIBBON_VERTEX_BYTES, 20);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, FLOW_RIBBON_VERTEX_BYTES, 36);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  upload(vertices: Float32Array, indices: Uint32Array): void {
    const gl = this.gl;
    this.indexCount = indices.length;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  draw(viewProj: Float32Array, pulseTime = -1): void {
    if (this.indexCount <= 0) return;
    const gl = this.gl;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);
    gl.uniform1f(this.uPulseTime, pulseTime);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);

    gl.enable(gl.CULL_FACE);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    gl.deleteBuffer(this.ibo);
  }
}
