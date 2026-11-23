import type { Vec3 } from '../model/spec';
import { MAX_ZOOM, MIN_ZOOM, type CameraPose } from '../model/state';

export interface AABB {
  readonly min: Vec3;
  readonly max: Vec3;
}

export interface ViewportMetrics {
  readonly hostOriginX: number;
  readonly hostOriginY: number;
  readonly hostWidth: number;
  readonly hostHeight: number;
  readonly insetTop: number;
  readonly insetBottom: number;
  readonly innerWidth: number;
  readonly innerHeight: number;
  readonly dpr: number;
}

export interface CameraBasis {
  readonly eyeDir: Vec3; // unit vector from target toward eye
  readonly right: Vec3;  // unit screen-right vector R
  readonly up: Vec3;     // unit screen-up vector U
}

export interface CameraFit {
  readonly target: Vec3;
  readonly baseHalfHeight: number;
  readonly near: number;
  readonly far: number;
  readonly basis: CameraBasis;
}

export interface CameraSnapshot {
  readonly pose: CameraPose;
  readonly basis: CameraBasis;
  readonly eye: Vec3;
  readonly baseHalfHeight: number;
  readonly halfHeight: number;
  readonly halfWidth: number;
  readonly near: number;
  readonly far: number;
  readonly view: Float32Array;
  readonly proj: Float32Array;
  readonly viewProj: Float32Array;
  readonly invViewProj: Float32Array;
  readonly metrics: ViewportMetrics;
}

export interface ProjectedPoint {
  readonly x: number;
  readonly y: number;
  readonly ndcZ: number;
}

export const OVERLAY_INSET_TOP = 56;
export const OVERLAY_INSET_BOTTOM = 72;
export const EYE_DISTANCE = 80;

const DEG_TO_RAD = Math.PI / 180;

export function makeViewportMetrics(
  hostOriginX: number,
  hostOriginY: number,
  hostWidth: number,
  hostHeight: number,
  devicePixelRatio = 1,
  insetTop = OVERLAY_INSET_TOP,
  insetBottom = OVERLAY_INSET_BOTTOM,
): ViewportMetrics {
  const safeW = Math.max(1, hostWidth);
  const safeH = Math.max(1, hostHeight);
  const clampedTop = Math.min(insetTop, Math.max(0, safeH - 1));
  const clampedBottom = Math.min(insetBottom, Math.max(0, safeH - clampedTop - 1));
  const innerWidth = safeW;
  const innerHeight = Math.max(1, safeH - clampedTop - clampedBottom);
  const dpr = Math.max(1, Math.min(devicePixelRatio, 2));
  return {
    hostOriginX,
    hostOriginY,
    hostWidth: safeW,
    hostHeight: safeH,
    insetTop: clampedTop,
    insetBottom: clampedBottom,
    innerWidth,
    innerHeight,
    dpr,
  };
}

export function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-12) return [0, 0, 1];
  return [v[0] / len, v[1] / len, v[2] / len];
}

export function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale3(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

/**
 * Orthonormal camera basis from azimuth a and elevation b (degrees).
 * Eye direction from target: e = (sin a cos b, sin b, cos a cos b).
 * R = normalize(up × e), U = e × R.
 */
export function computeCameraBasis(azimuthDeg: number, elevationDeg: number): CameraBasis {
  const a = azimuthDeg * DEG_TO_RAD;
  const b = elevationDeg * DEG_TO_RAD;
  const cosB = Math.cos(b);
  const eyeDir: Vec3 = normalize3([Math.sin(a) * cosB, Math.sin(b), Math.cos(a) * cosB]);
  const upRef: Vec3 = [0, 1, 0];
  const right = normalize3(cross3(upRef, eyeDir));
  const up = normalize3(cross3(eyeDir, right));
  return { eyeDir, right, up };
}

export function aabbCorners(bounds: AABB): Vec3[] {
  const [x0, y0, z0] = bounds.min;
  const [x1, y1, z1] = bounds.max;
  return [
    [x0, y0, z0],
    [x1, y0, z0],
    [x0, y1, z0],
    [x1, y1, z0],
    [x0, y0, z1],
    [x1, y0, z1],
    [x0, y1, z1],
    [x1, y1, z1],
  ];
}

/**
 * Fit orthographic camera to world AABB per §3:
 * H0 = max(rangeU / (2 * 0.76), rangeR / (2 * 0.88 * A)).
 */
export function fitCameraToBounds(
  bounds: AABB,
  azimuthDeg: number,
  elevationDeg: number,
  innerWidth: number,
  innerHeight: number,
): CameraFit {
  const basis = computeCameraBasis(azimuthDeg, elevationDeg);
  const corners = aabbCorners(bounds);

  let minR = Infinity;
  let maxR = -Infinity;
  let minU = Infinity;
  let maxU = -Infinity;
  let minE = Infinity;
  let maxE = -Infinity;

  for (const p of corners) {
    const r = dot3(p, basis.right);
    const u = dot3(p, basis.up);
    const e = dot3(p, basis.eyeDir);
    if (r < minR) minR = r;
    if (r > maxR) maxR = r;
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (e < minE) minE = e;
    if (e > maxE) maxE = e;
  }

  const midR = 0.5 * (minR + maxR);
  const midU = 0.5 * (minU + maxU);
  const midE = 0.5 * (minE + maxE);

  const target: Vec3 = [
    midR * basis.right[0] + midU * basis.up[0] + midE * basis.eyeDir[0],
    midR * basis.right[1] + midU * basis.up[1] + midE * basis.eyeDir[1],
    midR * basis.right[2] + midU * basis.up[2] + midE * basis.eyeDir[2],
  ];

  const rangeR = maxR - minR;
  const rangeU = maxU - minU;
  const aspect = Math.max(1e-4, innerWidth / Math.max(1, innerHeight));
  const baseHalfHeight = Math.max(
    rangeU / (2 * 0.76),
    rangeR / (2 * 0.88 * aspect),
  );

  const eye = add3(target, scale3(basis.eyeDir, EYE_DISTANCE));
  const { near, far } = computeNearFar(bounds, eye, basis.eyeDir);

  return { target, baseHalfHeight, near, far, basis };
}

function computeNearFar(bounds: AABB, eye: Vec3, eyeDir: Vec3): { near: number; far: number } {
  const corners = aabbCorners(bounds);
  let minDist = Infinity;
  let maxDist = -Infinity;
  for (const p of corners) {
    // Forward direction is -eyeDir, so positive distance from eye along forward axis is dot(eye - p, eyeDir).
    const dist = dot3(sub3(eye, p), eyeDir);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;
  }
  const near = Math.max(0.01, minDist - 2);
  const far = Math.max(near + 1, maxDist + 2);
  return { near, far };
}

/** Column-major 4x4 look-at view matrix from eye, right (R), up (U), eyeDir (e). */
function makeViewMatrix(eye: Vec3, basis: CameraBasis): Float32Array {
  const { right: R, up: U, eyeDir: E } = basis;
  const tx = -dot3(R, eye);
  const ty = -dot3(U, eye);
  const tz = -dot3(E, eye);

  // Column-major layout: m[col * 4 + row]
  return new Float32Array([
    R[0], U[0], E[0], 0,
    R[1], U[1], E[1], 0,
    R[2], U[2], E[2], 0,
    tx,   ty,   tz,   1,
  ]);
}

/** Column-major 4x4 WebGL orthographic matrix (NDC z in [-1, +1]). */
function makeOrthoMatrix(
  halfWidth: number,
  halfHeight: number,
  near: number,
  far: number,
): Float32Array {
  const sx = 1 / halfWidth;
  const sy = 1 / halfHeight;
  const sz = -2 / (far - near);
  const tz = -(far + near) / (far - near);

  return new Float32Array([
    sx, 0,  0,  0,
    0,  sy, 0,  0,
    0,  0,  sz, 0,
    0,  0,  tz, 1,
  ]);
}

/** Multiply two column-major 4x4 matrices: out = a * b. */
export function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    const b0 = b[col * 4 + 0];
    const b1 = b[col * 4 + 1];
    const b2 = b[col * 4 + 2];
    const b3 = b[col * 4 + 3];
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b0 +
        a[1 * 4 + row] * b1 +
        a[2 * 4 + row] * b2 +
        a[3 * 4 + row] * b3;
    }
  }
  return out;
}

/** General 4x4 matrix inversion (column-major). */
export function invertMat4(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-18) {
    throw new Error('Singular matrix in invertMat4');
  }
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/** Transform vec4 (x, y, z, w) by column-major 4x4 matrix, returning [x, y, z, w]. */
export function transformVec4(
  m: Float32Array,
  x: number,
  y: number,
  z: number,
  w: number,
): readonly [number, number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12] * w,
    m[1] * x + m[5] * y + m[9] * z + m[13] * w,
    m[2] * x + m[6] * y + m[10] * z + m[14] * w,
    m[3] * x + m[7] * y + m[11] * z + m[15] * w,
  ];
}

export function buildCameraSnapshot(
  pose: CameraPose,
  bounds: AABB,
  metrics: ViewportMetrics,
): CameraSnapshot {
  const fit = fitCameraToBounds(
    bounds,
    pose.azimuthDeg,
    pose.elevationDeg,
    metrics.innerWidth,
    metrics.innerHeight,
  );
  const clampedZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, pose.zoom));
  const aspect = Math.max(1e-4, metrics.innerWidth / Math.max(1, metrics.innerHeight));
  const halfHeight = fit.baseHalfHeight / clampedZoom;
  const halfWidth = aspect * halfHeight;
  const eye = add3(pose.target, scale3(fit.basis.eyeDir, EYE_DISTANCE));
  const { near, far } = computeNearFar(bounds, eye, fit.basis.eyeDir);

  const view = makeViewMatrix(eye, fit.basis);
  const proj = makeOrthoMatrix(halfWidth, halfHeight, near, far);
  const viewProj = multiplyMat4(proj, view);
  const invViewProj = invertMat4(viewProj);

  return {
    pose: {
      target: [pose.target[0], pose.target[1], pose.target[2]],
      azimuthDeg: pose.azimuthDeg,
      elevationDeg: pose.elevationDeg,
      zoom: clampedZoom,
    },
    basis: fit.basis,
    eye,
    baseHalfHeight: fit.baseHalfHeight,
    halfHeight,
    halfWidth,
    near,
    far,
    view,
    proj,
    viewProj,
    invViewProj,
    metrics,
  };
}

export function projectWorldToNdc(p: Vec3, snapshot: CameraSnapshot): Vec3 {
  const [cx, cy, cz, cw] = transformVec4(snapshot.viewProj, p[0], p[1], p[2], 1);
  return [cx / cw, cy / cw, cz / cw];
}

/** Project world point into inner-viewport CSS pixels (origin top-left of inner drawing viewport). */
export function projectWorldToInnerCss(p: Vec3, snapshot: CameraSnapshot): ProjectedPoint {
  const [ndcX, ndcY, ndcZ] = projectWorldToNdc(p, snapshot);
  return {
    x: (ndcX + 1) * 0.5 * snapshot.metrics.innerWidth,
    y: (1 - ndcY) * 0.5 * snapshot.metrics.innerHeight,
    ndcZ,
  };
}

/** Project world point into host-relative CSS pixels (origin top-left of stage host, including top inset). */
export function projectWorldToHostCss(p: Vec3, snapshot: CameraSnapshot): ProjectedPoint {
  const inner = projectWorldToInnerCss(p, snapshot);
  return {
    x: inner.x,
    y: snapshot.metrics.insetTop + inner.y,
    ndcZ: inner.ndcZ,
  };
}

/** Project world point into client CSS coordinates (including host origin and overlay inset). */
export function projectWorldToClientCss(p: Vec3, snapshot: CameraSnapshot): ProjectedPoint {
  const host = projectWorldToHostCss(p, snapshot);
  return {
    x: snapshot.metrics.hostOriginX + host.x,
    y: snapshot.metrics.hostOriginY + host.y,
    ndcZ: host.ndcZ,
  };
}

/** Pan camera in the R,U plane by CSS pixel drag delta. */
export function panCameraByCssDelta(
  pose: CameraPose,
  snapshot: CameraSnapshot,
  deltaCssX: number,
  deltaCssY: number,
): CameraPose {
  const scale = (2 * snapshot.halfHeight) / snapshot.metrics.innerHeight;
  const shiftR = -deltaCssX * scale;
  const shiftU = deltaCssY * scale;
  const newTarget = add3(
    pose.target,
    add3(scale3(snapshot.basis.right, shiftR), scale3(snapshot.basis.up, shiftU)),
  );
  return {
    ...pose,
    target: newTarget,
  };
}

/**
 * Refit camera on host resize while preserving zoom and projected pan offset in units of H (§3).
 */
export function refitCameraOnResize(
  pose: CameraPose,
  bounds: AABB,
  oldMetrics: ViewportMetrics,
  newMetrics: ViewportMetrics,
): CameraPose {
  const oldFit = fitCameraToBounds(
    bounds,
    pose.azimuthDeg,
    pose.elevationDeg,
    oldMetrics.innerWidth,
    oldMetrics.innerHeight,
  );
  const newFit = fitCameraToBounds(
    bounds,
    pose.azimuthDeg,
    pose.elevationDeg,
    newMetrics.innerWidth,
    newMetrics.innerHeight,
  );
  const oldH = oldFit.baseHalfHeight / pose.zoom;
  const newH = newFit.baseHalfHeight / pose.zoom;
  const delta = sub3(pose.target, oldFit.target);
  const panRInH = dot3(delta, oldFit.basis.right) / oldH;
  const panUInH = dot3(delta, oldFit.basis.up) / oldH;

  const newTarget = add3(
    newFit.target,
    add3(
      scale3(newFit.basis.right, panRInH * newH),
      scale3(newFit.basis.up, panUInH * newH),
    ),
  );
  return {
    ...pose,
    target: newTarget,
  };
}
