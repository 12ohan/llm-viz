import type { Vec3 } from '../model/spec';
import {
  DEFAULT_FOV_DEG,
  MAX_CAMERA_DISTANCE,
  MAX_ELEVATION_DEG,
  MIN_CAMERA_DISTANCE,
  MIN_ELEVATION_DEG,
  type CameraPose,
} from '../model/state';

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

export interface CameraSnapshot {
  readonly pose: CameraPose;
  readonly basis: CameraBasis;
  readonly eye: Vec3;
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
  readonly visible: boolean;
}

const DEG_TO_RAD = Math.PI / 180;

export function makeViewportMetrics(
  hostOriginX: number,
  hostOriginY: number,
  hostWidth: number,
  hostHeight: number,
  devicePixelRatio = 1,
  insetTop = 0,
  insetBottom = 0,
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

export function computeCameraBasis(azimuthDeg: number, elevationDeg: number): CameraBasis {
  const a = azimuthDeg * DEG_TO_RAD;
  const b = Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, elevationDeg)) * DEG_TO_RAD;
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

export interface OpticalOffsetNdc {
  /** Optional horizontal NDC shift applied to the projected target (positive shifts geometry right). */
  readonly x?: number;
  /** Optional vertical NDC shift applied to the projected target (positive shifts geometry up). */
  readonly y?: number;
}

/**
 * Compute a perspective orbit pose that cleanly frames `bounds` inside the viewport.
 * Uses exact per-corner projective distance so tall or deep bounds (like the 52-layer tower)
 * fill the intended fraction without excessive depth-diagonal backoff, while keeping every
 * corner strictly inside `[-fillFraction, +fillFraction]`.
 */
export function fitPerspectiveToBounds(
  bounds: AABB,
  azimuthDeg: number,
  elevationDeg: number,
  innerWidth: number,
  innerHeight: number,
  fovDeg = DEFAULT_FOV_DEG,
  fillFraction = 0.86,
  opticalOffsetNdc?: OpticalOffsetNdc,
): CameraPose {
  const basis = computeCameraBasis(azimuthDeg, elevationDeg);
  const corners = aabbCorners(bounds);

  let minR = Infinity, maxR = -Infinity;
  let minU = Infinity, maxU = -Infinity;
  let minE = Infinity, maxE = -Infinity;

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

  const aspect = Math.max(0.25, innerWidth / Math.max(1, innerHeight));
  const tanHalfFovY = Math.tan((fovDeg * 0.5) * DEG_TO_RAD);
  const tanHalfFovX = tanHalfFovY * aspect;
  const safeFill = Math.max(0.15, Math.min(0.98, Number.isFinite(fillFraction) ? fillFraction : 0.86));

  let requiredDist = MIN_CAMERA_DISTANCE;
  for (const p of corners) {
    const dr = Math.abs(dot3(p, basis.right) - midR);
    const du = Math.abs(dot3(p, basis.up) - midU);
    const de = dot3(p, basis.eyeDir) - midE;
    const dCorner = Math.max(
      dr / (tanHalfFovX * safeFill) + de,
      du / (tanHalfFovY * safeFill) + de,
      de + 2.0,
    );
    if (Number.isFinite(dCorner) && dCorner > requiredDist) {
      requiredDist = dCorner;
    }
  }

  const distance = Math.max(
    MIN_CAMERA_DISTANCE,
    Math.min(MAX_CAMERA_DISTANCE, requiredDist),
  );

  const ndcShiftX = Number.isFinite(opticalOffsetNdc?.x) ? opticalOffsetNdc!.x! : 0;
  const ndcShiftY = Number.isFinite(opticalOffsetNdc?.y) ? opticalOffsetNdc!.y! : 0;
  const targetShiftR = -ndcShiftX * distance * tanHalfFovX;
  const targetShiftU = -ndcShiftY * distance * tanHalfFovY;

  const target: Vec3 = [
    (midR + targetShiftR) * basis.right[0] + (midU + targetShiftU) * basis.up[0] + midE * basis.eyeDir[0],
    (midR + targetShiftR) * basis.right[1] + (midU + targetShiftU) * basis.up[1] + midE * basis.eyeDir[1],
    (midR + targetShiftR) * basis.right[2] + (midU + targetShiftU) * basis.up[2] + midE * basis.eyeDir[2],
  ];

  return {
    target,
    azimuthDeg,
    elevationDeg: Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, elevationDeg)),
    distance,
    fovDeg,
  };
}

export interface OpBoundsFitOptions {
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
  readonly fovDeg?: number;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly leftInsetPx?: number;
  readonly rightInsetPx?: number;
  /** Target fill fraction of the unobstructed viewport band (defaults to 0.80, i.e. ~75–85%). */
  readonly fillFraction?: number;
  readonly fillX?: number;
  readonly fillY?: number;
  readonly isNarrow?: boolean;
  readonly minDistance?: number;
}

/**
 * Projective, inset-aware perspective camera fitter for any operation AABB.
 * Computes the tightest perspective distance that frames every corner of `bounds`
 * inside the unobstructed canvas band `[leftInset, innerWidth - rightInset] × [topInset, innerHeight - bottomInset]`
 * at ~75–85% fill (default 0.80), shifting the camera target along screen-up / screen-right
 * so the operation cluster sits centered between the floating top toolbar and bottom Math Dock.
 */
export function fitOpBoundsCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: OpBoundsFitOptions,
): CameraPose {
  const safeW = Math.max(1, Number.isFinite(metrics.innerWidth) ? metrics.innerWidth : 960);
  const safeH = Math.max(1, Number.isFinite(metrics.innerHeight) ? metrics.innerHeight : 640);
  const isNarrow = options?.isNarrow ?? safeW < 760;

  const azimuthDeg = Number.isFinite(options?.azimuthDeg)
    ? options!.azimuthDeg!
    : isNarrow
      ? 17
      : 20;
  const rawElevation = Number.isFinite(options?.elevationDeg)
    ? options!.elevationDeg!
    : isNarrow
      ? 25
      : 23.5;
  const elevationDeg = Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, rawElevation));
  const rawFov = Number.isFinite(options?.fovDeg) ? options!.fovDeg! : DEFAULT_FOV_DEG;
  const fovDeg = Math.max(12, Math.min(110, rawFov));

  const rawTop = Math.max(0, Number.isFinite(options?.topInsetPx) ? options!.topInsetPx! : isNarrow ? 12 : 84);
  const rawBottom = Math.max(0, Number.isFinite(options?.bottomInsetPx) ? options!.bottomInsetPx! : isNarrow ? 12 : 118);
  const rawLeft = Math.max(0, Number.isFinite(options?.leftInsetPx) ? options!.leftInsetPx! : 0);
  const rawRight = Math.max(0, Number.isFinite(options?.rightInsetPx) ? options!.rightInsetPx! : 0);

  const clampedTop = Math.min(rawTop, safeH * 0.40);
  const clampedBottom = Math.min(rawBottom, safeH * 0.40);
  const clampedLeft = Math.min(rawLeft, safeW * 0.40);
  const clampedRight = Math.min(rawRight, safeW * 0.40);

  const usableW = Math.max(1, safeW - clampedLeft - clampedRight);
  const usableH = Math.max(1, safeH - clampedTop - clampedBottom);
  const usableFracX = Math.max(0.20, Math.min(1, usableW / safeW));
  const usableFracY = Math.max(0.20, Math.min(1, usableH / safeH));

  const baseFill = Math.max(
    0.25,
    Math.min(0.98, Number.isFinite(options?.fillFraction) ? options!.fillFraction! : 0.80),
  );
  const fillX = Math.max(
    0.25,
    Math.min(0.98, Number.isFinite(options?.fillX) ? options!.fillX! : baseFill),
  );
  const fillY = Math.max(
    0.25,
    Math.min(0.98, Number.isFinite(options?.fillY) ? options!.fillY! : baseFill),
  );

  // Normalize bounds defensively so inverted or non-finite values never produce NaN
  const safeVal = (v: number): number => (Number.isFinite(v) ? v : 0);
  const normBounds: AABB = {
    min: [
      Math.min(safeVal(bounds.min[0]), safeVal(bounds.max[0])),
      Math.min(safeVal(bounds.min[1]), safeVal(bounds.max[1])),
      Math.min(safeVal(bounds.min[2]), safeVal(bounds.max[2])),
    ],
    max: [
      Math.max(safeVal(bounds.min[0]), safeVal(bounds.max[0])),
      Math.max(safeVal(bounds.min[1]), safeVal(bounds.max[1])),
      Math.max(safeVal(bounds.min[2]), safeVal(bounds.max[2])),
    ],
  };

  const basis = computeCameraBasis(azimuthDeg, elevationDeg);
  const corners = aabbCorners(normBounds);

  let minR = Infinity, maxR = -Infinity;
  let minU = Infinity, maxU = -Infinity;
  let minE = Infinity, maxE = -Infinity;

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

  const aspect = Math.max(0.25, safeW / safeH);
  const tanHalfFovY = Math.tan((fovDeg * 0.5) * DEG_TO_RAD);
  const tanHalfFovX = tanHalfFovY * aspect;

  // Target NDC intervals strictly inside the unobstructed viewport band
  const ndcXCenter = (clampedLeft - clampedRight) / safeW;
  const ndcXHalfSpan = Math.min(usableFracX, Math.max(0.12, fillX * usableFracX));
  const ndcXMax = ndcXCenter + ndcXHalfSpan;
  const ndcXMin = ndcXCenter - ndcXHalfSpan;

  const ndcYCenter = (clampedBottom - clampedTop) / safeH;
  const ndcYHalfSpan = Math.min(usableFracY, Math.max(0.12, fillY * usableFracY));
  const ndcYMax = ndcYCenter + ndcYHalfSpan;
  const ndcYMin = ndcYCenter - ndcYHalfSpan;

  const minDist = Math.max(
    MIN_CAMERA_DISTANCE,
    Number.isFinite(options?.minDistance) ? options!.minDistance! : MIN_CAMERA_DISTANCE,
  );
  let requiredDist = minDist;

  for (const p of corners) {
    const dr = dot3(p, basis.right) - midR;
    const du = dot3(p, basis.up) - midU;
    const de = dot3(p, basis.eyeDir) - midE;

    const dXRight = (dr / tanHalfFovX + de * ndcXMax) / ndcXHalfSpan;
    const dXLeft = (-dr / tanHalfFovX - de * ndcXMin) / ndcXHalfSpan;
    const dYTop = (du / tanHalfFovY + de * ndcYMax) / ndcYHalfSpan;
    const dYBot = (-du / tanHalfFovY - de * ndcYMin) / ndcYHalfSpan;
    const dCorner = Math.max(dXRight, dXLeft, dYTop, dYBot, de + 2.0);
    if (Number.isFinite(dCorner) && dCorner > requiredDist) {
      requiredDist = dCorner;
    }
  }

  const distance = Math.max(minDist, Math.min(MAX_CAMERA_DISTANCE, requiredDist));
  const targetShiftR = -ndcXCenter * distance * tanHalfFovX;
  const targetShiftU = -ndcYCenter * distance * tanHalfFovY;

  const target: Vec3 = [
    (midR + targetShiftR) * basis.right[0] + (midU + targetShiftU) * basis.up[0] + midE * basis.eyeDir[0],
    (midR + targetShiftR) * basis.right[1] + (midU + targetShiftU) * basis.up[1] + midE * basis.eyeDir[1],
    (midR + targetShiftR) * basis.right[2] + (midU + targetShiftU) * basis.up[2] + midE * basis.eyeDir[2],
  ];

  return {
    target,
    azimuthDeg,
    elevationDeg,
    distance,
    fovDeg,
  };
}

function makeViewMatrix(eye: Vec3, basis: CameraBasis): Float32Array {
  const { right: R, up: U, eyeDir: E } = basis;
  const tx = -dot3(R, eye);
  const ty = -dot3(U, eye);
  const tz = -dot3(E, eye);

  return new Float32Array([
    R[0], U[0], E[0], 0,
    R[1], U[1], E[1], 0,
    R[2], U[2], E[2], 0,
    tx,   ty,   tz,   1,
  ]);
}

/** Column-major WebGL perspective projection matrix (NDC z in [-1, +1]). */
function makePerspectiveMatrix(
  fovDeg: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const f = 1 / Math.tan((fovDeg * 0.5) * DEG_TO_RAD);
  const nf = 1 / (near - far);

  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
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

function invertMat4(m: Float32Array): Float32Array {
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
  sceneBounds: AABB,
  metrics: ViewportMetrics,
): CameraSnapshot {
  const clampedElev = Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, pose.elevationDeg));
  const clampedDist = Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, pose.distance));
  const basis = computeCameraBasis(pose.azimuthDeg, clampedElev);
  const eye = add3(pose.target, scale3(basis.eyeDir, clampedDist));

  // Tight near/far envelope around the scene bounds for high 24-bit depth precision
  const corners = aabbCorners(sceneBounds);
  let minDepth = Infinity;
  let maxDepth = -Infinity;
  for (const c of corners) {
    const d = dot3(sub3(eye, c), basis.eyeDir);
    if (d < minDepth) minDepth = d;
    if (d > maxDepth) maxDepth = d;
  }
  const near = Math.max(0.25, Math.min(clampedDist * 0.08, minDepth - 2));
  const far = Math.max(near + 20, maxDepth + 40, clampedDist + 160);

  const aspect = Math.max(0.1, metrics.innerWidth / Math.max(1, metrics.innerHeight));
  const view = makeViewMatrix(eye, basis);
  const proj = makePerspectiveMatrix(pose.fovDeg, aspect, near, far);
  const viewProj = multiplyMat4(proj, view);
  const invViewProj = invertMat4(viewProj);

  return {
    pose: {
      target: [pose.target[0], pose.target[1], pose.target[2]],
      azimuthDeg: pose.azimuthDeg,
      elevationDeg: clampedElev,
      distance: clampedDist,
      fovDeg: pose.fovDeg,
    },
    basis,
    eye,
    near,
    far,
    view,
    proj,
    viewProj,
    invViewProj,
    metrics,
  };
}

export function projectWorldToInnerCss(p: Vec3, snapshot: CameraSnapshot): ProjectedPoint {
  const [cx, cy, cz, cw] = transformVec4(snapshot.viewProj, p[0], p[1], p[2], 1);
  if (cw <= 1e-5) {
    return { x: -9999, y: -9999, ndcZ: 2, visible: false };
  }
  const ndcX = cx / cw;
  const ndcY = cy / cw;
  const ndcZ = cz / cw;
  const visible = ndcZ >= -1 && ndcZ <= 1;
  return {
    x: (ndcX + 1) * 0.5 * snapshot.metrics.innerWidth,
    y: (1 - ndcY) * 0.5 * snapshot.metrics.innerHeight,
    ndcZ,
    visible,
  };
}

export function projectWorldToHostCss(p: Vec3, snapshot: CameraSnapshot): ProjectedPoint {
  const inner = projectWorldToInnerCss(p, snapshot);
  return {
    x: inner.x,
    y: snapshot.metrics.insetTop + inner.y,
    ndcZ: inner.ndcZ,
    visible: inner.visible,
  };
}

export function projectWorldToClientCss(p: Vec3, snapshot: CameraSnapshot): ProjectedPoint {
  const host = projectWorldToHostCss(p, snapshot);
  return {
    x: snapshot.metrics.hostOriginX + host.x,
    y: snapshot.metrics.hostOriginY + host.y,
    ndcZ: host.ndcZ,
    visible: host.visible,
  };
}

/** Orbit camera around active target (left-drag). */
export function orbitCameraByCssDelta(
  pose: CameraPose,
  deltaCssX: number,
  deltaCssY: number,
): CameraPose {
  const sensitivity = 0.34;
  return {
    ...pose,
    azimuthDeg: pose.azimuthDeg - deltaCssX * sensitivity,
    elevationDeg: Math.max(
      MIN_ELEVATION_DEG,
      Math.min(MAX_ELEVATION_DEG, pose.elevationDeg + deltaCssY * sensitivity),
    ),
  };
}

/** Pan camera smoothly in the camera view plane (right-drag or Shift+left-drag). */
export function panCameraByCssDelta(
  pose: CameraPose,
  snapshot: CameraSnapshot,
  deltaCssX: number,
  deltaCssY: number,
): CameraPose {
  const worldHeightAtTarget =
    2 * snapshot.pose.distance * Math.tan((snapshot.pose.fovDeg * 0.5) * DEG_TO_RAD);
  const scale = worldHeightAtTarget / Math.max(1, snapshot.metrics.innerHeight);
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

/** Smoothly zoom camera distance (scroll / pinch). */
export function zoomPerspectiveDistance(pose: CameraPose, scaleFactor: number): CameraPose {
  const nextDist = Math.max(
    MIN_CAMERA_DISTANCE,
    Math.min(MAX_CAMERA_DISTANCE, pose.distance * scaleFactor),
  );
  return {
    ...pose,
    distance: nextDist,
  };
}

/** Smoothstep ease t^2 * (3 - 2t) between two perspective camera poses. */
export function interpolateCameraPose(from: CameraPose, to: CameraPose, rawT: number): CameraPose {
  const t = Math.max(0, Math.min(1, rawT));
  const e = t * t * (3 - 2 * t);

  // Shortest angular arc for azimuth
  let dAz = ((to.azimuthDeg - from.azimuthDeg + 180) % 360 + 360) % 360 - 180;
  if (dAz < -180) dAz += 360;

  return {
    target: [
      from.target[0] + (to.target[0] - from.target[0]) * e,
      from.target[1] + (to.target[1] - from.target[1]) * e,
      from.target[2] + (to.target[2] - from.target[2]) * e,
    ],
    azimuthDeg: from.azimuthDeg + dAz * e,
    elevationDeg: from.elevationDeg + (to.elevationDeg - from.elevationDeg) * e,
    distance: from.distance + (to.distance - from.distance) * e,
    fovDeg: from.fovDeg + (to.fovDeg - from.fovDeg) * e,
  };
}
