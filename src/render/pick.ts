import type { Vec3 } from '../model/spec';
import { MAX_ZOOM, MIN_ZOOM, type CameraPose } from '../model/state';
import {
  add3,
  buildCameraSnapshot,
  dot3,
  normalize3,
  scale3,
  sub3,
  transformVec4,
  type AABB,
  type CameraSnapshot,
  type ViewportMetrics,
} from './camera';

export interface Ray {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

export interface PickBox {
  readonly bayId: number;
  readonly partId?: string;
  readonly min: Vec3;
  readonly max: Vec3;
}

export interface PickHit {
  readonly bayId: number;
  readonly partId?: string;
  readonly t: number;
  readonly point: Vec3;
  readonly isFascia: boolean;
}

const PARALLEL_EPS = 1e-8;
const TIE_EPS = 1e-6;

/**
 * Cast an orthographic world ray from client CSS coordinates per §4:
 * Subtract host origin and overlay top inset, map to NDC, unproject z=-1 and z=+1 via invViewProj.
 */
export function clientPointToRay(
  clientX: number,
  clientY: number,
  snapshot: CameraSnapshot,
): Ray {
  const { hostOriginX, hostOriginY, insetTop, innerWidth, innerHeight } = snapshot.metrics;
  const localX = clientX - hostOriginX;
  const localY = clientY - hostOriginY - insetTop;
  const ndcX = (2 * localX) / innerWidth - 1;
  const ndcY = 1 - (2 * localY) / innerHeight;

  const [nx, ny, nz, nw] = transformVec4(snapshot.invViewProj, ndcX, ndcY, -1, 1);
  const [fx, fy, fz, fw] = transformVec4(snapshot.invViewProj, ndcX, ndcY, 1, 1);

  const nearPoint: Vec3 = [nx / nw, ny / nw, nz / nw];
  const farPoint: Vec3 = [fx / fw, fy / fw, fz / fw];
  const direction = normalize3(sub3(farPoint, nearPoint));

  return { origin: nearPoint, direction };
}

/**
 * Axis-aligned slab intersection per §4.
 * A parallel axis (|dir| < 1e-8) misses only if origin lies outside [min, max].
 * Accepts when exit >= max(enter, 0); returns smallest nonnegative entry t or null.
 */
export function intersectRayAabb(ray: Ray, min: Vec3, max: Vec3): number | null {
  let enter = -Infinity;
  let exit = Infinity;

  for (let axis = 0; axis < 3; axis++) {
    const o = ray.origin[axis];
    const d = ray.direction[axis];
    const bMin = min[axis];
    const bMax = max[axis];

    if (Math.abs(d) < PARALLEL_EPS) {
      if (o < bMin || o > bMax) {
        return null;
      }
      continue;
    }

    const invD = 1 / d;
    let t0 = (bMin - o) * invD;
    let t1 = (bMax - o) * invD;
    if (t0 > t1) {
      const tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    if (t0 > enter) enter = t0;
    if (t1 < exit) exit = t1;
    if (exit < enter) return null;
  }

  const entry = Math.max(enter, 0);
  if (exit < entry) return null;
  return entry;
}

/** Intersect ray with plane passing through `planePoint` with unit normal `planeNormal`. */
export function intersectRayPlane(
  ray: Ray,
  planePoint: Vec3,
  planeNormal: Vec3,
): Vec3 | null {
  const denom = dot3(ray.direction, planeNormal);
  if (Math.abs(denom) < PARALLEL_EPS) return null;
  const t = dot3(sub3(planePoint, ray.origin), planeNormal) / denom;
  return add3(ray.origin, scale3(ray.direction, t));
}

/** Shortest Euclidean distance from world point `p` to the infinite line / ray. */
export function pointToRayDistance(point: Vec3, ray: Ray): number {
  const toPoint = sub3(point, ray.origin);
  const proj = Math.max(0, dot3(toPoint, ray.direction));
  const closest = add3(ray.origin, scale3(ray.direction, proj));
  const diff = sub3(point, closest);
  return Math.hypot(diff[0], diff[1], diff[2]);
}

function findNearestBoxHit(ray: Ray, boxes: readonly PickBox[], isFascia: boolean): PickHit | null {
  let bestT = Infinity;
  let bestBox: PickBox | null = null;

  for (const box of boxes) {
    const t = intersectRayAabb(ray, box.min, box.max);
    if (t === null) continue;
    if (
      bestBox === null ||
      t < bestT - TIE_EPS ||
      (Math.abs(t - bestT) <= TIE_EPS && box.bayId < bestBox.bayId)
    ) {
      bestT = t;
      bestBox = box;
    }
  }

  if (!bestBox) return null;
  return {
    bayId: bestBox.bayId,
    partId: bestBox.partId,
    t: bestT,
    point: add3(ray.origin, scale3(ray.direction, bestT)),
    isFascia,
  };
}

/**
 * Pick macro bay: test actual solids first (including rails & slices);
 * apply thin picking fascia only if real geometry misses (§4).
 */
export function pickMacroBay(
  ray: Ray,
  solidBoxes: readonly PickBox[],
  fasciaBoxes: readonly PickBox[],
): PickHit | null {
  const solidHit = findNearestBoxHit(ray, solidBoxes, false);
  if (solidHit) return solidHit;
  return findNearestBoxHit(ray, fasciaBoxes, true);
}

/**
 * Anchor zoom at a cursor or pinch midpoint (§4):
 * Intersect ray with plane through target normal to e before & after scaling,
 * translate target by oldHit - newHit.
 */
export function zoomCameraAtClientPoint(
  pose: CameraPose,
  bounds: AABB,
  metrics: ViewportMetrics,
  oldClientX: number,
  oldClientY: number,
  newZoomRaw: number,
  newClientX = oldClientX,
  newClientY = oldClientY,
): CameraPose {
  const clampedZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoomRaw));
  const beforeSnap = buildCameraSnapshot(pose, bounds, metrics);
  const oldRay = clientPointToRay(oldClientX, oldClientY, beforeSnap);
  const oldHit = intersectRayPlane(oldRay, pose.target, beforeSnap.basis.eyeDir);

  const scaledPose: CameraPose = {
    ...pose,
    zoom: clampedZoom,
  };
  if (!oldHit) return scaledPose;

  const afterSnap = buildCameraSnapshot(scaledPose, bounds, metrics);
  const newRay = clientPointToRay(newClientX, newClientY, afterSnap);
  const newHit = intersectRayPlane(newRay, scaledPose.target, afterSnap.basis.eyeDir);
  if (!newHit) return scaledPose;

  const shift = sub3(oldHit, newHit);
  return {
    ...scaledPose,
    target: add3(scaledPose.target, shift),
  };
}
