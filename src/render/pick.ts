import type { Vec3 } from '../model/spec';
import {
  add3,
  dot3,
  normalize3,
  scale3,
  sub3,
  transformVec4,
  type CameraSnapshot,
} from './camera';

export interface Ray {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

export interface PickBox {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly min: Vec3;
  readonly max: Vec3;
}

export interface PickHit {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly t: number;
  readonly point: Vec3;
  readonly isFascia: boolean;
}

const PARALLEL_EPS = 1e-8;
const TIE_EPS = 1e-6;

/**
 * Unproject client CSS coordinates through invViewProj at NDC z = -1 and z = +1.
 * Identical formula works seamlessly for both perspective and orthographic cameras.
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
 * Axis-aligned slab intersection.
 * Returns smallest nonnegative ray distance t, or null on miss.
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

/** Shortest Euclidean distance from world point `p` to ray. */
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
    headIndex: bestBox.headIndex,
    kvHeadIndex: bestBox.kvHeadIndex,
    t: bestT,
    point: add3(ray.origin, scale3(ray.direction, bestT)),
    isFascia,
  };
}

/** Test real solid AABBs first; fall back to fascia boxes only if solids miss. */
export function pickMacroBay(
  ray: Ray,
  solidBoxes: readonly PickBox[],
  fasciaBoxes: readonly PickBox[],
): PickHit | null {
  const solidHit = findNearestBoxHit(ray, solidBoxes, false);
  if (solidHit) return solidHit;
  return findNearestBoxHit(ray, fasciaBoxes, true);
}
