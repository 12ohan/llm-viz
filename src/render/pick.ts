import type { GqaCellRef, GqaTensorKey } from '../math/gqaSim';
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
  readonly tensorKey?: GqaTensorKey;
  readonly gridCols?: number;
  readonly gridRows?: number;
  readonly tokenIndex?: number;
  readonly min: Vec3;
  readonly max: Vec3;
}

export interface PickHit {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly tensorKey?: GqaTensorKey;
  readonly tokenIndex?: number;
  readonly hitRow?: number;
  readonly hitCol?: number;
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

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
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
  const point = add3(ray.origin, scale3(ray.direction, bestT));

  let hitRow: number | undefined;
  let hitCol: number | undefined;
  if (bestBox.gridCols && bestBox.gridRows && bestBox.gridCols > 0 && bestBox.gridRows > 0) {
    // Matches SheetsPass top-face UV: col 0 at min.x (-X), row 0 at min.z (-Z)
    const u = clamp01((point[0] - bestBox.min[0]) / Math.max(1e-6, bestBox.max[0] - bestBox.min[0]));
    const vFromTop = clamp01((point[2] - bestBox.min[2]) / Math.max(1e-6, bestBox.max[2] - bestBox.min[2]));
    hitCol = Math.min(bestBox.gridCols - 1, Math.max(0, Math.floor(u * bestBox.gridCols)));
    hitRow = Math.min(bestBox.gridRows - 1, Math.max(0, Math.floor(vFromTop * bestBox.gridRows)));
  }

  return {
    bayId: bestBox.bayId,
    partId: bestBox.partId,
    headIndex: bestBox.headIndex,
    kvHeadIndex: bestBox.kvHeadIndex,
    tensorKey: bestBox.tensorKey,
    tokenIndex: bestBox.tokenIndex,
    hitRow,
    hitCol,
    t: bestT,
    point,
    isFascia,
  };
}

/** Resolve a ray-cast PickHit on a live GQA sheet into an exact logical GqaCellRef. */
export function pickHitToGqaCell(
  hit: PickHit | null,
  activeGroup0Head = 0,
  activeGroup1Head = 16,
): GqaCellRef | null {
  if (!hit || !hit.tensorKey || hit.hitRow === undefined || hit.hitCol === undefined) {
    return null;
  }
  const tensor = hit.tensorKey;
  if (tensor === 'scores' || tensor === 'softmax') {
    const grp: 0 | 1 = hit.kvHeadIndex === 1 ? 1 : 0;
    const headIdx = grp === 0 ? activeGroup0Head : activeGroup1Head;
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      headIndex: headIdx,
      kvHeadIndex: grp,
    };
  }
  if (tensor === 'kv_k' || tensor === 'kv_v') {
    return {
      tensor,
      row: 0,
      col: hit.hitCol,
      kvHeadIndex: hit.kvHeadIndex === 1 ? 1 : 0,
      tokenIndex: hit.tokenIndex ?? 0,
    };
  }
  return {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    headIndex: hit.headIndex,
    kvHeadIndex: hit.kvHeadIndex,
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
