import {
  add3,
  dot3,
  normalize3,
  scale3,
  sub3,
  transformVec4,
  type CameraSnapshot,
} from './camera';
import type { Vec3 } from './primitives';
import {
  intersectRaySheetFrame,
  type SheetFrame3D,
  type SheetFrameRayHit,
} from './sheetFrame';

export interface Ray {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

export interface PickBox {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly tensorKey?: string;
  readonly gridCols?: number;
  readonly gridRows?: number;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  /** Optional multi-tower ownership metadata (used by TwoTower spatial views; omitted in Nano). */
  readonly towerRole?: 'context' | 'denoiser' | 'shared';
  readonly executionId?: string;
  readonly traceOrigin?: string;
  readonly semanticTag?: string;
  readonly absoluteTokenOffset?: number;
  readonly min: Vec3;
  readonly max: Vec3;
  /** Optional oriented 3D sheet frame. When provided, exact nonuniform-extent OBB picking is used. */
  readonly frame?: SheetFrame3D;
}

export interface PickHit {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly tensorKey?: string;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  /** Optional multi-tower ownership metadata propagated from PickBox. */
  readonly towerRole?: 'context' | 'denoiser' | 'shared';
  readonly executionId?: string;
  readonly traceOrigin?: string;
  readonly semanticTag?: string;
  readonly absoluteTokenOffset?: number;
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
function intersectRayAabb(ray: Ray, min: Vec3, max: Vec3): number | null {
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
  let bestFrameHit: SheetFrameRayHit | null = null;

  for (const box of boxes) {
    let t: number | null;
    let frameHit: SheetFrameRayHit | null = null;
    if (box.frame) {
      frameHit = intersectRaySheetFrame(ray.origin, ray.direction, box.frame);
      t = frameHit ? frameHit.t : null;
    } else {
      t = intersectRayAabb(ray, box.min, box.max);
    }
    if (t === null) continue;
    if (
      bestBox === null ||
      t < bestT - TIE_EPS ||
      (Math.abs(t - bestT) <= TIE_EPS && box.bayId < bestBox.bayId)
    ) {
      bestT = t;
      bestBox = box;
      bestFrameHit = frameHit;
    }
  }

  if (!bestBox) return null;
  const point = bestFrameHit ? bestFrameHit.point : add3(ray.origin, scale3(ray.direction, bestT));

  let hitRow: number | undefined;
  let hitCol: number | undefined;
  if (bestBox.gridCols && bestBox.gridRows && bestBox.gridCols > 0 && bestBox.gridRows > 0) {
    let u: number;
    let vFromTop: number;
    if (bestFrameHit) {
      u = bestFrameHit.u;
      vFromTop = bestFrameHit.vRow;
    } else {
      u = clamp01((point[0] - bestBox.min[0]) / Math.max(1e-6, bestBox.max[0] - bestBox.min[0]));
      vFromTop = clamp01((point[2] - bestBox.min[2]) / Math.max(1e-6, bestBox.max[2] - bestBox.min[2]));
    }
    hitCol = Math.min(bestBox.gridCols - 1, Math.max(0, Math.floor(u * bestBox.gridCols)));
    hitRow = Math.min(bestBox.gridRows - 1, Math.max(0, Math.floor(vFromTop * bestBox.gridRows)));
  }

  return {
    bayId: bestBox.bayId,
    partId: bestBox.partId,
    headIndex: bestBox.headIndex,
    kvHeadIndex: bestBox.kvHeadIndex,
    groupIndex: bestBox.groupIndex,
    tensorKey: bestBox.tensorKey,
    tokenIndex: bestBox.tokenIndex,
    tapIndex: bestBox.tapIndex,
    stateTokenIndex: bestBox.stateTokenIndex,
    ...(bestBox.towerRole !== undefined ? { towerRole: bestBox.towerRole } : {}),
    ...(bestBox.executionId !== undefined ? { executionId: bestBox.executionId } : {}),
    ...(bestBox.traceOrigin !== undefined ? { traceOrigin: bestBox.traceOrigin } : {}),
    ...(bestBox.semanticTag !== undefined ? { semanticTag: bestBox.semanticTag } : {}),
    ...(bestBox.absoluteTokenOffset !== undefined
      ? { absoluteTokenOffset: bestBox.absoluteTokenOffset }
      : {}),
    hitRow,
    hitCol,
    t: bestT,
    point,
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
