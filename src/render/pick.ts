import { inferGqaCellHead, type GqaCellRef, type GqaTensorKey } from '../math/gqaSim';
import {
  inferMambaCellHead,
  MAMBA_SIM_CONFIG,
  mambaGroupForHead,
  type MambaCellRef,
  type MambaTensorKey,
} from '../math/mambaSim';
import {
  MOE_SIM_CONFIG,
  type MoeCellRef,
  type MoeTensorKey,
} from '../math/moeSim';
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
  readonly groupIndex?: number;
  readonly tensorKey?: string;
  readonly gridCols?: number;
  readonly gridRows?: number;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  readonly min: Vec3;
  readonly max: Vec3;
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
  readonly hitRow?: number;
  readonly hitCol?: number;
  readonly t: number;
  readonly point: Vec3;
  readonly isFascia: boolean;
}

const PARALLEL_EPS = 1e-8;
const TIE_EPS = 1e-6;

const GQA_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'w_q', 'w_k', 'w_v', 'q', 'kv_k', 'kv_v',
  'scores', 'softmax', 'v_out', 'concat', 'w_o', 'residual',
]);

const MAMBA_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'w_in', 'z', 'xbc_pre', 'conv_buf', 'x',
  'b', 'c', 'dt', 'h_prev', 'h_curr', 'y', 'y_norm', 'w_out', 'residual',
]);

const MAMBA_SEQUENCE_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'z', 'xbc_pre', 'x', 'b', 'c', 'dt', 'y', 'y_norm', 'residual',
]);

const MOE_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'w_router', 'router_bank', 'topk_weights',
  'routed_w_up', 'routed_pre', 'routed_act', 'routed_w_down', 'routed_sum',
  'shared_w_up', 'shared_act', 'shared_w_down', 'shared_out', 'moe_out', 'residual',
]);

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
    groupIndex: bestBox.groupIndex,
    tensorKey: bestBox.tensorKey,
    tokenIndex: bestBox.tokenIndex,
    tapIndex: bestBox.tapIndex,
    stateTokenIndex: bestBox.stateTokenIndex,
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
  if (
    !hit ||
    !hit.tensorKey ||
    !GQA_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  const tensor = hit.tensorKey as GqaTensorKey;
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
  const baseCell: GqaCellRef = {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    headIndex: hit.headIndex,
    kvHeadIndex: hit.kvHeadIndex,
  };
  const inferredHead = inferGqaCellHead(baseCell);
  if (inferredHead !== null && baseCell.headIndex === undefined) {
    return {
      ...baseCell,
      headIndex: inferredHead,
      kvHeadIndex: baseCell.kvHeadIndex ?? (inferredHead < 16 ? 0 : 1),
    };
  }
  return baseCell;
}

/** Resolve a ray-cast PickHit on a live Mamba-2 sheet into an exact logical MambaCellRef. */
export function pickHitToMambaCell(
  hit: PickHit | null,
  activeMambaHead = 0,
  tokens = 8,
): MambaCellRef | null {
  if (
    !hit ||
    !hit.tensorKey ||
    !MAMBA_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  const tensor = hit.tensorKey as MambaTensorKey;
  if (tensor === 'conv_buf') {
    const slot = hit.tapIndex ?? hit.tokenIndex ?? 0;
    return {
      tensor,
      row: slot,
      col: hit.hitCol,
      tapIndex: slot,
      tokenIndex: hit.stateTokenIndex ?? Math.max(0, tokens - 1),
    };
  }
  if (tensor === 'h_prev' || tensor === 'h_curr') {
    const headIdx = hit.headIndex ?? activeMambaHead;
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      headIndex: headIdx,
      groupIndex: mambaGroupForHead(headIdx),
      tokenIndex: hit.tokenIndex ?? Math.max(0, tokens - 1),
    };
  }
  if (tensor === 'b' || tensor === 'c') {
    const grp = Math.floor(hit.hitCol / MAMBA_SIM_CONFIG.dState);
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      groupIndex: grp,
      tokenIndex: hit.hitRow,
    };
  }
  const baseCell: MambaCellRef = {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    headIndex: hit.headIndex,
    groupIndex: hit.groupIndex,
    tokenIndex: MAMBA_SEQUENCE_TENSOR_KEYS.has(tensor) ? hit.hitRow : undefined,
  };
  const inferredHead = inferMambaCellHead(baseCell);
  if (inferredHead !== null && baseCell.headIndex === undefined) {
    return {
      ...baseCell,
      headIndex: inferredHead,
      groupIndex: baseCell.groupIndex ?? mambaGroupForHead(inferredHead),
    };
  }
  return baseCell;
}

/** Resolve a ray-cast PickHit on a live MoE sheet into an exact logical MoeCellRef. */
export function pickHitToMoeCell(
  hit: PickHit | null,
  focusToken: number,
  focusExpert: number,
  topkIndices?: readonly number[] | Int32Array,
): MoeCellRef | null {
  if (
    !hit ||
    !hit.tensorKey ||
    !MOE_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  const tensor = hit.tensorKey as MoeTensorKey;
  const tokIdx = hit.tokenIndex ?? focusToken;

  if (tensor === 'router_bank') {
    const eid = hit.headIndex ?? focusExpert;
    const bankRow = Math.floor(eid / MOE_SIM_CONFIG.bankCols);
    const bankCol = eid % MOE_SIM_CONFIG.bankCols;
    return {
      tensor,
      row: bankRow,
      col: bankCol,
      tokenIndex: tokIdx,
      expertId: eid,
    };
  }
  if (tensor === 'topk_weights') {
    const slot = Math.max(0, Math.min(MOE_SIM_CONFIG.topK - 1, hit.hitCol));
    const eid = topkIndices && topkIndices[slot] !== undefined ? topkIndices[slot] : focusExpert;
    return {
      tensor,
      row: 0,
      col: slot,
      tokenIndex: tokIdx,
      expertId: eid,
      rankSlot: slot,
    };
  }
  if (
    tensor === 'routed_w_up' ||
    tensor === 'routed_pre' ||
    tensor === 'routed_act' ||
    tensor === 'routed_w_down'
  ) {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      tokenIndex: tokIdx,
      expertId: hit.headIndex ?? focusExpert,
    };
  }
  if (tensor === 'w_router') {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      tokenIndex: tokIdx,
      expertId: hit.hitRow,
    };
  }
  if (tensor === 'shared_w_up' || tensor === 'shared_act' || tensor === 'shared_w_down') {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      tokenIndex: tokIdx,
    };
  }
  // Sequence tensors (input, rmsnorm, routed_sum, shared_out, moe_out, residual): row IS tokenIndex
  return {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    tokenIndex: hit.hitRow,
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
