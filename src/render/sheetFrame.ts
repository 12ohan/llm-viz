import type { Vec3 } from '../model/spec';
import { add3, cross3, dot3, normalize3, scale3, sub3, type AABB } from './camera';

/**
 * Right-handed orthonormal 3D frame for a thin matrix sheet.
 *
 * Basis convention (strictly right-handed, det(M) = +ex * ey * ez > 0):
 *  - `axisU`: unit column axis (+col direction, local x in [-0.5, +0.5] -> col 0 .. cols-1)
 *  - `axisN`: unit sheet normal axis (thickness axis, local y in [-0.5, +0.5], thin ey <= 0.09)
 *  - `axisV`: unit row axis = cross(axisU, axisN) (+row direction, local z in [-0.5, +0.5] -> row 0 .. rows-1)
 *
 * World mapping for normalized local coordinates (lx, ly, lz) in [-0.5, +0.5]^3:
 *   world = center + (lx * extent[0]) * axisU + (ly * extent[1]) * axisN + (lz * extent[2]) * axisV
 *
 * Both +N (ly = +0.5) and -N (ly = -0.5) faces derive logical (u, vRow) from the exact same
 * (lx, lz) sheet-local coordinates so every physical cell displays and picks the exact same
 * numerical tensor entry from either side.
 */
export interface SheetFrame3D {
  readonly center: Vec3;
  readonly extent: Vec3;
  readonly axisU: Vec3;
  readonly axisN: Vec3;
  readonly axisV: Vec3;
}

export interface SheetBasisAxes {
  readonly axisU: Vec3;
  readonly axisN: Vec3;
  readonly axisV: Vec3;
}

/** Horizontal XZ drafting-table sheet: cols +X, normal +Y (up), rows +Z (back->front). */
export const FRAME_XZ_AXES: SheetBasisAxes = {
  axisU: [1, 0, 0],
  axisN: [0, 1, 0],
  axisV: [0, 0, 1],
};

/** Upright XY wall facing +Z: cols +X (left->right), normal +Z (front), rows -Y (top->bottom). */
export const FRAME_XY_AXES: SheetBasisAxes = {
  axisU: [1, 0, 0],
  axisN: [0, 0, 1],
  axisV: [0, -1, 0],
};

/** Standing YZ fin facing +X: cols -Z (front-left->back-right from +X/+Z), normal +X, rows -Y (top->bottom). */
export const FRAME_YZ_AXES: SheetBasisAxes = {
  axisU: [0, 0, -1],
  axisN: [1, 0, 0],
  axisV: [0, -1, 0],
};

/**
 * Construct a strictly right-handed orthonormal SheetFrame3D from center, positive full extents
 * `[ex, ey, ez]`, and desired `(axisU, axisN)` directions. `axisV` is always `cross(u, n)`.
 */
export function makeSheetFrame3D(
  center: Vec3,
  extent: Vec3,
  axisU: Vec3 = FRAME_XZ_AXES.axisU,
  axisN: Vec3 = FRAME_XZ_AXES.axisN,
): SheetFrame3D {
  const u = normalize3(axisU);
  const nProj = dot3(axisN, u);
  let nCand: Vec3 = [
    axisN[0] - nProj * u[0],
    axisN[1] - nProj * u[1],
    axisN[2] - nProj * u[2],
  ];
  const nLen = Math.hypot(nCand[0], nCand[1], nCand[2]);
  if (nLen < 1e-8) {
    // Fallback orthogonal normal if axisN was collinear with axisU
    const fallback: Vec3 = Math.abs(u[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const fp = dot3(fallback, u);
    nCand = [
      fallback[0] - fp * u[0],
      fallback[1] - fp * u[1],
      fallback[2] - fp * u[2],
    ];
  }
  const n = normalize3(nCand);
  const v = normalize3(cross3(u, n));
  return {
    center: [center[0], center[1], center[2]],
    extent: [Math.max(1e-5, extent[0]), Math.max(1e-5, extent[1]), Math.max(1e-5, extent[2])],
    axisU: u,
    axisN: n,
    axisV: v,
  };
}

/** Map normalized local sheet coordinates `(lx, ly, lz) in [-0.5, +0.5]^3` to world space. */
export function localToWorldOnFrame(frame: SheetFrame3D, lx: number, ly: number, lz: number): Vec3 {
  const du = lx * frame.extent[0];
  const dn = ly * frame.extent[1];
  const dv = lz * frame.extent[2];
  return [
    frame.center[0] + du * frame.axisU[0] + dn * frame.axisN[0] + dv * frame.axisV[0],
    frame.center[1] + du * frame.axisU[1] + dn * frame.axisN[1] + dv * frame.axisV[1],
    frame.center[2] + du * frame.axisU[2] + dn * frame.axisN[2] + dv * frame.axisV[2],
  ];
}

/** Map a world point into normalized local sheet coordinates `(lx, ly, lz)` where box is `[-0.5, +0.5]^3`. */
export function worldToLocalOnFrame(frame: SheetFrame3D, worldPoint: Vec3): Vec3 {
  const rel = sub3(worldPoint, frame.center);
  return [
    dot3(rel, frame.axisU) / frame.extent[0],
    dot3(rel, frame.axisN) / frame.extent[1],
    dot3(rel, frame.axisV) / frame.extent[2],
  ];
}

/** Exact tight world-space AABB enclosing all 8 oriented corners of `frame`. */
export function sheetFrameWorldAabb(frame: SheetFrame3D): AABB {
  const hx = frame.extent[0] * 0.5;
  const hy = frame.extent[1] * 0.5;
  const hz = frame.extent[2] * 0.5;
  const rx =
    Math.abs(frame.axisU[0]) * hx +
    Math.abs(frame.axisN[0]) * hy +
    Math.abs(frame.axisV[0]) * hz;
  const ry =
    Math.abs(frame.axisU[1]) * hx +
    Math.abs(frame.axisN[1]) * hy +
    Math.abs(frame.axisV[1]) * hz;
  const rz =
    Math.abs(frame.axisU[2]) * hx +
    Math.abs(frame.axisN[2]) * hy +
    Math.abs(frame.axisV[2]) * hz;
  return {
    min: [frame.center[0] - rx, frame.center[1] - ry, frame.center[2] - rz],
    max: [frame.center[0] + rx, frame.center[1] + ry, frame.center[2] + rz],
  };
}

export interface SheetFrameRayHit {
  /** Exact world-space ray distance parameter t >= 0 (un-renormalized local ray preserves t). */
  readonly t: number;
  /** World-space hit point `origin + t * direction`. */
  readonly point: Vec3;
  /** Normalized local hit coordinates in `[-0.5, +0.5]^3`. */
  readonly localHit: Vec3;
  /** Logical column coordinate `u in [0, 1]` derived directly from `localHit[0] + 0.5`. */
  readonly u: number;
  /** Logical row coordinate `vRow in [0, 1]` derived directly from `localHit[2] + 0.5`. */
  readonly vRow: number;
  /** +1 if hit entered on/near +N face, -1 if -N face. */
  readonly faceSign: 1 | -1;
}

const PARALLEL_EPS = 1e-8;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Exact ray vs. oriented nonuniform-extent sheet intersection.
 *
 * Transforms `ray` into the sheet's normalized local cube `[-0.5, +0.5]^3` by projecting
 * `(origin - center)` and `direction` onto `(axisU, axisN, axisV)` and dividing each component
 * by its corresponding full extent `(ex, ey, ez)`. Crucially does NOT renormalize `dLocal`,
 * so the returned ray parameter `t` is the exact world-space distance along `ray.direction`
 * and can be compared directly against axis-aligned `intersectRayAabb` hits.
 */
export function intersectRaySheetFrame(
  origin: Vec3,
  direction: Vec3,
  frame: SheetFrame3D,
): SheetFrameRayHit | null {
  const relO = sub3(origin, frame.center);
  const oLocal: Vec3 = [
    dot3(relO, frame.axisU) / frame.extent[0],
    dot3(relO, frame.axisN) / frame.extent[1],
    dot3(relO, frame.axisV) / frame.extent[2],
  ];
  const dLocal: Vec3 = [
    dot3(direction, frame.axisU) / frame.extent[0],
    dot3(direction, frame.axisN) / frame.extent[1],
    dot3(direction, frame.axisV) / frame.extent[2],
  ];

  let enter = -Infinity;
  let exit = Infinity;

  for (let axis = 0; axis < 3; axis++) {
    const o = oLocal[axis];
    const d = dLocal[axis];
    if (Math.abs(d) < PARALLEL_EPS) {
      if (o < -0.5 || o > 0.5) return null;
      continue;
    }
    const invD = 1 / d;
    let t0 = (-0.5 - o) * invD;
    let t1 = (0.5 - o) * invD;
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

  const localHit: Vec3 = [
    oLocal[0] + entry * dLocal[0],
    oLocal[1] + entry * dLocal[1],
    oLocal[2] + entry * dLocal[2],
  ];
  const point = add3(origin, scale3(direction, entry));
  const u = clamp01(localHit[0] + 0.5);
  const vRow = clamp01(localHit[2] + 0.5);
  const faceSign: 1 | -1 = localHit[1] >= 0 ? 1 : -1;

  return { t: entry, point, localHit, u, vRow, faceSign };
}
