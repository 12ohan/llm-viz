/**
 * Shared typed `LinesPass` GPU instance packing primitive (18 floats = 72 bytes per line).
 *
 * Exact per-instance stride layout matching `LINES_VS` (`src/render/lines.ts`) and
 * `buildMacroSpine` (`src/scene/spine.ts`):
 *   [0..2]   iPointA      (vec3,  byte offset  0, loc  2) — segment start (ax, ay, az)
 *   [3..5]   iPointB      (vec3,  byte offset 12, loc  3) — segment end   (bx, by, bz)
 *   [6]      iCssWidth    (float, byte offset 24, loc  4) — screen-space width in CSS px
 *   [7]      iStyle       (float, byte offset 28, loc  5) — 0 neutral, 1 active, 2 guide, 3 bf16 gold
 *   [8]      iAlpha       (float, byte offset 32, loc  6) — line opacity in [0, 1]
 *   [9]      iOwnerId     (float, byte offset 36, loc  7) — owning cuboid ID (`-1` for free/guide lines)
 *   [10]     iFace0       (float, byte offset 40, loc  8) — adjacent face 0 (`-1` disables backface culling)
 *   [11]     iFace1       (float, byte offset 44, loc  9) — adjacent face 1 (`-1` disables backface culling)
 *   [12..14] iOwnerCenter (vec3,  byte offset 48, loc 10) — owning cuboid center (`0,0,0` for free lines)
 *   [15..17] iOwnerExtent (vec3,  byte offset 60, loc 11) — owning cuboid extent (`0,0,0` for free lines)
 */

export const LINE_INSTANCE_FLOATS = 18;
export const LINE_INSTANCE_BYTES = LINE_INSTANCE_FLOATS * 4;

export const LINE_STYLE_NEUTRAL = 0;
export const LINE_STYLE_ACTIVE = 1;
export const LINE_STYLE_GUIDE = 2;
export const LINE_STYLE_BF16_GOLD = 3;

export const LINE_FREE_OWNER_ID = -1;
export const LINE_FREE_FACE = -1;

/** Named float offsets within each 18-float `LinesPass` instance record. */
export const LINE_OFFSETS = Object.freeze({
  pointA: 0,
  pointB: 3,
  cssWidth: 6,
  style: 7,
  alpha: 8,
  ownerId: 9,
  face0: 10,
  face1: 11,
  ownerCenter: 12,
  ownerExtent: 15,
} as const);

export interface LineInstanceAttributeSpec {
  readonly name: string;
  readonly location: number;
  readonly size: number;
  readonly floatOffset: number;
  readonly byteOffset: number;
}

/** Authoritative `LinesPass` per-instance WebGL2 attribute bindings derived from `LINE_OFFSETS`. */
export const LINE_INSTANCE_ATTRIBS: readonly LineInstanceAttributeSpec[] = Object.freeze([
  { name: 'iPointA', location: 2, size: 3, floatOffset: LINE_OFFSETS.pointA, byteOffset: LINE_OFFSETS.pointA * 4 },
  { name: 'iPointB', location: 3, size: 3, floatOffset: LINE_OFFSETS.pointB, byteOffset: LINE_OFFSETS.pointB * 4 },
  { name: 'iCssWidth', location: 4, size: 1, floatOffset: LINE_OFFSETS.cssWidth, byteOffset: LINE_OFFSETS.cssWidth * 4 },
  { name: 'iStyle', location: 5, size: 1, floatOffset: LINE_OFFSETS.style, byteOffset: LINE_OFFSETS.style * 4 },
  { name: 'iAlpha', location: 6, size: 1, floatOffset: LINE_OFFSETS.alpha, byteOffset: LINE_OFFSETS.alpha * 4 },
  { name: 'iOwnerId', location: 7, size: 1, floatOffset: LINE_OFFSETS.ownerId, byteOffset: LINE_OFFSETS.ownerId * 4 },
  { name: 'iFace0', location: 8, size: 1, floatOffset: LINE_OFFSETS.face0, byteOffset: LINE_OFFSETS.face0 * 4 },
  { name: 'iFace1', location: 9, size: 1, floatOffset: LINE_OFFSETS.face1, byteOffset: LINE_OFFSETS.face1 * 4 },
  { name: 'iOwnerCenter', location: 10, size: 3, floatOffset: LINE_OFFSETS.ownerCenter, byteOffset: LINE_OFFSETS.ownerCenter * 4 },
  { name: 'iOwnerExtent', location: 11, size: 3, floatOffset: LINE_OFFSETS.ownerExtent, byteOffset: LINE_OFFSETS.ownerExtent * 4 },
]);

/**
 * Append one free/guide line segment in the exact 18-float `LinesPass` instance layout.
 * Sets `iOwnerId = -1` and `iFace0 = iFace1 = -1` so backface edge culling is disabled.
 */
export function pushFreeGuideLine(
  out: number[],
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cssWidth = 1.0,
  style = LINE_STYLE_GUIDE,
  alpha = 0.5,
): void {
  out.push(
    ax, ay, az,
    bx, by, bz,
    cssWidth,
    style,
    alpha,
    LINE_FREE_OWNER_ID,
    LINE_FREE_FACE,
    LINE_FREE_FACE,
    0, 0, 0,
    0, 0, 0,
  );
}

/**
 * Append a 4-edge solid horizontal XZ rectangular frame at elevation `y`.
 */
export function pushBoxFrameXZ(
  out: number[],
  x0: number,
  z0: number,
  x1: number,
  z1: number,
  y: number,
  cssWidth = 1.0,
  style = LINE_STYLE_GUIDE,
  alpha = 0.45,
): void {
  pushFreeGuideLine(out, x0, y, z0, x1, y, z0, cssWidth, style, alpha);
  pushFreeGuideLine(out, x1, y, z0, x1, y, z1, cssWidth, style, alpha);
  pushFreeGuideLine(out, x1, y, z1, x0, y, z1, cssWidth, style, alpha);
  pushFreeGuideLine(out, x0, y, z1, x0, y, z0, cssWidth, style, alpha);
}

/**
 * Append a dashed horizontal XZ rectangular frame at elevation `y`.
 */
export function pushDashedFrameXZ(
  out: number[],
  x0: number,
  z0: number,
  x1: number,
  z1: number,
  y: number,
  style = LINE_STYLE_GUIDE,
  alpha = 0.42,
  dash = 0.22,
  gap = 0.14,
  cssWidth = 0.9,
): void {
  const step = dash + gap;
  if (dash <= 0 || step <= 0) return;

  const dashedLine = (ax: number, az: number, bx: number, bz: number) => {
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz);
    if (len < 1e-4) return;
    const ux = dx / len;
    const uz = dz / len;
    for (let d = 0; d < len; d += step) {
      const dEnd = Math.min(len, d + dash);
      if (dEnd - d < 1e-4) continue;
      pushFreeGuideLine(
        out,
        ax + ux * d,
        y,
        az + uz * d,
        ax + ux * dEnd,
        y,
        az + uz * dEnd,
        cssWidth,
        style,
        alpha,
      );
    }
  };

  dashedLine(x0, z0, x1, z0);
  dashedLine(x1, z0, x1, z1);
  dashedLine(x1, z1, x0, z1);
  dashedLine(x0, z1, x0, z0);
}
