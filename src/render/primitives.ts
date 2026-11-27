/**
 * Pure WebGL2 Primitive Layout Constants & Scene-Agnostic Anchor Contracts
 * (`src/render/primitives.ts`)
 *
 * Strictly leaf module with ZERO runtime dependencies on `src/model/`, `src/math/`,
 * `src/scene/`, or `src/ui/`. Shared across all MPA entries without pulling any
 * model-specific geometry builders or simulators into `shared(allN)`.
 */

export type Vec3 = readonly [number, number, number];

/** Solid instance layout: 12 floats = 48 bytes per instance. */
export const SOLID_INSTANCE_FLOATS = 12;
export const SOLID_INSTANCE_BYTES = SOLID_INSTANCE_FLOATS * 4;

/** Translucent ribbon / arrow instance layout: 12 floats = 48 bytes per instance. */
export const RIBBON_INSTANCE_FLOATS = 12;
export const RIBBON_INSTANCE_BYTES = RIBBON_INSTANCE_FLOATS * 4;

export const FACE_POS_X = 0;
export const FACE_NEG_X = 1;
export const FACE_POS_Y = 2;
export const FACE_NEG_Y = 3;
export const FACE_POS_Z = 4;
export const FACE_NEG_Z = 5;

/** Procedural matrix-cell color families (llm-viz visual language). */
export const CELL_NEUTRAL = 0;
export const CELL_WEIGHT_BLUE = 1;    // Periwinkle-blue #637BF5
export const CELL_ACTIVATION_GREEN = 2; // Emerald-green #4ADE80
export const CELL_SSM_CYAN = 3;       // Cyan/teal #38BDF8
export const CELL_MOE_AMBER = 4;      // Warm amber/gold #F59E0B
export const CELL_EMBED_SLATE = 5;    // Charcoal-slate + periwinkle accent

export const RIBBON_KIND_BOX = 0;
export const RIBBON_KIND_ARROW_NEG_X = 1; // Arrow pointing -X
export const RIBBON_KIND_ARROW_POS_X = 2; // Arrow pointing +X
export const RIBBON_KIND_ARROW_NEG_Y = 3; // Arrow pointing -Y (downward flow)
export const RIBBON_KIND_ARROW_POS_Y = 4; // Arrow pointing +Y (upward feedback return)
export const RIBBON_KIND_ARROW_POS_Z = 5; // Arrow pointing +Z (outward toward KV cache)
export const RIBBON_KIND_ARROW_NEG_Z = 6; // Arrow pointing -Z (inward from KV cache)

export const RIBBON_COLOR_EMERALD = 0;    // rgba(74, 222, 128, 0.34)
export const RIBBON_COLOR_PERIWINKLE = 1; // rgba(129, 140, 248, 0.45)
export const RIBBON_COLOR_CYAN = 2;       // rgba(56, 189, 248, 0.36)
export const RIBBON_COLOR_AMBER = 3;      // rgba(245, 158, 11, 0.40)

export interface LandmarkAnchor {
  readonly id: string;
  readonly label: string;
  readonly sublabel?: string;
  readonly world: Vec3;
  readonly bayId?: number;
  readonly partId?: string;
  readonly tone?: 'default' | 'gold' | 'cyan' | 'muted';
  readonly align?: 'left' | 'right' | 'top' | 'bottom' | 'center';
  readonly tier?: 'primary' | 'detail';
}

export interface DimensionCaliper {
  readonly id: string;
  readonly aWorld: Vec3;
  readonly bWorld: Vec3;
  readonly label: string;
  readonly partIds?: readonly string[];
  readonly minPixels?: number;
  readonly offsetPx?: number;
}

export interface KvDepthTickAnchor {
  readonly token: number;
  readonly world: Vec3;
}
