import {
  bayCenterX,
  bodyPrecision,
  kvCachePrecision,
  type BayDescriptor,
  type NemotronSpec,
  type PrecisionMode,
  type Vec3,
} from '../model/spec';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';

/** Solid instance layout: 12 floats = 48 bytes per instance (§3). */
export const SOLID_INSTANCE_FLOATS = 12;
export const SOLID_INSTANCE_BYTES = SOLID_INSTANCE_FLOATS * 4;

/** Line instance layout: 18 floats = 72 bytes per instance (§3). */
export const LINE_INSTANCE_FLOATS = 18;
export const LINE_INSTANCE_BYTES = LINE_INSTANCE_FLOATS * 4;

export const FACE_POS_X = 0;
export const FACE_NEG_X = 1;
export const FACE_POS_Y = 2;
export const FACE_NEG_Y = 3;
export const FACE_POS_Z = 4;
export const FACE_NEG_Z = 5;

export const MASK_POS_Y = 1 << FACE_POS_Y;
export const MASK_POS_Z = 1 << FACE_POS_Z;

export const PATTERN_NONE = 0;
export const PATTERN_REGISTER = 1;
export const PATTERN_CABINET = 2;

export const TONE_MATTE = 0;
export const TONE_BOOKEND = 1;
export const TONE_GUIDE = 2;

export const LINE_STYLE_NEUTRAL = 0;
export const LINE_STYLE_ACTIVE = 1;
export const LINE_STYLE_RULER = 2;

export const RULER_X = 28.5;
export const RULER_Y = 4.5;
export const RAIL_START_Z = -0.30;
export const RAIL_LENGTH = 9.60;
export const RAIL_END_Z = RAIL_START_Z - RAIL_LENGTH; // -9.90
export const SLICE_PITCH = 0.60;
export const SLICE_DEPTH = 0.48;
export const KV_RAIL_OFFSETS: readonly number[] = [-0.33, -0.15, 0.15, 0.33];

/** Fixed macro fit AABB spanning embedding plate through depth ruler and full 16-token capacity. */
export const MACRO_FIT_BOUNDS: AABB = {
  min: [-28.0, -0.30, -9.90],
  max: [29.1, 6.80, 0.60],
};

export interface BayAnchor {
  readonly id: number;
  readonly kind: BayDescriptor['kind'];
  readonly topWorld: Vec3;
  readonly frontWorld: Vec3;
}

export interface BracketAnchor {
  readonly index: 0 | 1 | 2 | 3;
  readonly first: number;
  readonly last: number;
  readonly leftWorld: Vec3;
  readonly rightWorld: Vec3;
  readonly midWorld: Vec3;
}

export interface RulerTickAnchor {
  readonly token: number;
  readonly tickWorld: Vec3;
  readonly labelWorld: Vec3;
}

export interface BookendAnchor {
  readonly id: 'embed' | 'norm' | 'output';
  readonly label: string;
  readonly world: Vec3;
}

export interface MacroSceneGeometry {
  readonly solidData: Float32Array;
  readonly solidCount: number;
  readonly lineData: Float32Array;
  readonly lineCount: number;
  readonly neutralLineCount: number;
  readonly activeLineCount: number;
  readonly fitBounds: AABB;
  readonly bayAnchors: readonly BayAnchor[];
  readonly bracketAnchors: readonly BracketAnchor[];
  readonly rulerTicks: readonly RulerTickAnchor[];
  readonly bookendAnchors: readonly BookendAnchor[];
  readonly solidPickBoxes: readonly PickBox[];
  readonly fasciaPickBoxes: readonly PickBox[];
}

interface SolidSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  familyPattern: number;
  fp8Flag: number;
  phase: number;
  bayId: number;
  faceMask: number;
  toneSelector: number;
  lineStyle: number;
  cssWidth: number;
  lineAlpha: number;
  pickPart?: string;
  skipEdges?: boolean;
}

interface LineSpec {
  ax: number;
  ay: number;
  az: number;
  bx: number;
  by: number;
  bz: number;
  cssWidth: number;
  style: number;
  alpha: number;
  ownerId: number;
  face0: number;
  face1: number;
  ocx: number;
  ocy: number;
  ocz: number;
  oex: number;
  oey: number;
  oez: number;
}

/** Back Z coordinate of slice j (1..16) along the K/V tracks and depth ruler. */
export function sliceCenterZ(sliceIndex1Based: number): number {
  return RAIL_START_Z - SLICE_PITCH * (sliceIndex1Based - 1) - SLICE_DEPTH * 0.5;
}

export function sliceBackZ(sliceIndex1Based: number): number {
  return RAIL_START_Z - SLICE_PITCH * (sliceIndex1Based - 1) - SLICE_DEPTH;
}

export function bayTopY(kind: BayDescriptor['kind']): number {
  switch (kind) {
    case 'E':
      return 3.05;
    case 'M':
      return 4.80;
    case '*':
      return 6.80;
  }
}

function quant(v: number): number {
  return Math.round(v * 10000);
}

function segmentKey(ax: number, ay: number, az: number, bx: number, by: number, bz: number): string {
  const qax = quant(ax), qay = quant(ay), qaz = quant(az);
  const qbx = quant(bx), qby = quant(by), qbz = quant(bz);
  const aFirst =
    qax < qbx ||
    (qax === qbx && (qay < qby || (qay === qby && qaz <= qbz)));
  return aFirst
    ? `${qax},${qay},${qaz}|${qbx},${qby},${qbz}`
    : `${qbx},${qby},${qbz}|${qax},${qay},${qaz}`;
}

function areOppositeFaces(a: number, b: number): boolean {
  return (a ^ 1) === b;
}

function mergeFaces(f0a: number, f1a: number, f0b: number, f1b: number): readonly [number, number] {
  if (f0a < 0 || f1a < 0 || f0b < 0 || f1b < 0) return [-1, -1];
  // If two coplanar adjacent faces share a visible face and have opposite mates, keep the shared face.
  if (f0a === f0b && areOppositeFaces(f1a, f1b)) return [f0a, f0a];
  if (f0a === f1b && areOppositeFaces(f1a, f0b)) return [f0a, f0a];
  if (f1a === f0b && areOppositeFaces(f0a, f1b)) return [f1a, f1a];
  if (f1a === f1b && areOppositeFaces(f0a, f0b)) return [f1a, f1a];
  return [f0a, f1a];
}

function emitCuboidEdges(
  s: SolidSpec,
  dedupMap: Map<string, LineSpec>,
): void {
  if (s.skipEdges) return;
  const hx = s.ex * 0.5;
  const hy = s.ey * 0.5;
  const hz = s.ez * 0.5;
  const x0 = s.cx - hx, x1 = s.cx + hx;
  const y0 = s.cy - hy, y1 = s.cy + hy;
  const z0 = s.cz - hz, z1 = s.cz + hz;

  const rawEdges: readonly [number, number, number, number, number, number, number, number][] = [
    // X edges
    [x0, y0, z0, x1, y0, z0, FACE_NEG_Y, FACE_NEG_Z],
    [x0, y1, z0, x1, y1, z0, FACE_POS_Y, FACE_NEG_Z],
    [x0, y0, z1, x1, y0, z1, FACE_NEG_Y, FACE_POS_Z],
    [x0, y1, z1, x1, y1, z1, FACE_POS_Y, FACE_POS_Z],
    // Y edges
    [x0, y0, z0, x0, y1, z0, FACE_NEG_X, FACE_NEG_Z],
    [x1, y0, z0, x1, y1, z0, FACE_POS_X, FACE_NEG_Z],
    [x0, y0, z1, x0, y1, z1, FACE_NEG_X, FACE_POS_Z],
    [x1, y0, z1, x1, y1, z1, FACE_POS_X, FACE_POS_Z],
    // Z edges
    [x0, y0, z0, x0, y0, z1, FACE_NEG_X, FACE_NEG_Y],
    [x1, y0, z0, x1, y0, z1, FACE_POS_X, FACE_NEG_Y],
    [x0, y1, z0, x0, y1, z1, FACE_NEG_X, FACE_POS_Y],
    [x1, y1, z0, x1, y1, z1, FACE_POS_X, FACE_POS_Y],
  ];

  for (const [ax, ay, az, bx, by, bz, face0, face1] of rawEdges) {
    const key = segmentKey(ax, ay, az, bx, by, bz);
    const existing = dedupMap.get(key);
    const candidate: LineSpec = {
      ax, ay, az, bx, by, bz,
      cssWidth: s.cssWidth,
      style: s.lineStyle,
      alpha: s.lineAlpha,
      ownerId: s.bayId,
      face0,
      face1,
      ocx: s.cx, ocy: s.cy, ocz: s.cz,
      oex: s.ex, oey: s.ey, oez: s.ez,
    };

    if (!existing) {
      dedupMap.set(key, candidate);
    } else if (candidate.style === LINE_STYLE_ACTIVE && existing.style !== LINE_STYLE_ACTIVE) {
      dedupMap.set(key, candidate);
    } else if (existing.style === candidate.style && existing.ownerId === candidate.ownerId) {
      const [mf0, mf1] = mergeFaces(existing.face0, existing.face1, candidate.face0, candidate.face1);
      existing.face0 = mf0;
      existing.face1 = mf1;
    }
  }
}

export interface BuildSpineOptions {
  readonly spec: NemotronSpec;
  readonly tokens: number;
  readonly precision: PrecisionMode;
  readonly selectedBay: number | null;
  readonly hoveredBay: number | null;
}

/**
 * Build the complete Milestone 1 macro 52-bay spine geometry with Patch 1 applied:
 * - Residual strip centered at y = -0.15 (y in [-0.30, 0.00]) so all 52 bays & bookends sit flush on y = 0.
 * - M uprights height = 4.52, center y = 2.26 (flush beneath crossbar y = 4.52..4.80).
 * - M Conv1D band center z = 0.36 (z in [0.29, 0.43], flush in front of uprights).
 * - Attention crosspiece width = 0.42 (x in [-0.21, +0.21], cleanly between tines).
 * - KV slice center y = 4.64 (flush atop rails at y = 4.53).
 */
export function buildMacroSpine(options: BuildSpineOptions): MacroSceneGeometry {
  const { spec, tokens, precision, selectedBay, hoveredBay } = options;
  const clampedTokens = Math.max(1, Math.min(16, Math.round(tokens)));
  const registerPhase = (clampedTokens % 4) / 4;

  const solids: SolidSpec[] = [];
  const solidPickBoxes: PickBox[] = [];
  const fasciaPickBoxes: PickBox[] = [];
  const bayAnchors: BayAnchor[] = [];

  const pushSolid = (s: SolidSpec, pickableBay: boolean) => {
    solids.push(s);
    if (pickableBay && s.bayId > 0) {
      const hx = s.ex * 0.5;
      const hy = s.ey * 0.5;
      const hz = s.ez * 0.5;
      solidPickBoxes.push({
        bayId: s.bayId,
        partId: s.pickPart,
        min: [s.cx - hx, s.cy - hy, s.cz - hz],
        max: [s.cx + hx, s.cy + hy, s.cz + hz],
      });
    }
  };

  // --- Residual strip & bookends (Patch 1: residual strip y in [-0.30, 0.00]) ---
  pushSolid(
    {
      cx: 0, cy: -0.15, cz: 0,
      ex: 56.0, ey: 0.30, ez: 0.78,
      familyPattern: PATTERN_NONE,
      fp8Flag: 0,
      phase: 0,
      bayId: 0,
      faceMask: 0,
      toneSelector: TONE_GUIDE,
      lineStyle: LINE_STYLE_NEUTRAL,
      cssWidth: 1.0,
      lineAlpha: 0.60,
    },
    false,
  );

  // Embedding plate: 1.20 x 4.0 x 1.20 at (-27.4, 2.0, 0)
  pushSolid(
    {
      cx: -27.4, cy: 2.0, cz: 0,
      ex: 1.20, ey: 4.0, ez: 1.20,
      familyPattern: PATTERN_NONE,
      fp8Flag: 0,
      phase: 0,
      bayId: 0,
      faceMask: 0,
      toneSelector: TONE_BOOKEND,
      lineStyle: LINE_STYLE_NEUTRAL,
      cssWidth: 1.0,
      lineAlpha: 0.60,
    },
    false,
  );

  // Final RMSNorm wafer: 0.24 x 2.6 x 1.0 at (26.45, 1.3, 0)
  pushSolid(
    {
      cx: 26.45, cy: 1.3, cz: 0,
      ex: 0.24, ey: 2.6, ez: 1.0,
      familyPattern: PATTERN_NONE,
      fp8Flag: 0,
      phase: 0,
      bayId: 0,
      faceMask: 0,
      toneSelector: TONE_BOOKEND,
      lineStyle: LINE_STYLE_NEUTRAL,
      cssWidth: 1.0,
      lineAlpha: 0.60,
    },
    false,
  );

  // Output plate: 1.20 x 4.4 x 1.20 at (27.4, 2.2, 0)
  pushSolid(
    {
      cx: 27.4, cy: 2.2, cz: 0,
      ex: 1.20, ey: 4.4, ez: 1.20,
      familyPattern: PATTERN_NONE,
      fp8Flag: 0,
      phase: 0,
      bayId: 0,
      faceMask: 0,
      toneSelector: TONE_BOOKEND,
      lineStyle: LINE_STYLE_NEUTRAL,
      cssWidth: 1.0,
      lineAlpha: 0.60,
    },
    false,
  );

  // --- 52 Bays ---
  for (const bay of spec.bays) {
    const cx = bayCenterX(bay.id);
    const isSelected = selectedBay === bay.id;
    const isHovered = hoveredBay === bay.id;
    const isActive = isSelected || isHovered;
    const lineStyle = isActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL;
    const cssWidth = isSelected ? 1.6 : isHovered ? 1.35 : 1.0;
    const lineAlpha = isSelected ? 1.0 : isHovered ? 0.90 : 0.60;
    const bodyFp8 = bodyPrecision(bay, precision) === 'fp8' ? 1 : 0;

    bayAnchors.push({
      id: bay.id,
      kind: bay.kind,
      topWorld: [cx, bayTopY(bay.kind), 0],
      frontWorld: [cx, bay.kind === 'M' ? 1.55 : bay.kind === 'E' ? 1.35 : 3.60, 0.50],
    });

    // Picking fascia spanning full pitch in X, y in [-0.15, 6.8], z in [0.50, 0.52] (§4)
    fasciaPickBoxes.push({
      bayId: bay.id,
      partId: 'fascia',
      min: [cx - 0.50, -0.15, 0.50],
      max: [cx + 0.50, 6.80, 0.52],
    });

    if (bay.kind === 'M') {
      // Patch 1: uprights height = 4.52, center y = 2.26 (flush below crossbar y = 4.52..4.80)
      pushSolid(
        {
          cx: cx - 0.31, cy: 2.26, cz: 0,
          ex: 0.16, ey: 4.52, ez: 0.58,
          familyPattern: PATTERN_NONE,
          fp8Flag: bodyFp8,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'upright-left',
        },
        true,
      );
      pushSolid(
        {
          cx: cx + 0.31, cy: 2.26, cz: 0,
          ex: 0.16, ey: 4.52, ez: 0.58,
          familyPattern: PATTERN_NONE,
          fp8Flag: bodyFp8,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'upright-right',
        },
        true,
      );
      // Crossbar: 0.78 x 0.28 x 0.58 at y = 4.66 (y in [4.52, 4.80])
      pushSolid(
        {
          cx, cy: 4.66, cz: 0,
          ex: 0.78, ey: 0.28, ez: 0.58,
          familyPattern: PATTERN_NONE,
          fp8Flag: bodyFp8,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'crossbar',
        },
        true,
      );
      // Hatched SSM register insert: 0.46 x 2.50 x 0.40 at (cx, 1.55, 0) -> front at z = +0.20
      // Always FP32 storage (fp8Flag = 0)
      pushSolid(
        {
          cx, cy: 1.55, cz: 0,
          ex: 0.46, ey: 2.50, ez: 0.40,
          familyPattern: PATTERN_REGISTER,
          fp8Flag: 0,
          phase: registerPhase,
          bayId: bay.id,
          faceMask: MASK_POS_Z | MASK_POS_Y,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'register',
        },
        true,
      );
      // Patch 1: Conv1D band 0.78 x 0.25 x 0.14 at (cx, 0.30, 0.36) -> z in [0.29, 0.43]
      // Always BF16 in both precision modes (fp8Flag = 0)
      pushSolid(
        {
          cx, cy: 0.30, cz: 0.36,
          ex: 0.78, ey: 0.25, ez: 0.14,
          familyPattern: PATTERN_NONE,
          fp8Flag: 0,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'conv1d',
        },
        true,
      );
    } else if (bay.kind === 'E') {
      // E body: 0.86 x 2.70 x 0.84 at (cx, 1.35, 0), grid pattern 2 on +Z front
      pushSolid(
        {
          cx, cy: 1.35, cz: 0,
          ex: 0.86, ey: 2.70, ez: 0.84,
          familyPattern: PATTERN_CABINET,
          fp8Flag: bodyFp8,
          phase: 0,
          bayId: bay.id,
          faceMask: MASK_POS_Z,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'cabinet',
        },
        true,
      );
      // E cap: 0.64 x 0.35 x 0.60 at (cx, 2.875, 0) (y in [2.70, 3.05])
      pushSolid(
        {
          cx, cy: 2.875, cz: 0,
          ex: 0.64, ey: 0.35, ez: 0.60,
          familyPattern: PATTERN_NONE,
          fp8Flag: bodyFp8,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'cap',
        },
        true,
      );
    } else {
      // Attention (*) tall fork — body always BF16 (fp8Flag = 0)
      // Foot: 0.86 x 0.40 x 0.92 at (cx, 0.20, 0)
      pushSolid(
        {
          cx, cy: 0.20, cz: 0,
          ex: 0.86, ey: 0.40, ez: 0.92,
          familyPattern: PATTERN_NONE,
          fp8Flag: 0,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'foot',
        },
        true,
      );
      // Two tines: 0.20 x 6.40 x 0.58 at x = cx +- 0.31, y = 3.60, z = 0 (y in [0.40, 6.80])
      pushSolid(
        {
          cx: cx - 0.31, cy: 3.60, cz: 0,
          ex: 0.20, ey: 6.40, ez: 0.58,
          familyPattern: PATTERN_NONE,
          fp8Flag: 0,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'tine-left',
        },
        true,
      );
      pushSolid(
        {
          cx: cx + 0.31, cy: 3.60, cz: 0,
          ex: 0.20, ey: 6.40, ez: 0.58,
          familyPattern: PATTERN_NONE,
          fp8Flag: 0,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'tine-right',
        },
        true,
      );
      // Patch 1: crosspiece width = 0.42 (x in [-0.21, +0.21], cleanly between tines)
      pushSolid(
        {
          cx, cy: 4.50, cz: 0,
          ex: 0.42, ey: 0.22, ez: 0.62,
          familyPattern: PATTERN_NONE,
          fp8Flag: 0,
          phase: 0,
          bayId: bay.id,
          faceMask: 0,
          toneSelector: TONE_MATTE,
          lineStyle,
          cssWidth,
          lineAlpha,
          pickPart: 'crosspiece',
        },
        true,
      );

      // Four fixed KV guide rails at local x = -0.33, -0.15, +0.15, +0.33, y = 4.50, z in [-9.90, -0.30]
      const kvFp8 = kvCachePrecision(bay, precision) === 'fp8' ? 1 : 0;
      const railCenterZ = RAIL_START_Z - RAIL_LENGTH * 0.5; // -5.10
      for (let r = 0; r < KV_RAIL_OFFSETS.length; r++) {
        const rx = cx + KV_RAIL_OFFSETS[r];
        pushSolid(
          {
            cx: rx, cy: 4.50, cz: railCenterZ,
            ex: 0.045, ey: 0.06, ez: RAIL_LENGTH,
            familyPattern: PATTERN_NONE,
            fp8Flag: 0,
            phase: 0,
            bayId: bay.id,
            faceMask: 0,
            toneSelector: TONE_GUIDE,
            lineStyle: isActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
            cssWidth: isSelected ? 1.3 : 0.85,
            lineAlpha: isSelected ? 0.95 : 0.45,
            pickPart: `rail-${r}`,
          },
          true,
        );

        // Patch 1: KV slices j = 1..T at center y = 4.64 (flush atop rails at y = 4.53)
        for (let j = 1; j <= clampedTokens; j++) {
          const isFrontEdgeSlice = j === clampedTokens;
          const sliceActive = isActive;
          pushSolid(
            {
              cx: rx,
              cy: 4.64,
              cz: sliceCenterZ(j),
              ex: 0.12,
              ey: 0.22,
              ez: SLICE_DEPTH,
              familyPattern: PATTERN_NONE,
              fp8Flag: kvFp8,
              phase: 0,
              bayId: bay.id,
              faceMask: 0,
              toneSelector: TONE_MATTE,
              lineStyle: sliceActive || isFrontEdgeSlice ? (sliceActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL) : LINE_STYLE_NEUTRAL,
              cssWidth: sliceActive ? (isSelected ? 1.5 : 1.25) : 0.95,
              lineAlpha: sliceActive ? 1.0 : 0.58,
              pickPart: `kv-slice-${r}-${j}`,
            },
            true,
          );
        }
      }
    }
  }

  // --- Pack solid instances (48 bytes = 12 floats each) ---
  const solidData = new Float32Array(solids.length * SOLID_INSTANCE_FLOATS);
  const edgeDedup = new Map<string, LineSpec>();

  for (let i = 0; i < solids.length; i++) {
    const s = solids[i];
    const o = i * SOLID_INSTANCE_FLOATS;
    solidData[o + 0] = s.cx;
    solidData[o + 1] = s.cy;
    solidData[o + 2] = s.cz;
    solidData[o + 3] = s.ex;
    solidData[o + 4] = s.ey;
    solidData[o + 5] = s.ez;
    solidData[o + 6] = s.familyPattern;
    solidData[o + 7] = s.fp8Flag;
    solidData[o + 8] = s.phase;
    solidData[o + 9] = s.bayId;
    solidData[o + 10] = s.faceMask;
    solidData[o + 11] = s.toneSelector;

    emitCuboidEdges(s, edgeDedup);
  }

  // --- Depth ruler free wires at x = 28.5, y = 4.5 ---
  const freeLines: LineSpec[] = [];
  const pushFreeLine = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth: number, style: number, alpha: number,
  ) => {
    freeLines.push({
      ax, ay, az, bx, by, bz,
      cssWidth, style, alpha,
      ownerId: 0,
      face0: -1, face1: -1,
      ocx: 0, ocy: 0, ocz: 0,
      oex: 0, oey: 0, oez: 0,
    });
  };

  // Full capacity ruler rail (neutral)
  pushFreeLine(RULER_X, RULER_Y, RAIL_START_Z, RULER_X, RULER_Y, RAIL_END_Z, 1.0, LINE_STYLE_RULER, 0.55);

  // Active depth span on ruler from front datum (-0.30) to back of slice T (oxide red)
  const activeBackZ = sliceBackZ(clampedTokens);
  pushFreeLine(RULER_X, RULER_Y, RAIL_START_Z, RULER_X, RULER_Y, activeBackZ, 1.6, LINE_STYLE_ACTIVE, 1.0);

  const rulerTicks: RulerTickAnchor[] = [];
  for (let t = 1; t <= 16; t++) {
    const z = sliceBackZ(t);
    const isMajor = t === 1 || t === 8 || t === 16;
    const isCurrent = t === clampedTokens;
    const tickLen = isCurrent ? 0.55 : isMajor ? 0.42 : 0.24;
    pushFreeLine(
      RULER_X - tickLen * 0.4, RULER_Y, z,
      RULER_X + tickLen, RULER_Y, z,
      isCurrent ? 1.6 : 1.0,
      isCurrent ? LINE_STYLE_ACTIVE : LINE_STYLE_RULER,
      isCurrent ? 1.0 : isMajor ? 0.70 : 0.42,
    );
    rulerTicks.push({
      token: t,
      tickWorld: [RULER_X, RULER_Y, z],
      labelWorld: [RULER_X + 0.65, RULER_Y, z],
    });
  }

  // Subtle dashed-style transverse datum wire connecting current depth T across the 6 attention forks
  const firstAttnX = bayCenterX(6) - 0.45;
  pushFreeLine(firstAttnX, RULER_Y, activeBackZ, RULER_X, RULER_Y, activeBackZ, 0.9, LINE_STYLE_ACTIVE, 0.35);

  // --- Separate neutral and active line runs (neutral first, active last per §3) ---
  const neutralLines: LineSpec[] = [];
  const activeLines: LineSpec[] = [];

  for (const l of edgeDedup.values()) {
    if (l.style === LINE_STYLE_ACTIVE) activeLines.push(l);
    else neutralLines.push(l);
  }
  for (const l of freeLines) {
    if (l.style === LINE_STYLE_ACTIVE) activeLines.push(l);
    else neutralLines.push(l);
  }

  const orderedLines = [...neutralLines, ...activeLines];
  const lineData = new Float32Array(orderedLines.length * LINE_INSTANCE_FLOATS);
  for (let i = 0; i < orderedLines.length; i++) {
    const l = orderedLines[i];
    const o = i * LINE_INSTANCE_FLOATS;
    lineData[o + 0] = l.ax;
    lineData[o + 1] = l.ay;
    lineData[o + 2] = l.az;
    lineData[o + 3] = l.bx;
    lineData[o + 4] = l.by;
    lineData[o + 5] = l.bz;
    lineData[o + 6] = l.cssWidth;
    lineData[o + 7] = l.style;
    lineData[o + 8] = l.alpha;
    lineData[o + 9] = l.ownerId;
    lineData[o + 10] = l.face0;
    lineData[o + 11] = l.face1;
    lineData[o + 12] = l.ocx;
    lineData[o + 13] = l.ocy;
    lineData[o + 14] = l.ocz;
    lineData[o + 15] = l.oex;
    lineData[o + 16] = l.oey;
    lineData[o + 17] = l.oez;
  }

  // --- Figure 2 bracket anchors along residual baseline y = -0.30, z = 0.55 (§1) ---
  const bracketAnchors: BracketAnchor[] = spec.brackets.map((b, idx) => {
    const [first, last] = b;
    const leftX = bayCenterX(first) - 0.50;
    const rightX = bayCenterX(last) + 0.50;
    const midX = 0.5 * (leftX + rightX);
    return {
      index: idx as 0 | 1 | 2 | 3,
      first,
      last,
      leftWorld: [leftX, -0.30, 0.55],
      rightWorld: [rightX, -0.30, 0.55],
      midWorld: [midX, -0.30, 0.55],
    };
  });

  const bookendAnchors: BookendAnchor[] = [
    { id: 'embed', label: 'Embed', world: [-27.4, 4.0, 0] },
    { id: 'norm', label: 'RMSNorm', world: [26.45, 2.6, 0] },
    { id: 'output', label: 'Output', world: [27.4, 4.4, 0] },
  ];

  return {
    solidData,
    solidCount: solids.length,
    lineData,
    lineCount: orderedLines.length,
    neutralLineCount: neutralLines.length,
    activeLineCount: activeLines.length,
    fitBounds: MACRO_FIT_BOUNDS,
    bayAnchors,
    bracketAnchors,
    rulerTicks,
    bookendAnchors,
    solidPickBoxes,
    fasciaPickBoxes,
  };
}
