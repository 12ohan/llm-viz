import { linkProgram } from './device';

/**
 * Per-sheet instanced buffer layout: 20 floats = 80 bytes per flat matrix sheet.
 *
 *   [0..2]   iCenter     vec3  world center (cx, cy, cz)
 *   [3..5]   iExtent     vec3  world extent (ex, ey, ez) — ey ~ 0.06..0.09 thin sheet
 *   [6..7]   iGridDims   vec2  logical cell columns & rows (cols, rows) for adaptive LOD
 *   [8]      iSheetKind  float 0=standard, 1=causal_scores, 2=causal_softmax, 3=ghost_rail
 *   [9]      iRole       float 0=neutral, 1=weight, 2=activation, 3=norm, 4=kv_k, 5=kv_v, 6=scores, 7=softmax, 8=residual
 *   [10]     iFp8        float 0=BF16/FP32, 1=FP8 frosted KV storage
 *   [11]     iHighlight  float 0=normal, 1=active (oxide-red), 2=pathway (soft accent), 3=dimmed, 4=bf16_gold, 5=operand_a (indigo), 6=operand_b (teal)
 *   [12]     iSeed       float deterministic subtle value variation seed (until Increment 2 live Float32 texture)
 *   [13..16] iTexRect    vec4  (u0, v0, uWidth, vHeight) in uValueTex — uWidth <= 0 uses calm structural fill
 *   [17]     iValueScale float normalization scale for Float32 value texture (Increment 2)
 *   [18..19] iPad        vec2  reserved for Increment 2 flags (signed vs prob palette, etc.)
 */
export const SHEET_INSTANCE_FLOATS = 20;
export const SHEET_INSTANCE_BYTES = SHEET_INSTANCE_FLOATS * 4;

/**
 * Optional parallel per-sheet 3D orientation buffer: 6 floats = 24 bytes per sheet.
 *   [0..2] iAxisU  vec3  unit column axis (+col direction)
 *   [3..5] iAxisN  vec3  unit sheet normal axis (thin thickness axis)
 *   axisV = cross(axisU, axisN) is derived in-shader (strictly right-handed, det = +1).
 * When omitted in upload(), identity XZ axes U=(1,0,0), N=(0,1,0) are used automatically.
 */
export const SHEET_BASIS_FLOATS = 6;
export const SHEET_BASIS_BYTES = SHEET_BASIS_FLOATS * 4;

/**
 * Optional parallel per-sheet progressive reveal buffer: 4 floats = 16 bytes per sheet.
 *   [0] revealMode   0=none, 1=row-major progressive, 2=col-major progressive, 3=full uncomputed ghost
 *   [1] colFrontier  integer active column frontier (inclusive)
 *   [2] rowFrontier  integer active row frontier (inclusive)
 *   [3] pulsePhase   0..1 accumulation pulse phase for the active frontier cell
 * When omitted in upload(), defaults to (0, 0, 0, 0) = SHEET_REVEAL_NONE (100% revealed).
 */
export const SHEET_REVEAL_FLOATS = 4;
export const SHEET_REVEAL_BYTES = SHEET_REVEAL_FLOATS * 4;

export const SHEET_REVEAL_NONE = 0;
export const SHEET_REVEAL_ROW_MAJOR = 1;
export const SHEET_REVEAL_COL_MAJOR = 2;
export const SHEET_REVEAL_UNCOMPUTED = 3;

export const SHEET_KIND_STANDARD = 0;
export const SHEET_KIND_CAUSAL_SCORES = 1;
export const SHEET_KIND_CAUSAL_SOFTMAX = 2;
export const SHEET_KIND_GHOST_RAIL = 3;

export const SHEET_ROLE_NEUTRAL = 0;
export const SHEET_ROLE_WEIGHT = 1;
export const SHEET_ROLE_ACTIVATION = 2;
export const SHEET_ROLE_NORM = 3;
export const SHEET_ROLE_KV_K = 4;
export const SHEET_ROLE_KV_V = 5;
export const SHEET_ROLE_SCORES = 6;
export const SHEET_ROLE_SOFTMAX = 7;
export const SHEET_ROLE_RESIDUAL = 8;

export const SHEET_HL_NORMAL = 0;
export const SHEET_HL_ACTIVE = 1;
export const SHEET_HL_PATHWAY = 2;
export const SHEET_HL_DIMMED = 3;
export const SHEET_HL_BF16_GOLD = 4;
export const SHEET_HL_OPERAND_A = 5;
export const SHEET_HL_OPERAND_B = 6;

// Contiguous instance scalars [8..11] and [17..19] are packed as vec4/vec3 vertex attributes
// while preserving the exact 20-float (80-byte) SHEET_INSTANCE layout, keeping total vertex
// attribute slots at 13 (locations 0..12 <= WebGL2 guaranteed GL_MAX_VERTEX_ATTRIBS = 16).
const SHEETS_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

layout(location = 3) in vec3 iCenter;
layout(location = 4) in vec3 iExtent;
layout(location = 5) in vec2 iGridDims;
layout(location = 6) in vec4 iMeta;       // (sheetKind, role, fp8, highlight)
layout(location = 7) in float iSeed;
layout(location = 8) in vec4 iTexRect;
layout(location = 9) in vec3 iValueFlags; // (valueScale, flags.x, flags.y)
layout(location = 10) in vec3 iAxisU;
layout(location = 11) in vec3 iAxisN;
layout(location = 12) in vec4 iReveal;    // (revealMode, colFrontier, rowFrontier, pulsePhase)

uniform mat4 uViewProj;

out vec2 vUv;
flat out vec3 vLocalNormal;
flat out vec3 vWorldNormal;
flat out vec2 vGridDims;
flat out int vSheetKind;
flat out int vRole;
flat out float vFp8;
flat out int vHighlight;
flat out float vSeed;
flat out vec4 vTexRect;
flat out float vValueScale;
flat out vec2 vFlags;
flat out vec4 vReveal;

void main() {
  vec3 uAxis = normalize(iAxisU);
  vec3 nCand = iAxisN - dot(iAxisN, uAxis) * uAxis;
  vec3 nAxis = length(nCand) > 1e-5 ? normalize(nCand) : vec3(0.0, 1.0, 0.0);
  vec3 vAxis = normalize(cross(uAxis, nAxis));

  vec3 worldPos =
    iCenter +
    (aPos.x * iExtent.x) * uAxis +
    (aPos.y * iExtent.y) * nAxis +
    (aPos.z * iExtent.z) * vAxis;

  gl_Position = uViewProj * vec4(worldPos, 1.0);
  vUv = aUv;
  vLocalNormal = aNormal;
  vWorldNormal = normalize(
    aNormal.x * uAxis +
    aNormal.y * nAxis +
    aNormal.z * vAxis
  );
  vGridDims = max(iGridDims, vec2(1.0));
  vSheetKind = int(floor(iMeta.x + 0.5));
  vRole = int(floor(iMeta.y + 0.5));
  vFp8 = iMeta.z;
  vHighlight = int(floor(iMeta.w + 0.5));
  vSeed = iSeed;
  vTexRect = iTexRect;
  vValueScale = iValueFlags.x;
  vFlags = iValueFlags.yz;
  vReveal = iReveal;
}
`;

/**
 * Adapted from bbycroft/llm-viz blockRender.ts adaptive cell-grid LOD + calm studio
 * surface shader. Uses fwidth() on logical cell coordinates for both fine grid lines
 * and numerical value contrast LOD so dense sheets stay crisp and unaliased at every zoom.
 * Samples live Float32 RG32F values (R = value, G = masked -inf flag) consistently on both
 * +N and -N faces of arbitrarily oriented 3D thin sheets, and supports V2 progressive
 * accumulation reveal masking (uncomputed ghost vs computed vs active frontier cell).
 */
const SHEETS_FS = `#version 300 es
precision highp float;

in vec2 vUv;
flat in vec3 vLocalNormal;
flat in vec3 vWorldNormal;
flat in vec2 vGridDims;
flat in int vSheetKind;
flat in int vRole;
flat in float vFp8;
flat in int vHighlight;
flat in float vSeed;
flat in vec4 vTexRect;
flat in float vValueScale;
flat in vec2 vFlags;
flat in vec4 vReveal;

uniform sampler2D uValueTex;

out vec4 fragColor;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// Compute antialiased grid line coverage for a 1D coordinate using screen-space derivatives
// (adapted from Bycroft's zoom-dependent cell subdivision principle).
float gridAxisAlpha(float coord, float lineWeightPx) {
  float fw = max(fwidth(coord), 1e-5);
  // Fade smoothly out when cells become smaller than ~4.5 px so distant sheets never shimmer/alias
  float cellPx = 1.0 / fw;
  float lodFade = smoothstep(3.8, 10.5, cellPx);
  float distToEdge = abs(fract(coord - 0.5) - 0.5) / fw;
  float line = 1.0 - smoothstep(lineWeightPx * 0.5, lineWeightPx * 0.5 + 1.0, distToEdge);
  return line * lodFade;
}

vec3 roleBaseColor(int role) {
  // Calibrated editorial architectural slate palette with crisp figure-ground substance against #F7F8FA
  if (role == 1) return vec3(0.755, 0.790, 0.858); // weight: cool drafting blue-slate
  if (role == 2) return vec3(0.768, 0.818, 0.828); // activation: calm editorial slate-sage
  if (role == 3) return vec3(0.842, 0.816, 0.758); // norm: warm architectural stone
  if (role == 4) return vec3(0.728, 0.824, 0.802); // kv_k: crisp teal-sage
  if (role == 5) return vec3(0.730, 0.806, 0.860); // kv_v: crisp steel-cyan
  if (role == 6) return vec3(0.775, 0.796, 0.848); // scores: crisp slate-blue
  if (role == 7) return vec3(0.785, 0.825, 0.796); // softmax: mineral green-slate
  if (role == 8) return vec3(0.742, 0.842, 0.792); // residual: clean editorial emerald-slate
  return vec3(0.792, 0.810, 0.838);                // neutral matte paper-slate
}

vec3 rolePositiveColor(int role) {
  // High-clarity semantic tints for positive live Float32 values
  if (role == 1) return vec3(0.22, 0.32, 0.62); // weights: deep indigo-slate
  if (role == 2) return vec3(0.11, 0.49, 0.39); // activations: rich emerald-teal
  if (role == 3) return vec3(0.60, 0.42, 0.14); // rmsnorm: warm ochre-bronze
  if (role == 4) return vec3(0.10, 0.48, 0.45); // kv_k: deep teal-slate
  if (role == 5) return vec3(0.13, 0.40, 0.62); // kv_v: rich steel-blue
  if (role == 6) return vec3(0.19, 0.31, 0.64); // scores: crisp indigo-slate
  if (role == 7) return vec3(0.08, 0.50, 0.34); // softmax: crisp mineral emerald
  if (role == 8) return vec3(0.09, 0.50, 0.33); // residual: crisp emerald
  return vec3(0.24, 0.33, 0.48);
}

vec3 roleCellAccent(int role) {
  if (role == 1) return vec3(0.27, 0.37, 0.64);
  if (role == 2) return vec3(0.19, 0.44, 0.47);
  if (role == 3) return vec3(0.58, 0.43, 0.19);
  if (role == 4) return vec3(0.11, 0.49, 0.43);
  if (role == 5) return vec3(0.14, 0.42, 0.61);
  if (role == 6) return vec3(0.22, 0.33, 0.60);
  if (role == 7) return vec3(0.14, 0.49, 0.37);
  if (role == 8) return vec3(0.12, 0.50, 0.35);
  return vec3(0.29, 0.35, 0.44);
}

void main() {
  const vec3 OXIDE_RED = vec3(0.863, 0.298, 0.204); // #DC4C34 target highlight accent
  const vec3 OPERAND_A = vec3(0.220, 0.380, 0.741); // #3861BD indigo upstream operand A
  const vec3 OPERAND_B = vec3(0.102, 0.541, 0.478); // #1A8A7A teal upstream operand B
  const vec3 BF16_GOLD = vec3(0.765, 0.545, 0.145); // restrained warm bronze-gold
  const vec3 PAPER_BG  = vec3(0.969, 0.973, 0.980); // #F7F8FA
  const vec3 UNCOMPUTED_GHOST = vec3(0.945, 0.953, 0.969); // #F1F3F7 pale translucent paper-slate
  const vec3 CHARCOAL_NEG = vec3(0.18, 0.23, 0.31); // crisp charcoal-slate for negative values

  // Gentle studio directional shading from transformed world normal so 3D plane orientation
  // reads immediately without compromising Float32 numerical value contrast.
  vec3 keyLight = normalize(vec3(0.32, 0.86, 0.40));
  float ndl = dot(normalize(vWorldNormal), keyLight);
  float faceShade = clamp(0.905 + 0.105 * (0.5 + 0.5 * ndl), 0.90, 1.00);

  float revealMode = vReveal.x;
  bool isFullUncomputed = revealMode >= 2.5;

  // Side rims (local X/Z edges): crisp matte architectural card rim
  if (abs(vLocalNormal.y) < 0.5) {
    vec3 sideCol = vec3(0.54, 0.60, 0.69);
    if (vSheetKind == 3) {
      sideCol = vec3(0.80, 0.84, 0.89);
      if (vHighlight == 3) {
        sideCol = mix(sideCol, PAPER_BG, 0.60);
      }
    } else if (isFullUncomputed) {
      sideCol = mix(vec3(0.78, 0.82, 0.88), PAPER_BG, 0.48);
    } else if (vHighlight == 1) {
      sideCol = mix(sideCol, OXIDE_RED, 0.74);
    } else if (vHighlight == 5) {
      sideCol = mix(sideCol, OPERAND_A, 0.68);
    } else if (vHighlight == 6) {
      sideCol = mix(sideCol, OPERAND_B, 0.68);
    } else if (vHighlight == 2) {
      sideCol = mix(sideCol, OXIDE_RED, 0.34);
    } else if (vHighlight == 4) {
      sideCol = mix(sideCol, BF16_GOLD, 0.58);
    } else if (vHighlight == 3) {
      sideCol = mix(sideCol, PAPER_BG, 0.62);
    }
    float sideShade = 0.84 + 0.14 * (0.5 + 0.5 * ndl);
    fragColor = vec4(sideCol * sideShade, 1.0);
    return;
  }

  // Ghost capacity rail (1..16 KV capacity outline)
  if (vSheetKind == 3) {
    vec2 cell = vUv * vGridDims;
    float gx = gridAxisAlpha(cell.x, 0.8);
    float gy = gridAxisAlpha(cell.y, 0.8);
    float grid = max(gx, gy);

    vec2 edgeDist = min(vUv, 1.0 - vUv);
    vec2 edgePx = edgeDist / max(fwidth(vUv), vec2(1e-5));
    float border = 1.0 - smoothstep(0.6, 1.6, min(edgePx.x, edgePx.y));

    vec3 fill = vec3(0.925, 0.938, 0.955);
    vec3 lineCol = vec3(0.68, 0.73, 0.81);
    if (vHighlight == 1 || vHighlight == 2) {
      lineCol = mix(lineCol, OXIDE_RED, vHighlight == 1 ? 0.55 : 0.30);
    } else if (vHighlight == 5) {
      lineCol = mix(lineCol, OPERAND_A, 0.50);
    } else if (vHighlight == 6) {
      lineCol = mix(lineCol, OPERAND_B, 0.50);
    } else if (vHighlight == 3) {
      fill = mix(fill, PAPER_BG, 0.60);
      lineCol = mix(lineCol, PAPER_BG, 0.55);
    }
    vec3 col = mix(fill, lineCol, clamp(max(grid * 0.55, border * 0.85), 0.0, 1.0));
    fragColor = vec4(col * faceShade, 1.0);
    return;
  }

  vec3 base = roleBaseColor(vRole);
  vec3 posTint = rolePositiveColor(vRole);
  vec3 accent = roleCellAccent(vRole);

  // FP8 frosted KV storage indicator (honest production precision metadata on KV tracks only)
  if (vFp8 > 0.5) {
    base = mix(base, vec3(0.885, 0.904, 0.932), 0.42);
    posTint = mix(posTint, vec3(0.32, 0.41, 0.54), 0.32);
    accent = mix(accent, vec3(0.38, 0.45, 0.56), 0.38);
  }

  vec2 gridCoords = vUv * vGridDims;
  float cellCol = clamp(floor(vUv.x * vGridDims.x), 0.0, vGridDims.x - 1.0);
  float cellRow = clamp(floor((1.0 - vUv.y) * vGridDims.y), 0.0, vGridDims.y - 1.0);
  vec2 cellIdx = vec2(cellCol, cellRow);

  // V2 progressive accumulation reveal mask evaluation
  float colFrontier = floor(vReveal.y + 0.5);
  float rowFrontier = floor(vReveal.z + 0.5);
  float pulsePhase = clamp(vReveal.w, 0.0, 1.0);
  bool isUncomputed = false;
  bool isFrontierCell = false;

  if (isFullUncomputed) {
    isUncomputed = true;
  } else if (revealMode >= 1.5) {
    // Column-major progressive accumulation (SHEET_REVEAL_COL_MAJOR = 2)
    if (cellCol > colFrontier + 0.5 || (abs(cellCol - colFrontier) < 0.5 && cellRow > rowFrontier + 0.5)) {
      isUncomputed = true;
    } else if (abs(cellCol - colFrontier) < 0.5 && abs(cellRow - rowFrontier) < 0.5) {
      isFrontierCell = true;
    }
  } else if (revealMode >= 0.5) {
    // Row-major progressive accumulation (SHEET_REVEAL_ROW_MAJOR = 1)
    if (cellRow > rowFrontier + 0.5 || (abs(cellRow - rowFrontier) < 0.5 && cellCol > colFrontier + 0.5)) {
      isUncomputed = true;
    } else if (abs(cellRow - rowFrontier) < 0.5 && abs(cellCol - colFrontier) < 0.5) {
      isFrontierCell = true;
    }
  }

  // Screen-space cell size for numerical value LOD (prevents dense sheets turning into checkerboard noise)
  // Compute fwidth(gridCoords) here in uniform control flow for both LOD and frontier cell border AA.
  vec2 cellGridFw = max(fwidth(gridCoords), vec2(1e-5));
  float cellFwMax = max(cellGridFw.x, cellGridFw.y);
  float cellPxMin = 1.0 / cellFwMax;
  float valueLod = smoothstep(1.35, 4.6, cellPxMin);

  bool isCausal = (vSheetKind == 1 || vSheetKind == 2);
  bool hasLiveTex = (vTexRect.z > 0.0 && vTexRect.w > 0.0);
  // For live RG32F value textures, channel G carries the exact per-cell causal mask (supports rectangular [S, P+S]).
  // Fall back to geometric cellCol > cellRow only when no live value texture is bound.
  bool isMaskedUpper = isCausal && !hasLiveTex && (cellCol > cellRow);

  // --- Value surface: Uncomputed ghost OR Live Float32 RG32F texture OR fallback structural tone ---
  if (isUncomputed) {
    base = UNCOMPUTED_GHOST;
  } else if (hasLiveTex) {
    vec2 sampleUv = vTexRect.xy + vec2((cellCol + 0.5) / vGridDims.x, (cellRow + 0.5) / vGridDims.y) * vTexRect.zw;
    vec2 texVal = texture(uValueTex, sampleUv).rg;
    if (texVal.g > 0.5) {
      isMaskedUpper = true;
    }

    if (isMaskedUpper) {
      // Distinct muted paper-slate + delicate 45-deg drafting hatch for masked (-inf / 0) future cells
      base = vec3(0.942, 0.949, 0.962);
      float hatchCoord = (vUv.x + vUv.y) * max(vGridDims.x, 4.0) * 1.45;
      float hatchLine = gridAxisAlpha(hatchCoord, 0.55) * 0.24;
      base = mix(base, vec3(0.70, 0.74, 0.81), hatchLine);
    } else if (vFlags.x > 0.5) {
      // Probability [0..1] encoding (Causal Softmax): deepened near-zero slate -> crisp mineral emerald
      float prob = clamp(texVal.r * vValueScale, 0.0, 1.0);
      float shaped = pow(prob, 0.56);
      vec3 zeroCol = mix(vec3(0.875, 0.892, 0.890), base, 0.46);
      float probGain = mix(0.68, 0.90, valueLod);
      base = mix(zeroCol, posTint, shaped * probGain);
    } else {
      // Calibrated signed Float32 diverging encoding with zoom-adaptive contrast LOD:
      // deepened neutral zero-value tone gives solid editorial slab presence against #F7F8FA.
      float normVal = clamp(texVal.r * vValueScale, -1.0, 1.0);
      float mag = pow(abs(normVal), 0.74);
      vec3 neutralZero = mix(vec3(0.848, 0.865, 0.890), base, 0.50);
      float contrast = mix(0.38, 1.0, valueLod);
      vec3 negCol = mix(neutralZero, CHARCOAL_NEG, mag * 0.78 * contrast);
      vec3 posCol = mix(neutralZero, posTint, mag * 0.82 * contrast);
      base = normVal >= 0.0 ? posCol : negCol;
    }
  } else if (isMaskedUpper) {
    base = vec3(0.942, 0.949, 0.962);
    float hatchCoord = (vUv.x + vUv.y) * max(vGridDims.x, 4.0) * 1.5;
    float hatchLine = gridAxisAlpha(hatchCoord, 0.55) * 0.18;
    base = mix(base, vec3(0.74, 0.78, 0.84), hatchLine);
  } else {
    float detailVis = smoothstep(3.5, 9.0, cellPxMin);
    float h = hash21(cellIdx + vec2(vSeed * 13.7, vSeed * 5.3));
    float subtle = (h - 0.5) * 0.075 * detailVis;
    base = mix(base, accent, max(0.0, subtle * 2.4)) + vec3(min(0.0, subtle));
    if (isCausal) {
      float diagProximity = 1.0 - clamp((cellRow - cellCol) / max(vGridDims.x, 1.0), 0.0, 1.0);
      float strength = vSheetKind == 2 ? (0.16 + 0.26 * pow(diagProximity, 1.6)) : (0.10 + 0.16 * diagProximity);
      base = mix(base, accent, strength);
    }
  }

  // --- Adaptive Zoom-Dependent Cell Grid LOD (fine cells + coarse 4x/8x subdivisions) ---
  float fineX = gridAxisAlpha(gridCoords.x, isUncomputed ? 0.60 : 0.78);
  float fineY = gridAxisAlpha(gridCoords.y, isUncomputed ? 0.60 : 0.78);
  float fineGrid = max(fineX, fineY);

  vec2 coarseDims = max(floor(vGridDims / 4.0), vec2(1.0));
  float coarseX = vGridDims.x >= 8.0 ? gridAxisAlpha(vUv.x * coarseDims.x, isUncomputed ? 0.72 : 0.98) : 0.0;
  float coarseY = vGridDims.y >= 8.0 ? gridAxisAlpha(vUv.y * coarseDims.y, isUncomputed ? 0.72 : 0.98) : 0.0;
  float coarseGrid = max(coarseX, coarseY);

  vec3 gridLineCol = isUncomputed
    ? vec3(0.845, 0.865, 0.898)
    : isMaskedUpper
      ? vec3(0.82, 0.85, 0.89)
      : mix(base * 0.72, accent * 0.66, 0.38);

  float gridStrength = isUncomputed ? 0.34 : isMaskedUpper ? 0.45 : 0.58;
  float coarseStrength = isUncomputed ? 0.22 : 0.46;
  vec3 col = mix(base, gridLineCol, clamp(max(fineGrid * gridStrength, coarseGrid * coarseStrength), 0.0, 0.78));

  // Subtle luminous oxide-red accumulation border & tint on the active frontier cell
  if (isFrontierCell && !isUncomputed && !isMaskedUpper) {
    vec2 cellUv = vec2(vUv.x * vGridDims.x, (1.0 - vUv.y) * vGridDims.y);
    vec2 cellLocal = fract(cellUv);
    vec2 cellEdgeDist = min(cellLocal, 1.0 - cellLocal);
    vec2 cellEdgePx = cellEdgeDist / max(cellGridFw, vec2(1e-4));
    float minCellEdgePx = min(cellEdgePx.x, cellEdgePx.y);
    float pulseAmt = 0.55 + 0.45 * sin(pulsePhase * 6.2831853);
    float frontierBorderPx = mix(1.45, 2.35, pulseAmt);
    float frontierMask = 1.0 - smoothstep(frontierBorderPx - 0.45, frontierBorderPx + 0.55, minCellEdgePx);
    vec3 luminousRed = mix(OXIDE_RED, vec3(1.0, 0.56, 0.44), 0.24 * pulseAmt);
    col = mix(col, mix(OXIDE_RED, vec3(1.0, 0.96, 0.94), 0.32), 0.13 + 0.07 * pulseAmt);
    col = mix(col, luminousRed, frontierMask * 0.92);
  }

  // --- Crisp Antialiased Outer Sheet Frame Border ---
  vec2 edgeDist = min(vUv, 1.0 - vUv);
  vec2 edgePx = edgeDist / max(fwidth(vUv), vec2(1e-5));
  float minEdgePx = min(edgePx.x, edgePx.y);

  float borderPx =
    isFullUncomputed
      ? 1.05
      : vHighlight == 1
        ? 2.35
        : (vHighlight == 5 || vHighlight == 6
            ? 2.05
            : (vHighlight == 2 || vHighlight == 4 ? 1.65 : 1.25));
  float borderMask = 1.0 - smoothstep(borderPx - 0.5, borderPx + 0.6, minEdgePx);

  // Calibrated crisp slate border (#617087 / #6E7B8F family) for solid presence against #F7F8FA
  vec3 borderCol = vec3(0.38, 0.44, 0.53);
  if (vFp8 > 0.5) {
    borderCol = vec3(0.47, 0.53, 0.62);
  }

  // Apply selection / operand / pathway / dimming hierarchy
  if (isFullUncomputed) {
    // Full-sheet uncomputed ghost recedes cleanly into pale drafting outline
    borderCol = vec3(0.76, 0.80, 0.86);
  } else if (vHighlight == 1) {
    // Active hovered/selected/target sheet: crisp oxide-red border + delicate warm face lift (computed cells only)
    if (!isUncomputed) {
      col = mix(col, OXIDE_RED, isMaskedUpper ? 0.04 : 0.14);
    }
    borderCol = OXIDE_RED;
  } else if (vHighlight == 5) {
    // Upstream Operand A sheet: crisp indigo border (#3861BD) + subtle cool lift
    if (!isUncomputed) {
      col = mix(col, OPERAND_A, isMaskedUpper ? 0.03 : 0.09);
    }
    borderCol = OPERAND_A;
  } else if (vHighlight == 6) {
    // Upstream Operand B sheet: crisp teal border (#1A8A7A) + subtle mineral lift
    if (!isUncomputed) {
      col = mix(col, OPERAND_B, isMaskedUpper ? 0.03 : 0.09);
    }
    borderCol = OPERAND_B;
  } else if (vHighlight == 2) {
    // Connected pathway sheet (e.g., shared KV head, causal scores, V out for selected Q_i)
    if (!isUncomputed) {
      col = mix(col, OXIDE_RED, isMaskedUpper ? 0.02 : 0.07);
    }
    borderCol = mix(borderCol, OXIDE_RED, 0.68);
  } else if (vHighlight == 4) {
    // BF16 preserved border in FP8 Selective mode (quiet restrained gold-bronze edge)
    borderCol = BF16_GOLD;
  } else if (vHighlight == 3) {
    // Unrelated sheets clearly recede toward paper background while remaining readable
    col = mix(col, PAPER_BG, 0.62);
    borderCol = mix(borderCol, PAPER_BG, 0.58);
  }

  // Subtle FP8 diagonal corner tick in top-right of KV cache sheets
  if (!isUncomputed && vFp8 > 0.5 && vUv.x > 0.82 && vUv.y > 0.82 && (vUv.x + vUv.y) > 1.78) {
    col = mix(col, vec3(0.45, 0.52, 0.64), 0.38);
  }

  col = mix(col * faceShade, borderCol, borderMask);
  fragColor = vec4(col, 1.0);
}
`;

function buildUnitSheetGeometry(): { vertices: Float32Array; indices: Uint16Array } {
  // Position (3), Normal (3), UV (2) -> 8 floats = 32 bytes/vert
  const verts: number[] = [];
  const indices: number[] = [];

  const faces: readonly {
    n: readonly [number, number, number];
    corners: readonly (readonly [number, number, number, number, number])[];
  }[] = [
    // +Y (+N) top/front data face: u = x + 0.5 (col 0 at -0.5), v = 0.5 - z (row 0 at z = -0.5 -> v = 1)
    {
      n: [0, 1, 0],
      corners: [
        [-0.5, 0.5, 0.5, 0, 0],
        [0.5, 0.5, 0.5, 1, 0],
        [0.5, 0.5, -0.5, 1, 1],
        [-0.5, 0.5, -0.5, 0, 1],
      ],
    },
    // -Y (-N) bottom/back data face: derives UV from the exact same sheet-local (x, z) coordinates
    // (u = x + 0.5, v = 0.5 - z) with outward CCW winding so every physical cell displays and picks
    // the exact same numerical tensor value from either side.
    {
      n: [0, -1, 0],
      corners: [
        [-0.5, -0.5, -0.5, 0, 1],
        [0.5, -0.5, -0.5, 1, 1],
        [0.5, -0.5, 0.5, 1, 0],
        [-0.5, -0.5, 0.5, 0, 0],
      ],
    },
    // +X side
    {
      n: [1, 0, 0],
      corners: [
        [0.5, -0.5, 0.5, 0, 0],
        [0.5, -0.5, -0.5, 1, 0],
        [0.5, 0.5, -0.5, 1, 1],
        [0.5, 0.5, 0.5, 0, 1],
      ],
    },
    // -X side
    {
      n: [-1, 0, 0],
      corners: [
        [-0.5, -0.5, -0.5, 0, 0],
        [-0.5, -0.5, 0.5, 1, 0],
        [-0.5, 0.5, 0.5, 1, 1],
        [-0.5, 0.5, -0.5, 0, 1],
      ],
    },
    // +Z side
    {
      n: [0, 0, 1],
      corners: [
        [-0.5, -0.5, 0.5, 0, 0],
        [0.5, -0.5, 0.5, 1, 0],
        [0.5, 0.5, 0.5, 1, 1],
        [-0.5, 0.5, 0.5, 0, 1],
      ],
    },
    // -Z side
    {
      n: [0, 0, -1],
      corners: [
        [0.5, -0.5, -0.5, 0, 0],
        [-0.5, -0.5, -0.5, 1, 0],
        [-0.5, 0.5, -0.5, 1, 1],
        [0.5, 0.5, -0.5, 0, 1],
      ],
    },
  ];

  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    const base = f * 4;
    for (const c of face.corners) {
      verts.push(c[0], c[1], c[2], face.n[0], face.n[1], face.n[2], c[3], c[4]);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { vertices: new Float32Array(verts), indices: new Uint16Array(indices) };
}

/**
 * Instanced WebGL2 pass for thin, dimension-aware flat matrix sheets in arbitrary 3D orientations.
 * Supports adaptive fwidth() grid & value LOD, double-sided consistent RG32F cell lookup,
 * causal lower-triangle rendering, restrained matte edges, and zero-overhead XZ defaults.
 */
export class SheetsPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private quadVbo!: WebGLBuffer;
  private quadIbo!: WebGLBuffer;
  private instanceVbo!: WebGLBuffer;
  private basisVbo!: WebGLBuffer;
  private revealVbo!: WebGLBuffer;
  private dummyValueTex!: WebGLTexture;
  private liveValueTex: WebGLTexture | null = null;
  private boundValueTex: WebGLTexture | null = null;
  private uViewProj!: WebGLUniformLocation | null;
  private uValueTex!: WebGLUniformLocation | null;
  private instanceCount = 0;
  private hasBasisBuffer = false;
  private hasRevealBuffer = false;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.init();
  }

  init(): void {
    const gl = this.gl;
    this.program = linkProgram(gl, SHEETS_VS, SHEETS_FS);
    this.uViewProj = gl.getUniformLocation(this.program, 'uViewProj');
    this.uValueTex = gl.getUniformLocation(this.program, 'uValueTex');

    const { vertices, indices } = buildUnitSheetGeometry();
    const vao = gl.createVertexArray();
    const quadVbo = gl.createBuffer();
    const quadIbo = gl.createBuffer();
    const instanceVbo = gl.createBuffer();
    const basisVbo = gl.createBuffer();
    const revealVbo = gl.createBuffer();
    const dummyTex = gl.createTexture();
    const liveTex = gl.createTexture();
    if (!vao || !quadVbo || !quadIbo || !instanceVbo || !basisVbo || !revealVbo || !dummyTex || !liveTex) {
      throw new Error('Failed to allocate SheetsPass WebGL resources');
    }

    this.vao = vao;
    this.quadVbo = quadVbo;
    this.quadIbo = quadIbo;
    this.instanceVbo = instanceVbo;
    this.basisVbo = basisVbo;
    this.revealVbo = revealVbo;
    this.dummyValueTex = dummyTex;
    this.liveValueTex = liveTex;

    // 1x1 neutral zero texture fallback
    gl.bindTexture(gl.TEXTURE_2D, this.dummyValueTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

    // Live Float32 RG32F value atlas texture (NEAREST filtering is core WebGL2 — zero extensions needed)
    gl.bindTexture(gl.TEXTURE_2D, this.liveValueTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.bindVertexArray(this.vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.quadIbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    const bindInst = (loc: number, size: number, offsetFloats: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, SHEET_INSTANCE_BYTES, offsetFloats * 4);
      gl.vertexAttribDivisor(loc, 1);
    };
    bindInst(3, 3, 0);  // iCenter      [0..2]
    bindInst(4, 3, 3);  // iExtent      [3..5]
    bindInst(5, 2, 6);  // iGridDims    [6..7]
    bindInst(6, 4, 8);  // iMeta        [8..11]: (sheetKind, role, fp8, highlight)
    bindInst(7, 1, 12); // iSeed        [12]
    bindInst(8, 4, 13); // iTexRect     [13..16]
    bindInst(9, 3, 17); // iValueFlags  [17..19]: (valueScale, flags.x, flags.y)

    // Default identity XZ basis attributes (locations 10 & 11 disabled until basisData is uploaded)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.basisVbo);
    gl.vertexAttribPointer(10, 3, gl.FLOAT, false, SHEET_BASIS_BYTES, 0);
    gl.vertexAttribDivisor(10, 1);
    gl.vertexAttribPointer(11, 3, gl.FLOAT, false, SHEET_BASIS_BYTES, 12);
    gl.vertexAttribDivisor(11, 1);
    gl.disableVertexAttribArray(10);
    gl.disableVertexAttribArray(11);
    gl.vertexAttrib3f(10, 1, 0, 0);
    gl.vertexAttrib3f(11, 0, 1, 0);

    // Default progressive reveal attribute (location 12 disabled -> (0,0,0,0) = SHEET_REVEAL_NONE)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.revealVbo);
    gl.vertexAttribPointer(12, 4, gl.FLOAT, false, SHEET_REVEAL_BYTES, 0);
    gl.vertexAttribDivisor(12, 1);
    gl.disableVertexAttribArray(12);
    gl.vertexAttrib4f(12, 0, 0, 0, 0);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Bind a custom WebGLTexture for real activation/weight values. */
  setValueTexture(tex: WebGLTexture | null): void {
    this.boundValueTex = tex;
  }

  /** Upload a live Float32 RG32F value atlas (R = value, G = masked -inf flag) directly into uValueTex. */
  uploadValueAtlas(width: number, height: number, rg32fData: Float32Array): void {
    if (!this.liveValueTex) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.liveValueTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, width, height, 0, gl.RG, gl.FLOAT, rg32fData);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.boundValueTex = this.liveValueTex;
  }

  upload(
    data: Float32Array,
    count: number,
    basisData?: Float32Array,
    revealData?: Float32Array,
  ): void {
    const gl = this.gl;
    this.instanceCount = count;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);

    gl.bindVertexArray(this.vao);
    if (basisData && basisData.length >= count * SHEET_BASIS_FLOATS) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.basisVbo);
      gl.bufferData(gl.ARRAY_BUFFER, basisData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(10);
      gl.enableVertexAttribArray(11);
      this.hasBasisBuffer = true;
    } else {
      gl.disableVertexAttribArray(10);
      gl.disableVertexAttribArray(11);
      gl.vertexAttrib3f(10, 1, 0, 0);
      gl.vertexAttrib3f(11, 0, 1, 0);
      this.hasBasisBuffer = false;
    }

    if (revealData && revealData.length >= count * SHEET_REVEAL_FLOATS) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.revealVbo);
      gl.bufferData(gl.ARRAY_BUFFER, revealData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(12);
      this.hasRevealBuffer = true;
    } else {
      gl.disableVertexAttribArray(12);
      gl.vertexAttrib4f(12, 0, 0, 0, 0);
      this.hasRevealBuffer = false;
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  draw(viewProj: Float32Array): void {
    if (this.instanceCount <= 0) return;
    const gl = this.gl;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.boundValueTex ?? this.dummyValueTex);
    gl.uniform1i(this.uValueTex, 0);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.disable(gl.BLEND);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1.0, 1.0);

    gl.bindVertexArray(this.vao);
    if (!this.hasBasisBuffer) {
      gl.vertexAttrib3f(10, 1, 0, 0);
      gl.vertexAttrib3f(11, 0, 1, 0);
    }
    if (!this.hasRevealBuffer) {
      gl.vertexAttrib4f(12, 0, 0, 0, 0);
    }
    gl.drawElementsInstanced(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    gl.bindVertexArray(null);

    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.quadVbo);
    gl.deleteBuffer(this.quadIbo);
    gl.deleteBuffer(this.instanceVbo);
    gl.deleteBuffer(this.basisVbo);
    gl.deleteBuffer(this.revealVbo);
    gl.deleteTexture(this.dummyValueTex);
    if (this.liveValueTex) {
      gl.deleteTexture(this.liveValueTex);
      this.liveValueTex = null;
    }
  }
}
