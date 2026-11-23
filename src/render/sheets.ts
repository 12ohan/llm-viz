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
 *   [11]     iHighlight  float 0=normal, 1=active (oxide-red), 2=pathway (soft accent), 3=dimmed, 4=bf16_gold
 *   [12]     iSeed       float deterministic subtle value variation seed (until Increment 2 live Float32 texture)
 *   [13..16] iTexRect    vec4  (u0, v0, uWidth, vHeight) in uValueTex — uWidth <= 0 uses calm structural fill
 *   [17]     iValueScale float normalization scale for Float32 value texture (Increment 2)
 *   [18..19] iPad        vec2  reserved for Increment 2 flags (signed vs prob palette, etc.)
 */
export const SHEET_INSTANCE_FLOATS = 20;
export const SHEET_INSTANCE_BYTES = SHEET_INSTANCE_FLOATS * 4;

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

const SHEETS_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

layout(location = 3) in vec3 iCenter;
layout(location = 4) in vec3 iExtent;
layout(location = 5) in vec2 iGridDims;
layout(location = 6) in float iSheetKind;
layout(location = 7) in float iRole;
layout(location = 8) in float iFp8;
layout(location = 9) in float iHighlight;
layout(location = 10) in float iSeed;
layout(location = 11) in vec4 iTexRect;
layout(location = 12) in float iValueScale;

uniform mat4 uViewProj;

out vec2 vUv;
flat out vec3 vNormal;
flat out vec2 vGridDims;
flat out int vSheetKind;
flat out int vRole;
flat out float vFp8;
flat out int vHighlight;
flat out float vSeed;
flat out vec4 vTexRect;
flat out float vValueScale;

void main() {
  vec3 worldPos = iCenter + aPos * iExtent;
  gl_Position = uViewProj * vec4(worldPos, 1.0);
  vUv = aUv;
  vNormal = aNormal;
  vGridDims = max(iGridDims, vec2(1.0));
  vSheetKind = int(floor(iSheetKind + 0.5));
  vRole = int(floor(iRole + 0.5));
  vFp8 = iFp8;
  vHighlight = int(floor(iHighlight + 0.5));
  vSeed = iSeed;
  vTexRect = iTexRect;
  vValueScale = iValueScale;
}
`;

/**
 * Adapted from bbycroft/llm-viz blockRender.ts adaptive cell-grid LOD + calm drafting-table
 * surface shader. Uses fwidth() on logical cell coordinates so fine matrix cells smoothly
 * emerge as you zoom in and stay completely clean/unaliased when pulled back.
 */
const SHEETS_FS = `#version 300 es
precision highp float;

in vec2 vUv;
flat in vec3 vNormal;
flat in vec2 vGridDims;
flat in int vSheetKind;
flat in int vRole;
flat in float vFp8;
flat in int vHighlight;
flat in float vSeed;
flat in vec4 vTexRect;
flat in float vValueScale;

// Bound to a 1x1 dummy texture in Increment 1; Increment 2 uploads real Float32 tensors here
// without changing geometry or shader structure.
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
  // Calm matte neutral slate palette with delicate semantic tinting
  if (role == 1) return vec3(0.845, 0.870, 0.915); // weight: cool blue-slate
  if (role == 2) return vec3(0.855, 0.888, 0.892); // activation: calm slate-sage
  if (role == 3) return vec3(0.910, 0.896, 0.858); // norm: warm stone
  if (role == 4) return vec3(0.822, 0.895, 0.878); // kv_k: pale teal-sage
  if (role == 5) return vec3(0.824, 0.882, 0.918); // kv_v: pale steel-cyan
  if (role == 6) return vec3(0.862, 0.876, 0.908); // scores: crisp slate-blue
  if (role == 7) return vec3(0.872, 0.892, 0.876); // softmax: pale mineral green
  if (role == 8) return vec3(0.835, 0.910, 0.872); // residual: clean light emerald-slate
  return vec3(0.875, 0.888, 0.906);                // neutral matte paper-slate
}

vec3 roleCellAccent(int role) {
  if (role == 1) return vec3(0.32, 0.42, 0.68);
  if (role == 2) return vec3(0.24, 0.47, 0.50);
  if (role == 3) return vec3(0.62, 0.48, 0.24);
  if (role == 4) return vec3(0.14, 0.52, 0.46);
  if (role == 5) return vec3(0.18, 0.46, 0.64);
  if (role == 6) return vec3(0.27, 0.37, 0.62);
  if (role == 7) return vec3(0.18, 0.52, 0.40);
  if (role == 8) return vec3(0.15, 0.54, 0.38);
  return vec3(0.34, 0.39, 0.47);
}

void main() {
  const vec3 OXIDE_RED = vec3(0.863, 0.298, 0.204); // #DC4C34 single highlight accent
  const vec3 BF16_GOLD = vec3(0.765, 0.545, 0.145); // restrained warm bronze-gold
  const vec3 PAPER_BG  = vec3(0.969, 0.973, 0.980); // #F7F8FA

  // Side edges & bottom face: plain matte neutral rim — thin architectural card edge
  if (vNormal.y < 0.5) {
    vec3 sideCol = vec3(0.66, 0.70, 0.76);
    if (vSheetKind == 3) {
      sideCol = vec3(0.84, 0.87, 0.91);
    } else if (vHighlight == 1) {
      sideCol = mix(sideCol, OXIDE_RED, 0.72);
    } else if (vHighlight == 2) {
      sideCol = mix(sideCol, OXIDE_RED, 0.32);
    } else if (vHighlight == 4) {
      sideCol = mix(sideCol, BF16_GOLD, 0.55);
    } else if (vHighlight == 3) {
      sideCol = mix(sideCol, PAPER_BG, 0.45);
    }
    float sideShade = abs(vNormal.z) > 0.5 ? 0.94 : 0.87;
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

    vec3 fill = vec3(0.935, 0.945, 0.960);
    vec3 lineCol = vec3(0.76, 0.80, 0.86);
    if (vHighlight == 1 || vHighlight == 2) {
      lineCol = mix(lineCol, OXIDE_RED, vHighlight == 1 ? 0.55 : 0.30);
    }
    vec3 col = mix(fill, lineCol, clamp(max(grid * 0.55, border * 0.85), 0.0, 1.0));
    fragColor = vec4(col, 1.0);
    return;
  }

  vec3 base = roleBaseColor(vRole);
  vec3 accent = roleCellAccent(vRole);

  // FP8 frosted KV storage indicator (honest metadata tint on KV cache tracks only)
  if (vFp8 > 0.5) {
    base = mix(base, vec3(0.915, 0.928, 0.948), 0.52);
    accent = mix(accent, vec3(0.42, 0.48, 0.58), 0.45);
  }

  vec2 gridCoords = vUv * vGridDims;
  vec2 cellIdx = floor(gridCoords);

  // --- Value surface (Increment 2 live Float32 texture hook OR calm structural tone) ---
  float valTone = 0.0;
  if (vTexRect.z > 0.0 && vTexRect.w > 0.0) {
    vec2 sampleUv = vTexRect.xy + vUv * vTexRect.zw;
    float rawVal = texture(uValueTex, sampleUv).r;
    valTone = clamp(rawVal * vValueScale, -1.0, 1.0);
    // Diverging cool-slate <-> warm-oxide encoding ready for Increment 2
    vec3 negCol = mix(base, vec3(0.24, 0.39, 0.66), abs(valTone) * 0.75);
    vec3 posCol = mix(base, OXIDE_RED, abs(valTone) * 0.72);
    base = valTone >= 0.0 ? posCol : negCol;
  } else {
    // Increment 1: restrained, low-contrast structural micro-variation (never noisy checkerboard)
    float fwMax = max(fwidth(gridCoords.x), fwidth(gridCoords.y));
    float detailVis = smoothstep(3.5, 9.0, 1.0 / max(fwMax, 1e-5));
    float h = hash21(cellIdx + vec2(vSeed * 13.7, vSeed * 5.3));
    float subtle = (h - 0.5) * 0.065 * detailVis;
    base = mix(base, accent, max(0.0, subtle * 2.2)) + vec3(min(0.0, subtle));
  }

  // --- Causal Lower-Triangle vs Masked Upper-Triangle (-inf) for Scores & Softmax ---
  bool isCausal = (vSheetKind == 1 || vSheetKind == 2);
  bool isMaskedUpper = false;
  if (isCausal) {
    // Row 0 = top (vUv.y near 1 -> row = 0), Col 0 = left (vUv.x near 0 -> col = 0)
    // Valid causal entries satisfy col <= row; upper triangle col > row is masked (-inf -> 0).
    float col = floor(vUv.x * vGridDims.x);
    float row = floor((1.0 - vUv.y) * vGridDims.y);
    if (col > row) {
      isMaskedUpper = true;
      base = vec3(0.946, 0.952, 0.964);
      // Very subtle 45-degree drafting hatch in masked upper triangle when zoomed enough
      float hatchCoord = (vUv.x + vUv.y) * max(vGridDims.x, 4.0) * 1.5;
      float hatchLine = gridAxisAlpha(hatchCoord, 0.55) * 0.16;
      base = mix(base, vec3(0.76, 0.79, 0.84), hatchLine);
    } else {
      // Active causal lower triangle: softly emphasize diagonal & recent context
      float diagProximity = 1.0 - clamp((row - col) / max(vGridDims.x, 1.0), 0.0, 1.0);
      float strength = vSheetKind == 2 ? (0.14 + 0.24 * pow(diagProximity, 1.6)) : (0.08 + 0.14 * diagProximity);
      base = mix(base, accent, strength);
    }
  }

  // --- Adaptive Zoom-Dependent Cell Grid LOD (fine cells + coarse 4x/8x subdivisions) ---
  float fineX = gridAxisAlpha(gridCoords.x, 0.75);
  float fineY = gridAxisAlpha(gridCoords.y, 0.75);
  float fineGrid = max(fineX, fineY);

  vec2 coarseDims = max(floor(vGridDims / 4.0), vec2(1.0));
  float coarseX = vGridDims.x >= 8.0 ? gridAxisAlpha(vUv.x * coarseDims.x, 0.95) : 0.0;
  float coarseY = vGridDims.y >= 8.0 ? gridAxisAlpha(vUv.y * coarseDims.y, 0.95) : 0.0;
  float coarseGrid = max(coarseX, coarseY);

  vec3 gridLineCol = isMaskedUpper
    ? vec3(0.84, 0.86, 0.90)
    : mix(base * 0.78, accent * 0.72, 0.35);

  float gridStrength = isMaskedUpper ? 0.45 : 0.52;
  vec3 col = mix(base, gridLineCol, clamp(max(fineGrid * gridStrength, coarseGrid * 0.42), 0.0, 0.75));

  // --- Crisp Antialiased Outer Sheet Frame Border ---
  vec2 edgeDist = min(vUv, 1.0 - vUv);
  vec2 edgePx = edgeDist / max(fwidth(vUv), vec2(1e-5));
  float minEdgePx = min(edgePx.x, edgePx.y);

  float borderPx = vHighlight == 1 ? 2.2 : (vHighlight == 2 || vHighlight == 4 ? 1.55 : 1.05);
  float borderMask = 1.0 - smoothstep(borderPx - 0.5, borderPx + 0.6, minEdgePx);

  vec3 borderCol = vec3(0.46, 0.52, 0.60);
  if (vFp8 > 0.5) {
    borderCol = vec3(0.56, 0.62, 0.70);
  }

  // Apply selection / pathway / dimming hierarchy
  if (vHighlight == 1) {
    // Active hovered/selected sheet: crisp oxide-red border + delicate warm face lift
    col = mix(col, OXIDE_RED, isMaskedUpper ? 0.04 : 0.14);
    borderCol = OXIDE_RED;
  } else if (vHighlight == 2) {
    // Connected pathway sheet (e.g., shared KV head, causal scores, V out for selected Q_i)
    col = mix(col, OXIDE_RED, isMaskedUpper ? 0.02 : 0.07);
    borderCol = mix(borderCol, OXIDE_RED, 0.68);
  } else if (vHighlight == 4) {
    // BF16 preserved border in FP8 Selective mode (quiet restrained gold-bronze edge)
    borderCol = BF16_GOLD;
  } else if (vHighlight == 3) {
    // Non-selected background head/lane gently recedes so active pathway reads instantaneously
    col = mix(col, PAPER_BG, 0.46);
    borderCol = mix(borderCol, PAPER_BG, 0.42);
  }

  // Subtle FP8 diagonal corner tick in top-right of KV cache sheets
  if (vFp8 > 0.5 && vUv.x > 0.82 && vUv.y > 0.82 && (vUv.x + vUv.y) > 1.78) {
    col = mix(col, vec3(0.50, 0.57, 0.68), 0.35);
  }

  col = mix(col, borderCol, borderMask);
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
    // +Y top face: UV.x runs -X -> +X (0..1), UV.y runs +Z -> -Z (0..1, so row 0 at -Z is UV.y=1)
    {
      n: [0, 1, 0],
      corners: [
        [-0.5, 0.5, 0.5, 0, 0],
        [0.5, 0.5, 0.5, 1, 0],
        [0.5, 0.5, -0.5, 1, 1],
        [-0.5, 0.5, -0.5, 0, 1],
      ],
    },
    // -Y bottom face
    {
      n: [0, -1, 0],
      corners: [
        [-0.5, -0.5, -0.5, 0, 0],
        [0.5, -0.5, -0.5, 1, 0],
        [0.5, -0.5, 0.5, 1, 1],
        [-0.5, -0.5, 0.5, 0, 1],
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
 * Instanced WebGL2 pass for thin, dimension-aware flat matrix sheets.
 * Supports adaptive fwidth() grid-LOD, causal lower-triangle rendering, restrained matte edges,
 * and direct binding of a Float32 value texture in Increment 2.
 */
export class SheetsPass {
  private readonly gl: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private quadVbo!: WebGLBuffer;
  private quadIbo!: WebGLBuffer;
  private instanceVbo!: WebGLBuffer;
  private dummyValueTex!: WebGLTexture;
  private boundValueTex: WebGLTexture | null = null;
  private uViewProj!: WebGLUniformLocation | null;
  private uValueTex!: WebGLUniformLocation | null;
  private instanceCount = 0;

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
    const dummyTex = gl.createTexture();
    if (!vao || !quadVbo || !quadIbo || !instanceVbo || !dummyTex) {
      throw new Error('Failed to allocate SheetsPass WebGL resources');
    }

    this.vao = vao;
    this.quadVbo = quadVbo;
    this.quadIbo = quadIbo;
    this.instanceVbo = instanceVbo;
    this.dummyValueTex = dummyTex;

    // 1x1 neutral zero texture so uValueTex is always valid even before Increment 2 binds live data
    gl.bindTexture(gl.TEXTURE_2D, this.dummyValueTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
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
    bindInst(3, 3, 0);  // iCenter
    bindInst(4, 3, 3);  // iExtent
    bindInst(5, 2, 6);  // iGridDims
    bindInst(6, 1, 8);  // iSheetKind
    bindInst(7, 1, 9);  // iRole
    bindInst(8, 1, 10); // iFp8
    bindInst(9, 1, 11); // iHighlight
    bindInst(10, 1, 12); // iSeed
    bindInst(11, 4, 13); // iTexRect
    bindInst(12, 1, 17); // iValueScale

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Increment 2 hook: bind a live Float32/R32F or RGBA texture for real activation/weight values. */
  setValueTexture(tex: WebGLTexture | null): void {
    this.boundValueTex = tex;
  }

  upload(data: Float32Array, count: number): void {
    const gl = this.gl;
    this.instanceCount = count;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
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
    gl.deleteTexture(this.dummyValueTex);
  }
}
