/**
 * View 2 (`hypersphere`): nGPT Hyperspherical Residual & Latent Trajectory Inspector
 * (`src/dory/scene/hypersphereScene.ts`)
 *
 * Implements DORY3 Slice 3B (Locked Decisions `D1`, `D2`, `D4`, `D5`, `D7`; Guards `DR4`,
 * `DR7`, `DR8`, `DR9`, `DR20`, `DR21`; Review Finding 9):
 *
 *   - **Left Stage (3D Channel-Wise NLERP Residual Cutaway):**
 *       * Live `RG32F` sheets for `a = N_d(x_res)` (`[S, 8]`), `b = N_d(x_sub)` (`[S, 8]`),
 *         chord displacement `delta = b - a` (`[S, 8]`), dual raw `alpha` vs. effective
 *         `|alpha_hat| = |alpha * (0.05 / 0.0177)|` (`[1, 8]`, spotlighting planted `> 1`
 *         unclamped extrapolation channels `[6, 7]` at physical layer `16`), pre-norm blend
 *         `a + |alpha_hat| * (b - a)` (`[S, 8]`), and unit-sphere output `N_d(...)` (`[S, 8]`).
 *       * Explicit anti-SLERP terminology plaque (`DR7`, `dory-nlerp-residual`): channel-wise
 *         normalized linear interpolation (NLERP) — NEVER trigonometric SLERP.
 *
 *   - **Right Stage (Strictly Separated Exact Subspace vs. Exploratory 2D PCA + Counterfactual Coda):**
 *       * **(a) Exact 2D Spanning-Subspace Diagram (`DorySpanningSubspaceDiagramData`):**
 *         Exact orthonormal basis of `span(a, b)` in binary64 mapping `a -> (1, 0)` and
 *         `b -> (cos omega, sin omega)`, exact uniform-scalar NLERP curve over `[0, max(1, maxLambda)]`
 *         (marking `lambda > 1` extrapolation), orthogonal projection of actual per-channel
 *         pre-norm blend & output onto `span(a, b)`, and exact out-of-plane norm induced by
 *         channel-wise rates `|alpha_hat_c|`.
 *       * **(b) Exploratory 2D PCA Trajectory Projection (`DoryExploratoryPcaTrajectoryData`):**
 *         Deterministic 8x8 symmetric Jacobi PCA of actual 8D states `{x, u_1, s_1, ..., u_k, s_k}`
 *         projected orthogonally onto the orthonormal PCA plane `span(pc1, pc2)` (guaranteeing
 *         `hypot(x, y) <= exact8dNorm <= 1.0 + 1e-6`), explicitly badged
 *         `'EXPLORATORY 2D PCA PROJECTION OF 8D TRACE (PROJECTED NORMS <= 1; ALL NUMERIC METRICS COMPUTED STRICTLY IN 8D)'`.
 *       * **Local `120-deg` Anchor-Map Jacobian Caveat (Audit §6.2.1, Finding 9):**
 *         Reports exact 8D `anchorMapJacobianNorm` (`eps = 1e-6`, capped at `1000` at antipodal
 *         `180 deg`) explicitly framed as a LOCAL boundary-map property ONLY — never claiming
 *         global recurrent-core contraction or convergence.
 *       * **Read-Only Diagnostic Counterfactual Coda (`DoryCounterfactualCodaPreview`):**
 *         Optional intermediate-loop Coda logit-lens evaluated via `evaluateCounterfactualIntermediateCoda`
 *         against committed historical Deep Coda KV (`writeKv: false`, ZERO KV/session mutation),
 *         badged `'DIAGNOSTIC COUNTERFACTUAL · CODA NOT RUN AT INTERMEDIATE LOOPS'`.
 *
 *   - **Atlas & Boundary Compliance:**
 *       * Compact `64 x 64` (`32.00 KiB`, expanding to `64 x 128` = `64.00 KiB` only for `k > 27`)
 *         `RG32F` atlas strictly `<= 128 KiB <= 512 KiB`, `0` overlaps, `0` out-of-bounds, `DR20`
 *         compliant, `DR21` pick encoding (zero edits to `src/render/pick.ts`), zero ambient RNG,
 *         and strict `D4` architectural boundary isolation (zero Gate 4b/4c or sibling-app imports).
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins, or ambient RNG.
 */

import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_EMERALD,
  RIBBON_KIND_ARROW_POS_X,
  type DimensionCaliper,
  type LandmarkAnchor,
  type Vec3,
} from '../../render/primitives';
import {
  FRAME_XZ_AXES,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import { fmtSigned } from '../../shared/format';
import {
  computeSheetCellWorldPos,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../../shared/overlayLayout';
import {
  anchorMapJacobianNorm,
  argmaxIndex,
  cosineSimilarity8d,
  l2Distance,
  l2NormVector,
  softmaxFloat64,
  vectorL2Norm,
} from '../ngptMath';
import { decodeDoryToken, getDoryPhysicalLayerSpec, getDoryStage } from '../spec';
import {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  DORY_PLANTED_EXTRAPOLATION_SPEC,
  type DoryCounterfactualCodaResult,
  type DoryForwardTrace,
  type DoryLoopTrajectoryMetrics,
  type DoryNlerpTrace,
  type DoryReconstructedStepResult,
  type DorySession,
} from '../types';
import {
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_FLOW_COLORS,
  DoryAtlasPacker,
  DoryFlatFlowBufferBuilder,
  DoryLineBufferBuilder,
  DoryRibbonBufferBuilder,
  DorySheetBufferBuilder,
  DorySolidBufferBuilder,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  computeBoundsFromPoints,
  encodeDoryPickMetadata,
  mergeAabbs,
  packPickBoxFromAabb,
  packPickBoxFromFrame,
  resolveDorySessionContext,
  validateSceneGeometryFinite,
} from './provenance';
import {
  DORY_SPATIAL_VIEW_MODES,
  type DoryBadgeTone,
  type DoryCounterfactualCodaPreview,
  type DoryCounterfactualTopToken,
  type DoryExploratoryPcaTrajectoryData,
  type DoryHypersphereSceneOptions,
  type DoryHypersphereViewMetadata,
  type DoryOverlayAnchor,
  type DoryPcaTrajectoryPoint,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneCallout,
  type DorySpanningSubspaceCurvePoint,
  type DorySpanningSubspaceDiagramData,
  type DorySpatialScene,
} from './types';

// ============================================================================
// VIEW 2 CONSTANTS & ATLAS BUDGETS
// ============================================================================

export const DORY_HYPERSPHERE_ATLAS_WIDTH = 64 as const;
export const DORY_HYPERSPHERE_ATLAS_DEFAULT_HEIGHT = 64 as const;
export const DORY_HYPERSPHERE_ATLAS_EXTENDED_HEIGHT = 128 as const;
export const DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES = 131_072 as const; // 128.00 KiB <= 512 KiB

const VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[1]!;
const COUNTERFACTUAL_TOP_K = 8;
const SPANNING_INTERP_SAMPLES = 24;
const SPANNING_EXTRAP_SAMPLES = 12;
const UNIT_CIRCLE_SEGMENTS = 48;
const JACOBI_MAX_SWEEPS = 40;
const JACOBI_OFFDIAG_TOL = 1e-14;

// ============================================================================
// PURE NUMERICAL HELPERS (BINARY64 ACCUMULATION, ZERO AMBIENT RNG)
// ============================================================================

function clampInt(val: number, lo: number, hi: number): number {
  if (!Number.isFinite(val)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

function clampUnitCos(v: number): number {
  return v > 1 ? 1 : v < -1 ? -1 : v;
}

function sliceCopyFloat32(src: Float32Array, offset: number, length: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = src[offset + i]!;
  }
  return out;
}

function dotFloat64(a: ArrayLike<number>, b: ArrayLike<number>, dim: number): number {
  let acc = 0;
  for (let i = 0; i < dim; i += 1) {
    acc += a[i]! * b[i]!;
  }
  return acc;
}

function normFloat64(v: ArrayLike<number>, dim: number): number {
  return Math.sqrt(dotFloat64(v, v, dim));
}

function normalizeFloat64(v: Float64Array): Float64Array {
  const n = normFloat64(v, v.length);
  const out = new Float64Array(v.length);
  if (n > 1e-15) {
    const inv = 1 / n;
    for (let i = 0; i < v.length; i += 1) {
      out[i] = v[i]! * inv;
    }
  } else if (v.length > 0) {
    out[0] = 1;
  }
  return out;
}

/**
 * Finds a deterministic unit vector strictly orthogonal to unit vector `u` in `R^dim`.
 */
function deterministicOrthogonalUnitVector(u: Float64Array): Float64Array {
  const dim = u.length;
  let bestAxis = 0;
  let minAbsProj = Infinity;
  for (let c = 0; c < dim; c += 1) {
    const absProj = Math.abs(u[c]!);
    if (absProj < minAbsProj) {
      minAbsProj = absProj;
      bestAxis = c;
    }
  }
  const cand = new Float64Array(dim);
  for (let c = 0; c < dim; c += 1) {
    cand[c] = (c === bestAxis ? 1 : 0) - u[bestAxis]! * u[c]!;
  }
  return normalizeFloat64(cand);
}

/**
 * Exact binary64 KL divergence `D_KL(p || q) = sum_i p_i * ln(p_i / q_i)` in nats.
 * Returns finite non-negative number (`0` when `p` and `q` match).
 */
export function computeKlDivergenceFloat64(
  p: ArrayLike<number>,
  q: ArrayLike<number>,
): number {
  const n = Math.min(p.length, q.length);
  let kl = 0;
  for (let i = 0; i < n; i += 1) {
    const pi = p[i]!;
    if (!Number.isFinite(pi) || pi <= 0) continue;
    const qiRaw = q[i]!;
    const qi = Number.isFinite(qiRaw) && qiRaw > 1e-300 ? qiRaw : 1e-300;
    kl += pi * Math.log(pi / qi);
  }
  if (!Number.isFinite(kl) || kl <= 0) {
    return 0;
  }
  return kl;
}

// ============================================================================
// COMPONENT A: EXACT 2D SPANNING-SUBSPACE NLERP DIAGRAM (FINDING 9, DR7)
// ============================================================================

export interface ComputeSpanningSubspaceInput {
  readonly physicalLayerIdx: number;
  readonly loopIteration: number;
  readonly tokenOffset: number;
  readonly absTokenPos: number;
  readonly rawAlpha: Float32Array;
  readonly nlerp: DoryNlerpTrace;
  readonly eps?: number;
}

/**
 * Constructs View 2 Diagram (a): the **Exact 2D Spanning-Subspace Diagram** `span(a, b)` for one
 * selected channel-wise nGPT NLERP residual update (Finding 9, `DR7`).
 *
 * Strictly distinct from the Exploratory 2D PCA trajectory projection:
 *   - Constructs an exact binary64 orthonormal basis `(e1, e2)` for `span(a, b)` in `R^8`.
 *   - Maps `a = N(x_res)` to `(1, 0)` and `b = N(x_sub)` to `(cos(omega), sin(omega))`.
 *   - Evaluates the exact uniform-scalar NLERP curve `N(a + lambda*(b - a))` across
 *     `lambda in [0, max(1, maxEffectiveAlpha)]` (tagging `lambda > 1` as extrapolation).
 *   - Orthogonally projects actual per-channel `blendedPreNorm` and `output` onto `span(a, b)`
 *     and computes the exact out-of-plane residual `||x_next - Pi_{span(a,b)}(x_next)||_2`.
 */
export function computeSpanningSubspaceDiagram(
  input: ComputeSpanningSubspaceInput,
): DorySpanningSubspaceDiagramData {
  const {
    physicalLayerIdx,
    loopIteration,
    tokenOffset,
    absTokenPos,
    rawAlpha,
    nlerp,
  } = input;
  const eps = input.eps ?? DORY_MINI_CONFIG.l2NormEps;
  const d = nlerp.hiddenSize;
  const rowOff = tokenOffset * d;
  const spec = getDoryPhysicalLayerSpec(physicalLayerIdx);

  const aVec = sliceCopyFloat32(nlerp.aNorm, rowOff, d);
  const bVec = sliceCopyFloat32(nlerp.bNorm, rowOff, d);
  const blendedVec = sliceCopyFloat32(nlerp.blendedPreNorm, rowOff, d);
  const outVec = sliceCopyFloat32(nlerp.output, rowOff, d);
  const rawAlphaCopy = sliceCopyFloat32(rawAlpha, 0, d);
  const effAlphaCopy = sliceCopyFloat32(nlerp.effectiveAlpha, 0, d);

  // Build exact binary64 orthonormal basis (e1, e2) of span(a, b)
  const a64 = new Float64Array(d);
  const b64 = new Float64Array(d);
  for (let c = 0; c < d; c += 1) {
    a64[c] = aVec[c]!;
    b64[c] = bVec[c]!;
  }
  const e1 = normalizeFloat64(a64);
  const bDotE1 = dotFloat64(b64, e1, d);
  const bPerp = new Float64Array(d);
  for (let c = 0; c < d; c += 1) {
    bPerp[c] = b64[c]! - bDotE1 * e1[c]!;
  }
  const e2 =
    normFloat64(bPerp, d) > 1e-12
      ? normalizeFloat64(bPerp)
      : deterministicOrthogonalUnitVector(e1);

  const cosineAB = clampUnitCos(cosineSimilarity8d(aVec, bVec, 0, 0, d));
  const omegaRadians = Math.acos(cosineAB);
  const omegaDegrees = (omegaRadians * 180) / Math.PI;
  const a2d: readonly [number, number] = Object.freeze([1, 0] as const);
  const b2d: readonly [number, number] = Object.freeze([
    Math.cos(omegaRadians),
    Math.sin(omegaRadians),
  ] as const);

  // Per-channel effective scale stats & unclamped > 1 extrapolation detection (DR7)
  let minEffectiveAlpha = Infinity;
  let maxEffectiveAlpha = 0;
  let sumEffectiveAlpha = 0;
  const extrapolatedChannels: number[] = [];

  for (let c = 0; c < d; c += 1) {
    const lam = effAlphaCopy[c]!;
    if (lam < minEffectiveAlpha) minEffectiveAlpha = lam;
    if (lam > maxEffectiveAlpha) maxEffectiveAlpha = lam;
    sumEffectiveAlpha += lam;
    if (lam > 1.0) {
      extrapolatedChannels.push(c);
    }
  }
  if (!Number.isFinite(minEffectiveAlpha)) {
    minEffectiveAlpha = 0;
  }
  const meanEffectiveAlpha = d > 0 ? sumEffectiveAlpha / d : 0;

  // Sample exact uniform-scalar NLERP curve N_eps(a2d + lambda * (b2d - a2d))
  const curve: DorySpanningSubspaceCurvePoint[] = [];
  const pushCurvePoint = (lambda: number): void => {
    const vx = a2d[0] + lambda * (b2d[0] - a2d[0]);
    const vy = a2d[1] + lambda * (b2d[1] - a2d[1]);
    const inv = 1 / Math.sqrt(vx * vx + vy * vy + eps);
    curve.push(
      Object.freeze({
        lambda,
        x: vx * inv,
        y: vy * inv,
        isExtrapolation: lambda > 1.0,
      }),
    );
  };

  for (let i = 0; i <= SPANNING_INTERP_SAMPLES; i += 1) {
    pushCurvePoint(i / SPANNING_INTERP_SAMPLES);
  }
  if (maxEffectiveAlpha > 1.0) {
    const extrapSpan = maxEffectiveAlpha - 1.0;
    for (let i = 1; i <= SPANNING_EXTRAP_SAMPLES; i += 1) {
      pushCurvePoint(1.0 + (extrapSpan * i) / SPANNING_EXTRAP_SAMPLES);
    }
  }

  // Orthogonal projections of actual 8D per-channel blended pre-norm and final NLERP output
  const blendedX = dotFloat64(blendedVec, e1, d);
  const blendedY = dotFloat64(blendedVec, e2, d);
  const outX = dotFloat64(outVec, e1, d);
  const outY = dotFloat64(outVec, e2, d);

  let perpSumSq = 0;
  for (let c = 0; c < d; c += 1) {
    const perp = outVec[c]! - (outX * e1[c]! + outY * e2[c]!);
    perpSumSq += perp * perp;
  }
  const outOfPlaneNorm = Math.sqrt(perpSumSq);

  const loopSuffix = spec.stage === 'recurrent' ? ` · Loop t=${loopIteration}` : ` · ${spec.stage}`;
  const diagramTitle = `Exact 2D Spanning Subspace span(a, b) — Phys L${physicalLayerIdx} (${spec.transformersPairBadge}${loopSuffix})`;
  const extrapSummary =
    extrapolatedChannels.length > 0
      ? ` Unclamped |alpha_hat| > 1 extrapolation on ch [${extrapolatedChannels.join(', ')}] (max ${maxEffectiveAlpha.toFixed(2)}).`
      : ` All 8 channels within [0, 1] (max ${maxEffectiveAlpha.toFixed(2)}).`;
  const antiSlerpCaption = `${DORY_ANTI_SLERP_PLAQUE} (${DORY_NLERP_MARKER}): chord projection N(a + |alpha_hat|*(b-a)) with unclamped |alpha_hat| >= 0.${extrapSummary}`;

  return Object.freeze({
    diagramTitle,
    antiSlerpCaption,
    physicalLayerIdx,
    pairedLayerIdx: spec.pairedLayerIdx,
    loopIteration,
    tokenOffset,
    absTokenPos,
    a2d,
    b2d,
    omegaRadians,
    omegaDegrees,
    cosineAB,
    uniformAlphaCurve: Object.freeze(curve),
    projectedBlendedPreNorm2d: Object.freeze([blendedX, blendedY] as const),
    projectedOutput2d: Object.freeze([outX, outY] as const),
    outOfPlaneNorm,
    rawAlpha: rawAlphaCopy,
    effectiveAlpha: effAlphaCopy,
    minEffectiveAlpha,
    meanEffectiveAlpha,
    maxEffectiveAlpha,
    extrapolatedChannels: Object.freeze(extrapolatedChannels),
  });
}

// ============================================================================
// COMPONENT B: EXPLORATORY 2D PCA TRAJECTORY PROJECTION (8D METRICS ONLY)
// ============================================================================

/**
 * Deterministic cyclic Jacobi eigendecomposition of an `n x n` real symmetric matrix `sym`
 * (stored row-major in `Float64Array(n * n)`). Returns eigenvalues and column-major orthonormal
 * eigenvectors (each column `j` is `vec[row * n + j]`).
 */
function jacobiSymmetricEigen(
  sym: Float64Array,
  n: number,
): { readonly eigenvalues: Float64Array; readonly eigenvectors: Float64Array } {
  const A = new Float64Array(sym);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    V[i * n + i] = 1.0;
  }

  for (let sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep += 1) {
    let maxOffDiag = 0;
    for (let p = 0; p < n - 1; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = Math.abs(A[p * n + q]!);
        if (apq > maxOffDiag) maxOffDiag = apq;
      }
    }
    if (maxOffDiag <= JACOBI_OFFDIAG_TOL) {
      break;
    }

    for (let p = 0; p < n - 1; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = A[p * n + q]!;
        if (Math.abs(apq) <= 1e-15) continue;

        const app = A[p * n + p]!;
        const aqq = A[q * n + q]!;
        const tau = (aqq - app) / (2 * apq);
        const t =
          tau >= 0
            ? 1 / (tau + Math.hypot(1, tau))
            : -1 / (-tau + Math.hypot(1, tau));
        const c = 1 / Math.hypot(1, t);
        const s = t * c;

        A[p * n + p] = app - t * apq;
        A[q * n + q] = aqq + t * apq;
        A[p * n + q] = 0;
        A[q * n + p] = 0;

        for (let r = 0; r < n; r += 1) {
          if (r === p || r === q) continue;
          const arp = A[r * n + p]!;
          const arq = A[r * n + q]!;
          const newArp = c * arp - s * arq;
          const newArq = s * arp + c * arq;
          A[r * n + p] = newArp;
          A[p * n + r] = newArp;
          A[r * n + q] = newArq;
          A[q * n + r] = newArq;
        }

        for (let r = 0; r < n; r += 1) {
          const vrp = V[r * n + p]!;
          const vrq = V[r * n + q]!;
          V[r * n + p] = c * vrp - s * vrq;
          V[r * n + q] = s * vrp + c * vrq;
        }
      }
    }
  }

  const eigenvalues = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const val = A[i * n + i]!;
    eigenvalues[i] = val > 0 ? val : 0;
  }

  return { eigenvalues, eigenvectors: V };
}

function canonicalizeAxisSign(axis: Float64Array, reference: Float32Array): void {
  const proj = dotFloat64(reference, axis, axis.length);
  if (proj < -1e-12) {
    for (let c = 0; c < axis.length; c += 1) axis[c] = -axis[c]!;
    return;
  }
  if (Math.abs(proj) <= 1e-12) {
    for (let c = 0; c < axis.length; c += 1) {
      if (Math.abs(axis[c]!) > 1e-8) {
        if (axis[c]! < 0) {
          for (let j = 0; j < axis.length; j += 1) axis[j] = -axis[j]!;
        }
        return;
      }
    }
  }
}

export interface ComputeExploratoryPcaInput {
  readonly xAnchor: Float32Array;
  readonly loopEntryStates: readonly Float32Array[];
  readonly loopExitStates: readonly Float32Array[];
}

/**
 * Constructs View 2 Diagram (b): the **Exploratory 2D PCA Trajectory Projection** of actual 8D
 * latent states `{x, u_1, s_1, ..., u_k, s_k}` (Audit §12.1 View 2, Finding 9).
 *
 * Mathematical guarantees:
 *   - Computes centered covariance eigenvalues/eigenvectors in binary64 via cyclic Jacobi to find
 *     the orthonormal 2D plane `(pc1, pc2)` of maximum trajectory variance.
 *   - Projects unit-sphere states `v in S^7` orthogonally onto `span(pc1, pc2)` via
 *     `(dot(v, pc1), dot(v, pc2))`, preserving exact pairwise centered-PCA relative vectors while
 *     guaranteeing `projectedNorm = hypot(x, y) <= exact8dNorm <= 1.0 + 1e-6` (points sit inside
 *     the unit disk `D^2`, visually disclosing 8D->2D norm collapse).
 *   - ALL numeric metrics (`exact8dNorm`, `exact8dAnchorCosine`, `exact8dAnchorAngleDeg`,
 *     `exact8dConsecutiveCosine`, `exact8dConsecutiveDisplacement`) are computed STRICTLY in
 *     original 8D.
 */
export function computeExploratoryPcaTrajectory(
  input: ComputeExploratoryPcaInput,
): DoryExploratoryPcaTrajectoryData {
  const { xAnchor, loopEntryStates, loopExitStates } = input;
  const d = xAnchor.length;
  const kLoops = Math.min(loopEntryStates.length, loopExitStates.length);

  interface RawTrajEntry {
    readonly id: string;
    readonly label: string;
    readonly kind: 'anchor' | 'loop_entry' | 'loop_exit';
    readonly loopIteration: number;
    readonly vec: Float32Array;
    readonly prevVec: Float32Array | null;
  }

  const entries: RawTrajEntry[] = [
    {
      id: 'x_anchor',
      label: 'x (Anchor s_0)',
      kind: 'anchor',
      loopIteration: 0,
      vec: xAnchor,
      prevVec: null,
    },
  ];

  for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
    const t = loopIdx + 1;
    const prevExit = loopIdx === 0 ? xAnchor : loopExitStates[loopIdx - 1]!;
    const uVec = loopEntryStates[loopIdx]!;
    const sVec = loopExitStates[loopIdx]!;
    entries.push({
      id: `u_${t}`,
      label: t === 1 ? 'u_1 = N(2x)' : `u_${t} = N(s_${t - 1}+x)`,
      kind: 'loop_entry',
      loopIteration: t,
      vec: uVec,
      prevVec: prevExit,
    });
    entries.push({
      id: `s_${t}`,
      label: `s_${t} = N(F(u_${t}))`,
      kind: 'loop_exit',
      loopIteration: t,
      vec: sVec,
      prevVec: prevExit,
    });
  }

  const count = entries.length;
  const mean64 = new Float64Array(d);
  for (let i = 0; i < count; i += 1) {
    const v = entries[i]!.vec;
    for (let c = 0; c < d; c += 1) {
      mean64[c] = mean64[c]! + v[c]!;
    }
  }
  const invCount = count > 0 ? 1 / count : 0;
  for (let c = 0; c < d; c += 1) {
    mean64[c] = mean64[c]! * invCount;
  }

  const cov = new Float64Array(d * d);
  for (let i = 0; i < count; i += 1) {
    const v = entries[i]!.vec;
    for (let r = 0; r < d; r += 1) {
      const dr = v[r]! - mean64[r]!;
      for (let c = r; c < d; c += 1) {
        const val = cov[r * d + c]! + dr * (v[c]! - mean64[c]!);
        cov[r * d + c] = val;
        if (c !== r) cov[c * d + r] = val;
      }
    }
  }
  let totalVar = 0;
  for (let r = 0; r < d; r += 1) {
    for (let c = 0; c < d; c += 1) {
      cov[r * d + c] = cov[r * d + c]! * invCount;
    }
    totalVar += Math.max(0, cov[r * d + r]!);
  }

  const { eigenvalues, eigenvectors } = jacobiSymmetricEigen(cov, d);
  const order = Array.from({ length: d }, (_, idx) => idx);
  order.sort((a, b) => {
    const diff = eigenvalues[b]! - eigenvalues[a]!;
    if (Math.abs(diff) > 1e-15) return diff;
    return a - b;
  });

  const eig0 = eigenvalues[order[0]!]!;
  const eig1 = d > 1 ? eigenvalues[order[1]!]! : 0;

  let pc1: Float64Array = new Float64Array(d);
  let pc2: Float64Array = new Float64Array(d);
  for (let r = 0; r < d; r += 1) {
    pc1[r] = eigenvectors[r * d + order[0]!]!;
    if (d > 1) {
      pc2[r] = eigenvectors[r * d + order[1]!]!;
    }
  }
  pc1 = normalizeFloat64(pc1);

  // If the trajectory is rank-1 (e.g. k=1 where u_1 ≈ x), pick the orthogonal component of xAnchor
  // as the zero-eigenvalue PC2 so span(pc1, pc2) preserves the exact radial plane of {x, s_1}.
  if (eig1 <= 1e-12 * Math.max(1, eig0)) {
    const xDotPc1 = dotFloat64(xAnchor, pc1, d);
    const xPerp = new Float64Array(d);
    for (let c = 0; c < d; c += 1) {
      xPerp[c] = xAnchor[c]! - xDotPc1 * pc1[c]!;
    }
    if (normFloat64(xPerp, d) > 1e-8) {
      pc2 = normalizeFloat64(xPerp);
    } else {
      pc2 = deterministicOrthogonalUnitVector(pc1);
    }
  } else {
    // Re-orthogonalize pc2 against pc1 for exact machine-precision orthonormality
    const proj12 = dotFloat64(pc2, pc1, d);
    for (let c = 0; c < d; c += 1) {
      pc2[c] = pc2[c]! - proj12 * pc1[c]!;
    }
    pc2 = normFloat64(pc2, d) > 1e-12 ? normalizeFloat64(pc2) : deterministicOrthogonalUnitVector(pc1);
  }

  const lastExitRef = kLoops > 0 ? loopExitStates[kLoops - 1]! : xAnchor;
  canonicalizeAxisSign(pc1, xAnchor);
  canonicalizeAxisSign(pc2, lastExitRef);

  let ratio0 = 0;
  let ratio1 = 0;
  if (totalVar > 1e-15) {
    ratio0 = Math.max(0, Math.min(1, eig0 / totalVar));
    ratio1 = Math.max(0, Math.min(1 - ratio0, eig1 / totalVar));
  }

  const points: DoryPcaTrajectoryPoint[] = [];
  let allProjectedNormsLeqOne = true;

  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i]!;
    const x = dotFloat64(entry.vec, pc1, d);
    const y = dotFloat64(entry.vec, pc2, d);
    const projectedNorm = Math.hypot(x, y);
    if (projectedNorm > 1.0 + 1e-6) {
      allProjectedNormsLeqOne = false;
    }

    const exact8dNorm = vectorL2Norm(entry.vec, 0, d);
    const exact8dAnchorCosine = clampUnitCos(cosineSimilarity8d(entry.vec, xAnchor, 0, 0, d));
    const exact8dAnchorAngleDeg = (Math.acos(exact8dAnchorCosine) * 180) / Math.PI;
    const exact8dConsecutiveCosine =
      entry.prevVec !== null
        ? clampUnitCos(cosineSimilarity8d(entry.vec, entry.prevVec, 0, 0, d))
        : null;
    const exact8dConsecutiveDisplacement =
      entry.prevVec !== null ? l2Distance(entry.vec, entry.prevVec, 0, 0, d) : null;

    points.push(
      Object.freeze({
        id: entry.id,
        label: entry.label,
        kind: entry.kind,
        loopIteration: entry.loopIteration,
        x,
        y,
        projectedNorm,
        exact8dNorm,
        exact8dAnchorCosine,
        exact8dAnchorAngleDeg,
        exact8dConsecutiveCosine,
        exact8dConsecutiveDisplacement,
      }),
    );
  }

  return Object.freeze({
    disclosureBadge: DORY_EXPLORATORY_PCA_BADGE,
    metricsComputedIn8d: true,
    allProjectedNormsLeqOne,
    explainedVarianceRatio: Object.freeze([ratio0, ratio1] as const),
    mean8d: Float32Array.from(mean64),
    pc1Axis8d: Float32Array.from(pc1),
    pc2Axis8d: Float32Array.from(pc2),
    points: Object.freeze(points),
  });
}

// ============================================================================
// COMPONENT C: READ-ONLY DIAGNOSTIC COUNTERFACTUAL CODA PREVIEW (ZERO KV MUTATION)
// ============================================================================

/**
 * Formats a read-only `DoryCounterfactualCodaResult` (produced with `writeKv: false` against
 * committed historical Deep Coda KV) into `DoryCounterfactualCodaPreview` with binary64
 * `D_KL(p_final || p_counterfactual)` computed at matched `temperature = 1.0`.
 */
export function computeCounterfactualCodaPreview(
  rawResult: DoryCounterfactualCodaResult,
  finalLoopScaledLogits: Float32Array,
  topK: number = COUNTERFACTUAL_TOP_K,
): DoryCounterfactualCodaPreview {
  const vocabSize = rawResult.probs.length;
  const finalLoopProbsT1 = softmaxFloat64(finalLoopScaledLogits, 0, vocabSize, 1.0);
  const klFromFinalLoopNats = computeKlDivergenceFloat64(finalLoopProbsT1, rawResult.probs);

  const argMaxTokenId = argmaxIndex(rawResult.scaledLogits, 0, vocabSize);
  const finalLoopArgMaxTokenId = argmaxIndex(finalLoopScaledLogits, 0, vocabSize);
  const argMaxChanged = argMaxTokenId !== finalLoopArgMaxTokenId;

  const indices = Array.from({ length: vocabSize }, (_, idx) => idx);
  indices.sort((a, b) => {
    const diff = rawResult.probs[b]! - rawResult.probs[a]!;
    if (diff !== 0) return diff;
    return a - b;
  });

  const k = Math.max(1, Math.min(vocabSize, Math.trunc(topK)));
  const topTokens: DoryCounterfactualTopToken[] = [];
  for (let i = 0; i < k; i += 1) {
    const tokenId = indices[i]!;
    topTokens.push(
      Object.freeze({
        tokenId,
        tokenText: decodeDoryToken(tokenId),
        prob: rawResult.probs[tokenId]!,
        scaledLogit: rawResult.scaledLogits[tokenId]!,
      }),
    );
  }

  return Object.freeze({
    badge: DORY_COUNTERFACTUAL_CODA_BADGE,
    recordId: rawResult.recordId,
    loopIteration: rawResult.loopIteration,
    kLoops: rawResult.kLoops,
    absTokenPos: rawResult.absTokenPos,
    readOnlyZeroKvMutation: true,
    topTokens: Object.freeze(topTokens),
    klFromFinalLoopNats,
    argMaxTokenId,
    finalLoopArgMaxTokenId,
    argMaxChanged,
    rawResult,
  });
}

// ============================================================================
// INTERNAL TRACE EXTRACTION & HOVER POPOVER HELPERS
// ============================================================================

function resolveContext(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  recordId?: string,
): DoryResolvedSessionContext {
  if ('resolveForwardTrace' in sessionOrCtx) {
    if (recordId !== undefined && recordId !== sessionOrCtx.record.recordId) {
      return resolveDorySessionContext(sessionOrCtx.session, recordId);
    }
    return sessionOrCtx;
  }
  return resolveDorySessionContext(sessionOrCtx, recordId);
}

function resolveTraceTokens(
  ctx: DoryResolvedSessionContext,
  reconstructed: DoryReconstructedStepResult,
  trace: DoryForwardTrace,
): readonly number[] {
  const tokens: number[] = new Array<number>(trace.seqLen);
  for (let i = 0; i < trace.seqLen; i += 1) {
    let tok: number | undefined;
    if (reconstructed.forwardResult !== null && i < reconstructed.forwardResult.tokens.length) {
      tok = reconstructed.forwardResult.tokens[i];
    } else if (
      reconstructed.specPrefillResult !== null &&
      i < reconstructed.specPrefillResult.promptTokens.length
    ) {
      tok = reconstructed.specPrefillResult.promptTokens[i];
    } else if (
      reconstructed.specCycleResult !== null &&
      i < reconstructed.specCycleResult.verifyPositions.length
    ) {
      tok = reconstructed.specCycleResult.verifyPositions[i]!.inputTokenId;
    } else {
      const absPos = trace.startPos + i;
      tok =
        absPos < ctx.record.committedSequence.length
          ? ctx.record.committedSequence[absPos]
          : ctx.record.inputTokens[i];
    }
    tokens[i] = tok ?? 0;
  }
  return tokens;
}

function resolveTokenLoopStatesAndMetrics(
  ctx: DoryResolvedSessionContext,
  reconstructed: DoryReconstructedStepResult,
  trace: DoryForwardTrace,
  tokenOffset: number,
): {
  readonly xAnchor: Float32Array;
  readonly loopEntryStates: readonly Float32Array[];
  readonly loopExitStates: readonly Float32Array[];
  readonly loopMetrics8d: readonly DoryLoopTrajectoryMetrics[];
} {
  const d = ctx.params.config.hiddenSize;
  const eps = ctx.params.config.l2NormEps;
  const kLoops = ctx.kLoops;
  const absTokenPos = trace.startPos + tokenOffset;
  const off = tokenOffset * d;

  let xAnchor: Float32Array;
  const loopEntryStates: Float32Array[] = [];
  const loopExitStates: Float32Array[] = [];

  if (reconstructed.forwardResult !== null) {
    xAnchor = sliceCopyFloat32(reconstructed.forwardResult.xAnchor, off, d);
    for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
      loopEntryStates.push(
        sliceCopyFloat32(reconstructed.forwardResult.loopEntryStates[loopIdx]!, off, d),
      );
      loopExitStates.push(
        sliceCopyFloat32(reconstructed.forwardResult.loopExitStates[loopIdx]!, off, d),
      );
    }
  } else if (reconstructed.specPrefillResult !== null) {
    xAnchor = sliceCopyFloat32(reconstructed.specPrefillResult.xAnchor, off, d);
    for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
      loopEntryStates.push(
        sliceCopyFloat32(reconstructed.specPrefillResult.loopEntryStates[loopIdx]!, off, d),
      );
      loopExitStates.push(
        sliceCopyFloat32(reconstructed.specPrefillResult.loopExitStates[loopIdx]!, off, d),
      );
    }
  } else if (
    reconstructed.specCycleResult !== null &&
    absTokenPos === reconstructed.specCycleResult.endCommittedLen - 1
  ) {
    xAnchor = sliceCopyFloat32(reconstructed.specCycleResult.lastCommittedXAnchor, 0, d);
    for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
      loopEntryStates.push(
        sliceCopyFloat32(reconstructed.specCycleResult.lastCommittedLoopEntryStates[loopIdx]!, 0, d),
      );
      loopExitStates.push(
        sliceCopyFloat32(reconstructed.specCycleResult.lastCommittedLoopExitStates[loopIdx]!, 0, d),
      );
    }
  } else {
    // Reconstruct exact loop boundary states directly from trace layers 15 (Prelude exit) & 55 (Core exit)
    const preludeLast = ctx.resolveSublayerRecord(ctx.params.config.nInputLayers - 1, 0);
    xAnchor = sliceCopyFloat32(preludeLast.trace.nlerp.output, off, d);
    const coreLastPhys = ctx.params.config.nInputLayers + ctx.params.config.nRecurrentLayers - 1;
    let prev = xAnchor;
    for (let t = 1; t <= kLoops; t += 1) {
      const sumBuf = new Float32Array(d);
      for (let c = 0; c < d; c += 1) {
        sumBuf[c] = prev[c]! + xAnchor[c]!;
      }
      const uVec = l2NormVector(sumBuf, eps);
      const coreExitRec = ctx.resolveSublayerRecord(coreLastPhys, t);
      const sVec = l2NormVector(sliceCopyFloat32(coreExitRec.trace.nlerp.output, off, d), eps);
      loopEntryStates.push(uVec);
      loopExitStates.push(sVec);
      prev = sVec;
    }
  }

  // Prefer exact precomputed trace loop metrics for tokenOffset; fall back to exact 8D math if absent
  const traceTokenMetrics = trace.loopMetrics
    .filter((m) => m.tokenOffset === tokenOffset)
    .sort((a, b) => a.loopIteration - b.loopIteration);

  let loopMetrics8d: readonly DoryLoopTrajectoryMetrics[];
  if (traceTokenMetrics.length === kLoops) {
    loopMetrics8d = Object.freeze(
      traceTokenMetrics.map((m) =>
        Object.freeze({
          ...m,
          entryAnchorJacobian: Object.freeze({ ...m.entryAnchorJacobian }),
        }),
      ),
    );
  } else {
    const built: DoryLoopTrajectoryMetrics[] = [];
    for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
      const prev = loopIdx === 0 ? xAnchor : loopExitStates[loopIdx - 1]!;
      const uVec = loopEntryStates[loopIdx]!;
      const sVec = loopExitStates[loopIdx]!;
      const prevAnchorCos = clampUnitCos(cosineSimilarity8d(prev, xAnchor, 0, 0, d));
      const entryAngleRad = Math.acos(prevAnchorCos);
      built.push(
        Object.freeze({
          loopIteration: loopIdx + 1,
          tokenOffset,
          absTokenPos,
          uNorm: vectorL2Norm(uVec, 0, d),
          sNorm: vectorL2Norm(sVec, 0, d),
          consecutiveCosine: clampUnitCos(cosineSimilarity8d(sVec, prev, 0, 0, d)),
          consecutiveDisplacement: l2Distance(sVec, prev, 0, 0, d),
          anchorCosine: clampUnitCos(cosineSimilarity8d(sVec, xAnchor, 0, 0, d)),
          entryAnchorJacobian: anchorMapJacobianNorm(entryAngleRad, eps),
        }),
      );
    }
    loopMetrics8d = Object.freeze(built);
  }

  return {
    xAnchor,
    loopEntryStates: Object.freeze(loopEntryStates),
    loopExitStates: Object.freeze(loopExitStates),
    loopMetrics8d,
  };
}

interface PlacedSheetRecord {
  readonly partId: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly frame: SheetFrame3D;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly values: Float32Array;
  readonly grid: ActiveSheetCellGrid;
}

function buildHoveredCellPopover(
  hovered: DoryPickIdentity | null | undefined,
  sheetsByPartId: ReadonlyMap<string, PlacedSheetRecord>,
  spanning: DorySpanningSubspaceDiagramData,
  exploratoryPca: DoryExploratoryPcaTrajectoryData,
  counterfactual: DoryCounterfactualCodaPreview | null,
  tokenLabels: readonly string[],
  counterfactualTokenIds: readonly number[],
  finalProbsT1: Float64Array,
): CellPopoverSpec | null {
  if (!hovered || !hovered.partId) return null;
  const rec = sheetsByPartId.get(hovered.partId);
  if (!rec) return null;

  const row = clampInt(hovered.row ?? 0, 0, rec.gridRows - 1);
  const col = clampInt(hovered.col ?? 0, 0, rec.gridCols - 1);
  const world = computeSheetCellWorldPos(rec.grid, col, row, 0.5);
  const val = rec.values[row * rec.gridCols + col] ?? 0;

  if (rec.partId === 'nlerp_raw_alpha') {
    const eff = spanning.effectiveAlpha[col] ?? 0;
    const isExtrap = eff > 1.0;
    return Object.freeze({
      world,
      tensorLabel: `Raw alpha[${col}] (Phys L${spanning.physicalLayerIdx})`,
      coordLabel: `ch c=${col} · sigma_0=0.0177`,
      badge: isExtrap ? 'UNCLAMPED |alpha_hat| > 1 (DR7)' : 'RAW CHECKPOINT SCALE',
      equation: `|alpha_hat[${col}]| = |${fmtSigned(val, 4)} * 0.05 / 0.0177| = ${eff.toFixed(4)}`,
      valueText: `raw = ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'nlerp_effective_alpha') {
    const raw = spanning.rawAlpha[col] ?? 0;
    const isExtrap = val > 1.0;
    return Object.freeze({
      world,
      tensorLabel: `Effective |alpha_hat[${col}]|`,
      coordLabel: `ch c=${col} · ${isExtrap ? 'EXTRAPOLATION > 1' : 'INTERPOLATION [0, 1]'}`,
      badge: isExtrap ? 'DR7 UNCLAMPED EXTRAPOLATION' : DORY_NLERP_MARKER,
      equation: `|${fmtSigned(raw, 4)} * (0.05 / 0.0177)| (unclamped >= 0)`,
      valueText: `= ${val.toFixed(4)}`,
    });
  }

  if (
    rec.partId === 'nlerp_a_norm' ||
    rec.partId === 'nlerp_b_norm' ||
    rec.partId === 'nlerp_delta' ||
    rec.partId === 'nlerp_blended_pre_norm' ||
    rec.partId === 'nlerp_output'
  ) {
    const aSheet = sheetsByPartId.get('nlerp_a_norm');
    const bSheet = sheetsByPartId.get('nlerp_b_norm');
    const tokLabel = tokenLabels[row] ?? `t${row}`;
    const aVal = aSheet ? (aSheet.values[row * rec.gridCols + col] ?? 0) : 0;
    const bVal = bSheet ? (bSheet.values[row * rec.gridCols + col] ?? 0) : 0;
    const lam = spanning.effectiveAlpha[col] ?? 0;

    if (rec.partId === 'nlerp_a_norm') {
      return Object.freeze({
        world,
        tensorLabel: 'a = N_8(x_res)',
        coordLabel: `${tokLabel} · ch c=${col}`,
        badge: 'UNIT-SPHERE RESIDUAL (eps=1e-6)',
        equation: `a[${row},${col}] = x_res / sqrt(||x_res||_2^2 + 1e-6)`,
        valueText: `= ${fmtSigned(val, 4)}`,
        operandColors: Object.freeze(['indigo'] as const),
      });
    }
    if (rec.partId === 'nlerp_b_norm') {
      return Object.freeze({
        world,
        tensorLabel: 'b = N_8(x_sub)',
        coordLabel: `${tokLabel} · ch c=${col}`,
        badge: 'UNIT-SPHERE SUBLAYER OUT',
        equation: `b[${row},${col}] = x_sub / sqrt(||x_sub||_2^2 + 1e-6)`,
        valueText: `= ${fmtSigned(val, 4)}`,
        operandColors: Object.freeze(['teal'] as const),
      });
    }
    if (rec.partId === 'nlerp_delta') {
      return Object.freeze({
        world,
        tensorLabel: 'Chord Delta (b - a)',
        coordLabel: `${tokLabel} · ch c=${col}`,
        badge: 'NLERP CHORD VECTOR',
        equation: `${fmtSigned(bVal, 3)} - (${fmtSigned(aVal, 3)})`,
        valueText: `= ${fmtSigned(val, 4)}`,
        operandColors: Object.freeze(['indigo', 'teal'] as const),
        sourceWorlds:
          aSheet && bSheet
            ? Object.freeze([
                { world: computeSheetCellWorldPos(aSheet.grid, col, row, 0.5), role: 'indigo' as const },
                { world: computeSheetCellWorldPos(bSheet.grid, col, row, 0.5), role: 'teal' as const },
              ])
            : undefined,
      });
    }
    if (rec.partId === 'nlerp_blended_pre_norm') {
      return Object.freeze({
        world,
        tensorLabel: 'Pre-Norm Blend a + |alpha_hat|*(b-a)',
        coordLabel: `${tokLabel} · ch c=${col}`,
        badge: lam > 1 ? 'EXTRAPOLATED PAST b (|alpha_hat|>1)' : 'OFF-SPHERE LINEAR BLEND',
        equation: `${fmtSigned(aVal, 3)} + ${lam.toFixed(3)}*(${fmtSigned(bVal - aVal, 3)})`,
        valueText: `= ${fmtSigned(val, 4)}`,
        operandColors: Object.freeze(['indigo', 'teal', 'red'] as const),
        sourceWorlds:
          aSheet && bSheet
            ? Object.freeze([
                { world: computeSheetCellWorldPos(aSheet.grid, col, row, 0.5), role: 'indigo' as const },
                { world: computeSheetCellWorldPos(bSheet.grid, col, row, 0.5), role: 'teal' as const },
              ])
            : undefined,
      });
    }
    return Object.freeze({
      world,
      tensorLabel: 'x_next = N_8(a + |alpha_hat|*(b-a))',
      coordLabel: `${tokLabel} · ch c=${col}`,
      badge: 'nGPT NLERP (NOT SLERP · DR7)',
      equation: `N_8(a + |alpha_hat|*(b - a)) · |alpha_hat[${col}]|=${lam.toFixed(3)}`,
      valueText: `= ${fmtSigned(val, 4)}`,
      operandColors: Object.freeze(['red'] as const),
    });
  }

  if (rec.partId === 'latent_trajectory_8d') {
    const pt = exploratoryPca.points[row];
    const ptLabel = pt?.label ?? `step ${row}`;
    const normStr = pt ? pt.exact8dNorm.toFixed(4) : '1.0000';
    const cosStr = pt ? pt.exact8dAnchorCosine.toFixed(4) : '1.0000';
    return Object.freeze({
      world,
      tensorLabel: `8D Latent State: ${ptLabel}`,
      coordLabel: `ch c=${col} · 8D ||v||=${normStr}`,
      badge: 'ALL METRICS IN ORIGINAL 8D',
      equation: `cos_8D(v, x)=${cosStr} · 2D PCA ||Pi(v)||=${pt ? pt.projectedNorm.toFixed(3) : '0'} <= 1`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'counterfactual_top_probs' && counterfactual !== null) {
    const tokId = counterfactualTokenIds[col] ?? 0;
    const tokText = decodeDoryToken(tokId);
    const pFinal = finalProbsT1[tokId] ?? 0;
    const pCf = counterfactual.rawResult.probs[tokId] ?? 0;
    const isCfRow = row === 1;
    return Object.freeze({
      world,
      tensorLabel: isCfRow
        ? `Counterfactual Coda Loop t=${counterfactual.loopIteration}`
        : `Final Deep Coda Loop k=${counterfactual.kLoops}`,
      coordLabel: `tok #${tokId} "${tokText}" · pos ${counterfactual.absTokenPos}`,
      badge: DORY_COUNTERFACTUAL_CODA_BADGE,
      equation: `p_final=${pFinal.toFixed(4)} vs p_cf=${pCf.toFixed(4)} · D_KL=${counterfactual.klFromFinalLoopNats.toFixed(4)} nats`,
      valueText: `prob = ${val.toFixed(4)}`,
    });
  }

  return null;
}

// ============================================================================
// MAIN VIEW 2 SCENE BUILDER (`buildDoryHypersphereScene`)
// ============================================================================

/**
 * Builds the complete headless 5-pass WebGL2 spatial scene for **View 2 (`hypersphere`)**:
 * nGPT Hyperspherical Residual & Latent Trajectory Inspector.
 *
 * Strictly non-mutating on `DorySession` (even when `includeCounterfactualCoda !== false`:
 * intermediate Coda runs with `writeKv: false` against a read-only historical KV snapshot).
 */
export function buildDoryHypersphereScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options: DoryHypersphereSceneOptions = {},
): DorySpatialScene {
  const ctx = resolveContext(sessionOrCtx, options.recordId);
  const config = ctx.params.config;
  const d = config.hiddenSize; // 8

  const reconstructed = ctx.resolveStepTrace();
  const trace = ctx.resolveForwardTrace();
  const seqLen = trace.seqLen;

  // 1. Resolve selected physical sublayer, loop pass, and token offset
  const selectedPhysicalLayerIdx = clampInt(
    options.physicalLayerIdx ?? DORY_PLANTED_EXTRAPOLATION_SPEC.physicalLayerIdx,
    0,
    config.numHiddenLayers - 1,
  );
  const layerSpec = getDoryPhysicalLayerSpec(selectedPhysicalLayerIdx);
  const selectedStage = getDoryStage(selectedPhysicalLayerIdx);
  const selectedLoopIteration =
    selectedStage === 'recurrent'
      ? clampInt(options.loopIteration ?? 1, 1, ctx.kLoops)
      : 0;

  const defaultTokenOffset = ctx.resolveDefaultTokenOffset();
  const selectedTokenOffset = clampInt(
    options.tokenOffset ?? defaultTokenOffset,
    0,
    Math.max(0, seqLen - 1),
  );
  const selectedAbsTokenPos = trace.startPos + selectedTokenOffset;

  const traceTokens = resolveTraceTokens(ctx, reconstructed, trace);
  const selectedTokenId = traceTokens[selectedTokenOffset] ?? 0;
  const selectedTokenText = decodeDoryToken(selectedTokenId);

  const sublayerRecord = ctx.resolveSublayerRecord(selectedPhysicalLayerIdx, selectedLoopIteration);
  const nlerp = sublayerRecord.trace.nlerp;
  const layerWeights = ctx.params.layers[selectedPhysicalLayerIdx]!;
  const rawAlpha = layerWeights.kind === 'attention' ? layerWeights.attnAlpha : layerWeights.mlpAlpha;

  // 2. Compute Component A: Exact 2D Spanning-Subspace NLERP diagram (`span(a, b)`)
  const spanningSubspace = computeSpanningSubspaceDiagram({
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
    tokenOffset: selectedTokenOffset,
    absTokenPos: selectedAbsTokenPos,
    rawAlpha,
    nlerp,
    eps: config.l2NormEps,
  });

  // 3. Extract exact 8D trajectory states {x, u_1, s_1, ..., u_k, s_k} & Compute Component B: 2D PCA
  const { xAnchor, loopEntryStates, loopExitStates, loopMetrics8d } =
    resolveTokenLoopStatesAndMetrics(ctx, reconstructed, trace, selectedTokenOffset);

  const exploratoryPca = computeExploratoryPcaTrajectory({
    xAnchor,
    loopEntryStates,
    loopExitStates,
  });

  // 4. Compute Component C: Read-Only Diagnostic Counterfactual Coda (ZERO KV mutation!)
  const includeCounterfactualCoda = options.includeCounterfactualCoda ?? true;
  const defaultCfLoop = selectedLoopIteration >= 1 ? selectedLoopIteration : 1;
  const cfLoop = clampInt(options.counterfactualLoop ?? defaultCfLoop, 1, ctx.kLoops);
  const finalLoopProbsT1 = softmaxFloat64(
    ctx.record.lastTokenScaledLogits,
    0,
    config.vocabSize,
    1.0,
  );

  let counterfactualCoda: DoryCounterfactualCodaPreview | null = null;
  if (includeCounterfactualCoda) {
    const rawCf = ctx.resolveCounterfactualCoda(cfLoop);
    counterfactualCoda = computeCounterfactualCodaPreview(
      rawCf,
      ctx.record.lastTokenScaledLogits,
      COUNTERFACTUAL_TOP_K,
    );
  }

  // 5. Pack live RG32F Atlas (compact 64x64 = 32 KiB for k <= 27, 64x128 = 64 KiB for larger k)
  const trajRows = exploratoryPca.points.length; // 1 + 2*kLoops
  const atlasHeight =
    trajRows <= 56
      ? DORY_HYPERSPHERE_ATLAS_DEFAULT_HEIGHT
      : DORY_HYPERSPHERE_ATLAS_EXTENDED_HEIGHT;
  const atlasPacker = new DoryAtlasPacker(
    DORY_HYPERSPHERE_ATLAS_WIDTH,
    atlasHeight,
    DORY_HYPERSPHERE_ATLAS_BUDGET_BYTES,
  );

  const aSlice = atlasPacker.packMatrix('nlerp_a_norm', nlerp.aNorm, seqLen, d, {
    mode: 0,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });
  const bSlice = atlasPacker.packMatrix('nlerp_b_norm', nlerp.bNorm, seqLen, d, {
    mode: 0,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });
  const deltaSlice = atlasPacker.packMatrix('nlerp_delta', nlerp.delta, seqLen, d, {
    mode: 0,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });
  const blendedSlice = atlasPacker.packMatrix(
    'nlerp_blended_pre_norm',
    nlerp.blendedPreNorm,
    seqLen,
    d,
    {
      mode: 0,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const outputSlice = atlasPacker.packMatrix('nlerp_output', nlerp.output, seqLen, d, {
    mode: 0,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });

  // Pack full 8D latent trajectory matrix [1 + 2k, 8]
  const trajMatrix = new Float32Array(trajRows * d);
  trajMatrix.set(xAnchor, 0);
  for (let loopIdx = 0; loopIdx < ctx.kLoops; loopIdx += 1) {
    trajMatrix.set(loopEntryStates[loopIdx]!, (1 + loopIdx * 2) * d);
    trajMatrix.set(loopExitStates[loopIdx]!, (2 + loopIdx * 2) * d);
  }
  const trajSlice = atlasPacker.packMatrix('latent_trajectory_8d', trajMatrix, trajRows, d, {
    mode: 0,
    loopIteration: selectedLoopIteration,
  });

  const rawAlphaSlice = atlasPacker.packMatrix(
    'nlerp_raw_alpha',
    spanningSubspace.rawAlpha,
    1,
    d,
    {
      mode: 0,
      minFloor: 0.02,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const effAlphaSlice = atlasPacker.packMatrix(
    'nlerp_effective_alpha',
    spanningSubspace.effectiveAlpha,
    1,
    d,
    {
      mode: 1,
      fixedScale: 1.0 / Math.max(1.0, spanningSubspace.maxEffectiveAlpha),
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );

  // Optional counterfactual probability comparison strip [2, 8] (Row 0 = final k, Row 1 = cf loop t)
  const counterfactualTokenIds: number[] = [];
  let cfProbsMatrix: Float32Array | null = null;
  let cfProbsSlice = null;
  if (counterfactualCoda !== null) {
    const cfCols = counterfactualCoda.topTokens.length;
    cfProbsMatrix = new Float32Array(2 * cfCols);
    for (let i = 0; i < cfCols; i += 1) {
      const tok = counterfactualCoda.topTokens[i]!;
      counterfactualTokenIds.push(tok.tokenId);
      cfProbsMatrix[i] = finalLoopProbsT1[tok.tokenId] ?? 0;
      cfProbsMatrix[cfCols + i] = tok.prob;
    }
    cfProbsSlice = atlasPacker.packMatrix(
      'counterfactual_top_probs',
      cfProbsMatrix,
      2,
      cfCols,
      {
        mode: 1,
        fixedScale: 1.0,
        loopIteration: counterfactualCoda.loopIteration,
      },
    );
  }

  const atlasExtraTag = `b=${ctx.branch.branchId}:r=${ctx.record.recordId}:p=${selectedPhysicalLayerIdx}:t=${selectedLoopIteration}:tok=${selectedTokenOffset}:cf=${counterfactualCoda ? counterfactualCoda.loopIteration : 'off'}`;
  const packedAtlas = atlasPacker.build('hypersphere', atlasExtraTag);

  // 6. Build 5-pass 3D Editorial Drafting-Table Geometry
  const solids = new DorySolidBufferBuilder();
  const lines = new DoryLineBufferBuilder();
  const ribbons = new DoryRibbonBufferBuilder();
  const sheets = new DorySheetBufferBuilder();
  const flatFlow = new DoryFlatFlowBufferBuilder();

  const pickBoxes: PickBox[] = [];
  const aabbs: AABB[] = [];
  const overlayAnchors: DoryOverlayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const sheetsByPartId = new Map<string, PlacedSheetRecord>();

  const tokenLabels = Object.freeze(
    traceTokens.map((tokId, i) => `p${trace.startPos + i}:${decodeDoryToken(tokId)}`),
  );
  const channelLabels = Object.freeze(Array.from({ length: d }, (_, c) => `c${c}`));
  const trajRowLabels = Object.freeze(exploratoryPca.points.map((pt) => pt.label));

  const cellS = 0.28;
  const sheetH = 0.068;
  const dEx = d * cellS; // 2.24
  const seqEz = Math.max(0.56, Math.min(4.8, seqLen * cellS));
  const stripEz = 0.42;
  const trajEz = Math.max(0.9, Math.min(4.2, trajRows * 0.26));

  const stageTone: DoryBadgeTone =
    selectedStage === 'input'
      ? 'prelude_indigo'
      : selectedStage === 'recurrent'
        ? 'recurrent_teal'
        : 'coda_amber';

  const registerSheet = (spec: {
    readonly partId: string;
    readonly tensorKey: string;
    readonly label: string;
    readonly cx: number;
    readonly cy: number;
    readonly cz: number;
    readonly ex: number;
    readonly ey: number;
    readonly ez: number;
    readonly gridCols: number;
    readonly gridRows: number;
    readonly role: number;
    readonly highlight: number;
    readonly slice: typeof aSlice;
    readonly values: Float32Array;
    readonly format: 'signed' | 'prob';
    readonly colLabels: readonly string[];
    readonly rowLabels: readonly string[];
    readonly colDimLabel: string;
    readonly rowDimLabel: string;
    readonly activeRow?: number;
    readonly activeCol?: number;
    readonly physicalLayerIdx?: number | null;
    readonly loopIteration?: number | null;
    readonly extraTags?: Readonly<Record<string, string | number | boolean>>;
  }): PlacedSheetRecord => {
    const frame = makeSheetFrame3D(
      [spec.cx, spec.cy, spec.cz],
      [spec.ex, spec.ey, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    sheets.push({
      cx: spec.cx,
      cy: spec.cy,
      cz: spec.cz,
      ex: spec.ex,
      ey: spec.ey,
      ez: spec.ez,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      sheetKind: SHEET_KIND_STANDARD,
      role: spec.role,
      highlight: spec.highlight,
      slice: spec.slice,
      frame,
    });

    const meta = encodeDoryPickMetadata({
      physicalLayerIdx: spec.physicalLayerIdx !== undefined ? spec.physicalLayerIdx : selectedPhysicalLayerIdx,
      executedLayerIdx: sublayerRecord.executedLayerIdx,
      loopIteration: spec.loopIteration !== undefined ? spec.loopIteration : selectedLoopIteration,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      absTokenPos: selectedAbsTokenPos,
      modelWeightId: ctx.params.modelWeightId,
      kvPolicy: ctx.kvCacheMode,
      cacheRole: sublayerRecord.coord.cacheOwnership.role,
      extraTags: {
        view: 'hypersphere',
        ...(spec.extraTags ?? {}),
      },
    });
    pickBoxes.push(packPickBoxFromFrame(frame, meta));
    aabbs.push(sheetFrameWorldAabb(frame));

    const grid: ActiveSheetCellGrid = Object.freeze({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: spec.gridCols,
      rows: spec.gridRows,
      values: spec.values,
      format: spec.format,
      activeRow: spec.activeRow,
      activeCol: spec.activeCol,
      colLabels: spec.colLabels,
      rowLabels: spec.rowLabels,
      colDimLabel: spec.colDimLabel,
      rowDimLabel: spec.rowDimLabel,
    });
    activeCellGrids.push(grid);

    const placed: PlacedSheetRecord = {
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      label: spec.label,
      frame,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      values: spec.values,
      grid,
    };
    sheetsByPartId.set(spec.partId, placed);
    return placed;
  };

  // --- LEFT STAGE: 3D CHANNEL-WISE nGPT NLERP CUTAWAY ---
  const leftFrameZSpan = Math.max(5.4, seqEz + 3.0);
  lines.addBoxFrameXZ(-12.6, -leftFrameZSpan, 3.2, leftFrameZSpan, -0.06, 1.0, LINE_STYLE_GUIDE, 0.36);

  const zBranchOffset = Math.max(1.8, seqEz * 0.5 + 0.9);

  const sheetA = registerSheet({
    partId: 'nlerp_a_norm',
    tensorKey: 'nlerp_a_norm',
    label: 'a = N_8(x_res)',
    cx: -10.8,
    cy: 0.0,
    cz: -zBranchOffset,
    ex: dEx,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_OPERAND_A,
    slice: aSlice,
    values: nlerp.aNorm,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
  });

  const sheetB = registerSheet({
    partId: 'nlerp_b_norm',
    tensorKey: 'nlerp_b_norm',
    label: 'b = N_8(x_sub)',
    cx: -10.8,
    cy: 0.0,
    cz: zBranchOffset,
    ex: dEx,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_B,
    slice: bSlice,
    values: nlerp.bNorm,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
  });

  const sheetDelta = registerSheet({
    partId: 'nlerp_delta',
    tensorKey: 'nlerp_delta',
    label: 'Chord (b - a)',
    cx: -6.8,
    cy: 0.0,
    cz: 0.25,
    ex: dEx,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_NEUTRAL,
    highlight: SHEET_HL_NORMAL,
    slice: deltaSlice,
    values: nlerp.delta,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
  });

  const rawAlphaZ = -(seqEz * 0.5 + 2.55);
  const effAlphaZ = -(seqEz * 0.5 + 1.5);

  const sheetRawAlpha = registerSheet({
    partId: 'nlerp_raw_alpha',
    tensorKey: layerWeights.kind === 'attention' ? 'attn_alpha' : 'mlp_alpha',
    label: 'Raw alpha (init 0.0177)',
    cx: -2.8,
    cy: 0.0,
    cz: rawAlphaZ,
    ex: dEx,
    ey: sheetH,
    ez: stripEz,
    gridCols: d,
    gridRows: 1,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_BF16_GOLD,
    slice: rawAlphaSlice,
    values: spanningSubspace.rawAlpha,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: Object.freeze(['raw']),
    colDimLabel: `d=${d}`,
    rowDimLabel: '1',
    activeCol: spanningSubspace.extrapolatedChannels[0],
  });

  const hasExtrapolation = spanningSubspace.extrapolatedChannels.length > 0;
  const sheetEffAlpha = registerSheet({
    partId: 'nlerp_effective_alpha',
    tensorKey: 'nlerp_effective_alpha',
    label: '|alpha_hat| = |alpha * 0.05/0.0177|',
    cx: -2.8,
    cy: 0.0,
    cz: effAlphaZ,
    ex: dEx,
    ey: sheetH,
    ez: stripEz,
    gridCols: d,
    gridRows: 1,
    role: SHEET_ROLE_NORM,
    highlight: hasExtrapolation ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
    slice: effAlphaSlice,
    values: spanningSubspace.effectiveAlpha,
    format: 'prob',
    colLabels: channelLabels,
    rowLabels: Object.freeze(['|a_hat|']),
    colDimLabel: `d=${d}`,
    rowDimLabel: '1',
    activeCol: spanningSubspace.extrapolatedChannels[0],
    extraTags: { nlerpMarker: DORY_NLERP_MARKER },
  });

  const sheetBlended = registerSheet({
    partId: 'nlerp_blended_pre_norm',
    tensorKey: 'nlerp_blended_pre_norm',
    label: 'a + |alpha_hat| * (b - a)',
    cx: -2.8,
    cy: 0.0,
    cz: 0.65,
    ex: dEx,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_PATHWAY,
    slice: blendedSlice,
    values: nlerp.blendedPreNorm,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
  });

  const sheetOutput = registerSheet({
    partId: 'nlerp_output',
    tensorKey: 'nlerp_output',
    label: 'x_next = N_8(blend)',
    cx: 1.4,
    cy: 0.0,
    cz: 0.65,
    ex: dEx,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_ACTIVE,
    slice: outputSlice,
    values: nlerp.output,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
    extraTags: { nlerpMarker: DORY_NLERP_MARKER },
  });

  // Highlight selected token row across all 5 [S, 8] NLERP sheets
  const rowFill = { r: 0.863, g: 0.298, b: 0.204, a: 0.14, borderAlpha: 0.72 };
  const rowFrame = { r: 0.863, g: 0.298, b: 0.204, a: 0.88, borderAlpha: 0.94 };
  const nlerpSeqSheets = [sheetA, sheetB, sheetDelta, sheetBlended, sheetOutput];
  for (let i = 0; i < nlerpSeqSheets.length; i += 1) {
    const s = nlerpSeqSheets[i]!;
    flatFlow.addCellHighlightOnFrame(
      s.frame,
      selectedTokenOffset,
      selectedTokenOffset,
      0,
      d - 1,
      seqLen,
      d,
      0.042,
      rowFill,
      rowFrame,
      0.02,
      true,
    );
  }

  // Highlight planted/extrapolated |alpha_hat| > 1 channels with raised 3D bars & highlight frames
  for (let i = 0; i < spanningSubspace.extrapolatedChannels.length; i += 1) {
    const ch = spanningSubspace.extrapolatedChannels[i]!;
    const effVal = spanningSubspace.effectiveAlpha[ch]!;
    flatFlow.addCellHighlightOnFrame(
      sheetEffAlpha.frame,
      0,
      0,
      ch,
      ch,
      1,
      d,
      0.045,
      { r: 0.88, g: 0.26, b: 0.28, a: 0.26, borderAlpha: 0.92 },
      DORY_FLOW_COLORS.extrapolationCoral,
      0.024,
      true,
    );
    const barPos = computeSheetCellWorldPos(sheetEffAlpha.grid, ch, 0, 0.5);
    const barHeight = Math.min(1.1, Math.max(0.24, effVal * 0.42));
    const barCenterY = barPos[1] + barHeight * 0.5 + 0.04;
    solids.push({
      cx: barPos[0],
      cy: barCenterY,
      cz: barPos[2],
      ex: 0.17,
      ey: barHeight,
      ez: 0.17,
      cellFamily: CELL_MOE_AMBER,
      bayId: selectedPhysicalLayerIdx + 1,
      cellScale: 0.14,
      highlight: 1,
    });
    lines.addBoxEdges(
      barPos[0],
      barCenterY,
      barPos[2],
      0.17,
      barHeight,
      0.17,
      selectedPhysicalLayerIdx + 1,
      1.1,
      LINE_STYLE_ACTIVE,
      0.86,
    );
    const barMeta = encodeDoryPickMetadata({
      physicalLayerIdx: selectedPhysicalLayerIdx,
      executedLayerIdx: sublayerRecord.executedLayerIdx,
      loopIteration: selectedLoopIteration,
      partId: `nlerp_extrap_ch_${ch}`,
      tensorKey: 'nlerp_effective_alpha',
      modelWeightId: ctx.params.modelWeightId,
      extraTags: { channel: ch, extrapolated: true, nlerpMarker: DORY_NLERP_MARKER },
    });
    pickBoxes.push(
      packPickBoxFromAabb(barPos[0], barCenterY, barPos[2], 0.2, barHeight, 0.2, barMeta),
    );
    aabbs.push({
      min: [barPos[0] - 0.1, barCenterY - barHeight * 0.5, barPos[2] - 0.1],
      max: [barPos[0] + 0.1, barCenterY + barHeight * 0.5, barPos[2] + 0.1],
    });
  }

  // Left-stage drafting flow ribbons connecting the NLERP pipeline
  const flowY = -0.025;
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetA.frame.center[0] + dEx * 0.5 + 0.08, z: sheetA.frame.center[2] },
      { x: -8.6, z: sheetA.frame.center[2] },
      { x: -8.6, z: sheetDelta.frame.center[2] - 0.35 },
      { x: sheetDelta.frame.center[0] - dEx * 0.5 - 0.08, z: sheetDelta.frame.center[2] - 0.35 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.preludeIndigo,
    0.26,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetB.frame.center[0] + dEx * 0.5 + 0.08, z: sheetB.frame.center[2] },
      { x: -8.6, z: sheetB.frame.center[2] },
      { x: -8.6, z: sheetDelta.frame.center[2] + 0.35 },
      { x: sheetDelta.frame.center[0] - dEx * 0.5 - 0.08, z: sheetDelta.frame.center[2] + 0.35 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.recurrentTeal,
    0.26,
    true,
  );
  // Residual anchor skip from a = N(x_res) directly into blended pre-norm
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetA.frame.center[0], z: sheetA.frame.center[2] - seqEz * 0.5 - 0.1 },
      { x: sheetA.frame.center[0], z: rawAlphaZ - 0.72 },
      { x: -4.85, z: rawAlphaZ - 0.72 },
      { x: -4.85, z: sheetBlended.frame.center[2] - 0.4 },
      { x: sheetBlended.frame.center[0] - dEx * 0.5 - 0.08, z: sheetBlended.frame.center[2] - 0.4 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.nlerpEmerald,
    0.28,
    true,
  );
  // Chord delta -> pre-norm blend
  flatFlow.addPolylineXZ(
    [
      { x: sheetDelta.frame.center[0] + dEx * 0.5 + 0.08, z: sheetBlended.frame.center[2] },
      { x: sheetBlended.frame.center[0] - dEx * 0.5 - 0.08, z: sheetBlended.frame.center[2] },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.swaCyan,
    true,
  );
  // Raw alpha -> effective |alpha_hat| -> pre-norm blend
  flatFlow.addPolylineXZ(
    [
      { x: sheetRawAlpha.frame.center[0], z: sheetRawAlpha.frame.center[2] + stripEz * 0.5 + 0.06 },
      { x: sheetEffAlpha.frame.center[0], z: sheetEffAlpha.frame.center[2] - stripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetEffAlpha.frame.center[0], z: sheetEffAlpha.frame.center[2] + stripEz * 0.5 + 0.06 },
      { x: sheetBlended.frame.center[0], z: sheetBlended.frame.center[2] - seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.11,
    hasExtrapolation ? DORY_FLOW_COLORS.extrapolationCoral : DORY_FLOW_COLORS.anchorGold,
    true,
  );
  // Pre-norm blend -> final unit-sphere N_8 output
  flatFlow.addPolylineXZ(
    [
      { x: sheetBlended.frame.center[0] + dEx * 0.5 + 0.08, z: sheetOutput.frame.center[2] },
      { x: sheetOutput.frame.center[0] - dEx * 0.5 - 0.08, z: sheetOutput.frame.center[2] },
    ],
    flowY,
    0.13,
    DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );
  ribbons.push({
    cx: (sheetBlended.frame.center[0] + sheetOutput.frame.center[0]) * 0.5,
    cy: 0.14,
    cz: sheetOutput.frame.center[2],
    ex: 0.72,
    ey: 0.08,
    ez: 0.36,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.42,
  });

  // --- RIGHT STAGE: 8D LATENT TRAJECTORY, EXPLORATORY 2D PCA DISK, EXACT 2D SUBSPACE & COUNTERFACTUAL ---
  lines.addBoxFrameXZ(4.6, -leftFrameZSpan, 16.2, leftFrameZSpan, -0.06, 1.0, LINE_STYLE_GUIDE, 0.36);

  // Sheet: Full 8D Latent Trajectory Matrix [1 + 2k, 8]
  const sheetTrajectory = registerSheet({
    partId: 'latent_trajectory_8d',
    tensorKey: 'latent_trajectory_8d',
    label: `8D Latent Trace {x, u_1..s_${ctx.kLoops}}`,
    cx: 6.7,
    cy: 0.0,
    cz: -2.15,
    ex: dEx,
    ey: sheetH,
    ez: trajEz,
    gridCols: d,
    gridRows: trajRows,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: trajSlice,
    values: trajMatrix,
    format: 'signed',
    colLabels: channelLabels,
    rowLabels: trajRowLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `1+2k=${trajRows}`,
    activeRow: selectedLoopIteration >= 1 ? selectedLoopIteration * 2 : 0,
    physicalLayerIdx: null,
    loopIteration: selectedLoopIteration,
    extraTags: { metricsIn8d: true },
  });

  // Optional Sheet: Read-Only Diagnostic Counterfactual Coda Probability Strip [2, 8]
  let sheetCounterfactual: PlacedSheetRecord | null = null;
  if (counterfactualCoda !== null && cfProbsMatrix !== null && cfProbsSlice !== null) {
    const cfCols = counterfactualCoda.topTokens.length;
    const cfEx = Math.max(2.0, cfCols * 0.28);
    const cfEz = 0.78;
    const cfCx = 6.7;
    const cfCz = 2.65;

    sheetCounterfactual = registerSheet({
      partId: 'counterfactual_top_probs',
      tensorKey: 'counterfactual_top_probs',
      label: `Counterfactual Coda (Loop t=${counterfactualCoda.loopIteration} vs k=${counterfactualCoda.kLoops})`,
      cx: cfCx,
      cy: 0.0,
      cz: cfCz,
      ex: cfEx,
      ey: sheetH,
      ez: cfEz,
      gridCols: cfCols,
      gridRows: 2,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_BF16_GOLD,
      slice: cfProbsSlice,
      values: cfProbsMatrix,
      format: 'prob',
      colLabels: Object.freeze(counterfactualCoda.topTokens.map((t) => t.tokenText)),
      rowLabels: Object.freeze([`k=${counterfactualCoda.kLoops}`, `cf t=${counterfactualCoda.loopIteration}`]),
      colDimLabel: `top-${cfCols}`,
      rowDimLabel: '2',
      activeRow: 1,
      physicalLayerIdx: null,
      loopIteration: counterfactualCoda.loopIteration,
      extraTags: { counterfactual: true, readOnlyZeroKvMutation: true },
    });

    lines.addDashedFrameXZ(
      cfCx - cfEx * 0.5 - 0.34,
      cfCz - cfEz * 0.5 - 0.34,
      cfCx + cfEx * 0.5 + 0.34,
      cfCz + cfEz * 0.5 + 0.34,
      0.01,
      LINE_STYLE_BF16_GOLD,
      0.78,
      0.2,
      0.12,
      1.15,
    );
    ribbons.push({
      cx: cfCx,
      cy: 0.14,
      cz: cfCz - cfEz * 0.5 - 0.65,
      ex: 0.44,
      ey: 0.06,
      ez: 0.44,
      kind: RIBBON_KIND_ARROW_POS_X,
      colorId: RIBBON_COLOR_AMBER,
      alpha: 0.34,
    });
  }

  // --- PEDESTAL (b): EXPLORATORY 2D PCA UNIT DISK (PROJECTED NORMS <= 1; ALL METRICS IN 8D) ---
  const pcaCenterX = 12.6;
  const pcaCenterZ = -2.35;
  const pcaRadius = 2.15;
  const pcaPlaneY = 0.02;

  const drawUnitDiskWireframe = (
    cx: number,
    cz: number,
    radius: number,
    y: number,
    ringStyle: number,
  ): void => {
    for (let i = 0; i < UNIT_CIRCLE_SEGMENTS; i += 1) {
      const a0 = (i / UNIT_CIRCLE_SEGMENTS) * Math.PI * 2;
      const a1 = ((i + 1) / UNIT_CIRCLE_SEGMENTS) * Math.PI * 2;
      lines.addSegment(
        cx + Math.cos(a0) * radius,
        y,
        cz - Math.sin(a0) * radius,
        cx + Math.cos(a1) * radius,
        y,
        cz - Math.sin(a1) * radius,
        1.25,
        ringStyle,
        0.74,
      );
      // Inner r = 0.5 reference ring emphasizing projected norms <= 1
      lines.addSegment(
        cx + Math.cos(a0) * radius * 0.5,
        y,
        cz - Math.sin(a0) * radius * 0.5,
        cx + Math.cos(a1) * radius * 0.5,
        y,
        cz - Math.sin(a1) * radius * 0.5,
        0.8,
        LINE_STYLE_GUIDE,
        0.32,
      );
    }
    lines.addSegment(cx - radius * 1.08, y, cz, cx + radius * 1.08, y, cz, 0.9, LINE_STYLE_GUIDE, 0.45);
    lines.addSegment(cx, y, cz - radius * 1.08, cx, y, cz + radius * 1.08, 0.9, LINE_STYLE_GUIDE, 0.45);
    aabbs.push({
      min: [cx - radius - 0.3, y - 0.1, cz - radius - 0.3],
      max: [cx + radius + 0.3, y + 0.7, cz + radius + 0.3],
    });
  };

  drawUnitDiskWireframe(pcaCenterX, pcaCenterZ, pcaRadius, pcaPlaneY, LINE_STYLE_NEUTRAL);

  const pcaWorldPath: Vec3[] = [];
  for (let i = 0; i < exploratoryPca.points.length; i += 1) {
    const pt = exploratoryPca.points[i]!;
    const wx = pcaCenterX + pt.x * pcaRadius;
    const wz = pcaCenterZ - pt.y * pcaRadius;
    const wy =
      pcaPlaneY +
      (pt.kind === 'anchor' ? 0.2 : pt.kind === 'loop_entry' ? 0.14 : 0.24 + pt.loopIteration * 0.03);

    pcaWorldPath.push([wx, wy, wz]);

    const nodeSize = pt.kind === 'anchor' ? 0.22 : pt.kind === 'loop_exit' ? 0.2 : 0.16;
    const nodeFamily =
      pt.kind === 'anchor'
        ? CELL_MOE_AMBER
        : pt.kind === 'loop_entry'
          ? CELL_SSM_CYAN
          : CELL_ACTIVATION_GREEN;
    const isFocusedLoop = pt.kind === 'loop_exit' && pt.loopIteration === selectedLoopIteration;

    solids.push({
      cx: wx,
      cy: wy,
      cz: wz,
      ex: nodeSize,
      ey: nodeSize,
      ez: nodeSize,
      cellFamily: nodeFamily,
      bayId: 0,
      cellScale: 0.14,
      highlight: isFocusedLoop ? 1 : pt.kind === 'anchor' ? 2 : 0,
    });
    lines.addBoxEdges(
      wx,
      wy,
      wz,
      nodeSize,
      nodeSize,
      nodeSize,
      0,
      isFocusedLoop ? 1.25 : 0.95,
      isFocusedLoop ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      0.78,
    );
    // Drop vertical stem onto the 2D unit disk plane
    lines.addSegment(wx, pcaPlaneY, wz, wx, wy - nodeSize * 0.5, wz, 0.85, LINE_STYLE_GUIDE, 0.55);

    const nodeMeta = encodeDoryPickMetadata({
      physicalLayerIdx: null,
      loopIteration: pt.loopIteration,
      partId: `pca_point_${pt.id}`,
      tensorKey: 'exploratory_pca_2d',
      absTokenPos: selectedAbsTokenPos,
      modelWeightId: ctx.params.modelWeightId,
      groupRole: 'loop',
      tapRole: 'loop',
      extraTags: { pcaKind: pt.kind, metricsIn8d: true },
    });
    pickBoxes.push(packPickBoxFromAabb(wx, wy, wz, nodeSize + 0.06, nodeSize + 0.06, nodeSize + 0.06, nodeMeta));
  }

  if (pcaWorldPath.length >= 2) {
    flatFlow.addPolyline3D(
      pcaWorldPath,
      [0, 1, 0],
      0.065,
      DORY_FLOW_COLORS.recurrentTeal,
      true,
      0.2,
      0.18,
      true,
    );
  }

  // --- PEDESTAL (a): EXACT 2D SPANNING-SUBSPACE DIAGRAM span(a, b) ---
  const spanCenterX = 12.6;
  const spanCenterZ = 2.65;
  const spanRadius = 1.95;
  const spanPlaneY = 0.02;

  drawUnitDiskWireframe(spanCenterX, spanCenterZ, spanRadius, spanPlaneY, LINE_STYLE_ACTIVE);

  // Draw uniform-scalar NLERP interpolation curve (emerald) & >1 extrapolation curve (coral)
  const interpCurvePts: Vec3[] = [];
  const extrapCurvePts: Vec3[] = [];
  for (let i = 0; i < spanningSubspace.uniformAlphaCurve.length; i += 1) {
    const pt = spanningSubspace.uniformAlphaCurve[i]!;
    const wPos: Vec3 = [
      spanCenterX + pt.x * spanRadius,
      spanPlaneY + 0.08,
      spanCenterZ - pt.y * spanRadius,
    ];
    if (!pt.isExtrapolation) {
      interpCurvePts.push(wPos);
    } else {
      if (extrapCurvePts.length === 0 && interpCurvePts.length > 0) {
        extrapCurvePts.push(interpCurvePts[interpCurvePts.length - 1]!);
      }
      extrapCurvePts.push(wPos);
    }
  }
  if (interpCurvePts.length >= 2) {
    flatFlow.addPolyline3D(
      interpCurvePts,
      [0, 1, 0],
      0.075,
      DORY_FLOW_COLORS.nlerpEmerald,
      false,
      0.18,
      0.16,
      true,
    );
  }
  if (extrapCurvePts.length >= 2) {
    flatFlow.addPolyline3D(
      extrapCurvePts,
      [0, 1, 0],
      0.085,
      DORY_FLOW_COLORS.extrapolationCoral,
      true,
      0.2,
      0.18,
      true,
    );
  }

  // Exact basis vectors a2d, b2d and chord line a2d -> b2d
  const aWorldX = spanCenterX + spanningSubspace.a2d[0] * spanRadius;
  const aWorldZ = spanCenterZ - spanningSubspace.a2d[1] * spanRadius;
  const bWorldX = spanCenterX + spanningSubspace.b2d[0] * spanRadius;
  const bWorldZ = spanCenterZ - spanningSubspace.b2d[1] * spanRadius;

  // Radial vectors from origin & straight chord a -> b (shows NLERP chord vs unit arc)
  lines.addSegment(spanCenterX, spanPlaneY + 0.04, spanCenterZ, aWorldX, spanPlaneY + 0.04, aWorldZ, 1.35, LINE_STYLE_NEUTRAL, 0.8);
  lines.addSegment(spanCenterX, spanPlaneY + 0.04, spanCenterZ, bWorldX, spanPlaneY + 0.04, bWorldZ, 1.35, LINE_STYLE_NEUTRAL, 0.8);
  lines.addSegment(aWorldX, spanPlaneY + 0.06, aWorldZ, bWorldX, spanPlaneY + 0.06, bWorldZ, 1.15, LINE_STYLE_BF16_GOLD, 0.72);

  // Solid markers for a, b, projected pre-norm blend, and projected 8D NLERP output (+ out-of-plane stem)
  const pushSpanNode = (
    partId: string,
    wx: number,
    wy: number,
    wz: number,
    size: number,
    cellFamily: number,
    highlight: number,
    extraTags: Readonly<Record<string, string | number | boolean>>,
  ): void => {
    solids.push({
      cx: wx,
      cy: wy,
      cz: wz,
      ex: size,
      ey: size,
      ez: size,
      cellFamily,
      bayId: selectedPhysicalLayerIdx + 1,
      cellScale: 0.14,
      highlight,
    });
    lines.addBoxEdges(wx, wy, wz, size, size, size, selectedPhysicalLayerIdx + 1, 1.05, highlight === 1 ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL, 0.82);
    const meta = encodeDoryPickMetadata({
      physicalLayerIdx: selectedPhysicalLayerIdx,
      executedLayerIdx: sublayerRecord.executedLayerIdx,
      loopIteration: selectedLoopIteration,
      partId,
      tensorKey: 'spanning_subspace_2d',
      absTokenPos: selectedAbsTokenPos,
      modelWeightId: ctx.params.modelWeightId,
      extraTags,
    });
    pickBoxes.push(packPickBoxFromAabb(wx, wy, wz, size + 0.06, size + 0.06, size + 0.06, meta));
    aabbs.push({
      min: [wx - size, wy - size, wz - size],
      max: [wx + size, wy + size, wz + size],
    });
  };

  pushSpanNode('span_vec_a', aWorldX, spanPlaneY + 0.14, aWorldZ, 0.2, CELL_WEIGHT_BLUE, 0, { vec: 'a' });
  pushSpanNode('span_vec_b', bWorldX, spanPlaneY + 0.14, bWorldZ, 0.2, CELL_SSM_CYAN, 0, { vec: 'b' });

  const outProjX = spanCenterX + spanningSubspace.projectedOutput2d[0] * spanRadius;
  const outProjZ = spanCenterZ - spanningSubspace.projectedOutput2d[1] * spanRadius;
  const outOfPlaneLift = 0.16 + Math.min(1.4, spanningSubspace.outOfPlaneNorm * 1.35);
  pushSpanNode(
    'span_nlerp_out',
    outProjX,
    spanPlaneY + outOfPlaneLift,
    outProjZ,
    0.22,
    CELL_ACTIVATION_GREEN,
    1,
    { vec: 'nlerp_out', nlerpMarker: DORY_NLERP_MARKER },
  );
  // Stem illustrating out-of-plane displacement caused by per-channel |alpha_hat_c|
  lines.addSegment(
    outProjX,
    spanPlaneY,
    outProjZ,
    outProjX,
    spanPlaneY + outOfPlaneLift - 0.11,
    outProjZ,
    1.3,
    LINE_STYLE_ACTIVE,
    0.85,
  );

  // Connect Left Stage output to Right Stage trajectory & spanning subspace pedestals
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetOutput.frame.center[0] + dEx * 0.5 + 0.1, z: sheetOutput.frame.center[2] },
      { x: 4.1, z: sheetOutput.frame.center[2] },
      { x: 4.1, z: sheetTrajectory.frame.center[2] },
      { x: sheetTrajectory.frame.center[0] - dEx * 0.5 - 0.1, z: sheetTrajectory.frame.center[2] },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.recurrentTeal,
    0.28,
    true,
  );

  // 7. Overlay Anchors, Landmarks & Dimension Calipers
  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_nlerp_a',
      label: 'a = N_8(x_res)',
      sublabel: `[${seqLen} x ${d}] unit-sphere residual`,
      worldPos: localToWorldOnFrame(sheetA.frame, 0, 0.65, -0.62),
      tone: 'prelude_indigo',
    }),
    Object.freeze({
      id: 'anchor_nlerp_b',
      label: 'b = N_8(x_sub)',
      sublabel: `[${seqLen} x ${d}] sublayer output`,
      worldPos: localToWorldOnFrame(sheetB.frame, 0, 0.65, 0.62),
      tone: 'recurrent_teal',
    }),
    Object.freeze({
      id: 'anchor_nlerp_delta',
      label: 'Chord Delta (b - a)',
      sublabel: `omega = ${spanningSubspace.omegaDegrees.toFixed(1)} deg`,
      worldPos: localToWorldOnFrame(sheetDelta.frame, 0, 0.65, -0.62),
      tone: 'muted',
    }),
    Object.freeze({
      id: 'anchor_raw_eff_alpha',
      label: 'Raw alpha -> |alpha_hat|',
      sublabel: hasExtrapolation
        ? `UNCLAMPED > 1 on ch [${spanningSubspace.extrapolatedChannels.join(',')}]`
        : `|alpha * 0.05 / 0.0177| (max ${spanningSubspace.maxEffectiveAlpha.toFixed(2)})`,
      worldPos: localToWorldOnFrame(sheetRawAlpha.frame, 0, 0.85, -0.85),
      tone: hasExtrapolation ? 'warn' : 'gold',
      badge: 'DR7',
    }),
    Object.freeze({
      id: 'anchor_nlerp_output',
      label: 'x_next = N_8(a + |alpha_hat|*(b - a))',
      sublabel: 'nGPT NLERP — NOT Trigonometric SLERP',
      worldPos: localToWorldOnFrame(sheetOutput.frame, 0, 0.75, -0.62),
      tone: 'nlerp_emerald',
      badge: DORY_NLERP_MARKER,
    }),
    Object.freeze({
      id: 'anchor_latent_trace',
      label: `8D Latent Trace (k=${ctx.kLoops})`,
      sublabel: `{x, u_1, s_1..s_${ctx.kLoops}} · pos ${selectedAbsTokenPos} "${selectedTokenText}"`,
      worldPos: localToWorldOnFrame(sheetTrajectory.frame, 0, 0.75, -0.62),
      tone: 'recurrent_teal',
    }),
    Object.freeze({
      id: 'anchor_pca_disk',
      label: 'Exploratory 2D PCA Disk',
      sublabel: 'PROJECTED NORMS <= 1; ALL METRICS IN 8D',
      worldPos: Object.freeze([pcaCenterX, 0.68, pcaCenterZ - pcaRadius - 0.35] as const),
      tone: 'swa_cyan',
      badge: 'EXPLORATORY 2D PCA',
    }),
    Object.freeze({
      id: 'anchor_spanning_subspace',
      label: 'Exact 2D span(a, b)',
      sublabel: `omega=${spanningSubspace.omegaDegrees.toFixed(1)} deg · perp=${spanningSubspace.outOfPlaneNorm.toFixed(3)}`,
      worldPos: Object.freeze([spanCenterX, 0.68, spanCenterZ + spanRadius + 0.35] as const),
      tone: 'nlerp_emerald',
      badge: 'EXACT 2D SUBSPACE',
    }),
  );

  if (sheetCounterfactual !== null && counterfactualCoda !== null) {
    overlayAnchors.push(
      Object.freeze({
        id: 'anchor_counterfactual_coda',
        label: `Counterfactual Coda (Loop t=${counterfactualCoda.loopIteration})`,
        sublabel: `D_KL=${counterfactualCoda.klFromFinalLoopNats.toFixed(4)} nats · writeKv=false`,
        worldPos: localToWorldOnFrame(sheetCounterfactual.frame, 0, 0.75, 0.78),
        tone: 'gold',
        badge: 'DIAGNOSTIC COUNTERFACTUAL',
      }),
    );
  }

  landmarks.push(
    Object.freeze({
      id: 'lm_nlerp_cutaway',
      label: 'nGPT NLERP Cutaway (NOT SLERP)',
      sublabel: `Phys L${selectedPhysicalLayerIdx} (${layerSpec.transformersPairBadge})`,
      world: Object.freeze([-4.8, 0.8, -leftFrameZSpan - 0.35] as const),
      partId: 'nlerp_output',
      tone: 'default',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_pca_pedestal',
      label: 'Exploratory 2D PCA (8D Metrics)',
      sublabel: `Var ${(exploratoryPca.explainedVarianceRatio[0] * 100).toFixed(0)}% + ${(exploratoryPca.explainedVarianceRatio[1] * 100).toFixed(0)}%`,
      world: Object.freeze([pcaCenterX, 0.9, pcaCenterZ - pcaRadius - 0.4] as const),
      tone: 'cyan',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_span_pedestal',
      label: 'Exact 2D Spanning Subspace span(a, b)',
      sublabel: hasExtrapolation
        ? `Extrap ch [${spanningSubspace.extrapolatedChannels.join(',')}] > 1`
        : 'Chord NLERP vs Unit Circle',
      world: Object.freeze([spanCenterX, 0.9, spanCenterZ + spanRadius + 0.4] as const),
      tone: 'gold',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'caliper_d8',
      aWorld: localToWorldOnFrame(sheetOutput.frame, -0.5, 0.5, 0.58),
      bWorld: localToWorldOnFrame(sheetOutput.frame, 0.5, 0.5, 0.58),
      label: `d = ${d} (prod 2,560)`,
      partIds: Object.freeze(['nlerp_output']),
    }),
    Object.freeze({
      id: 'caliper_seq',
      aWorld: localToWorldOnFrame(sheetOutput.frame, 0.62, 0.5, -0.5),
      bWorld: localToWorldOnFrame(sheetOutput.frame, 0.62, 0.5, 0.5),
      label: `S = ${seqLen}`,
      partIds: Object.freeze(['nlerp_output']),
    }),
  );

  // 8. Authoritative Epistemic Badges & Callouts (D7, DR4, DR7, DR8, Finding 9)
  const activeLoopMetricIdx =
    selectedLoopIteration >= 1
      ? Math.min(loopMetrics8d.length - 1, selectedLoopIteration - 1)
      : loopMetrics8d.length - 1;
  const activeLoopMetric = loopMetrics8d[activeLoopMetricIdx]!;
  const activeJac = activeLoopMetric.entryAnchorJacobian;

  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({
      label: DORY_ANTI_SLERP_PLAQUE,
      tone: 'nlerp_emerald',
      discrepancyId: 'DR7',
    }),
    Object.freeze({
      label: DORY_EXPLORATORY_PCA_BADGE,
      tone: 'swa_cyan',
    }),
    Object.freeze({
      label: DORY_ANCHOR_MAP_CAVEAT_BADGE,
      tone: 'warn',
    }),
    Object.freeze({
      label: `Phys L${selectedPhysicalLayerIdx} · ${layerSpec.transformersPairBadge}`,
      tone: stageTone,
      discrepancyId: 'DR4',
    }),
  ];

  if (hasExtrapolation) {
    provenanceBadges.push(
      Object.freeze({
        label: `Unclamped |alpha_hat| > 1 on ch [${spanningSubspace.extrapolatedChannels.join(', ')}] (max ${spanningSubspace.maxEffectiveAlpha.toFixed(2)})`,
        tone: 'warn',
        discrepancyId: 'DR7',
      }),
    );
  }
  if (counterfactualCoda !== null) {
    provenanceBadges.push(
      Object.freeze({
        label: DORY_COUNTERFACTUAL_CODA_BADGE,
        tone: 'gold',
      }),
    );
  }

  const callouts: DorySceneCallout[] = [
    Object.freeze({
      id: 'callout_anti_slerp',
      title: DORY_ANTI_SLERP_PLAQUE,
      body:
        `Phys L${selectedPhysicalLayerIdx} (${layerSpec.transformersPairBadge}) computes ` +
        `N_8(a + |alpha * 0.05 / 0.0177| * (b - a)) in binary64 accumulation with binary32 tensor write-back (D5, DR7, DR9). ` +
        `Despite vLLM's helper name 'slerp_norm', Dory NEVER runs trigonometric sin/arccos SLERP and NEVER clamps |alpha_hat| to [0, 1]. ` +
        (hasExtrapolation
          ? `Channels [${spanningSubspace.extrapolatedChannels.join(', ')}] extrapolate past b (max |alpha_hat| = ${spanningSubspace.maxEffectiveAlpha.toFixed(3)}).`
          : `Max effective |alpha_hat| = ${spanningSubspace.maxEffectiveAlpha.toFixed(3)}.`),
      tone: 'nlerp_emerald',
      worldPos: sheetOutput.frame.center,
      badge: DORY_NLERP_MARKER,
      discrepancyIds: Object.freeze(['DR7', 'DR9'] as const),
    }),
    Object.freeze({
      id: 'callout_exact_spanning_subspace',
      title: 'Diagram (a): Exact 2D Spanning Subspace span(a, b)',
      body:
        `Exact planar geometry of unit vectors a = N(x_res) and b = N(x_sub) separated by ` +
        `omega = ${spanningSubspace.omegaDegrees.toFixed(2)} deg (cos = ${spanningSubspace.cosineAB.toFixed(4)}). ` +
        `Uniform-scalar NLERP follows the chord projection onto the unit circle; channel-wise |alpha_hat_c| ` +
        `induces exact out-of-plane norm ||x_next - Pi(x_next)||_2 = ${spanningSubspace.outOfPlaneNorm.toFixed(4)} in R^8.`,
      tone: 'nlerp_emerald',
      worldPos: Object.freeze([spanCenterX, 0.2, spanCenterZ] as const),
      badge: 'EXACT 2D SUBSPACE',
      discrepancyIds: Object.freeze(['DR7'] as const),
    }),
    Object.freeze({
      id: 'callout_exploratory_pca',
      title: 'Diagram (b): Exploratory 2D PCA Trajectory Projection',
      body:
        `${DORY_EXPLORATORY_PCA_BADGE}. Orthogonal projection of 8D unit-sphere states ` +
        `{x, u_1, s_1..s_${ctx.kLoops}} onto (PC1, PC2) explains ` +
        `${((exploratoryPca.explainedVarianceRatio[0] + exploratoryPca.explainedVarianceRatio[1]) * 100).toFixed(1)}% of trajectory variance. ` +
        `Literal DR8 boundaries u_1 = N(2x) and s_t = N(F_theta(u_t)) are never elided.`,
      tone: 'swa_cyan',
      worldPos: Object.freeze([pcaCenterX, 0.2, pcaCenterZ] as const),
      badge: 'PROJECTED NORMS <= 1; 8D METRICS',
      discrepancyIds: Object.freeze(['DR8'] as const),
    }),
    Object.freeze({
      id: 'callout_anchor_jacobian_caveat',
      title: 'Local 120-deg Anchor-Map Jacobian Caveat',
      body:
        `${DORY_ANCHOR_MAP_CAVEAT_BADGE}. At Loop t=${activeLoopMetric.loopIteration}, ` +
        `theta = ${activeJac.angleDegrees.toFixed(2)} deg gives eps-stabilized boundary-map singular value ` +
        `1 / sqrt(4 cos^2(theta/2) + 1e-6) = ${activeJac.epsStabilizedJacobianNorm.toFixed(4)} ` +
        `(${activeJac.isLocalContractiveRegime ? '< 1 local boundary regime' : '>= 1 expansive boundary regime'}; capped at 1000 at 180 deg). ` +
        `The 40-sublayer shared recurrent core F_theta has NO unit-Lipschitz guarantee.`,
      tone: 'warn',
      worldPos: sheetTrajectory.frame.center,
      badge: 'LOCAL BOUNDARY MAP ONLY',
    }),
  ];

  if (counterfactualCoda !== null && sheetCounterfactual !== null) {
    callouts.push(
      Object.freeze({
        id: 'callout_counterfactual_coda',
        title: DORY_COUNTERFACTUAL_CODA_BADGE,
        body:
          `Read-only Deep Coda evaluated on intermediate state s_${counterfactualCoda.loopIteration} at pos ${counterfactualCoda.absTokenPos} ` +
          `against committed historical Deep Coda KV [0..${Math.max(0, counterfactualCoda.absTokenPos - 1)}] with writeKv=false (ZERO KV or session mutation). ` +
          `D_KL(p_final || p_cf) = ${counterfactualCoda.klFromFinalLoopNats.toFixed(4)} nats; ` +
          `argmax "${decodeDoryToken(counterfactualCoda.argMaxTokenId)}" ` +
          `${counterfactualCoda.argMaxChanged ? `differs from final-loop "${decodeDoryToken(counterfactualCoda.finalLoopArgMaxTokenId)}"` : 'matches final-loop argmax'}.`,
        tone: 'gold',
        worldPos: sheetCounterfactual.frame.center,
        badge: 'READ-ONLY · ZERO KV MUTATION',
      }),
    );
  }

  // 9. Active Cell Popover & World Fit Bounds
  const activeCellPopover = buildHoveredCellPopover(
    options.hoveredCell,
    sheetsByPartId,
    spanningSubspace,
    exploratoryPca,
    counterfactualCoda,
    tokenLabels,
    counterfactualTokenIds,
    finalLoopProbsT1,
  );

  const stageBoundaryAabb = computeBoundsFromPoints(
    [
      [-12.6, -0.1, -leftFrameZSpan],
      [16.2, 1.4, leftFrameZSpan],
    ],
    0.4,
  );
  const fitBounds = mergeAabbs([...aabbs, stageBoundaryAabb]);

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flatFlow.pack();

  const hypersphereMetadata: DoryHypersphereViewMetadata = Object.freeze({
    antiSlerpPlaque: DORY_ANTI_SLERP_PLAQUE,
    antiSlerpMarker: DORY_NLERP_MARKER,
    anchorMapCaveatBadge: DORY_ANCHOR_MAP_CAVEAT_BADGE,
    selectedPhysicalLayerIdx,
    selectedPairedLayerIdx: layerSpec.pairedLayerIdx,
    selectedLoopIteration,
    selectedTokenOffset,
    selectedAbsTokenPos,
    selectedTokenId,
    selectedTokenText,
    loopMetrics8d,
    spanningSubspace,
    exploratoryPca,
    counterfactualCoda,
  });

  const loopText =
    selectedStage === 'recurrent'
      ? `Loop ${selectedLoopIteration}/${ctx.kLoops}`
      : `${selectedStage} (1x)`;

  const scene: DorySpatialScene = Object.freeze({
    viewMode: 'hypersphere',
    title: VIEW_DESCRIPTOR.label,
    subtitle: `Phys L${selectedPhysicalLayerIdx} (${layerSpec.transformersPairBadge}) · ${loopText} · pos ${selectedAbsTokenPos} "${selectedTokenText}" · nGPT NLERP (NOT SLERP)`,
    modelWeightId: ctx.params.modelWeightId,
    branchId: ctx.branch.branchId,
    recordId: ctx.record.recordId,
    kLoops: ctx.kLoops,
    kvCacheMode: ctx.kvCacheMode,
    executionMode: ctx.executionMode,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
    solidData: packedSolids.data,
    solidCount: packedSolids.count,
    lineData: packedLines.data,
    lineCount: packedLines.count,
    ribbonData: packedRibbons.data,
    ribbonCount: packedRibbons.count,
    sheetData: packedSheets.sheetData,
    sheetBasisData: packedSheets.sheetBasisData,
    sheetRevealData: packedSheets.sheetRevealData,
    sheetCount: packedSheets.sheetCount,
    flowVertices: packedFlow.flowVertices,
    flowIndices: packedFlow.flowIndices,
    pickBoxes: Object.freeze(pickBoxes),
    cameraPreset: Object.freeze({
      azimuthDeg: VIEW_DESCRIPTOR.defaultCameraAzimuthDeg,
      elevationDeg: VIEW_DESCRIPTOR.defaultCameraElevationDeg,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: packedAtlas.valueAtlas,
    atlasKey: packedAtlas.atlasKey,
    atlasRegions: packedAtlas.atlasRegions,
    atlasAudit: packedAtlas.atlasAudit,
    provenanceBadges: Object.freeze(provenanceBadges),
    callouts: Object.freeze(callouts),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    activeCellGrids: Object.freeze(activeCellGrids),
    activeCellPopover,
    hypersphereMetadata,
  });

  // Self-audit geometry finiteness, atlas budget/non-overlap, and DR20 compliance before returning
  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `View 2 hypersphereScene validation failed: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
