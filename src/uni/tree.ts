/**
 * BalancedSemanticTree, Progressive Self-Refinement Math & U26 Diagnostic (`src/uni/tree.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching the pinned
 * Nemotron-Labs-Diffusion-Uni UNI3 Slice 3A contracts (`UNI0_ARCHITECTURE_AUDIT.md`
 * §5.1, §6 `U9–U12`, `U23`, `U26`, §8.1, §8.3A, §10.1):
 *   - Deterministic `recursive_centered_pca_rank_median_v1` binary ball tree using
 *     FP64 row-normalization, exact `normalized_squared_l2_tree_ultrametric` internal
 *     node radii (monotone non-increasing along every root-to-leaf path), and
 *     deterministic Jacobi symmetric eigendecomposition (zero random initialization).
 *   - Exact `3,068 B` core typed-array footprint for `V = 256` (`1,024 + 1,024 + 1,020 B`).
 *   - Top-3 deterministic PCA codebook projection (`projectCodebookPca3d`) for View 4.
 *   - Reversed inference code-stage polarity (`U9`), additive continuous mask embedding
 *     (`U10`), bitwise O(1) subtree representative sampling (`sampleRepresentatives`),
 *     shifted activation quotas + `N - 1` flush rounds (`U12`), deterministic quadtree
 *     stratified order (`stratifiedRandom`), Paper Eq. 16 continuous-time `agesAtTime`,
 *     strictly immutable state initialization/advancement, and dedicated `U26`
 *     prime-depth `D = 17` train-vs-inference floor-rounding mismatch diagnostic.
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG calls, or sibling-app imports.
 */

import { createXoshiro128ss } from './prng';
import {
  UNI_MINI_CONFIG,
  UNI_MINI_GEN_IMAGE_END_ID,
  UNI_MINI_GEN_IMAGE_MASK_ID,
  UNI_MINI_GEN_ROW_MARKER_ID,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_SEMANTIC_TREE_ALGORITHM,
  UNI_SEMANTIC_TREE_METRIC,
  UNI_STRATIFIED_DEFAULT_SEED,
  type UniBalancedSemanticTree,
  type UniPrimeDepthFloorMismatchReport,
  type UniPrimeDepthFloorMismatchRow,
  type UniSemanticTreeSchedule,
  type UniSemanticTreeState,
  type UniStagePolarityPair,
  type UniTreeBallMembership,
  type UniXoshiro128ssGenerator,
} from './types';

// ============================================================================
// DETERMINISTIC FP64 NORMALIZATION & JACOBI SYMMETRIC EIGENDECOMPOSITION
// ============================================================================

const ZERO_COVARIANCE_EPS = 1e-24;
const JACOBI_CONVERGENCE_EPS = 1e-14;

interface SymmetricEigenResult {
  readonly eigenvalues: Float64Array;
  /** Descending-eigenvalue sorted unit eigenvectors, each canonicalized in sign. */
  readonly eigenvectors: readonly Float64Array[];
}

function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
}

/**
 * Row-normalize `[vocabSize, latentDim]` codebook embeddings in FP64 to unit L2 norm:
 * `u[i, d] = e[i, d] / max(1e-12, ||e[i]||_2)`.
 */
function normalizeCodebookRowsFp64(
  codebook: Float32Array,
  vocabSize: number,
  latentDim: number,
): Float64Array {
  if (!Number.isInteger(vocabSize) || vocabSize < 1) {
    throw new RangeError(`vocabSize must be an integer >= 1, got ${vocabSize}`);
  }
  if (!Number.isInteger(latentDim) || latentDim < 1) {
    throw new RangeError(`latentDim must be an integer >= 1, got ${latentDim}`);
  }
  if (codebook.length !== vocabSize * latentDim) {
    throw new RangeError(
      `codebook length (${codebook.length}) must equal vocabSize * latentDim (${vocabSize * latentDim})`,
    );
  }

  const normalized = new Float64Array(vocabSize * latentDim);
  for (let i = 0; i < vocabSize; i++) {
    const base = i * latentDim;
    let sumSq = 0.0;
    for (let d = 0; d < latentDim; d++) {
      const val = codebook[base + d];
      sumSq += val * val;
    }
    const norm = Math.max(1e-12, Math.sqrt(sumSq));
    const invNorm = 1.0 / norm;
    for (let d = 0; d < latentDim; d++) {
      normalized[base + d] = codebook[base + d] * invNorm;
    }
  }
  return normalized;
}

/**
 * Canonicalize eigenvector sign so the component with largest `|v[d]|`
 * (tie-breaking to smallest `d`) is strictly `> 0`.
 */
function canonicalizeEigenvectorSign(vec: Float64Array): void {
  let maxAbs = -1.0;
  let maxIdx = 0;
  for (let d = 0; d < vec.length; d++) {
    const absVal = Math.abs(vec[d]);
    if (absVal > maxAbs) {
      maxAbs = absVal;
      maxIdx = d;
    }
  }
  if (vec[maxIdx] < 0.0) {
    for (let d = 0; d < vec.length; d++) {
      vec[d] = -vec[d];
    }
  }
  // Eliminate signed -0.0 for deterministic comparisons
  for (let d = 0; d < vec.length; d++) {
    if (vec[d] === 0) {
      vec[d] = 0;
    }
  }
}

/**
 * Deterministic Jacobi symmetric eigendecomposition for a `dim x dim` symmetric FP64 matrix.
 * Uses zero random initialization; selects the largest-magnitude off-diagonal element
 * `(p < q, tie-break smallest p then smallest q)` on each rotation until convergence.
 */
function jacobiSymmetricEigen(matrix: Float64Array, dim: number): SymmetricEigenResult {
  let maxEntry = 0.0;
  for (let i = 0; i < matrix.length; i++) {
    const a = Math.abs(matrix[i]);
    if (a > maxEntry) {
      maxEntry = a;
    }
  }

  // Degenerate zero-covariance fallback: deterministic standard basis `[e_0, e_1, ...]`
  if (maxEntry <= ZERO_COVARIANCE_EPS) {
    const eigenvalues = new Float64Array(dim);
    const eigenvectors: Float64Array[] = [];
    for (let j = 0; j < dim; j++) {
      const basis = new Float64Array(dim);
      basis[j] = 1.0;
      eigenvectors.push(basis);
    }
    return { eigenvalues, eigenvectors };
  }

  const m = new Float64Array(matrix);
  const v = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    v[i * dim + i] = 1.0;
  }

  const maxIterations = Math.max(200, 50 * dim * dim);
  const tol = JACOBI_CONVERGENCE_EPS * Math.max(1.0, maxEntry);

  for (let iter = 0; iter < maxIterations; iter++) {
    let p = 0;
    let q = 1;
    let maxOffDiag = 0.0;

    for (let r = 0; r < dim; r++) {
      const rBase = r * dim;
      for (let c = r + 1; c < dim; c++) {
        const absVal = Math.abs(m[rBase + c]);
        if (absVal > maxOffDiag) {
          maxOffDiag = absVal;
          p = r;
          q = c;
        }
      }
    }

    if (maxOffDiag <= tol) {
      break;
    }

    const app = m[p * dim + p];
    const aqq = m[q * dim + q];
    const apq = m[p * dim + q];

    const tau = (aqq - app) / (2.0 * apq);
    const t =
      tau >= 0.0
        ? 1.0 / (tau + Math.hypot(1.0, tau))
        : -1.0 / (-tau + Math.hypot(1.0, tau));
    const c = 1.0 / Math.hypot(1.0, t);
    const s = t * c;

    for (let k = 0; k < dim; k++) {
      if (k === p || k === q) {
        continue;
      }
      const mkp = m[k * dim + p];
      const mkq = m[k * dim + q];
      const newP = c * mkp - s * mkq;
      const newQ = s * mkp + c * mkq;
      m[k * dim + p] = newP;
      m[p * dim + k] = newP;
      m[k * dim + q] = newQ;
      m[q * dim + k] = newQ;
    }

    m[p * dim + p] = app - t * apq;
    m[q * dim + q] = aqq + t * apq;
    m[p * dim + q] = 0.0;
    m[q * dim + p] = 0.0;

    for (let k = 0; k < dim; k++) {
      const vkp = v[k * dim + p];
      const vkq = v[k * dim + q];
      v[k * dim + p] = c * vkp - s * vkq;
      v[k * dim + q] = s * vkp + c * vkq;
    }
  }

  const order: number[] = [];
  for (let j = 0; j < dim; j++) {
    order.push(j);
  }
  order.sort((a, b) => {
    const diff = m[b * dim + b] - m[a * dim + a];
    return diff !== 0 ? diff : a - b;
  });

  const eigenvalues = new Float64Array(dim);
  const eigenvectors: Float64Array[] = [];
  for (let rank = 0; rank < dim; rank++) {
    const col = order[rank];
    eigenvalues[rank] = m[col * dim + col];
    const vec = new Float64Array(dim);
    for (let r = 0; r < dim; r++) {
      vec[r] = v[r * dim + col];
    }
    canonicalizeEigenvectorSign(vec);
    eigenvectors.push(vec);
  }

  return { eigenvalues, eigenvectors };
}

/**
 * Compute subset centroid `mu[dim]` and centered covariance `C[dim, dim]` in FP64.
 */
function computeSubsetCentroidAndCovariance(
  normalized: Float64Array,
  indices: readonly number[],
  latentDim: number,
  centroidOut: Float64Array,
  covOut: Float64Array,
): void {
  centroidOut.fill(0.0);
  const count = indices.length;
  for (let i = 0; i < count; i++) {
    const base = indices[i] * latentDim;
    for (let d = 0; d < latentDim; d++) {
      centroidOut[d] += normalized[base + d];
    }
  }
  const invCount = 1.0 / count;
  for (let d = 0; d < latentDim; d++) {
    centroidOut[d] *= invCount;
  }

  covOut.fill(0.0);
  for (let i = 0; i < count; i++) {
    const base = indices[i] * latentDim;
    for (let r = 0; r < latentDim; r++) {
      const dr = normalized[base + r] - centroidOut[r];
      const rowOffset = r * latentDim;
      for (let c = r; c < latentDim; c++) {
        covOut[rowOffset + c] += dr * (normalized[base + c] - centroidOut[c]);
      }
    }
  }
  for (let r = 0; r < latentDim; r++) {
    for (let c = 0; c < r; c++) {
      covOut[r * latentDim + c] = covOut[c * latentDim + r];
    }
  }
}

// ============================================================================
// 1. BALANCED SEMANTIC TREE (`recursive_centered_pca_rank_median_v1`)
// ============================================================================

/**
 * Build a balanced binary semantic tree over a VQ codebook using deterministic
 * top-down centered-PCA rank-median splitting (`recursive_centered_pca_rank_median_v1`)
 * and exact squared-L2 diameter internal-node radii (`normalized_squared_l2_tree_ultrametric`).
 *
 * For `vocabSize = 256, latentDim = 8`, produces exact `3,068 B` typed-array footprint
 * (`leafToToken`: `1,024 B`, `tokenToLeaf`: `1,024 B`, `heapRadii`: `1,020 B`).
 */
export function buildBalancedSemanticTree(
  codebook: Float32Array,
  vocabSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
  latentDim: number = UNI_MINI_CONFIG.vqLatentDim,
): UniBalancedSemanticTree {
  if (!isPowerOfTwo(vocabSize) || vocabSize < 2) {
    throw new RangeError(
      `vocabSize must be a power of two >= 2, got ${vocabSize}`,
    );
  }
  if (!Number.isInteger(latentDim) || latentDim < 1) {
    throw new RangeError(`latentDim must be an integer >= 1, got ${latentDim}`);
  }
  if (codebook.length !== vocabSize * latentDim) {
    throw new RangeError(
      `codebook length (${codebook.length}) must equal vocabSize * latentDim (${vocabSize * latentDim})`,
    );
  }

  const depth = Math.round(Math.log2(vocabSize));
  const normalized = normalizeCodebookRowsFp64(codebook, vocabSize, latentDim);

  const leafToToken = new Int32Array(vocabSize);
  const tokenToLeaf = new Int32Array(vocabSize);
  const heapRadii = new Float32Array(vocabSize - 1);

  const centroid = new Float64Array(latentDim);
  const covariance = new Float64Array(latentDim * latentDim);
  const projections = new Float64Array(vocabSize);

  let leafCursor = 0;

  function splitNode(heapIdx: number, indices: number[]): void {
    const count = indices.length;
    if (count === 1) {
      const tokenId = indices[0];
      leafToToken[leafCursor] = tokenId;
      tokenToLeaf[tokenId] = leafCursor;
      leafCursor++;
      return;
    }

    // Exact squared-L2 subset diameter in FP64 (guarantees parent >= child monotonicity)
    let maxDistSq = 0.0;
    for (let a = 0; a < count; a++) {
      const baseA = indices[a] * latentDim;
      for (let b = a + 1; b < count; b++) {
        const baseB = indices[b] * latentDim;
        let distSq = 0.0;
        for (let d = 0; d < latentDim; d++) {
          const diff = normalized[baseA + d] - normalized[baseB + d];
          distSq += diff * diff;
        }
        if (distSq > maxDistSq) {
          maxDistSq = distSq;
        }
      }
    }
    heapRadii[heapIdx] = Math.fround(maxDistSq);

    computeSubsetCentroidAndCovariance(
      normalized,
      indices,
      latentDim,
      centroid,
      covariance,
    );

    const { eigenvectors } = jacobiSymmetricEigen(covariance, latentDim);
    const principal = eigenvectors[0];

    for (let i = 0; i < count; i++) {
      const tokenId = indices[i];
      const base = tokenId * latentDim;
      let proj = 0.0;
      for (let d = 0; d < latentDim; d++) {
        proj += (normalized[base + d] - centroid[d]) * principal[d];
      }
      projections[tokenId] = proj;
    }

    indices.sort((a, b) => {
      const diff = projections[a] - projections[b];
      return diff !== 0 ? diff : a - b;
    });

    const half = count >>> 1;
    splitNode(2 * heapIdx + 1, indices.slice(0, half));
    splitNode(2 * heapIdx + 2, indices.slice(half));
  }

  const rootIndices: number[] = [];
  for (let i = 0; i < vocabSize; i++) {
    rootIndices.push(i);
  }
  splitNode(0, rootIndices);

  const byteLength =
    leafToToken.byteLength + tokenToLeaf.byteLength + heapRadii.byteLength;

  return Object.freeze({
    algorithm: UNI_SEMANTIC_TREE_ALGORITHM,
    metric: UNI_SEMANTIC_TREE_METRIC,
    vocabSize,
    depth,
    latentDim,
    leafToToken,
    tokenToLeaf,
    heapRadii,
    byteLength,
  });
}

// ============================================================================
// 2. DETERMINISTIC 3D PCA CODEBOOK PROJECTION (View 4 Ball-Tree Inspector)
// ============================================================================

/**
 * Project the unit-normalized centered VQ codebook onto its top-3 deterministic
 * Jacobi PCA axes, returning row-major `[vocabSize * 3]` Float32 coordinates.
 */
export function projectCodebookPca3d(
  codebook: Float32Array,
  vocabSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
  latentDim: number = UNI_MINI_CONFIG.vqLatentDim,
): Float32Array {
  const normalized = normalizeCodebookRowsFp64(codebook, vocabSize, latentDim);

  const allIndices: number[] = [];
  for (let i = 0; i < vocabSize; i++) {
    allIndices.push(i);
  }

  const centroid = new Float64Array(latentDim);
  const covariance = new Float64Array(latentDim * latentDim);
  computeSubsetCentroidAndCovariance(
    normalized,
    allIndices,
    latentDim,
    centroid,
    covariance,
  );

  const { eigenvectors } = jacobiSymmetricEigen(covariance, latentDim);
  const numAxes = Math.min(3, latentDim);
  const out = new Float32Array(vocabSize * 3);

  for (let i = 0; i < vocabSize; i++) {
    const srcBase = i * latentDim;
    const dstBase = i * 3;
    for (let axis = 0; axis < numAxes; axis++) {
      const vec = eigenvectors[axis];
      let dot = 0.0;
      for (let d = 0; d < latentDim; d++) {
        dot += (normalized[srcBase + d] - centroid[d]) * vec[d];
      }
      out[dstBase + axis] = Math.fround(dot);
    }
  }

  return out;
}

// ============================================================================
// 3 & 4. REVERSED CODE-STAGE POLARITY & LEVEL / NOISE-WEIGHT MAPPING (`U9`)
// ============================================================================

/**
 * Map reversed inference code stages (`-1` unactivated, `0..N-1` refining, `N` committed)
 * to binary-tree split levels, additive mask-noise weights `eta`, and candidate ball sizes.
 */
export function semanticTreeLevelsForAges(
  stages: ArrayLike<number>,
  depth: number,
  refinementSteps: number,
): { levels: Int32Array; noiseWeights: Float32Array; ballSizes: Int32Array } {
  if (!Number.isInteger(depth) || depth < 1 || depth > 30) {
    throw new RangeError(`depth must be an integer in [1, 30], got ${depth}`);
  }
  if (!Number.isInteger(refinementSteps) || refinementSteps < 1) {
    throw new RangeError(
      `refinementSteps must be an integer >= 1, got ${refinementSteps}`,
    );
  }

  const len = stages.length;
  const levels = new Int32Array(len);
  const noiseWeights = new Float32Array(len);
  const ballSizes = new Int32Array(len);
  const fullBall = 1 << depth;

  for (let i = 0; i < len; i++) {
    const stage = stages[i];
    if (stage < 0) {
      levels[i] = 0;
      noiseWeights[i] = 1.0;
      ballSizes[i] = fullBall;
    } else if (stage >= refinementSteps) {
      levels[i] = depth;
      noiseWeights[i] = 0.0;
      ballSizes[i] = 1;
    } else {
      const level = Math.floor((stage * depth) / refinementSteps);
      levels[i] = level;
      noiseWeights[i] = Math.fround(1.0 - stage / refinementSteps);
      ballSizes[i] = 1 << (depth - level);
    }
  }

  return { levels, noiseWeights, ballSizes };
}

/**
 * U9 dual polarity readout pairing inference code stage `{-1, 0..N}` with
 * paper age `paperAge = max(0, N - max(0, codeStage))`.
 */
export function getStagePolarityPair(
  codeStage: number,
  depth: number,
  refinementSteps: number,
): UniStagePolarityPair {
  const { levels, noiseWeights, ballSizes } = semanticTreeLevelsForAges(
    [codeStage],
    depth,
    refinementSteps,
  );
  const clampedStage = Math.max(0, codeStage);
  const paperAge = Math.max(0, refinementSteps - clampedStage);

  return Object.freeze({
    codeStage,
    paperAge,
    level: levels[0],
    ballSize: ballSizes[0],
    noiseWeight: noiseWeights[0],
    isCommitted: codeStage >= refinementSteps,
  });
}

// ============================================================================
// 5. TREE BALL MEMBERSHIP & 6. BITWISE O(1) REPRESENTATIVE SAMPLING
// ============================================================================

/**
 * Inspect the exact subtree ball containing `tokenId` at split depth `level`.
 */
export function getTreeBallMembership(
  tree: UniBalancedSemanticTree,
  tokenId: number,
  level: number,
): UniTreeBallMembership {
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= tree.vocabSize) {
    throw new RangeError(
      `tokenId must be an integer in [0, ${tree.vocabSize - 1}], got ${tokenId}`,
    );
  }
  if (!Number.isInteger(level) || level < 0 || level > tree.depth) {
    throw new RangeError(
      `level must be an integer in [0, ${tree.depth}], got ${level}`,
    );
  }

  const leafIndex = tree.tokenToLeaf[tokenId];
  const shift = tree.depth - level;
  const ballStartLeaf = (leafIndex >>> shift) << shift;
  const ballSize = 1 << shift;
  const heapNodeIndex =
    level < tree.depth ? ((1 << level) - 1) + (leafIndex >>> shift) : null;
  const radius = heapNodeIndex !== null ? tree.heapRadii[heapNodeIndex] : 0.0;

  const memberTokenIds = new Int32Array(ballSize);
  for (let k = 0; k < ballSize; k++) {
    memberTokenIds[k] = tree.leafToToken[ballStartLeaf + k];
  }

  return Object.freeze({
    tokenId,
    leafIndex,
    level,
    shift,
    ballStartLeaf,
    ballSize,
    heapNodeIndex,
    radius,
    memberTokenIds,
  });
}

/**
 * Bitwise O(1) subtree representative sampling (`tree.py:210-226`).
 * When `level >= tree.depth` (`shift <= 0`), returns `proposals[i]` identically.
 * Otherwise masks `randBits[i]` to the lower `shift = depth - level` bits inside
 * `proposals[i]`'s aligned subtree leaf block.
 */
export function sampleRepresentatives(
  tree: UniBalancedSemanticTree,
  proposals: ArrayLike<number>,
  levels: ArrayLike<number>,
  randBits: ArrayLike<number>,
): Int32Array {
  const len = proposals.length;
  if (levels.length !== len || randBits.length !== len) {
    throw new RangeError(
      `proposals (${len}), levels (${levels.length}), and randBits (${randBits.length}) must have identical lengths`,
    );
  }

  const out = new Int32Array(len);
  const { depth, vocabSize, tokenToLeaf, leafToToken } = tree;

  for (let i = 0; i < len; i++) {
    const proposal = proposals[i];
    if (!Number.isInteger(proposal) || proposal < 0 || proposal >= vocabSize) {
      throw new RangeError(
        `proposals[${i}] must be an integer in [0, ${vocabSize - 1}], got ${proposal}`,
      );
    }
    const level = levels[i];
    if (!Number.isInteger(level) || level < 0 || level > depth) {
      throw new RangeError(
        `levels[${i}] must be an integer in [0, ${depth}], got ${level}`,
      );
    }

    const shift = depth - level;
    if (shift <= 0) {
      out[i] = proposal;
    } else {
      const leaf = tokenToLeaf[proposal];
      const start = (leaf >>> shift) << shift;
      const mask = (1 << shift) - 1;
      const leafPrime = start + ((randBits[i] >>> 0) & mask);
      out[i] = leafToToken[leafPrime];
    }
  }

  return out;
}

// ============================================================================
// 7. ADDITIVE CONTINUOUS MASK EMBEDDINGS (`U10`)
// ============================================================================

/**
 * Compute progressive self-refining generation embeddings (`U10`, `tree.py:351-382`):
 *   - `eta === 0`: `genEmbedding[rep]`
 *   - `eta > 0`:   `f32(genEmbedding[rep] + f32(eta * genEmbedding[maskRowId]))`
 */
export function buildSemanticTreeEmbeddings(
  representatives: ArrayLike<number>,
  noiseWeights: ArrayLike<number>,
  genEmbedding: Float32Array,
  hiddenSize: number = UNI_MINI_CONFIG.hiddenSize,
  maskRowId: number = UNI_MINI_GEN_IMAGE_MASK_ID,
): Float32Array {
  const len = representatives.length;
  if (noiseWeights.length !== len) {
    throw new RangeError(
      `representatives (${len}) and noiseWeights (${noiseWeights.length}) must have identical lengths`,
    );
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new RangeError(`hiddenSize must be an integer >= 1, got ${hiddenSize}`);
  }
  if (!Number.isInteger(maskRowId) || maskRowId < 0) {
    throw new RangeError(`maskRowId must be an integer >= 0, got ${maskRowId}`);
  }
  const maskBase = maskRowId * hiddenSize;
  if (maskBase + hiddenSize > genEmbedding.length) {
    throw new RangeError(
      `genEmbedding length (${genEmbedding.length}) too small for maskRowId=${maskRowId}, hiddenSize=${hiddenSize}`,
    );
  }

  const out = new Float32Array(len * hiddenSize);
  for (let i = 0; i < len; i++) {
    const rep = representatives[i];
    const repBase = rep * hiddenSize;
    if (!Number.isInteger(rep) || rep < 0 || repBase + hiddenSize > genEmbedding.length) {
      throw new RangeError(
        `representatives[${i}] (${rep}) out of bounds for genEmbedding (${genEmbedding.length} / ${hiddenSize})`,
      );
    }

    const eta = Math.fround(noiseWeights[i]);
    const dstBase = i * hiddenSize;
    if (eta === 0) {
      for (let c = 0; c < hiddenSize; c++) {
        out[dstBase + c] = genEmbedding[repBase + c];
      }
    } else {
      for (let c = 0; c < hiddenSize; c++) {
        out[dstBase + c] = Math.fround(
          genEmbedding[repBase + c] + Math.fround(eta * genEmbedding[maskBase + c]),
        );
      }
    }
  }

  return out;
}

// ============================================================================
// 8. SHIFTED ACTIVATION QUOTAS & FLUSH ROUNDS (`U12`)
// ============================================================================

/**
 * Compute deterministic shifted activation counts (`shift = 5.0`) + trailing
 * flush rounds `(refinementSteps - 1) + refinementSteps * refinementGap` (`U12`).
 */
export function semanticTreeActivationCounts(
  contentCount: number,
  steps: number,
  refinementSteps: number,
  shift: number = 5.0,
  refinementGap: number = 0,
): UniSemanticTreeSchedule {
  if (!Number.isInteger(contentCount) || contentCount < 1) {
    throw new RangeError(`contentCount must be an integer >= 1, got ${contentCount}`);
  }
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError(`steps must be an integer >= 1, got ${steps}`);
  }
  if (!Number.isInteger(refinementSteps) || refinementSteps < 1) {
    throw new RangeError(
      `refinementSteps must be an integer >= 1, got ${refinementSteps}`,
    );
  }
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new RangeError(`shift must be a finite number > 0, got ${shift}`);
  }
  if (!Number.isInteger(refinementGap) || refinementGap < 0) {
    throw new RangeError(
      `refinementGap must be an integer >= 0, got ${refinementGap}`,
    );
  }

  const activationSteps = Math.min(steps, contentCount);
  const boundaries = new Int32Array(activationSteps + 1);
  for (let m = 0; m <= activationSteps; m++) {
    if (m === activationSteps) {
      boundaries[m] = contentCount;
    } else {
      const t = m / activationSteps;
      const sigma = (shift * t) / (1.0 + (shift - 1.0) * t);
      boundaries[m] = Math.floor(sigma * contentCount);
    }
  }

  const rawCounts = new Int32Array(activationSteps);
  let sum = 0;
  for (let m = 0; m < activationSteps; m++) {
    const q = Math.max(1, boundaries[m + 1] - boundaries[m]);
    rawCounts[m] = q;
    sum += q;
  }

  // Round-robin decrement excess wherever q[idx] > 1 until exact sum === contentCount
  let idx = 0;
  while (sum > contentCount) {
    if (rawCounts[idx] > 1) {
      rawCounts[idx]--;
      sum--;
    }
    idx = (idx + 1) % activationSteps;
  }

  // Reverse (`counts.flip(0)`)
  rawCounts.reverse();

  const flushRounds = (refinementSteps - 1) + refinementSteps * refinementGap;
  const totalRounds = activationSteps + flushRounds;
  const activationCounts = new Int32Array(totalRounds);
  const cumulativeActivations = new Int32Array(totalRounds);

  let running = 0;
  for (let r = 0; r < totalRounds; r++) {
    const count = r < activationSteps ? rawCounts[r] : 0;
    activationCounts[r] = count;
    running += count;
    cumulativeActivations[r] = running;
  }

  return Object.freeze({
    contentCount,
    activationSteps,
    refinementSteps,
    refinementGap,
    shift,
    flushRounds,
    totalRounds,
    activationCounts,
    cumulativeActivations,
  });
}

// ============================================================================
// 9. DETERMINISTIC QUADTREE STRATIFIED SPATIAL ORDER (`U12`, `U23`)
// ============================================================================

function fisherYatesShuffle(
  arr: number[],
  rng: UniXoshiro128ssGenerator,
): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.nextUint32() % (i + 1);
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/**
 * Deterministic recursive quadtree stratified permutation over an `n x n` power-of-two
 * grid using `createXoshiro128ss(seed)` from `./prng` (`U12`, `U23`).
 *
 * Guarantees for `n >= 2`: the first 4 entries hit all 4 top-level `(n/2) x (n/2)`
 * quadrants (1 per quadrant), and the output is a valid bijection over `0 .. n*n - 1`.
 */
export function stratifiedRandom(
  n: number = UNI_MINI_CONFIG.vqGridSize,
  seed: number = UNI_STRATIFIED_DEFAULT_SEED,
): Int32Array {
  if (!isPowerOfTwo(n)) {
    throw new RangeError(`n must be a power of two >= 1, got ${n}`);
  }

  const rng = createXoshiro128ss(seed);
  const total = n * n;
  const levels = Math.round(Math.log2(n));
  const occupied = new Uint8Array(total);
  const order = new Int32Array(total);
  let cursor = 0;

  for (let k = 0; k < levels; k++) {
    const blocksPerSide = 1 << k;
    const numBlocks = blocksPerSide * blocksPerSide;
    const childSize = n >> (k + 1);
    const blockSize = childSize * 2;
    const childArea = childSize * childSize;

    const blockOrder: number[] = [];
    for (let b = 0; b < numBlocks; b++) {
      blockOrder.push(b);
    }
    fisherYatesShuffle(blockOrder, rng);

    for (let bi = 0; bi < numBlocks; bi++) {
      const bIdx = blockOrder[bi];
      const br = Math.floor(bIdx / blocksPerSide);
      const bc = bIdx % blocksPerSide;

      const quadOrder = [0, 1, 2, 3];
      fisherYatesShuffle(quadOrder, rng);

      for (let qi = 0; qi < 4; qi++) {
        const q = quadOrder[qi];
        const qr = q >> 1;
        const qc = q & 1;
        const r0 = br * blockSize + qr * childSize;
        const c0 = bc * blockSize + qc * childSize;

        let hasOccupied = false;
        for (let dr = 0; dr < childSize && !hasOccupied; dr++) {
          const rowBase = (r0 + dr) * n + c0;
          for (let dc = 0; dc < childSize; dc++) {
            if (occupied[rowBase + dc] === 1) {
              hasOccupied = true;
              break;
            }
          }
        }

        if (!hasOccupied) {
          const pick = rng.nextUint32() % childArea;
          const dr = Math.floor(pick / childSize);
          const dc = pick % childSize;
          const flat = (r0 + dr) * n + (c0 + dc);
          occupied[flat] = 1;
          order[cursor++] = flat;
        }
      }
    }
  }

  if (cursor < total) {
    const remaining: number[] = [];
    for (let flat = 0; flat < total; flat++) {
      if (occupied[flat] === 0) {
        remaining.push(flat);
      }
    }
    fisherYatesShuffle(remaining, rng);
    for (let i = 0; i < remaining.length; i++) {
      order[cursor++] = remaining[i];
    }
  }

  return order;
}

// ============================================================================
// 10. PAPER EQ. 16 CONTINUOUS-TIME PER-TOKEN INTERVAL AGES
// ============================================================================

/**
 * Exact Paper Eq. 16 (`appendix.tex:169-217`) continuous-time paper age schedule:
 *   - `0` when `t <= tau[i]` (clean)
 *   - `totalTransitionsK` when `t >= tau[i] + epsilon` (terminal mask)
 *   - `Math.floor((totalTransitionsK * (t - tau[i])) / epsilon)` otherwise.
 */
export function agesAtTime(
  t: number,
  tau: ArrayLike<number>,
  epsilon: number,
  totalTransitionsK: number,
): Int32Array {
  if (!Number.isFinite(t)) {
    throw new RangeError(`t must be finite, got ${t}`);
  }
  if (!Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError(`epsilon must be a finite number > 0, got ${epsilon}`);
  }
  if (!Number.isInteger(totalTransitionsK) || totalTransitionsK < 1) {
    throw new RangeError(
      `totalTransitionsK must be an integer >= 1, got ${totalTransitionsK}`,
    );
  }

  const len = tau.length;
  const out = new Int32Array(len);
  for (let i = 0; i < len; i++) {
    const ti = tau[i];
    if (t <= ti) {
      out[i] = 0;
    } else if (t >= ti + epsilon) {
      out[i] = totalTransitionsK;
    } else {
      out[i] = Math.floor((totalTransitionsK * (t - ti)) / epsilon);
    }
  }
  return out;
}

// ============================================================================
// 11. IMMUTABLE SEMANTIC TREE STATE INIT & ADVANCEMENT
// ============================================================================

export interface InitSemanticTreeStateOptions {
  readonly rows?: number;
  readonly contentCols?: number;
  readonly includeMarkers?: boolean;
  readonly depth?: number;
  readonly refinementSteps?: number;
  /** Optional `[rows * contentCols]` initial root representative VQ codes (`0..255`). */
  readonly initialContentCodes?: ArrayLike<number>;
  /** Seed used when `initialContentCodes` is omitted (default `UNI_STRATIFIED_DEFAULT_SEED`). */
  readonly seed?: number;
  /** Optional `[rows * contentCols]` binary lock mask (`1` = locked source padding cell). */
  readonly lockedContentMask?: ArrayLike<number>;
  /** Optional `[rows * contentCols]` clean source VQ codes for locked padding cells. */
  readonly lockedSourceCodes?: ArrayLike<number>;
}

/**
 * Map a content-grid index `c in 0..rows*contentCols-1` to its marked-grid index
 * `r * markedCols + col` (where `contentCols = markedCols - 2`).
 */
export function contentIndexToMarkedGridIndex(
  contentIndex: number,
  contentCols: number = UNI_MINI_CONFIG.vqGridSize,
  markedCols: number = UNI_MINI_CONFIG.vqMarkedGridCols,
): number {
  const r = Math.floor(contentIndex / contentCols);
  const col = contentIndex % contentCols;
  return r * markedCols + col;
}

/**
 * Initialize an immutable `UniSemanticTreeState` over a marked (`rows x (contentCols + 2)`)
 * or unmarked (`rows x contentCols`) VQ lattice:
 *   - Unlocked content cells start unactivated (`stage = -1`, `level = 0`, `noise = 1.0`).
 *   - Locked source-padding cells (`lockedContentMask[c] === 1`) start committed
 *     (`stage = refinementSteps`, `level = depth`, `noise = 0.0`) with `lockedSourceCodes[c]`.
 *   - Structural `+2` row/end markers (`256` on rows `0..rows-2`, `258` on row `rows-1`)
 *     start committed (`stage = refinementSteps`, `level = depth`, `noise = 0.0`, `isMarker = 1`).
 */
export function initSemanticTreeState(
  options: InitSemanticTreeStateOptions = {},
): UniSemanticTreeState {
  const rows = options.rows ?? UNI_MINI_CONFIG.vqGridSize;
  const contentCols = options.contentCols ?? UNI_MINI_CONFIG.vqGridSize;
  const includeMarkers = options.includeMarkers ?? true;
  const depth = options.depth ?? UNI_MINI_CONFIG.vqTreeDepth;
  const refinementSteps =
    options.refinementSteps ?? UNI_MINI_CONFIG.defaultRefinementSteps;

  if (!Number.isInteger(rows) || rows < 1) {
    throw new RangeError(`rows must be an integer >= 1, got ${rows}`);
  }
  if (!Number.isInteger(contentCols) || contentCols < 1) {
    throw new RangeError(`contentCols must be an integer >= 1, got ${contentCols}`);
  }
  if (!Number.isInteger(depth) || depth < 1 || depth > 30) {
    throw new RangeError(`depth must be an integer in [1, 30], got ${depth}`);
  }
  if (!Number.isInteger(refinementSteps) || refinementSteps < 1) {
    throw new RangeError(
      `refinementSteps must be an integer >= 1, got ${refinementSteps}`,
    );
  }

  const vocabSize = 1 << depth;
  const contentCount = rows * contentCols;
  const markedCols = includeMarkers ? contentCols + 2 : contentCols;
  const gridLen = rows * markedCols;

  let initialCodes: ArrayLike<number>;
  if (options.initialContentCodes !== undefined) {
    if (options.initialContentCodes.length !== contentCount) {
      throw new RangeError(
        `initialContentCodes length (${options.initialContentCodes.length}) must equal contentCount (${contentCount})`,
      );
    }
    initialCodes = options.initialContentCodes;
  } else {
    const rng = createXoshiro128ss(options.seed ?? UNI_STRATIFIED_DEFAULT_SEED);
    const drawn = new Int32Array(contentCount);
    const mask = vocabSize - 1;
    for (let c = 0; c < contentCount; c++) {
      drawn[c] = rng.nextUint32() & mask;
    }
    initialCodes = drawn;
  }

  const lockedMask = options.lockedContentMask;
  if (lockedMask !== undefined && lockedMask.length !== contentCount) {
    throw new RangeError(
      `lockedContentMask length (${lockedMask.length}) must equal contentCount (${contentCount})`,
    );
  }
  const lockedSource = options.lockedSourceCodes;
  if (lockedSource !== undefined && lockedSource.length !== contentCount) {
    throw new RangeError(
      `lockedSourceCodes length (${lockedSource.length}) must equal contentCount (${contentCount})`,
    );
  }

  const stages = new Int32Array(gridLen);
  const representatives = new Int32Array(gridLen);
  const levels = new Int32Array(gridLen);
  const noiseWeights = new Float32Array(gridLen);
  const isMarker = new Uint8Array(gridLen);

  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < contentCols; col++) {
      const c = r * contentCols + col;
      const g = r * markedCols + col;
      const isLocked = lockedMask !== undefined && lockedMask[c] === 1;
      isMarker[g] = 0;

      if (isLocked) {
        const srcCode = lockedSource !== undefined ? lockedSource[c] : initialCodes[c];
        stages[g] = refinementSteps;
        representatives[g] = srcCode;
        levels[g] = depth;
        noiseWeights[g] = 0.0;
      } else {
        stages[g] = -1;
        representatives[g] = initialCodes[c];
        levels[g] = 0;
        noiseWeights[g] = 1.0;
      }
    }

    if (includeMarkers) {
      const markerId =
        r < rows - 1 ? UNI_MINI_GEN_ROW_MARKER_ID : UNI_MINI_GEN_IMAGE_END_ID;
      for (let m = 0; m < 2; m++) {
        const g = r * markedCols + contentCols + m;
        isMarker[g] = 1;
        stages[g] = refinementSteps;
        representatives[g] = markerId;
        levels[g] = depth;
        noiseWeights[g] = 0.0;
      }
    }
  }

  return Object.freeze({
    stages,
    representatives,
    levels,
    noiseWeights,
    isMarker,
    refinementSteps,
  });
}

/**
 * Strictly non-mutating progressive self-refinement state transition:
 *   - Maps content index `c in 0..contentCount-1` to marked-grid flat index
 *     `r * markedCols + col` (derived from non-marker positions in `state.isMarker`).
 *   - Structural markers (`isMarker[g] === 1`), locked cells (`lockedContentMask?.[c] === 1`),
 *     and already committed cells (`stage >= refinementSteps`) NEVER change.
 *   - Unactivated cells (`stage === -1`) in `activatedContentIndices` advance to `stage = 1`.
 *   - Refining cells (`0 <= stage < refinementSteps`) advance to `stage + 1`.
 *   - Accepts `proposals` and `randBits` either in marked-grid (`[gridLen]`) or
 *     content-grid (`[contentCount]`) layout.
 */
export function advanceRevisableSemanticTree(
  tree: UniBalancedSemanticTree,
  state: UniSemanticTreeState,
  proposals: ArrayLike<number>,
  activatedContentIndices: ArrayLike<number>,
  randBits: ArrayLike<number>,
  lockedContentMask?: ArrayLike<number>,
): UniSemanticTreeState {
  const gridLen = state.stages.length;
  if (
    state.representatives.length !== gridLen ||
    state.levels.length !== gridLen ||
    state.noiseWeights.length !== gridLen ||
    state.isMarker.length !== gridLen
  ) {
    throw new RangeError(`Inconsistent UniSemanticTreeState array lengths`);
  }

  const contentToGrid: number[] = [];
  for (let g = 0; g < gridLen; g++) {
    if (state.isMarker[g] === 0) {
      contentToGrid.push(g);
    }
  }
  const contentCount = contentToGrid.length;

  const proposalsByGrid = proposals.length === gridLen;
  if (!proposalsByGrid && proposals.length !== contentCount) {
    throw new RangeError(
      `proposals length (${proposals.length}) must equal gridLen (${gridLen}) or contentCount (${contentCount})`,
    );
  }

  const randBitsByGrid = randBits.length === gridLen;
  if (!randBitsByGrid && randBits.length !== contentCount) {
    throw new RangeError(
      `randBits length (${randBits.length}) must equal gridLen (${gridLen}) or contentCount (${contentCount})`,
    );
  }

  let lockByGrid = false;
  if (lockedContentMask !== undefined) {
    if (lockedContentMask.length === contentCount) {
      lockByGrid = false;
    } else if (lockedContentMask.length === gridLen) {
      lockByGrid = true;
    } else {
      throw new RangeError(
        `lockedContentMask length (${lockedContentMask.length}) must equal contentCount (${contentCount}) or gridLen (${gridLen})`,
      );
    }
  }

  const activatedMask = new Uint8Array(contentCount);
  for (let i = 0; i < activatedContentIndices.length; i++) {
    const c = activatedContentIndices[i];
    if (!Number.isInteger(c) || c < 0 || c >= contentCount) {
      throw new RangeError(
        `activatedContentIndices[${i}] (${c}) out of bounds for contentCount=${contentCount}`,
      );
    }
    activatedMask[c] = 1;
  }

  const nextStages = new Int32Array(state.stages);
  const nextRepresentatives = new Int32Array(state.representatives);
  const nextLevels = new Int32Array(state.levels);
  const nextNoiseWeights = new Float32Array(state.noiseWeights);
  const nextIsMarker = new Uint8Array(state.isMarker);
  const N = state.refinementSteps;

  const advancingGridIndices: number[] = [];
  const advancingNextStages: number[] = [];
  const advancingProposals: number[] = [];
  const advancingRandBits: number[] = [];

  for (let c = 0; c < contentCount; c++) {
    const g = contentToGrid[c];
    if (
      lockedContentMask !== undefined &&
      lockedContentMask[lockByGrid ? g : c] === 1
    ) {
      continue;
    }

    const currentStage = state.stages[g];
    if (currentStage >= N) {
      continue;
    }

    let nextStage: number | null = null;
    if (currentStage < 0) {
      if (activatedMask[c] === 1) {
        nextStage = 1;
      }
    } else {
      nextStage = currentStage + 1;
    }

    if (nextStage !== null) {
      advancingGridIndices.push(g);
      advancingNextStages.push(nextStage);
      advancingProposals.push(proposalsByGrid ? proposals[g] : proposals[c]);
      advancingRandBits.push(randBitsByGrid ? randBits[g] : randBits[c]);
    }
  }

  if (advancingGridIndices.length > 0) {
    const { levels, noiseWeights } = semanticTreeLevelsForAges(
      advancingNextStages,
      tree.depth,
      N,
    );
    const reps = sampleRepresentatives(
      tree,
      advancingProposals,
      levels,
      advancingRandBits,
    );

    for (let i = 0; i < advancingGridIndices.length; i++) {
      const g = advancingGridIndices[i];
      nextStages[g] = advancingNextStages[i];
      nextLevels[g] = levels[i];
      nextNoiseWeights[g] = noiseWeights[i];
      nextRepresentatives[g] = reps[i];
    }
  }

  return Object.freeze({
    stages: nextStages,
    representatives: nextRepresentatives,
    levels: nextLevels,
    noiseWeights: nextNoiseWeights,
    isMarker: nextIsMarker,
    refinementSteps: N,
  });
}

// ============================================================================
// 12. U26 PRIME-DEPTH TRAIN-VS-INFERENCE FLOOR-ROUNDING MISMATCH DIAGNOSTIC
// ============================================================================

/**
 * Analyze the `U26` integer floor-rounding mismatch between training
 * (`minimal_training/corruption.py:58`: `trainLevel = D - floor(a_trn * D / K)`)
 * and inference (`tree.py:243`: `inferLevel = floor(a_inf * D / K)`) across all
 * internal stages `a_inf = 1 .. K - 1` (`a_trn = K - a_inf`).
 *
 * For production prime depth `D = 17`, every internal stage at `K ∈ {4, 8, 16}`
 * suffers an exact `1`-level (`2×` wider ball) mismatch; for our power-of-two
 * miniature `D = 8`, `K ∈ {1, 2, 4, 8}` yields exact `0` mismatches.
 */
export function analyzePrimeDepthFloorMismatch(
  treeDepth: number,
  refinementSteps: number,
): UniPrimeDepthFloorMismatchReport {
  if (!Number.isInteger(treeDepth) || treeDepth < 1 || treeDepth > 30) {
    throw new RangeError(
      `treeDepth must be an integer in [1, 30], got ${treeDepth}`,
    );
  }
  if (!Number.isInteger(refinementSteps) || refinementSteps < 1) {
    throw new RangeError(
      `refinementSteps must be an integer >= 1, got ${refinementSteps}`,
    );
  }

  const totalInternalStages = Math.max(0, refinementSteps - 1);
  const rows: UniPrimeDepthFloorMismatchRow[] = [];
  let mismatchedStageCount = 0;

  for (let stageIndex = 1; stageIndex < refinementSteps; stageIndex++) {
    const trainAge = refinementSteps - stageIndex;
    const trainLevel =
      treeDepth - Math.floor((trainAge * treeDepth) / refinementSteps);
    const inferLevel = Math.floor((stageIndex * treeDepth) / refinementSteps);
    const levelDelta = trainLevel - inferLevel;
    const trainBallSize = Math.pow(2, treeDepth - trainLevel);
    const inferBallSize = Math.pow(2, treeDepth - inferLevel);
    const ballSizeRatio = inferBallSize / trainBallSize;

    if (levelDelta !== 0) {
      mismatchedStageCount++;
    }

    rows.push(
      Object.freeze({
        stageIndex,
        trainAge,
        trainLevel,
        inferLevel,
        levelDelta,
        trainBallSize,
        inferBallSize,
        ballSizeRatio,
      }),
    );
  }

  return Object.freeze({
    treeDepth,
    refinementSteps,
    hasMismatch: mismatchedStageCount > 0,
    mismatchedStageCount,
    totalInternalStages,
    rows: Object.freeze(rows),
  });
}
