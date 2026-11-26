/**
 * PackedLinearImageCodec (`2x2`), Row/End Markers, Fixed Prefix & Padding Lock (`src/uni/codec.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching the pinned
 * Nemotron-Labs-Diffusion-Uni multimodal codec contracts (`U13`, `U18`, `U30`, `D3`):
 *   - Exact 4-tensor (`528`-float) `PackedLinearImageCodec` (`down_norm`,
 *     `down_projection [8, 32]`, `up_norm`, `up_projection [32, 8]`)
 *   - `+2` row/end markers (`256` on rows `0..rows-2`, `258` on row `rows-1`)
 *   - `4` uncompressed fixed-prefix tokens (`259..262`)
 *   - `D3` dual gain presets (`mini_default` `0.5` vs `prod_exact` `0.0053573 * 0.4498539`)
 *     + additive `2D_SinCos` with `markerColumn = !isSourceEdit` (`0`-ULP content alignment)
 *   - Source-padding VQ lock (`computeSourcePaddingVqMask`, `applySourcePaddingVqLock`,
 *     `applySourcePaddingEmbeddingLock`, `cropPaddedImageToSource`)
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG calls, or sibling-app imports.
 */

import { applyUniRmsNorm, linearProject } from './attention';
import { compute2dSinCosPosEmbed } from './rope';
import {
  UNI_GAIN_PRESETS,
  UNI_MINI_CODEC_FLOAT_COUNT,
  UNI_MINI_CODEC_TENSOR_COUNT,
  UNI_MINI_CONFIG,
  UNI_MINI_GEN_FIXED_PREFIX_COUNT,
  UNI_MINI_GEN_FIXED_PREFIX_START_ID,
  UNI_MINI_GEN_IMAGE_END_ID,
  UNI_MINI_GEN_ROW_MARKER_ID,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MULTIMODAL_SEED,
  type UniCodecCompressTrace,
  type UniCodecExpandTrace,
  type UniGainPresetId,
  type UniLetterboxInfo,
  type UniMultimodalTensorEntry,
  type UniPackedCodecWeights,
  type UniRgbImage,
  type UniSourcePaddingVqMask,
} from './types';

// ============================================================================
// ISOLATED PER-TENSOR PRNG (identical contract to `src/uni/parameters.ts`)
// ============================================================================

const FNV1A_32_OFFSET = 0x811c9dc5;
const FNV1A_32_PRIME = 0x01000193;

function fnv1a32Seed(baseSeed: number, label: string): number {
  let h = (FNV1A_32_OFFSET ^ (baseSeed >>> 0)) >>> 0;
  h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i) & 0xff;
    h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createGaussianGenerator(seed: number): () => number {
  const nextUniform = mulberry32(seed);
  let spare: number | null = null;

  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u1 = nextUniform();
    while (u1 <= 1e-12) {
      u1 = nextUniform();
    }
    const u2 = nextUniform();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const angle = 2.0 * Math.PI * u2;
    spare = mag * Math.sin(angle);
    return mag * Math.cos(angle);
  };
}

function sampleGaussianTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
  std: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.fround(gauss() * std);
  }
  return out;
}

function sampleRmsNormTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const v = 1.0 + 0.05 * gauss();
    out[i] = Math.fround(v < 0.5 ? 0.5 : v);
  }
  return out;
}

// ============================================================================
// WEIGHT FACTORY & 4-TENSOR CANONICAL COLLECTOR
// ============================================================================

/**
 * Create the 4 deterministic tensors (`528` floats) for `PackedLinearImageCodec`.
 */
export function createUniPackedCodecWeights(
  seed: number = UNI_MULTIMODAL_SEED,
): UniPackedCodecWeights {
  const s = seed >>> 0;
  const d = UNI_MINI_CONFIG.hiddenSize; // 8
  const packedDim = d * 4;              // 32

  const weights: UniPackedCodecWeights = Object.freeze({
    seed: s,
    downNorm: sampleRmsNormTensor(s, 'encoder.packed_image_codec.down_norm.weight', d),
    downProjection: sampleGaussianTensor(
      s,
      'encoder.packed_image_codec.down_projection.weight',
      d * packedDim,
      0.25 / Math.sqrt(packedDim),
    ),
    upNorm: sampleRmsNormTensor(s, 'encoder.packed_image_codec.up_norm.weight', d),
    upProjection: sampleGaussianTensor(
      s,
      'encoder.packed_image_codec.up_projection.weight',
      packedDim * d,
      0.25 / Math.sqrt(d),
    ),
  });

  const collected = collectUniCodecWeights(weights);
  if (collected.length !== UNI_MINI_CODEC_TENSOR_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_CODEC_TENSOR_COUNT} codec tensors, got ${collected.length}`,
    );
  }
  let totalFloats = 0;
  for (let i = 0; i < collected.length; i++) {
    totalFloats += collected[i].data.length;
  }
  if (totalFloats !== UNI_MINI_CODEC_FLOAT_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_CODEC_FLOAT_COUNT} codec floats, got ${totalFloats}`,
    );
  }

  return weights;
}

/**
 * Collect the exact 4 `PackedLinearImageCodec` tensors (`528` floats,
 * `subsystem: 'packed_image_codec'`) in canonical order.
 */
export function collectUniCodecWeights(
  weights: UniPackedCodecWeights,
): UniMultimodalTensorEntry[] {
  const d = UNI_MINI_CONFIG.hiddenSize;
  const packedDim = d * 4;
  return [
    {
      name: 'encoder.packed_image_codec.down_norm.weight',
      subsystem: 'packed_image_codec',
      shape: [d],
      data: weights.downNorm,
    },
    {
      name: 'encoder.packed_image_codec.down_projection.weight',
      subsystem: 'packed_image_codec',
      shape: [d, packedDim],
      data: weights.downProjection,
    },
    {
      name: 'encoder.packed_image_codec.up_norm.weight',
      subsystem: 'packed_image_codec',
      shape: [d],
      data: weights.upNorm,
    },
    {
      name: 'encoder.packed_image_codec.up_projection.weight',
      subsystem: 'packed_image_codec',
      shape: [packedDim, d],
      data: weights.upProjection,
    },
  ];
}

// ============================================================================
// +2 ROW/END MARKERS & 4 FIXED-PREFIX TOKENS (`U13`)
// ============================================================================

/**
 * Append `+2` structural marker tokens per row (`4x4 -> 4x6 = 24`):
 *   - Rows `0 .. rows - 2`: `[256, 256]` (`UNI_MINI_GEN_ROW_MARKER_ID`)
 *   - Row `rows - 1`: `[258, 258]` (`UNI_MINI_GEN_IMAGE_END_ID`)
 */
export function addGenerationRowMarkers(
  codes: ArrayLike<number>,
  rows: number = UNI_MINI_CONFIG.vqGridSize,
  cols: number = UNI_MINI_CONFIG.vqGridSize,
): Int32Array {
  if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(cols) || cols < 1) {
    throw new RangeError(`Invalid grid (${rows}x${cols})`);
  }
  if (codes.length !== rows * cols) {
    throw new RangeError(
      `codes length (${codes.length}) must equal rows * cols (${rows * cols})`,
    );
  }

  const markedCols = cols + 2;
  const out = new Int32Array(rows * markedCols);
  for (let r = 0; r < rows; r++) {
    const srcBase = r * cols;
    const dstBase = r * markedCols;
    for (let c = 0; c < cols; c++) {
      out[dstBase + c] = codes[srcBase + c];
    }
    const marker =
      r < rows - 1 ? UNI_MINI_GEN_ROW_MARKER_ID : UNI_MINI_GEN_IMAGE_END_ID;
    out[dstBase + cols] = marker;
    out[dstBase + cols + 1] = marker;
  }
  return out;
}

/**
 * Strip the trailing `+2` structural marker columns per row (`4x6 -> 4x4 = 16`).
 */
export function stripGenerationRowMarkers(
  markedCodes: ArrayLike<number>,
  rows: number = UNI_MINI_CONFIG.vqGridSize,
  markedCols: number = UNI_MINI_CONFIG.vqMarkedGridCols,
): Int32Array {
  if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(markedCols) || markedCols < 3) {
    throw new RangeError(`Invalid marked grid (${rows}x${markedCols})`);
  }
  if (markedCodes.length !== rows * markedCols) {
    throw new RangeError(
      `markedCodes length (${markedCodes.length}) must equal rows * markedCols (${rows * markedCols})`,
    );
  }

  const cols = markedCols - 2;
  const out = new Int32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const srcBase = r * markedCols;
    const dstBase = r * cols;
    for (let c = 0; c < cols; c++) {
      out[dstBase + c] = markedCodes[srcBase + c];
    }
  }
  return out;
}

/**
 * Return the `4` fixed-prefix generation token IDs `[259, 260, 261, 262]`.
 */
export function buildFixedPrefixTokenIds(): Int32Array {
  const count = UNI_MINI_GEN_FIXED_PREFIX_COUNT;
  const start = UNI_MINI_GEN_FIXED_PREFIX_START_ID;
  const ids = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    ids[i] = start + i;
  }
  return ids;
}

/**
 * Look up the `4` uncompressed fixed-prefix token embeddings `[4, hiddenSize]`
 * (`259..262`) directly from `genEmbedding`.
 */
export function embedFixedPrefixTokens(
  genEmbedding: Float32Array,
  hiddenSize: number = UNI_MINI_CONFIG.hiddenSize,
): Float32Array {
  const count = UNI_MINI_GEN_FIXED_PREFIX_COUNT;
  const start = UNI_MINI_GEN_FIXED_PREFIX_START_ID;
  const minLength = (start + count) * hiddenSize;
  if (genEmbedding.length < minLength) {
    throw new RangeError(
      `genEmbedding length (${genEmbedding.length}) must be >= ${minLength}`,
    );
  }

  const out = new Float32Array(count * hiddenSize);
  for (let i = 0; i < count; i++) {
    const srcBase = (start + i) * hiddenSize;
    const dstBase = i * hiddenSize;
    for (let d = 0; d < hiddenSize; d++) {
      out[dstBase + d] = genEmbedding[srcBase + d];
    }
  }
  return out;
}

// ============================================================================
// PACKED LINEAR IMAGE CODEC COMPRESS & EXPAND (`U13`, `D3`)
// ============================================================================

/**
 * Compress `[childRows * childCols, 8]` child visual embeddings via `2x2`
 * `PackedLinearImageCodec`:
 *   1. Per-child `RMSNorm(8, eps=1e-6)` (`down_norm`)
 *   2. `2x2` row-major `[TL, TR, BL, BR]` concat `[parentRows * parentCols, 32]`
 *   3. `Linear(32 -> 8)` (`down_projection`)
 *   4. Scale by `UNI_GAIN_PRESETS[gainPresetId].combinedGenerationScale` (`D3`)
 *   5. Add `compute2dSinCosPosEmbed(parentRows, parentCols, 8, !isSourceEdit)` (`U13`)
 */
export function compressPackedImageEmbeddings(
  childEmbeddings: Float32Array,
  childRows: number,
  childCols: number,
  weights: UniPackedCodecWeights,
  options: { isSourceEdit: boolean; gainPresetId?: UniGainPresetId },
): UniCodecCompressTrace {
  if (
    !Number.isInteger(childRows) ||
    !Number.isInteger(childCols) ||
    childRows < 2 ||
    childCols < 2 ||
    childRows % 2 !== 0 ||
    childCols % 2 !== 0
  ) {
    throw new RangeError(
      `childRows (${childRows}) and childCols (${childCols}) must be even integers >= 2`,
    );
  }

  const d = UNI_MINI_CONFIG.hiddenSize;
  const numChildren = childRows * childCols;
  if (childEmbeddings.length < numChildren * d) {
    throw new RangeError(
      `childEmbeddings length (${childEmbeddings.length}) is smaller than childRows * childCols * ${d} (${numChildren * d})`,
    );
  }

  const parentRows = childRows / 2;
  const parentCols = childCols / 2;
  const numParents = parentRows * parentCols;
  const packedDim = d * 4;

  const normedChildren = applyUniRmsNorm(
    childEmbeddings,
    numChildren,
    d,
    weights.downNorm,
    UNI_MINI_CONFIG.codecNormEps,
  );

  const packedConcat = new Float32Array(numParents * packedDim);
  for (let pr = 0; pr < parentRows; pr++) {
    const cr0 = 2 * pr;
    const cr1 = cr0 + 1;
    for (let pc = 0; pc < parentCols; pc++) {
      const cc0 = 2 * pc;
      const cc1 = cc0 + 1;
      const dstBase = (pr * parentCols + pc) * packedDim;

      const tlBase = (cr0 * childCols + cc0) * d;
      const trBase = (cr0 * childCols + cc1) * d;
      const blBase = (cr1 * childCols + cc0) * d;
      const brBase = (cr1 * childCols + cc1) * d;

      for (let k = 0; k < d; k++) {
        packedConcat[dstBase + k] = normedChildren[tlBase + k];
        packedConcat[dstBase + d + k] = normedChildren[trBase + k];
        packedConcat[dstBase + 2 * d + k] = normedChildren[blBase + k];
        packedConcat[dstBase + 3 * d + k] = normedChildren[brBase + k];
      }
    }
  }

  const projected = linearProject(
    packedConcat,
    numParents,
    packedDim,
    weights.downProjection,
    d,
  );

  const gainPresetId = options.gainPresetId ?? UNI_MINI_CONFIG.defaultGainPreset;
  const combinedScale = UNI_GAIN_PRESETS[gainPresetId].combinedGenerationScale;
  const combinedScaleF32 = Math.fround(combinedScale);

  const scaled = new Float32Array(numParents * d);
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] = Math.fround(projected[i] * combinedScaleF32);
  }

  const markerColumn = !options.isSourceEdit;
  const sinCos2d = compute2dSinCosPosEmbed(parentRows, parentCols, d, markerColumn);

  const output = new Float32Array(numParents * d);
  for (let i = 0; i < output.length; i++) {
    output[i] = Math.fround(scaled[i] + sinCos2d.embeddings[i]);
  }

  return {
    childRows,
    childCols,
    parentRows,
    parentCols,
    isSourceEdit: options.isSourceEdit,
    markerColumn,
    gainPresetId,
    combinedScale,
    normedChildren,
    packedConcat,
    projected,
    scaled,
    sinCos2d,
    output,
  };
}

/**
 * Expand `[parentRows * parentCols, 8]` transformer hidden states back to
 * `[childRows * childCols, 8]` via `up_norm -> up_projection -> 2x2 unpack`,
 * and optionally project through `genPredictor [256, 8]` to `[childRows * childCols, 256]`.
 */
export function expandPackedImageHiddenStates(
  parentHiddenStates: Float32Array,
  parentRows: number,
  parentCols: number,
  weights: UniPackedCodecWeights,
  genPredictor?: Float32Array,
): UniCodecExpandTrace {
  if (!Number.isInteger(parentRows) || parentRows < 1 || !Number.isInteger(parentCols) || parentCols < 1) {
    throw new RangeError(`Invalid parent grid (${parentRows}x${parentCols})`);
  }

  const d = UNI_MINI_CONFIG.hiddenSize;
  const packedDim = d * 4;
  const numParents = parentRows * parentCols;
  if (parentHiddenStates.length < numParents * d) {
    throw new RangeError(
      `parentHiddenStates length (${parentHiddenStates.length}) is smaller than parentRows * parentCols * ${d} (${numParents * d})`,
    );
  }

  const childRows = parentRows * 2;
  const childCols = parentCols * 2;
  const numChildren = childRows * childCols;

  const normedParents = applyUniRmsNorm(
    parentHiddenStates,
    numParents,
    d,
    weights.upNorm,
    UNI_MINI_CONFIG.codecNormEps,
  );

  const upProjected = linearProject(
    normedParents,
    numParents,
    d,
    weights.upProjection,
    packedDim,
  );

  const unpackedChildren = new Float32Array(numChildren * d);
  for (let pr = 0; pr < parentRows; pr++) {
    const cr0 = 2 * pr;
    const cr1 = cr0 + 1;
    for (let pc = 0; pc < parentCols; pc++) {
      const cc0 = 2 * pc;
      const cc1 = cc0 + 1;
      const srcBase = (pr * parentCols + pc) * packedDim;

      const tlBase = (cr0 * childCols + cc0) * d;
      const trBase = (cr0 * childCols + cc1) * d;
      const blBase = (cr1 * childCols + cc0) * d;
      const brBase = (cr1 * childCols + cc1) * d;

      for (let k = 0; k < d; k++) {
        unpackedChildren[tlBase + k] = upProjected[srcBase + k];
        unpackedChildren[trBase + k] = upProjected[srcBase + d + k];
        unpackedChildren[blBase + k] = upProjected[srcBase + 2 * d + k];
        unpackedChildren[brBase + k] = upProjected[srcBase + 3 * d + k];
      }
    }
  }

  let logits: Float32Array | null = null;
  if (genPredictor !== undefined) {
    const vqSize = UNI_MINI_VQ_CODEBOOK_SIZE;
    if (genPredictor.length < vqSize * d) {
      throw new RangeError(
        `genPredictor length (${genPredictor.length}) is smaller than ${vqSize * d}`,
      );
    }
    logits = linearProject(unpackedChildren, numChildren, d, genPredictor, vqSize);
  }

  return {
    parentRows,
    parentCols,
    childRows,
    childCols,
    normedParents,
    upProjected,
    unpackedChildren,
    logits,
  };
}

// ============================================================================
// SOURCE-PADDING VQ LOCK & LETTERBOX CROP (`U18`, `U30`)
// ============================================================================

/**
 * Compute the binary lock mask (`1` = locked letterbox border, `0` = editable
 * active content) over the `[vqGridRows, vqGridCols]` VQ lattice.
 * A cell `(r, c)` is locked iff its `[r*stride..(r+1)*stride) x [c*stride..(c+1)*stride)`
 * pixel footprint extends outside `[padTop..padTop+resizedHeight) x [padLeft..padLeft+resizedWidth)`.
 */
export function computeSourcePaddingVqMask(
  letterbox: UniLetterboxInfo,
  vqStride: number = UNI_MINI_CONFIG.vitPatchSize,
): UniSourcePaddingVqMask {
  if (!Number.isInteger(vqStride) || vqStride < 1) {
    throw new RangeError(`vqStride must be an integer >= 1, got ${vqStride}`);
  }
  if (letterbox.targetSize % vqStride !== 0) {
    throw new RangeError(
      `letterbox.targetSize (${letterbox.targetSize}) must be divisible by vqStride (${vqStride})`,
    );
  }

  const vqGridRows = letterbox.targetSize / vqStride;
  const vqGridCols = letterbox.targetSize / vqStride;
  const activeTop = letterbox.padTop;
  const activeBottom = letterbox.padTop + letterbox.resizedHeight;
  const activeLeft = letterbox.padLeft;
  const activeRight = letterbox.padLeft + letterbox.resizedWidth;

  const lockMask = new Uint8Array(vqGridRows * vqGridCols);
  let lockedCellCount = 0;

  for (let r = 0; r < vqGridRows; r++) {
    const rowStart = r * vqStride;
    const rowEnd = (r + 1) * vqStride;
    const rowInside = rowStart >= activeTop && rowEnd <= activeBottom;

    for (let c = 0; c < vqGridCols; c++) {
      const colStart = c * vqStride;
      const colEnd = (c + 1) * vqStride;
      const colInside = colStart >= activeLeft && colEnd <= activeRight;

      const locked = rowInside && colInside ? 0 : 1;
      lockMask[r * vqGridCols + c] = locked;
      lockedCellCount += locked;
    }
  }

  return {
    vqGridRows,
    vqGridCols,
    letterbox,
    lockMask,
    lockedCellCount,
    editableCellCount: lockMask.length - lockedCellCount,
  };
}

/**
 * Immutably replace locked border cells in `targetCodes` with `sourceCodes`.
 */
export function applySourcePaddingVqLock(
  targetCodes: ArrayLike<number>,
  sourceCodes: ArrayLike<number>,
  mask: UniSourcePaddingVqMask,
): Int32Array {
  const n = mask.lockMask.length;
  if (targetCodes.length !== n || sourceCodes.length !== n) {
    throw new RangeError(
      `targetCodes (${targetCodes.length}) and sourceCodes (${sourceCodes.length}) must match mask length (${n})`,
    );
  }

  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = mask.lockMask[i] === 1 ? sourceCodes[i] : targetCodes[i];
  }
  return out;
}

/**
 * Immutably replace locked border embeddings `[N, hiddenSize]` in `targetEmbeddings`
 * with `sourceEmbeddings`.
 */
export function applySourcePaddingEmbeddingLock(
  targetEmbeddings: Float32Array,
  sourceEmbeddings: Float32Array,
  mask: UniSourcePaddingVqMask,
  hiddenSize: number = UNI_MINI_CONFIG.hiddenSize,
): Float32Array {
  const n = mask.lockMask.length;
  const expectedFloats = n * hiddenSize;
  if (targetEmbeddings.length < expectedFloats || sourceEmbeddings.length < expectedFloats) {
    throw new RangeError(
      `Embedding buffers (${targetEmbeddings.length}, ${sourceEmbeddings.length}) must be >= ${expectedFloats}`,
    );
  }

  const out = new Float32Array(expectedFloats);
  for (let i = 0; i < n; i++) {
    const src = mask.lockMask[i] === 1 ? sourceEmbeddings : targetEmbeddings;
    const base = i * hiddenSize;
    for (let d = 0; d < hiddenSize; d++) {
      out[base + d] = src[base + d];
    }
  }
  return out;
}

/**
 * Crop a letterboxed square RGB image back to its exact source `(sourceWidth, sourceHeight)`.
 * When `resizedWidth === sourceWidth && resizedHeight === sourceHeight` (e.g., `16x16 -> 16x8`),
 * performs an exact bit-for-bit integer center crop; otherwise bilinearly resizes the cropped box.
 */
export function cropPaddedImageToSource(
  paddedImage: UniRgbImage,
  letterbox: UniLetterboxInfo,
): UniRgbImage {
  const { targetSize, padLeft, padTop, resizedWidth, resizedHeight, sourceWidth, sourceHeight } =
    letterbox;
  if (paddedImage.width !== targetSize || paddedImage.height !== targetSize) {
    throw new RangeError(
      `paddedImage (${paddedImage.width}x${paddedImage.height}) must match letterbox.targetSize (${targetSize})`,
    );
  }
  if (paddedImage.data.length < 3 * targetSize * targetSize) {
    throw new RangeError(`paddedImage.data buffer too small`);
  }

  const cropped = new Float32Array(3 * resizedHeight * resizedWidth);
  const srcPlane = targetSize * targetSize;
  const dstPlane = resizedHeight * resizedWidth;

  for (let c = 0; c < 3; c++) {
    const sBase = c * srcPlane;
    const dBase = c * dstPlane;
    for (let y = 0; y < resizedHeight; y++) {
      const sRow = sBase + (padTop + y) * targetSize + padLeft;
      const dRow = dBase + y * resizedWidth;
      for (let x = 0; x < resizedWidth; x++) {
        cropped[dRow + x] = paddedImage.data[sRow + x];
      }
    }
  }

  if (resizedWidth === sourceWidth && resizedHeight === sourceHeight) {
    return {
      width: sourceWidth,
      height: sourceHeight,
      data: cropped,
    };
  }

  const out = new Float32Array(3 * sourceHeight * sourceWidth);
  const outPlane = sourceHeight * sourceWidth;
  for (let c = 0; c < 3; c++) {
    const sBase = c * dstPlane;
    const oBase = c * outPlane;
    for (let y = 0; y < sourceHeight; y++) {
      const sy = ((y + 0.5) * resizedHeight) / sourceHeight - 0.5;
      const y0 = Math.max(0, Math.min(resizedHeight - 1, Math.floor(sy)));
      const y1 = Math.max(0, Math.min(resizedHeight - 1, y0 + 1));
      const wy = Math.max(0, Math.min(1, sy - y0));

      for (let x = 0; x < sourceWidth; x++) {
        const sx = ((x + 0.5) * resizedWidth) / sourceWidth - 0.5;
        const x0 = Math.max(0, Math.min(resizedWidth - 1, Math.floor(sx)));
        const x1 = Math.max(0, Math.min(resizedWidth - 1, x0 + 1));
        const wx = Math.max(0, Math.min(1, sx - x0));

        const v00 = cropped[sBase + y0 * resizedWidth + x0];
        const v01 = cropped[sBase + y0 * resizedWidth + x1];
        const v10 = cropped[sBase + y1 * resizedWidth + x0];
        const v11 = cropped[sBase + y1 * resizedWidth + x1];

        const top = v00 + (v01 - v00) * wx;
        const bot = v10 + (v11 - v10) * wx;
        const val = Math.fround(top + (bot - top) * wy);
        out[oBase + y * sourceWidth + x] = val < 0 ? 0 : val > 1 ? 1 : val;
      }
    }
  }

  return {
    width: sourceWidth,
    height: sourceHeight,
    data: out,
  };
}
