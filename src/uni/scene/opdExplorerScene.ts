/**
 * Milestone UNI5 — Slice 5C · View 7 (`opd_readonly`):
 * Read-Only Cross-Paradigm On-Policy Distillation (OPD) Explorer Scene
 * & `128 × 64` `RG32F` Scalar/Mask Atlas (`src/uni/scene/opdExplorerScene.ts`)
 *
 * Clean-room, strictly READ-ONLY, forward-only headless 3D spatial scene builder
 * (`0` gradients, `0` backward passes, `0` optimizer steps, `0` `Math.random()`):
 *
 *   - **Left Bay (Transitive Understanding OPD · Indigo `U`):**
 *       * Pipeline: `AR Teacher (p_t^AR) → Student AR (p_s^AR) → stopgrad → Student Diff (p_s^Diff)`
 *       * Rect 1 (`sbd_dual_stream_mask`, `32×32`, `SHEET_KIND_CAUSAL_SCORES`, strict `U25`):
 *         Dual-stream `[noisy (0..15) || clean (16..31)]` SBD `block_diffusion_mask` (`L=16, B=4`)
 *         via `buildDualStreamBlockDiffusionMask`, with `G = 1.0` ONLY on blocked cells and
 *         `G = 0.0` on all visible cells.
 *       * Rect 2 (`opd_transitive_tail_kl`, `6×43`, `G = 0.0`):
 *         Exact `3×43 = 129`-bucket AR Teacher (`p_t^AR`) vs `3×43 = 129`-bucket Student AR (`p_s^AR`)
 *         `tail_kl` distribution (`4.0 * reverse_tail_kl`).
 *       * Explicit `"New Francisco"` context-mismatch callout explaining why naive direct
 *         AR Teacher → Student Diff score copying corrupts bidirectional masked contexts.
 *       * Rect 3 (`opd_self_consistency_kl`, `6×43`, `G = 0.0`):
 *         Exact `3×43 = 129`-bucket `stopgrad(Student AR)` vs `3×43 = 129`-bucket `Student Diff`
 *         self-consistency `tail_kl` distribution (`forward_tail_kl / (0.5005 * R)`).
 *       * Explicit **`U28`** 3-bug reference-trainer diagnostic callout (`minimal_training/`):
 *         (1) VLM SFT double-normalization (`~6.915×` avg / `1,000×` spike at `t = 0.001`),
 *         (2) accidental `<|im_end|>` (`id = 11`) unmasking/supervision on system/user turns,
 *         (3) `132,101 - 64 = 132,037` `<LOC_*>` tail leak (`1,029` tokens).
 *
 *   - **Right Bay (Continuous-Flow Editing OPD · Teal `G`):**
 *       * Pipeline: `Student VQ Rollout → VAE Encode → x^r (r=0.9, matched_sigmas) → Multi-Step Teacher ODE → Re-Quantized Target`
 *       * Rect 4 (`opd_matched_sigmas`, `2×steps` for `steps <= 88` or wrapped `rows×64` for `steps > 88`):
 *         Qwen-Image inverted dynamic time-shift `matched_sigmas(r, seqLen, steps)` schedule (`r → 0.02`),
 *         with `G = 0.0` on all active schedule cells (and `G = 1.0` strictly on unused padding if wrapped).
 *       * Rect 5 (`opd_flow_latents`, `3×16`, `G = 0.0`):
 *         Linear flow-noising comparison `[x^0, δ, x^r = (1 - r) * x^0 + r * δ]`.
 *       * Hybrid editing loss readout: `exact_ce + 0.2 * grouped_ce`.
 *
 * Strict `D4` discipline: `rgbPreviews: []` (zero 3-channel RGB packed into `RG32F`).
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import type { Vec3 } from '../../model/spec';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import {
  FRAME_XZ_AXES,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NORM,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
} from '../../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import { buildDualStreamBlockDiffusionMask } from '../masks';
import {
  createDefaultOpdDemoInputs,
  evaluateFlowEditingOpdRef,
  evaluateUnderstandingOpdRef,
  getTrainerBugDiagnosticsU28,
} from '../opdRef';
import type { UniSession } from '../session';
import type {
  UniElasticPrefillMode,
  UniExecutionViewId,
  UniFlowEditingOpdResult,
  UniTrainerBugDiagnosticU28,
  UniUnderstandingOpdResult,
} from '../types';
import {
  auditAtlasRegions,
  computeAtlasContentKey,
  packPickBoxFromFrame,
  resolveUniSessionContext,
  UNI_FLOW_COLORS,
  UniFlatFlowBufferBuilder,
  UniLineBufferBuilder,
  UniRibbonBufferBuilder,
  UniSheetBufferBuilder,
  UniSolidBufferBuilder,
  validateSceneGeometryFinite,
} from './provenance';
import type {
  UniAtlasLayoutAudit,
  UniAtlasRegionBox,
  UniOpdBayFocus,
  UniOpdReadOnlyMetadata,
  UniOpdSceneOptions,
  UniOverlayAnchor,
  UniProvenanceBadge,
  UniSpatialScene,
} from './types';

// ============================================================================
// CONSTANTS & EXPORTED TYPES
// ============================================================================

export const UNI_OPD_ATLAS_WIDTH = 128 as const;
export const UNI_OPD_ATLAS_HEIGHT = 64 as const;
export const UNI_OPD_ATLAS_BUDGET_BYTES = 65_536 as const; // 128 * 64 * 8 B = 64.00 KiB
export const UNI_OPD_REGION_COUNT = 5 as const;

export const UNI_OPD_DEFAULT_SEED = 0x4f504431 as const;
export const UNI_OPD_DEFAULT_R = 0.9 as const;
export const UNI_OPD_DEFAULT_STEPS = 40 as const;
export const UNI_OPD_DEFAULT_SEQ_LEN = 1024 as const;
export const UNI_OPD_SBD_CLEAN_LEN = 16 as const;
export const UNI_OPD_SBD_BLOCK_SIZE = 4 as const;
export const UNI_OPD_READONLY_BADGE_LABEL = 'READ-ONLY · NO GRADIENTS' as const;

export type UniOpdPlateId =
  | 'sbd_dual_stream_mask'
  | 'opd_transitive_tail_kl'
  | 'opd_self_consistency_kl'
  | 'opd_matched_sigmas'
  | 'opd_flow_latents';

export type UniOpdStageCardId =
  | 'opd_u_ar_teacher_card'
  | 'opd_u_student_ar_card'
  | 'opd_u_stopgrad_card'
  | 'opd_u_new_francisco_callout'
  | 'opd_u28_diagnostics_card'
  | 'opd_g_vq_rollout_card'
  | 'opd_g_vae_encode_card'
  | 'opd_g_teacher_ode_card'
  | 'opd_g_requant_target_card';

export interface UniOpdAtlasSliceEntry {
  readonly id: UniOpdPlateId;
  readonly region: UniAtlasRegionBox;
  readonly slice: AtlasSlice;
}

export interface UniOpdAtlasPackResult {
  readonly atlas: Rg32fValueAtlas;
  readonly atlasKey: string;
  readonly regions: readonly UniAtlasRegionBox[];
  readonly audit: UniAtlasLayoutAudit;
  readonly slices: Readonly<Record<UniOpdPlateId, UniOpdAtlasSliceEntry>>;
  readonly sbdMaskVisible: Uint8Array;
  readonly matchedSigmasWrapped: boolean;
}

interface OpdMatrixPackParams {
  readonly id: UniOpdPlateId;
  readonly x0: number;
  readonly y0: number;
  readonly rows: number;
  readonly cols: number;
  readonly values: ArrayLike<number>;
  readonly mode: 0 | 1;
  readonly isMaskOrScore: boolean;
  /** Optional explicit per-cell G-channel flag (`1.0` ONLY on blocked/masked or unused padding cells). */
  readonly gChannelFlags?: ArrayLike<number>;
  readonly explicitScale?: number;
}

// ============================================================================
// INTERNAL NON-EXPORTED RG32F PACKER (NO COLLISION WITH `packMatrixToRg32f`)
// ============================================================================

function packOpdMatrixToRg32f(
  writer: Rg32fAtlasWriter,
  params: OpdMatrixPackParams,
): UniOpdAtlasSliceEntry {
  const { id, x0, y0, rows, cols, values, mode, isMaskOrScore, gChannelFlags, explicitScale } =
    params;

  if (
    !Number.isInteger(x0) ||
    !Number.isInteger(y0) ||
    !Number.isInteger(rows) ||
    !Number.isInteger(cols) ||
    rows < 1 ||
    cols < 1
  ) {
    throw new RangeError(`Invalid OPD RG32F region "${id}": (${x0},${y0}) ${cols}x${rows}`);
  }
  if (x0 < 0 || y0 < 0 || x0 + cols > writer.width || y0 + rows > writer.height) {
    throw new RangeError(
      `OPD RG32F region "${id}" (${x0},${y0} + ${cols}x${rows}) exceeds atlas ${writer.width}x${writer.height}`,
    );
  }
  const cellCount = rows * cols;
  if (values.length < cellCount) {
    throw new RangeError(
      `OPD RG32F region "${id}" values length (${values.length}) < rows*cols (${cellCount})`,
    );
  }

  let scale = explicitScale;
  if (scale === undefined) {
    if (mode === 1) {
      scale = 1.0;
    } else {
      const buf =
        values instanceof Float32Array ? values.subarray(0, cellCount) : Float32Array.from(values);
      scale = 1.0 / computeMaxAbs(buf, 0.25);
    }
  }

  for (let r = 0; r < rows; r++) {
    const srcRow = r * cols;
    const dstY = y0 + r;
    for (let c = 0; c < cols; c++) {
      const idx = srcRow + c;
      const rawVal = values[idx]!;
      const isMasked = gChannelFlags !== undefined && gChannelFlags[idx]! > 0.5;
      if (isMasked) {
        // Blocked SBD cell or unused padding slot: write G = 1.0 so SheetsPass hatches it cleanly.
        writer.setTexel(x0 + c, dstY, 0.0, true);
      } else {
        // Active scalar/probability cell: G = 0.0 strictly preserved!
        writer.setTexel(x0 + c, dstY, rawVal, false);
      }
    }
  }

  const region: UniAtlasRegionBox = Object.freeze({
    id,
    x0,
    y0,
    cols,
    rows,
    mode,
    isMaskOrScore,
  });

  const slice: AtlasSlice = Object.freeze({
    rect: writer.makeRect(x0, y0, cols, rows),
    scale,
    mode,
  });

  return Object.freeze({ id, region, slice });
}

function mergeAabb(target: { min: [number, number, number]; max: [number, number, number] }, box: AABB): void {
  for (let i = 0; i < 3; i++) {
    if (box.min[i]! < target.min[i]!) target.min[i] = box.min[i]!;
    if (box.max[i]! > target.max[i]!) target.max[i] = box.max[i]!;
  }
}

// ============================================================================
// 128 × 64 RG32F SCALAR/MASK ATLAS PACKER (EXACTLY 5 NON-OVERLAPPING RECTS)
// ============================================================================

/**
 * Pack the strictly read-only `128 × 64` `RG32F` scalar/mask atlas (`65,536 B` = `64.00 KiB`)
 * containing exactly 5 non-overlapping plates:
 *   1. `sbd_dual_stream_mask`: `32×32` at `(0, 0)` — `G = 1.0` where blocked, `G = 0.0` where visible
 *   2. `opd_transitive_tail_kl`: `6×43` at `(36, 0)` — `2 × 129` exact buckets, `G = 0.0`
 *   3. `opd_self_consistency_kl`: `6×43` at `(36, 10)` — `2 × 129` exact buckets, `G = 0.0`
 *   4. `opd_matched_sigmas`: `2×steps` at `(36, 20)` (`steps <= 88`, `G = 0.0`), or wrapped
 *      `rows×64` (`steps > 88`, `G = 0.0` on live steps, `G = 1.0` ONLY on trailing padding)
 *   5. `opd_flow_latents`: `3×16` at `(0, 36)` — `[x^0, δ, x^r]`, `G = 0.0`
 */
export function packUniOpdExplorerAtlas(
  understandingOpd: UniUnderstandingOpdResult,
  flowEditingOpd: UniFlowEditingOpdResult,
  extraKeyTag = '',
): UniOpdAtlasPackResult {
  const writer = createRg32fAtlas(UNI_OPD_ATLAS_WIDTH, UNI_OPD_ATLAS_HEIGHT);

  // --- Rect 1: 32×32 Dual-Stream [noisy || clean] SBD block_diffusion_mask (L=16, B=4) ---
  const cleanLen = UNI_OPD_SBD_CLEAN_LEN;
  const totalLen = cleanLen * 2; // 32
  const sbdMaskVisible = buildDualStreamBlockDiffusionMask(cleanLen, UNI_OPD_SBD_BLOCK_SIZE);
  const sbdR = new Float32Array(totalLen * totalLen);
  const sbdG = new Float32Array(totalLen * totalLen);

  for (let q = 0; q < totalLen; q++) {
    const rowBase = q * totalLen;
    for (let k = 0; k < totalLen; k++) {
      const idx = rowBase + k;
      if (sbdMaskVisible[idx] === 1) {
        sbdG[idx] = 0.0;
        // Differentiate the 3 visible attention zones while keeping all visible values in (0, 1]
        if (q < cleanLen && k < cleanLen) {
          sbdR[idx] = 0.95; // Noisy-to-same-noisy-block bidirectional diagonal
        } else if (q < cleanLen && k >= cleanLen) {
          sbdR[idx] = 0.62; // Noisy-to-strictly-earlier-clean-blocks cross-stream context
        } else {
          sbdR[idx] = 0.80; // Clean-to-clean causal lower triangle
        }
      } else {
        sbdR[idx] = 0.0;
        sbdG[idx] = 1.0; // Blocked cell -> hatched by SheetsPass!
      }
    }
  }

  const sbdEntry = packOpdMatrixToRg32f(writer, {
    id: 'sbd_dual_stream_mask',
    x0: 0,
    y0: 0,
    rows: totalLen,
    cols: totalLen,
    values: sbdR,
    gChannelFlags: sbdG,
    mode: 1,
    explicitScale: 1.0,
    isMaskOrScore: true,
  });

  // --- Rect 2: Transitive AR OPD 129-bucket tail_kl (6×43 = 3×43 Teacher + 3×43 Student AR) ---
  const tailCols = 43;
  const tailRows = 6;
  const bucketCount = 129; // 3 * 43 === 129 exact (zero padding!)
  const arTailVals = new Float32Array(tailRows * tailCols);
  for (let b = 0; b < bucketCount; b++) {
    arTailVals[b] = understandingOpd.arTailKl.pTargetBuckets[b] ?? 0;
    arTailVals[bucketCount + b] = understandingOpd.arTailKl.pStudentBuckets[b] ?? 0;
  }
  const arTailScale = 1.0 / computeMaxAbs(arTailVals, 1e-4);

  const transitiveEntry = packOpdMatrixToRg32f(writer, {
    id: 'opd_transitive_tail_kl',
    x0: 36,
    y0: 0,
    rows: tailRows,
    cols: tailCols,
    values: arTailVals,
    mode: 1,
    explicitScale: arTailScale,
    isMaskOrScore: false,
  });

  // --- Rect 3: Self-Consistency 129-bucket tail_kl (6×43 = 3×43 sg(Student AR) + 3×43 Student Diff) ---
  const consTailVals = new Float32Array(tailRows * tailCols);
  for (let b = 0; b < bucketCount; b++) {
    consTailVals[b] = understandingOpd.consistencyTailKl.pTargetBuckets[b] ?? 0;
    consTailVals[bucketCount + b] = understandingOpd.consistencyTailKl.pStudentBuckets[b] ?? 0;
  }
  const consTailScale = 1.0 / computeMaxAbs(consTailVals, 1e-4);

  const consistencyEntry = packOpdMatrixToRg32f(writer, {
    id: 'opd_self_consistency_kl',
    x0: 36,
    y0: 10,
    rows: tailRows,
    cols: tailCols,
    values: consTailVals,
    mode: 1,
    explicitScale: consTailScale,
    isMaskOrScore: false,
  });

  // --- Rect 4: Continuous-Flow Editing matched_sigmas(r, seqLen, steps) ---
  const matched = flowEditingOpd.matchedSigmas;
  const stepCount = matched.steps;
  const expMu = Math.exp(matched.mu);
  const tStart = matched.sigmaStart / (matched.sigmaStart + expMu * (1 - matched.sigmaStart));
  const tEnd = matched.sigmaEnd / (matched.sigmaEnd + expMu * (1 - matched.sigmaEnd));

  const maxLinearSteps = 88; // x0 = 36 -> 36 + 88 = 124 <= 128
  const matchedSigmasWrapped = stepCount > maxLinearSteps;
  let sigmaRows: number;
  let sigmaCols: number;
  let sigmaVals: Float32Array;
  let sigmaG: Float32Array | undefined;

  if (!matchedSigmasWrapped) {
    // 2×steps: Row 0 = dynamic-shifted sigma_k (r -> 0.02), Row 1 = unshifted linear t_k (G = 0.0 everywhere)
    sigmaRows = 2;
    sigmaCols = stepCount;
    sigmaVals = new Float32Array(sigmaRows * sigmaCols);
    for (let k = 0; k < stepCount; k++) {
      sigmaVals[k] = matched.sigmas[k]!;
      const t = tStart + (tEnd - tStart) * (k / (stepCount - 1));
      sigmaVals[stepCount + k] = Math.fround(t);
    }
  } else {
    // Wrapped multi-row layout (cols = 64, rows <= 40) with G = 1.0 ONLY on unused trailing padding
    sigmaCols = 64;
    sigmaRows = Math.min(40, Math.ceil(stepCount / sigmaCols));
    const totalSlots = sigmaRows * sigmaCols;
    sigmaVals = new Float32Array(totalSlots);
    sigmaG = new Float32Array(totalSlots);
    for (let idx = 0; idx < totalSlots; idx++) {
      if (idx < stepCount) {
        sigmaVals[idx] = matched.sigmas[idx]!;
        sigmaG[idx] = 0.0;
      } else {
        sigmaVals[idx] = 0.0;
        sigmaG[idx] = 1.0;
      }
    }
  }

  const sigmasEntry = packOpdMatrixToRg32f(writer, {
    id: 'opd_matched_sigmas',
    x0: 36,
    y0: 20,
    rows: sigmaRows,
    cols: sigmaCols,
    values: sigmaVals,
    ...(sigmaG !== undefined ? { gChannelFlags: sigmaG } : {}),
    mode: 1,
    explicitScale: 1.0,
    isMaskOrScore: false,
  });

  // --- Rect 5: Flow Interpolation [3, 16] Latent Comparison (x^0, δ, x^r) ---
  const latentCols = flowEditingOpd.x0.length; // 16
  const latentRows = 3;
  const latentVals = new Float32Array(latentRows * latentCols);
  for (let i = 0; i < latentCols; i++) {
    latentVals[i] = flowEditingOpd.x0[i]!;
    latentVals[latentCols + i] = flowEditingOpd.noiseDelta[i]!;
    latentVals[2 * latentCols + i] = flowEditingOpd.xNoisyR[i]!;
  }
  const latentScale = 1.0 / computeMaxAbs(latentVals, 0.25);

  const latentsEntry = packOpdMatrixToRg32f(writer, {
    id: 'opd_flow_latents',
    x0: 0,
    y0: 36,
    rows: latentRows,
    cols: latentCols,
    values: latentVals,
    mode: 0,
    explicitScale: latentScale,
    isMaskOrScore: false,
  });

  const regions: readonly UniAtlasRegionBox[] = Object.freeze([
    sbdEntry.region,
    transitiveEntry.region,
    consistencyEntry.region,
    sigmasEntry.region,
    latentsEntry.region,
  ]);

  const audit = auditAtlasRegions(
    UNI_OPD_ATLAS_WIDTH,
    UNI_OPD_ATLAS_HEIGHT,
    regions,
    UNI_OPD_ATLAS_BUDGET_BYTES,
  );

  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0 || !audit.withinBudget) {
    throw new Error(
      `OPD 128x64 RG32F atlas audit failed: overlap=${String(audit.overlapPair)}, oob=[${audit.outOfBoundsIds.join(',')}], bytes=${audit.byteLength}/${audit.maxBudgetBytes}`,
    );
  }

  const atlas: Rg32fValueAtlas = Object.freeze({
    width: UNI_OPD_ATLAS_WIDTH,
    height: UNI_OPD_ATLAS_HEIGHT,
    data: writer.data,
  });

  const atlasKey = computeAtlasContentKey('opd_readonly', atlas, extraKeyTag);

  const slices: Readonly<Record<UniOpdPlateId, UniOpdAtlasSliceEntry>> = Object.freeze({
    sbd_dual_stream_mask: sbdEntry,
    opd_transitive_tail_kl: transitiveEntry,
    opd_self_consistency_kl: consistencyEntry,
    opd_matched_sigmas: sigmasEntry,
    opd_flow_latents: latentsEntry,
  });

  return Object.freeze({
    atlas,
    atlasKey,
    regions,
    audit,
    slices,
    sbdMaskVisible,
    matchedSigmasWrapped,
  });
}

// ============================================================================
// MAIN HEADLESS SCENE BUILDER: VIEW 7 (`opd_readonly`)
// ============================================================================

interface OpdPlateLayoutSpec {
  readonly id: UniOpdPlateId;
  readonly bay: 'understanding' | 'editing';
  readonly branch: 'u' | 'g';
  readonly towerRole: 'context' | 'denoiser';
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ez: number;
  readonly sheetKind: number;
  readonly role: number;
  readonly semanticTag: string;
  readonly label: string;
  readonly sublabel: string;
  readonly badge?: string;
}

interface OpdCardLayoutSpec {
  readonly id: UniOpdStageCardId;
  readonly bay: 'understanding' | 'editing';
  readonly branch: 'u' | 'g';
  readonly towerRole: 'context' | 'denoiser';
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly cellFamily: number;
  readonly isWarningCard?: boolean;
  readonly semanticTag: string;
  readonly anchorId: string;
  readonly label: string;
  readonly sublabel: string;
  readonly tone: 'u_indigo' | 'g_teal' | 'warn' | 'readonly';
  readonly badge?: string;
}

/**
 * Build View 7 (`opd_readonly`): Strictly Read-Only Cross-Paradigm OPD Explorer.
 *
 * Works both standalone (`session === null`) and alongside any live `UniSession`
 * without ever mutating session state, computing gradients, or running backward passes.
 */
export function buildUniOpdExplorerScene(
  session: UniSession | null,
  options: UniOpdSceneOptions = {},
): UniSpatialScene {
  const seed = options.seed ?? UNI_OPD_DEFAULT_SEED;
  const r = options.r ?? UNI_OPD_DEFAULT_R;
  const steps = options.steps ?? UNI_OPD_DEFAULT_STEPS;
  const seqLen = options.seqLen ?? UNI_OPD_DEFAULT_SEQ_LEN;
  const selectedBay: UniOpdBayFocus = options.selectedBay ?? 'both';
  const hovered = options.hoveredCell ?? null;

  // Non-mutating optional session provenance resolution
  const sessionCtx =
    session !== null && session.getActiveRecordId() !== null
      ? resolveUniSessionContext(session)
      : null;

  const executionViewId: UniExecutionViewId = sessionCtx
    ? sessionCtx.executionViewId
    : 'interleaved_mixed';
  const prefillMode: UniElasticPrefillMode = sessionCtx
    ? sessionCtx.prefillMode
    : 'hf_faithful';

  // Strictly forward-only, zero-gradient OPD reference evaluations
  const demoInputs = createDefaultOpdDemoInputs(seed);
  const understandingOpd = evaluateUnderstandingOpdRef(
    demoInputs.teacherArProbs,
    demoInputs.studentArProbs,
    demoInputs.studentDiffProbs,
    { responseLen: demoInputs.responseLen },
  );
  const flowEditingOpd = evaluateFlowEditingOpdRef(
    demoInputs.x0,
    demoInputs.noiseDelta,
    demoInputs.exactLogits,
    demoInputs.targetCode,
    demoInputs.groupLogits,
    demoInputs.targetGroup,
    { r, seqLen, steps },
  );
  const u28Diagnostics: UniTrainerBugDiagnosticU28 = getTrainerBugDiagnosticsU28();

  const opdReadOnly: UniOpdReadOnlyMetadata = Object.freeze({
    readOnly: true,
    hasGradients: false,
    badgeLabel: UNI_OPD_READONLY_BADGE_LABEL,
    understandingOpd,
    flowEditingOpd,
    u28Diagnostics,
  });

  // Pack 128×64 RG32F scalar/mask atlas (exactly 5 non-overlapping rects, 65,536 B)
  const extraTag = `s${seed}_r${r.toFixed(3)}_st${steps}_q${seqLen}`;
  const packed = packUniOpdExplorerAtlas(understandingOpd, flowEditingOpd, extraTag);

  // Initialize 5-pass GPU builders
  const solids = new UniSolidBufferBuilder();
  const lines = new UniLineBufferBuilder();
  const ribbons = new UniRibbonBufferBuilder();
  const sheets = new UniSheetBufferBuilder();
  const flow = new UniFlatFlowBufferBuilder();

  const pickBoxes: PickBox[] = [];
  const overlayAnchors: UniOverlayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];

  const boundsAcc: { min: [number, number, number]; max: [number, number, number] } = {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };

  // Spatial layout constants
  const LEFT_BAY_X = -5.6;
  const RIGHT_BAY_X = 5.6;
  const CENTER_X = 0.0;
  const SHEET_Y = 0.12;
  const SHEET_EY = 0.07;
  const CARD_Y = 0.08;
  const CARD_EY = 0.16;
  const RIBBON_Y = 0.035;

  const leftActive = selectedBay !== 'editing';
  const rightActive = selectedBay !== 'understanding';

  // --- Architectural Drafting Table Bay Mats, Dividers & Read-Only Header Plinth ---
  // Subtle bay floor drafting outlines
  lines.addBoxFrameXZ(
    -11.9,
    -6.7,
    -2.2,
    7.45,
    0.01,
    leftActive ? 1.15 : 0.75,
    leftActive ? LINE_STYLE_NEUTRAL : LINE_STYLE_GUIDE,
    leftActive ? 0.58 : 0.25,
  );
  lines.addBoxFrameXZ(
    2.2,
    -6.7,
    9.1,
    6.65,
    0.01,
    rightActive ? 1.15 : 0.75,
    rightActive ? LINE_STYLE_NEUTRAL : LINE_STYLE_GUIDE,
    rightActive ? 0.58 : 0.25,
  );

  // Highlight active bay floor outline when a single bay is focused
  if (selectedBay === 'understanding') {
    flow.addCellHighlightXZ(
      -7.05,
      0.375,
      9.6,
      14.05,
      0.014,
      { r: 0.22, g: 0.38, b: 0.74, a: 0.06, borderAlpha: 0.55 },
      UNI_FLOW_COLORS.uIndigo,
      0.04,
    );
  } else if (selectedBay === 'editing') {
    flow.addCellHighlightXZ(
      5.65,
      -0.025,
      6.8,
      13.25,
      0.014,
      { r: 0.102, g: 0.541, b: 0.478, a: 0.06, borderAlpha: 0.55 },
      UNI_FLOW_COLORS.gTeal,
      0.04,
    );
  }

  // Subtle translucent bay zone backdrops in RibbonsPass
  ribbons.push({
    cx: -7.05,
    cy: 0.005,
    cz: 0.375,
    ex: 9.7,
    ey: 0.01,
    ez: 14.15,
    kind: 0,
    colorId: 0,
    alpha: leftActive ? 0.12 : 0.04,
  });
  ribbons.push({
    cx: 5.65,
    cy: 0.005,
    cz: -0.025,
    ex: 6.9,
    ey: 0.01,
    ez: 13.35,
    kind: 0,
    colorId: 1,
    alpha: rightActive ? 0.12 : 0.04,
  });

  // Center vertical read-only firewall divider between Left & Right Bays
  lines.addSegment(CENTER_X, 0.02, -6.9, CENTER_X, 0.02, 7.3, 1.1, LINE_STYLE_GUIDE, 0.48);

  // Top Read-Only Architectural Header Plinth (non-training firewall)
  solids.push({
    cx: CENTER_X,
    cy: 0.09,
    cz: -7.55,
    ex: 8.6,
    ey: 0.18,
    ez: 0.72,
    cellFamily: CELL_EMBED_SLATE,
    bayId: 0,
    highlight: 0,
  });
  lines.addBoxEdges(CENTER_X, 0.09, -7.55, 8.6, 0.18, 0.72, 1, 1.1, LINE_STYLE_NEUTRAL, 0.78);
  mergeAabb(boundsAcc, { min: [-12.1, -0.1, -8.2], max: [9.3, 0.9, 7.8] });

  // --- Define 5 Live RG32F Scalar/Mask Plates ---
  const arTail = understandingOpd.arTailKl;
  const consTail = understandingOpd.consistencyTailKl;
  const matched = flowEditingOpd.matchedSigmas;
  const matchedRegion = packed.slices.opd_matched_sigmas.region;

  const plateSpecs: readonly OpdPlateLayoutSpec[] = [
    // Left Bay (Understanding OPD) — 3 plates
    {
      id: 'sbd_dual_stream_mask',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_BAY_X,
      cy: SHEET_Y,
      cz: -2.3,
      ex: 4.0,
      ez: 4.0,
      sheetKind: SHEET_KIND_CAUSAL_SCORES, // Hard U25: live RG32F G=1.0 blocked / G=0.0 visible
      role: SHEET_ROLE_SCORES,
      semanticTag:
        'branch=u;bay=understanding;stage=sbd_dual_stream_mask;op=block_diffusion_mask;u25=live_rg32f;readonly=1',
      label: 'Dual-Stream [noisy || clean] SBD Mask (32×32)',
      sublabel:
        'L=16, B=4 · Noisy attends same block + earlier clean · Clean causal · G=1 blocked / G=0 visible',
      badge: 'U25 Live RG32F G-Mask',
    },
    {
      id: 'opd_transitive_tail_kl',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_BAY_X,
      cy: SHEET_Y,
      cz: 1.35,
      ex: 5.0,
      ez: 1.4,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      semanticTag:
        'branch=u;bay=understanding;stage=transitive_tail_kl;op=reverse_tail_kl_129;readonly=1',
      label: 'Transitive AR OPD · 129-Bucket tail_kl (6×43)',
      sublabel: `Rows 0..2 AR Teacher vs Rows 3..5 Student AR · 4.0 × D_KL = ${understandingOpd.arOpdLoss.toFixed(4)} (rev=${arTail.reverseKl.toFixed(4)}, tail=${arTail.pTargetTailMass.toFixed(3)}/${arTail.pStudentTailMass.toFixed(3)})`,
      badge: '128 Union + 1 Tail',
    },
    {
      id: 'opd_self_consistency_kl',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_BAY_X,
      cy: SHEET_Y,
      cz: 4.45,
      ex: 5.0,
      ez: 1.4,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      semanticTag:
        'branch=u;bay=understanding;stage=self_consistency_kl;op=forward_tail_kl_129;readonly=1',
      label: 'Self-Consistency · sg(Student AR) → Student Diff (6×43)',
      sublabel: `D_KL(sg(p_s^AR) || p_s^Diff) / (0.5005·R) = ${understandingOpd.selfConsistencyLoss.toFixed(4)} (fwd=${consTail.forwardKl.toFixed(4)}, denom=${understandingOpd.expectedMaskDenominator.toFixed(2)})`,
      badge: 'stopgrad(p_s^AR)',
    },

    // Right Bay (Continuous-Flow Editing OPD) — 2 plates
    {
      id: 'opd_flow_latents',
      bay: 'editing',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_BAY_X,
      cy: SHEET_Y,
      cz: -2.55,
      ex: 5.0,
      ez: 1.8,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      semanticTag: `branch=g;bay=editing;stage=flow_latents;op=linear_flow_noising;r=${r.toFixed(3)};readonly=1`,
      label: `Continuous-Flow Latents · x^r = (1 - r)·x^0 + r·δ (3×16)`,
      sublabel: `r = ${r.toFixed(2)} · Row 0: x^0 (VAE), Row 1: δ ~ N(0,I), Row 2: x^r = ${(1 - r).toFixed(2)}·x^0 + ${r.toFixed(2)}·δ`,
      badge: `r = ${r.toFixed(2)}`,
    },
    {
      id: 'opd_matched_sigmas',
      bay: 'editing',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_BAY_X,
      cy: SHEET_Y,
      cz: 0.45,
      ex: 5.4,
      ez: 1.4,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_NORM,
      semanticTag: `branch=g;bay=editing;stage=matched_sigmas;op=qwen_time_shift;steps=${steps};seqLen=${seqLen};readonly=1`,
      label: `Qwen-Image matched_sigmas (${matchedRegion.rows}×${matchedRegion.cols})`,
      sublabel: `σ: ${matched.sigmas[0]!.toFixed(3)} → ${matched.sigmas[matched.steps - 1]!.toFixed(3)} · μ(${seqLen}) = ${matched.mu.toFixed(4)} · ${steps} ODE steps`,
      badge: `steps = ${steps}`,
    },
  ];

  const plateFrames = {} as Record<UniOpdPlateId, SheetFrame3D>;

  for (let i = 0; i < plateSpecs.length; i++) {
    const spec = plateSpecs[i]!;
    const sliceEntry = packed.slices[spec.id];
    const gridCols = sliceEntry.region.cols;
    const gridRows = sliceEntry.region.rows;

    const frame = makeSheetFrame3D(
      [spec.cx, spec.cy, spec.cz],
      [spec.ex, SHEET_EY, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    plateFrames[spec.id] = frame;
    mergeAabb(boundsAcc, sheetFrameWorldAabb(frame));

    const isBayFocused =
      (spec.bay === 'understanding' && leftActive) || (spec.bay === 'editing' && rightActive);
    const isHoveredPlate =
      hovered !== null && (hovered.tensorKey === spec.id || hovered.partId === spec.id);

    const baseHl = !isBayFocused
      ? SHEET_HL_DIMMED
      : spec.bay === 'understanding'
        ? selectedBay === 'understanding'
          ? SHEET_HL_OPERAND_A
          : SHEET_HL_NORMAL
        : selectedBay === 'editing'
          ? SHEET_HL_OPERAND_B
          : SHEET_HL_NORMAL;

    const highlight = isHoveredPlate ? SHEET_HL_ACTIVE : baseHl;

    sheets.push({
      cx: spec.cx,
      cy: spec.cy,
      cz: spec.cz,
      ex: spec.ex,
      ey: SHEET_EY,
      ez: spec.ez,
      gridCols,
      gridRows,
      sheetKind: spec.sheetKind,
      role: spec.role,
      fp8: 0,
      highlight,
      seed: 0.71 + i * 0.43,
      slice: sliceEntry.slice,
      frame,
    });

    // Subtle bay-colored framing accent via FlatFlowRibbons (never misusing RG32F G-channel!)
    if (isBayFocused) {
      const bayColor =
        spec.bay === 'understanding' ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.gTeal;
      flow.addCellHighlightOnFrame(
        frame,
        0,
        gridRows - 1,
        0,
        gridCols - 1,
        gridRows,
        gridCols,
        SHEET_EY * 0.5 + 0.008,
        { r: bayColor.r, g: bayColor.g, b: bayColor.b, a: 0.0, borderAlpha: 0.42 },
        bayColor,
        0.016,
        false,
      );
    }

    // Highlight hovered cell if within plate grid bounds
    if (
      isHoveredPlate &&
      hovered !== null &&
      hovered.row !== null &&
      hovered.col !== null &&
      hovered.row >= 0 &&
      hovered.row < gridRows &&
      hovered.col >= 0 &&
      hovered.col < gridCols
    ) {
      flow.addCellHighlightOnFrame(
        frame,
        hovered.row,
        hovered.row,
        hovered.col,
        hovered.col,
        gridRows,
        gridCols,
        SHEET_EY * 0.5 + 0.015,
        { r: 0.863, g: 0.298, b: 0.204, a: 0.34, borderAlpha: 0.95 },
        UNI_FLOW_COLORS.activeOxide,
        0.024,
        true,
      );
    }

    // Emit PickBox via packPickBoxFromFrame for exact cell round-trip
    pickBoxes.push(
      packPickBoxFromFrame(frame, {
        bayId: 0,
        partId: spec.id,
        tensorKey: spec.id,
        gridCols,
        gridRows,
        towerRole: spec.towerRole,
        semanticTag: spec.semanticTag,
      }),
    );

    overlayAnchors.push(
      Object.freeze({
        id: `anchor_${spec.id}`,
        label: spec.label,
        sublabel: spec.sublabel,
        worldPos: [spec.cx, spec.cy + 0.22, spec.cz - spec.ez * 0.5 - 0.16] as Vec3,
        tone: spec.bay === 'understanding' ? 'u_indigo' : 'g_teal',
        ...(spec.badge !== undefined ? { badge: spec.badge } : {}),
      }),
    );
  }

  // --- Define Disjoint Architectural Stage Cards (Left Bay + Right Bay + Callouts) ---
  const { sftDoubleNorm, accidentalImEndSupervision, opdLocTailLeak } = u28Diagnostics;

  const cardSpecs: readonly OpdCardLayoutSpec[] = [
    // Left Bay Stage Cards
    {
      id: 'opd_u_ar_teacher_card',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: -7.2,
      cy: CARD_Y,
      cz: -5.6,
      ex: 2.6,
      ey: CARD_EY,
      ez: 1.0,
      cellFamily: CELL_WEIGHT_BLUE,
      semanticTag: 'branch=u;bay=understanding;stage=ar_teacher;op=frozen_teacher_ar;readonly=1',
      anchorId: 'anchor_opd_u_ar_teacher',
      label: 'AR VLM Teacher (p_t^AR)',
      sublabel: `Frozen 131,072-vocab teacher · Top-64 support`,
      tone: 'u_indigo',
      badge: 'Frozen Teacher',
    },
    {
      id: 'opd_u_student_ar_card',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: -4.0,
      cy: CARD_Y,
      cz: -5.6,
      ex: 2.6,
      ey: CARD_EY,
      ez: 1.0,
      cellFamily: CELL_WEIGHT_BLUE,
      semanticTag: 'branch=u;bay=understanding;stage=student_ar;op=on_policy_rollout;readonly=1',
      anchorId: 'anchor_opd_u_student_ar',
      label: 'Student AR View (p_s^AR)',
      sublabel: `On-policy rollout X^0 (R=${understandingOpd.responseLen}) · Top-128 union`,
      tone: 'u_indigo',
      badge: 'On-Policy AR',
    },
    {
      id: 'opd_u_stopgrad_card',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_BAY_X,
      cy: CARD_Y,
      cz: 2.85,
      ex: 3.8,
      ey: CARD_EY,
      ez: 0.72,
      cellFamily: CELL_EMBED_SLATE,
      semanticTag: 'branch=u;bay=understanding;stage=stopgrad;op=stopgrad_bridge;readonly=1',
      anchorId: 'anchor_opd_u_stopgrad',
      label: 'stopgrad(p_s^AR) → Student Diff (p_s^Diff)',
      sublabel: `Transitive bridge · Total U-OPD = ${understandingOpd.totalUnderstandingOpdLoss.toFixed(4)} (READ-ONLY)`,
      tone: 'readonly',
      badge: 'stopgrad',
    },
    {
      id: 'opd_u_new_francisco_callout',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: -10.1,
      cy: CARD_Y,
      cz: 1.35,
      ex: 2.8,
      ey: CARD_EY,
      ez: 1.8,
      cellFamily: CELL_NEUTRAL,
      isWarningCard: true,
      semanticTag:
        'branch=u;bay=understanding;stage=new_francisco_callout;op=context_mismatch_artifact;readonly=1',
      anchorId: 'anchor_opd_new_francisco',
      label: 'Why Transitive? "New Francisco" Context-Mismatch',
      sublabel:
        'Direct AR Teacher → Student Diff score copying mixes incompatible bidirectional & causal contexts ("New" + "Francisco" instead of "New York" / "San Francisco")',
      tone: 'warn',
      badge: 'Context Mismatch',
    },
    {
      id: 'opd_u28_diagnostics_card',
      bay: 'understanding',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_BAY_X,
      cy: CARD_Y,
      cz: 6.45,
      ex: 5.4,
      ey: CARD_EY,
      ez: 1.25,
      cellFamily: CELL_NEUTRAL,
      isWarningCard: true,
      semanticTag:
        'branch=u;bay=understanding;stage=u28_diagnostics;op=minimal_training_3_bugs;readonly=1',
      anchorId: 'anchor_opd_u28_diagnostics',
      label: 'U28 Reference-Trainer Diagnostics (3 Verified Bugs)',
      sublabel: `1) SFT Double-Norm: ~${sftDoubleNorm.expectedInflationFactor.toFixed(3)}× avg / ${sftDoubleNorm.peakSpikeAtTMin.toLocaleString('en-US')}× spike @ t=${sftDoubleNorm.tMin} · 2) Accidental <|im_end|> (id=${accidentalImEndSupervision.eosTokenId}) unmasking on ${accidentalImEndSupervision.affectedTurns} · 3) ${opdLocTailLeak.fullVocabSize.toLocaleString('en-US')} - ${opdLocTailLeak.topK} = ${opdLocTailLeak.leakedTailDenom.toLocaleString('en-US')} <LOC_*> tail leak (${opdLocTailLeak.locExtensionTokens.toLocaleString('en-US')} tokens)`,
      tone: 'warn',
      badge: 'U28 · 3 Trainer Bugs',
    },

    // Right Bay Stage Cards
    {
      id: 'opd_g_vq_rollout_card',
      bay: 'editing',
      branch: 'g',
      towerRole: 'denoiser',
      cx: 4.0,
      cy: CARD_Y,
      cz: -5.6,
      ex: 2.6,
      ey: CARD_EY,
      ez: 1.0,
      cellFamily: CELL_SSM_CYAN,
      semanticTag: 'branch=g;bay=editing;stage=vq_rollout;op=student_vq_rollout;readonly=1',
      anchorId: 'anchor_opd_g_vq_rollout',
      label: 'Student VQ Rollout',
      sublabel: 'V_out = Decode_VQ(X_out^0) (64 steps, crop=False)',
      tone: 'g_teal',
      badge: 'Discrete VQ',
    },
    {
      id: 'opd_g_vae_encode_card',
      bay: 'editing',
      branch: 'g',
      towerRole: 'denoiser',
      cx: 7.2,
      cy: CARD_Y,
      cz: -5.6,
      ex: 2.6,
      ey: CARD_EY,
      ez: 1.0,
      cellFamily: CELL_SSM_CYAN,
      semanticTag: 'branch=g;bay=editing;stage=vae_encode;op=teacher_vae_encode;readonly=1',
      anchorId: 'anchor_opd_g_vae_encode',
      label: 'Teacher VAE Encode',
      sublabel: 'x^0 = Encode_VAE(V_out) → continuous flow latent',
      tone: 'g_teal',
      badge: 'VAE Latent',
    },
    {
      id: 'opd_g_teacher_ode_card',
      bay: 'editing',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_BAY_X,
      cy: CARD_Y,
      cz: 2.85,
      ex: 4.6,
      ey: CARD_EY,
      ez: 1.1,
      cellFamily: CELL_ACTIVATION_GREEN,
      semanticTag: `branch=g;bay=editing;stage=teacher_ode;op=multistep_teacher_ode;steps=${steps};readonly=1`,
      anchorId: 'anchor_opd_g_teacher_ode',
      label: 'Multi-Step Teacher ODE (Qwen-Image)',
      sublabel: `ODE denoise x^r → x̂^0 over ${steps} matched_sigmas steps (true_cfg_scale=4.0)`,
      tone: 'g_teal',
      badge: 'Continuous Flow ODE',
    },
    {
      id: 'opd_g_requant_target_card',
      bay: 'editing',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_BAY_X,
      cy: CARD_Y,
      cz: 5.35,
      ex: 5.0,
      ey: CARD_EY,
      ez: 1.4,
      cellFamily: CELL_ACTIVATION_GREEN,
      semanticTag: 'branch=g;bay=editing;stage=requant_target;op=exact_ce_plus_0p2_grouped_ce;readonly=1',
      anchorId: 'anchor_opd_g_requant_target',
      label: 'Re-Quantized Target · exact_ce + 0.2 * grouped_ce',
      sublabel: `exact_ce=${flowEditingOpd.exactCe.toFixed(4)} + 0.2 × grouped_ce(${flowEditingOpd.groupedCe.toFixed(4)}) = ${flowEditingOpd.totalEditingOpdLoss.toFixed(4)} (stopgrad)`,
      tone: 'g_teal',
      badge: 'Hybrid VQ CE',
    },
  ];

  const cardFrames = {} as Record<UniOpdStageCardId, SheetFrame3D>;

  for (let i = 0; i < cardSpecs.length; i++) {
    const card = cardSpecs[i]!;
    const isBayFocused =
      (card.bay === 'understanding' && leftActive) || (card.bay === 'editing' && rightActive);
    const isHoveredCard =
      hovered !== null && (hovered.partId === card.id || hovered.tensorKey === card.id);

    solids.push({
      cx: card.cx,
      cy: card.cy,
      cz: card.cz,
      ex: card.ex,
      ey: card.ey,
      ez: card.ez,
      cellFamily: isBayFocused ? card.cellFamily : CELL_NEUTRAL,
      bayId: 0,
      highlight: isHoveredCard ? 1 : 0,
    });

    lines.addBoxEdges(
      card.cx,
      card.cy,
      card.cz,
      card.ex,
      card.ey,
      card.ez,
      10 + i,
      card.isWarningCard ? 1.2 : 0.95,
      card.isWarningCard ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      isBayFocused ? (card.isWarningCard ? 0.88 : 0.68) : 0.28,
    );

    if (card.isWarningCard) {
      lines.addDashedFrameXZ(
        card.cx - card.ex * 0.5 - 0.1,
        card.cz - card.ez * 0.5 - 0.1,
        card.cx + card.ex * 0.5 + 0.1,
        card.cz + card.ez * 0.5 + 0.1,
        0.02,
        LINE_STYLE_ACTIVE,
        isBayFocused ? 0.78 : 0.3,
      );
    }

    const cardFrame = makeSheetFrame3D(
      [card.cx, card.cy, card.cz],
      [card.ex, card.ey, card.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    cardFrames[card.id] = cardFrame;
    mergeAabb(boundsAcc, sheetFrameWorldAabb(cardFrame));

    pickBoxes.push(
      packPickBoxFromFrame(cardFrame, {
        bayId: 0,
        partId: card.id,
        tensorKey: card.id,
        gridCols: 1,
        gridRows: 1,
        towerRole: card.towerRole,
        semanticTag: card.semanticTag,
      }),
    );

    overlayAnchors.push(
      Object.freeze({
        id: card.anchorId,
        label: card.label,
        sublabel: card.sublabel,
        worldPos: [card.cx, card.cy + 0.24, card.cz] as Vec3,
        tone: card.tone,
        ...(card.badge !== undefined ? { badge: card.badge } : {}),
      }),
    );
  }

  // --- FlatFlowRibbons: Left Bay Transitive Understanding OPD Flow ---
  const xz = (x: number, z: number) => ({ x, z });
  const uFlowColor = leftActive ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.ghostSlate;
  const gFlowColor = rightActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate;
  const warnFlowColor = leftActive ? UNI_FLOW_COLORS.wasteAmber : UNI_FLOW_COLORS.ghostSlate;

  // AR Teacher -> Student AR (horizontal reverse tail_kl coupling)
  flow.addPolylineXZ(
    [xz(-5.85, -5.6), xz(-5.35, -5.6)],
    RIBBON_Y,
    0.16,
    uFlowColor,
    true,
  );
  // AR Teacher + Student AR -> Dual-Stream SBD Mask (32x32)
  flow.addCubicXZ(
    xz(-7.2, -5.05),
    xz(-7.2, -4.55),
    xz(-6.2, -4.65),
    xz(-6.2, -4.35),
    RIBBON_Y,
    0.15,
    uFlowColor,
    true,
    20,
  );
  flow.addCubicXZ(
    xz(-4.0, -5.05),
    xz(-4.0, -4.55),
    xz(-5.0, -4.65),
    xz(-5.0, -4.35),
    RIBBON_Y,
    0.15,
    uFlowColor,
    true,
    20,
  );
  // Dual-Stream SBD Mask -> Transitive AR OPD tail_kl plate
  flow.addPolylineXZ(
    [xz(LEFT_BAY_X, -0.25), xz(LEFT_BAY_X, 0.58)],
    RIBBON_Y,
    0.18,
    uFlowColor,
    true,
  );
  // Transitive tail_kl -> stopgrad barrier -> Self-Consistency tail_kl (Student Diff)
  flow.addPolylineXZ(
    [xz(LEFT_BAY_X, 2.08), xz(LEFT_BAY_X, 2.45)],
    RIBBON_Y,
    0.18,
    uFlowColor,
    true,
  );
  flow.addPolylineXZ(
    [xz(LEFT_BAY_X, 3.24), xz(LEFT_BAY_X, 3.68)],
    RIBBON_Y,
    0.18,
    uFlowColor,
    true,
  );
  // Warning connector from Transitive tail_kl to "New Francisco" context-mismatch callout
  flow.addPolylineXZ(
    [xz(-8.15, 1.35), xz(-8.65, 1.35)],
    RIBBON_Y,
    0.12,
    warnFlowColor,
    true,
  );
  // Warning connector from Self-Consistency plate down to U28 Reference-Trainer Diagnostics card
  flow.addPolylineXZ(
    [xz(LEFT_BAY_X, 5.18), xz(LEFT_BAY_X, 5.76)],
    RIBBON_Y,
    0.14,
    warnFlowColor,
    true,
  );

  // --- FlatFlowRibbons: Right Bay Continuous-Flow Editing OPD Flow ---
  // Student VQ Rollout -> VAE Encode
  flow.addPolylineXZ(
    [xz(5.35, -5.6), xz(5.85, -5.6)],
    RIBBON_Y,
    0.16,
    gFlowColor,
    true,
  );
  // VAE Encode -> Flow Interpolation Latents (x^r)
  flow.addCubicXZ(
    xz(7.2, -5.05),
    xz(7.2, -4.15),
    xz(RIGHT_BAY_X, -4.25),
    xz(RIGHT_BAY_X, -3.5),
    RIBBON_Y,
    0.18,
    gFlowColor,
    true,
    20,
  );
  // Flow Latents (x^r) -> matched_sigmas schedule
  flow.addPolylineXZ(
    [xz(RIGHT_BAY_X, -1.62), xz(RIGHT_BAY_X, -0.32)],
    RIBBON_Y,
    0.18,
    gFlowColor,
    true,
  );
  // matched_sigmas -> Multi-Step Teacher ODE
  flow.addPolylineXZ(
    [xz(RIGHT_BAY_X, 1.18), xz(RIGHT_BAY_X, 2.24)],
    RIBBON_Y,
    0.18,
    gFlowColor,
    true,
  );
  // Multi-Step Teacher ODE -> Re-Quantized Target (exact_ce + 0.2 * grouped_ce)
  flow.addPolylineXZ(
    [xz(RIGHT_BAY_X, 3.44), xz(RIGHT_BAY_X, 4.58)],
    RIBBON_Y,
    0.18,
    gFlowColor,
    true,
  );

  // --- Top Banner Anchor, Landmarks & Dimension Calipers ---
  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_opd_readonly_banner',
      label: UNI_OPD_READONLY_BADGE_LABEL,
      sublabel:
        'Cross-Paradigm On-Policy Distillation Reference · Forward-Only · 0 Backward Passes · 0 Optimizer Steps',
      worldPos: [CENTER_X, 0.34, -7.55] as Vec3,
      tone: 'readonly',
      badge: UNI_OPD_READONLY_BADGE_LABEL,
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_opd_readonly_boundary',
      label: UNI_OPD_READONLY_BADGE_LABEL,
      sublabel: 'Strictly Forward-Only Reference (0 Gradients)',
      world: [CENTER_X, 0.55, -7.55] as Vec3,
      tone: 'gold',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_opd_understanding_bay',
      label: 'Left Bay · Transitive Understanding OPD',
      sublabel: `AR Teacher → Student AR → stopgrad → Student Diff · Loss ${understandingOpd.totalUnderstandingOpdLoss.toFixed(4)}`,
      world: [LEFT_BAY_X, 0.48, -6.55] as Vec3,
      tone: 'default',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_opd_editing_bay',
      label: 'Right Bay · Continuous-Flow Editing OPD',
      sublabel: `Student VQ → VAE → x^r (r=${r.toFixed(2)}) → ODE → exact_ce + 0.2·grouped_ce = ${flowEditingOpd.totalEditingOpdLoss.toFixed(4)}`,
      world: [RIGHT_BAY_X, 0.48, -6.55] as Vec3,
      tone: 'cyan',
      align: 'top',
      tier: 'primary',
    }),
  );

  const sbdFrame = plateFrames.sbd_dual_stream_mask;
  const transFrame = plateFrames.opd_transitive_tail_kl;
  const sigmasFrame = plateFrames.opd_matched_sigmas;
  const latentsFrame = plateFrames.opd_flow_latents;

  calipers.push(
    Object.freeze({
      id: 'cal_sbd_dual_stream_mask',
      aWorld: [
        sbdFrame.center[0] - sbdFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        sbdFrame.center[2] - sbdFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        sbdFrame.center[0] + sbdFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        sbdFrame.center[2] - sbdFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: '32×32 [noisy||clean] SBD Mask (B=4)',
      partIds: ['sbd_dual_stream_mask'],
    }),
    Object.freeze({
      id: 'cal_opd_transitive_tail_kl',
      aWorld: [
        transFrame.center[0] - transFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        transFrame.center[2] - transFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        transFrame.center[0] + transFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        transFrame.center[2] - transFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: '6×43 (2 × 129 tail_kl buckets)',
      partIds: ['opd_transitive_tail_kl'],
    }),
    Object.freeze({
      id: 'cal_opd_matched_sigmas',
      aWorld: [
        sigmasFrame.center[0] - sigmasFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        sigmasFrame.center[2] - sigmasFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        sigmasFrame.center[0] + sigmasFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        sigmasFrame.center[2] - sigmasFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: `${matchedRegion.rows}×${matchedRegion.cols} matched_sigmas(r=${r.toFixed(2)})`,
      partIds: ['opd_matched_sigmas'],
    }),
    Object.freeze({
      id: 'cal_opd_flow_latents',
      aWorld: [
        latentsFrame.center[0] - latentsFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        latentsFrame.center[2] - latentsFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        latentsFrame.center[0] + latentsFrame.extent[0] * 0.5,
        SHEET_Y + 0.06,
        latentsFrame.center[2] - latentsFrame.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: '3×16 [x^0, δ, x^r]',
      partIds: ['opd_flow_latents'],
    }),
  );

  // --- Prominent Read-Only & Diagnostic Provenance Badges ---
  const provenanceBadges: UniProvenanceBadge[] = [
    Object.freeze({
      label: UNI_OPD_READONLY_BADGE_LABEL,
      tone: 'readonly',
    }),
    Object.freeze({
      label: 'Forward-Only Reference · 0 Backward Passes · 0 Optimizer Steps',
      tone: 'readonly',
    }),
    Object.freeze({
      label: `U24: 129-Bucket tail_kl (4.0×rev + fwd/0.5005R) & exact_ce + 0.2×grouped_ce`,
      tone: 'u_indigo',
    }),
    Object.freeze({
      label: `U28: 3 Minimal-Trainer Bugs (~${sftDoubleNorm.expectedInflationFactor.toFixed(3)}×/1,000× SFT, <|im_end|> id=11, 132,101-64 <LOC_*> Leak)`,
      tone: 'warn',
    }),
    Object.freeze({
      label: 'D4: 128×64 RG32F Scalar/Mask Only (64.00 KiB · 0 RGB Previews)',
      tone: 'muted',
    }),
    Object.freeze({
      label: sessionCtx
        ? `Session Attached (${sessionCtx.viewReadout.label} · Non-Mutating Read-Only OPD)`
        : 'Standalone Read-Only Mode (session = null)',
      tone: 'muted',
    }),
  ];

  // --- Pack 5-Pass Buffers & Finalize Scene ---
  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flow.pack();

  const fitBounds: AABB = Object.freeze({
    min: [boundsAcc.min[0] - 0.5, boundsAcc.min[1] - 0.2, boundsAcc.min[2] - 0.5] as Vec3,
    max: [boundsAcc.max[0] + 0.5, boundsAcc.max[1] + 1.1, boundsAcc.max[2] + 0.5] as Vec3,
  });

  const scene: UniSpatialScene = Object.freeze({
    viewMode: 'opd_readonly' as const,
    title: 'View 7 · Cross-Paradigm OPD Explorer (Read-Only)',
    subtitle: `${UNI_OPD_READONLY_BADGE_LABEL} · 129-bucket tail_kl & matched_sigmas(r=${r.toFixed(2)}, steps=${steps}) · ${packed.regions.length} RG32F rects (${packed.audit.utilizationPct.toFixed(1)}%)`,
    layerIndex: null,
    executionViewId,
    prefillMode,
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
      azimuthDeg: 14,
      elevationDeg: 48,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: packed.atlas,
    atlasKey: packed.atlasKey,
    atlasRegions: packed.regions,
    atlasAudit: packed.audit,
    rgbPreviews: Object.freeze([]),
    provenanceBadges: Object.freeze(provenanceBadges),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    d5Comparison: sessionCtx ? sessionCtx.d5Comparison : null,
    viewReadout: sessionCtx ? sessionCtx.viewReadout : null,
    opdReadOnly,
  });

  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `Invalid View 7 (opd_readonly) scene geometry: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
