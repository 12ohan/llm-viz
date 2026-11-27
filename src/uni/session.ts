/**
 * Bounded `<= 12.00 MiB` Interactive Session, 0-ULP Lazy Reconstruction &
 * `Set<ArrayBufferLike>` Memory Auditor (`src/uni/session.ts`)
 *
 * Clean-room headless TypeScript session engine for Nemotron-Labs-Diffusion-Uni:
 *   - Hard bounds: `maxBranches = 12` (`UNI_SESSION_MAX_BRANCHES`),
 *     `maxRecords = 128` (`UNI_SESSION_MAX_RECORDS`),
 *     `maxAllowedBytes = 12,582,912 B` (`12.00 MiB`, `UNI_MAX_SESSION_RETAINED_BYTES`).
 *   - Supports all 5 interactive task modes:
 *       * `text_understanding` (block-diffusion text decode)
 *       * `visual_understanding` (Mini RADIO-H + block-diffusion text decode)
 *       * `object_grounding` (Mini RADIO-H + `<LOC_0..16>` iterative unmasking)
 *       * `text_to_image` (`B_cfg ∈ {1, 2}` progressive semantic-tree diffusion)
 *       * `image_edit` (`B_cfg = 3` Dual-CFG editing with RADIO-H + Emu3VQ + `U18` lock)
 *   - Zero-duplication immutable `prefixCache` slab sharing on `forkBranch`, plus
 *     live `D5` `prefillMode` switching (`hf_faithful` <-> `elastic_pruned`).
 *   - Compact history records retain ONLY tight `[B_cfg][35]` layer residual
 *     checkpoints (`[seqLen, 8]`) + compact step state (NEVER full logits or traces).
 *   - Exact `0`-ULP lazy reconstruction for any historical record & layer
 *     (`reconstructLayerTrace`, `reconstructStepHeadLogits`) and active-record-only
 *     eager inspection (`UNI_SESSION_EAGER_TRACE_LAYERS = [0, 17, 18, 33]`).
 *   - Strict `Set<ArrayBufferLike>` memory auditor (`auditMemory`,
 *     `measureSessionRetainedBytes`) excluding shared read-only model weights.
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or sibling-app imports.
 */

import { applyRoutedUniRmsNorm, linearProject, softmaxRows } from './attention';
import { expandPackedImageHiddenStates } from './codec';
import { embedUniTextTokens, runUniElasticLayer, runUniForwardPass } from './forward';
import {
  applyEditDualCfgLogits,
  applyT2iCfgLogits,
  createDefaultUniRuntimeBundle,
  decodeMarkedCodesToImage,
  prepareImageEditPrefix,
  prepareT2iPrefix,
  runSingleImageDenoiseRound,
  type UniFullRuntimeBundle,
  type UniPreparedGenerationPrefix,
} from './imageSampler';
import {
  appendSingleRowKvsToCache,
  createEmptyUniKvCache,
  cropUniKvCache,
  extractRowPastKvs,
} from './kvCache';
import { canonicalizeTokenRouting } from './masks';
import { createXoshiro128ss, mixSeed32 } from './prng';
import { computeUniYarnInvFreq } from './rope';
import { sampleLogDomainGumbelArgmax } from './textSampler';
import {
  initSemanticTreeState,
  semanticTreeActivationCounts,
  stratifiedRandom,
} from './tree';
import {
  UNI_MAX_SESSION_RETAINED_BYTES,
  UNI_MINI_BOS_TOKEN_ID,
  UNI_MINI_BOX_CLOSE_TOKEN_ID,
  UNI_MINI_BOX_OPEN_TOKEN_ID,
  UNI_MINI_CONFIG,
  UNI_MINI_EOS_TOKEN_ID,
  UNI_MINI_IM_START_TOKEN_ID,
  UNI_MINI_LOC_BEGIN_TOKEN_ID,
  UNI_MINI_LOC_END_ID,
  UNI_MINI_LOC_END_TOKEN_ID,
  UNI_MINI_LOC_START_ID,
  UNI_MINI_MASK_TOKEN_ID,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_UNK_TOKEN_ID,
  UNI_MINI_VOCAB,
  UNI_ROUTE_G,
  UNI_SESSION_EAGER_TRACE_LAYERS,
  UNI_SESSION_MAX_BRANCHES,
  UNI_SESSION_MAX_RECORDS,
  UNI_STRATIFIED_DEFAULT_SEED,
  type UniAttentionMaskMode,
  type UniCfgBatchSize,
  type UniEditGuidanceMode,
  type UniElasticPrefillMode,
  type UniGainPresetId,
  type UniKvCacheState,
  type UniLayerKvSlice,
  type UniLayerTrace,
  type UniLetterboxInfo,
  type UniRgbImage,
  type UniScenePresetId,
  type UniSemanticTreeSchedule,
  type UniSemanticTreeState,
  type UniSessionActiveInspection,
  type UniSessionBranch,
  type UniSessionMemoryAudit,
  type UniSessionRecord,
  type UniSessionRecordKind,
  type UniSessionSnapshot,
  type UniSessionTaskMode,
  type UniXoshiro128ssGenerator,
} from './types';
import {
  createUniScenePreset,
  fixOnePrompt,
  runMiniVisionPipeline,
  spliceVisualEmbeddingsIntoPrompt,
} from './vision';

// ============================================================================
// PUBLIC SESSION OPTIONS & RECONSTRUCTION TYPES
// ============================================================================

export interface UniSessionStartBranchOptions {
  readonly taskMode: UniSessionTaskMode;
  readonly label?: string;
  readonly promptTokenIds?: ArrayLike<number>;
  readonly uncondPromptTokenIds?: ArrayLike<number>;
  readonly scenePresetId?: UniScenePresetId;
  readonly sourceImage?: UniRgbImage;
  readonly prefillMode?: UniElasticPrefillMode;
  readonly gainPresetId?: UniGainPresetId;
  readonly guidanceMode?: UniEditGuidanceMode;
  readonly guidanceScaleText?: number;
  readonly guidanceScaleImage?: number;
  readonly refinementSteps?: 1 | 2 | 4 | 8;
  readonly activationSteps?: 4 | 8 | 16;
  readonly shift?: number;
  readonly temperature?: number;
  readonly topK?: number;
  readonly topP?: number;
  readonly lockSourcePaddingVq?: boolean;
  readonly cropToSource?: boolean;
  readonly seed?: number;
  readonly stratifiedSeed?: number;
  // Text / visual understanding options
  readonly blockLength?: number;
  readonly stepsPerBlock?: number;
  readonly maxNewTokens?: number;
  readonly confidenceThreshold?: number;
  readonly causalContext?: boolean;
  readonly suppressMaskLogit?: boolean;
  // Object grounding options
  readonly groundingTags?: readonly string[];
  readonly groundingSteps?: 1 | 2 | 4;
  readonly enforceLocDomainGuard?: boolean;
}

export interface UniSessionForkBranchOptions {
  readonly label?: string;
  readonly prefillMode?: UniElasticPrefillMode;
  readonly guidanceMode?: UniEditGuidanceMode;
  readonly guidanceScaleText?: number;
  readonly guidanceScaleImage?: number;
  readonly seed?: number;
}

export interface UniReconstructedHeadLogits {
  readonly perBranchFinalNorm: readonly (Float32Array | null)[];
  readonly perBranchTextLogits: readonly (Float32Array | null)[];
  readonly perBranchGenLogits: readonly (Float32Array | null)[];
  readonly guidedGenLogits: Float32Array | null;
}

// ============================================================================
// DEFAULT PROMPTS & TIGHT-BUFFER HELPERS
// ============================================================================

const EMPTY_TRACE_LAYERS: ReadonlySet<number> = new Set<number>();

const DEFAULT_YARN = computeUniYarnInvFreq(UNI_MINI_CONFIG.rope);

const DEFAULT_SESSION_SEED = 0x554e4933; // "UNI3"

/**
 * Extract contiguous `[seqLen, 16]` K/V slice and `[seqLen]` routing for a single
 * layer `layerIndex` and CFG row `batchRow` (bit-exact equivalent to
 * `extractRowPastKvs(cache, batchRow)` at `pastKvs[layerIndex]`, avoiding 33 unused layers).
 */
function extractSingleLayerRowPastKv(
  cache: UniKvCacheState,
  batchRow: number,
  layerIndex: number,
): { pastKv: UniLayerKvSlice | null; pastRouting: Uint8Array } {
  const seqLen = cache.seqLen;
  const kvDim = UNI_MINI_CONFIG.kvProjOutDim;
  const pastRouting = new Uint8Array(seqLen);
  let anyAllocated = false;

  for (let t = 0; t < seqLen; t++) {
    const slab = cache.slabs[t];
    pastRouting[t] = slab.routing[batchRow];
    if (slab.layers[layerIndex] !== null) {
      anyAllocated = true;
    }
  }

  if (seqLen === 0 || !anyAllocated) {
    return { pastKv: null, pastRouting };
  }

  const rowOffset = batchRow * kvDim;
  const k = new Float32Array(seqLen * kvDim);
  const v = new Float32Array(seqLen * kvDim);

  for (let t = 0; t < seqLen; t++) {
    const slab = cache.slabs[t];
    const layerKv = slab.layers[layerIndex];
    if (layerKv === null) {
      if (slab.routing[batchRow] === UNI_ROUTE_G) {
        throw new RangeError(
          `Missing KV slice at layer ${layerIndex}, pos ${t}, batchRow ${batchRow} for G token`,
        );
      }
      continue;
    }
    const dstOffset = t * kvDim;
    for (let c = 0; c < kvDim; c++) {
      k[dstOffset + c] = layerKv.k[rowOffset + c];
      v[dstOffset + c] = layerKv.v[rowOffset + c];
    }
  }

  return { pastKv: { k, v, seqLen }, pastRouting };
}

const DEFAULT_PROMPTS: Readonly<Record<UniSessionTaskMode, readonly number[]>> = Object.freeze({
  text_understanding: Object.freeze([
    UNI_MINI_BOS_TOKEN_ID,
    UNI_MINI_IM_START_TOKEN_ID,
    26, // the
    27, // image
    28, // shows
    3,  // a
    6,  // cat
  ]),
  visual_understanding: Object.freeze([
    UNI_MINI_BOS_TOKEN_ID,
    UNI_MINI_IM_START_TOKEN_ID,
    UNI_MINI_UND_PAD_TOKEN_ID, // <SPECIAL_23>
    26, // the
    27, // image
    28, // shows
  ]),
  object_grounding: Object.freeze([
    UNI_MINI_BOS_TOKEN_ID,
    UNI_MINI_IM_START_TOKEN_ID,
    UNI_MINI_UND_PAD_TOKEN_ID, // <SPECIAL_23>
    26, // the
    27, // image
  ]),
  text_to_image: Object.freeze([
    UNI_MINI_BOS_TOKEN_ID,
    UNI_MINI_IM_START_TOKEN_ID,
    29, // generate
    3,  // a
    4,  // red
    6,  // cat
  ]),
  image_edit: Object.freeze([
    UNI_MINI_BOS_TOKEN_ID,
    UNI_MINI_IM_START_TOKEN_ID,
    UNI_MINI_UND_PAD_TOKEN_ID, // <SPECIAL_23>
    30, // edit
    26, // the
    27, // image
    36, // bright
    9,  // sky
  ]),
});

/** Ensure a Float32Array owns an exact-sized tight ArrayBuffer (never retaining a larger slice). */
function tightenFloat32(arr: Float32Array): Float32Array {
  if (arr.byteOffset === 0 && arr.byteLength === arr.buffer.byteLength) {
    return arr;
  }
  return new Float32Array(arr);
}

/** Ensure an Int32Array owns an exact-sized tight ArrayBuffer. */
function tightenInt32(arr: Int32Array): Int32Array {
  if (arr.byteOffset === 0 && arr.byteLength === arr.buffer.byteLength) {
    return arr;
  }
  return new Int32Array(arr);
}

/** Ensure a Uint8Array owns an exact-sized tight ArrayBuffer. */
function tightenUint8(arr: Uint8Array): Uint8Array {
  if (arr.byteOffset === 0 && arr.byteLength === arr.buffer.byteLength) {
    return arr;
  }
  return new Uint8Array(arr);
}

function tightenResidualCheckpoints(
  checkpoints: readonly (Float32Array | null)[],
): readonly (Float32Array | null)[] {
  const out = new Array<Float32Array | null>(checkpoints.length);
  for (let i = 0; i < checkpoints.length; i++) {
    const cp = checkpoints[i];
    out[i] = cp === null ? null : tightenFloat32(cp);
  }
  return Object.freeze(out);
}

function tightenPerBranchResiduals(
  perBranch: readonly (readonly (Float32Array | null)[])[],
): readonly (readonly (Float32Array | null)[])[] {
  const out = new Array<readonly (Float32Array | null)[]>(perBranch.length);
  for (let b = 0; b < perBranch.length; b++) {
    out[b] = tightenResidualCheckpoints(perBranch[b]);
  }
  return Object.freeze(out);
}

function argmaxFinite(values: ArrayLike<number>, length: number): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = values[i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function resolveGroundingTagTokenId(tag: string): number {
  const normalized = tag.trim().toLowerCase();
  if (normalized.length === 0) {
    return UNI_MINI_UNK_TOKEN_ID;
  }
  for (let id = 0; id < UNI_MINI_VOCAB.length; id++) {
    if (UNI_MINI_VOCAB[id].toLowerCase() === normalized) {
      return id;
    }
  }
  return UNI_MINI_UNK_TOKEN_ID;
}

function validateRefinementSteps(steps: number): 1 | 2 | 4 | 8 {
  if (steps !== 1 && steps !== 2 && steps !== 4 && steps !== 8) {
    throw new RangeError(`refinementSteps must be 1, 2, 4, or 8, got ${String(steps)}`);
  }
  return steps;
}

function validateActivationSteps(steps: number): 4 | 8 | 16 {
  if (steps !== 4 && steps !== 8 && steps !== 16) {
    throw new RangeError(`activationSteps must be 4, 8, or 16, got ${String(steps)}`);
  }
  return steps;
}

function buildInitialImageMarkers(): Uint8Array {
  const rows = UNI_MINI_CONFIG.vqGridSize;
  const markedCols = UNI_MINI_CONFIG.vqMarkedGridCols;
  const contentCols = UNI_MINI_CONFIG.vqGridSize;
  const isMarker = new Uint8Array(rows * markedCols);
  for (let r = 0; r < rows; r++) {
    isMarker[r * markedCols + contentCols] = 1;
    isMarker[r * markedCols + contentCols + 1] = 1;
  }
  return isMarker;
}

const CANONICAL_IMAGE_MARKER_LAYOUT = buildInitialImageMarkers();

// ============================================================================
// INTERNAL BRANCH & RECORD METADATA (COMPACT — ZERO LOGITS / ZERO TRACES)
// ============================================================================

interface InternalBranchState {
  promptTokenIds: Int32Array;
  uncondPromptTokenIds?: Int32Array;
  /** Only retained when caller passed a custom RGB image without a `scenePresetId`. */
  customSourceImage?: UniRgbImage;
  sourceLetterbox?: UniLetterboxInfo;
  lockSourcePaddingVq: boolean;
  cropToSource: boolean;
  shift: number;
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  treeInitSeed: number;
  stratifiedSeed: number;
  rng: UniXoshiro128ssGenerator;
  // Image diffusion branch cursor
  treeState?: UniSemanticTreeState;
  nextRoundIndex: number;
  // Text / visual block-diffusion branch cursor
  initialPromptLen: number;
  blockLength: number;
  stepsPerBlock: number;
  maxNewTokens: number;
  generatedCount: number;
  confidenceThreshold: number;
  causalContext: boolean;
  suppressMaskLogit: boolean;
  nextLogitsContext?: Float32Array;
  currentBlockTokens?: Int32Array;
  currentBlockStep: number;
  currentBlockPastLen: number;
  pendingPreSeedGuardInterventions: number;
  // Object grounding branch cursor
  groundingTags?: readonly string[];
  groundingSteps: 1 | 2 | 4;
  enforceLocDomainGuard: boolean;
  initialGroundingSuffixTokens?: Int32Array;
  groundingSuffixTokens?: Int32Array;
  groundingSlotSuffixIndices?: Int32Array;
  groundingStepIndex: number;
  // Common cursor
  lastRecordId: string | null;
  nextStepIndex: number;
  finished: boolean;
}

interface InternalRecordMeta {
  /** Exact prefix KV length active when this record's forward pass ran. */
  readonly pastSeqLen: number;
  /** Whether `useCache` was true on this record's forward pass. */
  readonly usedCache: boolean;
  /** For `text_block_step`: block denoising step count after this record (`0` if committed). */
  readonly blockStepAfter?: number;
  /** For `text_block_step`: true iff this step triggered a causal block commit. */
  readonly committedOnStep?: boolean;
  /** For `text_block_step`: exact KV cache seqLen immediately after commit (accounting for EOS/maxNewTokens crop). */
  readonly committedCacheLen?: number;
  /** For `text_block_step`: true iff EOS or maxNewTokens terminated the branch on this step. */
  readonly finishedOnStep?: boolean;
  /** For `grounding_step`: grounding step index after this step (`1..groundingSteps`). */
  readonly groundingStepAfter?: number;
}

// ============================================================================
// STRICT `Set<ArrayBufferLike>` MEMORY AUDITOR
// ============================================================================

/**
 * Recursively collect every distinct `ArrayBufferLike` reachable from `root`,
 * deduplicating shared typed-array backing buffers and skipping any object in
 * `excludedObjects` (such as the shared read-only `UniFullRuntimeBundle`).
 */
export function collectRetainedArrayBuffers(
  root: unknown,
  excludedObjects?: ReadonlySet<object>,
): Set<ArrayBufferLike> {
  const buffers = new Set<ArrayBufferLike>();
  const visited = new Set<object>();

  const walk = (value: unknown): void => {
    if (value === null || value === undefined || typeof value !== 'object') {
      return;
    }
    const obj = value as object;
    if (excludedObjects?.has(obj)) {
      return;
    }
    if (visited.has(obj)) {
      return;
    }
    visited.add(obj);

    if (ArrayBuffer.isView(obj)) {
      buffers.add(obj.buffer);
      return;
    }
    if (
      obj instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && obj instanceof SharedArrayBuffer)
    ) {
      buffers.add(obj);
      return;
    }

    if (Array.isArray(obj)) {
      for (let i = 0; i < obj.length; i++) {
        walk(obj[i]);
      }
      return;
    }
    if (obj instanceof Map) {
      for (const [k, v] of obj.entries()) {
        walk(k);
        walk(v);
      }
      return;
    }
    if (obj instanceof Set) {
      for (const item of obj.values()) {
        walk(item);
      }
      return;
    }

    const keys = Object.keys(obj);
    for (let i = 0; i < keys.length; i++) {
      walk((obj as Record<string, unknown>)[keys[i]]);
    }
  };

  walk(root);
  return buffers;
}

function recordHasForbiddenRetainedPayload(record: UniSessionRecord): boolean {
  const rec = record as unknown as Record<string, unknown>;
  return (
    rec.perBranchFullLogits !== undefined ||
    rec.guidedLogits !== undefined ||
    rec.layerTraces !== undefined ||
    rec.textLogits !== undefined ||
    rec.genLogits !== undefined
  );
}

/**
 * Standalone `Set<ArrayBufferLike>` retained-memory auditor for a `UniSession`.
 * Walks all session-retained state (branches, `prefixCache.slabs`, records,
 * `activeInspection`, and internal branch/record cursors) while excluding the
 * shared read-only `bundle` model weights (`UNI_MINI_TOTAL_RESIDENT_BYTES`).
 */
export function measureSessionRetainedBytes(session: UniSession): UniSessionMemoryAudit {
  return session.auditMemory();
}

// ============================================================================
// BOUNDED INTERACTIVE `UniSession`
// ============================================================================

export class UniSession {
  readonly bundle: UniFullRuntimeBundle;
  readonly maxBranches: number = UNI_SESSION_MAX_BRANCHES;
  readonly maxRecords: number = UNI_SESSION_MAX_RECORDS;
  readonly maxAllowedBytes: number = UNI_MAX_SESSION_RETAINED_BYTES;

  private readonly branchesById = new Map<string, UniSessionBranch>();
  private readonly branchInternals = new Map<string, InternalBranchState>();
  private readonly branchOrder: string[] = [];

  private readonly recordsById = new Map<string, UniSessionRecord>();
  private readonly recordInternals = new Map<string, InternalRecordMeta>();
  private readonly recordOrder: string[] = [];

  private activeBranchId: string | null = null;
  private activeRecordId: string | null = null;
  private activeInspection: UniSessionActiveInspection | null = null;

  private nextBranchSeq = 0;
  private nextRecordSeq = 0;

  constructor(bundle?: UniFullRuntimeBundle) {
    this.bundle = bundle ?? createDefaultUniRuntimeBundle();
  }

  // --------------------------------------------------------------------------
  // PUBLIC READ ACCESSORS
  // --------------------------------------------------------------------------

  getBranch(branchId: string): UniSessionBranch | undefined {
    return this.branchesById.get(branchId);
  }

  getRecord(recordId: string): UniSessionRecord | undefined {
    return this.recordsById.get(recordId);
  }

  getActiveBranchId(): string | null {
    return this.activeBranchId;
  }

  getActiveRecordId(): string | null {
    return this.activeRecordId;
  }

  private ensureActiveInspection(): UniSessionActiveInspection | null {
    if (this.activeRecordId === null) {
      return null;
    }
    if (this.activeInspection === null || this.activeInspection.recordId !== this.activeRecordId) {
      return this.setActiveRecord(this.activeRecordId);
    }
    return this.activeInspection;
  }

  getActiveInspection(): UniSessionActiveInspection | null {
    return this.ensureActiveInspection();
  }

  getAllBranches(): readonly UniSessionBranch[] {
    const out = new Array<UniSessionBranch>(this.branchOrder.length);
    for (let i = 0; i < this.branchOrder.length; i++) {
      out[i] = this.branchesById.get(this.branchOrder[i])!;
    }
    return Object.freeze(out);
  }

  getAllRecords(): readonly UniSessionRecord[] {
    const out = new Array<UniSessionRecord>(this.recordOrder.length);
    for (let i = 0; i < this.recordOrder.length; i++) {
      out[i] = this.recordsById.get(this.recordOrder[i])!;
    }
    return Object.freeze(out);
  }

  getSnapshot(): UniSessionSnapshot {
    const activeInspection = this.ensureActiveInspection();
    return Object.freeze({
      activeBranchId: this.activeBranchId,
      activeRecordId: this.activeRecordId,
      branches: this.getAllBranches(),
      activeInspection,
      memoryAudit: this.auditMemory(),
    });
  }

  // --------------------------------------------------------------------------
  // MEMORY AUDIT (`Set<ArrayBufferLike>`)
  // --------------------------------------------------------------------------

  auditMemory(): UniSessionMemoryAudit {
    // Always include the materialized active-record inspection in memory accounting.
    this.ensureActiveInspection();

    const excluded = new Set<object>([
      this.bundle,
      this.bundle.core,
      this.bundle.vision,
      this.bundle.vqvae,
      this.bundle.codec,
      this.bundle.tree,
    ]);

    const seenBuffers = collectRetainedArrayBuffers(this, excluded);
    let totalRetainedBytes = 0;
    for (const buf of seenBuffers) {
      totalRetainedBytes += buf.byteLength;
    }

    let nonActiveRecordsWithFullLogits = 0;
    for (const [id, rec] of this.recordsById.entries()) {
      if (id !== this.activeRecordId && recordHasForbiddenRetainedPayload(rec)) {
        nonActiveRecordsWithFullLogits += 1;
      }
    }

    return Object.freeze({
      uniqueBufferCount: seenBuffers.size,
      totalRetainedBytes,
      maxAllowedBytes: this.maxAllowedBytes,
      withinBudget: totalRetainedBytes <= this.maxAllowedBytes,
      branchCount: this.branchesById.size,
      maxBranches: this.maxBranches,
      recordCount: this.recordsById.size,
      maxRecords: this.maxRecords,
      activeRecordId: this.activeRecordId,
      nonActiveRecordsWithFullLogits,
    });
  }

  // --------------------------------------------------------------------------
  // BRANCH CREATION (`startBranch`)
  // --------------------------------------------------------------------------

  startBranch(options: UniSessionStartBranchOptions): UniSessionBranch {
    const taskMode = options.taskMode;
    const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';
    const gainPresetId: UniGainPresetId = options.gainPresetId ?? UNI_MINI_CONFIG.defaultGainPreset;
    const refinementSteps = validateRefinementSteps(
      options.refinementSteps ?? UNI_MINI_CONFIG.defaultRefinementSteps,
    );
    const activationSteps = validateActivationSteps(
      options.activationSteps ?? UNI_MINI_CONFIG.defaultActivationSteps,
    );
    const shift = options.shift ?? UNI_MINI_CONFIG.defaultShift;
    const createdAtIndex = ++this.nextBranchSeq;
    const branchId = `branch-${createdAtIndex}`;
    const label = options.label ?? `${taskMode} #${createdAtIndex}`;
    const seed = (options.seed ?? mixSeed32(DEFAULT_SESSION_SEED, createdAtIndex)) >>> 0;
    const stratifiedSeed = (options.stratifiedSeed ?? UNI_STRATIFIED_DEFAULT_SEED) >>> 0;

    const promptTokenIds = tightenInt32(
      Int32Array.from(options.promptTokenIds ?? DEFAULT_PROMPTS[taskMode]),
    );
    const uncondPromptTokenIds =
      options.uncondPromptTokenIds !== undefined
        ? tightenInt32(Int32Array.from(options.uncondPromptTokenIds))
        : undefined;

    // Enforce <= 12 branches before inserting the new branch
    this.evictBranchesIfNeeded(null);

    if (taskMode === 'text_to_image' || taskMode === 'image_edit') {
      return this.startImageGenerationBranch({
        branchId,
        label,
        createdAtIndex,
        taskMode,
        promptTokenIds,
        uncondPromptTokenIds,
        scenePresetId: options.scenePresetId,
        sourceImage: options.sourceImage,
        prefillMode,
        gainPresetId,
        guidanceMode: options.guidanceMode,
        guidanceScaleText: options.guidanceScaleText,
        guidanceScaleImage: options.guidanceScaleImage,
        refinementSteps,
        activationSteps,
        shift,
        temperature: options.temperature ?? 0.8,
        topK: options.topK ?? 0,
        topP: options.topP ?? 1.0,
        lockSourcePaddingVq: options.lockSourcePaddingVq ?? true,
        cropToSource: options.cropToSource ?? true,
        seed,
        stratifiedSeed,
      });
    }

    if (taskMode === 'object_grounding') {
      return this.startGroundingBranch({
        branchId,
        label,
        createdAtIndex,
        promptTokenIds,
        scenePresetId: options.scenePresetId,
        sourceImage: options.sourceImage,
        prefillMode,
        gainPresetId,
        refinementSteps,
        activationSteps,
        shift,
        seed,
        stratifiedSeed,
        groundingTags: options.groundingTags,
        groundingSteps: options.groundingSteps ?? 2,
        enforceLocDomainGuard: options.enforceLocDomainGuard ?? true,
      });
    }

    // `text_understanding` or `visual_understanding`
    return this.startTextOrVisualBranch({
      branchId,
      label,
      createdAtIndex,
      taskMode,
      promptTokenIds,
      scenePresetId: options.scenePresetId,
      sourceImage: options.sourceImage,
      prefillMode,
      gainPresetId,
      refinementSteps,
      activationSteps,
      shift,
      temperature: options.temperature ?? 0,
      seed,
      stratifiedSeed,
      blockLength: options.blockLength ?? 4,
      stepsPerBlock: options.stepsPerBlock ?? (options.blockLength ?? 4),
      maxNewTokens: options.maxNewTokens ?? 8,
      confidenceThreshold: options.confidenceThreshold ?? 0.9,
      causalContext: options.causalContext ?? true,
      suppressMaskLogit: options.suppressMaskLogit ?? true,
    });
  }

  // --------------------------------------------------------------------------
  // BRANCH FORKING (`forkBranch`)
  // --------------------------------------------------------------------------

  forkBranch(
    fromRecordId: string,
    options?: UniSessionForkBranchOptions,
  ): UniSessionBranch {
    const fromRecord = this.recordsById.get(fromRecordId);
    if (!fromRecord) {
      throw new Error(`Cannot fork from unknown recordId: ${fromRecordId}`);
    }
    const parentBranch = this.branchesById.get(fromRecord.branchId);
    const parentInternal = this.branchInternals.get(fromRecord.branchId);
    const fromMeta = this.recordInternals.get(fromRecordId);
    if (!parentBranch || !parentInternal || !fromMeta) {
      throw new Error(`Missing branch metadata for recordId: ${fromRecordId}`);
    }

    // Enforce <= 12 branches while protecting parentBranch from eviction
    this.evictBranchesIfNeeded(parentBranch.branchId);

    const createdAtIndex = ++this.nextBranchSeq;
    const branchId = `branch-${createdAtIndex}`;
    const label = options?.label ?? `${parentBranch.label} (fork #${createdAtIndex})`;
    const prefillMode: UniElasticPrefillMode = options?.prefillMode ?? parentBranch.prefillMode;
    const guidanceMode = options?.guidanceMode ?? parentBranch.guidanceMode;
    const guidanceScaleText = options?.guidanceScaleText ?? parentBranch.guidanceScaleText;
    const guidanceScaleImage = options?.guidanceScaleImage ?? parentBranch.guidanceScaleImage;
    const seed = (options?.seed ?? mixSeed32(parentInternal.seed, createdAtIndex)) >>> 0;

    let prefixCache: UniKvCacheState = parentBranch.prefixCache;
    let cfgBatchSize: UniCfgBatchSize = parentBranch.cfgBatchSize;
    let sourcePaddingMask = parentBranch.sourcePaddingMask;
    let sourceVqCodes = parentBranch.sourceVqCodes;
    let sourceLetterbox = parentInternal.sourceLetterbox;

    if (parentBranch.taskMode === 'text_to_image') {
      const targetCfgBatchSize: UniCfgBatchSize = guidanceScaleText === 0 ? 1 : 2;
      if (
        prefillMode !== parentBranch.prefillMode ||
        targetCfgBatchSize !== parentBranch.cfgBatchSize
      ) {
        const prepared = prepareT2iPrefix(this.bundle, {
          promptTokenIds: parentInternal.promptTokenIds,
          uncondPromptTokenIds: parentInternal.uncondPromptTokenIds,
          guidanceScale: guidanceScaleText,
          prefillMode,
          gainPresetId: parentBranch.gainPresetId,
        });
        prefixCache = prepared.prefixCache;
        cfgBatchSize = prepared.cfgBatchSize;
      }
    } else if (parentBranch.taskMode === 'image_edit') {
      if (prefillMode !== parentBranch.prefillMode) {
        const srcImg = this.resolveBranchSourceImage(parentBranch, parentInternal);
        const prepared = prepareImageEditPrefix(this.bundle, {
          sourceImage: srcImg,
          promptTokenIds: parentInternal.promptTokenIds,
          uncondPromptTokenIds: parentInternal.uncondPromptTokenIds,
          guidanceMode: guidanceMode ?? 'legacy',
          guidanceScaleText,
          guidanceScaleImage,
          lockSourcePaddingVq: parentInternal.lockSourcePaddingVq,
          cropToSource: parentInternal.cropToSource,
          prefillMode,
          gainPresetId: parentBranch.gainPresetId,
        });
        prefixCache = prepared.prefixCache;
        cfgBatchSize = prepared.cfgBatchSize;
        sourcePaddingMask = prepared.sourcePaddingMask;
        sourceVqCodes = prepared.sourceVqCodes;
        sourceLetterbox = prepared.sourceLetterbox?.info;
      }
    } else {
      // Understanding / grounding: share immutable slabs cropped to exact fork record KV horizon
      let forkCacheLen: number;
      if (fromRecord.kind === 'prefill') {
        forkCacheLen = Math.min(parentBranch.prefixCache.seqLen, fromRecord.routing.length);
      } else if (
        fromRecord.kind === 'text_block_step' &&
        fromMeta.committedOnStep &&
        fromMeta.committedCacheLen !== undefined
      ) {
        forkCacheLen = Math.min(parentBranch.prefixCache.seqLen, fromMeta.committedCacheLen);
      } else {
        forkCacheLen = Math.min(parentBranch.prefixCache.seqLen, fromMeta.pastSeqLen);
      }
      prefixCache = cropUniKvCache(parentBranch.prefixCache, forkCacheLen);
    }

    // Reconstruct exact historical image semantic-tree state from `fromRecord`
    let treeState: UniSemanticTreeState | undefined;
    let nextRoundIndex = 0;
    if (parentBranch.taskMode === 'text_to_image' || parentBranch.taskMode === 'image_edit') {
      if (fromRecord.kind === 'image_denoise' && fromRecord.imageState) {
        treeState = Object.freeze({
          stages: new Int32Array(fromRecord.imageState.stages),
          representatives: new Int32Array(fromRecord.imageState.representatives),
          levels: new Int32Array(fromRecord.imageState.levels),
          noiseWeights: new Float32Array(fromRecord.imageState.noiseWeights),
          isMarker: new Uint8Array(CANONICAL_IMAGE_MARKER_LAYOUT),
          refinementSteps: parentBranch.refinementSteps,
        });
        nextRoundIndex = fromRecord.imageState.roundIndex + 1;
      } else {
        treeState = initSemanticTreeState({
          refinementSteps: parentBranch.refinementSteps,
          seed: parentInternal.treeInitSeed,
          lockedContentMask: parentInternal.lockSourcePaddingVq
            ? sourcePaddingMask?.lockMask
            : undefined,
          lockedSourceCodes: parentInternal.lockSourcePaddingVq ? sourceVqCodes : undefined,
        });
        nextRoundIndex = 0;
      }
    }

    // Reconstruct exact historical Understanding / Grounding cursor from `fromRecord`
    // (never leaking later parent tokens, logits, or grounding steps!)
    const vocabSize = this.bundle.core.config.vocabSize;
    let forkGeneratedCount = 0;
    let forkNextLogitsContext: Float32Array | undefined;
    let forkCurrentBlockTokens: Int32Array | undefined;
    let forkCurrentBlockStep = 0;
    let forkCurrentBlockPastLen = prefixCache.seqLen;
    let forkGroundingSuffixTokens: Int32Array | undefined;
    let forkGroundingStepIndex = 0;
    let forkFinished =
      parentBranch.schedule !== undefined && nextRoundIndex >= parentBranch.schedule.totalRounds;

    if (
      parentBranch.taskMode === 'text_understanding' ||
      parentBranch.taskMode === 'visual_understanding'
    ) {
      forkGeneratedCount = Math.max(0, prefixCache.seqLen - parentInternal.initialPromptLen);

      if (fromRecord.kind === 'prefill') {
        const head = this.reconstructStepHeadLogits(fromRecordId);
        const logits = head.perBranchTextLogits[0];
        if (logits) {
          const seqLen = fromRecord.routing.length;
          forkNextLogitsContext = new Float32Array(
            logits.subarray((seqLen - 1) * vocabSize, seqLen * vocabSize),
          );
        }
        forkFinished = parentInternal.maxNewTokens <= 0;
      } else if (fromRecord.kind === 'text_block_step') {
        if (fromMeta.committedOnStep) {
          forkCurrentBlockTokens = undefined;
          forkCurrentBlockStep = 0;
          forkCurrentBlockPastLen = prefixCache.seqLen;
          forkFinished =
            Boolean(fromMeta.finishedOnStep) || forkGeneratedCount >= parentInternal.maxNewTokens;

          if (fromRecord.maskMode === 'causal') {
            // AR limit (B=1): `fromRecord` itself holds the causal commit residuals
            const head = this.reconstructStepHeadLogits(fromRecordId);
            const logits = head.perBranchTextLogits[0];
            if (logits) {
              const seqLen = fromRecord.routing.length;
              forkNextLogitsContext = new Float32Array(
                logits.subarray((seqLen - 1) * vocabSize, seqLen * vocabSize),
              );
            }
          } else if (fromRecord.textState) {
            // Recompute exact 0-ULP causal commit tail logits for `fromRecord`'s committed block
            const blockTokens = fromRecord.textState.tokenIds;
            const preCommitCache = cropUniKvCache(parentBranch.prefixCache, fromMeta.pastSeqLen);
            const { pastKvs, pastRouting } = extractRowPastKvs(preCommitCache, 0);
            const commitFwd = runUniForwardPass(this.bundle.core, {
              tokenIds: blockTokens,
              maskMode: 'causal',
              useCache: true,
              prefillMode: 'hf_faithful',
              downstreamTarget: 'u',
              pastKvs,
              pastRouting,
              computeTextHead: true,
              computeGenHead: false,
              retainedTraceLayers: EMPTY_TRACE_LAYERS,
            });
            const commitLogits = commitFwd.textLogits!;
            const bLen = blockTokens.length;
            forkNextLogitsContext = new Float32Array(
              commitLogits.subarray((bLen - 1) * vocabSize, bLen * vocabSize),
            );
          }
        } else {
          // Mid-block denoising step: preserve partially unmasked block tokens & recover prior causal logits
          forkCurrentBlockTokens = fromRecord.textState
            ? new Int32Array(fromRecord.textState.tokenIds)
            : undefined;
          forkCurrentBlockStep = fromMeta.blockStepAfter ?? 1;
          forkCurrentBlockPastLen = prefixCache.seqLen;
          forkFinished = false;
          forkNextLogitsContext = this.reconstructPriorCausalLogits(fromRecord, parentBranch);
        }
      }
    } else if (parentBranch.taskMode === 'object_grounding') {
      // Recover exact prefill slot-0 shifted logits from the grounding prefill record
      const prefillId =
        fromRecord.kind === 'prefill'
          ? fromRecordId
          : this.findBranchPrefillRecordId(fromRecord, parentBranch);
      if (prefillId !== null) {
        const prefillRec = this.recordsById.get(prefillId)!;
        const head = this.reconstructStepHeadLogits(prefillId);
        const logits = head.perBranchTextLogits[0];
        if (logits) {
          const seqLen = prefillRec.routing.length;
          forkNextLogitsContext = new Float32Array(
            logits.subarray((seqLen - 1) * vocabSize, seqLen * vocabSize),
          );
        }
      }

      if (fromRecord.kind === 'grounding_step' && fromRecord.textState) {
        forkGroundingSuffixTokens = new Int32Array(fromRecord.textState.tokenIds);
        forkGroundingStepIndex = fromMeta.groundingStepAfter ?? fromRecord.stepIndex;
        let masksLeft = 0;
        const slots = parentInternal.groundingSlotSuffixIndices;
        if (slots) {
          for (let k = 0; k < slots.length; k++) {
            if (forkGroundingSuffixTokens[slots[k]] === UNI_MINI_MASK_TOKEN_ID) {
              masksLeft += 1;
            }
          }
        }
        forkFinished =
          masksLeft === 0 || forkGroundingStepIndex >= parentInternal.groundingSteps;
      } else {
        forkGroundingSuffixTokens =
          parentInternal.initialGroundingSuffixTokens !== undefined
            ? new Int32Array(parentInternal.initialGroundingSuffixTokens)
            : undefined;
        forkGroundingStepIndex = 0;
        forkFinished = false;
      }
    }

    const forkedBranch: UniSessionBranch = Object.freeze({
      branchId,
      parentBranchId: parentBranch.branchId,
      forkRecordId: fromRecordId,
      label,
      taskMode: parentBranch.taskMode,
      prefillMode,
      gainPresetId: parentBranch.gainPresetId,
      cfgBatchSize,
      ...(guidanceMode !== undefined ? { guidanceMode } : {}),
      guidanceScaleText,
      guidanceScaleImage,
      refinementSteps: parentBranch.refinementSteps,
      activationSteps: parentBranch.activationSteps,
      ...(parentBranch.scenePresetId !== undefined
        ? { scenePresetId: parentBranch.scenePresetId }
        : {}),
      prefixCache,
      ...(sourcePaddingMask !== undefined ? { sourcePaddingMask } : {}),
      ...(sourceVqCodes !== undefined ? { sourceVqCodes } : {}),
      ...(parentBranch.unmaskOrder !== undefined ? { unmaskOrder: parentBranch.unmaskOrder } : {}),
      ...(parentBranch.schedule !== undefined ? { schedule: parentBranch.schedule } : {}),
      recordIds: Object.freeze([]),
      createdAtIndex,
    });

    const forkedInternal: InternalBranchState = {
      promptTokenIds: parentInternal.promptTokenIds,
      uncondPromptTokenIds: parentInternal.uncondPromptTokenIds,
      customSourceImage: parentInternal.customSourceImage,
      sourceLetterbox,
      lockSourcePaddingVq: parentInternal.lockSourcePaddingVq,
      cropToSource: parentInternal.cropToSource,
      shift: parentInternal.shift,
      temperature: parentInternal.temperature,
      topK: parentInternal.topK,
      topP: parentInternal.topP,
      seed,
      treeInitSeed: parentInternal.treeInitSeed,
      stratifiedSeed: parentInternal.stratifiedSeed,
      rng: createXoshiro128ss(seed),
      treeState,
      nextRoundIndex,
      initialPromptLen: parentInternal.initialPromptLen,
      blockLength: parentInternal.blockLength,
      stepsPerBlock: parentInternal.stepsPerBlock,
      maxNewTokens: parentInternal.maxNewTokens,
      generatedCount: forkGeneratedCount,
      confidenceThreshold: parentInternal.confidenceThreshold,
      causalContext: parentInternal.causalContext,
      suppressMaskLogit: parentInternal.suppressMaskLogit,
      nextLogitsContext: forkNextLogitsContext,
      currentBlockTokens: forkCurrentBlockTokens,
      currentBlockStep: forkCurrentBlockStep,
      currentBlockPastLen: forkCurrentBlockPastLen,
      pendingPreSeedGuardInterventions: 0,
      groundingTags: parentInternal.groundingTags,
      groundingSteps: parentInternal.groundingSteps,
      enforceLocDomainGuard: parentInternal.enforceLocDomainGuard,
      initialGroundingSuffixTokens: parentInternal.initialGroundingSuffixTokens,
      groundingSuffixTokens: forkGroundingSuffixTokens,
      groundingSlotSuffixIndices: parentInternal.groundingSlotSuffixIndices,
      groundingStepIndex: forkGroundingStepIndex,
      lastRecordId: fromRecordId,
      nextStepIndex: fromRecord.stepIndex + 1,
      finished: forkFinished,
    };

    this.branchesById.set(branchId, forkedBranch);
    this.branchInternals.set(branchId, forkedInternal);
    this.branchOrder.push(branchId);
    this.activeBranchId = branchId;

    return forkedBranch;
  }

  // --------------------------------------------------------------------------
  // STEPPING (`stepBranch`, `runBranchToCompletion`)
  // --------------------------------------------------------------------------

  stepBranch(branchId?: string): UniSessionRecord | null {
    const targetBranchId = branchId ?? this.activeBranchId;
    if (!targetBranchId) {
      throw new Error('No active branch to step; call startBranch() first');
    }
    const branch = this.branchesById.get(targetBranchId);
    const internal = this.branchInternals.get(targetBranchId);
    if (!branch || !internal) {
      throw new Error(`Unknown branchId: ${targetBranchId}`);
    }
    if (internal.finished) {
      return null;
    }

    if (branch.taskMode === 'text_to_image' || branch.taskMode === 'image_edit') {
      return this.stepImageGenerationBranch(branch, internal);
    }
    if (branch.taskMode === 'object_grounding') {
      return this.stepGroundingBranch(branch, internal);
    }
    return this.stepTextOrVisualBranch(branch, internal);
  }

  runBranchToCompletion(branchId?: string, maxSteps = 64): readonly UniSessionRecord[] {
    const targetBranchId = branchId ?? this.activeBranchId;
    if (!targetBranchId) {
      throw new Error('No active branch to run; call startBranch() first');
    }
    const produced: UniSessionRecord[] = [];
    for (let i = 0; i < maxSteps; i++) {
      const rec = this.stepBranch(targetBranchId);
      if (rec === null) {
        break;
      }
      produced.push(rec);
    }
    return Object.freeze(produced);
  }

  // --------------------------------------------------------------------------
  // ACTIVE INSPECTION & 0-ULP LAZY RECONSTRUCTION
  // --------------------------------------------------------------------------

  setActiveRecord(recordId: string): UniSessionActiveInspection {
    const record = this.recordsById.get(recordId);
    if (!record) {
      throw new Error(`Unknown recordId: ${recordId}`);
    }

    // Evict previous activeInspection immediately before building the new one
    this.activeInspection = null;
    this.activeRecordId = recordId;
    this.activeBranchId = record.branchId;

    const eagerTracesByBranch: ReadonlyMap<number, UniLayerTrace>[] = [];
    for (let b = 0; b < record.cfgBatchSize; b++) {
      const traceMap = new Map<number, UniLayerTrace>();
      const rowResiduals = record.perBranchResiduals[b];
      for (let i = 0; i < UNI_SESSION_EAGER_TRACE_LAYERS.length; i++) {
        const layerIndex = UNI_SESSION_EAGER_TRACE_LAYERS[i];
        if (rowResiduals[layerIndex] !== null && rowResiduals[layerIndex + 1] !== null) {
          traceMap.set(layerIndex, this.reconstructLayerTrace(recordId, layerIndex, b));
        }
      }
      eagerTracesByBranch.push(traceMap);
    }

    const head = this.reconstructStepHeadLogits(recordId);

    const inspection: UniSessionActiveInspection = Object.freeze({
      recordId,
      eagerTracesByBranch: Object.freeze(eagerTracesByBranch),
      perBranchFinalNorm: head.perBranchFinalNorm,
      perBranchTextLogits: head.perBranchTextLogits,
      perBranchGenLogits: head.perBranchGenLogits,
      guidedGenLogits: head.guidedGenLogits,
    });

    this.activeInspection = inspection;
    return inspection;
  }

  /**
   * 0-ULP lazy reconstruction of any executed layer trace (`L0..L33`) on any retained
   * historical or active record from `record.perBranchResiduals[batchRow][layerIndex]`
   * + `extractRowPastKvs(branch.prefixCache, batchRow)` via `runUniElasticLayer`.
   */
  reconstructLayerTrace(
    recordId: string,
    layerIndex: number,
    batchRow = 0,
  ): UniLayerTrace {
    const record = this.recordsById.get(recordId);
    if (!record) {
      throw new Error(`Unknown recordId: ${recordId}`);
    }
    const branch = this.branchesById.get(record.branchId);
    const meta = this.recordInternals.get(recordId);
    if (!branch || !meta) {
      throw new Error(`Missing branch or metadata for recordId: ${recordId}`);
    }

    const cfg = this.bundle.core.config;
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= cfg.numLayers) {
      throw new RangeError(
        `layerIndex (${String(layerIndex)}) must be an integer in [0, ${cfg.numLayers - 1}]`,
      );
    }
    if (!Number.isInteger(batchRow) || batchRow < 0 || batchRow >= record.cfgBatchSize) {
      throw new RangeError(
        `batchRow (${String(batchRow)}) out of range for cfgBatchSize=${record.cfgBatchSize}`,
      );
    }

    const rowResiduals = record.perBranchResiduals[batchRow];
    const inputResidual = rowResiduals[layerIndex];
    const outputResidual = rowResiduals[layerIndex + 1];
    if (inputResidual === null || outputResidual === null) {
      throw new RangeError(
        `Layer ${layerIndex} was skipped for branch row ${batchRow} on record ${recordId} (prefillMode=${record.prefillMode})`,
      );
    }

    const seqLen = record.routing.length;
    const rowRouting = this.resolveRecordRowRouting(record, branch, batchRow);
    const canonicalRouting = canonicalizeTokenRouting(rowRouting).canonical;
    const isSplit = layerIndex >= cfg.dualTowerLayers;
    const pruneUnderstandingTokens =
      record.prefillMode === 'elastic_pruned' && isSplit && canonicalRouting === 'mixed';

    let pastKv: UniLayerKvSlice | null = null;
    let pastRouting: Uint8Array | null = null;

    if (record.kind === 'image_denoise') {
      const extracted = extractSingleLayerRowPastKv(branch.prefixCache, batchRow, layerIndex);
      pastKv = extracted.pastKv;
      pastRouting = extracted.pastRouting;
    } else if (
      (record.kind === 'text_block_step' || record.kind === 'grounding_step') &&
      meta.pastSeqLen > 0
    ) {
      const historicalCache = cropUniKvCache(branch.prefixCache, meta.pastSeqLen);
      const extracted = extractSingleLayerRowPastKv(historicalCache, batchRow, layerIndex);
      pastKv = extracted.pastKv;
      pastRouting = extracted.pastRouting;
    }

    const yarn = cfg.rope === UNI_MINI_CONFIG.rope ? DEFAULT_YARN : computeUniYarnInvFreq(cfg.rope);
    return runUniElasticLayer({
      hiddenStates: inputResidual,
      seqLen,
      routing: rowRouting,
      layerWeights: this.bundle.core.layers[layerIndex],
      config: cfg,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds: record.positionIds,
      cachePositions: record.positionIds,
      maskMode: record.maskMode,
      useCache: meta.usedCache,
      pastKv,
      pastRouting,
      pruneUnderstandingTokens,
    });
  }

  /**
   * 0-ULP lazy reconstruction of final RMSNorm, per-branch text/generation head logits,
   * and CFG-mixed `guidedGenLogits` from `record.perBranchResiduals[b][34]`.
   */
  reconstructStepHeadLogits(recordId: string): UniReconstructedHeadLogits {
    const record = this.recordsById.get(recordId);
    if (!record) {
      throw new Error(`Unknown recordId: ${recordId}`);
    }
    const branch = this.branchesById.get(record.branchId);
    if (!branch) {
      throw new Error(`Missing branch for recordId: ${recordId}`);
    }

    const core = this.bundle.core;
    const cfg = core.config;
    const d = cfg.hiddenSize;
    const numLayers = cfg.numLayers;
    const seqLen = record.routing.length;
    const batchSize = record.cfgBatchSize;

    const perBranchFinalNorm: (Float32Array | null)[] = new Array<Float32Array | null>(
      batchSize,
    ).fill(null);
    const perBranchTextLogits: (Float32Array | null)[] = new Array<Float32Array | null>(
      batchSize,
    ).fill(null);
    const perBranchGenLogits: (Float32Array | null)[] = new Array<Float32Array | null>(
      batchSize,
    ).fill(null);

    const isUnderstandingTask =
      record.taskMode === 'text_understanding' ||
      record.taskMode === 'visual_understanding' ||
      record.taskMode === 'object_grounding';

    for (let b = 0; b < batchSize; b++) {
      const hFinal = record.perBranchResiduals[b][numLayers];
      if (hFinal === null) {
        continue;
      }

      const rowRouting = this.resolveRecordRowRouting(record, branch, b);
      const canonicalRouting = canonicalizeTokenRouting(rowRouting).canonical;
      const pruneUnderstandingAtSplit =
        record.prefillMode === 'elastic_pruned' && canonicalRouting === 'mixed';

      const finalNorm = applyRoutedUniRmsNorm(
        hFinal,
        seqLen,
        d,
        rowRouting,
        core.finalNorm,
        core.finalNormGen,
        cfg.rmsNormEps,
        pruneUnderstandingAtSplit,
      );
      perBranchFinalNorm[b] = finalNorm;

      if (record.kind === 'image_denoise') {
        const fixedPrefixCount = cfg.genFixedPrefixTokens; // 4
        const parentRows = cfg.packedGridRows; // 2
        const parentCols = cfg.packedGridCols; // 3
        const parentHidden = finalNorm.subarray(
          fixedPrefixCount * d,
          (fixedPrefixCount + parentRows * parentCols) * d,
        );
        const expanded = expandPackedImageHiddenStates(
          parentHidden,
          parentRows,
          parentCols,
          this.bundle.codec,
          core.genPredictor,
        );
        perBranchGenLogits[b] = expanded.logits!;
      } else if (isUnderstandingTask) {
        perBranchTextLogits[b] = linearProject(
          finalNorm,
          seqLen,
          d,
          core.diffusionHead,
          cfg.vocabSize,
        );
      }
    }

    let guidedGenLogits: Float32Array | null = null;
    if (record.kind === 'image_denoise') {
      if (batchSize === 1) {
        guidedGenLogits = perBranchGenLogits[0];
      } else if (batchSize === 2) {
        guidedGenLogits = applyT2iCfgLogits(
          perBranchGenLogits[0]!,
          perBranchGenLogits[1]!,
          branch.guidanceScaleText,
        );
      } else {
        guidedGenLogits = applyEditDualCfgLogits(
          perBranchGenLogits[0]!,
          perBranchGenLogits[1]!,
          perBranchGenLogits[2]!,
          branch.guidanceMode ?? 'legacy',
          branch.guidanceScaleText,
          branch.guidanceScaleImage,
        );
      }
    }

    return Object.freeze({
      perBranchFinalNorm: Object.freeze(perBranchFinalNorm),
      perBranchTextLogits: Object.freeze(perBranchTextLogits),
      perBranchGenLogits: Object.freeze(perBranchGenLogits),
      guidedGenLogits,
    });
  }

  /**
   * Lazily decode the marked VQ codes of a branch (or specific image-denoise record)
   * into an RGB image without retaining decoded pixel buffers in session state.
   */
  decodeBranchImage(
    branchId?: string,
    recordId?: string,
  ): { contentCodes: Int32Array; decodedPaddedImage: UniRgbImage; outputImage: UniRgbImage } {
    let targetBranchId = branchId ?? this.activeBranchId;
    let markedCodes: Int32Array | undefined;

    if (recordId !== undefined) {
      const rec = this.recordsById.get(recordId);
      if (!rec || !rec.imageState) {
        throw new Error(`Record ${recordId} does not contain imageState`);
      }
      targetBranchId = rec.branchId;
      markedCodes = rec.imageState.representatives;
    }

    if (!targetBranchId) {
      throw new Error('No target branch available for decodeBranchImage');
    }
    const branch = this.branchesById.get(targetBranchId);
    const internal = this.branchInternals.get(targetBranchId);
    if (!branch || !internal) {
      throw new Error(`Unknown branchId: ${targetBranchId}`);
    }
    if (!markedCodes) {
      if (!internal.treeState) {
        throw new Error(`Branch ${targetBranchId} is not an image generation branch`);
      }
      markedCodes = internal.treeState.representatives;
    }

    return decodeMarkedCodesToImage(
      this.bundle,
      markedCodes,
      internal.sourceLetterbox,
      internal.cropToSource,
    );
  }

  // --------------------------------------------------------------------------
  // PRIVATE TASK STARTUP IMPLEMENTATIONS
  // --------------------------------------------------------------------------

  private startImageGenerationBranch(params: {
    branchId: string;
    label: string;
    createdAtIndex: number;
    taskMode: 'text_to_image' | 'image_edit';
    promptTokenIds: Int32Array;
    uncondPromptTokenIds?: Int32Array;
    scenePresetId?: UniScenePresetId;
    sourceImage?: UniRgbImage;
    prefillMode: UniElasticPrefillMode;
    gainPresetId: UniGainPresetId;
    guidanceMode?: UniEditGuidanceMode;
    guidanceScaleText?: number;
    guidanceScaleImage?: number;
    refinementSteps: 1 | 2 | 4 | 8;
    activationSteps: 4 | 8 | 16;
    shift: number;
    temperature: number;
    topK: number;
    topP: number;
    lockSourcePaddingVq: boolean;
    cropToSource: boolean;
    seed: number;
    stratifiedSeed: number;
  }): UniSessionBranch {
    const isEdit = params.taskMode === 'image_edit';
    const guidanceMode: UniEditGuidanceMode | undefined = isEdit
      ? (params.guidanceMode ?? 'legacy')
      : undefined;
    const guidanceScaleText =
      params.guidanceScaleText ?? (isEdit ? 1.6 : 3.0);
    const guidanceScaleImage =
      params.guidanceScaleImage ?? (isEdit ? 2.0 : 0.0);

    let scenePresetId: UniScenePresetId | undefined = params.scenePresetId;
    let customSourceImage: UniRgbImage | undefined;
    let prepared: UniPreparedGenerationPrefix;

    if (isEdit) {
      let sourceImage: UniRgbImage;
      if (params.sourceImage !== undefined) {
        sourceImage = params.sourceImage;
        scenePresetId = params.scenePresetId ?? params.sourceImage.presetId;
        if (scenePresetId === undefined) {
          customSourceImage = sourceImage;
        }
      } else {
        scenePresetId = params.scenePresetId ?? 'harbor_panorama_16x8';
        sourceImage = createUniScenePreset(scenePresetId);
      }

      prepared = prepareImageEditPrefix(this.bundle, {
        sourceImage,
        promptTokenIds: params.promptTokenIds,
        uncondPromptTokenIds: params.uncondPromptTokenIds,
        guidanceMode,
        guidanceScaleText,
        guidanceScaleImage,
        lockSourcePaddingVq: params.lockSourcePaddingVq,
        cropToSource: params.cropToSource,
        prefillMode: params.prefillMode,
        gainPresetId: params.gainPresetId,
      });
    } else {
      prepared = prepareT2iPrefix(this.bundle, {
        promptTokenIds: params.promptTokenIds,
        uncondPromptTokenIds: params.uncondPromptTokenIds,
        guidanceScale: guidanceScaleText,
        prefillMode: params.prefillMode,
        gainPresetId: params.gainPresetId,
      });
    }

    const contentCount = UNI_MINI_CONFIG.vqGridSize * UNI_MINI_CONFIG.vqGridSize;
    const schedule: UniSemanticTreeSchedule = semanticTreeActivationCounts(
      contentCount,
      params.activationSteps,
      params.refinementSteps,
      params.shift,
    );
    const unmaskOrder = stratifiedRandom(UNI_MINI_CONFIG.vqGridSize, params.stratifiedSeed);
    const treeInitSeed = mixSeed32(params.seed, 0x54524545); // "TREE"
    const treeState = initSemanticTreeState({
      refinementSteps: params.refinementSteps,
      seed: treeInitSeed,
      lockedContentMask:
        isEdit && params.lockSourcePaddingVq
          ? prepared.sourcePaddingMask?.lockMask
          : undefined,
      lockedSourceCodes:
        isEdit && params.lockSourcePaddingVq ? prepared.sourceVqCodes : undefined,
    });

    // Reuse compact prefill residuals directly from `prepare*Prefix` (zero duplicate 34L passes)
    const prefillResiduals: readonly (readonly (Float32Array | null)[])[] =
      prepared.perBranchResiduals !== undefined
        ? tightenPerBranchResiduals(prepared.perBranchResiduals)
        : Object.freeze(
            prepared.perBranchEmbeds.map((emb, b) => {
              const fwd = runUniForwardPass(this.bundle.core, {
                inputsEmbeds: emb,
                routing: prepared.perBranchRouting[b],
                maskMode: 'causal',
                useCache: true,
                prefillMode: params.prefillMode,
                downstreamTarget: 'g',
                computeTextHead: false,
                computeGenHead: false,
                retainedTraceLayers: EMPTY_TRACE_LAYERS,
              });
              return tightenResidualCheckpoints(fwd.residualCheckpoints);
            }),
          );

    const branch: UniSessionBranch = Object.freeze({
      branchId: params.branchId,
      parentBranchId: null,
      forkRecordId: null,
      label: params.label,
      taskMode: params.taskMode,
      prefillMode: params.prefillMode,
      gainPresetId: params.gainPresetId,
      cfgBatchSize: prepared.cfgBatchSize,
      ...(guidanceMode !== undefined ? { guidanceMode } : {}),
      guidanceScaleText,
      guidanceScaleImage,
      refinementSteps: params.refinementSteps,
      activationSteps: params.activationSteps,
      ...(scenePresetId !== undefined ? { scenePresetId } : {}),
      prefixCache: prepared.prefixCache,
      ...(prepared.sourcePaddingMask !== undefined
        ? { sourcePaddingMask: prepared.sourcePaddingMask }
        : {}),
      ...(prepared.sourceVqCodes !== undefined ? { sourceVqCodes: prepared.sourceVqCodes } : {}),
      unmaskOrder,
      schedule,
      recordIds: Object.freeze([]),
      createdAtIndex: params.createdAtIndex,
    });

    const internal: InternalBranchState = {
      promptTokenIds: params.promptTokenIds,
      uncondPromptTokenIds: params.uncondPromptTokenIds,
      customSourceImage,
      sourceLetterbox: prepared.sourceLetterbox?.info,
      lockSourcePaddingVq: params.lockSourcePaddingVq,
      cropToSource: params.cropToSource,
      shift: params.shift,
      temperature: params.temperature,
      topK: params.topK,
      topP: params.topP,
      seed: params.seed,
      treeInitSeed,
      stratifiedSeed: params.stratifiedSeed,
      rng: createXoshiro128ss(params.seed),
      treeState,
      nextRoundIndex: 0,
      initialPromptLen: prepared.prefixLen,
      blockLength: 4,
      stepsPerBlock: 4,
      maxNewTokens: 0,
      generatedCount: 0,
      confidenceThreshold: 0.9,
      causalContext: true,
      suppressMaskLogit: true,
      currentBlockStep: 0,
      currentBlockPastLen: prepared.prefixLen,
      pendingPreSeedGuardInterventions: 0,
      groundingSteps: 2,
      enforceLocDomainGuard: true,
      groundingStepIndex: 0,
      lastRecordId: null,
      nextStepIndex: 0,
      finished: false,
    };

    this.branchesById.set(params.branchId, branch);
    this.branchInternals.set(params.branchId, internal);
    this.branchOrder.push(params.branchId);
    this.activeBranchId = params.branchId;

    // Record initial prefill state cleanly
    const positionIds = new Int32Array(prepared.prefixLen);
    for (let i = 0; i < prepared.prefixLen; i++) {
      positionIds[i] = i;
    }
    const primaryRouting = tightenUint8(
      prepared.perBranchRouting[prepared.cfgBatchSize - 1],
    );

    this.appendRecord(
      params.branchId,
      {
        parentRecordId: null,
        stepIndex: 0,
        kind: 'prefill',
        taskMode: params.taskMode,
        prefillMode: params.prefillMode,
        cfgBatchSize: prepared.cfgBatchSize,
        maskMode: 'causal',
        routing: primaryRouting,
        positionIds,
        perBranchResiduals: Object.freeze(prefillResiduals),
      },
      { pastSeqLen: 0, usedCache: true },
    );

    return this.branchesById.get(params.branchId)!;
  }

  private startTextOrVisualBranch(params: {
    branchId: string;
    label: string;
    createdAtIndex: number;
    taskMode: 'text_understanding' | 'visual_understanding';
    promptTokenIds: Int32Array;
    scenePresetId?: UniScenePresetId;
    sourceImage?: UniRgbImage;
    prefillMode: UniElasticPrefillMode;
    gainPresetId: UniGainPresetId;
    refinementSteps: 1 | 2 | 4 | 8;
    activationSteps: 4 | 8 | 16;
    shift: number;
    temperature: number;
    seed: number;
    stratifiedSeed: number;
    blockLength: number;
    stepsPerBlock: number;
    maxNewTokens: number;
    confidenceThreshold: number;
    causalContext: boolean;
    suppressMaskLogit: boolean;
  }): UniSessionBranch {
    const core = this.bundle.core;
    const cfg = core.config;
    const vocabSize = cfg.vocabSize;

    let scenePresetId: UniScenePresetId | undefined = params.scenePresetId;
    let customSourceImage: UniRgbImage | undefined;
    let expandedTokenIds: Int32Array = params.promptTokenIds;
    let prefillEmbeds: Float32Array | undefined;

    if (params.taskMode === 'visual_understanding') {
      let sourceImage: UniRgbImage;
      if (params.sourceImage !== undefined) {
        sourceImage = params.sourceImage;
        scenePresetId = params.scenePresetId ?? params.sourceImage.presetId;
        if (scenePresetId === undefined) {
          customSourceImage = sourceImage;
        }
      } else {
        scenePresetId = params.scenePresetId ?? 'sunlit_cat_16x16';
        sourceImage = createUniScenePreset(scenePresetId);
      }

      const visionTrace = runMiniVisionPipeline(
        sourceImage,
        this.bundle.vision,
        params.gainPresetId,
      );
      const fixed = fixOnePrompt(
        params.promptTokenIds,
        visionTrace.projector.tokenGridRows,
        visionTrace.projector.tokenGridCols,
      );
      expandedTokenIds = tightenInt32(fixed.expandedTokenIds);
      const textEmbeds = embedUniTextTokens(expandedTokenIds, core);
      prefillEmbeds = spliceVisualEmbeddingsIntoPrompt(
        expandedTokenIds,
        textEmbeds,
        fixed.visualSlotIndices,
        visionTrace.projector.visualEmbeddings,
        cfg.hiddenSize,
      );
    }

    const promptLen = expandedTokenIds.length;
    const prefillRouting = new Uint8Array(promptLen);
    const prefill = runUniForwardPass(core, {
      ...(prefillEmbeds !== undefined
        ? { inputsEmbeds: prefillEmbeds, routing: prefillRouting }
        : { tokenIds: expandedTokenIds }),
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
      computeTextHead: true,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_TRACE_LAYERS,
    });

    let prefixCache = createEmptyUniKvCache({
      batchSize: 1,
      capacity: cfg.kvCapacity,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
    });
    prefixCache = appendSingleRowKvsToCache(prefixCache, prefill.newKvs, prefill.routing);

    const nextLogitsContext = new Float32Array(
      prefill.textLogits!.subarray((promptLen - 1) * vocabSize, promptLen * vocabSize),
    );

    const branch: UniSessionBranch = Object.freeze({
      branchId: params.branchId,
      parentBranchId: null,
      forkRecordId: null,
      label: params.label,
      taskMode: params.taskMode,
      prefillMode: 'hf_faithful',
      gainPresetId: params.gainPresetId,
      cfgBatchSize: 1,
      guidanceScaleText: 0,
      guidanceScaleImage: 0,
      refinementSteps: params.refinementSteps,
      activationSteps: params.activationSteps,
      ...(scenePresetId !== undefined ? { scenePresetId } : {}),
      prefixCache,
      recordIds: Object.freeze([]),
      createdAtIndex: params.createdAtIndex,
    });

    const internal: InternalBranchState = {
      promptTokenIds: params.promptTokenIds,
      customSourceImage,
      lockSourcePaddingVq: false,
      cropToSource: false,
      shift: params.shift,
      temperature: params.temperature,
      topK: 0,
      topP: 1.0,
      seed: params.seed,
      treeInitSeed: params.seed,
      stratifiedSeed: params.stratifiedSeed,
      rng: createXoshiro128ss(params.seed),
      nextRoundIndex: 0,
      initialPromptLen: promptLen,
      blockLength: params.blockLength,
      stepsPerBlock: params.stepsPerBlock,
      maxNewTokens: params.maxNewTokens,
      generatedCount: 0,
      confidenceThreshold: params.confidenceThreshold,
      causalContext: params.causalContext,
      suppressMaskLogit: params.suppressMaskLogit,
      nextLogitsContext,
      currentBlockTokens: undefined,
      currentBlockStep: 0,
      currentBlockPastLen: prefixCache.seqLen,
      pendingPreSeedGuardInterventions: 0,
      groundingSteps: 2,
      enforceLocDomainGuard: true,
      groundingStepIndex: 0,
      lastRecordId: null,
      nextStepIndex: 0,
      finished: params.maxNewTokens <= 0,
    };

    this.branchesById.set(params.branchId, branch);
    this.branchInternals.set(params.branchId, internal);
    this.branchOrder.push(params.branchId);
    this.activeBranchId = params.branchId;

    this.appendRecord(
      params.branchId,
      {
        parentRecordId: null,
        stepIndex: 0,
        kind: 'prefill',
        taskMode: params.taskMode,
        prefillMode: 'hf_faithful',
        cfgBatchSize: 1,
        maskMode: 'causal',
        routing: tightenUint8(prefill.routing),
        positionIds: tightenInt32(prefill.positionIds),
        perBranchResiduals: Object.freeze([tightenResidualCheckpoints(prefill.residualCheckpoints)]),
        textState: Object.freeze({
          tokenIds: expandedTokenIds,
          unmaskedOffsets: Object.freeze([]),
          guardInterventions: 0,
        }),
      },
      { pastSeqLen: 0, usedCache: true },
    );

    return this.branchesById.get(params.branchId)!;
  }

  private startGroundingBranch(params: {
    branchId: string;
    label: string;
    createdAtIndex: number;
    promptTokenIds: Int32Array;
    scenePresetId?: UniScenePresetId;
    sourceImage?: UniRgbImage;
    prefillMode: UniElasticPrefillMode;
    gainPresetId: UniGainPresetId;
    refinementSteps: 1 | 2 | 4 | 8;
    activationSteps: 4 | 8 | 16;
    shift: number;
    seed: number;
    stratifiedSeed: number;
    groundingTags?: readonly string[];
    groundingSteps: 1 | 2 | 4;
    enforceLocDomainGuard: boolean;
  }): UniSessionBranch {
    const core = this.bundle.core;
    const cfg = core.config;
    const d = cfg.hiddenSize;
    const vocabSize = cfg.vocabSize;

    let scenePresetId: UniScenePresetId | undefined = params.scenePresetId;
    let customSourceImage: UniRgbImage | undefined;
    let sourceImage: UniRgbImage;
    if (params.sourceImage !== undefined) {
      sourceImage = params.sourceImage;
      scenePresetId = params.scenePresetId ?? params.sourceImage.presetId;
      if (scenePresetId === undefined) {
        customSourceImage = sourceImage;
      }
    } else {
      scenePresetId = params.scenePresetId ?? 'sunlit_cat_16x16';
      sourceImage = createUniScenePreset(scenePresetId);
    }

    const visionTrace = runMiniVisionPipeline(
      sourceImage,
      this.bundle.vision,
      params.gainPresetId,
    );
    const fixed = fixOnePrompt(
      params.promptTokenIds,
      visionTrace.projector.tokenGridRows,
      visionTrace.projector.tokenGridCols,
    );
    const expandedPromptIds = fixed.expandedTokenIds;
    const promptTextEmbeds = embedUniTextTokens(expandedPromptIds, core);
    const promptSplicedEmbeds = spliceVisualEmbeddingsIntoPrompt(
      expandedPromptIds,
      promptTextEmbeds,
      fixed.visualSlotIndices,
      visionTrace.projector.visualEmbeddings,
      d,
    );

    const tags: readonly string[] =
      params.groundingTags && params.groundingTags.length > 0
        ? params.groundingTags
        : sourceImage.groundTruthBoxes && sourceImage.groundTruthBoxes.length > 0
          ? [sourceImage.groundTruthBoxes[0].tag]
          : ['cat'];

    // Assemble single-tag or multi-tag grounding template (matching `src/uni/textSampler.ts`)
    const templateList: number[] = [];
    const slotTemplateIndices: number[] = [];
    if (tags.length === 1) {
      templateList.push(UNI_MINI_LOC_BEGIN_TOKEN_ID);
      for (let k = 0; k < 4; k++) {
        slotTemplateIndices.push(templateList.length);
        templateList.push(UNI_MINI_MASK_TOKEN_ID);
      }
      templateList.push(UNI_MINI_LOC_END_TOKEN_ID, UNI_MINI_EOS_TOKEN_ID);
    } else {
      for (let i = 0; i < tags.length; i++) {
        const tagId = resolveGroundingTagTokenId(tags[i]);
        templateList.push(
          UNI_MINI_BOX_OPEN_TOKEN_ID,
          tagId,
          UNI_MINI_BOX_CLOSE_TOKEN_ID,
          UNI_MINI_LOC_BEGIN_TOKEN_ID,
        );
        for (let k = 0; k < 4; k++) {
          slotTemplateIndices.push(templateList.length);
          templateList.push(UNI_MINI_MASK_TOKEN_ID);
        }
        templateList.push(UNI_MINI_LOC_END_TOKEN_ID);
      }
      templateList.push(UNI_MINI_EOS_TOKEN_ID);
    }

    const assembledTemplateIds = Int32Array.from(templateList);
    const promptLen = expandedPromptIds.length;
    const firstSlotTemplateIdx = slotTemplateIndices[0];
    const firstSlot = promptLen + firstSlotTemplateIdx;
    const slotSuffixIndices = new Int32Array(slotTemplateIndices.length);
    for (let k = 0; k < slotTemplateIndices.length; k++) {
      slotSuffixIndices[k] = slotTemplateIndices[k] - firstSlotTemplateIdx;
    }

    const prefillEmbeds = new Float32Array(firstSlot * d);
    prefillEmbeds.set(promptSplicedEmbeds, 0);
    if (firstSlotTemplateIdx > 0) {
      prefillEmbeds.set(
        embedUniTextTokens(assembledTemplateIds.subarray(0, firstSlotTemplateIdx), core),
        promptLen * d,
      );
    }

    const prefill = runUniForwardPass(core, {
      inputsEmbeds: prefillEmbeds,
      routing: new Uint8Array(firstSlot),
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
      computeTextHead: true,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_TRACE_LAYERS,
    });

    let prefixCache = createEmptyUniKvCache({
      batchSize: 1,
      capacity: cfg.kvCapacity,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
    });
    prefixCache = appendSingleRowKvsToCache(prefixCache, prefill.newKvs, prefill.routing);

    const prevLogits = new Float32Array(
      prefill.textLogits!.subarray((firstSlot - 1) * vocabSize, firstSlot * vocabSize),
    );
    const suffixTokens = tightenInt32(
      new Int32Array(assembledTemplateIds.subarray(firstSlotTemplateIdx)),
    );

    const prefillTokenIds = new Int32Array(firstSlot);
    prefillTokenIds.set(expandedPromptIds, 0);
    prefillTokenIds.set(assembledTemplateIds.subarray(0, firstSlotTemplateIdx), promptLen);

    const branch: UniSessionBranch = Object.freeze({
      branchId: params.branchId,
      parentBranchId: null,
      forkRecordId: null,
      label: params.label,
      taskMode: 'object_grounding',
      prefillMode: 'hf_faithful',
      gainPresetId: params.gainPresetId,
      cfgBatchSize: 1,
      guidanceScaleText: 0,
      guidanceScaleImage: 0,
      refinementSteps: params.refinementSteps,
      activationSteps: params.activationSteps,
      ...(scenePresetId !== undefined ? { scenePresetId } : {}),
      prefixCache,
      recordIds: Object.freeze([]),
      createdAtIndex: params.createdAtIndex,
    });

    const internal: InternalBranchState = {
      promptTokenIds: params.promptTokenIds,
      customSourceImage,
      lockSourcePaddingVq: false,
      cropToSource: false,
      shift: params.shift,
      temperature: 0,
      topK: 0,
      topP: 1.0,
      seed: params.seed,
      treeInitSeed: params.seed,
      stratifiedSeed: params.stratifiedSeed,
      rng: createXoshiro128ss(params.seed),
      nextRoundIndex: 0,
      initialPromptLen: firstSlot,
      blockLength: 4,
      stepsPerBlock: 4,
      maxNewTokens: 0,
      generatedCount: 0,
      confidenceThreshold: 0.9,
      causalContext: false,
      suppressMaskLogit: true,
      nextLogitsContext: prevLogits,
      currentBlockStep: 0,
      currentBlockPastLen: prefixCache.seqLen,
      pendingPreSeedGuardInterventions: 0,
      groundingTags: tags,
      groundingSteps: params.groundingSteps,
      enforceLocDomainGuard: params.enforceLocDomainGuard,
      initialGroundingSuffixTokens: new Int32Array(suffixTokens),
      groundingSuffixTokens: suffixTokens,
      groundingSlotSuffixIndices: slotSuffixIndices,
      groundingStepIndex: 0,
      lastRecordId: null,
      nextStepIndex: 0,
      finished: false,
    };

    this.branchesById.set(params.branchId, branch);
    this.branchInternals.set(params.branchId, internal);
    this.branchOrder.push(params.branchId);
    this.activeBranchId = params.branchId;

    this.appendRecord(
      params.branchId,
      {
        parentRecordId: null,
        stepIndex: 0,
        kind: 'prefill',
        taskMode: 'object_grounding',
        prefillMode: 'hf_faithful',
        cfgBatchSize: 1,
        maskMode: 'causal',
        routing: tightenUint8(prefill.routing),
        positionIds: tightenInt32(prefill.positionIds),
        perBranchResiduals: Object.freeze([tightenResidualCheckpoints(prefill.residualCheckpoints)]),
        textState: Object.freeze({
          tokenIds: prefillTokenIds,
          unmaskedOffsets: Object.freeze([]),
          guardInterventions: 0,
        }),
      },
      { pastSeqLen: 0, usedCache: true },
    );

    return this.branchesById.get(params.branchId)!;
  }

  // --------------------------------------------------------------------------
  // PRIVATE TASK STEP IMPLEMENTATIONS
  // --------------------------------------------------------------------------

  private stepImageGenerationBranch(
    branch: UniSessionBranch,
    internal: InternalBranchState,
  ): UniSessionRecord | null {
    const schedule = branch.schedule;
    const unmaskOrder = branch.unmaskOrder;
    const treeState = internal.treeState;
    if (!schedule || !unmaskOrder || !treeState) {
      internal.finished = true;
      return null;
    }
    if (internal.nextRoundIndex >= schedule.totalRounds) {
      internal.finished = true;
      return null;
    }

    const roundIndex = internal.nextRoundIndex;
    const { stepRecord, nextTreeState } = runSingleImageDenoiseRound(this.bundle, {
      prefixCache: branch.prefixCache,
      treeState,
      schedule,
      unmaskOrder,
      roundIndex,
      taskKind: branch.taskMode as 'text_to_image' | 'image_edit',
      guidanceMode: branch.guidanceMode,
      guidanceScaleText: branch.guidanceScaleText,
      guidanceScaleImage: branch.guidanceScaleImage,
      temperature: internal.temperature,
      topK: internal.topK,
      topP: internal.topP,
      gainPresetId: branch.gainPresetId,
      sourcePaddingMask: internal.lockSourcePaddingVq ? branch.sourcePaddingMask : undefined,
      sourceVqCodes: internal.lockSourcePaddingVq ? branch.sourceVqCodes : undefined,
      nextUniformOpenUnit: () => internal.rng.nextUniformOpenUnit(),
      nextRandBits: () => internal.rng.nextUint32(),
      retainFullLogits: false,
    });

    internal.treeState = nextTreeState;
    internal.nextRoundIndex = roundIndex + 1;
    if (internal.nextRoundIndex >= schedule.totalRounds) {
      internal.finished = true;
    }

    const windowLen = stepRecord.parentPositionIds.length; // 10 (4 fixed prefix + 6 packed parents)
    const routing = new Uint8Array(windowLen).fill(UNI_ROUTE_G);

    const record = this.appendRecord(
      branch.branchId,
      {
        parentRecordId: internal.lastRecordId,
        stepIndex: internal.nextStepIndex,
        kind: 'image_denoise',
        taskMode: branch.taskMode,
        prefillMode: branch.prefillMode,
        cfgBatchSize: branch.cfgBatchSize,
        maskMode: 'bidirectional',
        routing,
        positionIds: tightenInt32(stepRecord.parentPositionIds),
        perBranchResiduals: tightenPerBranchResiduals(stepRecord.perBranchResiduals),
        imageState: Object.freeze({
          roundIndex: stepRecord.roundIndex,
          isFlushRound: stepRecord.isFlushRound,
          stages: tightenInt32(stepRecord.stagesAfter),
          representatives: tightenInt32(stepRecord.representativesAfter),
          proposals: tightenInt32(stepRecord.proposals),
          levels: tightenInt32(stepRecord.levelsAfter),
          noiseWeights: tightenFloat32(stepRecord.noiseWeightsAfter),
          top4Guided: Object.freeze({
            tokenIds: tightenInt32(stepRecord.top4Guided.tokenIds),
            probs: tightenFloat32(stepRecord.top4Guided.probs),
          }),
          lockedCellCount: stepRecord.lockedCellCount,
          committedContentCount: stepRecord.committedContentCount,
        }),
      },
      { pastSeqLen: branch.prefixCache.seqLen, usedCache: false },
    );

    return record;
  }

  private stepTextOrVisualBranch(
    branch: UniSessionBranch,
    internal: InternalBranchState,
  ): UniSessionRecord | null {
    const core = this.bundle.core;
    const cfg = core.config;
    const vocabSize = cfg.vocabSize;
    const blockLength = internal.blockLength;

    if (internal.generatedCount >= internal.maxNewTokens) {
      internal.finished = true;
      return null;
    }

    // Initialize a new block if none is active
    if (!internal.currentBlockTokens) {
      if (branch.prefixCache.seqLen + blockLength > branch.prefixCache.capacity) {
        internal.finished = true;
        return null;
      }
      const blockTokens = new Int32Array(blockLength).fill(UNI_MINI_MASK_TOKEN_ID);
      internal.currentBlockPastLen = branch.prefixCache.seqLen;
      internal.currentBlockStep = 0;

      // Optional causalContext pre-seeding of slot 0 (preserving U21 guard intervention count!)
      internal.pendingPreSeedGuardInterventions = 0;
      if (internal.causalContext && internal.nextLogitsContext) {
        const outcome0 = this.decodeTextSlotRow(
          internal.nextLogitsContext,
          internal.suppressMaskLogit,
          internal.temperature,
          internal.rng,
          true,
        );
        blockTokens[0] = outcome0.proposed;
        if (outcome0.guardIntervened) {
          internal.pendingPreSeedGuardInterventions += 1;
        }
      }
      internal.currentBlockTokens = blockTokens;
    }

    const blockTokens = internal.currentBlockTokens;
    const pastLen = internal.currentBlockPastLen;
    const { pastKvs, pastRouting } = extractRowPastKvs(
      cropUniKvCache(branch.prefixCache, pastLen),
      0,
    );

    const maskedOffsets: number[] = [];
    for (let j = 0; j < blockLength; j++) {
      if (blockTokens[j] === UNI_MINI_MASK_TOKEN_ID) {
        maskedOffsets.push(j);
      }
    }

    // Special AR limit (`blockLength === 1` with `causalContext === true`): commit directly
    if (maskedOffsets.length === 0) {
      const commit = runUniForwardPass(core, {
        tokenIds: blockTokens,
        maskMode: 'causal',
        useCache: true,
        prefillMode: 'hf_faithful',
        downstreamTarget: 'u',
        pastKvs,
        pastRouting,
        computeTextHead: true,
        computeGenHead: false,
        retainedTraceLayers: EMPTY_TRACE_LAYERS,
      });

      const arGuardInterventions = internal.pendingPreSeedGuardInterventions;
      internal.pendingPreSeedGuardInterventions = 0;
      const committedCacheLen = this.finalizeTextBlockCommit(
        branch,
        internal,
        blockTokens,
        commit.newKvs,
        commit.routing,
        commit.textLogits!,
      );

      return this.appendRecord(
        branch.branchId,
        {
          parentRecordId: internal.lastRecordId,
          stepIndex: internal.nextStepIndex,
          kind: 'text_block_step',
          taskMode: branch.taskMode,
          prefillMode: branch.prefillMode,
          cfgBatchSize: 1,
          maskMode: 'causal',
          routing: tightenUint8(commit.routing),
          positionIds: tightenInt32(commit.positionIds),
          perBranchResiduals: Object.freeze([tightenResidualCheckpoints(commit.residualCheckpoints)]),
          textState: Object.freeze({
            tokenIds: new Int32Array(blockTokens),
            unmaskedOffsets: Object.freeze([0]),
            guardInterventions: arGuardInterventions,
          }),
        },
        {
          pastSeqLen: pastLen,
          usedCache: true,
          blockStepAfter: 0,
          committedOnStep: true,
          committedCacheLen,
          finishedOnStep: internal.finished,
        },
      );
    }

    // Standard bidirectional block-diffusion denoising step
    const fwd = runUniForwardPass(core, {
      tokenIds: blockTokens,
      maskMode: 'bidirectional',
      useCache: false,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
      pastKvs,
      pastRouting,
      computeTextHead: true,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_TRACE_LAYERS,
    });

    const fwdLogits = fwd.textLogits!;
    const shiftedLogits = new Float32Array(blockLength * vocabSize);
    if (internal.nextLogitsContext) {
      shiftedLogits.set(internal.nextLogitsContext, 0);
    }
    for (let j = 1; j < blockLength; j++) {
      shiftedLogits.set(
        fwdLogits.subarray((j - 1) * vocabSize, j * vocabSize),
        j * vocabSize,
      );
    }

    const slotConfidences = new Float32Array(blockLength);
    const proposals = new Int32Array(blockLength);
    let guardInterventions = internal.pendingPreSeedGuardInterventions;
    internal.pendingPreSeedGuardInterventions = 0;

    for (let j = 0; j < blockLength; j++) {
      const isMasked = blockTokens[j] === UNI_MINI_MASK_TOKEN_ID;
      const rowLogits = shiftedLogits.subarray(j * vocabSize, (j + 1) * vocabSize);
      const outcome = this.decodeTextSlotRow(
        rowLogits,
        internal.suppressMaskLogit,
        internal.temperature,
        internal.rng,
        isMasked,
      );
      slotConfidences[j] = outcome.confidence;
      proposals[j] = outcome.proposed;
      if (isMasked && outcome.guardIntervened) {
        guardInterventions += 1;
      }
    }

    const remainingSteps = Math.max(1, internal.stepsPerBlock - internal.currentBlockStep);
    const quota = Math.ceil(maskedOffsets.length / remainingSteps);
    const aboveThreshold: number[] = [];
    for (let i = 0; i < maskedOffsets.length; i++) {
      const off = maskedOffsets[i];
      if (slotConfidences[off] >= internal.confidenceThreshold) {
        aboveThreshold.push(off);
      }
    }

    let selected: number[];
    if (aboveThreshold.length >= quota) {
      selected = aboveThreshold;
    } else {
      const ranked = [...maskedOffsets].sort((a, b) => {
        const diff = slotConfidences[b] - slotConfidences[a];
        return diff !== 0 ? diff : a - b;
      });
      selected = ranked.slice(0, quota).sort((a, b) => a - b);
    }

    for (let i = 0; i < selected.length; i++) {
      const off = selected[i];
      blockTokens[off] = proposals[off];
    }
    internal.currentBlockStep += 1;

    const snapshotTokens = new Int32Array(blockTokens);

    // Check if the block is now fully unmasked -> run causal commit & append KV slabs
    let anyMaskLeft = false;
    for (let j = 0; j < blockLength; j++) {
      if (blockTokens[j] === UNI_MINI_MASK_TOKEN_ID) {
        anyMaskLeft = true;
        break;
      }
    }
    let committedOnStep = false;
    let committedCacheLen: number | undefined;
    if (!anyMaskLeft || internal.currentBlockStep >= internal.stepsPerBlock) {
      const commit = runUniForwardPass(core, {
        tokenIds: blockTokens,
        maskMode: 'causal',
        useCache: true,
        prefillMode: 'hf_faithful',
        downstreamTarget: 'u',
        pastKvs,
        pastRouting,
        computeTextHead: true,
        computeGenHead: false,
        retainedTraceLayers: EMPTY_TRACE_LAYERS,
      });
      committedOnStep = true;
      committedCacheLen = this.finalizeTextBlockCommit(
        branch,
        internal,
        blockTokens,
        commit.newKvs,
        commit.routing,
        commit.textLogits!,
      );
    }

    return this.appendRecord(
      branch.branchId,
      {
        parentRecordId: internal.lastRecordId,
        stepIndex: internal.nextStepIndex,
        kind: 'text_block_step',
        taskMode: branch.taskMode,
        prefillMode: branch.prefillMode,
        cfgBatchSize: 1,
        maskMode: 'bidirectional',
        routing: tightenUint8(fwd.routing),
        positionIds: tightenInt32(fwd.positionIds),
        perBranchResiduals: Object.freeze([tightenResidualCheckpoints(fwd.residualCheckpoints)]),
        textState: Object.freeze({
          tokenIds: snapshotTokens,
          unmaskedOffsets: Object.freeze(selected),
          guardInterventions,
        }),
      },
      {
        pastSeqLen: pastLen,
        usedCache: false,
        blockStepAfter: committedOnStep ? 0 : internal.currentBlockStep,
        committedOnStep,
        ...(committedCacheLen !== undefined ? { committedCacheLen } : {}),
        finishedOnStep: internal.finished,
      },
    );
  }

  private finalizeTextBlockCommit(
    branch: UniSessionBranch,
    internal: InternalBranchState,
    blockTokens: Int32Array,
    commitKvs: readonly (UniLayerKvSlice | null)[],
    commitRouting: Uint8Array,
    commitLogits: Float32Array,
  ): number {
    const vocabSize = this.bundle.core.config.vocabSize;
    const blockLength = blockTokens.length;
    const prevLen = branch.prefixCache.seqLen;
    let updatedCache = appendSingleRowKvsToCache(branch.prefixCache, commitKvs, commitRouting);

    internal.nextLogitsContext = new Float32Array(
      commitLogits.subarray((blockLength - 1) * vocabSize, blockLength * vocabSize),
    );

    const remainingBudget = internal.maxNewTokens - internal.generatedCount;
    const limit = Math.min(blockLength, remainingBudget);
    let eosInBlock = -1;
    for (let i = 0; i < limit; i++) {
      if (blockTokens[i] === UNI_MINI_EOS_TOKEN_ID) {
        eosInBlock = i;
        break;
      }
    }

    if (eosInBlock >= 0) {
      const kept = eosInBlock + 1;
      internal.generatedCount += kept;
      updatedCache = cropUniKvCache(updatedCache, prevLen + kept);
      internal.finished = true;
    } else {
      internal.generatedCount += limit;
      if (limit < blockLength) {
        updatedCache = cropUniKvCache(updatedCache, prevLen + limit);
        internal.finished = true;
      } else if (internal.generatedCount >= internal.maxNewTokens) {
        internal.finished = true;
      }
    }

    // Update branch's immutable prefixCache snapshot
    const updatedBranch: UniSessionBranch = Object.freeze({
      ...branch,
      prefixCache: updatedCache,
    });
    this.branchesById.set(branch.branchId, updatedBranch);
    internal.currentBlockTokens = undefined;
    internal.currentBlockStep = 0;
    return updatedCache.seqLen;
  }

  private stepGroundingBranch(
    branch: UniSessionBranch,
    internal: InternalBranchState,
  ): UniSessionRecord | null {
    const suffixTokens = internal.groundingSuffixTokens;
    const slotSuffixIndices = internal.groundingSlotSuffixIndices;
    const prevLogits = internal.nextLogitsContext;
    if (!suffixTokens || !slotSuffixIndices || !prevLogits) {
      internal.finished = true;
      return null;
    }

    const totalSlots = slotSuffixIndices.length;
    const remainingSlots: number[] = [];
    for (let k = 0; k < totalSlots; k++) {
      if (suffixTokens[slotSuffixIndices[k]] === UNI_MINI_MASK_TOKEN_ID) {
        remainingSlots.push(k);
      }
    }
    if (remainingSlots.length === 0 || internal.groundingStepIndex >= internal.groundingSteps) {
      internal.finished = true;
      return null;
    }

    const core = this.bundle.core;
    const vocabSize = core.config.vocabSize;
    const pastLen = branch.prefixCache.seqLen;
    const { pastKvs, pastRouting } = extractRowPastKvs(branch.prefixCache, 0);

    const fwd = runUniForwardPass(core, {
      tokenIds: suffixTokens,
      maskMode: 'bidirectional',
      useCache: false,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
      pastKvs,
      pastRouting,
      computeTextHead: true,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_TRACE_LAYERS,
    });

    const fwdLogits = fwd.textLogits!;
    const slotConfidences = new Float32Array(totalSlots);
    const proposals = new Int32Array(totalSlots);
    let guardInterventions = 0;

    for (let k = 0; k < totalSlots; k++) {
      const off = slotSuffixIndices[k];
      const logitsRow =
        off === 0
          ? prevLogits
          : fwdLogits.subarray((off - 1) * vocabSize, off * vocabSize);
      const rawTop1 = argmaxFinite(logitsRow, vocabSize);
      const isMasked = suffixTokens[off] === UNI_MINI_MASK_TOKEN_ID;

      if (internal.enforceLocDomainGuard) {
        const maskedRow = new Float32Array(vocabSize).fill(-Infinity);
        for (let v = UNI_MINI_LOC_START_ID; v <= UNI_MINI_LOC_END_ID; v++) {
          maskedRow[v] = logitsRow[v];
        }
        const probs = softmaxRows(maskedRow, 1, vocabSize);
        const proposed = argmaxFinite(maskedRow, vocabSize);
        proposals[k] = proposed;
        slotConfidences[k] = probs[proposed];
        if (isMasked && (rawTop1 < UNI_MINI_LOC_START_ID || rawTop1 > UNI_MINI_LOC_END_ID)) {
          guardInterventions += 1;
        }
      } else {
        const probs = softmaxRows(logitsRow, 1, vocabSize);
        proposals[k] = rawTop1;
        slotConfidences[k] = probs[rawTop1];
      }
    }

    const stepsLeft = Math.max(1, internal.groundingSteps - internal.groundingStepIndex);
    const quota = Math.ceil(remainingSlots.length / stepsLeft);
    const ranked = [...remainingSlots].sort((a, b) => {
      const diff = slotConfidences[b] - slotConfidences[a];
      return diff !== 0 ? diff : a - b;
    });
    const selected = ranked.slice(0, quota).sort((a, b) => a - b);

    for (let i = 0; i < selected.length; i++) {
      const slotIdx = selected[i];
      suffixTokens[slotSuffixIndices[slotIdx]] = proposals[slotIdx];
    }

    internal.groundingStepIndex += 1;
    let masksLeft = 0;
    for (let k = 0; k < totalSlots; k++) {
      if (suffixTokens[slotSuffixIndices[k]] === UNI_MINI_MASK_TOKEN_ID) {
        masksLeft += 1;
      }
    }
    if (masksLeft === 0 || internal.groundingStepIndex >= internal.groundingSteps) {
      internal.finished = true;
    }

    return this.appendRecord(
      branch.branchId,
      {
        parentRecordId: internal.lastRecordId,
        stepIndex: internal.nextStepIndex,
        kind: 'grounding_step',
        taskMode: 'object_grounding',
        prefillMode: branch.prefillMode,
        cfgBatchSize: 1,
        maskMode: 'bidirectional',
        routing: tightenUint8(fwd.routing),
        positionIds: tightenInt32(fwd.positionIds),
        perBranchResiduals: Object.freeze([tightenResidualCheckpoints(fwd.residualCheckpoints)]),
        textState: Object.freeze({
          tokenIds: new Int32Array(suffixTokens),
          unmaskedOffsets: Object.freeze(selected),
          guardInterventions,
        }),
      },
      { pastSeqLen: pastLen, usedCache: false, groundingStepAfter: internal.groundingStepIndex },
    );
  }

  /** Walk record ancestry to find the branch's prefill record ID (if still retained). */
  private findBranchPrefillRecordId(
    fromRecord: UniSessionRecord,
    parentBranch: UniSessionBranch,
  ): string | null {
    let cursor: UniSessionRecord | undefined = fromRecord;
    let hops = 0;
    while (cursor && hops < 256) {
      if (cursor.kind === 'prefill') {
        return cursor.recordId;
      }
      cursor =
        cursor.parentRecordId !== null
          ? this.recordsById.get(cursor.parentRecordId)
          : undefined;
      hops += 1;
    }
    const firstId = parentBranch.recordIds[0];
    if (firstId) {
      const firstRec = this.recordsById.get(firstId);
      if (firstRec && firstRec.kind === 'prefill') {
        return firstRec.recordId;
      }
    }
    return null;
  }

  /**
   * Reconstruct the exact causal context tail logits (`[vocabSize]`) immediately preceding
   * a mid-block `text_block_step` record `fromRecord` (never leaking later parent blocks).
   */
  private reconstructPriorCausalLogits(
    fromRecord: UniSessionRecord,
    parentBranch: UniSessionBranch,
  ): Float32Array | undefined {
    const vocabSize = this.bundle.core.config.vocabSize;
    let cursorId = fromRecord.parentRecordId;
    let hops = 0;
    while (cursorId !== null && hops < 256) {
      const rec = this.recordsById.get(cursorId);
      const meta = this.recordInternals.get(cursorId);
      if (!rec || !meta) break;
      if (rec.kind === 'prefill' || (rec.kind === 'text_block_step' && rec.maskMode === 'causal')) {
        const head = this.reconstructStepHeadLogits(rec.recordId);
        const logits = head.perBranchTextLogits[0];
        if (logits) {
          const seqLen = rec.routing.length;
          return new Float32Array(logits.subarray((seqLen - 1) * vocabSize, seqLen * vocabSize));
        }
      }
      if (rec.kind === 'text_block_step' && meta.committedOnStep && rec.textState) {
        const blockTokens = rec.textState.tokenIds;
        const preCommitCache = cropUniKvCache(parentBranch.prefixCache, meta.pastSeqLen);
        const { pastKvs, pastRouting } = extractRowPastKvs(preCommitCache, 0);
        const commitFwd = runUniForwardPass(this.bundle.core, {
          tokenIds: blockTokens,
          maskMode: 'causal',
          useCache: true,
          prefillMode: 'hf_faithful',
          downstreamTarget: 'u',
          pastKvs,
          pastRouting,
          computeTextHead: true,
          computeGenHead: false,
          retainedTraceLayers: EMPTY_TRACE_LAYERS,
        });
        const commitLogits = commitFwd.textLogits!;
        const bLen = blockTokens.length;
        return new Float32Array(commitLogits.subarray((bLen - 1) * vocabSize, bLen * vocabSize));
      }
      cursorId = rec.parentRecordId;
      hops += 1;
    }
    return undefined;
  }

  private decodeTextSlotRow(
    rawLogitsRow: Float32Array,
    suppressMaskLogit: boolean,
    temperature: number,
    rng: UniXoshiro128ssGenerator,
    consumeStochastic: boolean,
  ): { proposed: number; confidence: number; guardIntervened: boolean } {
    const vocabSize = rawLogitsRow.length;
    const rawTop1 = argmaxFinite(rawLogitsRow, vocabSize);
    const guardIntervened = suppressMaskLogit && rawTop1 === UNI_MINI_MASK_TOKEN_ID;

    const maskedLogits = new Float32Array(rawLogitsRow);
    if (suppressMaskLogit) {
      maskedLogits[UNI_MINI_MASK_TOKEN_ID] = -Infinity;
    }
    const probs = softmaxRows(maskedLogits, 1, vocabSize);
    const proposed =
      consumeStochastic && temperature > 0
        ? sampleLogDomainGumbelArgmax(maskedLogits, temperature, {
            nextOpenUnit: () => rng.nextUniformOpenUnit(),
          })
        : argmaxFinite(maskedLogits, vocabSize);

    return {
      proposed,
      confidence: probs[proposed],
      guardIntervened,
    };
  }

  // --------------------------------------------------------------------------
  // PRIVATE RECORD STORAGE, EVICTION & ROUTING HELPERS
  // --------------------------------------------------------------------------

  private appendRecord(
    branchId: string,
    payload: {
      parentRecordId: string | null;
      stepIndex: number;
      kind: UniSessionRecordKind;
      taskMode: UniSessionTaskMode;
      prefillMode: UniElasticPrefillMode;
      cfgBatchSize: UniCfgBatchSize;
      maskMode: UniAttentionMaskMode;
      routing: Uint8Array;
      positionIds: Int32Array;
      perBranchResiduals: readonly (readonly (Float32Array | null)[])[];
      imageState?: UniSessionRecord['imageState'];
      textState?: UniSessionRecord['textState'];
    },
    meta: InternalRecordMeta,
  ): UniSessionRecord {
    // Enforce <= 128 records by evicting oldest non-active record if full
    this.evictRecordsIfNeeded();

    const recordId = `rec-${++this.nextRecordSeq}`;
    const record: UniSessionRecord = Object.freeze({
      recordId,
      branchId,
      parentRecordId: payload.parentRecordId,
      stepIndex: payload.stepIndex,
      kind: payload.kind,
      taskMode: payload.taskMode,
      prefillMode: payload.prefillMode,
      cfgBatchSize: payload.cfgBatchSize,
      maskMode: payload.maskMode,
      routing: payload.routing,
      positionIds: payload.positionIds,
      perBranchResiduals: payload.perBranchResiduals,
      ...(payload.imageState !== undefined ? { imageState: payload.imageState } : {}),
      ...(payload.textState !== undefined ? { textState: payload.textState } : {}),
    });

    this.recordsById.set(recordId, record);
    this.recordInternals.set(recordId, Object.freeze(meta));
    this.recordOrder.push(recordId);

    const branch = this.branchesById.get(branchId)!;
    const internal = this.branchInternals.get(branchId)!;
    const updatedBranch: UniSessionBranch = Object.freeze({
      ...branch,
      recordIds: Object.freeze([...branch.recordIds, recordId]),
    });
    this.branchesById.set(branchId, updatedBranch);
    internal.lastRecordId = recordId;
    internal.nextStepIndex = payload.stepIndex + 1;

    // Evict previous activeInspection immediately; materialize lazily on inspection/snapshot/audit
    this.activeInspection = null;
    this.activeRecordId = recordId;
    this.activeBranchId = branchId;

    return record;
  }

  private evictBranchesIfNeeded(protectedBranchId: string | null): void {
    while (this.branchOrder.length >= this.maxBranches) {
      let victimIdx = -1;
      for (let i = 0; i < this.branchOrder.length; i++) {
        const candidateId = this.branchOrder[i];
        if (candidateId !== this.activeBranchId && candidateId !== protectedBranchId) {
          victimIdx = i;
          break;
        }
      }
      if (victimIdx < 0) {
        // Fallback: protect `protectedBranchId` over `activeBranchId`
        for (let i = 0; i < this.branchOrder.length; i++) {
          if (this.branchOrder[i] !== protectedBranchId) {
            victimIdx = i;
            break;
          }
        }
      }
      if (victimIdx < 0) {
        victimIdx = 0;
      }

      const victimBranchId = this.branchOrder.splice(victimIdx, 1)[0];
      const victimBranch = this.branchesById.get(victimBranchId);
      if (victimBranch) {
        const victimRecordSet = new Set(victimBranch.recordIds);
        for (const recId of victimRecordSet) {
          this.recordsById.delete(recId);
          this.recordInternals.delete(recId);
          if (this.activeRecordId === recId) {
            this.activeRecordId = null;
            this.activeInspection = null;
          }
        }
        // Compact `recordOrder` in place
        let write = 0;
        for (let i = 0; i < this.recordOrder.length; i++) {
          if (!victimRecordSet.has(this.recordOrder[i])) {
            this.recordOrder[write++] = this.recordOrder[i];
          }
        }
        this.recordOrder.length = write;
      }
      this.branchesById.delete(victimBranchId);
      this.branchInternals.delete(victimBranchId);
      if (this.activeBranchId === victimBranchId) {
        this.activeBranchId = null;
      }
    }
  }

  private evictRecordsIfNeeded(): void {
    while (this.recordOrder.length >= this.maxRecords) {
      let victimIdx = -1;
      for (let i = 0; i < this.recordOrder.length; i++) {
        if (this.recordOrder[i] !== this.activeRecordId) {
          victimIdx = i;
          break;
        }
      }
      if (victimIdx < 0) {
        victimIdx = 0;
      }

      const victimRecordId = this.recordOrder.splice(victimIdx, 1)[0];
      const victimRecord = this.recordsById.get(victimRecordId);
      this.recordsById.delete(victimRecordId);
      this.recordInternals.delete(victimRecordId);

      if (victimRecord) {
        const ownerBranch = this.branchesById.get(victimRecord.branchId);
        if (ownerBranch) {
          const filteredIds = ownerBranch.recordIds.filter((id) => id !== victimRecordId);
          this.branchesById.set(
            ownerBranch.branchId,
            Object.freeze({
              ...ownerBranch,
              recordIds: Object.freeze(filteredIds),
            }),
          );
        }
      }
      if (this.activeRecordId === victimRecordId) {
        this.activeRecordId = null;
        this.activeInspection = null;
      }
    }
  }

  private resolveRecordRowRouting(
    record: UniSessionRecord,
    branch: UniSessionBranch,
    batchRow: number,
  ): Uint8Array {
    if (
      record.kind === 'prefill' &&
      record.cfgBatchSize > 1 &&
      branch.prefixCache.seqLen >= record.routing.length
    ) {
      const seqLen = record.routing.length;
      const rowRouting = new Uint8Array(seqLen);
      for (let t = 0; t < seqLen; t++) {
        rowRouting[t] = branch.prefixCache.slabs[t].routing[batchRow];
      }
      return rowRouting;
    }
    return record.routing;
  }

  private resolveBranchSourceImage(
    branch: UniSessionBranch,
    internal: InternalBranchState,
  ): UniRgbImage {
    if (internal.customSourceImage !== undefined) {
      return internal.customSourceImage;
    }
    return createUniScenePreset(branch.scenePresetId ?? 'harbor_panorama_16x8');
  }
}

/**
 * Factory creating a bounded `<= 12.00 MiB` interactive `UniSession`.
 */
export function createUniSession(bundle?: UniFullRuntimeBundle): UniSession {
  return new UniSession(bundle);
}
