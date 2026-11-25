/**
 * TwoTower Embedding (NoPE) & Output / Untied LM Head Bookend Cutaways
 * (`src/twotower/scene/twoTowerBookendCutaways.ts`)
 *
 * Pure headless TypeScript (zero DOM, WebGL2, or `Math.random()`).
 * Directly reuses Nano's verified `buildEmbeddingCutaway` and `buildOutputCutaway` over each
 * tower's independently owned `HybridModelParameters` (`model.contextModel` vs `model.denoiserModel`,
 * `V = 34` with `[MASK]` `#32` & `<eos>` `#33`), actual executed token IDs/labels, actual
 * `residualStreams[0]` / `residualStreams[52]`, actual unmodulated `finalNorm`, and actual
 * `startTokenOffset` (`0` / `P` / `P - 1`), while:
 *   1. Re-packing a strictly collision-free RG32F value atlas for `V = 34` (so rows `#32` `[MASK]`
 *      and `#33` `<eos>` never collide with `y = 32` activation slices in Nano's 32-vocab layout).
 *   2. Propagating full tower-aware `PickBox` metadata (`towerRole`, `executionId`, `traceOrigin`,
 *      `absoluteTokenOffset`) and distinct independent weight table fingerprints (`W_embed_ctx` vs
 *      `W_embed_den`, `W_lm_head_ctx` vs `W_lm_head_den`).
 *   3. Explicitly distinguishing Context AR Next-Token Prediction (shifted causal `t -> next`) vs
 *      Denoiser Unshifted Block Logits (position `P + j` directly predicts clean `x_0[P + j]`)
 *      and confirming `norm_f` is strictly UNMODULATED in both towers.
 *   4. Attaching the full presentation-safe TT2 `SamplerStepPresentation` (strictly separating Raw
 *      LM Head, Constrained MDLM with `[MASK] = 0.0000`, Selected Candidates, Unscaled Confidence
 *      with `"Locked (prior step)"` — ZERO `Infinity` — and Commitment Decisions) plus explicit
 *      synthetic-miniature disclosure.
 */

import {
  applyRmsNorm,
  embedTokens,
  runUntiedLmHead,
  type HybridFinalNormResult,
  type HybridLmHeadResult,
  type HybridModelParameters,
  type HybridPrediction,
} from '../../math/hybridFlow';
import type { PickBox } from '../../render/pick';
import { SHEET_INSTANCE_FLOATS } from '../../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  buildEmbeddingCutaway,
  EMBEDDING_SPATIAL_FIT_BOUNDS,
  EMBEDDING_STEPPER_STEPS,
  fitSpatialEmbeddingCamera,
  pickHitToEmbeddingCell,
  type EmbeddingCellRef,
} from '../../scene/embeddingCutaway';
import {
  buildOutputCutaway,
  fitSpatialOutputCamera,
  OUTPUT_SPATIAL_FIT_BOUNDS,
  OUTPUT_STEPPER_STEPS,
  parseOutputCandidatePartId,
  pickHitToOutputCell,
  type OutputCellRef,
} from '../../scene/outputCutaway';
import type { LandmarkAnchor } from '../../scene/spine';
import type { TwoTowerTraceOrigin } from '../contracts';
import {
  assertPresentationSafeNoInfinity,
  buildSamplerStepPresentation,
  deriveRefinementUniformVariates,
  runConfidenceUnmaskStep,
  type SamplerStepPresentation,
} from '../sampler';
import type { TwoTowerSession } from '../session';
import {
  buildTwoTowerSelectionIdentity,
  resolveTwoTowerSessionViewState,
  type TwoTowerProvenanceBadge,
  type TwoTowerSessionViewState,
  type TwoTowerSpatialScene,
} from './provenance';

export {
  EMBEDDING_SPATIAL_FIT_BOUNDS,
  EMBEDDING_STEPPER_STEPS,
  fitSpatialEmbeddingCamera,
  pickHitToEmbeddingCell,
  OUTPUT_SPATIAL_FIT_BOUNDS,
  OUTPUT_STEPPER_STEPS,
  fitSpatialOutputCamera,
  pickHitToOutputCell,
  parseOutputCandidatePartId,
};

export const TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE =
  'Educational synthetic miniature TwoTower simulator (d = 8, V = 34 incl. miniature [MASK] #32 and <eos> #33, deterministic seeded weights). NOT 63.19B BF16 checkpoint weights and NOT real 131,072-token Nemotron tokenizer probabilities.' as const;

export interface BuildTwoTowerEmbeddingOptions {
  /** Tower to inspect (defaults to `'denoiser'` when a Denoiser pass exists, else `'context'`). */
  readonly towerRole?: 'context' | 'denoiser';
  readonly focusToken?: number;
  readonly selectedPart?: string | null;
  readonly hoveredPart?: string | null;
  readonly activeCell?: EmbeddingCellRef | null;
  /** Optional explicit Context segment index when inspecting split prefill/block-commit segments. */
  readonly contextSegmentIndex?: number;
}

export interface TwoTowerEmbeddingDiagnostics {
  readonly towerRole: 'context' | 'denoiser';
  readonly weightTableName: 'W_embed_ctx' | 'W_embed_den';
  readonly towerModelId: string;
  readonly towerParameterFingerprint: string;
  readonly otherTowerParameterFingerprint: string;
  readonly weightsIndependentOfOtherTower: true;
  readonly untiedFromLmHead: true;
  readonly positionalEncoding: 'NoPE';
  readonly vocabSize: number;
  readonly maskTokenId: number;
  readonly eosTokenId: number;
  readonly tokenCount: number;
  readonly startTokenOffset: number;
  readonly absoluteTokenRange: readonly [number, number];
  readonly tokenIds: readonly number[];
  readonly tokenLabels: readonly string[];
  readonly maskTokenLocalIndices: readonly number[];
  readonly maskTokenAbsolutePositions: readonly number[];
  readonly eosTokenLocalIndices: readonly number[];
  readonly eosTokenAbsolutePositions: readonly number[];
  readonly hasMaskTokens: boolean;
  readonly hasEosTokens: boolean;
  readonly contextTraceStatus: 'available' | 'pruned_unavailable' | 'not_applicable';
  readonly isHistoricalDenoiserInspection: boolean;
  readonly matchesCurrentContextBoundary: boolean;
  readonly denoiserSourceRecordIndex: number | null;
  readonly denoiserSourceBlockId: number | null;
  readonly denoiserSourceRefinementIndex: number | null;
  readonly denoiserSourceContextPrefixLen: number | null;
  readonly denoiserSourceContextCacheDigest: string | null;
  readonly historicalProvenanceWarning: string | null;
  readonly syntheticMiniatureDisclosure: string;
}

export type TwoTowerOutputPredictionSemantics =
  | 'causal_next_token_shifted'
  | 'unshifted_block_diffusion';

export interface BuildTwoTowerOutputOptions {
  /** Tower to inspect (defaults to `'denoiser'` when a Denoiser pass exists, else `'context'`). */
  readonly towerRole?: 'context' | 'denoiser';
  readonly focusToken?: number;
  readonly selectedPart?: string | null;
  readonly hoveredPart?: string | null;
  readonly activeCell?: OutputCellRef | null;
  readonly selectedCandidateTokenId?: number | null;
}

export interface TwoTowerOutputDiagnostics {
  readonly towerRole: 'context' | 'denoiser';
  readonly weightTableName: 'W_lm_head_ctx' | 'W_lm_head_den';
  readonly towerModelId: string;
  readonly towerParameterFingerprint: string;
  readonly otherTowerParameterFingerprint: string;
  readonly weightsIndependentOfOtherTower: true;
  readonly untiedFromEmbedding: true;
  readonly finalNormModulated: false;
  readonly finalNormNote: string;
  readonly predictionSemantics: TwoTowerOutputPredictionSemantics;
  readonly predictionSemanticsLabel: string;
  readonly vocabSize: number;
  readonly maskTokenId: number;
  readonly eosTokenId: number;
  readonly tokenCount: number;
  readonly startTokenOffset: number;
  readonly absoluteTokenRange: readonly [number, number];
  readonly focusToken: number;
  readonly absoluteFocusPosition: number;
  readonly contextTraceStatus: 'available' | 'pruned_unavailable' | 'not_applicable';
  readonly isHistoricalDenoiserInspection: boolean;
  readonly matchesCurrentContextBoundary: boolean;
  readonly denoiserSourceRecordIndex: number | null;
  readonly denoiserSourceBlockId: number | null;
  readonly denoiserSourceRefinementIndex: number | null;
  readonly denoiserSourceContextPrefixLen: number | null;
  readonly denoiserSourceContextCacheDigest: string | null;
  readonly historicalProvenanceWarning: string | null;
  readonly samplerPresentationSource: 'recorded_commit' | 'live_refinement_preview' | 'none';
  readonly samplerPresentation: SamplerStepPresentation | null;
  readonly top5AtFocus: readonly HybridPrediction[];
  readonly syntheticMiniatureDisclosure: string;
}

// ============================================================================
// COLLISION-FREE V = 34 RG32F ATLAS RE-PACKERS (Preserves Nano 0-edit rule)
// ============================================================================

function writeSheetAtlasSlice(
  sheetData: Float32Array,
  sheetIndex: number,
  slice: AtlasSlice,
): void {
  const base = sheetIndex * SHEET_INSTANCE_FLOATS;
  sheetData[base + 13] = slice.rect[0];
  sheetData[base + 14] = slice.rect[1];
  sheetData[base + 15] = slice.rect[2];
  sheetData[base + 16] = slice.rect[3];
  sheetData[base + 17] = slice.scale;
  sheetData[base + 18] = slice.mode;
}

function nextPowerOfTwo(v: number): number {
  let p = 64;
  while (p < v) p *= 2;
  return p;
}

function buildCollisionFreeEmbeddingAtlas(params: {
  readonly model: HybridModelParameters;
  readonly X: Float32Array;
  readonly tokenIds: readonly number[] | Int32Array;
  readonly T: number;
}): {
  readonly atlas: Rg32fValueAtlas;
  readonly wEmbed: AtlasSlice;
  readonly xEmbed: AtlasSlice;
  readonly tokenStrip: AtlasSlice;
} {
  const { model, X, tokenIds, T } = params;
  const d = model.config.d;
  const vocabSize = model.config.vocabSize;
  const yX = vocabSize + 2;
  const yTok = yX + d + 2;
  const atlasW = 32;
  const atlasH = nextPowerOfTwo(yTok + Math.max(T, 1) + 2);
  const writer = createRg32fAtlas(atlasW, atlasH);
  const { data, setTexel, makeRect } = writer;

  for (let v = 0; v < vocabSize; v++) {
    const rowOff = v * d;
    for (let ch = 0; ch < d; ch++) {
      setTexel(ch, v, model.W_embed[rowOff + ch]);
    }
  }
  for (let ch = 0; ch < d; ch++) {
    for (let t = 0; t < T; t++) {
      setTexel(t, yX + ch, X[ch * T + t]);
    }
  }
  for (let t = 0; t < T; t++) {
    const id = Math.max(0, Math.min(vocabSize - 1, Math.trunc(tokenIds[t] ?? 0)));
    setTexel(0, yTok + t, id);
  }

  return {
    atlas: { width: atlasW, height: atlasH, data },
    wEmbed: { rect: makeRect(0, 0, d, vocabSize), scale: 1 / computeMaxAbs(model.W_embed, 1.0), mode: 0 },
    xEmbed: { rect: makeRect(0, yX, T, d), scale: 1 / computeMaxAbs(X, 1.0), mode: 0 },
    tokenStrip: { rect: makeRect(0, yTok, 1, T), scale: 1 / Math.max(1, vocabSize - 1), mode: 1 },
  };
}

function buildCollisionFreeOutputAtlas(params: {
  readonly model: HybridModelParameters;
  readonly residualIn: Float32Array;
  readonly finalNorm: HybridFinalNormResult;
  readonly lmHead: HybridLmHeadResult;
  readonly T: number;
  readonly focusTok: number;
}): {
  readonly atlas: Rg32fValueAtlas;
  readonly slices: {
    readonly residualIn: AtlasSlice;
    readonly finalNorm: AtlasSlice;
    readonly rmsAgg: AtlasSlice;
    readonly gamma: AtlasSlice;
    readonly wLmHead: AtlasSlice;
    readonly logits: AtlasSlice;
    readonly softmaxAgg: AtlasSlice;
    readonly probs: AtlasSlice;
    readonly probSingleRow: (vocabId: number) => AtlasSlice;
  };
} {
  const { model, residualIn, finalNorm, lmHead, T, focusTok } = params;
  const d = model.config.d;
  const vocabSize = model.config.vocabSize;
  const ySeq = vocabSize + 2;
  const yGamma = ySeq + Math.max(T, 1) + 2;
  const atlasW = 32;
  const atlasH = nextPowerOfTwo(yGamma + 4);
  const writer = createRg32fAtlas(atlasW, atlasH);
  const { data, setTexel, makeRect } = writer;

  let maxLogit = -Infinity;
  let maxAbsLogit = 0.5;
  for (let v = 0; v < vocabSize; v++) {
    const wOff = v * d;
    for (let ch = 0; ch < d; ch++) {
      setTexel(ch, v, lmHead.W_lm_head[wOff + ch]);
    }
    const lv = lmHead.logits[v * T + focusTok];
    const pv = lmHead.probs[v * T + focusTok];
    if (lv > maxLogit) maxLogit = lv;
    if (Number.isFinite(lv) && Math.abs(lv) > maxAbsLogit) maxAbsLogit = Math.abs(lv);
    setTexel(8, v, lv);
    setTexel(9, v, pv);
  }
  let sumExp = 0;
  for (let v = 0; v < vocabSize; v++) {
    sumExp += Math.exp(lmHead.logits[v * T + focusTok] - maxLogit);
  }
  setTexel(10, 0, Math.fround(maxLogit));
  setTexel(10, 1, Math.fround(sumExp));

  for (let t = 0; t < T; t++) {
    for (let ch = 0; ch < d; ch++) {
      setTexel(ch, ySeq + t, residualIn[ch * T + t]);
      setTexel(8 + ch, ySeq + t, finalNorm.U_final[ch * T + t]);
    }
    setTexel(16, ySeq + t, finalNorm.rms[t]);
  }
  for (let ch = 0; ch < d; ch++) {
    setTexel(ch, yGamma, finalNorm.gamma[ch]);
  }

  return {
    atlas: { width: atlasW, height: atlasH, data },
    slices: {
      wLmHead: { rect: makeRect(0, 0, d, vocabSize), scale: 1 / computeMaxAbs(lmHead.W_lm_head, 0.5), mode: 0 },
      logits: { rect: makeRect(8, 0, 1, vocabSize), scale: 1 / Math.max(maxAbsLogit, 1.0), mode: 0 },
      probs: { rect: makeRect(9, 0, 1, vocabSize), scale: 1.0, mode: 1 },
      softmaxAgg: {
        rect: makeRect(10, 0, 1, 2),
        scale: 1 / Math.max(1.0, Math.abs(maxLogit), Math.abs(sumExp)),
        mode: 0,
      },
      residualIn: { rect: makeRect(0, ySeq, d, T), scale: 1 / computeMaxAbs(residualIn, 1.0), mode: 0 },
      finalNorm: { rect: makeRect(8, ySeq, d, T), scale: 1 / computeMaxAbs(finalNorm.U_final, 1.0), mode: 0 },
      rmsAgg: { rect: makeRect(16, ySeq, 1, T), scale: 1 / computeMaxAbs(finalNorm.rms, 1.0), mode: 0 },
      gamma: { rect: makeRect(0, yGamma, d, 1), scale: 1 / computeMaxAbs(finalNorm.gamma, 1.0), mode: 0 },
      probSingleRow: (vocabId: number): AtlasSlice => ({
        rect: makeRect(9, Math.max(0, Math.min(vocabSize - 1, vocabId)), 1, 1),
        scale: 1.0,
        mode: 1,
      }),
    },
  };
}

function transposeRowMajorToVocabBySeq(
  rowMajor: Float32Array,
  B: number,
  vocabSize: number,
): Float32Array {
  const out = new Float32Array(vocabSize * B);
  for (let j = 0; j < B; j++) {
    const rowOff = j * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      out[v * B + j] = rowMajor[rowOff + v];
    }
  }
  return out;
}

function concatSegmentXEmbed(
  segments: readonly { readonly tokens: number; readonly X: Float32Array }[],
  d: number,
): Float32Array {
  let totalT = 0;
  for (const seg of segments) totalT += seg.tokens;
  const out = new Float32Array(d * totalT);
  let colOffset = 0;
  for (const seg of segments) {
    for (let ch = 0; ch < d; ch++) {
      for (let t = 0; t < seg.tokens; t++) {
        out[ch * totalT + (colOffset + t)] = seg.X[ch * seg.tokens + t];
      }
    }
    colOffset += seg.tokens;
  }
  return out;
}

function computeTopKFromVector(
  logits: Float32Array,
  probs: Float32Array,
  vocab: readonly string[],
  k = 5,
): readonly HybridPrediction[] {
  const V = vocab.length;
  const order = Array.from({ length: V }, (_, v) => v);
  order.sort((a, b) => {
    const dLogit = logits[b] - logits[a];
    if (dLogit !== 0) return dLogit;
    return a - b;
  });
  const top: HybridPrediction[] = [];
  for (let i = 0; i < Math.min(k, V); i++) {
    const v = order[i];
    top.push(
      Object.freeze({
        tokenId: v,
        label: vocab[v] ?? `id:${v}`,
        logit: logits[v],
        prob: probs[v],
      }),
    );
  }
  return Object.freeze(top);
}

// ============================================================================
// 1. TWOTOWER EMBEDDING (NoPE) BOOKEND CUTAWAY
// ============================================================================

/**
 * Build the TwoTower Embedding (NoPE) 3D spatial cutaway directly from a `TwoTowerSession`,
 * reusing Nano's verified `buildEmbeddingCutaway` over the selected tower's independently owned
 * `W_embed_ctx` or `W_embed_den` (`V = 34`, `[MASK]` `#32` & `<eos>` `#33`), with collision-free
 * RG32F value atlas packing and full tower-aware provenance.
 */
export function buildTwoTowerEmbeddingScene(
  session: TwoTowerSession,
  options?: BuildTwoTowerEmbeddingOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const towerRole: 'context' | 'denoiser' =
    options?.towerRole ?? (view.denoiserLookup.pass !== null ? 'denoiser' : 'context');

  const towerModel: HybridModelParameters =
    towerRole === 'denoiser' ? view.model.denoiserModel : view.model.contextModel;
  const otherModel: HybridModelParameters =
    towerRole === 'denoiser' ? view.model.contextModel : view.model.denoiserModel;
  const weightTableName = towerRole === 'denoiser' ? 'W_embed_den' : 'W_embed_ctx';
  const d = towerModel.config.d;
  const vocabSize = towerModel.config.vocabSize;
  const maskTokenId = view.model.maskTokenId;
  const eosTokenId = view.model.eosTokenId;

  let tokenIds: Int32Array;
  let tokenLabels: readonly string[];
  let X_embed: Float32Array;
  let startTokenOffset: number;
  let executionId: string;
  let traceOrigin: TwoTowerTraceOrigin;
  let contextTraceStatus: 'available' | 'pruned_unavailable' | 'not_applicable' = 'not_applicable';
  const dl = view.denoiserLookup;
  const isHistoricalDenoiser = towerRole === 'denoiser' && dl.pass !== null && dl.isHistoricalInspection;
  const historicalTag =
    isHistoricalDenoiser && dl.pass !== null
      ? `Historical Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, ${dl.pass.passExecutionId}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`
      : null;

  const warningParts: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warningParts.push(`Context snapshot integrity error: ${view.snapshotIntegrity.error}`);
  }

  if (towerRole === 'denoiser') {
    const denPass = dl.pass;
    if (denPass !== null) {
      tokenIds = new Int32Array(denPass.tokenIds);
      tokenLabels = [...denPass.tokenLabels];
      X_embed = new Float32Array(denPass.residualStreams[0]);
      startTokenOffset = denPass.absoluteTokenStart;
      executionId = denPass.passExecutionId;
      traceOrigin = 'denoiser_scratch';
      if (isHistoricalDenoiser && dl.historicalProvenanceWarning !== null) {
        warningParts.push(dl.historicalProvenanceWarning);
      }
    } else if (view.activeBlock !== null) {
      tokenIds = Int32Array.from(view.activeBlock.currentTokens);
      tokenLabels = [...view.activeBlock.currentTokenLabels];
      X_embed = embedTokens(tokenIds, towerModel.W_embed, d);
      startTokenOffset = view.activeBlock.absoluteTokenStart;
      executionId = view.identity.recordId;
      traceOrigin = 'denoiser_scratch';
      warningParts.push(
        'Denoiser forward pass has not executed yet for this initialized block; previewing deterministic W_embed_den lookup on current block tokens.',
      );
    } else {
      // Fallback: display the latest committed token ID deterministically with clear disclosure
      const lastId = view.contextSnapshot.lastTokenId;
      const lastLabel = view.contextSnapshot.lastTokenLabel;
      tokenIds = Int32Array.from([lastId]);
      tokenLabels = [lastLabel];
      X_embed = embedTokens(tokenIds, towerModel.W_embed, d);
      startTokenOffset = Math.max(0, view.committedPrefixLen - 1);
      executionId = view.identity.recordId;
      traceOrigin = 'denoiser_scratch';
      warningParts.push(
        'No Denoiser forward pass has executed yet on this branch; previewing deterministic W_embed_den lookup on latest token.',
      );
    }
  } else {
    if (view.latestContextTrace.status === 'available' && view.latestContextTrace.execution) {
      contextTraceStatus = 'available';
      const exec = view.latestContextTrace.execution;
      if (options?.contextSegmentIndex !== undefined) {
        const segIdx = Math.max(
          0,
          Math.min(exec.segments.length - 1, Math.trunc(options.contextSegmentIndex)),
        );
        const seg = exec.segments[segIdx];
        const bay1 = seg.getLayerTrace(1);
        tokenIds = Int32Array.from(seg.tokenIds);
        tokenLabels = [...seg.tokenLabels];
        X_embed = new Float32Array(bay1.sim.X);
        startTokenOffset = seg.tokenRange[0];
        executionId = seg.stepExecutionId;
        traceOrigin = bay1.traceOrigin;
      } else if (exec.committedTokenIds.length <= 16) {
        // Combine split segments losslessly from actual executed Bay 01 X (= X_embed)
        const segXs: { readonly tokens: number; readonly X: Float32Array }[] = [];
        let anyReconstructed = false;
        for (const seg of exec.segments) {
          const bay1 = seg.getLayerTrace(1);
          if (bay1.traceOrigin === 'context_reconstructed') anyReconstructed = true;
          segXs.push({ tokens: seg.tokens, X: bay1.sim.X });
        }
        tokenIds = Int32Array.from(exec.committedTokenIds);
        tokenLabels = [...exec.committedTokenLabels];
        X_embed = concatSegmentXEmbed(segXs, d);
        startTokenOffset = exec.committedTokenRange[0];
        executionId = exec.segments[exec.segments.length - 1].stepExecutionId;
        traceOrigin = anyReconstructed ? 'context_reconstructed' : 'context_computed';
      } else {
        const tailSeg = exec.segments[exec.segments.length - 1];
        const bay1 = tailSeg.getLayerTrace(1);
        tokenIds = Int32Array.from(tailSeg.tokenIds);
        tokenLabels = [...tailSeg.tokenLabels];
        X_embed = new Float32Array(bay1.sim.X);
        startTokenOffset = tailSeg.tokenRange[0];
        executionId = tailSeg.stepExecutionId;
        traceOrigin = bay1.traceOrigin;
      }
    } else {
      contextTraceStatus = 'pruned_unavailable';
      const fullIds = view.contextSnapshot.ctxState.prefixTokenIds;
      const fullLabels = view.contextSnapshot.ctxState.prefixTokenLabels;
      const sliceStart = Math.max(0, fullIds.length - 16);
      tokenIds = Int32Array.from(fullIds.slice(sliceStart));
      tokenLabels = fullLabels.slice(sliceStart);
      X_embed = embedTokens(tokenIds, towerModel.W_embed, d);
      startTokenOffset = sliceStart;
      executionId = view.identity.sourceCacheDigest;
      traceOrigin = 'persistent_cache';
      warningParts.push(
        'Historical Context execution trace is unavailable after history pruning; displaying deterministic W_embed_ctx lookup from authoritative persistent Context prefix tokens.',
      );
    }
  }
  const warningBanner: string | null = warningParts.length > 0 ? warningParts.join(' | ') : null;

  const T = tokenIds.length;
  const focusToken = Math.max(0, Math.min(T - 1, Math.trunc(options?.focusToken ?? T - 1)));

  // Directly reuse Nano's verified 3D Embedding Cutaway builder!
  const nanoScene = buildEmbeddingCutaway({
    model: towerModel,
    embedding: { X: X_embed, d },
    tokenIds,
    tokenLabels,
    startTokenOffset,
    focusToken,
    precision: 'bf16',
    selectedPart: options?.selectedPart ?? null,
    hoveredPart: options?.hoveredPart ?? null,
    activeCell: options?.activeCell ?? null,
  });

  // Re-pack collision-free RG32F value atlas for V = 34 (so rows #32 [MASK] & #33 <eos> never collide!)
  const repacked = buildCollisionFreeEmbeddingAtlas({
    model: towerModel,
    X: X_embed,
    tokenIds,
    T,
  });
  const sheetData = nanoScene.sheetData ? new Float32Array(nanoScene.sheetData) : undefined;
  if (sheetData) {
    for (let i = 0; i < nanoScene.solidPickBoxes.length; i++) {
      const partId = nanoScene.solidPickBoxes[i].partId;
      if (partId === 'token-strip') writeSheetAtlasSlice(sheetData, i, repacked.tokenStrip);
      else if (partId === 'w-embed') writeSheetAtlasSlice(sheetData, i, repacked.wEmbed);
      else if (partId === 'x-embed' || partId === 'residual-handoff') {
        writeSheetAtlasSlice(sheetData, i, repacked.xEmbed);
      }
    }
  }

  // Locate [MASK] (#32) and <eos> (#33) tokens at their exact local & absolute positions
  const maskTokenLocalIndices: number[] = [];
  const maskTokenAbsolutePositions: number[] = [];
  const eosTokenLocalIndices: number[] = [];
  const eosTokenAbsolutePositions: number[] = [];
  for (let i = 0; i < T; i++) {
    if (tokenIds[i] === maskTokenId) {
      maskTokenLocalIndices.push(i);
      maskTokenAbsolutePositions.push(startTokenOffset + i);
    }
    if (tokenIds[i] === eosTokenId) {
      eosTokenLocalIndices.push(i);
      eosTokenAbsolutePositions.push(startTokenOffset + i);
    }
  }

  // Propagate full tower-aware PickBox provenance
  const solidPickBoxes: PickBox[] = nanoScene.solidPickBoxes.map((box) => ({
    ...box,
    bayId: 0,
    towerRole,
    executionId,
    traceOrigin,
    semanticTag: 'embedding',
    absoluteTokenOffset: startTokenOffset,
  }));

  const towerLabel = towerRole === 'denoiser' ? 'Denoiser Tower' : 'Context Tower';
  const absRange: readonly [number, number] = Object.freeze([
    startTokenOffset,
    startTokenOffset + T - 1,
  ] as const);

  // Augment partCatalog & landmarks with TwoTower independent weight identity & [MASK]/<eos> highlights
  const partCatalog: Record<string, CutawayPartInfo> = {
    ...nanoScene.partCatalog,
    'w-embed': {
      ...nanoScene.partCatalog['w-embed'],
      title: `${towerLabel} Untied Embedding Table (${weightTableName})`,
      badge: `${weightTableName} · fp ${towerModel.parameterFingerprint.slice(0, 8)} · V=${vocabSize}`,
      summary:
        `Independently owned ${towerLabel} token embedding matrix (${weightTableName} ≠ ${towerRole === 'denoiser' ? 'W_embed_ctx' : 'W_embed_den'} and strictly untied from W_lm_head). Includes miniature [MASK] #${maskTokenId} and <eos> #${eosTokenId}.`,
      specs: [
        ...nanoScene.partCatalog['w-embed'].specs,
        `Tower parameter fingerprint: ${towerModel.parameterFingerprint} (independent of other tower ${otherModel.parameterFingerprint})`,
        TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
      ],
    },
    'token-strip': {
      ...nanoScene.partCatalog['token-strip'],
      badge: `pos [${absRange[0]}..${absRange[1]}] · ${maskTokenLocalIndices.length} [MASK] · ${eosTokenLocalIndices.length} <eos>`,
      specs: [
        ...nanoScene.partCatalog['token-strip'].specs,
        `Absolute token positions: [${absRange[0]}..${absRange[1]}] (startTokenOffset = ${startTokenOffset})`,
        ...(historicalTag ? [`Historical Denoiser provenance: ${historicalTag}`] : []),
        maskTokenAbsolutePositions.length > 0
          ? `[MASK] (#${maskTokenId}) active at absolute positions: ${maskTokenAbsolutePositions.map((p) => `t${p}`).join(', ')}`
          : `No [MASK] (#${maskTokenId}) tokens in this slice`,
        eosTokenAbsolutePositions.length > 0
          ? `<eos> (#${eosTokenId}) present at absolute positions: ${eosTokenAbsolutePositions.map((p) => `t${p}`).join(', ')}`
          : `No <eos> (#${eosTokenId}) tokens in this slice`,
      ],
    },
  };

  const landmarks: LandmarkAnchor[] = [
    ...nanoScene.landmarks.map((lm) =>
      lm.id === 'emb-w'
        ? {
            ...lm,
            label: `${weightTableName} [${vocabSize}×${d}] (${towerLabel})`,
            sublabel: `fp ${towerModel.parameterFingerprint.slice(0, 8)} · Independent of other tower & W_lm_head`,
          }
        : lm,
    ),
    ...(maskTokenAbsolutePositions.length > 0 || eosTokenAbsolutePositions.length > 0
      ? [
          {
            id: 'twotower-emb-special-tokens',
            label:
              [
                maskTokenAbsolutePositions.length > 0
                  ? `[MASK] #${maskTokenId} @ ${maskTokenAbsolutePositions.map((p) => `t${p}`).join(',')}`
                  : null,
                eosTokenAbsolutePositions.length > 0
                  ? `<eos> #${eosTokenId} @ ${eosTokenAbsolutePositions.map((p) => `t${p}`).join(',')}`
                  : null,
              ]
                .filter(Boolean)
                .join(' · '),
            sublabel: `${towerLabel} active special tokens (V=${vocabSize} synthetic miniature)`,
            world: [-6.05, 0.54, -1.65] as readonly [number, number, number],
            partId: 'token-strip',
            align: 'top' as const,
            tier: 'primary' as const,
            tone: 'gold' as const,
          },
        ]
      : []),
  ];

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    { label: `${towerRole === 'denoiser' ? 'DENOISER' : 'CONTEXT'} · ${weightTableName}`, tone: towerRole, origin: traceOrigin },
    { label: `fp ${towerModel.parameterFingerprint.slice(0, 8)} (Independent)`, tone: 'shared' },
    { label: `pos [${absRange[0]}..${absRange[1]}] · V=${vocabSize} NoPE`, tone: 'shared' },
    ...(maskTokenLocalIndices.length > 0
      ? [{ label: `${maskTokenLocalIndices.length}× [MASK] (#${maskTokenId})`, tone: 'gold' as const }]
      : []),
    ...(eosTokenLocalIndices.length > 0
      ? [{ label: `${eosTokenLocalIndices.length}× <eos> (#${eosTokenId})`, tone: 'warn' as const }]
      : []),
    ...(isHistoricalDenoiser
      ? [
          {
            label: `Historical Denoiser Pass · Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`,
            tone: 'warn' as const,
            origin: 'denoiser_scratch' as const,
          },
        ]
      : []),
  ];

  const activeSelection = buildTwoTowerSelectionIdentity(view, {
    towerRole,
    bayId: 0,
    mixerKind: 'embed',
    tensorRole: options?.activeCell?.tensor ?? 'w_embed',
    traceOrigin,
    executionId,
    absoluteTokenPosition: startTokenOffset + focusToken,
    absoluteTokenRange: absRange,
    provenanceNote:
      `${towerLabel} ${weightTableName} (fp=${towerModel.parameterFingerprint}) · pos [${absRange[0]}..${absRange[1]}]` +
      (historicalTag ? ` · ${historicalTag}` : ''),
  });

  const diagnostics: TwoTowerEmbeddingDiagnostics = Object.freeze({
    towerRole,
    weightTableName,
    towerModelId: towerModel.modelId,
    towerParameterFingerprint: towerModel.parameterFingerprint,
    otherTowerParameterFingerprint: otherModel.parameterFingerprint,
    weightsIndependentOfOtherTower: true as const,
    untiedFromLmHead: true as const,
    positionalEncoding: 'NoPE' as const,
    vocabSize,
    maskTokenId,
    eosTokenId,
    tokenCount: T,
    startTokenOffset,
    absoluteTokenRange: absRange,
    tokenIds: Object.freeze(Array.from(tokenIds)),
    tokenLabels: Object.freeze([...tokenLabels]),
    maskTokenLocalIndices: Object.freeze(maskTokenLocalIndices),
    maskTokenAbsolutePositions: Object.freeze(maskTokenAbsolutePositions),
    eosTokenLocalIndices: Object.freeze(eosTokenLocalIndices),
    eosTokenAbsolutePositions: Object.freeze(eosTokenAbsolutePositions),
    hasMaskTokens: maskTokenLocalIndices.length > 0,
    hasEosTokens: eosTokenLocalIndices.length > 0,
    contextTraceStatus,
    isHistoricalDenoiserInspection: isHistoricalDenoiser,
    matchesCurrentContextBoundary: towerRole === 'denoiser' ? dl.matchesCurrentContextBoundary : true,
    denoiserSourceRecordIndex: towerRole === 'denoiser' ? dl.sourceRecordIndex : null,
    denoiserSourceBlockId: towerRole === 'denoiser' ? dl.sourceBlockId : null,
    denoiserSourceRefinementIndex: towerRole === 'denoiser' ? dl.sourceRefinementIndex : null,
    denoiserSourceContextPrefixLen: towerRole === 'denoiser' ? dl.sourceContextPrefixLen : null,
    denoiserSourceContextCacheDigest: towerRole === 'denoiser' ? dl.sourceContextCacheDigest : null,
    historicalProvenanceWarning: isHistoricalDenoiser ? dl.historicalProvenanceWarning : null,
    syntheticMiniatureDisclosure: TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
  });

  const atlasKey =
    `tt-emb|${towerModel.parameterFingerprint}|${towerRole}|${executionId}|` +
    `off${startTokenOffset}|toks${Array.from(tokenIds).join(',')}`;

  return {
    viewMode: 'embedding',
    title: `TwoTower Embedding (NoPE) · ${towerLabel} (${weightTableName})`,
    subtitle:
      (historicalTag ? `${historicalTag} · ` : '') +
      `V=${vocabSize} (incl. [MASK] #${maskTokenId}, <eos> #${eosTokenId}) · pos [${absRange[0]}..${absRange[1]}] · Independent Weights`,
    bayId: 0,
    layerIndex: null,
    activeTowerFocus: towerRole === 'denoiser' ? 'denoiser_solo' : 'context_solo',
    solidData: nanoScene.solidData,
    solidCount: nanoScene.solidCount,
    lineData: nanoScene.lineData,
    lineCount: nanoScene.lineCount,
    ribbonData: nanoScene.ribbonData,
    ribbonCount: nanoScene.ribbonCount,
    sheetData,
    sheetCount: nanoScene.sheetCount,
    sheetBasisData: nanoScene.sheetBasisData,
    flowVertices: nanoScene.flowVertices,
    flowIndices: nanoScene.flowIndices,
    valueAtlas: repacked.atlas,
    atlasKey,
    fitBounds: nanoScene.fitBounds,
    opFocusBounds: nanoScene.opFocusBounds,
    landmarks,
    kvDepthTicks: [],
    calipers: nanoScene.calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeOpPartIds: nanoScene.activeOpPartIds,
    activeCellGrids: nanoScene.activeCellGrids,
    cellPopover: nanoScene.cellPopover,
    activeFormula: nanoScene.activeFormula,
    focusedAccumulation: null,
    activeSelection,
    stepperSteps: EMBEDDING_STEPPER_STEPS,
    provenanceBadges,
    epistemicNote: TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}

// ============================================================================
// 2. TWOTOWER OUTPUT & UNTIED LM HEAD BOOKEND CUTAWAY
// ============================================================================

function resolveDenoiserSamplerPresentation(
  view: TwoTowerSessionViewState,
): {
  readonly presentation: SamplerStepPresentation | null;
  readonly source: 'recorded_commit' | 'live_refinement_preview' | 'none';
} {
  const rec = view.record;

  // 1. Exact recorded presentation on a confidence-commit transition
  if (rec.kind === 'diffusion_confidence_commit') {
    assertPresentationSafeNoInfinity(rec.samplerPresentation, 'recordedSamplerPresentation');
    return { presentation: rec.samplerPresentation, source: 'recorded_commit' };
  }

  // 2. Active diffusion_refinement transition: derive deterministic read-only preview from the
  //    actual Denoiser pass logits/probs (NEVER mutates session, NEVER uses Math.random()!)
  if (rec.kind === 'diffusion_refinement') {
    const pass = rec.denoiserPass;
    const cfg = view.historicalConfig;
    const B = pass.blockSize;
    const V = view.model.denoiserModel.config.vocabSize;
    let uniformVariates: Float32Array | undefined;
    if (cfg.temperature > 0) {
      uniformVariates = deriveRefinementUniformVariates({
        samplingSeed: cfg.samplingSeed,
        sessionId: view.sessionId,
        branchId: view.branchId,
        blockId: rec.blockConfig.blockId,
        refinementIndex: rec.refinementIndex,
        blockSize: B,
        vocabSize: V,
      }).variates;
    }
    const unmaskStep = runConfidenceUnmaskStep({
      xt: rec.xtInput,
      logits: pass.logits,
      vocabSize: V,
      maskTokenId: view.model.maskTokenId,
      stepIndex: rec.refinementIndex,
      stepsPerBlock: rec.blockConfig.stepsPerBlock,
      confidenceThreshold: cfg.confidenceThreshold,
      temperature: cfg.temperature,
      uniformVariates,
    });
    const presentation = buildSamplerStepPresentation({
      rawLogits: pass.logits,
      rawProbs: pass.probs,
      unmaskStep,
      vocab: view.model.vocab,
      maskTokenId: view.model.maskTokenId,
      confidenceThreshold: cfg.confidenceThreshold,
      temperature: cfg.temperature,
      absoluteTokenStart: pass.absoluteTokenStart,
      topK: 5,
    });
    assertPresentationSafeNoInfinity(presentation, 'liveRefinementSamplerPresentation');
    return { presentation, source: 'live_refinement_preview' };
  }

  // 3. Fallback to latest recorded sampler presentation on active block (if mask_diffusion pass)
  if (
    view.denoiserLookup.pass?.mode === 'mask_diffusion' &&
    view.latestSamplerPresentation !== null
  ) {
    assertPresentationSafeNoInfinity(view.latestSamplerPresentation, 'latestSamplerPresentation');
    return { presentation: view.latestSamplerPresentation, source: 'recorded_commit' };
  }

  return { presentation: null, source: 'none' };
}

function buildUnavailableOutputScene(params: {
  readonly view: TwoTowerSessionViewState;
  readonly towerRole: 'context' | 'denoiser';
  readonly contextTraceStatus: 'pruned_unavailable' | 'not_applicable';
  readonly warningBanner: string;
}): TwoTowerSpatialScene {
  const { view, towerRole, contextTraceStatus, warningBanner } = params;
  const towerModel = towerRole === 'denoiser' ? view.model.denoiserModel : view.model.contextModel;
  const otherModel = towerRole === 'denoiser' ? view.model.contextModel : view.model.denoiserModel;
  const weightTableName = towerRole === 'denoiser' ? 'W_lm_head_den' : 'W_lm_head_ctx';
  const top5AtFocus =
    towerRole === 'context'
      ? computeTopKFromVector(
          view.contextSnapshot.arLogits,
          view.contextSnapshot.arProbs,
          towerModel.vocab,
          5,
        )
      : Object.freeze([]);
  const absPos = Math.max(0, view.committedPrefixLen - 1);

  const activeSelection = buildTwoTowerSelectionIdentity(view, {
    towerRole,
    bayId: 53,
    mixerKind: 'output',
    tensorRole: 'output_unavailable',
    traceOrigin: towerRole === 'denoiser' ? 'denoiser_scratch' : 'persistent_cache',
    executionId: view.identity.recordId,
    absoluteTokenPosition: absPos,
    provenanceNote: warningBanner,
  });

  const diagnostics: TwoTowerOutputDiagnostics = Object.freeze({
    towerRole,
    weightTableName,
    towerModelId: towerModel.modelId,
    towerParameterFingerprint: towerModel.parameterFingerprint,
    otherTowerParameterFingerprint: otherModel.parameterFingerprint,
    weightsIndependentOfOtherTower: true as const,
    untiedFromEmbedding: true as const,
    finalNormModulated: false as const,
    finalNormNote: 'Final RMSNorm (norm_f, ε=1e-5) is strictly UNMODULATED in both Context and Denoiser towers.',
    predictionSemantics:
      towerRole === 'context' ? 'causal_next_token_shifted' : 'unshifted_block_diffusion',
    predictionSemanticsLabel:
      towerRole === 'context'
        ? 'Context AR Next-Token Prediction (shifted causal t → next)'
        : 'Denoiser Unshifted Block Logits (pos P+j → clean x₀[P+j])',
    vocabSize: towerModel.config.vocabSize,
    maskTokenId: view.model.maskTokenId,
    eosTokenId: view.model.eosTokenId,
    tokenCount: 0,
    startTokenOffset: absPos,
    absoluteTokenRange: Object.freeze([absPos, absPos] as const),
    focusToken: 0,
    absoluteFocusPosition: absPos,
    contextTraceStatus,
    isHistoricalDenoiserInspection: false,
    matchesCurrentContextBoundary: false,
    denoiserSourceRecordIndex: null,
    denoiserSourceBlockId: null,
    denoiserSourceRefinementIndex: null,
    denoiserSourceContextPrefixLen: null,
    denoiserSourceContextCacheDigest: null,
    historicalProvenanceWarning: null,
    samplerPresentationSource: 'none' as const,
    samplerPresentation: null,
    top5AtFocus,
    syntheticMiniatureDisclosure: TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
  });

  return {
    viewMode: 'output',
    title: `TwoTower Output & Untied LM Head · ${towerRole === 'denoiser' ? 'Denoiser Tower' : 'Context Tower'} (${weightTableName})`,
    subtitle: 'Bay 52 Residual Trace Unavailable · Final RMSNorm Strictly Unmodulated',
    bayId: 53,
    layerIndex: null,
    activeTowerFocus: towerRole === 'denoiser' ? 'denoiser_solo' : 'context_solo',
    solidData: new Float32Array(0),
    solidCount: 0,
    lineData: new Float32Array(0),
    lineCount: 0,
    ribbonData: new Float32Array(0),
    ribbonCount: 0,
    fitBounds: OUTPUT_SPATIAL_FIT_BOUNDS,
    landmarks: [
      {
        id: 'output-unavailable-notice',
        label: warningBanner,
        world: [0, 1.2, 0],
        partId: 'output-unavailable',
        align: 'top',
        tier: 'primary',
        tone: 'muted',
      },
    ],
    kvDepthTicks: [],
    solidPickBoxes: [],
    fasciaPickBoxes: [],
    partCatalog: {
      'output-unavailable': {
        id: 'output-unavailable',
        title: `${weightTableName} Trace Unavailable`,
        badge: 'No Synthetic Residual Substituted',
        summary: warningBanner,
        specs: [
          'Final RMSNorm (norm_f) is strictly UNMODULATED in both towers.',
          TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
        ],
      },
    },
    activeSelection,
    stepperSteps: OUTPUT_STEPPER_STEPS,
    provenanceBadges: [
      { label: `${towerRole === 'denoiser' ? 'DENOISER' : 'CONTEXT'} · ${weightTableName}`, tone: towerRole },
      { label: 'Final RMSNorm: Strictly Unmodulated', tone: 'shared' },
      { label: 'Trace Unavailable', tone: 'warn' },
    ],
    epistemicNote: TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}

/**
 * Build the TwoTower Output & Untied LM Head 3D spatial cutaway directly from a `TwoTowerSession`,
 * reusing Nano's verified `buildOutputCutaway` over the selected tower's independently owned
 * unmodulated `norm_f` and untied `W_lm_head_ctx` / `W_lm_head_den` (`V = 34`), cleanly
 * distinguishing Context AR next-token prediction vs unshifted Denoiser block logits and attaching
 * the presentation-safe `SamplerStepPresentation` (strictly ZERO `Infinity`).
 */
export function buildTwoTowerOutputScene(
  session: TwoTowerSession,
  options?: BuildTwoTowerOutputOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const towerRole: 'context' | 'denoiser' =
    options?.towerRole ?? (view.denoiserLookup.pass !== null ? 'denoiser' : 'context');

  const towerModel: HybridModelParameters =
    towerRole === 'denoiser' ? view.model.denoiserModel : view.model.contextModel;
  const otherModel: HybridModelParameters =
    towerRole === 'denoiser' ? view.model.contextModel : view.model.denoiserModel;
  const weightTableName = towerRole === 'denoiser' ? 'W_lm_head_den' : 'W_lm_head_ctx';
  const d = towerModel.config.d;
  const vocabSize = towerModel.config.vocabSize;
  const maskTokenId = view.model.maskTokenId;
  const eosTokenId = view.model.eosTokenId;

  let residualIn: Float32Array;
  let finalNorm: HybridFinalNormResult;
  let lmHead: HybridLmHeadResult;
  let tokenLabels: readonly string[];
  let startTokenOffset: number;
  let isArDecode: boolean;
  let predictionSemantics: TwoTowerOutputPredictionSemantics;
  let executionId: string;
  let traceOrigin: TwoTowerTraceOrigin;
  let contextTraceStatus: 'available' | 'pruned_unavailable' | 'not_applicable' = 'not_applicable';
  let samplerPresentation: SamplerStepPresentation | null = null;
  let samplerPresentationSource: 'recorded_commit' | 'live_refinement_preview' | 'none' = 'none';

  if (towerRole === 'denoiser') {
    const denPass = view.denoiserLookup.pass;
    if (!denPass) {
      return buildUnavailableOutputScene({
        view,
        towerRole: 'denoiser',
        contextTraceStatus: 'not_applicable',
        warningBanner:
          'No Denoiser forward pass has executed yet on this branch; switch to Context Tower or run a Denoiser step.',
      });
    }
    const B = denPass.blockSize;
    residualIn = new Float32Array(denPass.residualStreams[denPass.residualStreams.length - 1]);
    finalNorm = denPass.finalNorm;
    // Lossless transpose from Denoiser row-major [B, V] to column-major [V, B] for buildOutputCutaway
    const colMajorLogits = transposeRowMajorToVocabBySeq(denPass.logits, B, vocabSize);
    const colMajorProbs = transposeRowMajorToVocabBySeq(denPass.probs, B, vocabSize);
    lmHead = {
      W_lm_head: towerModel.W_lm_head,
      logits: colMajorLogits,
      probs: colMajorProbs,
      topKByToken: denPass.topKByPosition,
    };
    tokenLabels = [...denPass.tokenLabels];
    startTokenOffset = denPass.absoluteTokenStart;
    isArDecode = denPass.mode === 'mock_ar';
    predictionSemantics =
      denPass.mode === 'mask_diffusion'
        ? 'unshifted_block_diffusion'
        : 'causal_next_token_shifted';
    executionId = denPass.passExecutionId;
    traceOrigin = 'denoiser_scratch';

    if (denPass.mode === 'mask_diffusion') {
      const resolvedSampler = resolveDenoiserSamplerPresentation(view);
      samplerPresentation = resolvedSampler.presentation;
      samplerPresentationSource = resolvedSampler.source;
    }
  } else {
    const bay52Lookup = view.resolveContextBayTrace(52);
    if (bay52Lookup.status !== 'available' || !bay52Lookup.trace || !bay52Lookup.segment) {
      return buildUnavailableOutputScene({
        view,
        towerRole: 'context',
        contextTraceStatus: 'pruned_unavailable',
        warningBanner:
          bay52Lookup.reason ||
          'Historical Context Bay 52 residual trace is unavailable after pruning; persistent snapshot arLogits/arProbs remain authoritative (no synthetic hidden state substituted).',
      });
    }
    contextTraceStatus = 'available';
    const seg = bay52Lookup.segment;
    const S = seg.tokens;
    residualIn = new Float32Array(bay52Lookup.trace.sim.Y);
    const normRes = applyRmsNorm(
      residualIn,
      towerModel.finalNormGamma,
      d,
      S,
      towerModel.config.rmsNormEps,
    );
    finalNorm = Object.freeze({
      eps: towerModel.config.rmsNormEps,
      gamma: towerModel.finalNormGamma,
      rms: normRes.rms,
      X_norm: normRes.X_norm,
      U_final: normRes.U,
    });
    lmHead = runUntiedLmHead(finalNorm.U_final, towerModel.W_lm_head, towerModel.vocab, d, S, 5);
    tokenLabels = [...seg.tokenLabels];
    startTokenOffset = seg.tokenRange[0];
    isArDecode = true;
    predictionSemantics = 'causal_next_token_shifted';
    executionId = seg.stepExecutionId;
    traceOrigin = bay52Lookup.traceOrigin;
  }

  const T = tokenLabels.length;
  const focusToken = Math.max(0, Math.min(T - 1, Math.trunc(options?.focusToken ?? T - 1)));
  const absoluteFocusPosition = startTokenOffset + focusToken;

  // Directly reuse Nano's verified 3D Output & Untied LM Head builder!
  const nanoScene = buildOutputCutaway({
    model: towerModel,
    residualIn,
    finalNorm,
    lmHead,
    tokenLabels,
    startTokenOffset,
    focusToken,
    isArDecode,
    precision: 'bf16',
    selectedPart: options?.selectedPart ?? null,
    hoveredPart: options?.hoveredPart ?? null,
    activeCell: options?.activeCell ?? null,
    selectedCandidateTokenId: options?.selectedCandidateTokenId ?? null,
  });

  // Re-pack collision-free RG32F value atlas for V = 34 (so rows #32 [MASK] & #33 <eos> never collide!)
  const repacked = buildCollisionFreeOutputAtlas({
    model: towerModel,
    residualIn,
    finalNorm,
    lmHead,
    T,
    focusTok: focusToken,
  });
  const sheetData = nanoScene.sheetData ? new Float32Array(nanoScene.sheetData) : undefined;
  if (sheetData) {
    for (let i = 0; i < nanoScene.solidPickBoxes.length; i++) {
      const partId = nanoScene.solidPickBoxes[i].partId;
      if (partId === 'residual-in') writeSheetAtlasSlice(sheetData, i, repacked.slices.residualIn);
      else if (partId === 'rms-agg') writeSheetAtlasSlice(sheetData, i, repacked.slices.rmsAgg);
      else if (partId === 'final-gamma') writeSheetAtlasSlice(sheetData, i, repacked.slices.gamma);
      else if (partId === 'final-norm') writeSheetAtlasSlice(sheetData, i, repacked.slices.finalNorm);
      else if (partId === 'w-lm-head') writeSheetAtlasSlice(sheetData, i, repacked.slices.wLmHead);
      else if (partId === 'logits') writeSheetAtlasSlice(sheetData, i, repacked.slices.logits);
      else if (partId === 'softmax-agg') writeSheetAtlasSlice(sheetData, i, repacked.slices.softmaxAgg);
      else if (partId === 'probs') writeSheetAtlasSlice(sheetData, i, repacked.slices.probs);
      else {
        const candId = parseOutputCandidatePartId(partId);
        if (candId !== null) {
          writeSheetAtlasSlice(sheetData, i, repacked.slices.probSingleRow(candId));
        }
      }
    }
  }

  // Propagate full tower-aware PickBox provenance
  const solidPickBoxes: PickBox[] = nanoScene.solidPickBoxes.map((box) => ({
    ...box,
    bayId: 53,
    towerRole,
    executionId,
    traceOrigin,
    semanticTag: 'output',
    absoluteTokenOffset: startTokenOffset,
  }));

  const dlOut = view.denoiserLookup;
  const isHistoricalDenoiserOut = towerRole === 'denoiser' && dlOut.isHistoricalInspection;
  const historicalTagOut =
    isHistoricalDenoiserOut && dlOut.pass !== null
      ? `Historical Rec #${dlOut.sourceRecordIndex} (b${dlOut.sourceBlockId ?? '-'} r${dlOut.sourceRefinementIndex ?? '-'}, ${dlOut.pass.passExecutionId}, P_cond=${dlOut.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`
      : null;

  const towerLabel = towerRole === 'denoiser' ? 'Denoiser Tower' : 'Context Tower';
  const absRange: readonly [number, number] = Object.freeze([
    startTokenOffset,
    startTokenOffset + T - 1,
  ] as const);
  const predictionSemanticsLabel =
    predictionSemantics === 'unshifted_block_diffusion'
      ? `Denoiser Unshifted Block Logits (pos t${absoluteFocusPosition} → directly predicts clean x₀[t${absoluteFocusPosition}])`
      : `Context AR Next-Token Prediction (shifted causal t${absoluteFocusPosition} → predicts t${absoluteFocusPosition + 1})`;

  const finalNormNote =
    'Final RMSNorm (norm_f, ε = 1e-5) is strictly UNMODULATED in BOTH Context and Denoiser towers (zero AdaLN shift, scale, or gate).';

  const partCatalog: Record<string, CutawayPartInfo> = {
    ...nanoScene.partCatalog,
    'final-norm': {
      ...nanoScene.partCatalog['final-norm'],
      badge: 'Strictly UNMODULATED norm_f · ε = 1e−5',
      summary: `${nanoScene.partCatalog['final-norm'].summary} ${finalNormNote}`,
      specs: [
        ...nanoScene.partCatalog['final-norm'].specs,
        'finalNormModulated = false (verified across both Context & Denoiser towers)',
      ],
    },
    'w-lm-head': {
      ...nanoScene.partCatalog['w-lm-head'],
      title: `${towerLabel} Untied LM Head (${weightTableName})`,
      badge: `${weightTableName} · fp ${towerModel.parameterFingerprint.slice(0, 8)} · V=${vocabSize}`,
      summary:
        `Independently owned ${towerLabel} untied output head (${weightTableName} ≠ ${towerRole === 'denoiser' ? 'W_lm_head_ctx' : 'W_lm_head_den'} and ≠ W_embed).`,
      specs: [
        ...nanoScene.partCatalog['w-lm-head'].specs,
        `Tower parameter fingerprint: ${towerModel.parameterFingerprint} (independent of other tower ${otherModel.parameterFingerprint})`,
        TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
      ],
    },
    logits: {
      ...nanoScene.partCatalog.logits,
      badge:
        predictionSemantics === 'unshifted_block_diffusion'
          ? `Unshifted Block x₀[t${absoluteFocusPosition}] · V=${vocabSize}`
          : `Shifted Causal t${absoluteFocusPosition} → next · V=${vocabSize}`,
      specs: [
        ...nanoScene.partCatalog.logits.specs,
        predictionSemanticsLabel,
        ...(historicalTagOut ? [`Historical Denoiser provenance: ${historicalTagOut}`] : []),
        TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
      ],
    },
    probs: {
      ...nanoScene.partCatalog.probs,
      title:
        predictionSemantics === 'unshifted_block_diffusion'
          ? `Raw Unconstrained LM-Head Probs [${vocabSize} × 1] (Before MDLM [MASK]=0)`
          : `Context AR Next-Token Probabilities [${vocabSize} × 1]`,
      specs: [
        ...nanoScene.partCatalog.probs.specs,
        predictionSemantics === 'unshifted_block_diffusion'
          ? `MDLM sampler constrains [MASK] (#${maskTokenId}) to exact 0.0000 and displays locked prior slots as "Locked (prior step)" (never Infinity).`
          : 'Causal shifted AR distribution predicting next token t + 1.',
        TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
      ],
    },
  };

  const landmarks: LandmarkAnchor[] = nanoScene.landmarks.map((lm) => {
    if (lm.id === 'out-final-norm') {
      return {
        ...lm,
        label: 'Final RMSNorm norm_f (Strictly UNMODULATED)',
        sublabel: 'Zero shift/scale/gate in Context & Denoiser',
      };
    }
    if (lm.id === 'out-w-lm') {
      return {
        ...lm,
        label: `${weightTableName} [${vocabSize}×${d}] (${towerLabel})`,
        sublabel: `fp ${towerModel.parameterFingerprint.slice(0, 8)} · Independent & Untied`,
      };
    }
    if (lm.id === 'out-top5-header') {
      return {
        ...lm,
        label:
          predictionSemantics === 'unshifted_block_diffusion'
            ? `Unshifted Block x₀[t${absoluteFocusPosition}] Raw Top-5`
            : `Context AR Next-Token (t${absoluteFocusPosition} → t${absoluteFocusPosition + 1}) Top-5`,
        sublabel:
          predictionSemantics === 'unshifted_block_diffusion'
            ? `Zero shift-by-1 · MDLM sets [MASK] #${maskTokenId} = 0.0000`
            : `Shifted causal AR · V=${vocabSize} synthetic mini`,
      };
    }
    return lm;
  });

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    { label: `${towerRole === 'denoiser' ? 'DENOISER' : 'CONTEXT'} · ${weightTableName}`, tone: towerRole, origin: traceOrigin },
    { label: `fp ${towerModel.parameterFingerprint.slice(0, 8)} (Independent)`, tone: 'shared' },
    { label: 'Final RMSNorm norm_f: Unmodulated', tone: 'shared' },
    {
      label:
        predictionSemantics === 'unshifted_block_diffusion'
          ? 'Unshifted Block Logits (P+j → x₀[P+j])'
          : 'Shifted Causal AR (t → next)',
      tone: predictionSemantics === 'unshifted_block_diffusion' ? 'gold' : 'context',
    },
    ...(samplerPresentation !== null
      ? [{ label: 'MDLM Sampler: [MASK]=0.0000 · Zero Infinity', tone: 'gold' as const }]
      : []),
    ...(isHistoricalDenoiserOut
      ? [
          {
            label: `Historical Denoiser Pass · Rec #${dlOut.sourceRecordIndex} (b${dlOut.sourceBlockId ?? '-'} r${dlOut.sourceRefinementIndex ?? '-'}, P_cond=${dlOut.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`,
            tone: 'warn' as const,
            origin: 'denoiser_scratch' as const,
          },
        ]
      : []),
  ];

  const activeSelection = buildTwoTowerSelectionIdentity(view, {
    towerRole,
    bayId: 53,
    mixerKind: 'output',
    tensorRole: options?.activeCell?.tensor ?? 'probs',
    traceOrigin,
    executionId,
    absoluteTokenPosition: absoluteFocusPosition,
    absoluteTokenRange: absRange,
    provenanceNote:
      `${towerLabel} ${weightTableName} (fp=${towerModel.parameterFingerprint}) · ${predictionSemanticsLabel}` +
      (historicalTagOut ? ` · ${historicalTagOut}` : ''),
  });

  const top5AtFocus = lmHead.topKByToken[focusToken] ?? Object.freeze([]);
  const diagnostics: TwoTowerOutputDiagnostics = Object.freeze({
    towerRole,
    weightTableName,
    towerModelId: towerModel.modelId,
    towerParameterFingerprint: towerModel.parameterFingerprint,
    otherTowerParameterFingerprint: otherModel.parameterFingerprint,
    weightsIndependentOfOtherTower: true as const,
    untiedFromEmbedding: true as const,
    finalNormModulated: false as const,
    finalNormNote,
    predictionSemantics,
    predictionSemanticsLabel,
    vocabSize,
    maskTokenId,
    eosTokenId,
    tokenCount: T,
    startTokenOffset,
    absoluteTokenRange: absRange,
    focusToken,
    absoluteFocusPosition,
    contextTraceStatus,
    isHistoricalDenoiserInspection: isHistoricalDenoiserOut,
    matchesCurrentContextBoundary: towerRole === 'denoiser' ? dlOut.matchesCurrentContextBoundary : true,
    denoiserSourceRecordIndex: towerRole === 'denoiser' ? dlOut.sourceRecordIndex : null,
    denoiserSourceBlockId: towerRole === 'denoiser' ? dlOut.sourceBlockId : null,
    denoiserSourceRefinementIndex: towerRole === 'denoiser' ? dlOut.sourceRefinementIndex : null,
    denoiserSourceContextPrefixLen: towerRole === 'denoiser' ? dlOut.sourceContextPrefixLen : null,
    denoiserSourceContextCacheDigest: towerRole === 'denoiser' ? dlOut.sourceContextCacheDigest : null,
    historicalProvenanceWarning: isHistoricalDenoiserOut ? dlOut.historicalProvenanceWarning : null,
    samplerPresentationSource,
    samplerPresentation,
    top5AtFocus,
    syntheticMiniatureDisclosure: TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
  });

  const warningPartsOut: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warningPartsOut.push(`Context snapshot integrity error: ${view.snapshotIntegrity.error}`);
  }
  if (isHistoricalDenoiserOut && dlOut.historicalProvenanceWarning !== null) {
    warningPartsOut.push(dlOut.historicalProvenanceWarning);
  }
  const warningBanner = warningPartsOut.length > 0 ? warningPartsOut.join(' | ') : null;

  const atlasKey =
    `tt-out|${towerModel.parameterFingerprint}|${towerRole}|${executionId}|` +
    `off${startTokenOffset}|tok${focusToken}`;

  return {
    viewMode: 'output',
    title: `TwoTower Output & Untied LM Head · ${towerLabel} (${weightTableName})`,
    subtitle:
      (historicalTagOut ? `${historicalTagOut} · ` : '') +
      `${predictionSemanticsLabel} · Final RMSNorm Strictly Unmodulated`,
    bayId: 53,
    layerIndex: null,
    activeTowerFocus: towerRole === 'denoiser' ? 'denoiser_solo' : 'context_solo',
    solidData: nanoScene.solidData,
    solidCount: nanoScene.solidCount,
    lineData: nanoScene.lineData,
    lineCount: nanoScene.lineCount,
    ribbonData: nanoScene.ribbonData,
    ribbonCount: nanoScene.ribbonCount,
    sheetData,
    sheetCount: nanoScene.sheetCount,
    sheetBasisData: nanoScene.sheetBasisData,
    flowVertices: nanoScene.flowVertices,
    flowIndices: nanoScene.flowIndices,
    valueAtlas: repacked.atlas,
    atlasKey,
    fitBounds: nanoScene.fitBounds,
    opFocusBounds: nanoScene.opFocusBounds,
    landmarks,
    kvDepthTicks: [],
    calipers: nanoScene.calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeOpPartIds: nanoScene.activeOpPartIds,
    activeCellGrids: nanoScene.activeCellGrids,
    cellPopover: nanoScene.cellPopover,
    activeFormula: nanoScene.activeFormula,
    focusedAccumulation: null,
    activeSelection,
    stepperSteps: OUTPUT_STEPPER_STEPS,
    provenanceBadges,
    epistemicNote: TWOTOWER_SYNTHETIC_MINIATURE_DISCLOSURE,
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}
