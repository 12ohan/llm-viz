import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { validateDecodeCacheState } from '../src/math/incrementalDecode';
import {
  TWOTOWER_CONFIDENCE_PRESETS,
  TWOTOWER_MINI_MASK_TOKEN_ID,
  TWOTOWER_MINI_VOCAB,
  assertNonAliasedWithContextSnapshot,
  assertPresentationSafeNoInfinity,
  buildBlockConditioningBase,
  buildContextConditioningState,
  buildSamplerStepPresentation,
  cloneDecodeCacheState,
  commitTwoTowerBlockToContext,
  commitTwoTowerBlockToContextWithTrace,
  createDefaultTwoTowerModel,
  createTwoTowerModelParameters,
  deriveRefinementConditioningView,
  deriveRefinementUniformVariates,
  executeTwoTowerDiffusionRefinement,
  measureConditioningTypedArrayBytes,
  measureContextSnapshotTypedArrayBytes,
  runConfidenceUnmaskStep,
  runTwoTowerArStep,
  runTwoTowerContextPrefill,
  runTwoTowerContextPrefillWithTrace,
  runTwoTowerDenoiserPass,
  runTwoTowerDiffusionBlock,
  runTwoTowerMockArStep,
  validateContextConditioningState,
  validateContextSnapshot,
  type TwoTowerContextConditioningState,
  type TwoTowerContextSnapshot,
  type TwoTowerTraceOrigin,
} from '../src/twotower';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

function collectUniqueConditioningBuffers(
  states: readonly TwoTowerContextConditioningState[],
): { readonly bufferCount: number; readonly totalBytes: number } {
  const seen = new Set<ArrayBufferLike>();
  let totalBytes = 0;
  const add = (arr: ArrayBufferView) => {
    if (!seen.has(arr.buffer)) {
      seen.add(arr.buffer);
      totalBytes += arr.buffer.byteLength;
    }
  };
  for (const state of states) {
    for (const m of state.mambaSeedStates.values()) {
      add(m.convState);
      add(m.ssmState);
    }
    for (const g of state.gqaPrefixCaches.values()) {
      add(g.K);
      add(g.V);
    }
  }
  return { bufferCount: seen.size, totalBytes };
}

function softmaxRowF32(logits: Float32Array, vocabSize: number, B: number): Float32Array {
  const probs = new Float32Array(B * vocabSize);
  for (let j = 0; j < B; j++) {
    const off = j * vocabSize;
    let maxVal = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      if (logits[off + v] > maxVal) maxVal = logits[off + v];
    }
    let sum = 0;
    const exps = new Float64Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      exps[v] = Math.exp(logits[off + v] - maxVal);
      sum += exps[v];
    }
    for (let v = 0; v < vocabSize; v++) {
      probs[off + v] = Math.fround(exps[v] / sum);
    }
  }
  return probs;
}

describe('Stage TT2 Wave 1A: Context Trace Availability, Ownership Hardening, Conditioning Reuse & Sampler Presentation', () => {
  describe('1. Unified Trace Origin Taxonomy, Persistent Cache Validation & Deduplicated Block Conditioning', () => {
    it('enforces stateOrigin: "persistent_cache", exact snapshot/conditioning byte accounting, and zero-duplication refinement views', () => {
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });

      // Compile-time check covering all 4 unified taxonomy literals
      const taxonomy: readonly TwoTowerTraceOrigin[] = [
        'context_computed',
        'context_reconstructed',
        'persistent_cache',
        'denoiser_scratch',
      ];
      expect(taxonomy).toHaveLength(4);

      expect(snap.stateOrigin).toBe('persistent_cache');
      expect(() => validateContextSnapshot(snap, model.contextModel)).not.toThrow();

      // Relabeling stateOrigin on a persistent snapshot must be rejected
      const badOriginSnap: TwoTowerContextSnapshot = {
        ...snap,
        stateOrigin: 'denoiser_scratch' as unknown as 'persistent_cache',
      };
      expect(() => validateContextSnapshot(badOriginSnap, model.contextModel)).toThrow(TypeError);

      // Exact deduplicated byte count for persistent snapshot (P=8, kvCapacity=24, V=34):
      // prefixTokenIds (8*4=32) + ctxMamba (70656) + ctxGqa (9216) + mambaS2 (70656) + arLogits (136) + arProbs (136) = 150,832 B
      expect(measureContextSnapshotTypedArrayBytes(snap)).toBe(150_832);

      // Deduplicated per-block conditioning base (~78.00 KiB = 79,872 B at kvCapacity = 24)
      const base = buildBlockConditioningBase(model, snap, 0);
      expect(base.ownerRole).toBe('context_tower');
      expect(base.consumerRole).toBe('denoiser_tower');
      expect(base.mode).toBe('mask_diffusion');
      expect(base.blockId).toBe(0);
      expect(base.sourceStateDigest).toBe(snap.ctxState.stateDigest);
      expect(base.committedPrefixLen).toBe(8);
      expect(base.absoluteTokenStart).toBe(8);
      expect(base.allocatedKvCapacity).toBe(24);
      expect(base.byteLength).toBe(79_872);
      expect(measureConditioningTypedArrayBytes(base.baseConditioning)).toBe(79_872);

      // Derive 4 refinement views from the single base WITHOUT re-cloning 78 KiB typed arrays!
      const v0 = deriveRefinementConditioningView(base, 0);
      const v1 = deriveRefinementConditioningView(base, 1);
      const v2 = deriveRefinementConditioningView(base, 2);
      const v3 = deriveRefinementConditioningView(base, 3);
      const views = [v0, v1, v2, v3];

      expect(v0).toBe(base.baseConditioning);
      for (let r = 0; r < views.length; r++) {
        const view = views[r];
        expect(view.refinementIndex).toBe(r);
        expect(view.gqaPrefixCaches).toBe(base.baseConditioning.gqaPrefixCaches);
        expect(view.mambaSeedStates).toBe(base.baseConditioning.mambaSeedStates);
        expect(() => validateContextConditioningState(view, model, 4)).not.toThrow();
        expect(() => assertNonAliasedWithContextSnapshot(view, snap)).not.toThrow();

        // Conditioning digest matches a standalone freshly cloned buildContextConditioningState bit-for-bit
        const standalone = buildContextConditioningState(model, snap, {
          mode: 'mask_diffusion',
          blockId: 0,
          refinementIndex: r,
        });
        expect(view.conditioningDigest).toBe(standalone.conditioningDigest);
      }

      // Digests across refinements are strictly distinct
      expect(new Set(views.map((v) => v.conditioningDigest)).size).toBe(4);

      // Total unique ArrayBuffer bytes across ALL 4 refinement views is STILL exactly 79,872 B (ZERO extra 78 KiB allocation!)
      const dedup = collectUniqueConditioningBuffers(views);
      expect(dedup.totalBytes).toBe(79_872);
      expect(dedup.bufferCount).toBe(23 * 2 + 6 * 2);

      // Invalid refinementIndex rejected
      expect(() => deriveRefinementConditioningView(base, -1)).toThrow(RangeError);
      expect(() => deriveRefinementConditioningView(base, 1.5)).toThrow(RangeError);

      // cloneDecodeCacheState creates a valid, strictly non-aliased copy
      const clonedCtx = cloneDecodeCacheState(snap.ctxState);
      expect(() => validateDecodeCacheState(model.contextModel, clonedCtx)).not.toThrow();
      expect(clonedCtx.stateDigest).toBe(snap.ctxState.stateDigest);
      expect(clonedCtx.prefixTokenIds.buffer).not.toBe(snap.ctxState.prefixTokenIds.buffer);
      expect(clonedCtx.mambaStates.get(1)!.ssmState.buffer).not.toBe(
        snap.ctxState.mambaStates.get(1)!.ssmState.buffer,
      );
      expect(clonedCtx.gqaCaches.get(6)!.K.buffer).not.toBe(
        snap.ctxState.gqaCaches.get(6)!.K.buffer,
      );
    });
  });

  describe('2. Context Incremental Execution Trace Availability, Split Provenance, Reconstruction & Mutation Isolation', () => {
    it('preserves BOTH split segments for prefill (S >= 2) and block commit (B >= 2) plus single-segment (S=1 / B=1) provenance', () => {
      const model = createDefaultTwoTowerModel();

      // A. Prefill S = 8 -> 2 segments ([0..6] + [7..7])
      const prefillExec = runTwoTowerContextPrefillWithTrace(model, [0, 1, 2, 3, 4, 5, 6, 7], {
        kvCapacity: 24,
        branchId: 'ctx-branch-a',
      });
      const legacyPrefillSnap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], {
        kvCapacity: 24,
        branchId: 'ctx-branch-a',
      });

      expect(prefillExec.operation).toBe('prefill');
      expect(prefillExec.snapshot.snapshotDigest).toBe(legacyPrefillSnap.snapshotDigest);
      expect(prefillExec.nextStateDigest).toBe(prefillExec.snapshot.ctxState.stateDigest);
      expect(prefillExec.committedTokenRange).toEqual([0, 7]);
      expect(prefillExec.committedTokenIds).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      expect(prefillExec.committedTokenLabels).toHaveLength(8);
      expect(prefillExec.segments).toHaveLength(2);

      const [pSeg0, pSeg1] = prefillExec.segments;
      expect(pSeg0.segmentRole).toBe('split_prefix_for_mamba_s2');
      expect(pSeg0.segmentIndex).toBe(0);
      expect(pSeg0.segmentCount).toBe(2);
      expect(pSeg0.tokenRange).toEqual([0, 6]);
      expect(pSeg0.tokens).toBe(7);
      expect(pSeg0.prefixTokens).toBe(0);
      expect(pSeg0.totalTokens).toBe(7);

      expect(pSeg1.segmentRole).toBe('split_final_token');
      expect(pSeg1.segmentIndex).toBe(1);
      expect(pSeg1.segmentCount).toBe(2);
      expect(pSeg1.tokenRange).toEqual([7, 7]);
      expect(pSeg1.tokens).toBe(1);
      expect(pSeg1.prefixTokens).toBe(7);
      expect(pSeg1.totalTokens).toBe(8);

      // Unbroken provenance chain across both segments
      expect(prefillExec.prevStateDigest).toBe(pSeg0.provenance.prevStateDigest);
      expect(pSeg0.provenance.nextStateDigest).toBe(pSeg1.provenance.prevStateDigest);
      expect(pSeg1.provenance.parentStepExecutionId).toBe(pSeg0.stepExecutionId);
      expect(pSeg1.provenance.nextStateDigest).toBe(prefillExec.nextStateDigest);

      for (let pos = 0; pos <= 6; pos++) {
        expect(prefillExec.getSegmentForAbsolutePosition(pos)).toBe(pSeg0);
      }
      expect(prefillExec.getSegmentForAbsolutePosition(7)).toBe(pSeg1);
      expect(() => prefillExec.getSegmentForAbsolutePosition(-1)).toThrow(RangeError);
      expect(() => prefillExec.getSegmentForAbsolutePosition(8)).toThrow(RangeError);
      expect(() => prefillExec.getSegmentForAbsolutePosition(1.5)).toThrow(RangeError);

      // B. Commit B = 4 -> 2 segments ([8..10] + [11..11])
      const commitExec = commitTwoTowerBlockToContextWithTrace(
        model,
        prefillExec.snapshot,
        [16, 17, 18, 19],
        { branchId: 'ctx-branch-a' },
      );
      const legacyCommitSnap = commitTwoTowerBlockToContext(
        model,
        prefillExec.snapshot,
        [16, 17, 18, 19],
        { branchId: 'ctx-branch-a' },
      );

      expect(commitExec.operation).toBe('commit_block');
      expect(commitExec.snapshot.snapshotDigest).toBe(legacyCommitSnap.snapshotDigest);
      expect(commitExec.prevStateDigest).toBe(prefillExec.snapshot.ctxState.stateDigest);
      expect(commitExec.nextStateDigest).toBe(commitExec.snapshot.ctxState.stateDigest);
      expect(commitExec.committedTokenRange).toEqual([8, 11]);
      expect(commitExec.committedTokenIds).toEqual([16, 17, 18, 19]);
      expect(commitExec.segments).toHaveLength(2);

      const [cSeg0, cSeg1] = commitExec.segments;
      expect(cSeg0.segmentRole).toBe('split_prefix_for_mamba_s2');
      expect(cSeg0.segmentIndex).toBe(0);
      expect(cSeg0.segmentCount).toBe(2);
      expect(cSeg0.tokenRange).toEqual([8, 10]);
      expect(cSeg0.tokens).toBe(3);
      expect(cSeg0.prefixTokens).toBe(8);

      expect(cSeg1.segmentRole).toBe('split_final_token');
      expect(cSeg1.segmentIndex).toBe(1);
      expect(cSeg1.segmentCount).toBe(2);
      expect(cSeg1.tokenRange).toEqual([11, 11]);
      expect(cSeg1.tokens).toBe(1);
      expect(cSeg1.prefixTokens).toBe(11);
      expect(cSeg1.provenance.parentStepExecutionId).toBe(cSeg0.stepExecutionId);

      expect(commitExec.getSegmentForAbsolutePosition(8)).toBe(cSeg0);
      expect(commitExec.getSegmentForAbsolutePosition(10)).toBe(cSeg0);
      expect(commitExec.getSegmentForAbsolutePosition(11)).toBe(cSeg1);
      expect(() => commitExec.getSegmentForAbsolutePosition(7)).toThrow(RangeError);
      expect(() => commitExec.getSegmentForAbsolutePosition(12)).toThrow(RangeError);

      // C. Single-segment prefill (S = 1) and single-segment commit (B = 1) + AR / Mock-AR contextExecution
      const singlePrefill = runTwoTowerContextPrefillWithTrace(model, [5], { kvCapacity: 24 });
      expect(singlePrefill.segments).toHaveLength(1);
      expect(singlePrefill.segments[0].segmentRole).toBe('single_segment');
      expect(singlePrefill.segments[0].segmentIndex).toBe(0);
      expect(singlePrefill.segments[0].segmentCount).toBe(1);
      expect(singlePrefill.segments[0].tokenRange).toEqual([0, 0]);
      expect(singlePrefill.getSegmentForAbsolutePosition(0)).toBe(singlePrefill.segments[0]);

      const singleCommit = commitTwoTowerBlockToContextWithTrace(
        model,
        singlePrefill.snapshot,
        [9],
      );
      expect(singleCommit.segments).toHaveLength(1);
      expect(singleCommit.segments[0].segmentRole).toBe('single_segment');
      expect(singleCommit.segments[0].segmentIndex).toBe(0);
      expect(singleCommit.segments[0].segmentCount).toBe(1);
      expect(singleCommit.segments[0].tokenRange).toEqual([1, 1]);
      expect(singleCommit.getSegmentForAbsolutePosition(1)).toBe(singleCommit.segments[0]);

      const arStep = runTwoTowerArStep(model, singlePrefill.snapshot);
      expect(arStep.contextExecution.operation).toBe('commit_block');
      expect(arStep.contextExecution.snapshot).toBe(arStep.nextSnapshot);
      expect(arStep.contextExecution.segments).toHaveLength(1);
      expect(arStep.contextExecution.segments[0].getLayerTrace(1).traceOrigin).toBe(
        'context_computed',
      );

      const mockArStep = runTwoTowerMockArStep(model, singlePrefill.snapshot);
      expect(mockArStep.contextExecution.operation).toBe('commit_block');
      expect(mockArStep.contextExecution.snapshot).toBe(mockArStep.nextSnapshot);
      expect(mockArStep.contextExecution.segments).toHaveLength(1);
      expect(mockArStep.contextExecution.segments[0].getLayerTrace(6).traceOrigin).toBe(
        'context_computed',
      );
    });

    it('transitions traceOrigin cleanly across releaseEagerTraces(), reconstructs all 3 bay kinds (M, E, *) at 0-ULP, and uses strictly model.contextModel', () => {
      const model = createDefaultTwoTowerModel();
      const prefillExec = runTwoTowerContextPrefillWithTrace(model, [0, 1, 2, 3, 4, 5], {
        kvCapacity: 24,
        retainedTraceBays: [1, 2, 6, 43],
      });
      const commitExec = commitTwoTowerBlockToContextWithTrace(
        model,
        prefillExec.snapshot,
        [10, 11, 12, 13],
        { retainedTraceBays: [1, 2, 6, 43] },
      );

      // Verify both segments on BOTH prefill and commit across M (bay 1), E (bay 2), * (bay 6 & 43)
      for (const exec of [prefillExec, commitExec]) {
        for (const seg of exec.segments) {
          expect(seg.retainedTraceBayIds()).toEqual([1, 2, 6, 43]);
          expect(seg.getLayerTraceOrigin(1)).toBe('context_computed');
          expect(seg.getLayerTraceOrigin(2)).toBe('context_computed');
          expect(seg.getLayerTraceOrigin(6)).toBe('context_computed');
          expect(seg.getLayerTraceOrigin(13)).toBeNull(); // non-eager GQA bay

          const mEager = seg.getLayerTrace(1);
          const eEager = seg.getLayerTrace(2);
          const gEager = seg.getLayerTrace(6);
          expect(mEager.traceOrigin).toBe('context_computed');
          expect(eEager.traceOrigin).toBe('context_computed');
          expect(gEager.traceOrigin).toBe('context_computed');
          expect(mEager.segmentIndex).toBe(seg.segmentIndex);
          expect(mEager.segmentRole).toBe(seg.segmentRole);
          expect(mEager.stepExecutionId).toBe(seg.stepExecutionId);

          if (mEager.kind !== 'M' || eEager.kind !== 'E' || gEager.kind !== '*') {
            throw new Error('Unexpected bay kinds');
          }

          // Snapshot pristine eager tensors
          const mY = new Float32Array(mEager.sim.Y);
          const mH = new Float32Array(mEager.sim.H);
          const mConv = new Float32Array(mEager.sim.conv_buf);
          const eY = new Float32Array(eEager.sim.Y);
          const eTopk = new Int32Array(eEager.sim.topk_indices);
          const eWeights = new Float32Array(eEager.sim.topk_weights);
          const gY = new Float32Array(gEager.sim.Y);
          const gK = new Float32Array(gEager.sim.K);
          const gV = new Float32Array(gEager.sim.V);
          const gSoftmax = new Float32Array(gEager.sim.Softmax);

          // Access non-eager bay 13 (*) -> lazily reconstructed
          const g13 = seg.getLayerTrace(13);
          expect(g13.traceOrigin).toBe('context_reconstructed');
          expect(seg.getLayerTraceOrigin(13)).toBe('context_reconstructed');

          const bytesBeforeRelease = seg.retainedTypedArrayBytes();
          seg.releaseEagerTraces();
          const bytesAfterRelease = seg.retainedTypedArrayBytes();
          expect(seg.retainedTraceBayIds()).toEqual([]);
          expect(seg.getLayerTraceOrigin(1)).toBeNull();
          expect(seg.getLayerTraceOrigin(2)).toBeNull();
          expect(seg.getLayerTraceOrigin(6)).toBeNull();
          expect(bytesAfterRelease).toBeLessThan(bytesBeforeRelease);

          // Reconstruct all 3 bay kinds after release -> must be 'context_reconstructed' & 0-ULP bit-exact!
          const mRec = seg.getLayerTrace(1);
          const eRec = seg.getLayerTrace(2);
          const gRec = seg.getLayerTrace(6);
          expect(mRec.traceOrigin).toBe('context_reconstructed');
          expect(eRec.traceOrigin).toBe('context_reconstructed');
          expect(gRec.traceOrigin).toBe('context_reconstructed');
          expect(seg.getLayerTraceOrigin(1)).toBe('context_reconstructed');
          expect(seg.getLayerTraceOrigin(2)).toBe('context_reconstructed');
          expect(seg.getLayerTraceOrigin(6)).toBe('context_reconstructed');

          if (mRec.kind !== 'M' || eRec.kind !== 'E' || gRec.kind !== '*') {
            throw new Error('Unexpected reconstructed bay kinds');
          }

          const tag = `${exec.operation}.seg${seg.segmentIndex}`;
          assertBitExactFiniteArray(mRec.sim.Y, mY, `${tag}.M1.Y`);
          assertBitExactFiniteArray(mRec.sim.H, mH, `${tag}.M1.H`);
          assertBitExactFiniteArray(mRec.sim.conv_buf, mConv, `${tag}.M1.conv_buf`);
          assertBitExactFiniteArray(eRec.sim.Y, eY, `${tag}.E2.Y`);
          expect(Array.from(eRec.sim.topk_indices)).toEqual(Array.from(eTopk));
          assertBitExactFiniteArray(eRec.sim.topk_weights, eWeights, `${tag}.E2.topk_weights`);
          assertBitExactFiniteArray(gRec.sim.Y, gY, `${tag}.G6.Y`);
          assertBitExactFiniteArray(gRec.sim.K, gK, `${tag}.G6.K`);
          assertBitExactFiniteArray(gRec.sim.V, gV, `${tag}.G6.V`);
          assertBitExactFiniteArray(gRec.sim.Softmax, gSoftmax, `${tag}.G6.Softmax`);

          // Typed getters enforce kind checking
          expect(() => seg.getGqaTrace(1)).toThrow(TypeError);
          expect(() => seg.getMambaTrace(2)).toThrow(TypeError);
          expect(() => seg.getMoeTrace(6)).toThrow(TypeError);
        }
      }

      // Context trace strictly uses model.contextModel (independent of denoiser perturbationScale,
      // and sensitive to contextSeed — never substitutes unrelated standalone Nano fixtures!)
      const altDenoiserModel = createTwoTowerModelParameters({ perturbationScale: 0.45 });
      const altContextModel = createTwoTowerModelParameters({ contextSeed: 0x11223344 });
      const execSameCtx = runTwoTowerContextPrefillWithTrace(altDenoiserModel, [0, 1, 2, 3, 4, 5], {
        kvCapacity: 24,
        retainedTraceBays: [],
      });
      const execDiffCtx = runTwoTowerContextPrefillWithTrace(altContextModel, [0, 1, 2, 3, 4, 5], {
        kvCapacity: 24,
        retainedTraceBays: [],
      });

      assertBitExactFiniteArray(
        execSameCtx.segments[0].getMambaTrace(1).Y,
        prefillExec.segments[0].getMambaTrace(1).Y,
        'Context trace invariant to Denoiser perturbation',
      );
      let diffSum = 0;
      const a = execDiffCtx.segments[0].getMambaTrace(1).Y;
      const b = prefillExec.segments[0].getMambaTrace(1).Y;
      for (let i = 0; i < a.length; i++) diffSum += Math.abs(a[i] - b[i]);
      expect(diffSum).toBeGreaterThan(1e-3);
    });

    it('isolates historical Context segment reconstruction from external mutations on prevSnapshot, nextSnapshot, and returned traces', () => {
      const model = createDefaultTwoTowerModel();
      const prevSnap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });
      const commitExec = commitTwoTowerBlockToContextWithTrace(
        model,
        prevSnap,
        [16, 17, 18, 19],
        { retainedTraceBays: [1, 2, 6] },
      );

      // Capture pristine baselines for both segments across eager and lazy bays
      const s0M1Clean = new Float32Array(commitExec.segments[0].getMambaTrace(1).Y);
      const s0E2Clean = new Float32Array(commitExec.segments[0].getMoeTrace(2).Y);
      const s0G6Clean = new Float32Array(commitExec.segments[0].getGqaTrace(6).Y);
      const s0M15Clean = new Float32Array(commitExec.segments[0].getMambaTrace(15).Y);
      const s0G20Clean = new Float32Array(commitExec.segments[0].getGqaTrace(20).Y);

      const s1M1Clean = new Float32Array(commitExec.segments[1].getMambaTrace(1).Y);
      const s1G6Clean = new Float32Array(commitExec.segments[1].getGqaTrace(6).Y);
      const s1M15Clean = new Float32Array(commitExec.segments[1].getMambaTrace(15).Y);

      // Aggressively corrupt public prevSnapshot, public nextSnapshot, and returned trace buffers!
      prevSnap.ctxState.mambaStates.get(1)!.ssmState.fill(-9999);
      prevSnap.ctxState.mambaStates.get(1)!.convState.fill(9999);
      prevSnap.ctxState.mambaStates.get(15)!.ssmState.fill(8888);
      prevSnap.ctxState.gqaCaches.get(6)!.K.fill(-8888);
      prevSnap.ctxState.gqaCaches.get(6)!.V.fill(7777);
      prevSnap.ctxState.gqaCaches.get(20)!.K.fill(-7777);
      prevSnap.ctxState.prefixTokenIds.fill(0);

      commitExec.snapshot.ctxState.mambaStates.get(1)!.ssmState.fill(6666);
      commitExec.snapshot.ctxState.gqaCaches.get(6)!.K.fill(-6666);

      commitExec.segments[0].getMambaTrace(1).Y.fill(5555);
      commitExec.segments[0].getMambaTrace(1).X.fill(4444);
      commitExec.segments[0].getGqaTrace(6).K.fill(-4444);
      commitExec.segments[1].getMambaTrace(1).Y.fill(3333);
      commitExec.segments[1].getGqaTrace(6).Y.fill(-3333);

      // Release all cached traces and reconstruct from private isolated checkpoints
      commitExec.releaseEagerTraces();

      assertBitExactFiniteArray(
        commitExec.segments[0].getMambaTrace(1).Y,
        s0M1Clean,
        'isolated commit seg0 M1',
      );
      assertBitExactFiniteArray(
        commitExec.segments[0].getMoeTrace(2).Y,
        s0E2Clean,
        'isolated commit seg0 E2',
      );
      assertBitExactFiniteArray(
        commitExec.segments[0].getGqaTrace(6).Y,
        s0G6Clean,
        'isolated commit seg0 G6',
      );
      assertBitExactFiniteArray(
        commitExec.segments[0].getMambaTrace(15).Y,
        s0M15Clean,
        'isolated commit seg0 lazy M15',
      );
      assertBitExactFiniteArray(
        commitExec.segments[0].getGqaTrace(20).Y,
        s0G20Clean,
        'isolated commit seg0 lazy G20',
      );
      assertBitExactFiniteArray(
        commitExec.segments[1].getMambaTrace(1).Y,
        s1M1Clean,
        'isolated commit seg1 M1',
      );
      assertBitExactFiniteArray(
        commitExec.segments[1].getGqaTrace(6).Y,
        s1G6Clean,
        'isolated commit seg1 G6',
      );
      assertBitExactFiniteArray(
        commitExec.segments[1].getMambaTrace(15).Y,
        s1M15Clean,
        'isolated commit seg1 lazy M15',
      );
    });
  });

  describe('3. Denoiser Modulation & Trace Mutation Isolation, Trace Release & Shared Refinement Primitive', () => {
    it('isolates getLayerModulation and lazy/evicted reconstructions from mutating trace.modulation, sim, tEmb, timestepProjection, residualStreams, or tokenIds', () => {
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });
      const cond = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 0,
      });

      const pass = runTwoTowerDenoiserPass(
        model,
        cond,
        [TWOTOWER_MINI_MASK_TOKEN_ID, TWOTOWER_MINI_MASK_TOKEN_ID, 4, TWOTOWER_MINI_MASK_TOKEN_ID],
        { retainedTraceBays: [1, 2, 6] },
      );

      // Verify traceOrigin: 'denoiser_scratch' across all 3 bay kinds
      const tr1 = pass.getLayerTrace(1);
      const tr2 = pass.getLayerTrace(2);
      const tr6 = pass.getLayerTrace(6);
      expect(tr1.traceOrigin).toBe('denoiser_scratch');
      expect(tr2.traceOrigin).toBe('denoiser_scratch');
      expect(tr6.traceOrigin).toBe('denoiser_scratch');
      expect(tr1.modulation).not.toBeNull();
      expect(tr2.modulation).not.toBeNull();
      expect(tr6.modulation).not.toBeNull();

      // Snapshot pristine baselines before mutation
      const cleanMod1Shift = new Float32Array(pass.getLayerModulation(1)!.shift);
      const cleanMod1Scale = new Float32Array(pass.getLayerModulation(1)!.scale);
      const cleanMod1Gate = new Float32Array(pass.getLayerModulation(1)!.gate);
      const cleanMod2Shift = new Float32Array(pass.getLayerModulation(2)!.shift);
      const cleanMod6Gate = new Float32Array(pass.getLayerModulation(6)!.gate);
      const cleanMod15Shift = new Float32Array(pass.getLayerModulation(15)!.shift);

      const cleanM1Y = new Float32Array(tr1.sim.Y);
      const cleanE2Y = new Float32Array(tr2.sim.Y);
      const cleanG6Y = new Float32Array(tr6.sim.Y);

      // Run an independent reference pass to capture pristine non-eager bays 15 (M) and 20 (*)
      const refPass = runTwoTowerDenoiserPass(
        model,
        cond,
        [TWOTOWER_MINI_MASK_TOKEN_ID, TWOTOWER_MINI_MASK_TOKEN_ID, 4, TWOTOWER_MINI_MASK_TOKEN_ID],
        { retainedTraceBays: [] },
      );
      const cleanM15Y = new Float32Array(refPass.getMambaTrace(15).Y);
      const cleanG20Y = new Float32Array(refPass.getGqaTrace(20).Y);

      // Aggressively corrupt returned trace.modulation, sim buffers, tEmb, timestepProjection, residualStreams & tokenIds!
      tr1.modulation!.shift.fill(999);
      tr1.modulation!.scale.fill(-999);
      tr1.modulation!.gate.fill(888);
      tr2.modulation!.shift.fill(777);
      tr6.modulation!.gate.fill(-777);

      pass.tEmb!.fill(666);
      pass.timestepProjection!.phi.fill(555);
      pass.timestepProjection!.tRepr.fill(555);
      pass.timestepProjection!.tEmb.fill(555);
      pass.residualStreams[0].fill(444);
      pass.residualStreams[1].fill(444);
      pass.residualStreams[5].fill(444);
      pass.residualStreams[14].fill(444);
      pass.residualStreams[19].fill(444);
      pass.tokenIds.fill(0);

      tr1.sim.Y.fill(333);
      tr1.sim.X.fill(222);
      tr1.sim.tokenIds.fill(0);
      tr2.sim.Y.fill(-333);
      tr6.sim.Y.fill(111);

      // 1. getLayerModulation(bayId) MUST remain 0-ULP uncorrupted!
      assertBitExactFiniteArray(pass.getLayerModulation(1)!.shift, cleanMod1Shift, 'mod1.shift');
      assertBitExactFiniteArray(pass.getLayerModulation(1)!.scale, cleanMod1Scale, 'mod1.scale');
      assertBitExactFiniteArray(pass.getLayerModulation(1)!.gate, cleanMod1Gate, 'mod1.gate');
      assertBitExactFiniteArray(pass.getLayerModulation(2)!.shift, cleanMod2Shift, 'mod2.shift');
      assertBitExactFiniteArray(pass.getLayerModulation(6)!.gate, cleanMod6Gate, 'mod6.gate');
      assertBitExactFiniteArray(pass.getLayerModulation(15)!.shift, cleanMod15Shift, 'mod15.shift');

      // 2. Even on an immediate cache hit BEFORE releaseEagerTraces(), getLayerTrace(1) attaches a fresh uncorrupted modulation clone!
      const tr1Again = pass.getLayerTrace(1);
      expect(tr1Again.modulation).not.toBe(tr1.modulation);
      assertBitExactFiniteArray(tr1Again.modulation!.shift, cleanMod1Shift, 'cache-hit tr1.mod.shift');
      assertBitExactFiniteArray(tr1Again.modulation!.scale, cleanMod1Scale, 'cache-hit tr1.mod.scale');
      assertBitExactFiniteArray(tr1Again.modulation!.gate, cleanMod1Gate, 'cache-hit tr1.mod.gate');

      // 3. Lazy non-eager bays 15 & 20 MUST reconstruct 0-ULP identically despite all mutations!
      assertBitExactFiniteArray(pass.getMambaTrace(15).Y, cleanM15Y, 'lazy M15 after mutations');
      assertBitExactFiniteArray(pass.getGqaTrace(20).Y, cleanG20Y, 'lazy G20 after mutations');

      // 4. releaseEagerTraces() drops memory footprint while preserving 0-ULP rebuilds of bays 1, 2, 6!
      const bytesBefore = pass.retainedTypedArrayBytes();
      expect(pass.retainedTraceBayIds().length).toBeGreaterThanOrEqual(3);
      pass.releaseEagerTraces();
      expect(pass.retainedTraceBayIds()).toEqual([]);
      expect(pass.retainedTypedArrayBytes()).toBeLessThan(bytesBefore);

      const tr1Rebuilt = pass.getLayerTrace(1);
      const tr2Rebuilt = pass.getLayerTrace(2);
      const tr6Rebuilt = pass.getLayerTrace(6);
      expect(tr1Rebuilt.traceOrigin).toBe('denoiser_scratch');
      expect(tr2Rebuilt.traceOrigin).toBe('denoiser_scratch');
      expect(tr6Rebuilt.traceOrigin).toBe('denoiser_scratch');

      assertBitExactFiniteArray(tr1Rebuilt.sim.Y, cleanM1Y, 'rebuilt M1.Y after release');
      assertBitExactFiniteArray(tr1Rebuilt.modulation!.shift, cleanMod1Shift, 'rebuilt M1.mod.shift');
      assertBitExactFiniteArray(tr2Rebuilt.sim.Y, cleanE2Y, 'rebuilt E2.Y after release');
      assertBitExactFiniteArray(tr6Rebuilt.sim.Y, cleanG6Y, 'rebuilt G6.Y after release');
    });

    it('shares one refinement primitive (executeTwoTowerDiffusionRefinement) with runTwoTowerDiffusionBlock and deduplicates block conditioning', () => {
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });

      const block = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        temperature: 0.0,
      });

      expect(block.refinements).toHaveLength(4);
      expect(block.contextCommitExecution).not.toBeNull();
      expect(block.contextCommitExecution!.snapshot).toBe(block.nextSnapshot);
      expect(block.contextCommitExecution!.segments).toHaveLength(2);

      // All 4 refinements inside runTwoTowerDiffusionBlock share the single frozen conditioning Maps!
      const blockCondDedup = collectUniqueConditioningBuffers(
        block.refinements.map((r) => r.conditioning),
      );
      expect(blockCondDedup.totalBytes).toBe(79_872);

      // Driving the exact same 4 refinements manually via buildBlockConditioningBase + executeTwoTowerDiffusionRefinement
      // produces 0-ULP identical passes, logits, digests, and tokens
      const base = buildBlockConditioningBase(model, snap, 0);
      let xt: readonly number[] = [
        TWOTOWER_MINI_MASK_TOKEN_ID,
        TWOTOWER_MINI_MASK_TOKEN_ID,
        TWOTOWER_MINI_MASK_TOKEN_ID,
        TWOTOWER_MINI_MASK_TOKEN_ID,
      ];
      for (let r = 0; r < 4; r++) {
        const manual = executeTwoTowerDiffusionRefinement({
          model,
          conditioningSource: base,
          xt,
          refinementIndex: r,
          stepsPerBlock: 4,
          confidenceThreshold: 0.9999,
          temperature: 0.0,
        });
        const fromBlock = block.refinements[r];
        expect(manual.conditioning.conditioningDigest).toBe(fromBlock.conditioning.conditioningDigest);
        expect(manual.denoiserPass.passExecutionId).toBe(fromBlock.denoiserPass.passExecutionId);
        assertBitExactFiniteArray(
          manual.denoiserPass.logits,
          fromBlock.denoiserPass.logits,
          `refinement ${r} logits`,
        );
        expect(manual.unmaskStep.xtNext).toEqual(fromBlock.unmaskStep.xtNext);
        xt = manual.unmaskStep.xtNext;
      }

      // Reject non-mask_diffusion conditioning or mismatched refinementIndex
      const mockCond = buildContextConditioningState(model, snap, {
        mode: 'mock_ar',
        blockId: 0,
        refinementIndex: 0,
      });
      expect(() =>
        executeTwoTowerDiffusionRefinement({
          model,
          conditioningSource: mockCond,
          xt,
          refinementIndex: 0,
          stepsPerBlock: 4,
          confidenceThreshold: 0.22,
          temperature: 0.0,
        }),
      ).toThrow(RangeError);

      const diffCondR1 = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 1,
      });
      expect(() =>
        executeTwoTowerDiffusionRefinement({
          model,
          conditioningSource: diffCondR1,
          xt,
          refinementIndex: 0,
          stepsPerBlock: 4,
          confidenceThreshold: 0.22,
          temperature: 0.0,
        }),
      ).toThrow(RangeError);
    });
  });

  describe('4. Presentation-Safe Sampler Contract, Confidence Presets & Deterministic Keyed Variates', () => {
    it('exports calibrated miniature preset (γ = 0.22) alongside Paper (0.80) and Reference Code (0.90) presets', () => {
      expect(TWOTOWER_CONFIDENCE_PRESETS).toHaveLength(3);

      const mini = TWOTOWER_CONFIDENCE_PRESETS.find((p) => p.threshold === 0.22);
      const paper = TWOTOWER_CONFIDENCE_PRESETS.find((p) => p.threshold === 0.8);
      const refCode = TWOTOWER_CONFIDENCE_PRESETS.find((p) => p.threshold === 0.9);

      expect(mini).toBeDefined();
      expect(mini!.label).toBe('Miniature Educational Default (γ = 0.22)');
      expect(mini!.isCalibratedMiniatureDefault).toBe(true);

      expect(paper).toBeDefined();
      expect(paper!.label).toContain('0.80');
      expect(paper!.isCalibratedMiniatureDefault).toBe(false);

      expect(refCode).toBeDefined();
      expect(refCode!.label).toContain('0.90');
      expect(refCode!.isCalibratedMiniatureDefault).toBe(false);
    });

    it('derives strictly finite [0, 1) keyed uniform variates reproducibly and sensitively across every coordinate (ZERO Math.random)', () => {
      const baseParams = {
        samplingSeed: 20260701,
        sessionId: 'sess-alpha',
        branchId: 'branch-0',
        blockId: 0,
        refinementIndex: 0,
        blockSize: 4,
        vocabSize: 34,
      };

      const a = deriveRefinementUniformVariates(baseParams);
      const aRepeat = deriveRefinementUniformVariates({ ...baseParams });

      expect(a.variates).toBeInstanceOf(Float32Array);
      expect(a.variates).toHaveLength(4 * 34);
      expect(a.derivationKey).toBe(aRepeat.derivationKey);
      expect(a.variateDigest).toBe(aRepeat.variateDigest);
      expect(a.variateDigest).toHaveLength(16);
      assertBitExactFiniteArray(a.variates, aRepeat.variates, 'variates bit-exact repeat');

      // Every element strictly finite in [0, 1)
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let i = 0; i < a.variates.length; i++) {
        const u = a.variates[i];
        expect(Number.isFinite(u)).toBe(true);
        expect(u).toBeGreaterThanOrEqual(0);
        expect(u).toBeLessThan(1);
        if (u < minVal) minVal = u;
        if (u > maxVal) maxVal = u;
      }
      expect(maxVal - minVal).toBeGreaterThan(0.5);

      // Changing ANY single input key produces distinct variates & digests
      const variants = [
        deriveRefinementUniformVariates({ ...baseParams, samplingSeed: 20260702 }),
        deriveRefinementUniformVariates({ ...baseParams, sessionId: 'sess-beta' }),
        deriveRefinementUniformVariates({ ...baseParams, branchId: 'branch-1' }),
        deriveRefinementUniformVariates({ ...baseParams, blockId: 1 }),
        deriveRefinementUniformVariates({ ...baseParams, refinementIndex: 1 }),
        deriveRefinementUniformVariates({ ...baseParams, blockSize: 8 }),
        deriveRefinementUniformVariates({ ...baseParams, vocabSize: 16 }),
      ];

      const digests = new Set([a.variateDigest, ...variants.map((v) => v.variateDigest)]);
      expect(digests.size).toBe(1 + variants.length);

      // Length-prefixed framing prevents delimiter collision between sessionId and branchId
      const split1 = deriveRefinementUniformVariates({
        ...baseParams,
        sessionId: 'a|b',
        branchId: 'c',
      });
      const split2 = deriveRefinementUniformVariates({
        ...baseParams,
        sessionId: 'a',
        branchId: 'b|c',
      });
      expect(split1.variateDigest).not.toBe(split2.variateDigest);

      // Invalid coordinates rejected
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, samplingSeed: 1.5 }),
      ).toThrow(RangeError);
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, sessionId: '' }),
      ).toThrow(RangeError);
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, branchId: '' }),
      ).toThrow(RangeError);
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, blockId: -1 }),
      ).toThrow(RangeError);
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, refinementIndex: -1 }),
      ).toThrow(RangeError);
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, blockSize: 0 }),
      ).toThrow(RangeError);
      expect(() =>
        deriveRefinementUniformVariates({ ...baseParams, vocabSize: 1 }),
      ).toThrow(RangeError);
    });

    it('separates raw LM-head vs constrained MDLM, eliminates Infinity on locked slots, and classifies all 6 slot decisions', () => {
      const vocab = ['a', 'b', 'c', 'd', '[MASK]', '<eos>'];
      const V = vocab.length;
      const MASK = 4;

      // Scenario A: locked_prior + committed_threshold + remasked + adversarial MASK logit in raw LM head!
      // Slot 0: locked prior at token 2 (raw LM head favors token 0 & MASK!)
      // Slot 1: masked, raw LM head puts huge logit (+10) on MASK, second best token 1 (+2.0) -> high constrained prob > 0.7 -> committed_threshold!
      // Slot 2: masked, uniform non-mask logits -> constrained prob 0.2 <= 0.7 -> remasked!
      const rawLogitsA = new Float32Array([
        8.0, 0.0, -5.0, 0.0, 9.0, 0.0, // slot 0 (locked=2)
        0.0, 3.5, 0.0, 0.0, 10.0, 0.0, // slot 1 (masked -> raw top1 is MASK, constrained top1 is 'b' with prob ~0.89 > 0.7)
        0.0, 0.0, 0.0, 0.0, 6.0, 0.0, // slot 2 (masked -> raw top1 is MASK, constrained prob 0.2 <= 0.7 -> remasked)
      ]);
      const rawProbsA = softmaxRowF32(rawLogitsA, V, 3);
      const unmaskA = runConfidenceUnmaskStep({
        xt: [2, MASK, MASK],
        logits: rawLogitsA,
        vocabSize: V,
        maskTokenId: MASK,
        stepIndex: 0,
        stepsPerBlock: 3,
        confidenceThreshold: 0.7,
        temperature: 0.0,
      });

      // Internal unmaskStep still uses +Infinity for locked slot 0 (TT1 contract preserved)
      expect(unmaskA.confidence[0]).toBe(Infinity);

      const presA = buildSamplerStepPresentation({
        rawLogits: rawLogitsA,
        rawProbs: rawProbsA,
        unmaskStep: unmaskA,
        vocab,
        maskTokenId: MASK,
        confidenceThreshold: 0.7,
        temperature: 0.0,
        absoluteTokenStart: 8,
      });

      expect(() => assertPresentationSafeNoInfinity(presA)).not.toThrow();
      expect(presA.slots).toHaveLength(3);

      // Slot 0: locked prior -> ZERO Infinity, carry-over delta, locked_carry_over, locked_prior
      const slot0 = presA.slots[0];
      expect(slot0.absolutePosition).toBe(8);
      expect(slot0.rawLmHead.top1TokenId).toBe(MASK);
      expect(slot0.rawLmHead.maskTokenProb).toBeGreaterThan(0.5);
      expect(slot0.constrainedMdlm.maskTokenExcluded).toBe(true);
      expect(slot0.constrainedMdlm.maskTokenProb).toBe(0);
      expect(slot0.constrainedMdlm.isCarryOverDelta).toBe(true);
      expect(slot0.constrainedMdlm.topK).toEqual([
        { tokenId: 2, tokenLabel: 'c', logProb: 0, prob: 1 },
      ]);
      expect(slot0.selection).toEqual({
        samplingMode: 'locked_carry_over',
        temperature: 0.0,
        selectedTokenId: 2,
        selectedTokenLabel: 'c',
      });
      expect(slot0.confidenceDisplay).toEqual({
        kind: 'locked_prior',
        unscaledConfidence: null,
        displayText: 'Locked (prior step)',
        exceedsThreshold: null,
      });
      expect(slot0.decision).toBe('locked_prior');

      // Slot 1: raw LM head sees MASK as top-1 (>99%), constrained MDLM excludes MASK (0%) and commits token 1 via threshold!
      const slot1 = presA.slots[1];
      expect(slot1.absolutePosition).toBe(9);
      expect(slot1.rawLmHead.top1TokenId).toBe(MASK);
      expect(slot1.rawLmHead.maskTokenProb).toBeGreaterThan(0.99);
      expect(slot1.constrainedMdlm.maskTokenExcluded).toBe(true);
      expect(slot1.constrainedMdlm.maskTokenProb).toBe(0);
      expect(slot1.constrainedMdlm.isCarryOverDelta).toBe(false);
      expect(slot1.constrainedMdlm.topK.some((c) => c.tokenId === MASK)).toBe(false);
      expect(slot1.constrainedMdlm.topK[0].tokenId).toBe(1);
      expect(slot1.selection.samplingMode).toBe('greedy_argmax');
      expect(slot1.selection.selectedTokenId).toBe(1);
      expect(slot1.confidenceDisplay.kind).toBe('unscaled_probability');
      if (slot1.confidenceDisplay.kind === 'unscaled_probability') {
        expect(slot1.confidenceDisplay.unscaledConfidence).toBeGreaterThan(0.7);
        expect(slot1.confidenceDisplay.unscaledConfidence).toBeLessThanOrEqual(1);
        expect(slot1.confidenceDisplay.exceedsThreshold).toBe(true);
      }
      expect(slot1.decision).toBe('committed_threshold');

      // Slot 2: remasked
      const slot2 = presA.slots[2];
      expect(slot2.decision).toBe('remasked');
      expect(slot2.confidenceDisplay.kind).toBe('unscaled_probability');
      if (slot2.confidenceDisplay.kind === 'unscaled_probability') {
        expect(slot2.confidenceDisplay.exceedsThreshold).toBe(false);
      }

      // Scenario B: committed_progress_floor (threshold=0.99, K=4, B=3 -> minCommitFloor=1, commits top-1)
      const rawLogitsB = new Float32Array([
        0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
      ]);
      const unmaskB = runConfidenceUnmaskStep({
        xt: [MASK, MASK, MASK],
        logits: rawLogitsB,
        vocabSize: V,
        maskTokenId: MASK,
        stepIndex: 0,
        stepsPerBlock: 4,
        confidenceThreshold: 0.99,
        temperature: 0.0,
      });
      const presB = buildSamplerStepPresentation({
        rawLogits: rawLogitsB,
        rawProbs: softmaxRowF32(rawLogitsB, V, 3),
        unmaskStep: unmaskB,
        vocab,
        maskTokenId: MASK,
        confidenceThreshold: 0.99,
        temperature: 0.0,
      });
      expect(presB.slots.map((s) => s.decision)).toEqual([
        'remasked',
        'committed_progress_floor',
        'remasked',
      ]);

      // Scenario C: committed_budget_floor (B=4, K=2 -> ceil(4/2)=2 forced; 1 passes threshold + 1 forced by budget_floor!)
      const rawLogitsC = new Float32Array([
        5.0, 0.0, 0.0, 0.0, 0.0, 0.0, // slot 0: high prob > 0.8 -> committed_threshold
        0.0, 1.2, 0.0, 0.0, 0.0, 0.0, // slot 1: medium prob <= 0.8 -> committed_budget_floor
        0.0, 0.0, 0.2, 0.0, 0.0, 0.0, // slot 2: remasked
        0.0, 0.0, 0.0, 0.1, 0.0, 0.0, // slot 3: remasked
      ]);
      const unmaskC = runConfidenceUnmaskStep({
        xt: [MASK, MASK, MASK, MASK],
        logits: rawLogitsC,
        vocabSize: V,
        maskTokenId: MASK,
        stepIndex: 0,
        stepsPerBlock: 2,
        confidenceThreshold: 0.8,
        temperature: 0.0,
      });
      const presC = buildSamplerStepPresentation({
        rawLogits: rawLogitsC,
        rawProbs: softmaxRowF32(rawLogitsC, V, 4),
        unmaskStep: unmaskC,
        vocab,
        maskTokenId: MASK,
        confidenceThreshold: 0.8,
        temperature: 0.0,
      });
      expect(presC.slots.map((s) => s.decision)).toEqual([
        'committed_threshold',
        'committed_budget_floor',
        'remasked',
        'remasked',
      ]);

      // Scenario D: committed_final_step + gumbel_max sampling mode
      const derived = deriveRefinementUniformVariates({
        samplingSeed: 77,
        sessionId: 'sess-final',
        branchId: 'branch-0',
        blockId: 0,
        refinementIndex: 1,
        blockSize: 2,
        vocabSize: V,
      });
      const rawLogitsD = new Float32Array([
        0.1, 0.2, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.2, 0.0, 0.0,
      ]);
      const unmaskD = runConfidenceUnmaskStep({
        xt: [MASK, MASK],
        logits: rawLogitsD,
        vocabSize: V,
        maskTokenId: MASK,
        stepIndex: 1,
        stepsPerBlock: 2,
        confidenceThreshold: 0.99,
        temperature: 0.7,
        uniformVariates: derived.variates,
      });
      const presD = buildSamplerStepPresentation({
        rawLogits: rawLogitsD,
        rawProbs: softmaxRowF32(rawLogitsD, V, 2),
        unmaskStep: unmaskD,
        vocab,
        maskTokenId: MASK,
        confidenceThreshold: 0.99,
        temperature: 0.7,
      });
      expect(presD.slots.map((s) => s.decision)).toEqual([
        'committed_final_step',
        'committed_final_step',
      ]);
      expect(presD.slots.map((s) => s.selection.samplingMode)).toEqual([
        'gumbel_max',
        'gumbel_max',
      ]);

      // assertPresentationSafeNoInfinity detects Infinity, -Infinity, and NaN anywhere
      expect(() => assertPresentationSafeNoInfinity({ ok: 1, nested: [Infinity] })).toThrow(
        RangeError,
      );
      expect(() =>
        assertPresentationSafeNoInfinity({ nested: { val: -Infinity } }),
      ).toThrow(RangeError);
      expect(() =>
        assertPresentationSafeNoInfinity({ arr: new Float32Array([0.5, Number.NaN]) }),
      ).toThrow(RangeError);

      // End-to-end check on real TwoTower multi-step diffusion refinements (including locked slots in r >= 1)
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });
      const block = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        temperature: 0.0,
      });
      for (const ref of block.refinements) {
        const pres = buildSamplerStepPresentation({
          rawLogits: ref.denoiserPass.logits,
          rawProbs: ref.denoiserPass.probs,
          unmaskStep: ref.unmaskStep,
          vocab: TWOTOWER_MINI_VOCAB,
          maskTokenId: model.maskTokenId,
          confidenceThreshold: 0.9999,
          temperature: 0.0,
          absoluteTokenStart: ref.denoiserPass.absoluteTokenStart,
        });
        expect(() => assertPresentationSafeNoInfinity(pres)).not.toThrow();
        const lockedCount = pres.slots.filter((s) => s.decision === 'locked_prior').length;
        expect(lockedCount).toBe(ref.refinementIndex);
      }
    });
  });

  describe('5. Strict Headless & Zero-Math.random Guardrails', () => {
    it('enforces zero DOM, WebGL, scene, render imports, and zero Math.random() across all src/twotower/*.ts', () => {
      const files = [
        'src/twotower/spec.ts',
        'src/twotower/modulation.ts',
        'src/twotower/sampler.ts',
        'src/twotower/parameters.ts',
        'src/twotower/contracts.ts',
        'src/twotower/execution.ts',
        'src/twotower/index.ts',
      ];
      for (const rel of files) {
        const src = readFileSync(resolve(__dirname, '..', rel), 'utf-8');
        const codeOnly = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
        expect(codeOnly, `${rel} must not import scene or render modules`).not.toMatch(
          /from\s+['"][^'"]*(?:\/scene\/|\/render\/)/,
        );
        expect(codeOnly, `${rel} must not reference DOM or WebGL APIs`).not.toMatch(
          /\b(?:window|document|HTMLElement|WebGL2RenderingContext|HTMLCanvasElement)\b/,
        );
        expect(codeOnly, `${rel} must not call Math.random()`).not.toMatch(/Math\.random\s*\(/);
      }
    });
  });
});
