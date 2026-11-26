import { describe, expect, it } from 'vitest';
import { createDefaultUniCoreModel } from '../src/uni/parameters';
import { createXoshiro128ss } from '../src/uni/prng';
import {
  advanceRevisableSemanticTree,
  agesAtTime,
  analyzePrimeDepthFloorMismatch,
  buildBalancedSemanticTree,
  buildSemanticTreeEmbeddings,
  contentIndexToMarkedGridIndex,
  getStagePolarityPair,
  getTreeBallMembership,
  initSemanticTreeState,
  projectCodebookPca3d,
  sampleRepresentatives,
  semanticTreeActivationCounts,
  semanticTreeLevelsForAges,
  stratifiedRandom,
} from '../src/uni/tree';
import {
  UNI_MINI_GEN_IMAGE_END_ID,
  UNI_MINI_GEN_IMAGE_MASK_ID,
  UNI_MINI_GEN_ROW_MARKER_ID,
  UNI_MINI_TREE_BUFFER_BYTES,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MINI_VQ_TREE_DEPTH,
  UNI_SEMANTIC_TREE_ALGORITHM,
  UNI_SEMANTIC_TREE_METRIC,
  UNI_STRATIFIED_DEFAULT_SEED,
} from '../src/uni/types';
import { createUniEmu3VqWeights } from '../src/uni/vqvae';

describe('UNI3 Slice 3A: BalancedSemanticTree & Progressive Self-Refinement', () => {
  const vqWeights = createUniEmu3VqWeights();
  const tree = buildBalancedSemanticTree(
    vqWeights.codebook,
    UNI_MINI_VQ_CODEBOOK_SIZE,
    8,
  );

  describe('buildBalancedSemanticTree & projectCodebookPca3d', () => {
    it('builds exact 3,068 B tree with bijective permutations on real Mini Emu3 codebook', () => {
      expect(tree.algorithm).toBe(UNI_SEMANTIC_TREE_ALGORITHM);
      expect(tree.metric).toBe(UNI_SEMANTIC_TREE_METRIC);
      expect(tree.vocabSize).toBe(256);
      expect(tree.depth).toBe(8);
      expect(tree.latentDim).toBe(8);

      expect(tree.leafToToken.byteLength).toBe(1_024);
      expect(tree.tokenToLeaf.byteLength).toBe(1_024);
      expect(tree.heapRadii.byteLength).toBe(1_020);
      expect(tree.byteLength).toBe(UNI_MINI_TREE_BUFFER_BYTES);
      expect(tree.byteLength).toBe(3_068);

      // Verify exact bijection over 0..255
      const seenTokens = new Set<number>();
      for (let leaf = 0; leaf < 256; leaf++) {
        const tok = tree.leafToToken[leaf];
        expect(tok).toBeGreaterThanOrEqual(0);
        expect(tok).toBeLessThan(256);
        seenTokens.add(tok);
        expect(tree.tokenToLeaf[tok]).toBe(leaf);
      }
      expect(seenTokens.size).toBe(256);
    });

    it('guarantees strict root-to-leaf monotonicity of all 255 heapRadii and positive finite radii', () => {
      expect(tree.heapRadii.length).toBe(255);
      for (let i = 0; i < 255; i++) {
        expect(Number.isFinite(tree.heapRadii[i])).toBe(true);
        expect(tree.heapRadii[i]).toBeGreaterThanOrEqual(0);
        expect(tree.heapRadii[i]).toBeLessThanOrEqual(4.0001); // unit-norm vectors: max ||u-v||^2 <= 4
      }
      // Root diameter must be strictly positive on non-degenerate codebook
      expect(tree.heapRadii[0]).toBeGreaterThan(2.0);

      // Every internal node i in 0..126 has children 2*i+1 and 2*i+2 in 0..254
      for (let parent = 0; parent < 127; parent++) {
        const left = 2 * parent + 1;
        const right = 2 * parent + 2;
        expect(tree.heapRadii[left]).toBeLessThanOrEqual(tree.heapRadii[parent]);
        expect(tree.heapRadii[right]).toBeLessThanOrEqual(tree.heapRadii[parent]);
      }

      // Deterministic rebuild yields 0-ULP identical buffers
      const rebuilt = buildBalancedSemanticTree(vqWeights.codebook, 256, 8);
      expect(Array.from(rebuilt.leafToToken)).toEqual(Array.from(tree.leafToToken));
      expect(Array.from(rebuilt.tokenToLeaf)).toEqual(Array.from(tree.tokenToLeaf));
      expect(Array.from(rebuilt.heapRadii)).toEqual(Array.from(tree.heapRadii));
    });

    it('handles degenerate zero-variance codebook deterministically via standard basis fallback', () => {
      const flatCodebook = new Float32Array(16 * 4);
      for (let i = 0; i < 16; i++) {
        flatCodebook[i * 4] = 1.0; // all identical unit vectors -> zero covariance
      }
      const degTree = buildBalancedSemanticTree(flatCodebook, 16, 4);
      expect(degTree.depth).toBe(4);
      expect(degTree.byteLength).toBe(16 * 4 + 16 * 4 + 15 * 4);
      // Tie-breaks by ascending token ID -> identity permutation
      expect(Array.from(degTree.leafToToken)).toEqual(
        Array.from({ length: 16 }, (_, i) => i),
      );
      for (let i = 0; i < degTree.heapRadii.length; i++) {
        expect(degTree.heapRadii[i]).toBe(0);
      }
    });

    it('projects codebook deterministically to 3D PCA coordinates with decreasing variance', () => {
      const coords = projectCodebookPca3d(vqWeights.codebook, 256, 8);
      expect(coords.length).toBe(256 * 3);

      const means = [0, 0, 0];
      for (let i = 0; i < 256; i++) {
        for (let k = 0; k < 3; k++) {
          const val = coords[i * 3 + k];
          expect(Number.isFinite(val)).toBe(true);
          means[k] += val;
        }
      }
      for (let k = 0; k < 3; k++) {
        expect(Math.abs(means[k] / 256)).toBeLessThan(1e-5);
      }

      const vars = [0, 0, 0];
      for (let i = 0; i < 256; i++) {
        for (let k = 0; k < 3; k++) {
          const val = coords[i * 3 + k];
          vars[k] += val * val;
        }
      }
      expect(vars[0]).toBeGreaterThanOrEqual(vars[1]);
      expect(vars[1]).toBeGreaterThanOrEqual(vars[2]);
      expect(vars[2]).toBeGreaterThan(0);

      // 0-ULP deterministic replay
      const coords2 = projectCodebookPca3d(vqWeights.codebook, 256, 8);
      expect(Array.from(coords2)).toEqual(Array.from(coords));
    });

    it('rejects non-power-of-two vocabSize or mismatched codebook length', () => {
      expect(() => buildBalancedSemanticTree(new Float32Array(24), 3, 8)).toThrow(
        RangeError,
      );
      expect(() => buildBalancedSemanticTree(new Float32Array(10), 256, 8)).toThrow(
        RangeError,
      );
    });
  });

  describe('U26 Dedicated Production D=17 Integer Floor-Mismatch Verification', () => {
    it('reproduces exact 2x wider candidate subtrees at every internal stage for D=17, K=4', () => {
      const report = analyzePrimeDepthFloorMismatch(17, 4);
      expect(report.treeDepth).toBe(17);
      expect(report.refinementSteps).toBe(4);
      expect(report.hasMismatch).toBe(true);
      expect(report.totalInternalStages).toBe(3);
      expect(report.mismatchedStageCount).toBe(3);
      expect(report.rows.length).toBe(3);

      expect(report.rows.map((r) => r.stageIndex)).toEqual([1, 2, 3]);
      expect(report.rows.map((r) => r.trainAge)).toEqual([3, 2, 1]);
      expect(report.rows.map((r) => r.trainLevel)).toEqual([5, 9, 13]);
      expect(report.rows.map((r) => r.inferLevel)).toEqual([4, 8, 12]);
      expect(report.rows.map((r) => r.levelDelta)).toEqual([1, 1, 1]);
      expect(report.rows.map((r) => r.trainBallSize)).toEqual([4096, 256, 16]);
      expect(report.rows.map((r) => r.inferBallSize)).toEqual([8192, 512, 32]);
      expect(report.rows.map((r) => r.ballSizeRatio)).toEqual([2, 2, 2]);
    });

    it('verifies D=17 mismatch across K=8 (7/7), K=16 (15/15), and K=1 (0/0)', () => {
      const r8 = analyzePrimeDepthFloorMismatch(17, 8);
      expect(r8.hasMismatch).toBe(true);
      expect(r8.totalInternalStages).toBe(7);
      expect(r8.mismatchedStageCount).toBe(7);
      expect(r8.rows.every((r) => r.levelDelta === 1 && r.ballSizeRatio === 2)).toBe(
        true,
      );

      const r16 = analyzePrimeDepthFloorMismatch(17, 16);
      expect(r16.hasMismatch).toBe(true);
      expect(r16.totalInternalStages).toBe(15);
      expect(r16.mismatchedStageCount).toBe(15);
      expect(r16.rows.every((r) => r.levelDelta === 1 && r.ballSizeRatio === 2)).toBe(
        true,
      );

      const r1 = analyzePrimeDepthFloorMismatch(17, 1);
      expect(r1.hasMismatch).toBe(false);
      expect(r1.totalInternalStages).toBe(0);
      expect(r1.mismatchedStageCount).toBe(0);
      expect(r1.rows).toEqual([]);
    });

    it('verifies exact 0 mismatches on power-of-two miniature D=8 for all K in {1, 2, 4, 8}', () => {
      for (const k of [1, 2, 4, 8]) {
        const report = analyzePrimeDepthFloorMismatch(UNI_MINI_VQ_TREE_DEPTH, k);
        expect(report.hasMismatch).toBe(false);
        expect(report.mismatchedStageCount).toBe(0);
        expect(report.totalInternalStages).toBe(k - 1);
        for (const row of report.rows) {
          expect(row.trainLevel).toBe(row.inferLevel);
          expect(row.levelDelta).toBe(0);
          expect(row.trainBallSize).toBe(row.inferBallSize);
          expect(row.ballSizeRatio).toBe(1);
        }
      }
    });
  });

  describe('semanticTreeLevelsForAges & U9 Polarity Pair', () => {
    it('reduces identically to binary MDM at N=1', () => {
      const res = semanticTreeLevelsForAges([-1, 0, 1], 8, 1);
      expect(Array.from(res.levels)).toEqual([0, 0, 8]);
      expect(Array.from(res.noiseWeights)).toEqual([1.0, 1.0, 0.0]);
      expect(Array.from(res.ballSizes)).toEqual([256, 256, 1]);
    });

    it('maps stages across N in {2, 4, 8} accurately (including N=4 balls [256, 64, 16, 4, 1])', () => {
      const r2 = semanticTreeLevelsForAges([-1, 0, 1, 2], 8, 2);
      expect(Array.from(r2.levels)).toEqual([0, 0, 4, 8]);
      expect(Array.from(r2.noiseWeights)).toEqual([1.0, 1.0, 0.5, 0.0]);
      expect(Array.from(r2.ballSizes)).toEqual([256, 256, 16, 1]);

      const r4 = semanticTreeLevelsForAges([-1, 0, 1, 2, 3, 4], 8, 4);
      expect(Array.from(r4.levels)).toEqual([0, 0, 2, 4, 6, 8]);
      expect(Array.from(r4.noiseWeights)).toEqual([1.0, 1.0, 0.75, 0.5, 0.25, 0.0]);
      expect(Array.from(r4.ballSizes)).toEqual([256, 256, 64, 16, 4, 1]);

      const r8 = semanticTreeLevelsForAges([0, 1, 2, 3, 4, 5, 6, 7, 8], 8, 8);
      expect(Array.from(r8.levels)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
      expect(Array.from(r8.ballSizes)).toEqual([256, 128, 64, 32, 16, 8, 4, 2, 1]);
      expect(r8.noiseWeights[0]).toBe(1.0);
      expect(r8.noiseWeights[4]).toBe(0.5);
      expect(r8.noiseWeights[8]).toBe(0.0);
    });

    it('computes U9 reversed code-stage vs paper-age polarity pair accurately', () => {
      const unstarted = getStagePolarityPair(-1, 8, 4);
      expect(unstarted).toEqual({
        codeStage: -1,
        paperAge: 4,
        level: 0,
        ballSize: 256,
        noiseWeight: 1.0,
        isCommitted: false,
      });

      const mid = getStagePolarityPair(1, 8, 4);
      expect(mid).toEqual({
        codeStage: 1,
        paperAge: 3,
        level: 2,
        ballSize: 64,
        noiseWeight: 0.75,
        isCommitted: false,
      });

      const committed = getStagePolarityPair(4, 8, 4);
      expect(committed).toEqual({
        codeStage: 4,
        paperAge: 0,
        level: 8,
        ballSize: 1,
        noiseWeight: 0.0,
        isCommitted: true,
      });
    });
  });

  describe('getTreeBallMembership & sampleRepresentatives', () => {
    it('returns exact nested subtree ball memberships and heap nodes across levels 0..8', () => {
      const token = 73;
      const rootBall = getTreeBallMembership(tree, token, 0);
      expect(rootBall.shift).toBe(8);
      expect(rootBall.ballStartLeaf).toBe(0);
      expect(rootBall.ballSize).toBe(256);
      expect(rootBall.heapNodeIndex).toBe(0);
      expect(rootBall.radius).toBe(tree.heapRadii[0]);
      expect(rootBall.memberTokenIds.length).toBe(256);

      let prevMembers = new Set(rootBall.memberTokenIds);
      let prevRadius = rootBall.radius;

      for (let level = 1; level <= 8; level++) {
        const ball = getTreeBallMembership(tree, token, level);
        expect(ball.ballSize).toBe(1 << (8 - level));
        expect(ball.memberTokenIds.length).toBe(ball.ballSize);
        expect(ball.memberTokenIds).toContain(token);
        expect(ball.radius).toBeLessThanOrEqual(prevRadius);

        // Nested containment: every member at level L is inside level L-1
        for (const m of ball.memberTokenIds) {
          expect(prevMembers.has(m)).toBe(true);
        }

        if (level < 8) {
          expect(ball.heapNodeIndex).not.toBeNull();
          expect(ball.radius).toBe(tree.heapRadii[ball.heapNodeIndex!]);
        } else {
          expect(ball.heapNodeIndex).toBeNull();
          expect(ball.radius).toBe(0);
          expect(Array.from(ball.memberTokenIds)).toEqual([token]);
        }

        prevMembers = new Set(ball.memberTokenIds);
        prevRadius = ball.radius;
      }
    });

    it('confines sampleRepresentatives strictly inside proposal subtree and replays bit-identically', () => {
      const proposals = [0, 17, 42, 128, 255, 99];
      const levels = [0, 2, 4, 6, 8, 8];
      const randBits = [0x12345678, 0xabcdef01, 0x0000000f, 0xffffffff, 0x99999999, 0];

      const reps1 = sampleRepresentatives(tree, proposals, levels, randBits);
      const reps2 = sampleRepresentatives(tree, proposals, levels, randBits);
      expect(Array.from(reps1)).toEqual(Array.from(reps2));

      for (let i = 0; i < proposals.length; i++) {
        const ball = getTreeBallMembership(tree, proposals[i], levels[i]);
        expect(ball.memberTokenIds).toContain(reps1[i]);
        if (levels[i] === tree.depth) {
          // Committed level ignores randBits and returns exact proposal
          expect(reps1[i]).toBe(proposals[i]);
        }
      }
    });
  });

  describe('buildSemanticTreeEmbeddings (U10)', () => {
    it('matches exact U10 equation for eta = 0 (clean copy) and eta > 0 (additive mask)', () => {
      const core = createDefaultUniCoreModel();
      const genEmb = core.genEmbedding;
      const d = 8;
      const reps = [12, 200, UNI_MINI_GEN_ROW_MARKER_ID, 5];
      const noiseWeights = new Float32Array([0.0, 0.75, 0.0, 1.0]);

      const emb = buildSemanticTreeEmbeddings(
        reps,
        noiseWeights,
        genEmb,
        d,
        UNI_MINI_GEN_IMAGE_MASK_ID,
      );
      expect(emb.length).toBe(reps.length * d);

      for (let i = 0; i < reps.length; i++) {
        const rep = reps[i];
        const eta = noiseWeights[i];
        for (let c = 0; c < d; c++) {
          const baseVal = genEmb[rep * d + c];
          const maskVal = genEmb[UNI_MINI_GEN_IMAGE_MASK_ID * d + c];
          const expected =
            eta === 0 ? baseVal : Math.fround(baseVal + Math.fround(eta * maskVal));
          expect(emb[i * d + c]).toBe(expected);
        }
      }
    });
  });

  describe('semanticTreeActivationCounts (U12), agesAtTime (Eq. 16) & stratifiedRandom', () => {
    it('matches U12 shift=5.0 quotas and N-1 flush rounds for both prod (64+3=67) and mini (8+3=11)', () => {
      const prod = semanticTreeActivationCounts(4096, 64, 4, 5.0, 0);
      expect(prod.activationSteps).toBe(64);
      expect(prod.flushRounds).toBe(3);
      expect(prod.totalRounds).toBe(67);
      expect(prod.activationCounts.length).toBe(67);
      expect(prod.cumulativeActivations[63]).toBe(4096);
      expect(prod.cumulativeActivations[66]).toBe(4096);
      expect(Array.from(prod.activationCounts.slice(64))).toEqual([0, 0, 0]);
      // Flipped concave shift=5.0 starts slow and ends large
      expect(prod.activationCounts[0]).toBeLessThan(prod.activationCounts[63]);
      for (let r = 0; r < 64; r++) {
        expect(prod.activationCounts[r]).toBeGreaterThanOrEqual(1);
      }

      const mini = semanticTreeActivationCounts(16, 8, 4, 5.0, 0);
      expect(mini.activationSteps).toBe(8);
      expect(mini.flushRounds).toBe(3);
      expect(mini.totalRounds).toBe(11);
      expect(Array.from(mini.activationCounts)).toEqual([
        1, 1, 1, 1, 1, 2, 4, 5, 0, 0, 0,
      ]);
      expect(mini.cumulativeActivations[10]).toBe(16);

      // Binary MDM N=1 has 0 flush rounds
      const mdm = semanticTreeActivationCounts(16, 8, 1, 5.0, 0);
      expect(mdm.flushRounds).toBe(0);
      expect(mdm.totalRounds).toBe(8);

      // Non-zero refinementGap adds N * gap flush rounds
      const gapped = semanticTreeActivationCounts(16, 8, 4, 5.0, 2);
      expect(gapped.flushRounds).toBe(3 + 8);
      expect(gapped.totalRounds).toBe(19);
    });

    it('matches Paper Eq. 16 agesAtTime boundary points exactly', () => {
      const tau = [0.0, 0.25, 0.5, 0.75];
      const eps = 0.25;
      const K = 17;

      expect(Array.from(agesAtTime(0.0, tau, eps, K))).toEqual([0, 0, 0, 0]);
      expect(Array.from(agesAtTime(0.25, tau, eps, K))).toEqual([17, 0, 0, 0]);
      expect(Array.from(agesAtTime(0.375, tau, eps, K))).toEqual([17, 8, 0, 0]);
      expect(Array.from(agesAtTime(0.5, tau, eps, K))).toEqual([17, 17, 0, 0]);
      expect(Array.from(agesAtTime(1.0, tau, eps, K))).toEqual([17, 17, 17, 17]);
    });

    it('generates deterministic quadtree-stratified permutations for stratifiedRandom(4, 42)', () => {
      const order1 = stratifiedRandom(4, UNI_STRATIFIED_DEFAULT_SEED);
      const order2 = stratifiedRandom(4, 42);
      expect(order1.length).toBe(16);
      expect(Array.from(order1)).toEqual(Array.from(order2));

      // Valid permutation of 0..15
      expect(new Set(order1).size).toBe(16);
      for (const idx of order1) {
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThan(16);
      }

      // First 4 entries hit all 4 top-level 2x2 quadrants (1 per quadrant)
      const quadrants = new Set<number>();
      for (let i = 0; i < 4; i++) {
        const flat = order1[i];
        const r = Math.floor(flat / 4);
        const c = flat % 4;
        const q = ((r >= 2 ? 1 : 0) << 1) | (c >= 2 ? 1 : 0);
        quadrants.add(q);
      }
      expect(quadrants.size).toBe(4);

      // Works for n = 1 and n = 8 as well
      expect(Array.from(stratifiedRandom(1, 42))).toEqual([0]);
      const order8 = stratifiedRandom(8, 42);
      expect(order8.length).toBe(64);
      expect(new Set(order8).size).toBe(64);
    });
  });

  describe('initSemanticTreeState & advanceRevisableSemanticTree', () => {
    it('enforces strict immutability, marker preservation, finalization lock, and padding lock', () => {
      const initialContent = Int32Array.from({ length: 16 }, (_, i) => (i * 13) % 256);
      // Lock top row (4 cells) and bottom row (4 cells) like 16x8 letterbox padding
      const lockedMask = new Uint8Array([
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1,
      ]);
      const sourceCodes = Int32Array.from({ length: 16 }, (_, i) => 100 + i);

      let state = initSemanticTreeState({
        rows: 4,
        contentCols: 4,
        includeMarkers: true,
        depth: 8,
        refinementSteps: 4,
        initialContentCodes: initialContent,
        lockedContentMask: lockedMask,
        lockedSourceCodes: sourceCodes,
      });

      expect(state.stages.length).toBe(24);
      // Verify markers on cols 4..5
      for (let r = 0; r < 4; r++) {
        const expectedMarker =
          r < 3 ? UNI_MINI_GEN_ROW_MARKER_ID : UNI_MINI_GEN_IMAGE_END_ID;
        for (let m = 4; m < 6; m++) {
          const g = r * 6 + m;
          expect(state.isMarker[g]).toBe(1);
          expect(state.stages[g]).toBe(4);
          expect(state.levels[g]).toBe(8);
          expect(state.noiseWeights[g]).toBe(0.0);
          expect(state.representatives[g]).toBe(expectedMarker);
        }
      }

      // Verify locked border cells start committed with source codes
      for (let c = 0; c < 16; c++) {
        const g = contentIndexToMarkedGridIndex(c, 4, 6);
        if (lockedMask[c] === 1) {
          expect(state.stages[g]).toBe(4);
          expect(state.levels[g]).toBe(8);
          expect(state.noiseWeights[g]).toBe(0.0);
          expect(state.representatives[g]).toBe(sourceCodes[c]);
        } else {
          expect(state.stages[g]).toBe(-1);
          expect(state.levels[g]).toBe(0);
          expect(state.noiseWeights[g]).toBe(1.0);
          expect(state.representatives[g]).toBe(initialContent[c]);
        }
      }

      const schedule = semanticTreeActivationCounts(16, 8, 4, 5.0, 0);
      const unmaskOrder = stratifiedRandom(4, 42);
      const rng = createXoshiro128ss(0x554e4933);

      let cursor = 0;
      const firstCommitByContent = new Map<number, number>();

      for (let round = 0; round < schedule.totalRounds; round++) {
        const count = schedule.activationCounts[round];
        const activated = unmaskOrder.slice(cursor, cursor + count);
        cursor += count;

        // Distinct proposals every round to test finalization & padding immutability
        const proposals = Int32Array.from(
          { length: 16 },
          (_, c) => (round * 29 + c * 7 + 3) % 256,
        );
        const randBits = Int32Array.from({ length: 16 }, () => rng.nextUint32());

        const beforeStages = Array.from(state.stages);
        const beforeReps = Array.from(state.representatives);

        const nextState = advanceRevisableSemanticTree(
          tree,
          state,
          proposals,
          activated,
          randBits,
          lockedMask,
        );

        // Strict non-mutation of previous state
        expect(Array.from(state.stages)).toEqual(beforeStages);
        expect(Array.from(state.representatives)).toEqual(beforeReps);

        // Markers and locked cells NEVER change
        for (let r = 0; r < 4; r++) {
          const expectedMarker =
            r < 3 ? UNI_MINI_GEN_ROW_MARKER_ID : UNI_MINI_GEN_IMAGE_END_ID;
          expect(nextState.representatives[r * 6 + 4]).toBe(expectedMarker);
          expect(nextState.representatives[r * 6 + 5]).toBe(expectedMarker);
        }
        for (let c = 0; c < 16; c++) {
          const g = contentIndexToMarkedGridIndex(c, 4, 6);
          if (lockedMask[c] === 1) {
            expect(nextState.stages[g]).toBe(4);
            expect(nextState.representatives[g]).toBe(sourceCodes[c]);
          } else if (nextState.stages[g] === 4) {
            if (!firstCommitByContent.has(c)) {
              firstCommitByContent.set(c, nextState.representatives[g]);
              // On the round a cell commits, its representative equals its clean proposal
              expect(nextState.representatives[g]).toBe(proposals[c]);
            } else {
              // Once committed, NEVER changes in subsequent rounds
              expect(nextState.representatives[g]).toBe(firstCommitByContent.get(c));
            }
          }
        }

        state = nextState;
      }

      // After all 11 rounds (8 activation + 3 flush), EVERY cell is finalized at stage 4
      for (let g = 0; g < 24; g++) {
        expect(state.stages[g]).toBe(4);
        expect(state.levels[g]).toBe(8);
        expect(state.noiseWeights[g]).toBe(0.0);
      }
      expect(firstCommitByContent.size).toBe(8); // 8 editable interior cells
    });

    it('accepts marked-grid [24] proposals/randBits and commits in 1 step for N=1 binary MDM', () => {
      const state0 = initSemanticTreeState({
        rows: 4,
        contentCols: 4,
        includeMarkers: true,
        depth: 8,
        refinementSteps: 1,
        seed: 7,
      });

      const proposals24 = new Int32Array(24);
      const randBits24 = new Int32Array(24);
      for (let g = 0; g < 24; g++) {
        proposals24[g] = (g * 11 + 5) % 256;
        randBits24[g] = 0xdeadbeef;
      }

      const activated = [0, 5, 10, 15];
      const state1 = advanceRevisableSemanticTree(
        tree,
        state0,
        proposals24,
        activated,
        randBits24,
      );

      for (let c = 0; c < 16; c++) {
        const g = contentIndexToMarkedGridIndex(c, 4, 6);
        if (activated.includes(c)) {
          expect(state1.stages[g]).toBe(1);
          expect(state1.levels[g]).toBe(8);
          expect(state1.noiseWeights[g]).toBe(0.0);
          expect(state1.representatives[g]).toBe(proposals24[g]);
        } else {
          expect(state1.stages[g]).toBe(-1);
          expect(state1.levels[g]).toBe(0);
          expect(state1.noiseWeights[g]).toBe(1.0);
          expect(state1.representatives[g]).toBe(state0.representatives[g]);
        }
      }
    });
  });
});
