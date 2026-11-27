import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  fmtSigned as mathFmtSigned,
  type LayerFormulaReadout as MathLayerFormulaReadout,
  type StepperStepDef as MathStepperStepDef,
} from '../src/math/formulaFormat';
import {
  fmtSigned as sceneFmtSigned,
  type LayerFormulaReadout as SceneLayerFormulaReadout,
  type StepperStepDef as SceneStepperStepDef,
} from '../src/scene/sheetInspector';
import {
  createDefaultHybridModel,
  createHybridModelParameters,
  DEFAULT_MAX_RETAINED_WEIGHT_BAYS,
  type MoeBayWeights,
} from '../src/math/hybridFlow';
import { runMambaSim } from '../src/math/mambaSim';
import { describeMoeCellFormula, runMoeSim, type MoeTensorKey } from '../src/math/moeSim';
import {
  computeDecodeStateDigest,
  createDecodeSession,
  createExternalDecodeCacheState,
  createZeroDecodeCacheState,
  DEFAULT_SESSION_MAX_BRANCHES,
  DEFAULT_SESSION_MAX_UNIQUE_STEPS,
  deleteDecodeSessionBranch,
  getActiveDecodeBranch,
  getActiveDecodeStep,
  MAX_INCREMENTAL_STEP_TOKENS,
  measureDecodeSessionRetainedBytes,
  navigateDecodeSession,
  pruneDecodeSessionBranches,
  pruneInactiveStepTraces,
  runIncrementalDecodeStep,
  stepDecodeSession,
  stepDecodeSessionGreedy,
  switchDecodeSessionBranch,
  validateDecodeCacheState,
} from '../src/math/incrementalDecode';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

const REPO_ROOT = resolve(__dirname, '..');

function listTsFilesRecursive(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...listTsFilesRecursive(full));
    } else if (entry.isFile() && entry.name.endsWith('.ts')) {
      out.push(full);
    }
  }
  return out;
}

describe('Stage 1a Pre-Release Hardening: Math & State Contracts (RB-7, PR-4..PR-9)', () => {
  it('RB-7: tsconfig.json explicitly includes src/math/**/*.ts and covers every src/**/*.ts file', () => {
    const tsconfig = JSON.parse(readFileSync(resolve(REPO_ROOT, 'tsconfig.json'), 'utf-8')) as {
      include: string[];
    };
    expect(
      tsconfig.include.includes('src/**/*.ts') || tsconfig.include.includes('src/math/**/*.ts'),
    ).toBe(true);
    expect(tsconfig.include).toContain('playwright.config.ts');

    const allSrcTs = listTsFilesRecursive(resolve(REPO_ROOT, 'src')).map((abs) =>
      abs.slice(REPO_ROOT.length + 1).replace(/\\/g, '/'),
    );
    expect(allSrcTs.length).toBeGreaterThan(0);

    const coveredByInclude = (relPath: string): boolean =>
      tsconfig.include.some((pattern) => {
        if (pattern === relPath) return true;
        if (pattern.endsWith('/**/*.ts')) {
          const prefix = pattern.slice(0, -'/**/*.ts'.length) + '/';
          return relPath.startsWith(prefix) && relPath.endsWith('.ts');
        }
        return false;
      });

    for (const relPath of allSrcTs) {
      expect(coveredByInclude(relPath), `${relPath} must be covered by tsconfig.include`).toBe(true);
    }
  });

  it('PR-4: src/math/*.ts has zero imports from src/scene or src/render and preserves fmtSigned & type re-exports', () => {
    const mathFiles = listTsFilesRecursive(resolve(REPO_ROOT, 'src/math'));
    expect(mathFiles.length).toBeGreaterThanOrEqual(7);

    // Static check: disallow static imports, re-exports, or dynamic imports into scene/ or render/
    const forbiddenImportRe =
      /(?:from\s+['"]|import\s*\(\s*['"])(?:\.\.\/|\.\.?\/[^'"]*\/)?(?:scene|render)(?:\/|['"])/;

    for (const file of mathFiles) {
      const source = readFileSync(file, 'utf-8');
      expect(forbiddenImportRe.test(source), `${file} must not import from scene/ or render/`).toBe(
        false,
      );
    }

    // Backward-compatible re-export identity and exact sign formatting behavior
    expect(sceneFmtSigned).toBe(mathFmtSigned);
    expect(mathFmtSigned(0, 3)).toBe('+0.000');
    expect(mathFmtSigned(-0, 3)).toBe('+0.000');
    expect(mathFmtSigned(-0.0001, 3)).toBe('+0.000');
    expect(mathFmtSigned(1.2346, 3)).toBe('+1.235');
    expect(mathFmtSigned(-1.2346, 3)).toBe('−1.235');
    expect(mathFmtSigned(Number.POSITIVE_INFINITY, 3)).toBe('+∞');
    expect(mathFmtSigned(Number.NEGATIVE_INFINITY, 3)).toBe('−∞');

    // Compile-time equivalence check between math and scene re-exported interfaces
    const readout: SceneLayerFormulaReadout = {
      tensorLabel: 'T',
      coordLabel: 'c',
      headBadge: null,
      value: 0,
      isMasked: false,
      valueText: '+0.000',
      equation: 'e',
      subEquation: 's',
    };
    const mathReadout: MathLayerFormulaReadout = readout;
    expect(mathReadout.valueText).toBe('+0.000');

    const stepDef: SceneStepperStepDef = {
      index: 0,
      shortLabel: '1',
      title: 'Step 1',
      summary: 'Summary',
    };
    const mathStepDef: MathStepperStepDef = stepDef;
    expect(mathStepDef.index).toBe(0);
  });

  it('PR-5: describeMoeCellFormula prints absolute token coordinates (startTokenOffset + t) across all cell kinds', () => {
    const startOffset = 7;
    const sim = runMoeSim(2, {
      startTokenOffset: startOffset,
      tokenIds: [10, 11],
      tokenLabels: ['million', 'tokens'],
    });

    const activeExpT0 = sim.topk_indices[0];
    const activeExpT1 = sim.topk_indices[6];

    // Check all 8 newly fixed kinds + the other token-displaying kinds for both t=0 (abs t7) and t=1 (abs t8)
    const cases: readonly {
      readonly tensor: MoeTensorKey;
      readonly field: 'coordLabel' | 'equation' | 'both';
    }[] = [
      { tensor: 'input', field: 'both' },
      { tensor: 'rmsnorm', field: 'both' },
      { tensor: 'router_bank', field: 'coordLabel' },
      { tensor: 'topk_weights', field: 'coordLabel' },
      { tensor: 'routed_w_up', field: 'equation' },
      { tensor: 'routed_pre', field: 'both' },
      { tensor: 'routed_act', field: 'coordLabel' },
      { tensor: 'routed_sum', field: 'both' },
      { tensor: 'shared_w_up', field: 'equation' },
      { tensor: 'shared_act', field: 'both' },
      { tensor: 'shared_w_down', field: 'equation' },
      { tensor: 'shared_out', field: 'both' },
      { tensor: 'moe_out', field: 'both' },
      { tensor: 'residual', field: 'both' },
    ];

    for (const { tensor, field } of cases) {
      for (const localT of [0, 1]) {
        const absT = startOffset + localT;
        const expId = localT === 0 ? activeExpT0 : activeExpT1;
        const isTokenRowTensor =
          tensor === 'input' ||
          tensor === 'rmsnorm' ||
          tensor === 'routed_sum' ||
          tensor === 'shared_out' ||
          tensor === 'moe_out' ||
          tensor === 'residual';
        const formula = describeMoeCellFormula(
          sim,
          {
            tensor,
            row: isTokenRowTensor ? localT : 0,
            col: 0,
            tokenIndex: localT,
            expertId: expId,
            rankSlot: 0,
          },
          localT,
          expId,
        );

        const expectedTag = `t${absT}`;
        const rawLocalTag = `t${localT}`;
        if (field === 'coordLabel' || field === 'both') {
          expect(formula.coordLabel, `${tensor} coordLabel @ localT=${localT}`).toContain(
            expectedTag,
          );
          expect(formula.coordLabel, `${tensor} coordLabel must not print raw ${rawLocalTag}`).not.toContain(
            rawLocalTag,
          );
        }
        if (field === 'equation' || field === 'both') {
          expect(formula.equation, `${tensor} equation @ localT=${localT}`).toContain(expectedTag);
          expect(formula.equation, `${tensor} equation must not print raw ${rawLocalTag}`).not.toContain(
            rawLocalTag,
          );
        }
      }
    }
  });

  it('PR-6: retains all 52 miniature bays by default with stable buffer identity and fixes bounded LRU eviction', () => {
    // 1. Default 52-bay weight retention (~1.14 MiB) & stable buffer identity across decode steps
    expect(DEFAULT_MAX_RETAINED_WEIGHT_BAYS).toBe(52);
    const model = createDefaultHybridModel();
    expect(model.config.maxRetainedWeightBays).toBe(52);

    const firstPass = new Map<number, ReturnType<typeof model.getBayWeights>>();
    for (let bayId = 1; bayId <= 52; bayId++) {
      firstPass.set(bayId, model.getBayWeights(bayId));
    }
    // All 52 miniature bays retained = ~1.14 MiB (1,196,768 B)
    expect(model.retainedWeightBytes()).toBe(1_196_768);

    // Every subsequent access returns the exact same cached object & typed array identity
    for (let bayId = 1; bayId <= 52; bayId++) {
      expect(model.getBayWeights(bayId)).toBe(firstPass.get(bayId));
    }

    // No duplicate trace-owned weight allocations across decode steps (both pinned & non-pinned bays)
    const step0 = runIncrementalDecodeStep(model, { newTokenIds: [0, 1, 2, 3], prevState: null });
    const step1 = runIncrementalDecodeStep(model, { newTokenIds: [4], prevState: step0.nextState });
    const bay11Weights = model.getBayWeights(11) as MoeBayWeights; // Bay 11 is non-pinned MoE ('E')
    expect(step0.getMoeTrace(11).W_up).toBe(bay11Weights.W_up);
    expect(step1.getMoeTrace(11).W_up).toBe(bay11Weights.W_up);
    expect(step0.getMoeTrace(11).W_down).toBe(step1.getMoeTrace(11).W_down);
    expect(model.retainedWeightBytes()).toBe(1_196_768);

    // 2. Small-capacity LRU (capacity = 8, default 6 pinned bays [1, 2, 6, 42, 43, 52])
    const bounded = createHybridModelParameters({ maxRetainedWeightBays: 8 });
    expect(bounded.config.maxRetainedWeightBays).toBe(8);

    const pinnedBays = [1, 2, 6, 42, 43, 52];
    const pinnedRefs = new Map(pinnedBays.map((b) => [b, bounded.getBayWeights(b)]));
    const bay7First = bounded.getBayWeights(7);
    const bay8First = bounded.getBayWeights(8);
    // Cache is now full at 8: 6 pinned + [7, 8]
    expect(bounded.getBayWeights(7)).toBe(bay7First);
    expect(bounded.getBayWeights(8)).toBe(bay8First);

    // Accessing non-pinned bay 9 MUST insert bay 9 into weightCache and evict oldest non-pinned bay 7
    const bay9First = bounded.getBayWeights(9);
    expect(bounded.getBayWeights(9)).toBe(bay9First); // cached hit!
    expect(bounded.getBayWeights(8)).toBe(bay8First); // still cached (now refreshed to MRU!)
    for (const b of pinnedBays) {
      expect(bounded.getBayWeights(b)).toBe(pinnedRefs.get(b)); // all pinned bays preserved!
    }

    // Accessing bay 10 must evict LRU non-pinned bay 9 (NOT bay 8 which was just refreshed)
    const bay10First = bounded.getBayWeights(10);
    expect(bounded.getBayWeights(10)).toBe(bay10First);
    expect(bounded.getBayWeights(8)).toBe(bay8First);
    // Bay 7 was evicted earlier, so re-requesting it regenerates a new object with bit-identical values
    const bay7Regen = bounded.getBayWeights(7);
    expect(bay7Regen).not.toBe(bay7First);
    if (bay7First.kind === 'E' && bay7Regen.kind === 'E') {
      assertBitExactFiniteArray(bay7Regen.W_router, bay7First.W_router, 'evicted bay 7 values');
    }

    // 3. Capacity smaller than pinned count (capacity = 3, 6 pinned): still inserts & caches requested non-pinned bay
    const tiny = createHybridModelParameters({ maxRetainedWeightBays: 3 });
    tiny.getBayWeights(1);
    tiny.getBayWeights(2);
    tiny.getBayWeights(6);
    const tinyBay9 = tiny.getBayWeights(9);
    expect(tiny.getBayWeights(9)).toBe(tinyBay9);
  });

  it('PR-7/PR-8: includes length-prefixed prefixTokenLabels in digest, rejects tampered states, hardens external seeds and Mamba finiteness', () => {
    const model = createDefaultHybridModel();
    const step0 = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1],
      newTokenLabels: ['a', 'bc'],
      prevState: null,
    });
    const step0Alt = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1],
      newTokenLabels: ['ab', 'c'],
      prevState: null,
    });

    // Unambiguous length-prefixed prefixTokenLabels disambiguation
    expect(step0.nextState.stateDigest).not.toBe(step0Alt.nextState.stateDigest);
    expect(computeDecodeStateDigest(step0.nextState)).toBe(step0.nextState.stateDigest);

    // Same labels + state produce identical digest
    const step0Dup = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1],
      newTokenLabels: ['a', 'bc'],
      prevState: null,
    });
    expect(step0Dup.nextState.stateDigest).toBe(step0.nextState.stateDigest);

    // Digest tampering rejections in validateDecodeCacheState & runIncrementalDecodeStep
    expect(() =>
      validateDecodeCacheState(model, { ...step0.nextState, stateDigest: '0000000000000000' }),
    ).toThrow(/stateDigest mismatch/);

    // Swapping prefixTokenLabels without updating digest must throw
    expect(() =>
      validateDecodeCacheState(model, {
        ...step0.nextState,
        prefixTokenLabels: ['ab', 'c'],
      }),
    ).toThrow(/stateDigest mismatch/);

    // Mutating a single finite float inside Mamba state or populated GQA KV must throw
    const tamperMamba = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1],
      prevState: null,
    });
    tamperMamba.nextState.mambaStates.get(1)!.ssmState[0] += 0.01;
    expect(() => validateDecodeCacheState(model, tamperMamba.nextState)).toThrow(
      /stateDigest mismatch/,
    );
    expect(() =>
      runIncrementalDecodeStep(model, { newTokenIds: [2], prevState: tamperMamba.nextState }),
    ).toThrow(/stateDigest mismatch/);

    const tamperGqa = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1],
      prevState: null,
    });
    tamperGqa.nextState.gqaCaches.get(6)!.K[1] += 0.01; // populated slot k=1 < seqLen=2
    expect(() => validateDecodeCacheState(model, tamperGqa.nextState)).toThrow(
      /stateDigest mismatch/,
    );

    // Mutating unpopulated trailing KV slot (k >= seqLen) is excluded from digest per contract
    const trailingUntouched = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1],
      prevState: null,
    });
    trailingUntouched.nextState.gqaCaches.get(6)!.K[15] = 42.0; // slot 15 >= seqLen 2
    expect(() => validateDecodeCacheState(model, trailingUntouched.nextState)).not.toThrow();

    // External seed validation hardening:
    const makeValidExternalMaps = (P: number, kStride: number, vStride: number) => {
      const mambaStates = new Map<number, { convState: Float32Array; ssmState: Float32Array }>();
      const gqaCaches = new Map<number, { K: Float32Array; V: Float32Array }>();
      for (let i = 0; i < model.config.pattern.length; i++) {
        const bayId = i + 1;
        const ch = model.config.pattern[i];
        if (ch === 'M') {
          mambaStates.set(bayId, {
            convState: new Float32Array(128 * 4),
            ssmState: new Float32Array(16 * 4 * 4),
          });
        } else if (ch === '*') {
          gqaCaches.set(bayId, {
            K: new Float32Array(8 * kStride),
            V: new Float32Array(8 * vStride),
          });
        }
      }
      return { mambaStates, gqaCaches, processedTokens: P };
    };

    // (a) seed.maxTokens < P or invalid must throw (no silent clamping!)
    const tooSmallSeed = makeValidExternalMaps(4, 16, 16);
    expect(() =>
      createExternalDecodeCacheState(model, {
        ...tooSmallSeed,
        maxTokens: 2,
      }),
    ).toThrow(/maxTokens \(2\) < processedTokens \(4\)/);
    expect(() =>
      createExternalDecodeCacheState(model, {
        ...tooSmallSeed,
        maxTokens: 0,
      }),
    ).toThrow(RangeError);
    expect(() =>
      createExternalDecodeCacheState(model, {
        ...tooSmallSeed,
        maxTokens: 4.5,
      }),
    ).toThrow(RangeError);

    // (b) src.V.length !== src.K.length when inferring stride without explicit maxTokens must throw
    const mismatchedKvSeed = makeValidExternalMaps(4, 4, 16);
    expect(() =>
      createExternalDecodeCacheState(model, {
        ...mismatchedKvSeed,
        maxTokens: 16,
      }),
    ).toThrow(/K\/V length mismatch/);

    // Direct runMambaSim finiteness guards on initialConvState and initialSsmState
    const badConv = new Float32Array(128 * 4);
    badConv[17] = Number.NaN;
    expect(() => runMambaSim(1, { initialConvState: badConv })).toThrow(RangeError);
    expect(() => runMambaSim(1, { initialConvState: badConv })).toThrow(/Non-finite/);

    const badSsm = new Float32Array(16 * 4 * 4);
    badSsm[42] = Number.NEGATIVE_INFINITY;
    expect(() => runMambaSim(1, { initialSsmState: badSsm })).toThrow(RangeError);
    expect(() => runMambaSim(1, { initialSsmState: badSsm })).toThrow(/Non-finite/);
  });

  it('PR-7/PR-8: isolates lazy getLayerTrace reconstruction from post-step mutations to prevState, nextState, residualStreams, and eagerTrace.sim.Y', () => {
    const model = createDefaultHybridModel();
    const refPrefill = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1, 2, 3],
      prevState: null,
    });
    const refStep = runIncrementalDecodeStep(model, {
      newTokenIds: [5],
      prevState: refPrefill.nextState,
      retainedTraceBays: [1],
    });
    const refBay2 = refStep.getMoeTrace(2);
    const refBay5 = refStep.getMambaTrace(5);
    const refBay11 = refStep.getMoeTrace(11);
    const refBay20 = refStep.getGqaTrace(20);

    // Execute an identical fresh step whose lazy bays (2, 5, 11, 20) have NOT been reconstructed yet
    const prefill = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1, 2, 3],
      prevState: null,
    });
    const step = runIncrementalDecodeStep(model, {
      newTokenIds: [5],
      prevState: prefill.nextState,
      retainedTraceBays: [1],
    });

    // Verify residualStreams[1] is detached from eagerTrace.sim.Y (copy via .slice())
    const eager1 = step.getLayerTrace(1);
    const expectedRes1 = new Float32Array(step.residualStreams[1]);
    expect(step.residualStreams[1]).not.toBe(eager1.sim.Y);
    eager1.sim.Y.fill(333.0);
    assertBitExactFiniteArray(
      step.residualStreams[1],
      expectedRes1,
      'residualStreams[1] detached from eagerTrace.sim.Y',
    );

    // Now aggressively mutate ALL exposed surfaces before lazy reconstruction!
    step.prevState.mambaStates.get(5)!.convState.fill(999.0);
    step.prevState.mambaStates.get(5)!.ssmState.fill(-999.0);
    step.prevState.gqaCaches.get(20)!.K.fill(777.0);
    step.prevState.gqaCaches.get(20)!.V.fill(-777.0);
    step.prevState.prefixTokenIds.fill(31);
    (step.prevState.mambaStates as Map<number, unknown>).clear();
    (step.prevState.gqaCaches as Map<number, unknown>).clear();

    step.nextState.mambaStates.get(5)!.ssmState.fill(888.0);
    step.nextState.gqaCaches.get(20)!.K.fill(888.0);

    step.X_embed.fill(444.0);
    step.newTokenIds.fill(30);
    step.residualStreams[0].fill(555.0);
    step.residualStreams[1].fill(555.0);
    step.residualStreams[4].fill(555.0);
    step.residualStreams[10].fill(555.0);
    step.residualStreams[19].fill(555.0);

    // Every lazily reconstructed bay must STILL match the unmutated reference bit-for-bit!
    const lazyBay2 = step.getMoeTrace(2);
    const lazyBay5 = step.getMambaTrace(5);
    const lazyBay11 = step.getMoeTrace(11);
    const lazyBay20 = step.getGqaTrace(20);

    assertBitExactFiniteArray(lazyBay2.Y, refBay2.Y, 'isolated lazy Bay 2 Y');
    assertBitExactFiniteArray(lazyBay5.Y, refBay5.Y, 'isolated lazy Bay 5 Y');
    assertBitExactFiniteArray(lazyBay5.H, refBay5.H, 'isolated lazy Bay 5 H');
    assertBitExactFiniteArray(lazyBay5.H_prev, refBay5.H_prev, 'isolated lazy Bay 5 H_prev');
    assertBitExactFiniteArray(lazyBay5.conv_buf, refBay5.conv_buf, 'isolated lazy Bay 5 conv_buf');
    assertBitExactFiniteArray(lazyBay11.Y, refBay11.Y, 'isolated lazy Bay 11 Y');
    assertBitExactFiniteArray(lazyBay20.Y, refBay20.Y, 'isolated lazy Bay 20 Y');
    assertBitExactFiniteArray(lazyBay20.K, refBay20.K, 'isolated lazy Bay 20 K');
    assertBitExactFiniteArray(lazyBay20.V, refBay20.V, 'isolated lazy Bay 20 V');
    assertBitExactFiniteArray(lazyBay20.Softmax, refBay20.Softmax, 'isolated lazy Bay 20 Softmax');
    expect(Array.from(lazyBay20.tokenIds)).toEqual([5]);
    expect(Array.from(lazyBay20.keyTokenIds)).toEqual([0, 1, 2, 3, 5]);
  });

  it('PR-9: enforces S<=16 guard, conservative branch/step bounds, explicit delete/prune invariants, and deduplicated byte accounting', () => {
    const model = createDefaultHybridModel();

    // 1. S <= 16 guard (even if maxTokens > 16)
    expect(MAX_INCREMENTAL_STEP_TOKENS).toBe(16);
    const bigZero = createZeroDecodeCacheState(model, { maxTokens: 32 });
    const seventeenTokens = Array.from({ length: 17 }, (_, i) => i % 32);
    expect(() =>
      runIncrementalDecodeStep(model, {
        newTokenIds: seventeenTokens,
        prevState: bigZero,
      }),
    ).toThrow(RangeError);
    expect(() =>
      runIncrementalDecodeStep(model, {
        newTokenIds: seventeenTokens,
        prevState: bigZero,
      }),
    ).toThrow(/at most 16/);

    // 2. Conservative defaults & option validation
    const baseSession = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    expect(baseSession.maxBranches).toBe(DEFAULT_SESSION_MAX_BRANCHES);
    expect(baseSession.maxBranches).toBe(16);
    expect(baseSession.maxUniqueSteps).toBe(DEFAULT_SESSION_MAX_UNIQUE_STEPS);
    expect(baseSession.maxUniqueSteps).toBe(128);
    expect(baseSession.pruneInactiveStepTraces).toBe(true);

    expect(() => createDecodeSession(model, { maxBranches: 0 })).toThrow(RangeError);
    expect(() => createDecodeSession(model, { maxUniqueSteps: -2 })).toThrow(RangeError);

    // Root-only session accounting includes rootStep.prevState (76,800 B)
    const rootStats = measureDecodeSessionRetainedBytes(baseSession);
    expect(rootStats.uniqueSteps).toBe(1);
    expect(rootStats.branchCount).toBe(1);
    expect(rootStats.rootPrevStateBytes).toBe(76_800);
    expect(rootStats.totalRetainedBytes).toBeGreaterThan(2 * 76_800);

    // 3. 20-branch fan-out stays capped at maxBranches = 16, protects active branch & shared root,
    //    and prunes eager traces from inactive historical steps while keeping lazy reconstruction bit-exact.
    let session = stepDecodeSession(baseSession, 4);
    const rootStepId = getActiveDecodeBranch(session).steps[0].stepExecutionId;
    for (let b = 1; b < 20; b++) {
      session = navigateDecodeSession(session, 0);
      session = stepDecodeSession(session, 4 + (b % 20));
    }

    expect(session.branches.size).toBe(16);
    expect(session.activeBranchId).toBe('branch-19');
    expect(session.branches.has('branch-0')).toBe(false);
    expect(session.branches.has('branch-3')).toBe(false);
    expect(session.branches.has('branch-4')).toBe(true);
    expect(session.branches.has('branch-19')).toBe(true);

    // Shared root step survived across all 16 remaining branches!
    for (const br of session.branches.values()) {
      expect(br.steps[0].stepExecutionId).toBe(rootStepId);
      // Inactive root step dropped reconstructible eager traces
      expect(br.steps[0].retainedTraceBayIds()).toEqual([]);
    }
    // Active tip keeps eager traces
    const activeTip = getActiveDecodeStep(session);
    expect(activeTip.retainedTraceBayIds()).toEqual([1, 2, 6, 42, 43, 52]);
    // Inactive branch tip dropped eager traces, yet lazily reconstructs identically
    const inactiveTip = session.branches.get('branch-4')!.steps[1];
    expect(inactiveTip.retainedTraceBayIds()).toEqual([]);
    const rebuiltInactive = inactiveTip.getLayerTrace(43);
    expect(rebuiltInactive.traceOrigin).toBe('reconstructed_from_checkpoint');
    expect(rebuiltInactive.sim.Y.length).toBe(8);

    // Deduplicated session accounting: rootPrevStateBytes counted ONCE (76,800 B), not 16x
    const stats20 = measureDecodeSessionRetainedBytes(session);
    expect(stats20.branchCount).toBe(16);
    expect(stats20.uniqueSteps).toBe(17); // 1 shared root + 16 branch tips
    expect(stats20.rootPrevStateBytes).toBe(76_800);

    // 4. Explicit deleteDecodeSessionBranch invariants
    expect(() => deleteDecodeSessionBranch(session, session.activeBranchId)).toThrow(
      /Cannot delete active/,
    );
    expect(() => deleteDecodeSessionBranch(session, 'branch-does-not-exist')).toThrow(
      /Unknown decode session branch/,
    );
    expect(() => deleteDecodeSessionBranch(baseSession, 'branch-0')).toThrow(
      /Cannot delete active/,
    );

    const afterDelete = deleteDecodeSessionBranch(session, 'branch-4');
    expect(afterDelete.branches.size).toBe(15);
    expect(afterDelete.branches.has('branch-4')).toBe(false);
    expect(afterDelete.activeBranchId).toBe('branch-19');
    expect(getActiveDecodeBranch(afterDelete).steps[0].stepExecutionId).toBe(rootStepId);
    // All surviving branches retain valid cache snapshots and intact shared ancestors
    for (const br of afterDelete.branches.values()) {
      for (const st of br.steps) {
        expect(() => validateDecodeCacheState(model, st.nextState)).not.toThrow();
      }
    }
    expect(measureDecodeSessionRetainedBytes(afterDelete).totalRetainedBytes).toBeLessThan(
      stats20.totalRetainedBytes,
    );

    // Deduplication check: 2-branch session sharing root uses strictly less bytes than 2x 1-branch session
    const oneBranch = stepDecodeSession(baseSession, 4);
    const twoBranches = stepDecodeSession(navigateDecodeSession(oneBranch, 0), 5);
    const oneStats = measureDecodeSessionRetainedBytes(oneBranch);
    const twoStats = measureDecodeSessionRetainedBytes(twoBranches);
    expect(twoStats.uniqueSteps).toBe(3);
    expect(twoStats.rootPrevStateBytes).toBe(76_800);
    expect(twoStats.totalRetainedBytes).toBeLessThan(2 * oneStats.totalRetainedBytes);

    // 5. Explicit pruneDecodeSessionBranches protects active branch EVEN when active branch is oldest!
    const oldestActive = switchDecodeSessionBranch(afterDelete, 'branch-5');
    const prunedTo2 = pruneDecodeSessionBranches(oldestActive, { maxBranches: 2 });
    expect(prunedTo2.branches.size).toBe(2);
    expect(prunedTo2.activeBranchId).toBe('branch-5');
    expect(prunedTo2.branches.has('branch-5')).toBe(true); // active oldest branch protected!
    expect(prunedTo2.branches.has('branch-19')).toBe(true); // newest surviving branch kept!

    // 6. maxUniqueSteps bound prunes oldest inactive branches and NEVER truncates the active branch
    let stepBounded = createDecodeSession(model, {
      promptTokenIds: [0, 1, 2, 3],
      maxBranches: 16,
      maxUniqueSteps: 6,
    });
    // Build 3 steps on branch-0 (4 total steps including root)
    stepBounded = stepDecodeSessionGreedy(stepBounded);
    stepBounded = stepDecodeSessionGreedy(stepBounded);
    stepBounded = stepDecodeSessionGreedy(stepBounded);
    // Rewind to root and build 3 steps on branch-1 -> 1 + 3 + 3 = 7 > maxUniqueSteps(6) -> prunes branch-0!
    stepBounded = navigateDecodeSession(stepBounded, 0);
    stepBounded = stepDecodeSession(stepBounded, 15);
    stepBounded = stepDecodeSessionGreedy(stepBounded);
    stepBounded = stepDecodeSessionGreedy(stepBounded);
    expect(stepBounded.activeBranchId).toBe('branch-1');
    expect(stepBounded.branches.size).toBe(1);
    expect(getActiveDecodeBranch(stepBounded).steps).toHaveLength(4);

    // Active branch alone exceeding maxUniqueSteps(2) is NEVER truncated or corrupted!
    const tinyStepBound = pruneDecodeSessionBranches(stepBounded, { maxUniqueSteps: 2 });
    expect(tinyStepBound.branches.size).toBe(1);
    expect(tinyStepBound.activeBranchId).toBe('branch-1');
    expect(getActiveDecodeBranch(tinyStepBound).steps).toHaveLength(4);

    // 7. Explicit pruneInactiveStepTraces reduces retained bytes when pruneInactiveStepTraces was false
    let unpruned = createDecodeSession(model, {
      promptTokenIds: [0, 1, 2, 3],
      pruneInactiveStepTraces: false,
    });
    unpruned = stepDecodeSessionGreedy(unpruned);
    unpruned = stepDecodeSessionGreedy(unpruned);
    const beforeTracePrune = measureDecodeSessionRetainedBytes(unpruned);
    const afterTracePruneSession = pruneInactiveStepTraces(unpruned);
    const afterTracePrune = measureDecodeSessionRetainedBytes(afterTracePruneSession);
    expect(afterTracePrune.totalRetainedBytes).toBeLessThan(beforeTracePrune.totalRetainedBytes);
    // Original unpruned session remains untouched (immutable)
    expect(getActiveDecodeBranch(unpruned).steps[0].retainedTraceBayIds().length).toBeGreaterThan(0);
    expect(getActiveDecodeBranch(afterTracePruneSession).steps[0].retainedTraceBayIds()).toEqual([]);
  });
});
