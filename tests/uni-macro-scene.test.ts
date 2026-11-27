/**
 * UNI4 View 1 (`macro`) Verification Suite (`tests/uni-macro-scene.test.ts`):
 *   1. All 6 `UniExecutionViewId` modes (standalone `session === null` & live `UniSession`),
 *      including explicit `UNI_SCENE_INTERLEAVED_BADGE` on `interleaved_mixed`.
 *   2. Twin 34-layer Elastic-MoT geometry (`34 U`, `34 G`, `18` joint couplers `L0..17`,
 *      visually separated `L18..33` split towers, `L17 | L18` divider landmark, all 8 I/O bookends).
 *   3. Live `D5` toggle (`hf_faithful` vs `elastic_pruned`) on T2I (`47.1%`) & Editing (`31.4%`):
 *      distinct geometry/highlights/ribbons/anchors/landmarks/calipers + exact `d5Comparison` telemetry
 *      (`16` vs `0` wasted split-U layers, `stage2UlpDiff === 0`).
 *   4. Strictly `0 B` `RG32F` atlas (`valueAtlas: null`, `rgbPreviews: []`, `sheetCount: 0`) and
 *      `validateSceneGeometryFinite(scene).valid === true` across all modes.
 *   5. Real raycast (`pickMacroBay`) -> `pickHitToUniCell` round-trip across all 34 U bays,
 *      all 34 G bays, `L17` vs `L18` split boundary, all 18 joint cross-couplers, and all 8 I/O bookends.
 *   6. Clean-room hygiene (zero `Math.random()`, zero DOM/WebGL, zero forbidden sibling-app imports,
 *      deterministic output).
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { pickMacroBay, type PickBox, type Ray } from '../src/render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_WEIGHT_BLUE,
  SOLID_INSTANCE_FLOATS,
} from '../src/scene/spine';
import {
  createDefaultUniRuntimeBundle,
  createUniSession,
  type UniElasticPrefillMode,
  type UniExecutionViewId,
  type UniSessionTaskMode,
} from '../src/uni/index';
import {
  buildUniMacroScene,
  resolveUniMacroExecutionProfile,
  UNI_MACRO_IO_MODULE_IDS,
  UNI_MACRO_LAYOUT,
  uniMacroLayerCenterY,
} from '../src/uni/scene/macroScene';
import {
  pickHitToUniCell,
  UNI_SCENE_INTERLEAVED_BADGE,
  validateSceneGeometryFinite,
} from '../src/uni/scene/provenance';

const ALL_EXECUTION_VIEWS: readonly UniExecutionViewId[] = [
  'text_understanding',
  'visual_understanding',
  'object_grounding',
  'text_to_image',
  'image_edit',
  'interleaved_mixed',
];

const SESSION_TASK_MODES: readonly UniSessionTaskMode[] = [
  'text_understanding',
  'visual_understanding',
  'object_grounding',
  'text_to_image',
  'image_edit',
];

function rayIntoBoxFront(box: PickBox): Ray {
  const cx = 0.5 * (box.min[0] + box.max[0]);
  const cy = 0.5 * (box.min[1] + box.max[1]);
  const zStart = box.max[2] + 15.0;
  return {
    origin: [cx, cy, zStart],
    direction: [0, 0, -1],
  };
}

interface ParsedSolidInstance {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly cellFamily: number;
  readonly fp8: number;
  readonly bayId: number;
  readonly highlight: number;
}

function parseSolids(solidData: Float32Array, solidCount: number): ParsedSolidInstance[] {
  const result: ParsedSolidInstance[] = [];
  for (let i = 0; i < solidCount; i++) {
    const base = i * SOLID_INSTANCE_FLOATS;
    result.push({
      cx: solidData[base + 0]!,
      cy: solidData[base + 1]!,
      cz: solidData[base + 2]!,
      cellFamily: solidData[base + 6]!,
      fp8: solidData[base + 7]!,
      bayId: solidData[base + 9]!,
      highlight: solidData[base + 11]!,
    });
  }
  return result;
}

describe('UNI4 View 1 (macro) — Elastic Macro Architecture 34L Scene', () => {
  const sharedBundle = createDefaultUniRuntimeBundle();

  it('builds valid 0-byte-atlas scenes across all 6 standalone execution views (session === null)', () => {
    for (const viewId of ALL_EXECUTION_VIEWS) {
      for (const prefillMode of ['elastic_pruned', 'hf_faithful'] as const satisfies readonly UniElasticPrefillMode[]) {
        const scene = buildUniMacroScene(null, {
          executionViewId: viewId,
          prefillMode,
          prefixTokens: 16,
        });

        expect(scene.viewMode).toBe('macro');
        expect(scene.executionViewId).toBe(viewId);
        expect(scene.prefillMode).toBe(prefillMode);

        // Zero RG32F atlas & zero RGB previews
        expect(scene.valueAtlas).toBeNull();
        expect(scene.atlasKey).toBeNull();
        expect(scene.atlasRegions).toHaveLength(0);
        expect(scene.atlasAudit).toBeNull();
        expect(scene.rgbPreviews).toHaveLength(0);
        expect(scene.sheetCount).toBe(0);
        expect(scene.sheetData.length).toBe(0);
        expect(scene.sheetBasisData.length).toBe(0);
        expect(scene.sheetRevealData.length).toBe(0);

        // Non-empty 3D geometry across solids, lines, ribbons, and flat flow bridges
        expect(scene.solidCount).toBeGreaterThanOrEqual(94);
        expect(scene.lineCount).toBeGreaterThan(0);
        expect(scene.ribbonCount).toBeGreaterThan(0);
        expect(scene.flowVertices.length).toBeGreaterThan(0);
        expect(scene.flowIndices.length).toBeGreaterThan(0);

        // Finite geometry & stride validation
        const validation = validateSceneGeometryFinite(scene);
        expect(validation.valid, `${viewId}/${prefillMode}: ${validation.errors.join('; ')}`).toBe(
          true,
        );
        expect(validation.errors).toHaveLength(0);

        // Readouts wired to archReadouts.ts
        expect(scene.viewReadout?.viewId).toBe(viewId);
        expect(scene.d5Comparison?.viewId).toBe(viewId);
        expect(scene.d5Comparison?.prefixTokens).toBe(16);
        expect(scene.d5Comparison?.stage2UlpDiff).toBe(0);

        // Interleaved provenance badge strictly on interleaved_mixed
        const hasInterleavedBadge = scene.provenanceBadges.some(
          (b) => b.label === UNI_SCENE_INTERLEAVED_BADGE && b.tone === 'warn',
        );
        expect(hasInterleavedBadge).toBe(viewId === 'interleaved_mixed');
      }
    }
  });

  it('builds valid scenes from live UniSession across all 5 interactive task modes plus interleaved override', () => {
    const session = createUniSession(sharedBundle);

    for (const taskMode of SESSION_TASK_MODES) {
      const prefillMode: UniElasticPrefillMode =
        taskMode === 'text_to_image' || taskMode === 'image_edit'
          ? 'hf_faithful'
          : 'elastic_pruned';
      const branch = session.startBranch({
        taskMode,
        prefillMode,
        ...(taskMode === 'image_edit' ? { scenePresetId: 'harbor_panorama_16x8' as const } : {}),
      });

      const scene = buildUniMacroScene(session);
      expect(scene.executionViewId).toBe(taskMode);
      expect(scene.prefillMode).toBe(branch.prefillMode);
      expect(scene.d5Comparison?.prefixTokens).toBe(Math.max(1, branch.prefixCache.seqLen));
      expect(scene.valueAtlas).toBeNull();
      expect(scene.rgbPreviews).toHaveLength(0);
      expect(validateSceneGeometryFinite(scene).valid).toBe(true);

      // Verify live D5 toggle override on T2I and Image Edit branches backed by UniSession
      if (taskMode === 'text_to_image' || taskMode === 'image_edit') {
        const prunedScene = buildUniMacroScene(session, { prefillMode: 'elastic_pruned' });
        expect(prunedScene.prefillMode).toBe('elastic_pruned');
        expect(scene.solidCount - prunedScene.solidCount).toBe(16);
        expect(validateSceneGeometryFinite(prunedScene).valid).toBe(true);
      }
    }

    // Explicit interleaved_mixed override on a live session still surfaces UNI_SCENE_INTERLEAVED_BADGE
    const interleavedScene = buildUniMacroScene(session, {
      executionViewId: 'interleaved_mixed',
    });
    expect(interleavedScene.executionViewId).toBe('interleaved_mixed');
    expect(
      interleavedScene.provenanceBadges.some(
        (b) => b.label === UNI_SCENE_INTERLEAVED_BADGE && b.tone === 'warn',
      ),
    ).toBe(true);
    expect(validateSceneGeometryFinite(interleavedScene).valid).toBe(true);
  });

  it('enforces twin 34L geometry, L17 | L18 elastic split separation, 18 joint couplers, and all 8 I/O bookends', () => {
    const scene = buildUniMacroScene(null, {
      executionViewId: 'text_to_image',
      prefillMode: 'elastic_pruned',
    });

    const uLayerBoxes = scene.pickBoxes.filter(
      (pb) => pb.bayId >= 1 && (pb.semanticTag ?? '').includes('branch=u'),
    );
    const gLayerBoxes = scene.pickBoxes.filter(
      (pb) => pb.bayId >= 1 && (pb.semanticTag ?? '').includes('branch=g'),
    );
    const jointBoxes = scene.pickBoxes.filter(
      (pb) => pb.bayId >= 1 && (pb.semanticTag ?? '').includes('branch=joint'),
    );
    const ioBoxes = scene.pickBoxes.filter((pb) => pb.bayId === 0);

    expect(uLayerBoxes).toHaveLength(34);
    expect(gLayerBoxes).toHaveLength(34);
    expect(jointBoxes).toHaveLength(18);
    expect(ioBoxes).toHaveLength(UNI_MACRO_IO_MODULE_IDS.length);

    // Every U bay sits on left (x < 0), every G bay on right (x > 0), every coupler centered at x = 0
    for (const pb of uLayerBoxes) {
      expect(0.5 * (pb.min[0] + pb.max[0])).toBeLessThan(0);
      expect(pb.towerRole).toBe('context');
    }
    for (const pb of gLayerBoxes) {
      expect(0.5 * (pb.min[0] + pb.max[0])).toBeGreaterThan(0);
      expect(pb.towerRole).toBe('denoiser');
    }
    for (const pb of jointBoxes) {
      expect(0.5 * (pb.min[0] + pb.max[0])).toBeCloseTo(0, 6);
      expect(pb.towerRole).toBe('shared');
    }

    // Visual horizontal & vertical separation across L17 | L18 split boundary
    expect(Math.abs(UNI_MACRO_LAYOUT.uSplitX)).toBeGreaterThan(Math.abs(UNI_MACRO_LAYOUT.uJointX));
    expect(UNI_MACRO_LAYOUT.gSplitX).toBeGreaterThan(UNI_MACRO_LAYOUT.gJointX);
    const normalStep = uniMacroLayerCenterY(0) - uniMacroLayerCenterY(1);
    const splitStep = uniMacroLayerCenterY(17) - uniMacroLayerCenterY(18);
    expect(splitStep).toBeGreaterThan(normalStep * 2.0);

    // Explicit L17 | L18 elastic split divider landmark & overlay anchor
    expect(scene.landmarks.some((l) => l.id === 'elastic-split-boundary')).toBe(true);
    expect(scene.overlayAnchors.some((a) => a.id === 'anchor-split-divider')).toBe(true);

    // All 8 required I/O bookend modules present
    const ioIds = new Set(ioBoxes.map((pb) => pb.partId));
    for (const expectedId of UNI_MACRO_IO_MODULE_IDS) {
      expect(ioIds.has(expectedId), `Missing I/O bookend "${expectedId}"`).toBe(true);
    }
  });

  it('toggles D5 (hf_faithful vs elastic_pruned) on T2I (47.1%) and Image Edit (31.4%) with distinct geometry, amber highlights, and telemetry', () => {
    const cases: readonly {
      readonly viewId: 'text_to_image' | 'image_edit';
      readonly expectedPct: string;
      readonly expectedCfgBranches: number;
      readonly expectedPrunedLayersExecuted: number;
    }[] = [
      {
        viewId: 'text_to_image',
        expectedPct: '47.1',
        expectedCfgBranches: 2,
        expectedPrunedLayersExecuted: 18,
      },
      {
        viewId: 'image_edit',
        expectedPct: '31.4',
        expectedCfgBranches: 3,
        expectedPrunedLayersExecuted: 34,
      },
    ];

    for (const tc of cases) {
      const faithfulProfile = resolveUniMacroExecutionProfile(tc.viewId, 'hf_faithful');
      const prunedProfile = resolveUniMacroExecutionProfile(tc.viewId, 'elastic_pruned');
      expect(faithfulProfile.wastedSplitULayers).toBe(16);
      expect(faithfulProfile.uSplitState).toBe('wasted_kv');
      expect(prunedProfile.wastedSplitULayers).toBe(0);
      expect(prunedProfile.uSplitState).toBe('pruned_ghost');

      const faithful = buildUniMacroScene(null, {
        executionViewId: tc.viewId,
        prefillMode: 'hf_faithful',
        prefixTokens: 16,
      });
      const pruned = buildUniMacroScene(null, {
        executionViewId: tc.viewId,
        prefillMode: 'elastic_pruned',
        prefixTokens: 16,
      });

      // Exact D5 telemetry
      expect(faithful.d5Comparison).not.toBeNull();
      expect(pruned.d5Comparison).not.toBeNull();
      expect(faithful.d5Comparison!.cfgBranches).toBe(tc.expectedCfgBranches);
      expect(faithful.d5Comparison!.hfFaithfulLayersExecuted).toBe(34);
      expect(faithful.d5Comparison!.elasticPrunedLayersExecuted).toBe(
        tc.expectedPrunedLayersExecuted,
      );
      expect((faithful.d5Comparison!.wastedPrefixKvFraction * 100).toFixed(1)).toBe(
        tc.expectedPct,
      );
      expect(faithful.d5Comparison!.stage2UlpDiff).toBe(0);
      expect(pruned.d5Comparison!.stage2UlpDiff).toBe(0);

      // Distinct geometry: HF-Faithful adds 16 protruding amber wasted KV slabs on U L18..33
      expect(faithful.solidCount - pruned.solidCount).toBe(16);
      expect(faithful.lineCount).toBeGreaterThan(pruned.lineCount);
      expect(faithful.ribbonCount).not.toBe(pruned.ribbonCount);
      expect(faithful.flowVertices.length).not.toBe(pruned.flowVertices.length);

      // Inspect U L18..33 main bay shading in solidData
      const faithfulSolids = parseSolids(faithful.solidData, faithful.solidCount);
      const prunedSolids = parseSolids(pruned.solidData, pruned.solidCount);

      const faithfulSplitUBays = faithfulSolids.filter(
        (s) => s.bayId >= 19 && s.bayId <= 34 && Math.abs(s.cx - UNI_MACRO_LAYOUT.uSplitX) < 0.01,
      );
      const prunedSplitUBays = prunedSolids.filter(
        (s) => s.bayId >= 19 && s.bayId <= 34 && Math.abs(s.cx - UNI_MACRO_LAYOUT.uSplitX) < 0.01,
      );
      expect(faithfulSplitUBays).toHaveLength(16);
      expect(prunedSplitUBays).toHaveLength(16);

      for (const s of faithfulSplitUBays) {
        expect(s.cellFamily).toBe(CELL_MOE_AMBER);
        expect(s.fp8).toBe(0);
        expect(s.highlight).toBe(2);
      }
      for (const s of prunedSplitUBays) {
        expect(s.cellFamily).toBe(CELL_WEIGHT_BLUE);
        expect(s.fp8).toBe(1);
        expect(s.highlight).toBe(0);
      }

      // G L18..33 stays active emerald in both modes
      const faithfulSplitGBays = faithfulSolids.filter(
        (s) => s.bayId >= 19 && s.bayId <= 34 && Math.abs(s.cx - UNI_MACRO_LAYOUT.gSplitX) < 0.01,
      );
      expect(faithfulSplitGBays).toHaveLength(16);
      for (const s of faithfulSplitGBays) {
        expect(s.cellFamily).toBe(CELL_ACTIVATION_GREEN);
        expect(s.fp8).toBe(0);
      }

      // Distinct D5 annotations in overlayAnchors, landmarks, and calipers
      const faithfulAnchor = faithful.overlayAnchors.find((a) => a.id === 'anchor-d5-comparison');
      const prunedAnchor = pruned.overlayAnchors.find((a) => a.id === 'anchor-d5-comparison');
      expect(faithfulAnchor?.tone).toBe('warn');
      expect(faithfulAnchor?.label).toContain(`${tc.expectedPct}%`);
      expect(prunedAnchor?.tone).toBe('g_teal');
      expect(prunedAnchor?.label).toContain('0 Wasted KV');

      const faithfulLandmark = faithful.landmarks.find((l) => l.id === 'd5-split-u-status');
      const prunedLandmark = pruned.landmarks.find((l) => l.id === 'd5-split-u-status');
      expect(faithfulLandmark?.tone).toBe('gold');
      expect(faithfulLandmark?.label).toContain(`${tc.expectedPct}%`);
      expect(prunedLandmark?.tone).toBe('cyan');
      expect(prunedLandmark?.label).toContain('0 Wasted KV');

      const faithfulCaliper = faithful.calipers.find((c) => c.id === 'caliper-split-16l');
      const prunedCaliper = pruned.calipers.find((c) => c.id === 'caliper-split-16l');
      expect(faithfulCaliper?.label).not.toBe(prunedCaliper?.label);
      expect(faithfulCaliper?.label).toContain(`${tc.expectedPct}%`);
      expect(prunedCaliper?.label).toContain('0 Wasted KV');
    }
  });

  it('dims G tower on Understanding/Grounding and activates all 68 blocks on Interleaved Mixed', () => {
    const und = buildUniMacroScene(null, { executionViewId: 'text_understanding' });
    const undSolids = parseSolids(und.solidData, und.solidCount);
    const undGBays = undSolids.filter((s) => s.bayId >= 1 && s.cx > 1.0);
    const undUBays = undSolids.filter((s) => s.bayId >= 1 && s.cx < -1.0);
    expect(undGBays).toHaveLength(34);
    expect(undUBays).toHaveLength(34);
    expect(undGBays.every((s) => s.fp8 === 1)).toBe(true);
    expect(undUBays.every((s) => s.fp8 === 0)).toBe(true);

    const mixed = buildUniMacroScene(null, { executionViewId: 'interleaved_mixed' });
    const mixedSolids = parseSolids(mixed.solidData, mixed.solidCount);
    const mixedGBays = mixedSolids.filter((s) => s.bayId >= 1 && s.cx > 1.0);
    const mixedUBays = mixedSolids.filter((s) => s.bayId >= 1 && s.cx < -1.0);
    expect(mixedGBays.every((s) => s.fp8 === 0)).toBe(true);
    expect(mixedUBays.every((s) => s.fp8 === 0)).toBe(true);
  });

  it('round-trips raycast pickMacroBay -> pickHitToUniCell across all 34 U bays, 34 G bays, L17/L18 boundary, 18 joint couplers, and 8 I/O bookends', () => {
    const scene = buildUniMacroScene(null, {
      executionViewId: 'image_edit',
      prefillMode: 'hf_faithful',
      selectedLayerIndex: 17,
    });

    // All 34 U bays
    for (let l = 0; l < 34; l++) {
      const box = scene.pickBoxes.find((pb) => pb.partId === `u_layer_${l}`);
      expect(box, `Missing pickBox u_layer_${l}`).toBeDefined();
      const hit = pickMacroBay(rayIntoBoxFront(box!), scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      const cell = pickHitToUniCell(hit, 'macro');
      expect(cell).not.toBeNull();
      expect(cell!.viewMode).toBe('macro');
      expect(cell!.layerIndex).toBe(l);
      expect(cell!.isSplitLayer).toBe(l >= 18);
      expect(cell!.branch).toBe('u');
      expect(cell!.partId).toBe(`u_layer_${l}`);
    }

    // All 34 G bays
    for (let l = 0; l < 34; l++) {
      const box = scene.pickBoxes.find((pb) => pb.partId === `g_layer_${l}`);
      expect(box, `Missing pickBox g_layer_${l}`).toBeDefined();
      const hit = pickMacroBay(rayIntoBoxFront(box!), scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      const cell = pickHitToUniCell(hit, 'macro');
      expect(cell).not.toBeNull();
      expect(cell!.layerIndex).toBe(l);
      expect(cell!.isSplitLayer).toBe(l >= 18);
      expect(cell!.branch).toBe('g');
      expect(cell!.partId).toBe(`g_layer_${l}`);
    }

    // Explicit L17 (joint) vs L18 (split) boundary verification
    const u17 = pickHitToUniCell(
      pickMacroBay(
        rayIntoBoxFront(scene.pickBoxes.find((pb) => pb.partId === 'u_layer_17')!),
        scene.pickBoxes,
        [],
      ),
      'macro',
    );
    const u18 = pickHitToUniCell(
      pickMacroBay(
        rayIntoBoxFront(scene.pickBoxes.find((pb) => pb.partId === 'u_layer_18')!),
        scene.pickBoxes,
        [],
      ),
      'macro',
    );
    expect(u17?.layerIndex).toBe(17);
    expect(u17?.isSplitLayer).toBe(false);
    expect(u18?.layerIndex).toBe(18);
    expect(u18?.isSplitLayer).toBe(true);

    // All 18 joint cross-couplers
    for (let l = 0; l < 18; l++) {
      const box = scene.pickBoxes.find((pb) => pb.partId === `joint_coupler_${l}`);
      expect(box, `Missing pickBox joint_coupler_${l}`).toBeDefined();
      const hit = pickMacroBay(rayIntoBoxFront(box!), scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      const cell = pickHitToUniCell(hit, 'macro');
      expect(cell).not.toBeNull();
      expect(cell!.layerIndex).toBe(l);
      expect(cell!.isSplitLayer).toBe(false);
      expect(cell!.branch).toBe('joint');
      expect(cell!.partId).toBe(`joint_coupler_${l}`);
    }

    // All 8 I/O bookend modules
    for (const modId of UNI_MACRO_IO_MODULE_IDS) {
      const box = scene.pickBoxes.find((pb) => pb.partId === modId);
      expect(box, `Missing pickBox ${modId}`).toBeDefined();
      const hit = pickMacroBay(rayIntoBoxFront(box!), scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      const cell = pickHitToUniCell(hit, 'macro');
      expect(cell).not.toBeNull();
      expect(cell!.layerIndex).toBeNull();
      expect(cell!.isSplitLayer).toBeNull();
      expect(cell!.partId).toBe(modId);
      expect(cell!.branch).toBe(box!.towerRole === 'context' ? 'u' : 'g');
    }
  });

  it('produces bit-identical deterministic buffers and satisfies strict clean-room hygiene rules', () => {
    const a = buildUniMacroScene(null, {
      executionViewId: 'text_to_image',
      prefillMode: 'hf_faithful',
      selectedLayerIndex: 18,
      hoveredPartId: 'u_layer_18',
    });
    const b = buildUniMacroScene(null, {
      executionViewId: 'text_to_image',
      prefillMode: 'hf_faithful',
      selectedLayerIndex: 18,
      hoveredPartId: 'u_layer_18',
    });

    expect(Array.from(a.solidData)).toEqual(Array.from(b.solidData));
    expect(Array.from(a.lineData)).toEqual(Array.from(b.lineData));
    expect(Array.from(a.ribbonData)).toEqual(Array.from(b.ribbonData));
    expect(Array.from(a.flowVertices)).toEqual(Array.from(b.flowVertices));
    expect(Array.from(a.flowIndices)).toEqual(Array.from(b.flowIndices));

    const src = readFileSync(resolve(process.cwd(), 'src/uni/scene/macroScene.ts'), 'utf8');
    expect(src).not.toMatch(/Math\.random\s*\(/);
    expect(src).not.toMatch(/from\s+['"][^'"]*twotower/);
    expect(src).not.toMatch(/from\s+['"][^'"]*diffusion/);
    expect(src).not.toMatch(/from\s+['"][^'"]*ui\/controls/);
    expect(src).not.toMatch(/\b(document|window|WebGL2RenderingContext)\b/);
  });
});
