/**
 * View 1 (`macro`): Elastic Macro Architecture Spatial Scene (`src/uni/scene/macroScene.ts`)
 *
 * Clean-room headless 3D scene builder for Nemotron-Labs-Diffusion-Uni's twin 34-layer
 * Elastic Mixture-of-Transformers (Elastic-MoT) architecture:
 *   - Twin 34-layer Understanding (`U`, left `x < 0`, indigo) and Generation (`G`, right
 *     `x > 0`, emerald-teal) towers across physical layers `L0..L33`.
 *   - `18` luminous joint-attention cross-couplers bridging `L0..L17` (`split_tower = False`)
 *     with solids, crisp front-face-culled edges, and bidirectional flat flow bridges.
 *   - Visually separated `L18..L33` split towers (`split_tower = True`) with an explicit
 *     `L17 | L18` elastic split divider datum rail, flare connectors, and landmark.
 *   - All 8 I/O bookend modules (`embed_tokens`, `vision_tower`, `mm_projector`, `vqvae`,
 *     `packed_image_codec`, `gen_embedding`, `diffusion_head`, `gen_predictor`).
 *   - Task-reactive execution across all 6 `UniExecutionViewId` modes, including the
 *     explicit `UNI_SCENE_INTERLEAVED_BADGE` on `interleaved_mixed`.
 *   - Live `D5` comparison toggle (`hf_faithful` vs `elastic_pruned`) wired directly to
 *     `buildUniViewArchReadouts()` and `buildUniElasticD5Comparison()`:
 *       * `hf_faithful` highlights `U` `L18..33` as 16 amber wasted prefix-KV bays
 *         (`16/34 = 47.1%` T2I, `32/102 = 31.4%` 3-branch Editing) with protruding
 *         prefix-KV waste slabs & arrows.
 *       * `elastic_pruned` prunes/ghosts `U` `L18..33` (`0` wasted KV, `18L U + 34L G`,
 *         `0`-ULP Stage-2 equivalence).
 *   - Strictly `0 B` `RG32F` atlas upload (`valueAtlas: null`, `sheetCount: 0`, `rgbPreviews: []`)
 *     and complete pick coverage across all 68 layer bays, 18 joint couplers, and 8 I/O bookends.
 *
 * Pure headless TypeScript — zero DOM, zero WebGL execution, zero ambient RNG, and zero
 * cross-app imports.
 */

import type { Vec3 } from '../../model/spec';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_KIND_ARROW_NEG_X,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_BOX,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import {
  buildUniElasticD5Comparison,
  buildUniViewArchReadouts,
} from '../archReadouts';
import type { UniSession } from '../session';
import { formatSiIecBytes } from '../spec';
import {
  UNI_MINI_CONFIG,
  type UniElasticPrefillMode,
  type UniExecutionViewId,
} from '../types';
import {
  packPickBoxFromAabb,
  resolveUniSessionContext,
  UNI_FLOW_COLORS,
  UNI_SCENE_INTERLEAVED_BADGE,
  UNI_SCENE_PROJECTION_REVERSAL_BADGE,
  UniFlatFlowBufferBuilder,
  UniLineBufferBuilder,
  UniRibbonBufferBuilder,
  UniSolidBufferBuilder,
} from './provenance';
import type {
  UniMacroSceneOptions,
  UniOverlayAnchor,
  UniProvenanceBadge,
  UniSpatialScene,
} from './types';

// ============================================================================
// LAYOUT CONSTANTS & READONLY HELPERS
// ============================================================================

export type UniMacroBayVisualState =
  | 'active'
  | 'wasted_kv'
  | 'pruned_ghost'
  | 'ghosted';

export type UniMacroIoModuleId =
  | 'embed_tokens'
  | 'vision_tower'
  | 'mm_projector'
  | 'vqvae'
  | 'packed_image_codec'
  | 'gen_embedding'
  | 'diffusion_head'
  | 'gen_predictor';

export const UNI_MACRO_IO_MODULE_IDS: readonly UniMacroIoModuleId[] = Object.freeze([
  'embed_tokens',
  'vision_tower',
  'mm_projector',
  'vqvae',
  'packed_image_codec',
  'gen_embedding',
  'diffusion_head',
  'gen_predictor',
]);

export interface UniMacroExecutionProfile {
  readonly executionViewId: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly uJointState: UniMacroBayVisualState;
  readonly uSplitState: UniMacroBayVisualState;
  readonly gJointState: UniMacroBayVisualState;
  readonly gSplitState: UniMacroBayVisualState;
  readonly couplerState: 'active' | 'ghosted';
  readonly wastedSplitULayers: number;
  readonly activeIoModules: Readonly<Record<UniMacroIoModuleId, boolean>>;
  readonly vqEncodeActive: boolean;
  readonly vqDecodeActive: boolean;
}

const NUM_LAYERS = UNI_MINI_CONFIG.numLayers; // 34
const DUAL_TOWER_LAYERS = UNI_MINI_CONFIG.dualTowerLayers; // 18
const SPLIT_LAYERS = UNI_MINI_CONFIG.splitLayers; // 16

const BAY_PITCH_Y = 1.18;
const SUBGROUP_GAP_Y = 0.24;
const SPLIT_GAP_Y = 2.10;

/**
 * Deterministic vertical world center Y for 0-based physical layer `layerIndex` (`0..33`),
 * descending top-to-bottom along `-Y` with subtle 6-layer rhythm gaps and an unmistakable
 * elastic split gap between `L17` (joint) and `L18` (split-tower).
 */
export function uniMacroLayerCenterY(layerIndex: number): number {
  const l = Math.max(0, Math.min(NUM_LAYERS - 1, Math.trunc(layerIndex)));
  let y = -l * BAY_PITCH_Y;
  if (l >= 6) y -= SUBGROUP_GAP_Y;
  if (l >= 12) y -= SUBGROUP_GAP_Y;
  if (l >= DUAL_TOWER_LAYERS) y -= SPLIT_GAP_Y;
  if (l >= 24) y -= SUBGROUP_GAP_Y;
  if (l >= 30) y -= SUBGROUP_GAP_Y;
  return y;
}

const LAYER_0_Y = uniMacroLayerCenterY(0);
const LAYER_17_Y = uniMacroLayerCenterY(DUAL_TOWER_LAYERS - 1);
const LAYER_18_Y = uniMacroLayerCenterY(DUAL_TOWER_LAYERS);
const LAYER_33_Y = uniMacroLayerCenterY(NUM_LAYERS - 1);
const SPLIT_DIVIDER_Y = 0.5 * (LAYER_17_Y + LAYER_18_Y);

const TOP_IO_ROW_Y = LAYER_0_Y + 3.20;
const TOP_ENCODER_ROW_Y = LAYER_0_Y + 6.10;
const BOTTOM_IO_ROW_Y = LAYER_33_Y - 3.25;

export const UNI_MACRO_LAYOUT = Object.freeze({
  numLayers: NUM_LAYERS,
  dualTowerLayers: DUAL_TOWER_LAYERS,
  splitLayers: SPLIT_LAYERS,
  bayPitchY: BAY_PITCH_Y,
  subgroupGapY: SUBGROUP_GAP_Y,
  splitGapY: SPLIT_GAP_Y,
  uJointX: -3.85,
  gJointX: 3.85,
  uSplitX: -4.65,
  gSplitX: 4.65,
  couplerX: 0.0,
  bayExtent: Object.freeze([2.70, 0.78, 1.35] as Vec3),
  couplerExtent: Object.freeze([4.60, 0.34, 0.62] as Vec3),
  wastedKvSlabX: -7.15,
  wastedKvSlabExtent: Object.freeze([1.15, 0.56, 0.92] as Vec3),
  uJointSpineX: -5.58,
  gJointSpineX: 5.58,
  uSplitSpineX: -6.38,
  gSplitSpineX: 6.38,
  layer0Y: LAYER_0_Y,
  layer17Y: LAYER_17_Y,
  layer18Y: LAYER_18_Y,
  layer33Y: LAYER_33_Y,
  splitDividerY: SPLIT_DIVIDER_Y,
  topIoRowY: TOP_IO_ROW_Y,
  topEncoderRowY: TOP_ENCODER_ROW_Y,
  bottomIoRowY: BOTTOM_IO_ROW_Y,
  dividerSpanHalfX: 7.60,
});

const MACRO_FIT_BOUNDS: AABB = Object.freeze({
  min: Object.freeze([-10.6, BOTTOM_IO_ROW_Y - 2.1, -2.2] as Vec3),
  max: Object.freeze([10.6, TOP_ENCODER_ROW_Y + 2.2, 2.4] as Vec3),
});

interface IoBookendGeometrySpec {
  readonly id: UniMacroIoModuleId;
  readonly label: string;
  readonly sublabel: string;
  readonly tensorKey: string;
  readonly branch: 'u' | 'g';
  readonly role: 'input' | 'output' | 'io';
  readonly towerRole: 'context' | 'denoiser';
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly cellFamily: number;
}

const IO_BOOKENDS: readonly IoBookendGeometrySpec[] = Object.freeze([
  Object.freeze({
    id: 'embed_tokens',
    label: 'embed_tokens',
    sublabel: '[132,101 × 4,096] · U Text/Multimodal Embed',
    tensorKey: 'encoder.embed_tokens.weight',
    branch: 'u',
    role: 'input',
    towerRole: 'context',
    cx: UNI_MACRO_LAYOUT.uJointX,
    cy: TOP_IO_ROW_Y,
    cz: 0,
    ex: 2.85,
    ey: 1.24,
    ez: 1.25,
    cellFamily: CELL_WEIGHT_BLUE,
  }),
  Object.freeze({
    id: 'vision_tower',
    label: 'vision_tower (RADIO-H)',
    sublabel: '32L ViT-Huge/16 · Continuous Semantic Stream',
    tensorKey: 'encoder.vision_tower',
    branch: 'u',
    role: 'input',
    towerRole: 'context',
    cx: -8.10,
    cy: TOP_ENCODER_ROW_Y,
    cz: 0,
    ex: 2.75,
    ey: 1.28,
    ez: 1.18,
    cellFamily: CELL_WEIGHT_BLUE,
  }),
  Object.freeze({
    id: 'mm_projector',
    label: 'mm_projector',
    sublabel: '2×2 Gather → RMSNorm → SquaredReLU ×0.01',
    tensorKey: 'encoder.mm_projector',
    branch: 'u',
    role: 'input',
    towerRole: 'context',
    cx: -8.10,
    cy: TOP_IO_ROW_Y,
    cz: 0,
    ex: 2.55,
    ey: 1.05,
    ez: 1.10,
    cellFamily: CELL_SSM_CYAN,
  }),
  Object.freeze({
    id: 'vqvae',
    label: 'vqvae (Emu3 IBQ)',
    sublabel: '131,072-Codebook IBQ VQ-VAE Enc/Dec',
    tensorKey: 'encoder.vqvae',
    branch: 'g',
    role: 'io',
    towerRole: 'denoiser',
    cx: 8.10,
    cy: TOP_ENCODER_ROW_Y,
    cz: 0,
    ex: 2.75,
    ey: 1.28,
    ez: 1.18,
    cellFamily: CELL_ACTIVATION_GREEN,
  }),
  Object.freeze({
    id: 'packed_image_codec',
    label: 'packed_image_codec',
    sublabel: '2×2 Down/Up Linear Codec (16,384 ↔ 4,096)',
    tensorKey: 'encoder.packed_image_codec',
    branch: 'g',
    role: 'io',
    towerRole: 'denoiser',
    cx: 8.10,
    cy: TOP_IO_ROW_Y,
    cz: 0,
    ex: 2.55,
    ey: 1.05,
    ez: 1.10,
    cellFamily: CELL_SSM_CYAN,
  }),
  Object.freeze({
    id: 'gen_embedding',
    label: 'gen_embedding',
    sublabel: '[131,328 × 4,096] · VQ + 128 Fixed Prefix',
    tensorKey: 'encoder.gen_embedding.weight',
    branch: 'g',
    role: 'input',
    towerRole: 'denoiser',
    cx: UNI_MACRO_LAYOUT.gJointX,
    cy: TOP_IO_ROW_Y,
    cz: 0,
    ex: 2.85,
    ey: 1.24,
    ez: 1.25,
    cellFamily: CELL_ACTIVATION_GREEN,
  }),
  Object.freeze({
    id: 'diffusion_head',
    label: 'diffusion_head',
    sublabel: 'encoder.norm + Linear [4,096 → 132,101]',
    tensorKey: 'diffusion_head.weight',
    branch: 'u',
    role: 'output',
    towerRole: 'context',
    cx: UNI_MACRO_LAYOUT.uSplitX,
    cy: BOTTOM_IO_ROW_Y,
    cz: 0,
    ex: 2.90,
    ey: 1.28,
    ez: 1.25,
    cellFamily: CELL_WEIGHT_BLUE,
  }),
  Object.freeze({
    id: 'gen_predictor',
    label: 'gen_predictor',
    sublabel: 'encoder.norm_gen + Linear [4,096 → 131,072]',
    tensorKey: 'encoder.gen_predictor.weight',
    branch: 'g',
    role: 'output',
    towerRole: 'denoiser',
    cx: UNI_MACRO_LAYOUT.gSplitX,
    cy: BOTTOM_IO_ROW_Y,
    cz: 0,
    ex: 2.90,
    ey: 1.28,
    ez: 1.25,
    cellFamily: CELL_ACTIVATION_GREEN,
  }),
]);

// ============================================================================
// EXECUTION-VIEW & D5 STATE RESOLUTION
// ============================================================================

/**
 * Resolve per-tower, per-coupler, and per-I/O visual activity states for any of the 6
 * `UniExecutionViewId` modes and `D5` prefill mode (`hf_faithful` vs `elastic_pruned`).
 */
export function resolveUniMacroExecutionProfile(
  executionViewId: UniExecutionViewId,
  prefillMode: UniElasticPrefillMode,
): UniMacroExecutionProfile {
  switch (executionViewId) {
    case 'text_understanding':
      return Object.freeze({
        executionViewId,
        prefillMode,
        uJointState: 'active',
        uSplitState: 'active',
        gJointState: 'ghosted',
        gSplitState: 'ghosted',
        couplerState: 'ghosted',
        wastedSplitULayers: 0,
        activeIoModules: Object.freeze({
          embed_tokens: true,
          vision_tower: false,
          mm_projector: false,
          vqvae: false,
          packed_image_codec: false,
          gen_embedding: false,
          diffusion_head: true,
          gen_predictor: false,
        }),
        vqEncodeActive: false,
        vqDecodeActive: false,
      });

    case 'visual_understanding':
    case 'object_grounding':
      return Object.freeze({
        executionViewId,
        prefillMode,
        uJointState: 'active',
        uSplitState: 'active',
        gJointState: 'ghosted',
        gSplitState: 'ghosted',
        couplerState: 'ghosted',
        wastedSplitULayers: 0,
        activeIoModules: Object.freeze({
          embed_tokens: true,
          vision_tower: true,
          mm_projector: true,
          vqvae: false,
          packed_image_codec: false,
          gen_embedding: false,
          diffusion_head: true,
          gen_predictor: false,
        }),
        vqEncodeActive: false,
        vqDecodeActive: false,
      });

    case 'text_to_image': {
      const isHfFaithful = prefillMode === 'hf_faithful';
      return Object.freeze({
        executionViewId,
        prefillMode,
        uJointState: 'active',
        uSplitState: isHfFaithful ? 'wasted_kv' : 'pruned_ghost',
        gJointState: 'active',
        gSplitState: 'active',
        couplerState: 'active',
        wastedSplitULayers: isHfFaithful ? SPLIT_LAYERS : 0,
        activeIoModules: Object.freeze({
          embed_tokens: true,
          vision_tower: false,
          mm_projector: false,
          vqvae: true,
          packed_image_codec: true,
          gen_embedding: true,
          diffusion_head: false,
          gen_predictor: true,
        }),
        vqEncodeActive: false,
        vqDecodeActive: true,
      });
    }

    case 'image_edit': {
      const isHfFaithful = prefillMode === 'hf_faithful';
      return Object.freeze({
        executionViewId,
        prefillMode,
        uJointState: 'active',
        uSplitState: isHfFaithful ? 'wasted_kv' : 'pruned_ghost',
        gJointState: 'active',
        gSplitState: 'active',
        couplerState: 'active',
        wastedSplitULayers: isHfFaithful ? SPLIT_LAYERS : 0,
        activeIoModules: Object.freeze({
          embed_tokens: true,
          vision_tower: true,
          mm_projector: true,
          vqvae: true,
          packed_image_codec: true,
          gen_embedding: true,
          diffusion_head: false,
          gen_predictor: true,
        }),
        vqEncodeActive: true,
        vqDecodeActive: true,
      });
    }

    case 'interleaved_mixed':
      return Object.freeze({
        executionViewId,
        prefillMode,
        uJointState: 'active',
        uSplitState: 'active',
        gJointState: 'active',
        gSplitState: 'active',
        couplerState: 'active',
        wastedSplitULayers: 0,
        activeIoModules: Object.freeze({
          embed_tokens: true,
          vision_tower: true,
          mm_projector: true,
          vqvae: true,
          packed_image_codec: true,
          gen_embedding: true,
          diffusion_head: true,
          gen_predictor: true,
        }),
        vqEncodeActive: true,
        vqDecodeActive: true,
      });
  }
}

function formatBillions(params: number): string {
  return `${(params / 1_000_000_000).toFixed(2)}B`;
}

function executionViewOrdinal(viewId: UniExecutionViewId): number {
  switch (viewId) {
    case 'text_understanding':
      return 1;
    case 'visual_understanding':
      return 2;
    case 'object_grounding':
      return 3;
    case 'text_to_image':
      return 4;
    case 'image_edit':
      return 5;
    case 'interleaved_mixed':
      return 6;
  }
}

function resolveBayShading(
  baseCellFamily: number,
  state: UniMacroBayVisualState,
  isSelectedOrHovered: boolean,
): {
  readonly cellFamily: number;
  readonly fp8: number;
  readonly highlight: number;
  readonly lineStyle: number;
  readonly cssWidth: number;
  readonly lineAlpha: number;
} {
  if (state === 'wasted_kv') {
    return {
      cellFamily: CELL_MOE_AMBER,
      fp8: 0,
      highlight: isSelectedOrHovered ? 1 : 2,
      lineStyle: isSelectedOrHovered ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
      cssWidth: isSelectedOrHovered ? 1.85 : 1.45,
      lineAlpha: isSelectedOrHovered ? 1.0 : 0.94,
    };
  }
  if (state === 'pruned_ghost' || state === 'ghosted') {
    return {
      cellFamily: baseCellFamily,
      fp8: 1,
      highlight: isSelectedOrHovered ? 1 : 0,
      lineStyle: isSelectedOrHovered ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
      cssWidth: isSelectedOrHovered ? 1.55 : 0.78,
      lineAlpha: isSelectedOrHovered ? 0.92 : 0.30,
    };
  }
  return {
    cellFamily: baseCellFamily,
    fp8: 0,
    highlight: isSelectedOrHovered ? 1 : 0,
    lineStyle: isSelectedOrHovered ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: isSelectedOrHovered ? 1.85 : 1.02,
    lineAlpha: isSelectedOrHovered ? 1.0 : 0.68,
  };
}

// ============================================================================
// MAIN VIEW 1 SCENE BUILDER
// ============================================================================

/**
 * Build View 1 (`macro`) Elastic Macro Architecture spatial scene.
 *
 * Works both with a live `UniSession` (via non-mutating `resolveUniSessionContext`) and
 * standalone (`session === null`, defaulting to `text_to_image`, `elastic_pruned`, `prefixTokens = 16`).
 */
export function buildUniMacroScene(
  session: UniSession | null,
  options?: UniMacroSceneOptions,
): UniSpatialScene {
  const ctx = session ? resolveUniSessionContext(session, options?.recordId) : null;

  const executionViewId: UniExecutionViewId =
    options?.executionViewId ?? ctx?.executionViewId ?? 'text_to_image';
  const prefillMode: UniElasticPrefillMode =
    options?.prefillMode ?? ctx?.prefillMode ?? 'elastic_pruned';
  const rawPrefixTokens = options?.prefixTokens ?? (ctx ? ctx.branch.prefixCache.seqLen : 16);
  const prefixTokens = Math.max(1, Math.trunc(Number.isFinite(rawPrefixTokens) ? rawPrefixTokens : 16));

  const selectedLayerIndex =
    options?.selectedLayerIndex !== undefined && options.selectedLayerIndex !== null
      ? Math.max(0, Math.min(NUM_LAYERS - 1, Math.trunc(options.selectedLayerIndex)))
      : null;
  const hoveredPartId = options?.hoveredPartId ?? null;

  const viewReadouts = buildUniViewArchReadouts();
  const viewReadout =
    viewReadouts.find((v) => v.viewId === executionViewId) ?? viewReadouts[0]!;
  const d5Comparison = buildUniElasticD5Comparison(executionViewId, prefixTokens);
  const profile = resolveUniMacroExecutionProfile(executionViewId, prefillMode);
  const viewPhase = executionViewOrdinal(executionViewId) * 0.37;

  const solids = new UniSolidBufferBuilder();
  const lines = new UniLineBufferBuilder();
  const ribbons = new UniRibbonBufferBuilder();
  const flows = new UniFlatFlowBufferBuilder();
  const pickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const overlayAnchors: UniOverlayAnchor[] = [];
  const calipers: DimensionCaliper[] = [];

  const [bayEx, bayEy, bayEz] = UNI_MACRO_LAYOUT.bayExtent;
  const [coupEx, coupEy, coupEz] = UNI_MACRO_LAYOUT.couplerExtent;
  const [wasteEx, wasteEy, wasteEz] = UNI_MACRO_LAYOUT.wastedKvSlabExtent;

  const isD5GenerativeView =
    executionViewId === 'text_to_image' || executionViewId === 'image_edit';
  const wastedPctStr = (d5Comparison.wastedPrefixKvFraction * 100).toFixed(1);

  // --------------------------------------------------------------------------
  // 1. TWIN 34-LAYER BACKBONE (U LEFT x < 0, G RIGHT x > 0) + 18 JOINT COUPLERS
  // --------------------------------------------------------------------------

  for (let l = 0; l < NUM_LAYERS; l++) {
    const cy = uniMacroLayerCenterY(l);
    const bayId = l + 1;
    const isSplit = l >= DUAL_TOWER_LAYERS;
    const zone = isSplit ? 'split' : 'joint';
    const isLayerSelected = selectedLayerIndex === l;

    const uPartId = `u_layer_${l}`;
    const gPartId = `g_layer_${l}`;
    const uState = isSplit ? profile.uSplitState : profile.uJointState;
    const gState = isSplit ? profile.gSplitState : profile.gJointState;
    const uX = isSplit ? UNI_MACRO_LAYOUT.uSplitX : UNI_MACRO_LAYOUT.uJointX;
    const gX = isSplit ? UNI_MACRO_LAYOUT.gSplitX : UNI_MACRO_LAYOUT.gJointX;

    // --- Understanding Tower Bay (U, left x < 0) ---
    const uFocused = isLayerSelected || hoveredPartId === uPartId;
    const uShade = resolveBayShading(CELL_WEIGHT_BLUE, uState, uFocused);
    solids.push({
      cx: uX,
      cy,
      cz: 0,
      ex: bayEx,
      ey: bayEy,
      ez: bayEz,
      cellFamily: uShade.cellFamily,
      fp8: uShade.fp8,
      phase: l * 0.43 + viewPhase + (uState === 'wasted_kv' ? 7.0 : 0.0),
      bayId,
      cellScale: 0.24,
      highlight: uShade.highlight,
    });
    lines.addBoxEdges(
      uX,
      cy,
      0,
      bayEx,
      bayEy,
      bayEz,
      bayId,
      uShade.cssWidth,
      uShade.lineStyle,
      uShade.lineAlpha,
    );
    pickBoxes.push(
      packPickBoxFromAabb(uX, cy, 0, bayEx, bayEy, bayEz, {
        bayId,
        partId: uPartId,
        tensorKey: isSplit ? 'u_split_layer' : 'u_joint_layer',
        towerRole: 'context',
        semanticTag: `branch=u;zone=${zone};layer=${l};state=${uState};prefill=${prefillMode}`,
      }),
    );

    // D5 HF-Faithful wasted prefix-KV slabs protruding on the outer-left flank of U L18..33
    if (isSplit && uState === 'wasted_kv') {
      const wx = UNI_MACRO_LAYOUT.wastedKvSlabX;
      solids.push({
        cx: wx,
        cy,
        cz: 0.16,
        ex: wasteEx,
        ey: wasteEy,
        ez: wasteEz,
        cellFamily: CELL_MOE_AMBER,
        fp8: 0,
        phase: l * 0.71 + 3.1,
        bayId,
        cellScale: 0.19,
        highlight: uFocused ? 1 : 2,
      });
      lines.addBoxEdges(
        wx,
        cy,
        0.16,
        wasteEx,
        wasteEy,
        wasteEz,
        bayId,
        uFocused ? 1.55 : 1.15,
        uFocused ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
        0.92,
      );
      ribbons.push({
        cx: 0.5 * (uX - bayEx * 0.5 + wx + wasteEx * 0.5),
        cy,
        cz: 0.08,
        ex: 0.54,
        ey: 0.22,
        ez: 0.24,
        kind: RIBBON_KIND_ARROW_NEG_X,
        colorId: RIBBON_COLOR_AMBER,
        alpha: 0.78,
      });
    }

    // --- Generation Tower Bay (G, right x > 0) ---
    const gFocused = isLayerSelected || hoveredPartId === gPartId;
    const gShade = resolveBayShading(CELL_ACTIVATION_GREEN, gState, gFocused);
    solids.push({
      cx: gX,
      cy,
      cz: 0,
      ex: bayEx,
      ey: bayEy,
      ez: bayEz,
      cellFamily: gShade.cellFamily,
      fp8: gShade.fp8,
      phase: l * 0.59 + viewPhase + 13.0,
      bayId,
      cellScale: 0.24,
      highlight: gShade.highlight,
    });
    lines.addBoxEdges(
      gX,
      cy,
      0,
      bayEx,
      bayEy,
      bayEz,
      bayId,
      gShade.cssWidth,
      gShade.lineStyle,
      gShade.lineAlpha,
    );
    pickBoxes.push(
      packPickBoxFromAabb(gX, cy, 0, bayEx, bayEy, bayEz, {
        bayId,
        partId: gPartId,
        tensorKey: isSplit ? 'g_split_layer' : 'g_joint_layer',
        towerRole: 'denoiser',
        semanticTag: `branch=g;zone=${zone};layer=${l};state=${gState};prefill=${prefillMode}`,
      }),
    );

    // --- 18 Luminous Joint-Attention Cross-Couplers (L0..L17) ---
    if (!isSplit) {
      const couplerPartId = `joint_coupler_${l}`;
      const couplerFocused = isLayerSelected || hoveredPartId === couplerPartId;
      const couplerActive = profile.couplerState === 'active';
      const couplerShade = resolveBayShading(
        CELL_SSM_CYAN,
        couplerActive ? 'active' : 'ghosted',
        couplerFocused,
      );

      solids.push({
        cx: UNI_MACRO_LAYOUT.couplerX,
        cy,
        cz: 0,
        ex: coupEx,
        ey: coupEy,
        ez: coupEz,
        cellFamily: couplerShade.cellFamily,
        fp8: couplerShade.fp8,
        phase: l * 0.83 + viewPhase + 29.0,
        bayId,
        cellScale: 0.16,
        highlight: couplerFocused ? 1 : couplerActive ? 2 : 0,
      });
      lines.addBoxEdges(
        UNI_MACRO_LAYOUT.couplerX,
        cy,
        0,
        coupEx,
        coupEy,
        coupEz,
        bayId,
        couplerFocused ? 1.75 : couplerActive ? 1.15 : 0.72,
        couplerFocused ? LINE_STYLE_ACTIVE : couplerActive ? LINE_STYLE_NEUTRAL : LINE_STYLE_GUIDE,
        couplerFocused ? 1.0 : couplerActive ? 0.82 : 0.28,
      );
      pickBoxes.push(
        packPickBoxFromAabb(UNI_MACRO_LAYOUT.couplerX, cy, 0, coupEx, coupEy, coupEz, {
          bayId,
          partId: couplerPartId,
          tensorKey: 'joint_attention_coupler',
          towerRole: 'shared',
          semanticTag: `branch=joint;zone=joint;layer=${l};state=${profile.couplerState}`,
        }),
      );

      // Bidirectional flat flow bridges across U <-> G on the front face of every joint coupler
      const bridgeColor = couplerFocused
        ? UNI_FLOW_COLORS.activeOxide
        : couplerActive
          ? UNI_FLOW_COLORS.jointCyan
          : UNI_FLOW_COLORS.ghostSlate;
      const bridgeWidth = couplerFocused ? 0.11 : couplerActive ? 0.085 : 0.055;
      const uInnerX = UNI_MACRO_LAYOUT.uJointX + bayEx * 0.5;
      const gInnerX = UNI_MACRO_LAYOUT.gJointX - bayEx * 0.5;
      const bridgeZ = coupEz * 0.5 + 0.06;

      flows.addPolyline3D(
        [
          [uInnerX, cy + 0.08, bridgeZ],
          [gInnerX, cy + 0.08, bridgeZ],
        ],
        [0, 1, 0],
        bridgeWidth,
        bridgeColor,
        true,
        0.24,
        0.18,
        false,
      );
      flows.addPolyline3D(
        [
          [gInnerX, cy - 0.08, bridgeZ],
          [uInnerX, cy - 0.08, bridgeZ],
        ],
        [0, 1, 0],
        bridgeWidth,
        bridgeColor,
        true,
        0.24,
        0.18,
        false,
      );
    }
  }

  // --------------------------------------------------------------------------
  // 2. EXPLICIT L17 | L18 ELASTIC SPLIT DIVIDER DATUM / RAIL
  // --------------------------------------------------------------------------

  const divY = UNI_MACRO_LAYOUT.splitDividerY;
  const divSpanX = UNI_MACRO_LAYOUT.dividerSpanHalfX;

  // Translucent split divider rail bar across the elastic boundary
  ribbons.push({
    cx: 0,
    cy: divY,
    cz: 0,
    ex: divSpanX * 2,
    ey: 0.10,
    ez: 1.55,
    kind: RIBBON_KIND_BOX,
    colorId:
      isD5GenerativeView && prefillMode === 'hf_faithful'
        ? RIBBON_COLOR_AMBER
        : RIBBON_COLOR_CYAN,
    alpha: 0.24,
  });

  // Crisp drafting datum hairlines framing the L17 | L18 split plane
  lines.addDashedFrameXZ(
    -divSpanX,
    -0.88,
    divSpanX,
    0.88,
    divY,
    isD5GenerativeView && prefillMode === 'hf_faithful' ? LINE_STYLE_BF16_GOLD : LINE_STYLE_GUIDE,
    0.82,
    0.34,
    0.18,
    1.25,
  );
  lines.addSegment(
    -divSpanX - 0.45,
    divY,
    0.88,
    divSpanX + 0.45,
    divY,
    0.88,
    1.35,
    isD5GenerativeView && prefillMode === 'hf_faithful' ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
    0.72,
  );

  // Small end-cap datum markers at left & right ends of the L17 | L18 rail
  for (const capX of [-divSpanX - 0.15, divSpanX + 0.15]) {
    solids.push({
      cx: capX,
      cy: divY,
      cz: 0,
      ex: 0.30,
      ey: 0.24,
      ez: 0.30,
      cellFamily:
        isD5GenerativeView && prefillMode === 'hf_faithful' ? CELL_MOE_AMBER : CELL_SSM_CYAN,
      fp8: 0,
      phase: 17.5,
      bayId: 0,
      cellScale: 0.14,
      highlight: 2,
    });
  }

  // Elastic-Pruned cutoff bar right below U L17 to dramatize 18L U prefill stop (0-ULP Stage-2)
  if (isD5GenerativeView && prefillMode === 'elastic_pruned') {
    const cutoffY = LAYER_17_Y - bayEy * 0.5 - 0.42;
    ribbons.push({
      cx: UNI_MACRO_LAYOUT.uJointX,
      cy: cutoffY,
      cz: 0,
      ex: bayEx + 0.65,
      ey: 0.16,
      ez: bayEz + 0.30,
      kind: RIBBON_KIND_BOX,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: 0.68,
    });
    lines.addBoxFrameXZ(
      UNI_MACRO_LAYOUT.uJointX - bayEx * 0.5 - 0.32,
      -bayEz * 0.5 - 0.15,
      UNI_MACRO_LAYOUT.uJointX + bayEx * 0.5 + 0.32,
      bayEz * 0.5 + 0.15,
      cutoffY,
      1.4,
      LINE_STYLE_ACTIVE,
      0.85,
    );
  }

  // --------------------------------------------------------------------------
  // 3. VERTICAL RESIDUAL SPINES & L17 -> L18 SPLIT-FLARE CONNECTORS
  // --------------------------------------------------------------------------

  const jointTopY = LAYER_0_Y + bayEy * 0.5;
  const jointBotY = LAYER_17_Y - bayEy * 0.5;
  const splitTopY = LAYER_18_Y + bayEy * 0.5;
  const splitBotY = LAYER_33_Y - bayEy * 0.5;

  // U Joint spine (always active)
  ribbons.push({
    cx: UNI_MACRO_LAYOUT.uJointSpineX,
    cy: 0.5 * (jointTopY + jointBotY),
    cz: 0,
    ex: 0.28,
    ey: jointTopY - jointBotY,
    ez: 0.28,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.48,
  });

  // U Split spine (active indigo, wasted amber in HF-Faithful, or faint ghost when pruned)
  const uSplitSpineColorId =
    profile.uSplitState === 'wasted_kv'
      ? RIBBON_COLOR_AMBER
      : RIBBON_COLOR_PERIWINKLE;
  const uSplitSpineAlpha =
    profile.uSplitState === 'wasted_kv'
      ? 0.64
      : profile.uSplitState === 'active'
        ? 0.48
        : 0.12;
  ribbons.push({
    cx: UNI_MACRO_LAYOUT.uSplitSpineX,
    cy: 0.5 * (splitTopY + splitBotY),
    cz: 0,
    ex: profile.uSplitState === 'wasted_kv' ? 0.36 : 0.28,
    ey: splitTopY - splitBotY,
    ez: profile.uSplitState === 'wasted_kv' ? 0.36 : 0.28,
    kind: RIBBON_KIND_BOX,
    colorId: uSplitSpineColorId,
    alpha: uSplitSpineAlpha,
  });

  // G Joint & G Split spines
  const gActive = profile.gJointState === 'active';
  ribbons.push({
    cx: UNI_MACRO_LAYOUT.gJointSpineX,
    cy: 0.5 * (jointTopY + jointBotY),
    cz: 0,
    ex: 0.28,
    ey: jointTopY - jointBotY,
    ez: 0.28,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: gActive ? 0.48 : 0.12,
  });
  ribbons.push({
    cx: UNI_MACRO_LAYOUT.gSplitSpineX,
    cy: 0.5 * (splitTopY + splitBotY),
    cz: 0,
    ex: 0.28,
    ey: splitTopY - splitBotY,
    ez: 0.28,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: profile.gSplitState === 'active' ? 0.48 : 0.12,
  });

  // Smooth 3D cubic Bezier flare ribbons between L17 and L18 on both U and G towers
  const uFlareColor =
    profile.uSplitState === 'wasted_kv'
      ? UNI_FLOW_COLORS.wasteAmber
      : profile.uSplitState === 'active'
        ? UNI_FLOW_COLORS.uIndigo
        : UNI_FLOW_COLORS.ghostSlate;
  flows.addCubic3D(
    [UNI_MACRO_LAYOUT.uJointX, jointBotY, 0.35],
    [UNI_MACRO_LAYOUT.uJointX, divY + 0.25, 0.35],
    [UNI_MACRO_LAYOUT.uSplitX, divY - 0.25, 0.35],
    [UNI_MACRO_LAYOUT.uSplitX, splitTopY, 0.35],
    [0, 0, 1],
    profile.uSplitState === 'pruned_ghost' ? 0.10 : 0.20,
    uFlareColor,
    true,
    18,
    true,
    0.34,
    0.30,
  );

  const gFlareColor =
    profile.gSplitState === 'active'
      ? UNI_FLOW_COLORS.gTeal
      : UNI_FLOW_COLORS.ghostSlate;
  flows.addCubic3D(
    [UNI_MACRO_LAYOUT.gJointX, jointBotY, 0.35],
    [UNI_MACRO_LAYOUT.gJointX, divY + 0.25, 0.35],
    [UNI_MACRO_LAYOUT.gSplitX, divY - 0.25, 0.35],
    [UNI_MACRO_LAYOUT.gSplitX, splitTopY, 0.35],
    [0, 0, 1],
    profile.gSplitState === 'active' ? 0.20 : 0.10,
    gFlareColor,
    true,
    18,
    true,
    0.34,
    0.30,
  );

  // Amber vertical waste bracket along the left flank of the 16 wasted KV slabs in HF-Faithful
  if (profile.uSplitState === 'wasted_kv') {
    const bracketX = UNI_MACRO_LAYOUT.wastedKvSlabX - wasteEx * 0.5 - 0.38;
    flows.addPolyline3D(
      [
        [bracketX + 0.26, splitTopY, 0.16],
        [bracketX, splitTopY, 0.16],
        [bracketX, splitBotY, 0.16],
        [bracketX + 0.26, splitBotY, 0.16],
      ],
      [0, 0, 1],
      0.13,
      UNI_FLOW_COLORS.wasteAmber,
      false,
      0.2,
      0.2,
      false,
    );
  }

  // --------------------------------------------------------------------------
  // 4. ALL 8 I/O BOOKEND MODULES + TASK-REACTIVE I/O FLOW RIBBONS
  // --------------------------------------------------------------------------

  const bookendById = new Map<UniMacroIoModuleId, IoBookendGeometrySpec>();
  for (let i = 0; i < IO_BOOKENDS.length; i++) {
    const spec = IO_BOOKENDS[i]!;
    bookendById.set(spec.id, spec);

    const active = profile.activeIoModules[spec.id];
    const focused = hoveredPartId === spec.id;
    const shade = resolveBayShading(spec.cellFamily, active ? 'active' : 'ghosted', focused);

    solids.push({
      cx: spec.cx,
      cy: spec.cy,
      cz: spec.cz,
      ex: spec.ex,
      ey: spec.ey,
      ez: spec.ez,
      cellFamily: shade.cellFamily,
      fp8: shade.fp8,
      phase: 100 + i * 2.17 + viewPhase,
      bayId: 0,
      cellScale: 0.24,
      highlight: shade.highlight,
    });
    lines.addBoxEdges(
      spec.cx,
      spec.cy,
      spec.cz,
      spec.ex,
      spec.ey,
      spec.ez,
      0,
      shade.cssWidth,
      shade.lineStyle,
      shade.lineAlpha,
    );
    pickBoxes.push(
      packPickBoxFromAabb(spec.cx, spec.cy, spec.cz, spec.ex, spec.ey, spec.ez, {
        bayId: 0,
        partId: spec.id,
        tensorKey: spec.tensorKey,
        towerRole: spec.towerRole,
        semanticTag: `branch=${spec.branch};module=${spec.id};role=${spec.role};state=${active ? 'active' : 'ghosted'}`,
      }),
    );
  }

  const embedSpec = bookendById.get('embed_tokens')!;
  const visionSpec = bookendById.get('vision_tower')!;
  const projSpec = bookendById.get('mm_projector')!;
  const vqSpec = bookendById.get('vqvae')!;
  const codecSpec = bookendById.get('packed_image_codec')!;
  const genEmbSpec = bookendById.get('gen_embedding')!;
  const diffHeadSpec = bookendById.get('diffusion_head')!;
  const genPredSpec = bookendById.get('gen_predictor')!;

  // Input ribbon: embed_tokens -> U L0
  const embedActive = profile.activeIoModules.embed_tokens;
  ribbons.push({
    cx: embedSpec.cx,
    cy: 0.5 * (embedSpec.cy - embedSpec.ey * 0.5 + jointTopY),
    cz: 0,
    ex: 0.44,
    ey: embedSpec.cy - embedSpec.ey * 0.5 - jointTopY - 0.12,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: embedActive ? 0.68 : 0.16,
  });

  // Input ribbon: vision_tower -> mm_projector -> embed_tokens
  const visionActive = profile.activeIoModules.vision_tower;
  flows.addPolyline3D(
    [
      [visionSpec.cx, visionSpec.cy - visionSpec.ey * 0.5, 0.2],
      [projSpec.cx, projSpec.cy + projSpec.ey * 0.5, 0.2],
    ],
    [0, 0, 1],
    visionActive ? 0.20 : 0.10,
    visionActive ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.ghostSlate,
    true,
    0.34,
    0.30,
    true,
  );
  ribbons.push({
    cx: 0.5 * (projSpec.cx + projSpec.ex * 0.5 + embedSpec.cx - embedSpec.ex * 0.5),
    cy: projSpec.cy,
    cz: 0,
    ex: embedSpec.cx - embedSpec.ex * 0.5 - (projSpec.cx + projSpec.ex * 0.5) - 0.14,
    ey: 0.34,
    ez: 0.28,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: visionActive ? 0.66 : 0.14,
  });

  // Input ribbon: gen_embedding -> G L0
  const genEmbActive = profile.activeIoModules.gen_embedding;
  ribbons.push({
    cx: genEmbSpec.cx,
    cy: 0.5 * (genEmbSpec.cy - genEmbSpec.ey * 0.5 + jointTopY),
    cz: 0,
    ex: 0.44,
    ey: genEmbSpec.cy - genEmbSpec.ey * 0.5 - jointTopY - 0.12,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: genEmbActive ? 0.68 : 0.16,
  });

  // Discrete encode path: vqvae -> packed_image_codec -> gen_embedding (active in Image Edit & Interleaved)
  flows.addPolyline3D(
    [
      [vqSpec.cx, vqSpec.cy - vqSpec.ey * 0.5, 0.2],
      [codecSpec.cx, codecSpec.cy + codecSpec.ey * 0.5, 0.2],
    ],
    [0, 0, 1],
    profile.vqEncodeActive ? 0.20 : 0.10,
    profile.vqEncodeActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate,
    true,
    0.34,
    0.30,
    true,
  );
  ribbons.push({
    cx: 0.5 * (genEmbSpec.cx + genEmbSpec.ex * 0.5 + codecSpec.cx - codecSpec.ex * 0.5),
    cy: codecSpec.cy,
    cz: 0,
    ex: codecSpec.cx - codecSpec.ex * 0.5 - (genEmbSpec.cx + genEmbSpec.ex * 0.5) - 0.14,
    ey: 0.34,
    ez: 0.28,
    kind: RIBBON_KIND_ARROW_NEG_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: profile.activeIoModules.packed_image_codec ? 0.64 : 0.14,
  });

  // Output ribbon: U L33 -> diffusion_head
  const diffHeadActive = profile.activeIoModules.diffusion_head;
  ribbons.push({
    cx: diffHeadSpec.cx,
    cy: 0.5 * (splitBotY + diffHeadSpec.cy + diffHeadSpec.ey * 0.5),
    cz: 0,
    ex: 0.44,
    ey: splitBotY - (diffHeadSpec.cy + diffHeadSpec.ey * 0.5) - 0.12,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: diffHeadActive ? 0.68 : 0.14,
  });

  // Output ribbon: G L33 -> gen_predictor
  const genPredActive = profile.activeIoModules.gen_predictor;
  ribbons.push({
    cx: genPredSpec.cx,
    cy: 0.5 * (splitBotY + genPredSpec.cy + genPredSpec.ey * 0.5),
    cz: 0,
    ex: 0.44,
    ey: splitBotY - (genPredSpec.cy + genPredSpec.ey * 0.5) - 0.12,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: genPredActive ? 0.68 : 0.14,
  });

  // Output decode feedback arc: gen_predictor -> right outer lane -> packed_image_codec / vqvae (T2I, Edit, Interleaved)
  if (profile.vqDecodeActive) {
    const outerDecodeX = 9.85;
    flows.addPolyline3D(
      [
        [genPredSpec.cx + genPredSpec.ex * 0.5, genPredSpec.cy, 0.25],
        [outerDecodeX, genPredSpec.cy, 0.25],
        [outerDecodeX, vqSpec.cy, 0.25],
        [vqSpec.cx + vqSpec.ex * 0.5, vqSpec.cy, 0.25],
      ],
      [0, 0, 1],
      0.16,
      UNI_FLOW_COLORS.gTeal,
      true,
      0.38,
      0.32,
      true,
    );
  }

  // --------------------------------------------------------------------------
  // 5. LANDMARKS, OVERLAY ANCHORS & DIMENSION CALIPERS
  // --------------------------------------------------------------------------

  landmarks.push(
    {
      id: 'u-tower-header',
      label: 'Understanding Tower (U) · 34L',
      sublabel: 'Text + RADIO-H · 32Q/8KV GQA + 3.5× SwiGLU',
      world: [UNI_MACRO_LAYOUT.uJointX, TOP_ENCODER_ROW_Y + 1.35, 0],
      tone: 'default',
      align: 'center',
      tier: 'primary',
    },
    {
      id: 'g-tower-header',
      label: 'Generation Tower (G) · 34L',
      sublabel: 'Emu3 VQ + 128 Prefix · Semantic-Tree Diffusion',
      world: [UNI_MACRO_LAYOUT.gJointX, TOP_ENCODER_ROW_Y + 1.35, 0],
      tone: 'cyan',
      align: 'center',
      tier: 'primary',
    },
    {
      id: 'elastic-split-boundary',
      label: 'L17 | L18 Elastic Split Boundary',
      sublabel: '18 Joint-Attention (52.9%) → 16 Split-Tower (47.1%)',
      world: [0, divY, 0.95],
      tone: isD5GenerativeView && prefillMode === 'hf_faithful' ? 'gold' : 'cyan',
      align: 'center',
      tier: 'primary',
    },
    {
      id: 'layer-0-anchor',
      label: 'L0 · Joint Attention Entry',
      world: [0, LAYER_0_Y, 0.45],
      bayId: 1,
      partId: 'joint_coupler_0',
      tone: 'cyan',
      align: 'center',
      tier: 'detail',
    },
    {
      id: 'layer-33-anchor',
      label: 'L33 · Split-Tower Exit',
      world: [0, LAYER_33_Y, 0.45],
      bayId: 34,
      tone: 'muted',
      align: 'center',
      tier: 'detail',
    },
  );

  // D5 split-U landmark reacting directly to executionViewId + prefillMode
  const splitUMidY = 0.5 * (LAYER_18_Y + LAYER_33_Y);
  if (isD5GenerativeView) {
    if (prefillMode === 'hf_faithful') {
      landmarks.push({
        id: 'd5-split-u-status',
        label: `D5 HF-Faithful · ${SPLIT_LAYERS}L Wasted U Prefix KV (${wastedPctStr}%)`,
        sublabel: `L18..33 U cached & dropped · ${formatSiIecBytes(d5Comparison.prodSavedPrefixKvBytes)} wasted @ ${prefixTokens} tok`,
        world: [UNI_MACRO_LAYOUT.wastedKvSlabX - 1.15, splitUMidY, 0.25],
        bayId: 19,
        partId: 'u_layer_18',
        tone: 'gold',
        align: 'left',
        tier: 'primary',
      });
    } else {
      landmarks.push({
        id: 'd5-split-u-status',
        label: 'D5 Elastic-Pruned · 0 Wasted KV (0-ULP Stage-2)',
        sublabel: `18L U + 34L G · saves ${formatSiIecBytes(d5Comparison.prodSavedPrefixKvBytes)} (${wastedPctStr}%)`,
        world: [UNI_MACRO_LAYOUT.uSplitX - bayEx * 0.5 - 1.25, splitUMidY, 0.25],
        bayId: 19,
        partId: 'u_layer_18',
        tone: 'cyan',
        align: 'left',
        tier: 'primary',
      });
    }
  } else if (executionViewId === 'interleaved_mixed') {
    landmarks.push({
      id: 'd5-split-u-status',
      label: 'Dual 34L Mixed Routing (68 Blocks)',
      sublabel: UNI_SCENE_INTERLEAVED_BADGE,
      world: [0, splitUMidY, 0.35],
      tone: 'gold',
      align: 'center',
      tier: 'primary',
    });
  } else {
    landmarks.push({
      id: 'd5-split-u-status',
      label: 'U-Only 34L Execution · G Tower Omitted',
      sublabel: `${formatBillions(viewReadout.prodActiveParams)} active · 0 wasted KV`,
      world: [UNI_MACRO_LAYOUT.gSplitX + bayEx * 0.5 + 1.0, splitUMidY, 0.25],
      tone: 'muted',
      align: 'right',
      tier: 'primary',
    });
  }

  for (const spec of IO_BOOKENDS) {
    landmarks.push({
      id: `io-${spec.id}`,
      label: spec.label,
      sublabel: spec.sublabel,
      world: [spec.cx, spec.cy, spec.cz + spec.ez * 0.5 + 0.15],
      bayId: 0,
      partId: spec.id,
      tone: profile.activeIoModules[spec.id]
        ? spec.branch === 'g'
          ? 'cyan'
          : 'default'
        : 'muted',
      align: spec.branch === 'u' ? 'left' : 'right',
      tier: 'detail',
    });
  }

  // Overlay anchors for pooled DOM/SVG drafting callouts
  overlayAnchors.push(
    {
      id: 'anchor-u-tower',
      label: 'Understanding (U)',
      sublabel:
        isD5GenerativeView && prefillMode === 'elastic_pruned'
          ? '18L Active · 16L Pruned'
          : isD5GenerativeView && prefillMode === 'hf_faithful'
            ? '18L Active + 16L Wasted KV'
            : '34L Active',
      worldPos: [UNI_MACRO_LAYOUT.uJointX, LAYER_0_Y + 1.15, 0.75],
      tone: 'u_indigo',
    },
    {
      id: 'anchor-g-tower',
      label: 'Generation (G)',
      sublabel: gActive ? '34L Active (18 Joint + 16 Split)' : 'Omitted (0L)',
      worldPos: [UNI_MACRO_LAYOUT.gJointX, LAYER_0_Y + 1.15, 0.75],
      tone: gActive ? 'g_teal' : 'muted',
    },
    {
      id: 'anchor-joint-band',
      label: '18 Joint-Attention Couplers',
      sublabel: 'L0..L17 · Cross-Modal U ↔ G Attention',
      worldPos: [0, 0.5 * (LAYER_0_Y + LAYER_17_Y), 0.70],
      tone: profile.couplerState === 'active' ? 'joint' : 'muted',
    },
    {
      id: 'anchor-split-divider',
      label: 'L17 | L18 Elastic Split',
      sublabel: 'split_tower = layer_idx >= 18',
      worldPos: [0, divY, 0.95],
      tone: 'split',
    },
    {
      id: 'anchor-d5-comparison',
      label:
        isD5GenerativeView && prefillMode === 'hf_faithful'
          ? `HF-Faithful: 16L U Prefix KV Waste (${wastedPctStr}%)`
          : isD5GenerativeView && prefillMode === 'elastic_pruned'
            ? `Elastic-Pruned: 0 Wasted KV (Saved ${wastedPctStr}%)`
            : `${viewReadout.label} (${viewReadout.paperShorthand})`,
      sublabel: isD5GenerativeView
        ? `Stage-2 Equivalence: ${d5Comparison.stage2UlpDiff} ULP · B_cfg=${d5Comparison.cfgBranches}`
        : executionViewId === 'interleaved_mixed'
          ? UNI_SCENE_INTERLEAVED_BADGE
          : `Active ${formatBillions(viewReadout.prodActiveParams)} · Resident ${formatSiIecBytes(viewReadout.prodResidentHfBytes)}`,
      worldPos: [
        isD5GenerativeView ? UNI_MACRO_LAYOUT.wastedKvSlabX : 0,
        splitUMidY,
        0.75,
      ],
      tone:
        isD5GenerativeView && prefillMode === 'hf_faithful'
          ? 'warn'
          : isD5GenerativeView && prefillMode === 'elastic_pruned'
            ? 'g_teal'
            : executionViewId === 'interleaved_mixed'
              ? 'warn'
              : 'u_indigo',
      badge:
        executionViewId === 'interleaved_mixed'
          ? UNI_SCENE_INTERLEAVED_BADGE
          : isD5GenerativeView
            ? prefillMode === 'hf_faithful'
              ? '16 Wasted Split-U Bays'
              : '0 Wasted KV · 0 ULP'
            : undefined,
    },
  );

  // Dimension calipers along outer drafting edges
  calipers.push(
    {
      id: 'caliper-joint-18l',
      aWorld: [7.55, LAYER_0_Y + bayEy * 0.5, 0],
      bWorld: [7.55, LAYER_17_Y - bayEy * 0.5, 0],
      label: '18L Joint Attention (L0..17 · 52.9%)',
      minPixels: 36,
      offsetPx: 14,
    },
    {
      id: 'caliper-split-16l',
      aWorld: [7.55, LAYER_18_Y + bayEy * 0.5, 0],
      bWorld: [7.55, LAYER_33_Y - bayEy * 0.5, 0],
      label:
        isD5GenerativeView && prefillMode === 'hf_faithful'
          ? `16L Split-Tower (L18..33 · U Prefill Waste ${wastedPctStr}%)`
          : isD5GenerativeView && prefillMode === 'elastic_pruned'
            ? `16L Split-Tower (L18..33 · U Pruned · 0 Wasted KV)`
            : '16L Split-Tower (L18..33 · 47.1%)',
      minPixels: 36,
      offsetPx: 14,
    },
    {
      id: 'caliper-total-34l',
      aWorld: [-9.20, LAYER_0_Y + bayEy * 0.5, 0],
      bWorld: [-9.20, LAYER_33_Y - bayEy * 0.5, 0],
      label: '34L Elastic-MoT Backbone (18 Joint + 16 Split)',
      minPixels: 48,
      offsetPx: 16,
    },
  );

  // --------------------------------------------------------------------------
  // 6. PROVENANCE BADGES & FINAL ZERO-ATLAS SCENE ASSEMBLY
  // --------------------------------------------------------------------------

  const provenanceBadges: UniProvenanceBadge[] = [
    {
      label: `Task: ${viewReadout.label}`,
      tone:
        executionViewId === 'interleaved_mixed'
          ? 'joint'
          : isD5GenerativeView
            ? 'g_teal'
            : 'u_indigo',
    },
    {
      label: isD5GenerativeView
        ? prefillMode === 'hf_faithful'
          ? `D5: HF-Faithful (34L U Prefill · ${SPLIT_LAYERS}L / ${wastedPctStr}% KV Waste)`
          : `D5: Elastic-Pruned (18L U + 34L G · 0 Wasted KV · ${d5Comparison.stage2UlpDiff} ULP)`
        : prefillMode === 'elastic_pruned'
          ? `D5: Elastic-Pruned (34L Active · ${d5Comparison.stage2UlpDiff} ULP)`
          : 'D5: HF-Faithful (34L Active)',
      tone:
        isD5GenerativeView && prefillMode === 'hf_faithful'
          ? 'warn'
          : prefillMode === 'elastic_pruned'
            ? 'g_teal'
            : 'gold',
    },
  ];

  if (executionViewId === 'interleaved_mixed') {
    provenanceBadges.push({
      label: UNI_SCENE_INTERLEAVED_BADGE,
      tone: 'warn',
    });
  }

  if (ctx) {
    provenanceBadges.push({
      label:
        ctx.gainPresetId === 'prod_exact'
          ? 'D3 Gains: Production-Exact'
          : 'D3 Gains: Mini Default',
      tone: 'muted',
    });
    if (ctx.isHistoricalCursor) {
      provenanceBadges.push({
        label: `Historical Cursor (${ctx.record.recordId})`,
        tone: 'gold',
      });
    }
  }

  provenanceBadges.push({
    label: UNI_SCENE_PROJECTION_REVERSAL_BADGE,
    tone: 'muted',
  });

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedFlows = flows.pack();

  const subtitle = isD5GenerativeView
    ? prefillMode === 'hf_faithful'
      ? `${viewReadout.label} · HF-Faithful 34L U Prefill (${SPLIT_LAYERS}L / ${wastedPctStr}% prefix KV wasted) · Active ${formatBillions(viewReadout.prodActiveParams)} vs Eager HF ${formatBillions(viewReadout.prodEagerHfParams)}`
      : `${viewReadout.label} · Elastic-Pruned (18L U + 34L G · 0 wasted KV · ${d5Comparison.stage2UlpDiff}-ULP Stage-2) · Active ${formatBillions(viewReadout.prodActiveParams)}`
    : executionViewId === 'interleaved_mixed'
      ? `${viewReadout.label} (${viewReadout.paperShorthand}) · ${UNI_SCENE_INTERLEAVED_BADGE}`
      : `${viewReadout.label} (${viewReadout.paperShorthand}) · Active ${formatBillions(viewReadout.prodActiveParams)} · 34L U (G omitted)`;

  return Object.freeze({
    viewMode: 'macro',
    title: 'Elastic Macro Architecture (34L Twin-Tower Elastic-MoT)',
    subtitle,
    layerIndex: selectedLayerIndex,
    executionViewId,
    prefillMode,
    solidData: packedSolids.data,
    solidCount: packedSolids.count,
    lineData: packedLines.data,
    lineCount: packedLines.count,
    ribbonData: packedRibbons.data,
    ribbonCount: packedRibbons.count,
    sheetData: new Float32Array(0),
    sheetBasisData: new Float32Array(0),
    sheetRevealData: new Float32Array(0),
    sheetCount: 0,
    flowVertices: packedFlows.flowVertices,
    flowIndices: packedFlows.flowIndices,
    pickBoxes: Object.freeze(pickBoxes),
    cameraPreset: Object.freeze({
      azimuthDeg: 24,
      elevationDeg: 14,
      fitBounds: MACRO_FIT_BOUNDS,
    }),
    fitBounds: MACRO_FIT_BOUNDS,
    valueAtlas: null,
    atlasKey: null,
    atlasRegions: Object.freeze([]),
    atlasAudit: null,
    rgbPreviews: Object.freeze([]),
    provenanceBadges: Object.freeze(provenanceBadges),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    d5Comparison,
    viewReadout,
  });
}
