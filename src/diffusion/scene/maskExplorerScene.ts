/**
 * 5-Tab Attention Mask & 3-Coordinate Explorer (`src/diffusion/scene/maskExplorerScene.ts`)
 *
 * Interactive drafting plate controlled by `focus.selectedMaskTab` using real binary masks
 * from `src/diffusion/masks.ts`:
 *   1. `'joint_training'`: `buildJointTrainingMask(6, 2)` (`12×12` `[x_t || x_0]`), badged
 *      `Read-Only / No Gradients`, encoding `M_BD` / `M_OBC` / `M_BC` distinctly in `R` and
 *      masked cells as `G = 1.0`, with 3-coordinate axis strips (`seq_idx`, `rope position_ids`,
 *      `cache_position`), strict `M_OBC (b_q > b_k, Δp >= 1)` verification readout, and `D23`
 *      Llama-4 scale asymmetry callout.
 *   2. `'ar_causal'`: `buildArCausalMask(0, 4)` (`4×4` AR Prefill) AND `buildArCausalMask(4, 1)`
 *      (`1×5` AR Decode) side-by-side plates, both bound to live `RG32F` `G ∈ {0.0, 1.0}`.
 *   3. `'block_bidir'`: `buildBlockBidirMask(4, 4)` (`4×8` Block-Diff Denoise / Spec Draft —
 *      all visible `G = 0.0`, proving `SheetsPass` does NOT fall back to geometric `cellCol > cellRow`).
 *   4. `'causal_commit_verify'`: `buildCausalCommitVerifyMask(4, 4)` (`4×8` Causal Commit /
 *      Spec Verify — lower triangle visible `G = 0.0`, upper block triangle masked `G = 1.0`).
 *   5. `'vlm_quadratic_ss'`: `buildVlmQuadraticSsMask(2, 2)` (`6×8` `Quadratic SS · VLM/Paper Ref`,
 *      badged Read-Only Reference), showing interleaved `k(k+1)` queries & reordered
 *      `[draft_kv || clean_kv]` KV layout and overlapping RoPE `position_ids` vs `cache_position`.
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import type { LayerFormulaReadout, StepperStepDef } from '../../math/formulaFormat';
import { fmtSigned } from '../../shared/format';
import type { AABB } from '../../render/camera';
import { LINE_STYLE_BF16_GOLD, LINE_STYLE_GUIDE, LINE_STYLE_NEUTRAL } from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import { FlowRibbonBuilder } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NORM,
  SHEET_ROLE_SCORES,
} from '../../render/sheets';
import {
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
} from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  drawCellHighlight3D,
  spanFromSheetGrid,
  type SheetGeomRecord,
} from '../../scene/sheetInspector';
import type { DimensionCaliper, LandmarkAnchor } from '../../render/primitives';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../shared/overlayLayout';
import {
  buildArCausalMask,
  buildBlockBidirMask,
  buildCausalCommitVerifyMask,
  buildJointTrainingMask,
  buildVlmQuadraticSsMask,
  verifyMobcStrictOffset,
  type MobcStrictOffsetCheck,
} from '../masks';
import type {
  DiffusionExplorerMaskMode,
  DiffusionMaskMatrix,
  JointTrainingMaskRegion,
} from '../types';
import {
  auditAtlasRegions,
  DIFFUSION_EPISTEMIC_NOTE,
  LineBufferBuilder,
  RibbonBufferBuilder,
  SheetBufferBuilder,
  SolidBufferBuilder,
} from './provenance';
import type {
  DiffusionAtlasRegionBox,
  DiffusionFocusState,
  DiffusionPickIdentity,
  DiffusionProvenanceBadge,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
} from './types';

export const DIFFUSION_MASK_ATLAS_SIZE = 64 as const;

export const DIFFUSION_MASK_TAB_STEPPER: readonly StepperStepDef[] = Object.freeze([
  Object.freeze({
    index: 0,
    shortLabel: 'Joint [x_t‖x_0]',
    title: 'Joint Training Dual-Stream Mask (Read-Only)',
    summary: '12×12 [x_t || x_0] separating M_BD, strict M_OBC (b_q > b_k, Δp >= 1), and causal M_BC + D23 3-coordinate divergence.',
  }),
  Object.freeze({
    index: 1,
    shortLabel: 'AR Causal',
    title: 'AR Causal Prefill (4×4) & Decode (1×5)',
    summary: 'Standard token-causal prefill and single-token autoregressive decode masks.',
  }),
  Object.freeze({
    index: 2,
    shortLabel: 'Block Bidir',
    title: 'Block-Diff Denoise / Spec Draft (4×8)',
    summary: 'All 32 cells visible (G = 0.0) over [prefix(0..3) || block(4..7)] with transient use_cache=False.',
  }),
  Object.freeze({
    index: 3,
    shortLabel: 'Commit / Verify',
    title: 'Causal Commit / Spec Verify (4×8)',
    summary: 'Lower-triangle causal mask over active block (G = 1.0 upper block triangle), committing KV with use_cache=True.',
  }),
  Object.freeze({
    index: 4,
    shortLabel: 'VLM Quad SS',
    title: 'Quadratic SS · VLM/Paper Ref (6×8)',
    summary: 'Read-only VLM-8B reference showing interleaved k(k+1) queries, reordered [draft_kv || clean_kv], and overlapping RoPE IDs.',
  }),
]);

interface MaskPlateSpec {
  readonly id: string;
  readonly tensorKey: string;
  readonly title: string;
  readonly mask: DiffusionMaskMatrix;
  readonly cx: number;
  readonly cz: number;
  readonly cellSize: number;
  readonly atlasX0: number;
  readonly atlasY0: number;
  readonly rowLabels: readonly string[];
  readonly colLabels: readonly string[];
  readonly cellRegionFn: (r: number, c: number) => string;
  readonly cellValueFn: (r: number, c: number, visible: boolean) => number;
}

interface CoordStripSpec {
  readonly id: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly sublabel: string;
  readonly values: Float32Array;
  readonly labels: readonly string[];
  readonly cx: number;
  readonly cz: number;
  readonly ex: number;
  readonly atlasX0: number;
  readonly atlasY0: number;
  readonly role: number;
  readonly highlight: number;
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

function packMaskPlate(
  writer: Rg32fAtlasWriter,
  regions: DiffusionAtlasRegionBox[],
  plate: MaskPlateSpec,
): { readonly slice: AtlasSlice; readonly values: Float32Array; readonly masked: Uint8Array; readonly visibleCount: number; readonly maskedCount: number } {
  const { mask, atlasX0, atlasY0 } = plate;
  const rows = mask.qLen;
  const cols = mask.kvLen;
  const values = new Float32Array(rows * cols);
  const masked = new Uint8Array(rows * cols);
  let visibleCount = 0;
  let maskedCount = 0;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const isVis = mask.visible[idx] === 1;
      if (isVis) {
        const val = plate.cellValueFn(r, c, true);
        values[idx] = val;
        masked[idx] = 0;
        writer.setTexel(atlasX0 + c, atlasY0 + r, val, false); // G = 0.0
        visibleCount++;
      } else {
        values[idx] = 0;
        masked[idx] = 1;
        writer.setTexel(atlasX0 + c, atlasY0 + r, 0.0, true); // G = 1.0
        maskedCount++;
      }
    }
  }

  regions.push({
    id: plate.id,
    x0: atlasX0,
    y0: atlasY0,
    cols,
    rows,
    mode: 0,
    isMaskOrScore: true,
  });

  return {
    slice: {
      rect: writer.makeRect(atlasX0, atlasY0, cols, rows),
      scale: 1.0,
      mode: 0,
    },
    values,
    masked,
    visibleCount,
    maskedCount,
  };
}

function packCoordStrip(
  writer: Rg32fAtlasWriter,
  regions: DiffusionAtlasRegionBox[],
  strip: CoordStripSpec,
): AtlasSlice {
  const cols = strip.values.length;
  let maxAbs = 1.0;
  for (let c = 0; c < cols; c++) {
    const v = strip.values[c]!;
    writer.setTexel(strip.atlasX0 + c, strip.atlasY0, v, false);
    if (Number.isFinite(v) && Math.abs(v) > maxAbs) {
      maxAbs = Math.abs(v);
    }
  }
  regions.push({
    id: strip.id,
    x0: strip.atlasX0,
    y0: strip.atlasY0,
    cols,
    rows: 1,
    mode: 0,
    isMaskOrScore: false,
  });
  return {
    rect: writer.makeRect(strip.atlasX0, strip.atlasY0, cols, 1),
    scale: 1.0 / maxAbs,
    mode: 0,
  };
}

function int32ToFloat32(arr: Int32Array): Float32Array {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i]!;
  return out;
}

export function buildDiffusionMaskExplorerScene(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const tab: DiffusionExplorerMaskMode = focus.selectedMaskTab;
  const size = DIFFUSION_MASK_ATLAS_SIZE;
  const writer = createRg32fAtlas(size, size);
  const regions: DiffusionAtlasRegionBox[] = [];

  const plates: MaskPlateSpec[] = [];
  const strips: CoordStripSpec[] = [];
  const badges: DiffusionProvenanceBadge[] = [...ctx.provenanceBadges];
  const partCatalog: Record<string, CutawayPartInfo> = {};
  const calipers: DimensionCaliper[] = [];

  let title = '';
  let subtitle = '';
  let mobcCheck: MobcStrictOffsetCheck | null = null;
  let isReadOnlyReference = false;

  if (tab === 'joint_training') {
    const mask = buildJointTrainingMask(6, 2);
    mobcCheck = verifyMobcStrictOffset(mask);
    isReadOnlyReference = true;
    title = 'Joint Training Mask [x_t ‖ x_0] (12×12)';
    subtitle = `Read-Only Analytical Training Mask (No Gradients) · M_BD + strict M_OBC (b_q > b_k, min Δp=${mobcCheck.minDeltaP ?? '—'}, twin leaks=${mobcCheck.zeroTwinViolations}) + M_BC`;

    badges.push(
      { label: 'Read-Only / No Gradients', tone: 'gold' },
      {
        label: mobcCheck.valid
          ? `M_OBC Strict Δp >= ${mobcCheck.minDeltaP} (0 twin leaks)`
          : 'M_OBC Violation Detected',
        tone: mobcCheck.valid ? 'emerald' : 'warn',
      },
      {
        label: 'D23: x_t cache_pos [0..L-1] (1.000×) vs x_0 [L..2L-1] (1.069× at SFT 16K)',
        tone: 'warn',
      },
    );

    const rowLabels = Array.from({ length: 12 }, (_, i) =>
      i < 6 ? `xt${i}(p${i})` : `x0_${i - 6}(p${i - 6})`,
    );
    const colLabels = rowLabels.slice();

    plates.push({
      id: 'mask_joint_training',
      tensorKey: 'mask_joint_training',
      title: 'Joint Training [x_t(0..5) ‖ x_0(0..5)] · L=6, B=2',
      mask,
      cx: 0.0,
      cz: -0.6,
      cellSize: 0.56,
      atlasX0: 0,
      atlasY0: 0,
      rowLabels,
      colLabels,
      cellRegionFn: (r, c) => mask.regions?.[r * mask.kvLen + c] ?? 'none',
      cellValueFn: (r, c, vis) => {
        if (!vis) return 0.0;
        const reg: JointTrainingMaskRegion = mask.regions?.[r * mask.kvLen + c] ?? 'none';
        if (reg === 'M_BD') return 0.96; // strong indigo-slate block diagonal
        if (reg === 'M_OBC') return 0.48; // medium offset block-causal
        if (reg === 'M_BC') return -0.82; // contrasting charcoal-slate clean causal
        return 0.0;
      },
    });

    const plateW = 12 * 0.56;
    // 3-coordinate axis strips + D23 SFT Llama-4 scale strip beneath the 12×12 plate
    const sftLlama4 = new Float32Array(12);
    const sftLabels: string[] = [];
    for (let i = 0; i < 12; i++) {
      // At SFT L = 16384: x_t uses cache_pos < 16384 -> scale = 1.0; x_0 uses cache_pos >= 16384 -> scale = 1 + 0.1*ln(2) = 1.069315
      const sftScale = i < 6 ? 1.0 : 1.0 + 0.1 * Math.log(2);
      sftLlama4[i] = i < 6 ? 0.35 : -0.95; // distinct sign highlights D23 asymmetry
      sftLabels.push(sftScale.toFixed(3) + '×');
    }

    strips.push(
      {
        id: 'coord_seq_idx',
        tensorKey: 'coord_seq_idx',
        label: 'seq_idx (0..11)',
        sublabel: 'Attention Mask Index',
        values: int32ToFloat32(mask.queryCoords.seqIdx),
        labels: Array.from(mask.queryCoords.seqIdx, (v) => String(v)),
        cx: 0.0,
        cz: 3.45,
        ex: plateW,
        atlasX0: 16,
        atlasY0: 0,
        role: SHEET_ROLE_NORM,
        highlight: SHEET_HL_NORMAL,
      },
      {
        id: 'coord_rope_pos',
        tensorKey: 'coord_rope_pos',
        label: 'RoPE position_ids [0..5 ‖ 0..5]',
        sublabel: 'Chunk-Shared cos/sin',
        values: int32ToFloat32(mask.queryCoords.ropePositionIds),
        labels: Array.from(mask.queryCoords.ropePositionIds, (v) => `p${v}`),
        cx: 0.0,
        cz: 4.30,
        ex: plateW,
        atlasX0: 16,
        atlasY0: 4,
        role: SHEET_ROLE_ACTIVATION,
        highlight: SHEET_HL_OPERAND_A,
      },
      {
        id: 'coord_cache_pos',
        tensorKey: 'coord_cache_pos',
        label: 'HF cache_position (0..11)',
        sublabel: 'Feeds _get_llama_4_attn_scale',
        values: int32ToFloat32(mask.queryCoords.cachePositions),
        labels: Array.from(mask.queryCoords.cachePositions, (v) => `cp${v}`),
        cx: 0.0,
        cz: 5.15,
        ex: plateW,
        atlasX0: 16,
        atlasY0: 8,
        role: SHEET_ROLE_ACTIVATION,
        highlight: SHEET_HL_OPERAND_B,
      },
      {
        id: 'coord_d23_llama4',
        tensorKey: 'coord_d23_llama4',
        label: 'D23 Llama-4 SFT Scale (1.000× vs 1.069×)',
        sublabel: 'Asymmetry at L=16384',
        values: sftLlama4,
        labels: sftLabels,
        cx: 0.0,
        cz: 6.00,
        ex: plateW,
        atlasX0: 16,
        atlasY0: 12,
        role: SHEET_ROLE_NORM,
        highlight: SHEET_HL_BF16_GOLD,
      },
    );

    partCatalog.mask_joint_training = {
      id: 'mask_joint_training',
      title: 'Joint Training Dual-Stream Mask [x_t ‖ x_0]',
      badge: 'Read-Only / No Gradients',
      summary:
        'Single-backbone dual-stream training mask combining M_BD (noisy block-diagonal), strict M_OBC (noisy→clean b_q > b_k, Δp >= 1), and causal M_BC (clean→clean).',
      specs: [
        `Shape: 12×12 (L=6, B=2) · Read-only analytical visualization (zero gradients)`,
        `verifyMobcStrictOffset: valid=${String(mobcCheck.valid)}, minDeltaP=${String(mobcCheck.minDeltaP)}, zeroTwinViolations=${mobcCheck.zeroTwinViolations}`,
        'D23: RoPE position_ids=[0..L-1 || 0..L-1] diverges from cache_position=[0..2L-1], giving x_0 1.0693× Llama-4 scale at SFT L=16384',
      ],
    };
    partCatalog.coord_d23_llama4 = {
      id: 'coord_d23_llama4',
      title: 'D23 Llama-4 Query Scale Asymmetry',
      badge: 'Verified Upstream Bug D23',
      summary:
        'In forward(), concatenating [x_t || x_0] doubles cache_position to 0..2L-1 while position_ids stays [0..L-1 || 0..L-1].',
      specs: [
        'At SFT L=16,384: x_t (0..16383) gets s_L4 = 1.000000×',
        'Clean stream x_0 (16384..32767) gets s_L4 = 1 + 0.1·ln(2) = 1.069315×',
      ],
    };
  } else if (tab === 'ar_causal') {
    const prefill = buildArCausalMask(0, 4);
    const decode = buildArCausalMask(4, 1);
    title = 'AR Causal Attention Masks (Prefill 4×4 & Decode 1×5)';
    subtitle = 'ar_generate: explicit lower-triangular prefill (use_cache=True) and single-token decode attending over [0..P] (D8 explicit mask guard)';

    badges.push({ label: 'KV: token_causal · use_cache=True', tone: 'causal' });

    plates.push(
      {
        id: 'mask_ar_prefill',
        tensorKey: 'mask_ar_prefill',
        title: 'AR Causal Prefill (4×4 · P=0, Q=4)',
        mask: prefill,
        cx: -3.2,
        cz: 0.0,
        cellSize: 0.85,
        atlasX0: 0,
        atlasY0: 0,
        rowLabels: ['q0(p0)', 'q1(p1)', 'q2(p2)', 'q3(p3)'],
        colLabels: ['k0', 'k1', 'k2', 'k3'],
        cellRegionFn: (_r, _c) => 'AR_Prefill',
        cellValueFn: (_r, _c, vis) => (vis ? 0.90 : 0.0),
      },
      {
        id: 'mask_ar_decode',
        tensorKey: 'mask_ar_decode',
        title: 'AR Causal Decode (1×5 · P=4, Q=1)',
        mask: decode,
        cx: 3.3,
        cz: 0.0,
        cellSize: 0.85,
        atlasX0: 8,
        atlasY0: 0,
        rowLabels: ['q4(p4)'],
        colLabels: ['kv0', 'kv1', 'kv2', 'kv3', 'k4'],
        cellRegionFn: (_r, c) => (c < 4 ? 'Cached_Prefix_KV' : 'Current_Token'),
        cellValueFn: (_r, c, vis) => (vis ? (c < 4 ? 0.58 : 0.96) : 0.0),
      },
    );

    strips.push({
      id: 'coord_ar_decode_pos',
      tensorKey: 'coord_ar_decode_pos',
      label: 'KV RoPE = cache_position (0..4)',
      sublabel: 'Aligned coordinates in AR',
      values: int32ToFloat32(decode.kvRopePositionIds),
      labels: ['p0', 'p1', 'p2', 'p3', 'p4'],
      cx: 3.3,
      cz: 1.55,
      ex: 5 * 0.85,
      atlasX0: 16,
      atlasY0: 0,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_OPERAND_A,
    });

    partCatalog.mask_ar_prefill = {
      id: 'mask_ar_prefill',
      title: 'AR Causal Prefill (4×4)',
      badge: 'use_cache=True · Shifted +1',
      summary: 'Lower-triangular causal prefill writing 4 token_causal KV slabs; upper triangle explicitly masked with G = 1.0.',
      specs: [
        'D8 Guard: always applies explicit causal mask regardless of eager vs SDPA',
        'Last row logits predict token at position P=4',
      ],
    };
    partCatalog.mask_ar_decode = {
      id: 'mask_ar_decode',
      title: 'AR Causal Decode (1×5)',
      badge: '1 query × 5 KVs · G = 0.0',
      summary: 'Single decode token at position 4 attends to all 4 cached prefix KVs plus itself; appends 1 slab to KV cache.',
      specs: [
        'All 5 entries visible (G = 0.0) in live RG32F texture',
        'Shifted +1: row 0 at p=4 predicts token p=5',
      ],
    };
  } else if (tab === 'block_bidir') {
    const mask = buildBlockBidirMask(4, 4);
    title = 'Block-Diffusion Bidirectional Mask (4×8)';
    subtitle = 'generate() denoise / linear_spec draft · ALL 32 cells visible (G = 0.0) · Transient KV concat (use_cache=False) · Proves D24 no geometric fallback';

    badges.push(
      { label: 'All 32 Cells Visible (G = 0.0)', tone: 'bidir' },
      { label: 'use_cache=False (0 KV mutation)', tone: 'emerald' },
    );

    plates.push({
      id: 'mask_block_bidir',
      tensorKey: 'mask_block_bidir',
      title: 'Block-Diff Denoise / Spec Draft [4 × (4+4)]',
      mask,
      cx: 0.0,
      cz: -0.2,
      cellSize: 0.78,
      atlasX0: 0,
      atlasY0: 0,
      rowLabels: ['b0(p4)', 'b1(p5)', 'b2(p6)', 'b3(p7)'],
      colLabels: ['p0', 'p1', 'p2', 'p3', 'b0', 'b1', 'b2', 'b3'],
      cellRegionFn: (_r, c) => (c < 4 ? 'Prefix_KV_ReadOnly' : 'Active_Block_Bidir'),
      cellValueFn: (_r, c, vis) => (vis ? (c < 4 ? 0.56 : 0.96) : 0.0),
    });

    strips.push({
      id: 'coord_bidir_kv_pos',
      tensorKey: 'coord_bidir_kv_pos',
      label: 'KV Positions [prefix 0..3 ‖ transient block 4..7]',
      sublabel: 'RoPE position_ids == cache_position',
      values: int32ToFloat32(mask.kvRopePositionIds),
      labels: ['p0', 'p1', 'p2', 'p3', 'p4*', 'p5*', 'p6*', 'p7*'],
      cx: 0.0,
      cz: 2.35,
      ex: 8 * 0.78,
      atlasX0: 16,
      atlasY0: 0,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_OPERAND_B,
    });

    partCatalog.mask_block_bidir = {
      id: 'mask_block_bidir',
      title: 'Block-Diffusion Bidirectional Mask (4×8)',
      badge: 'D24 Verified · 32/32 Visible (G=0)',
      summary:
        'Every active block token sees the entire cached prefix AND all sibling tokens in the active block (including future block slots).',
      specs: [
        'Bound to live RG32F texture with G = 0.0 on all 32 cells — zero geometric cellCol > cellRow fallback',
        'use_cache=False: transient cat([old_k, K]) leaves persistent prefix KV immutable (0 ULP)',
        'Unshifted (+0) logits: row i predicts token at position P + i',
      ],
    };
  } else if (tab === 'causal_commit_verify') {
    const mask = buildCausalCommitVerifyMask(4, 4);
    title = 'Causal Commit / Speculative Verify Mask (4×8)';
    subtitle = 'generate(causal_context=True) post-block commit & linear_spec causal verify · Lower triangle visible (G = 0.0), upper block masked (G = 1.0)';

    badges.push(
      { label: 'use_cache=True (KV += B; Spec crops)', tone: 'causal' },
      { label: 'Shifted +1: row i verifies block[i+1]', tone: 'gold' },
    );

    plates.push({
      id: 'mask_causal_commit_verify',
      tensorKey: 'mask_causal_commit_verify',
      title: 'Causal Commit / Spec Verify [4 × (4+4)]',
      mask,
      cx: 0.0,
      cz: -0.2,
      cellSize: 0.78,
      atlasX0: 0,
      atlasY0: 0,
      rowLabels: ['b0(p4)→p5', 'b1(p5)→p6', 'b2(p6)→p7', 'b3(p7)→p8'],
      colLabels: ['p0', 'p1', 'p2', 'p3', 'b0', 'b1', 'b2', 'b3'],
      cellRegionFn: (r, c) =>
        c < 4 ? 'Prefix_KV' : c <= 4 + r ? 'Causal_Block_Visible' : 'Masked_Future_Block',
      cellValueFn: (_r, c, vis) => (vis ? (c < 4 ? 0.56 : 0.96) : 0.0),
    });

    strips.push({
      id: 'coord_verify_kv_pos',
      tensorKey: 'coord_verify_kv_pos',
      label: 'Committed KV Positions [0..3 ‖ 4..7]',
      sublabel: 'Exact RoPE reuse for accepted prefix',
      values: int32ToFloat32(mask.kvRopePositionIds),
      labels: ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
      cx: 0.0,
      cz: 2.35,
      ex: 8 * 0.78,
      atlasX0: 16,
      atlasY0: 0,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_OPERAND_A,
    });

    partCatalog.mask_causal_commit_verify = {
      id: 'mask_causal_commit_verify',
      title: 'Causal Commit / Speculative Verify (4×8)',
      badge: '26 Visible (G=0) · 6 Masked (G=1)',
      summary:
        'Causal attention over the active block so committed KV entries obey strict token-causal provenance and row i verifies draft slot i+1.',
      specs: [
        'Lower-triangle active block visible (G=0.0); strictly upper triangle masked (G=1.0 hatched)',
        'linear_spec ALWAYS disables o_proj LoRA during this verify pass (0 ULP AR invariance)',
      ],
    };
  } else {
    // 'vlm_quadratic_ss'
    const mask = buildVlmQuadraticSsMask(2, 2);
    isReadOnlyReference = true;
    title = 'Quadratic SS · VLM/Paper Ref (6×8)';
    subtitle = 'Read-Only Reference verified from VLM-8B (P=2, k=2 → k(k+1)=6 interleaved queries, reordered [draft_kv(0..5) ‖ clean_kv(0..1)])';

    badges.push(
      { label: 'Quadratic SS · VLM/Paper Ref (Read-Only)', tone: 'gold' },
      { label: 'Reordered KV: [draft_kv ‖ clean_kv]', tone: 'bidir' },
      { label: 'D24: VLM self_spec_generate is greedy-only', tone: 'warn' },
    );

    const rowLabels = [
      'd0(p2)',
      'm0.1(p3)',
      'm0.2(p4)',
      'd1(p3)',
      'm1.1(p4)',
      'm1.2(p5)',
    ];
    const colLabels = [
      'd0(p2)',
      'm0.1(p3)',
      'm0.2(p4)',
      'd1(p3)',
      'm1.1(p4)',
      'm1.2(p5)',
      'c0(p0)',
      'c1(p1)',
    ];

    plates.push({
      id: 'mask_vlm_quadratic_ss',
      tensorKey: 'mask_vlm_quadratic_ss',
      title: 'VLM-8B Cached Quadratic SS [6 × 8] · P=2, k=2',
      mask,
      cx: 0.0,
      cz: -0.7,
      cellSize: 0.68,
      atlasX0: 0,
      atlasY0: 0,
      rowLabels,
      colLabels,
      cellRegionFn: (r, c) => {
        if (c >= 6) return 'clean_part';
        if (c % 3 === 0 && r >= c) return 'first_clean_causal';
        if (Math.floor(r / 3) === Math.floor(c / 3) && r % 3 !== 0) return 'same_block_bidir';
        return 'masked';
      },
      cellValueFn: (r, c, vis) => {
        if (!vis) return 0.0;
        if (c >= 6) return 0.52; // clean_part
        if (c % 3 === 0 && r >= c) return -0.82; // first_clean causal anchors
        return 0.96; // same_block_except_first bidir draft subblock
      },
    });

    const plateW = 8 * 0.68;
    strips.push(
      {
        id: 'coord_vlm_kv_rope',
        tensorKey: 'coord_vlm_kv_rope',
        label: 'Reordered KV RoPE position_ids [2,3,4, 3,4,5 ‖ 0,1]',
        sublabel: 'Overlapping sibling branch RoPE IDs',
        values: int32ToFloat32(mask.kvRopePositionIds),
        labels: Array.from(mask.kvRopePositionIds, (v) => `p${v}`),
        cx: 0.0,
        cz: 2.35,
        ex: plateW,
        atlasX0: 16,
        atlasY0: 0,
        role: SHEET_ROLE_ACTIVATION,
        highlight: SHEET_HL_OPERAND_A,
      },
      {
        id: 'coord_vlm_kv_cache',
        tensorKey: 'coord_vlm_kv_cache',
        label: 'Reordered KV cache_position [2,3,4,5,6,7 ‖ 0,1]',
        sublabel: 'Diverges from overlapping RoPE IDs',
        values: int32ToFloat32(mask.kvCachePositions),
        labels: Array.from(mask.kvCachePositions, (v) => `cp${v}`),
        cx: 0.0,
        cz: 3.25,
        ex: plateW,
        atlasX0: 16,
        atlasY0: 4,
        role: SHEET_ROLE_ACTIVATION,
        highlight: SHEET_HL_OPERAND_B,
      },
    );

    partCatalog.mask_vlm_quadratic_ss = {
      id: 'mask_vlm_quadratic_ss',
      title: 'Quadratic SS · VLM/Paper Reference (6×8)',
      badge: 'Read-Only Reference (VLM-8B)',
      summary:
        'Simultaneously verifies k draft tokens and drafts speculative continuations for every possible acceptance count in 1 pass using k(k+1) interleaved queries.',
      specs: [
        'Absent from all 6 released text checkpoints (D10); verified directly from VLM-8B',
        'Reordered cached KV layout: [draft_kv(0..5) || clean_kv(0..1)]',
        'Overlapping RoPE position_ids [2,3,4,3,4,5] vs sequential cache_position [2..7]',
      ],
    };
  }

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const sheets = new SheetBufferBuilder();
  const flow = new FlowRibbonBuilder();

  const solidPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];

  const activeCell: DiffusionPickIdentity | null =
    focus.pinnedCell && focus.pinnedCell.viewMode === 'mask_explorer'
      ? focus.pinnedCell
      : focus.hoveredCell && focus.hoveredCell.viewMode === 'mask_explorer'
        ? focus.hoveredCell
        : null;

  let totalVisible = 0;
  let totalMasked = 0;
  const plateGeomMap = new Map<
    string,
    {
      plate: MaskPlateSpec;
      frame: SheetFrame3D;
      geom: SheetGeomRecord;
      values: Float32Array;
      masked: Uint8Array;
    }
  >();

  // Drafting table border
  lines.addBoxFrameXZ(-6.6, -4.6, 6.6, 6.8, -0.05, 1.0, LINE_STYLE_GUIDE, 0.34);

  for (const plate of plates) {
    const packed = packMaskPlate(writer, regions, plate);
    totalVisible += packed.visibleCount;
    totalMasked += packed.maskedCount;

    const ex = plate.mask.kvLen * plate.cellSize;
    const ez = plate.mask.qLen * plate.cellSize;
    const ey = 0.072;
    const frame = makeSheetFrame3D(
      [plate.cx, 0.0, plate.cz],
      [ex, ey, ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    const geom: SheetGeomRecord = {
      cx: plate.cx,
      cy: 0.0,
      cz: plate.cz,
      ex,
      ey,
      ez,
      gridCols: plate.mask.kvLen,
      gridRows: plate.mask.qLen,
      frame,
    };
    plateGeomMap.set(plate.id, {
      plate,
      frame,
      geom,
      values: packed.values,
      masked: packed.masked,
    });

    const isFocused = activeCell?.tensorKey === plate.tensorKey || focus.selectedPartId === plate.id;

    // D24: ALWAYS SHEET_KIND_CAUSAL_SCORES bound to a live RG32F slice with explicit G ∈ {0.0, 1.0}
    sheets.push({
      cx: plate.cx,
      cy: 0.0,
      cz: plate.cz,
      ex,
      ey,
      ez,
      gridCols: plate.mask.kvLen,
      gridRows: plate.mask.qLen,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      highlight: isFocused ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
      slice: packed.slice,
      frame,
    });

    const aabb = sheetFrameWorldAabb(frame);
    solidPickBoxes.push({
      bayId: 0,
      partId: plate.id,
      tensorKey: plate.tensorKey,
      gridCols: plate.mask.kvLen,
      gridRows: plate.mask.qLen,
      semanticTag: plate.mask.mode,
      min: aabb.min,
      max: aabb.max,
      frame,
    });

    landmarks.push({
      id: `mask-lm-${plate.id}`,
      label: plate.title,
      sublabel: `${packed.visibleCount}/${plate.mask.qLen * plate.mask.kvLen} visible · ${plate.mask.badge}`,
      world: [plate.cx, 0.16, plate.cz - ez * 0.5 - 0.34],
      partId: plate.id,
      tone: isReadOnlyReference ? 'gold' : 'default',
      tier: 'primary',
    });

    // Quadrant / prefix divider lines for immediate legibility
    if (tab === 'joint_training') {
      // Split between x_t (0..5) and x_0 (6..11) on both axes
      lines.addSegment(plate.cx, 0.06, plate.cz - ez * 0.5, plate.cx, 0.06, plate.cz + ez * 0.5, 1.8, LINE_STYLE_BF16_GOLD, 0.85);
      lines.addSegment(plate.cx - ex * 0.5, 0.06, plate.cz, plate.cx + ex * 0.5, 0.06, plate.cz, 1.8, LINE_STYLE_BF16_GOLD, 0.85);
      calipers.push(
        {
          id: 'mask-cal-xt',
          aWorld: [plate.cx - ex * 0.5, 0.10, plate.cz - ez * 0.5 - 0.12],
          bWorld: [plate.cx, 0.10, plate.cz - ez * 0.5 - 0.12],
          label: 'x_t Noisy Stream (0..5 · M_BD)',
          partIds: [plate.id],
        },
        {
          id: 'mask-cal-x0',
          aWorld: [plate.cx, 0.10, plate.cz - ez * 0.5 - 0.12],
          bWorld: [plate.cx + ex * 0.5, 0.10, plate.cz - ez * 0.5 - 0.12],
          label: 'x_0 Clean Stream (6..11 · M_OBC / M_BC)',
          partIds: [plate.id],
        },
      );
    } else if (tab === 'block_bidir' || tab === 'causal_commit_verify') {
      // Split between prefix (0..3) and active block (4..7)
      lines.addSegment(plate.cx, 0.06, plate.cz - ez * 0.5, plate.cx, 0.06, plate.cz + ez * 0.5, 1.6, LINE_STYLE_BF16_GOLD, 0.80);
      calipers.push(
        {
          id: 'mask-cal-prefix',
          aWorld: [plate.cx - ex * 0.5, 0.10, plate.cz - ez * 0.5 - 0.12],
          bWorld: [plate.cx, 0.10, plate.cz - ez * 0.5 - 0.12],
          label: 'Prefix KV (P=4)',
          partIds: [plate.id],
        },
        {
          id: 'mask-cal-block',
          aWorld: [plate.cx, 0.10, plate.cz - ez * 0.5 - 0.12],
          bWorld: [plate.cx + ex * 0.5, 0.10, plate.cz - ez * 0.5 - 0.12],
          label: tab === 'block_bidir' ? 'Active Block (Bidir · G=0)' : 'Active Block (Causal · Lower Tri)',
          partIds: [plate.id],
        },
      );
    } else if (tab === 'vlm_quadratic_ss') {
      const splitX = plate.cx - ex * 0.5 + 6 * plate.cellSize;
      lines.addSegment(splitX, 0.06, plate.cz - ez * 0.5, splitX, 0.06, plate.cz + ez * 0.5, 1.8, LINE_STYLE_BF16_GOLD, 0.85);
      calipers.push(
        {
          id: 'mask-cal-vlm-draft',
          aWorld: [plate.cx - ex * 0.5, 0.10, plate.cz - ez * 0.5 - 0.12],
          bWorld: [splitX, 0.10, plate.cz - ez * 0.5 - 0.12],
          label: 'draft_kv (k(k+1)=6)',
          partIds: [plate.id],
        },
        {
          id: 'mask-cal-vlm-clean',
          aWorld: [splitX, 0.10, plate.cz - ez * 0.5 - 0.12],
          bWorld: [plate.cx + ex * 0.5, 0.10, plate.cz - ez * 0.5 - 0.12],
          label: 'clean_kv (P=2)',
          partIds: [plate.id],
        },
      );
    }

    activeCellGrids.push({
      originWorld: [plate.cx, 0.0, plate.cz],
      extent: [ex, ey, ez],
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: plate.mask.kvLen,
      rows: plate.mask.qLen,
      values: packed.values,
      masked: packed.masked,
      format: 'signed',
      activeCol: activeCell?.tensorKey === plate.tensorKey ? (activeCell.col ?? undefined) : undefined,
      activeRow: activeCell?.tensorKey === plate.tensorKey ? (activeCell.row ?? undefined) : undefined,
      colLabels: plate.colLabels,
      rowLabels: plate.rowLabels,
      colDimLabel: `KV=${plate.mask.kvLen}`,
      rowDimLabel: `Q=${plate.mask.qLen}`,
    });
  }

  for (const strip of strips) {
    const slice = packCoordStrip(writer, regions, strip);
    const ez = 0.44;
    const ey = 0.056;
    const frame = makeSheetFrame3D(
      [strip.cx, 0.0, strip.cz],
      [strip.ex, ey, ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    sheets.push({
      cx: strip.cx,
      cy: 0.0,
      cz: strip.cz,
      ex: strip.ex,
      ey,
      ez,
      gridCols: strip.values.length,
      gridRows: 1,
      sheetKind: SHEET_KIND_STANDARD,
      role: strip.role,
      highlight: strip.highlight,
      slice,
      frame,
    });
    const aabb = sheetFrameWorldAabb(frame);
    solidPickBoxes.push({
      bayId: 0,
      partId: strip.id,
      tensorKey: strip.tensorKey,
      gridCols: strip.values.length,
      gridRows: 1,
      semanticTag: 'coordinate_strip',
      min: aabb.min,
      max: aabb.max,
      frame,
    });
    landmarks.push({
      id: `mask-lm-${strip.id}`,
      label: strip.label,
      sublabel: strip.sublabel,
      world: [strip.cx - strip.ex * 0.5 - 0.25, 0.10, strip.cz],
      partId: strip.id,
      align: 'right',
      tone: strip.highlight === SHEET_HL_BF16_GOLD ? 'gold' : 'muted',
      tier: 'detail',
    });
    lines.addBoxFrameXZ(
      strip.cx - strip.ex * 0.5,
      strip.cz - ez * 0.5,
      strip.cx + strip.ex * 0.5,
      strip.cz + ez * 0.5,
      0.03,
      0.8,
      LINE_STYLE_NEUTRAL,
      0.28,
    );
  }

  // Cell inspection formula & popover
  const primaryPlate = plates[0]!;
  const focusedEntry =
    (activeCell?.tensorKey ? plateGeomMap.get(activeCell.tensorKey) : undefined) ??
    plateGeomMap.get(primaryPlate.id)!;
  const fMask = focusedEntry.plate.mask;
  const selRow = clampInt(
    activeCell?.tensorKey === focusedEntry.plate.tensorKey ? (activeCell.row ?? 0) : (tab === 'joint_training' ? 2 : 0),
    0,
    fMask.qLen - 1,
  );
  const selCol = clampInt(
    activeCell?.tensorKey === focusedEntry.plate.tensorKey ? (activeCell.col ?? 0) : (tab === 'joint_training' ? 6 : Math.min(fMask.kvLen - 1, 4)),
    0,
    fMask.kvLen - 1,
  );
  const cellIdx = selRow * fMask.kvLen + selCol;
  const isVisible = fMask.visible[cellIdx] === 1;
  const regionName = focusedEntry.plate.cellRegionFn(selRow, selCol);
  const qSeqIdx = fMask.queryCoords.seqIdx[selRow]!;
  const qRopePos = fMask.queryCoords.ropePositionIds[selRow]!;
  const qCachePos = fMask.queryCoords.cachePositions[selRow]!;
  const kRopePos = fMask.kvRopePositionIds[selCol]!;
  const kCachePos = fMask.kvCachePositions[selCol]!;
  const deltaP = qRopePos - kRopePos;

  if (activeCell !== null && activeCell.tensorKey === focusedEntry.plate.tensorKey) {
    const span = spanFromSheetGrid(focusedEntry.geom, selRow, selRow, selCol, selCol);
    drawCellHighlight3D(flow, span, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.026, true);
  }

  const equation = `M[q=${selRow}, kv=${selCol}] = ${isVisible ? '1 (VISIBLE · G=0.0)' : '0 (MASKED −∞ · G=1.0)'} · region=${regionName}`;
  const subEquation = `seq_idx=${qSeqIdx} · RoPE pos (q=${qRopePos}, k=${kRopePos}, Δp=${deltaP}) · cache_pos (q=${qCachePos}, k=${kCachePos})`;

  const activeFormula: LayerFormulaReadout = {
    tensorLabel: focusedEntry.plate.title,
    coordLabel: `q=${selRow} (${focusedEntry.plate.rowLabels[selRow] ?? selRow}) × kv=${selCol} (${focusedEntry.plate.colLabels[selCol] ?? selCol})`,
    headBadge: isReadOnlyReference ? 'Read-Only Ref' : 'Live Mask',
    value: isVisible ? 1.0 : -Infinity,
    isMasked: !isVisible,
    valueText: isVisible ? `= ${fmtSigned(focusedEntry.values[cellIdx] ?? 1.0, 2)} (vis)` : '= −∞ (masked)',
    equation,
    subEquation,
  };

  const cellPopover: CellPopoverSpec | null =
    activeCell !== null && activeCell.tensorKey === focusedEntry.plate.tensorKey
      ? {
          world: localToWorldOnFrame(
            focusedEntry.frame,
            -0.5 + (selCol + 0.5) / fMask.kvLen,
            0.5 + 0.09 / Math.max(0.02, focusedEntry.frame.extent[1]),
            -0.5 + (selRow + 0.5) / fMask.qLen,
          ),
          tensorLabel: focusedEntry.plate.title,
          coordLabel: `q=${selRow}, kv=${selCol} · ${regionName}`,
          badge: isVisible ? 'Visible (G=0)' : 'Masked (G=1)',
          equation,
          valueText: isVisible ? 'Visible (1)' : 'Masked (−∞)',
          operandColors: isVisible ? ['teal'] : ['red'],
        }
      : null;

  const audit = auditAtlasRegions(size, size, regions);
  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0) {
    throw new Error(
      `Mask Explorer 64×64 atlas layout error: overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  const solidPacked = solids.pack();
  const linePacked = lines.pack();
  const ribbonPacked = ribbons.pack();
  const sheetPacked = sheets.pack();
  const flowPacked = flow.pack();

  const fitBounds: AABB = {
    min: [-6.8, -0.4, -4.8],
    max: [6.8, 1.6, 7.0],
  };

  return {
    viewMode: 'mask_explorer',
    title,
    subtitle,
    layerIndex: null,
    solidData: solidPacked.data,
    solidCount: solidPacked.count,
    lineData: linePacked.data,
    lineCount: linePacked.count,
    ribbonData: ribbonPacked.data,
    ribbonCount: ribbonPacked.count,
    sheetData: sheetPacked.sheetData,
    sheetCount: sheetPacked.sheetCount,
    sheetBasisData: sheetPacked.sheetBasisData,
    sheetRevealData: sheetPacked.sheetRevealData,
    flowVertices: flowPacked.vertices,
    flowIndices: flowPacked.indices,
    valueAtlas: {
      width: size,
      height: size,
      data: writer.data,
    },
    atlasKey: `mask_explorer|tab=${tab}`,
    atlasAudit: audit,
    fitBounds,
    defaultCameraAzimuthDeg: 18,
    defaultCameraElevationDeg: 52,
    landmarks,
    kvDepthTicks: [],
    calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeCellGrids,
    cellPopover,
    activeFormula,
    stepperSteps: DIFFUSION_MASK_TAB_STEPPER,
    provenanceBadges: Object.freeze(badges),
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner,
    diagnostics: Object.freeze({
      maskMode: tab,
      qLen: primaryPlate.mask.qLen,
      kvLen: primaryPlate.mask.kvLen,
      visibleCount: totalVisible,
      maskedCount: totalMasked,
      mobcCheck,
      isReadOnlyReference,
      atlasUtilizationPct: audit.utilizationPct,
    }),
  };
}
