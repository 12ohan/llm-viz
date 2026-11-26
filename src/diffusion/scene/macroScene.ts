/**
 * 40-Layer Backbone Macro Spatial Scene (`src/diffusion/scene/macroScene.ts`)
 *
 * Single physical 40-layer homogeneous dense Ministral-3 decoder spine
 * (`Layers 01..40` descending along `-Y`, paired `[RoPE GQA | SwiGLU]` solids),
 * untied `encoder.embed_tokens [36, 8]` and `encoder.norm + diffusion_head [36, 8]`
 * bookends, `0..31` KV cache rail along `+Z` distinguishing `'token_causal'` vs
 * `'block_bidirectional'` provenance, and ONE detachable `o_proj` LoRA rail that
 * bridges onto all 40 layers ONLY on `linear_spec_lora` draft passes.
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`,
 * or `src/twotower/` / `src/math/gqaSim` / `src/math/moeSim` imports.
 */

import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_ARROW_POS_Z,
  RIBBON_KIND_BOX,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from '../../scene/spine';
import {
  DIFFUSION_EPISTEMIC_NOTE,
  DIFFUSION_LORA_RAIL_BADGE,
  DIFFUSION_PROJECTION_REVERSAL_BADGE,
  LineBufferBuilder,
  RibbonBufferBuilder,
  SolidBufferBuilder,
} from './provenance';
import type {
  DiffusionFocusState,
  DiffusionProvenanceBadge,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
} from './types';

// ============================================================================
// MACRO GEOMETRY CONSTANTS (40-Layer Homogeneous Dense Spine)
// ============================================================================

const NUM_LAYERS = 40;
const LAYER_PITCH_Y = 1.45;
const DECADE_GAP_Y = 0.45;

const GQA_X = -1.40;
const SWIGLU_X = 1.40;
const SUBBLOCK_EX = 2.38;
const SUBBLOCK_EY = 0.96;
const SUBBLOCK_EZ = 1.12;

const RESIDUAL_RAIL_X = 3.55;
const LORA_RAIL_BRIDGED_X = -4.25;
const LORA_RAIL_DETACHED_X = -6.85;
const LORA_RAIL_EX = 0.72;
const LORA_RAIL_EZ = 0.88;

const KV_RAIL_X = -1.40;
const KV_START_Z = 1.65;
const KV_SLOT_PITCH_Z = 0.44;
const KV_SLOT_DEPTH_Z = 0.28;
const KV_SLAB_EX = 2.15;
const KV_SLAB_EY = 0.78;

function layerCenterY(layerIndex0: number): number {
  const decadesCompleted = Math.floor(layerIndex0 / 10);
  return -layerIndex0 * LAYER_PITCH_Y - decadesCompleted * DECADE_GAP_Y;
}

function kvSlotCenterZ(pos0: number): number {
  return KV_START_Z + pos0 * KV_SLOT_PITCH_Z + KV_SLOT_DEPTH_Z * 0.5;
}

function formatLayerPartId(layerIndex0: number): string {
  return `layer-${String(layerIndex0 + 1).padStart(2, '0')}`;
}

function clampLayer(layerIndex0: number): number {
  return Math.max(0, Math.min(NUM_LAYERS - 1, Math.trunc(layerIndex0)));
}

// ============================================================================
// MAIN BUILDER
// ============================================================================

export function buildDiffusionMacroScene(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const cfg = ctx.model.config;
  const selectedLayer = clampLayer(focus.selectedLayerIndex);
  const selectedPartId = focus.selectedPartId;
  const hoveredPartId = focus.hoveredPartId;
  const loraBridged = ctx.loraActiveOnDraft;
  const kvSeqLen = ctx.kvSnapshot.seqLen;
  const kvCapacity = ctx.kvSnapshot.capacity;
  const kvProvenance = ctx.kvSnapshot.kvMaskProvenance;

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const solidPickBoxes: PickBox[] = [];
  const fasciaPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const partCatalog: Record<string, CutawayPartInfo> = {};

  const topLayerY = layerCenterY(0);
  const bottomLayerY = layerCenterY(NUM_LAYERS - 1);
  const embedY = topLayerY + 3.10;
  const finalNormY = bottomLayerY - 2.10;
  const diffHeadY = bottomLayerY - 4.15;

  let ownerCounter = 1;

  // --------------------------------------------------------------------------
  // 1. TOP BOOKEND: `encoder.embed_tokens [V=36, d=8]`
  // --------------------------------------------------------------------------
  const embedActive =
    selectedPartId === 'embed_tokens' || hoveredPartId === 'embed_tokens';
  solids.push({
    cx: 0,
    cy: embedY,
    cz: 0,
    ex: 5.20,
    ey: 1.24,
    ez: 1.24,
    cellFamily: CELL_EMBED_SLATE,
    fp8: 0,
    phase: 0,
    bayId: 0,
    cellScale: 0.24,
    highlight: embedActive ? 1 : 0,
  });
  lines.addBoxEdges(
    0,
    embedY,
    0,
    5.20,
    1.24,
    1.24,
    ownerCounter++,
    embedActive ? 1.5 : 1.05,
    embedActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    embedActive ? 0.92 : 0.72,
  );
  solidPickBoxes.push({
    bayId: 0,
    partId: 'embed_tokens',
    semanticTag: 'embed_tokens',
    min: [-2.60, embedY - 0.62, -0.62],
    max: [2.60, embedY + 0.62, 0.62],
  });
  landmarks.push({
    id: 'lm-embed-tokens',
    label: 'encoder.embed_tokens [36, 8]',
    sublabel: 'Untied V=36 (32 [MASK], 33 <|im_end|>, 34 <s>, 35 </think>)',
    world: [3.15, embedY, 0],
    bayId: 0,
    partId: 'embed_tokens',
    align: 'left',
    tier: 'primary',
  });
  partCatalog['embed_tokens'] = {
    id: 'embed_tokens',
    title: 'encoder.embed_tokens',
    badge: 'Untied V=36 · d=8',
    summary:
      'Untied token embedding lookup table. Positional information is injected per layer via YaRN RoPE inside attention (no learned positional embeddings).',
    specs: [
      `Mini shape: [${cfg.vocabSize}, ${cfg.hiddenSize}] (${cfg.vocabSize * cfg.hiddenSize} floats)`,
      'Prod 14B shape: [131,072, 5,120] (671,088,640 params · 1.25 GiB BF16)',
      'Special IDs: [MASK]=32, <|im_end|>=33, <s>=34, </think>=35',
    ],
  };

  // Residual drop from embed_tokens into Layer 01
  ribbons.push({
    cx: 0,
    cy: (embedY - 0.62 + (topLayerY + SUBBLOCK_EY * 0.5)) * 0.5,
    cz: 0,
    ex: 0.42,
    ey: Math.max(0.4, embedY - 0.62 - (topLayerY + SUBBLOCK_EY * 0.5) - 0.14),
    ez: 0.28,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.46,
  });

  // --------------------------------------------------------------------------
  // 2. SINGLE PHYSICAL 40-LAYER SPINE (`Layers 01..40`, paired [RoPE GQA | SwiGLU])
  // --------------------------------------------------------------------------
  for (let l = 0; l < NUM_LAYERS; l++) {
    const y = layerCenterY(l);
    const bayId = l + 1;
    const partId = formatLayerPartId(l);
    const isFocusedLayer =
      l === selectedLayer ||
      selectedPartId === partId ||
      hoveredPartId === partId;

    // Left paired solid: 32Q/8KV YaRN RoPE GQA
    solids.push({
      cx: GQA_X,
      cy: y,
      cz: 0,
      ex: SUBBLOCK_EX,
      ey: SUBBLOCK_EY,
      ez: SUBBLOCK_EZ,
      cellFamily: CELL_WEIGHT_BLUE,
      fp8: 0,
      phase: l * 0.13,
      bayId,
      cellScale: 0.22,
      highlight: isFocusedLayer ? 1 : 0,
    });
    lines.addBoxEdges(
      GQA_X,
      y,
      0,
      SUBBLOCK_EX,
      SUBBLOCK_EY,
      SUBBLOCK_EZ,
      ownerCounter++,
      isFocusedLayer ? 1.55 : 0.95,
      isFocusedLayer ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      isFocusedLayer ? 0.94 : 0.68,
    );

    // Right paired solid: Dense SwiGLU (d=8 -> dFf=24 -> d=8)
    solids.push({
      cx: SWIGLU_X,
      cy: y,
      cz: 0,
      ex: SUBBLOCK_EX,
      ey: SUBBLOCK_EY,
      ez: SUBBLOCK_EZ,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8: 0,
      phase: l * 0.17 + 0.5,
      bayId,
      cellScale: 0.22,
      highlight: isFocusedLayer ? 1 : 0,
    });
    lines.addBoxEdges(
      SWIGLU_X,
      y,
      0,
      SUBBLOCK_EX,
      SUBBLOCK_EY,
      SUBBLOCK_EZ,
      ownerCounter++,
      isFocusedLayer ? 1.55 : 0.95,
      isFocusedLayer ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      isFocusedLayer ? 0.94 : 0.68,
    );

    // Intra-layer coupler ribbon: RoPE GQA -> SwiGLU
    ribbons.push({
      cx: 0,
      cy: y,
      cz: 0,
      ex: 0.36,
      ey: 0.24,
      ez: 0.24,
      kind: RIBBON_KIND_ARROW_POS_X,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: isFocusedLayer ? 0.68 : 0.32,
    });

    // Residual highway tap into +X spine
    ribbons.push({
      cx: (SWIGLU_X + SUBBLOCK_EX * 0.5 + RESIDUAL_RAIL_X) * 0.5,
      cy: y,
      cz: 0,
      ex: RESIDUAL_RAIL_X - (SWIGLU_X + SUBBLOCK_EX * 0.5),
      ey: 0.12,
      ez: 0.16,
      kind: RIBBON_KIND_BOX,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: isFocusedLayer ? 0.62 : 0.24,
    });

    // Inter-layer downward spine arrow
    if (l < NUM_LAYERS - 1) {
      const nextY = layerCenterY(l + 1);
      const midY = (y - SUBBLOCK_EY * 0.5 + (nextY + SUBBLOCK_EY * 0.5)) * 0.5;
      const gap = Math.max(0.18, y - SUBBLOCK_EY * 0.5 - (nextY + SUBBLOCK_EY * 0.5));
      ribbons.push({
        cx: RESIDUAL_RAIL_X,
        cy: midY,
        cz: 0,
        ex: 0.26,
        ey: gap * 0.88,
        ez: 0.22,
        kind: RIBBON_KIND_ARROW_NEG_Y,
        colorId: RIBBON_COLOR_EMERALD,
        alpha: 0.34,
      });
    }

    // Clickable PickBox covering the paired [RoPE GQA | SwiGLU] layer block
    solidPickBoxes.push({
      bayId,
      partId,
      semanticTag: 'layer_block',
      min: [GQA_X - SUBBLOCK_EX * 0.5, y - SUBBLOCK_EY * 0.5, -SUBBLOCK_EZ * 0.5],
      max: [SWIGLU_X + SUBBLOCK_EX * 0.5, y + SUBBLOCK_EY * 0.5, SUBBLOCK_EZ * 0.5],
    });

    // Decade & selected landmarks
    if (l === 0 || l === 9 || l === 19 || l === 29 || l === 39 || l === selectedLayer) {
      landmarks.push({
        id: `lm-${partId}`,
        label: `Layer ${String(bayId).padStart(2, '0')} · [RoPE GQA | SwiGLU]`,
        sublabel:
          l === selectedLayer
            ? `Active focus · ${ctx.modeAtRecord} · ${kvProvenance}`
            : '32Q/8KV YaRN RoPE + Dense SwiGLU (d=8→24→8)',
        world: [RESIDUAL_RAIL_X + 0.55, y, 0],
        bayId,
        partId,
        tone: l === selectedLayer ? 'gold' : 'default',
        align: 'left',
        tier: l === 0 || l === 39 || l === selectedLayer ? 'primary' : 'detail',
      });
    }

    partCatalog[partId] = {
      id: partId,
      title: `Layer ${String(bayId).padStart(2, '0')} / 40 (Homogeneous Dense)`,
      badge: loraBridged ? 'GQA + SwiGLU · LoRA Draft ON' : 'GQA + SwiGLU · Backbone Only',
      summary:
        'Single physical Ministral-3 decoder layer reused across AR, Block-Diffusion, and Linear Speculation via attention mask switching (zero duplicated towers).',
      specs: [
        'Attn: Pre-RMSNorm -> 32Q/8KV (4:1) GQA + Dyadic YaRN RoPE + Llama-4 Q-scale',
        DIFFUSION_PROJECTION_REVERSAL_BADGE,
        'MLP: Post-Attn RMSNorm -> Dense SwiGLU (mini 8→24→8 · prod 5120→16384→5120)',
        loraBridged
          ? 'o_proj LoRA: BRIDGED (+4.0·B(A·a) on draft pass; OFF on causal verify)'
          : 'o_proj LoRA: DETACHED (pure backbone W_O·a)',
      ],
    };
  }

  // Continuous vertical residual spine beam on +X
  const spineTopY = embedY - 0.62;
  const spineBotY = finalNormY + 0.30;
  ribbons.push({
    cx: RESIDUAL_RAIL_X,
    cy: (spineTopY + spineBotY) * 0.5,
    cz: 0,
    ex: 0.22,
    ey: Math.abs(spineTopY - spineBotY),
    ez: 0.20,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.30,
  });

  // --------------------------------------------------------------------------
  // 3. BOTTOM BOOKEND: `encoder.norm` + `diffusion_head [36, 8]`
  // --------------------------------------------------------------------------
  const headActive =
    selectedPartId === 'diffusion_head' || hoveredPartId === 'diffusion_head';

  // Final RMSNorm bar (`encoder.norm`)
  solids.push({
    cx: 0,
    cy: finalNormY,
    cz: 0,
    ex: 4.50,
    ey: 0.52,
    ez: 1.06,
    cellFamily: CELL_NEUTRAL,
    fp8: 0,
    phase: 0,
    bayId: 41,
    cellScale: 0.22,
    highlight: headActive ? 1 : 0,
  });
  lines.addBoxEdges(
    0,
    finalNormY,
    0,
    4.50,
    0.52,
    1.06,
    ownerCounter++,
    headActive ? 1.35 : 0.95,
    headActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    0.72,
  );

  // Untied `diffusion_head [36, 8]`
  solids.push({
    cx: 0,
    cy: diffHeadY,
    cz: 0,
    ex: 5.20,
    ey: 1.24,
    ez: 1.24,
    cellFamily: CELL_EMBED_SLATE,
    fp8: 0,
    phase: 1,
    bayId: 41,
    cellScale: 0.24,
    highlight: headActive ? 1 : 0,
  });
  lines.addBoxEdges(
    0,
    diffHeadY,
    0,
    5.20,
    1.24,
    1.24,
    ownerCounter++,
    headActive ? 1.55 : 1.05,
    headActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    headActive ? 0.92 : 0.74,
  );

  // Flow arrows Layer 40 -> encoder.norm -> diffusion_head
  ribbons.push({
    cx: 0,
    cy: (bottomLayerY - SUBBLOCK_EY * 0.5 + (finalNormY + 0.26)) * 0.5,
    cz: 0,
    ex: 0.40,
    ey: Math.max(0.3, bottomLayerY - SUBBLOCK_EY * 0.5 - (finalNormY + 0.26) - 0.10),
    ez: 0.26,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.44,
  });
  ribbons.push({
    cx: 0,
    cy: (finalNormY - 0.26 + (diffHeadY + 0.62)) * 0.5,
    cz: 0,
    ex: 0.40,
    ey: Math.max(0.3, finalNormY - 0.26 - (diffHeadY + 0.62) - 0.10),
    ez: 0.26,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.48,
  });

  solidPickBoxes.push({
    bayId: 41,
    partId: 'diffusion_head',
    semanticTag: 'diffusion_head',
    min: [-2.60, diffHeadY - 0.62, -0.62],
    max: [2.60, finalNormY + 0.26, 0.62],
  });
  landmarks.push({
    id: 'lm-diffusion-head',
    label: 'encoder.norm + diffusion_head [36, 8]',
    sublabel: 'Shared untied output head · causal (+1) & diffusion (+0)',
    world: [3.15, diffHeadY, 0],
    bayId: 41,
    partId: 'diffusion_head',
    align: 'left',
    tier: 'primary',
  });
  partCatalog['diffusion_head'] = {
    id: 'diffusion_head',
    title: 'encoder.norm + diffusion_head',
    badge: 'Untied V=36 · Raw Logits',
    summary:
      'Final Ministral-3 RMSNorm followed by a single untied diffusion_head [36, 8] used for AR next-token (+1), Block-Diffusion same-position (+0), and Speculative Verify logits. [MASK] logits are never suppressed.',
    specs: [
      `Final norm: [${cfg.hiddenSize}] (eps=${cfg.rmsNormEps})`,
      `Mini head: [${cfg.vocabSize}, ${cfg.hiddenSize}]`,
      'Prod 14B head: [131,072, 5,120] (671,088,640 params · 1.25 GiB BF16)',
    ],
  };

  // --------------------------------------------------------------------------
  // 4. 0..31 KV CACHE RAIL ALONG `+Z`
  // --------------------------------------------------------------------------
  const kvRailY = layerCenterY(selectedLayer);
  const kvActive = selectedPartId === 'kv_rail' || hoveredPartId === 'kv_rail';

  // Vertical collector bus on front of GQA stack showing all 40 layers write/read KV
  ribbons.push({
    cx: KV_RAIL_X,
    cy: (topLayerY + bottomLayerY) * 0.5,
    cz: SUBBLOCK_EZ * 0.5 + 0.22,
    ex: 0.28,
    ey: Math.abs(topLayerY - bottomLayerY) + SUBBLOCK_EY,
    ez: 0.14,
    kind: RIBBON_KIND_BOX,
    colorId: kvProvenance === 'token_causal' ? RIBBON_COLOR_EMERALD : RIBBON_COLOR_CYAN,
    alpha: 0.26,
  });

  // Outward arrow from selected layer GQA to KV rail
  ribbons.push({
    cx: KV_RAIL_X,
    cy: kvRailY,
    cz: (SUBBLOCK_EZ * 0.5 + KV_START_Z) * 0.5,
    ex: 0.42,
    ey: 0.24,
    ez: KV_START_Z - SUBBLOCK_EZ * 0.5 - 0.08,
    kind: RIBBON_KIND_ARROW_POS_Z,
    colorId: kvProvenance === 'token_causal' ? RIBBON_COLOR_EMERALD : RIBBON_COLOR_CYAN,
    alpha: 0.58,
  });

  // Render committed slabs 0..kvSeqLen-1 and empty ghost ticks kvSeqLen..kvCapacity-1
  for (let pos = 0; pos < kvCapacity; pos++) {
    const cz = kvSlotCenterZ(pos);
    const z0 = cz - KV_SLOT_DEPTH_Z * 0.5;
    const z1 = cz + KV_SLOT_DEPTH_Z * 0.5;

    if (pos < kvSeqLen) {
      const slab = ctx.kvSnapshot.slabs[pos];
      const slabProv = slab ? slab.maskProvenance : kvProvenance;
      const isCausalSlab = slabProv === 'token_causal';
      const isLlama4Extended = pos >= cfg.rope.originalMaxPositionEmbeddings;

      solids.push({
        cx: KV_RAIL_X,
        cy: kvRailY,
        cz,
        ex: KV_SLAB_EX,
        ey: KV_SLAB_EY,
        ez: KV_SLOT_DEPTH_Z,
        cellFamily: isCausalSlab ? CELL_ACTIVATION_GREEN : CELL_SSM_CYAN,
        fp8: 0,
        phase: isCausalSlab ? 1.0 + pos * 0.05 : 2.0 + pos * 0.05,
        bayId: 42,
        cellScale: 0.20,
        highlight: kvActive ? 1 : isLlama4Extended ? 2 : 0,
      });
      lines.addBoxEdges(
        KV_RAIL_X,
        kvRailY,
        cz,
        KV_SLAB_EX,
        KV_SLAB_EY,
        KV_SLOT_DEPTH_Z,
        ownerCounter++,
        kvActive ? 1.35 : isLlama4Extended ? 1.1 : 0.85,
        kvActive
          ? LINE_STYLE_ACTIVE
          : isLlama4Extended
            ? LINE_STYLE_BF16_GOLD
            : LINE_STYLE_NEUTRAL,
        kvActive ? 0.90 : 0.65,
      );
    } else {
      // Ghost capacity slot outline
      lines.addBoxFrameXZ(
        KV_RAIL_X - KV_SLAB_EX * 0.5,
        z0,
        KV_RAIL_X + KV_SLAB_EX * 0.5,
        z1,
        kvRailY - KV_SLAB_EY * 0.5,
        0.8,
        LINE_STYLE_GUIDE,
        0.34,
      );
      lines.addSegment(
        KV_RAIL_X - KV_SLAB_EX * 0.5,
        kvRailY - KV_SLAB_EY * 0.5,
        cz,
        KV_RAIL_X - KV_SLAB_EX * 0.5,
        kvRailY + KV_SLAB_EY * 0.25,
        cz,
        0.75,
        LINE_STYLE_GUIDE,
        0.28,
      );
    }
  }

  // Full 0..31 rail perimeter guide & Llama-4 pos=16 threshold line
  const kvMaxZ = kvSlotCenterZ(kvCapacity - 1) + KV_SLOT_DEPTH_Z * 0.5;
  lines.addDashedFrameXZ(
    KV_RAIL_X - KV_SLAB_EX * 0.5 - 0.12,
    KV_START_Z - 0.10,
    KV_RAIL_X + KV_SLAB_EX * 0.5 + 0.12,
    kvMaxZ + 0.10,
    kvRailY - KV_SLAB_EY * 0.5 - 0.04,
    LINE_STYLE_GUIDE,
    0.45,
  );
  if (cfg.rope.originalMaxPositionEmbeddings < kvCapacity) {
    const l4Z =
      kvSlotCenterZ(cfg.rope.originalMaxPositionEmbeddings) - KV_SLOT_PITCH_Z * 0.5;
    lines.addSegment(
      KV_RAIL_X - KV_SLAB_EX * 0.5 - 0.35,
      kvRailY - KV_SLAB_EY * 0.5,
      l4Z,
      KV_RAIL_X + KV_SLAB_EX * 0.5 + 0.35,
      kvRailY - KV_SLAB_EY * 0.5,
      l4Z,
      1.35,
      LINE_STYLE_BF16_GOLD,
      0.85,
    );
  }

  // KV depth ticks (1-based token ticks matching overlay conventions)
  const tickSet = new Set<number>([1, 8, 16, 24, kvCapacity]);
  if (kvSeqLen > 0 && kvSeqLen <= kvCapacity) {
    tickSet.add(kvSeqLen);
  }
  const sortedTicks = Array.from(tickSet).sort((a, b) => a - b);
  for (const tok of sortedTicks) {
    kvDepthTicks.push({
      token: tok,
      world: [KV_RAIL_X + KV_SLAB_EX * 0.5 + 0.32, kvRailY, kvSlotCenterZ(tok - 1)],
    });
  }

  solidPickBoxes.push({
    bayId: 42,
    partId: 'kv_rail',
    semanticTag: 'kv_rail',
    min: [
      KV_RAIL_X - KV_SLAB_EX * 0.5,
      kvRailY - KV_SLAB_EY * 0.5,
      KV_START_Z - 0.08,
    ],
    max: [
      KV_RAIL_X + KV_SLAB_EX * 0.5,
      kvRailY + KV_SLAB_EY * 0.5,
      kvMaxZ + 0.08,
    ],
  });
  landmarks.push({
    id: 'lm-kv-rail',
    label: `KV Cache Rail (${kvSeqLen}/${kvCapacity})`,
    sublabel:
      kvProvenance === 'token_causal'
        ? 'token_causal · 40L × 8KV × dHead=8 (20 KiB/tok)'
        : 'block_bidirectional · 40L × 8KV × dHead=8 (20 KiB/tok)',
    world: [KV_RAIL_X, kvRailY + KV_SLAB_EY * 0.5 + 0.45, kvSlotCenterZ(Math.min(kvCapacity - 1, Math.max(0, kvSeqLen - 1)))],
    bayId: 42,
    partId: 'kv_rail',
    tone: kvProvenance === 'token_causal' ? 'default' : 'cyan',
    align: 'bottom',
    tier: 'primary',
  });
  partCatalog['kv_rail'] = {
    id: 'kv_rail',
    title: '0..31 Persistent KV Cache Rail',
    badge: `${kvSeqLen}/${kvCapacity} slots · ${kvProvenance}`,
    summary:
      'Immutable per-position shared KV slab pool across all 40 layers (8 KV heads × 8 headDim × K/V = 20.00 KiB/tok). Active slabs distinguish token_causal vs block_bidirectional provenance; pos >= 16 triggers live Llama-4 query temperature scaling.',
    specs: [
      `Committed slabs: ${kvSeqLen} / ${kvCapacity} (${(kvSeqLen * 20).toFixed(0)} KiB)`,
      `KV provenance: ${kvProvenance}`,
      'Llama-4 threshold: pos 0..15 -> ×1.000 · pos 16..31 -> ×1.0693',
    ],
  };

  // --------------------------------------------------------------------------
  // 5. ONE DETACHABLE `o_proj` LoRA RAIL (Bridged ONLY on `linear_spec_lora` draft)
  // --------------------------------------------------------------------------
  const loraX = loraBridged ? LORA_RAIL_BRIDGED_X : LORA_RAIL_DETACHED_X;
  const loraMidY = (topLayerY + bottomLayerY) * 0.5;
  const loraSpanY = Math.abs(topLayerY - bottomLayerY) + SUBBLOCK_EY;
  const loraActive = selectedPartId === 'lora_rail' || hoveredPartId === 'lora_rail';

  solids.push({
    cx: loraX,
    cy: loraMidY,
    cz: 0,
    ex: LORA_RAIL_EX,
    ey: loraSpanY,
    ez: LORA_RAIL_EZ,
    cellFamily: loraBridged ? CELL_MOE_AMBER : CELL_NEUTRAL,
    fp8: loraBridged ? 0 : 1,
    phase: loraBridged ? 3.0 : 0.0,
    bayId: 43,
    cellScale: 0.24,
    highlight: loraActive || loraBridged ? 1 : 0,
  });
  lines.addBoxEdges(
    loraX,
    loraMidY,
    0,
    LORA_RAIL_EX,
    loraSpanY,
    LORA_RAIL_EZ,
    ownerCounter++,
    loraBridged ? 1.6 : 0.9,
    loraBridged
      ? LINE_STYLE_BF16_GOLD
      : loraActive
        ? LINE_STYLE_ACTIVE
        : LINE_STYLE_GUIDE,
    loraBridged ? 0.96 : 0.42,
  );

  const gqaLeftEdgeX = GQA_X - SUBBLOCK_EX * 0.5;
  const loraRightEdgeX = loraX + LORA_RAIL_EX * 0.5;

  if (loraBridged) {
    // Glowing amber bridge ribbons onto ALL 40 layers' o_proj nodes
    const bridgeCx = (loraRightEdgeX + gqaLeftEdgeX) * 0.5;
    const bridgeEx = Math.max(0.25, gqaLeftEdgeX - loraRightEdgeX - 0.08);
    for (let l = 0; l < NUM_LAYERS; l++) {
      const y = layerCenterY(l);
      ribbons.push({
        cx: bridgeCx,
        cy: y,
        cz: 0,
        ex: bridgeEx,
        ey: 0.22,
        ez: 0.26,
        kind: RIBBON_KIND_ARROW_POS_X,
        colorId: RIBBON_COLOR_AMBER,
        alpha: l === selectedLayer ? 0.88 : 0.64,
      });
    }
  } else {
    // Physically detached: subtle retraction guide ticks at decade layers (zero active bridges)
    for (const l of [0, 10, 20, 30, 39]) {
      const y = layerCenterY(l);
      lines.addSegment(
        loraRightEdgeX + 0.15,
        y,
        0,
        gqaLeftEdgeX - 0.35,
        y,
        0,
        0.85,
        LINE_STYLE_GUIDE,
        0.30,
      );
    }
  }

  solidPickBoxes.push({
    bayId: 43,
    partId: 'lora_rail',
    semanticTag: 'lora_rail',
    min: [
      loraX - LORA_RAIL_EX * 0.5,
      loraMidY - loraSpanY * 0.5,
      -LORA_RAIL_EZ * 0.5,
    ],
    max: [
      loraX + LORA_RAIL_EX * 0.5,
      loraMidY + loraSpanY * 0.5,
      LORA_RAIL_EZ * 0.5,
    ],
  });
  landmarks.push({
    id: 'lm-lora-rail',
    label: loraBridged ? 'o_proj LoRA: BRIDGED (draft only)' : 'o_proj LoRA: DETACHED',
    sublabel: DIFFUSION_LORA_RAIL_BADGE,
    world: [loraX - LORA_RAIL_EX * 0.5 - 0.45, topLayerY - 2.0, 0],
    bayId: 43,
    partId: 'lora_rail',
    tone: loraBridged ? 'gold' : 'muted',
    align: 'right',
    tier: 'primary',
  });
  partCatalog['lora_rail'] = {
    id: 'lora_rail',
    title: loraBridged ? 'o_proj LoRA Rail: BRIDGED' : 'o_proj LoRA Rail: DETACHED',
    badge: DIFFUSION_LORA_RAIL_BADGE,
    summary: loraBridged
      ? 'Draft-only o_proj LoRA rail is actively bridged onto all 40 layers (+4.0·B(A·a)) during linear_spec_lora draft passes, and strictly detached (0 ULP invariance) on causal verify.'
      : 'o_proj LoRA: DETACHED — retracted outward along -X. Pure backbone weights run with 0 ULP LoRA-OFF invariance across AR, DLM, linear_spec, and verify passes.',
    specs: [
      `Status: ${loraBridged ? 'BRIDGED onto all 40 layers (draft pass)' : 'DETACHED (retracted along -X)'}`,
      `Mini adapter: 80 tensors (40 × lora_A [4,256] + lora_B [8,4], r=${cfg.loraRank}, α=${cfg.loraAlpha}, scale=${cfg.loraScale.toFixed(1)})`,
      'Prod 14B adapter: +0.35% params (47,185,920 F32 params, r=128, α=512, scale=4.0)',
    ],
  };

  // Projection-width reversal caliper along Layer 01
  calipers.push({
    id: 'cal-macro-reversal',
    aWorld: [GQA_X - SUBBLOCK_EX * 0.5, topLayerY + 0.85, 0],
    bWorld: [SWIGLU_X + SUBBLOCK_EX * 0.5, topLayerY + 0.85, 0],
    label: DIFFUSION_PROJECTION_REVERSAL_BADGE,
    partIds: ['layer-01'],
  });

  // --------------------------------------------------------------------------
  // 6. PACK BUFFERS, PROVENANCE BADGES & FIT BOUNDS
  // --------------------------------------------------------------------------
  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();

  const provenanceBadges: DiffusionProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    { label: DIFFUSION_PROJECTION_REVERSAL_BADGE, tone: 'muted' },
    {
      label: DIFFUSION_LORA_RAIL_BADGE,
      tone: loraBridged ? 'lora' : 'muted',
    },
  ];

  const fitBounds: AABB = {
    min: [LORA_RAIL_DETACHED_X - 1.4, diffHeadY - 1.4, -2.0],
    max: [RESIDUAL_RAIL_X + 5.8, embedY + 2.2, kvMaxZ + 1.2],
  };

  return {
    viewMode: 'macro',
    title: '40-Layer Homogeneous Dense Backbone',
    subtitle: `Single 40L Ministral-3 spine · KV ${kvSeqLen}/${kvCapacity} (${kvProvenance}) · LoRA ${loraBridged ? 'BRIDGED (draft)' : 'DETACHED'}`,
    layerIndex: selectedLayer,
    solidData: packedSolids.data,
    solidCount: packedSolids.count,
    lineData: packedLines.data,
    lineCount: packedLines.count,
    ribbonData: packedRibbons.data,
    ribbonCount: packedRibbons.count,
    fitBounds,
    defaultCameraAzimuthDeg: 30,
    defaultCameraElevationDeg: 14,
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze(kvDepthTicks),
    calipers: Object.freeze(calipers),
    solidPickBoxes: Object.freeze(solidPickBoxes),
    fasciaPickBoxes: Object.freeze(fasciaPickBoxes),
    partCatalog: Object.freeze(partCatalog),
    provenanceBadges: Object.freeze(provenanceBadges),
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner,
    diagnostics: Object.freeze({
      layerCount: NUM_LAYERS,
      loraBridged,
      kvSeqLen,
      kvCapacity,
      loraRailX: loraX,
      kvMaskProvenance: kvProvenance,
    }),
  };
}
