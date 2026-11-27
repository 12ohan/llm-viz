/**
 * Hybrid Nemotron 3 Nano / TwoTower Logical Cell Picking Resolvers
 * (`src/scene/hybridPick.ts`)
 *
 * Maps generic WebGL2 `PickHit` results from `src/render/pick.ts` onto logical
 * GQA, Mamba-2, and Granular MoE cell references. Kept out of `src/render/pick.ts`
 * so non-hybrid visualizers (Diffusion-14B, NLD-Uni, future models) never pull
 * Nano's hybrid simulators into `shared(allN)`.
 */

import { inferGqaCellHead, type GqaCellRef, type GqaTensorKey } from '../math/gqaSim';
import {
  inferMambaCellHead,
  MAMBA_SIM_CONFIG,
  mambaGroupForHead,
  type MambaCellRef,
  type MambaTensorKey,
} from '../math/mambaSim';
import {
  MOE_SIM_CONFIG,
  type MoeCellRef,
  type MoeTensorKey,
} from '../math/moeSim';
import type { PickHit } from '../render/pick';

const GQA_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'w_q', 'w_k', 'w_v', 'q', 'kv_k', 'kv_v',
  'scores', 'softmax', 'v_out', 'concat', 'w_o', 'residual',
]);

const MAMBA_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'w_in', 'z', 'xbc_pre', 'conv_buf', 'x',
  'b', 'c', 'dt', 'h_prev', 'h_curr', 'y', 'y_norm', 'w_out', 'residual',
]);

const MAMBA_SEQUENCE_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'z', 'xbc_pre', 'x', 'b', 'c', 'dt', 'y', 'y_norm', 'residual',
]);

const MOE_TENSOR_KEYS = new Set<string>([
  'input', 'rmsnorm', 'w_router', 'router_bank', 'topk_weights',
  'routed_w_up', 'routed_pre', 'routed_act', 'routed_w_down', 'routed_sum',
  'shared_w_up', 'shared_act', 'shared_w_down', 'shared_out', 'moe_out', 'residual',
]);

/** Resolve a ray-cast PickHit on a live GQA sheet into an exact logical GqaCellRef. */
export function pickHitToGqaCell(
  hit: PickHit | null,
  activeGroup0Head = 0,
  activeGroup1Head = 16,
): GqaCellRef | null {
  if (
    !hit ||
    !hit.tensorKey ||
    !GQA_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  const tensor = hit.tensorKey as GqaTensorKey;
  if (tensor === 'scores' || tensor === 'softmax') {
    const grp: 0 | 1 = hit.kvHeadIndex === 1 ? 1 : 0;
    const headIdx = grp === 0 ? activeGroup0Head : activeGroup1Head;
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      headIndex: headIdx,
      kvHeadIndex: grp,
    };
  }
  if (tensor === 'kv_k' || tensor === 'kv_v') {
    return {
      tensor,
      row: 0,
      col: hit.hitCol,
      kvHeadIndex: hit.kvHeadIndex === 1 ? 1 : 0,
      tokenIndex: hit.tokenIndex ?? 0,
    };
  }
  const baseCell: GqaCellRef = {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    headIndex: hit.headIndex,
    kvHeadIndex: hit.kvHeadIndex,
  };
  const inferredHead = inferGqaCellHead(baseCell);
  if (inferredHead !== null && baseCell.headIndex === undefined) {
    return {
      ...baseCell,
      headIndex: inferredHead,
      kvHeadIndex: baseCell.kvHeadIndex ?? (inferredHead < 16 ? 0 : 1),
    };
  }
  return baseCell;
}

/** Resolve a ray-cast PickHit on a live Mamba-2 sheet into an exact logical MambaCellRef. */
export function pickHitToMambaCell(
  hit: PickHit | null,
  activeMambaHead = 0,
  tokens = 8,
): MambaCellRef | null {
  if (
    !hit ||
    !hit.tensorKey ||
    !MAMBA_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  const tensor = hit.tensorKey as MambaTensorKey;
  if (tensor === 'conv_buf') {
    const slot = hit.tapIndex ?? hit.tokenIndex ?? 0;
    return {
      tensor,
      row: slot,
      col: hit.hitCol,
      tapIndex: slot,
      tokenIndex: hit.stateTokenIndex ?? Math.max(0, tokens - 1),
    };
  }
  if (tensor === 'h_prev' || tensor === 'h_curr') {
    const headIdx = hit.headIndex ?? activeMambaHead;
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      headIndex: headIdx,
      groupIndex: mambaGroupForHead(headIdx),
      tokenIndex: hit.tokenIndex ?? Math.max(0, tokens - 1),
    };
  }
  if (tensor === 'b' || tensor === 'c') {
    const grp = Math.floor(hit.hitCol / MAMBA_SIM_CONFIG.dState);
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      groupIndex: grp,
      tokenIndex: hit.hitRow,
    };
  }
  const baseCell: MambaCellRef = {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    headIndex: hit.headIndex,
    groupIndex: hit.groupIndex,
    tokenIndex: MAMBA_SEQUENCE_TENSOR_KEYS.has(tensor) ? hit.hitRow : undefined,
  };
  const inferredHead = inferMambaCellHead(baseCell);
  if (inferredHead !== null && baseCell.headIndex === undefined) {
    return {
      ...baseCell,
      headIndex: inferredHead,
      groupIndex: baseCell.groupIndex ?? mambaGroupForHead(inferredHead),
    };
  }
  return baseCell;
}

/** Resolve a ray-cast PickHit on a live MoE sheet into an exact logical MoeCellRef. */
export function pickHitToMoeCell(
  hit: PickHit | null,
  focusToken: number,
  focusExpert: number,
  topkIndices?: readonly number[] | Int32Array,
): MoeCellRef | null {
  if (
    !hit ||
    !hit.tensorKey ||
    !MOE_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  const tensor = hit.tensorKey as MoeTensorKey;
  const tokIdx = hit.tokenIndex ?? focusToken;

  if (tensor === 'router_bank') {
    const eid = hit.headIndex ?? focusExpert;
    const bankRow = Math.floor(eid / MOE_SIM_CONFIG.bankCols);
    const bankCol = eid % MOE_SIM_CONFIG.bankCols;
    return {
      tensor,
      row: bankRow,
      col: bankCol,
      tokenIndex: tokIdx,
      expertId: eid,
    };
  }
  if (tensor === 'topk_weights') {
    const slot = Math.max(0, Math.min(MOE_SIM_CONFIG.topK - 1, hit.hitCol));
    const eid = topkIndices && topkIndices[slot] !== undefined ? topkIndices[slot] : focusExpert;
    return {
      tensor,
      row: 0,
      col: slot,
      tokenIndex: tokIdx,
      expertId: eid,
      rankSlot: slot,
    };
  }
  if (
    tensor === 'routed_w_up' ||
    tensor === 'routed_pre' ||
    tensor === 'routed_act' ||
    tensor === 'routed_w_down'
  ) {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      tokenIndex: tokIdx,
      expertId: hit.headIndex ?? focusExpert,
    };
  }
  if (tensor === 'w_router') {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      expertId: hit.hitRow,
    };
  }
  if (tensor === 'shared_w_up' || tensor === 'shared_w_down') {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
    };
  }
  if (tensor === 'shared_act') {
    return {
      tensor,
      row: hit.hitRow,
      col: hit.hitCol,
      tokenIndex: tokIdx,
    };
  }
  return {
    tensor,
    row: hit.hitRow,
    col: hit.hitCol,
    tokenIndex: hit.hitRow,
  };
}
