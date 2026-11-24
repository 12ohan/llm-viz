import {
  moeStepperCellForState,
  type MoeCellRef,
  type MoeSimResult,
  type MoeTensorKey,
} from '../math/moeSim';

/** Tensors displaying a focused-token routing snapshot rather than a multi-row T-sequence strip. */
export function isMoeSnapshotTensor(tensor: MoeTensorKey): boolean {
  return (
    tensor === 'router_bank' ||
    tensor === 'topk_weights' ||
    tensor === 'routed_w_up' ||
    tensor === 'routed_pre' ||
    tensor === 'routed_act' ||
    tensor === 'routed_w_down' ||
    tensor === 'shared_act'
  );
}

/** Token-independent learned weight matrices in MoE. */
export function isMoeWeightTensor(tensor: MoeTensorKey): boolean {
  return tensor === 'w_router' || tensor === 'shared_w_up' || tensor === 'shared_w_down';
}

/**
 * Strip any stale `tokenIndex` from token-independent weight cells (`w_router`, `shared_w_up`,
 * `shared_w_down`) so stationary weight hover/pinning never overrides the active routing token.
 */
export function normalizeMoeWeightCell(cell: MoeCellRef): MoeCellRef {
  if (!isMoeWeightTensor(cell.tensor) || cell.tokenIndex === undefined) {
    return cell;
  }
  const { tokenIndex: _omit, ...rest } = cell;
  return rest;
}

/** Clamp active MoE focus token to `[0..tokens-1]` (defaults to `tokens - 1`). */
export function resolveActiveMoeFocusToken(
  moeFocusToken: number | null,
  tokens: number,
): number {
  const clampedT = Math.max(1, Math.round(tokens));
  const raw = moeFocusToken === null ? clampedT - 1 : moeFocusToken;
  return Math.max(0, Math.min(clampedT - 1, Math.round(raw)));
}

/** Remap a MoE cell reference onto token `tok` (synchronizing Top-6 slot expert when applicable). */
export function remapMoeCellToken(
  cell: MoeCellRef,
  tok: number,
  sim: MoeSimResult,
): MoeCellRef {
  const clampedTok = Math.max(0, Math.min(sim.tokens - 1, Math.round(tok)));
  if (isMoeWeightTensor(cell.tensor)) {
    return normalizeMoeWeightCell(cell);
  }
  if (cell.tensor === 'topk_weights') {
    const slot = Math.max(0, Math.min(sim.topK - 1, cell.rankSlot ?? cell.col));
    const winnerId = sim.topk_indices[clampedTok * sim.topK + slot];
    return { ...cell, tokenIndex: clampedTok, rankSlot: slot, col: slot, expertId: winnerId };
  }
  if (isMoeSnapshotTensor(cell.tensor)) {
    return { ...cell, tokenIndex: clampedTok };
  }
  if (cell.tensor === 'routed_sum') {
    return { tensor: 'routed_sum', row: clampedTok, col: cell.col, tokenIndex: clampedTok };
  }
  return { ...cell, row: clampedTok, tokenIndex: clampedTok };
}

export interface MoeHoverCyclingResolution {
  readonly hoveredCell: MoeCellRef | null;
  readonly shouldPauseTokenCycling: boolean;
}

/**
 * Explicit Interaction Policy between Hover Inspection and MoE Per-Token Cycling:
 *
 * 1. Hovering a focused-token snapshot tensor (`router_bank`, `topk_weights`, `routed_*`,
 *    `shared_act`) while token cycling is active: cycling CONTINUES and the hovered cell tracks
 *    the advancing `moeFocusToken` live (e.g. hovering `topk_weights` slot #1 streams each token's
 *    rank-1 winner; hovering a bank tile watches it flip active/idle across tokens).
 * 2. Hovering token-independent weights (`w_router`, `shared_w_up`, `shared_w_down`) or empty
 *    canvas while cycling: cycling CONTINUES uninterrupted with `tokenIndex` stripped so the
 *    routing snapshot, atlas, and formula upstream spans advance smoothly across tokens.
 * 3. Hovering a sequence-row tensor (`input`, `rmsnorm`, `routed_sum`, `shared_out`, `moe_out`,
 *    `residual`) while cycling: cycling CLEANLY PAUSES so the user inspects that exact token row
 *    directly under the cursor without timer desynchronization.
 */
export function resolveMoeHoverDuringCycling(
  hitCell: MoeCellRef | null,
  moeTokenCycling: boolean,
  activeFocusToken: number,
  sim: MoeSimResult,
): MoeHoverCyclingResolution {
  if (!hitCell) {
    return { hoveredCell: null, shouldPauseTokenCycling: false };
  }
  if (isMoeWeightTensor(hitCell.tensor)) {
    return { hoveredCell: normalizeMoeWeightCell(hitCell), shouldPauseTokenCycling: false };
  }
  if (!moeTokenCycling) {
    return { hoveredCell: hitCell, shouldPauseTokenCycling: false };
  }
  if (isMoeSnapshotTensor(hitCell.tensor)) {
    return {
      hoveredCell: remapMoeCellToken(hitCell, activeFocusToken, sim),
      shouldPauseTokenCycling: false,
    };
  }
  return { hoveredCell: hitCell, shouldPauseTokenCycling: true };
}

export interface ResolveEffectiveMoeCellParams {
  readonly sim: MoeSimResult;
  readonly moeFocusToken: number | null;
  readonly moeActiveExpert: number | null;
  readonly moeTokenCycling: boolean;
  readonly moeHoveredCell: MoeCellRef | null;
  readonly moeSelectedCell: MoeCellRef | null;
  readonly moeStepperActive: boolean;
  readonly moeStepperStep: number;
  readonly moeStepperSubProgress: number;
}

/** Resolve the effective active MoE cell, preventing stale hover/weight tokens during token cycling. */
export function resolveEffectiveMoeCell(
  params: ResolveEffectiveMoeCellParams,
): MoeCellRef | null {
  const {
    sim,
    moeFocusToken,
    moeActiveExpert,
    moeTokenCycling,
    moeHoveredCell,
    moeSelectedCell,
    moeStepperActive,
    moeStepperStep,
    moeStepperSubProgress,
  } = params;
  const tok = resolveActiveMoeFocusToken(moeFocusToken, sim.tokens);
  const exp =
    moeActiveExpert !== null
      ? Math.max(0, Math.min(sim.nRoutedExperts - 1, moeActiveExpert))
      : sim.topk_indices[tok * sim.topK];

  if (moeHoveredCell) {
    if (isMoeWeightTensor(moeHoveredCell.tensor)) {
      return normalizeMoeWeightCell(moeHoveredCell);
    }
    if (moeTokenCycling && isMoeSnapshotTensor(moeHoveredCell.tensor)) {
      return remapMoeCellToken(moeHoveredCell, tok, sim);
    }
    return moeHoveredCell;
  }

  if (moeSelectedCell) {
    if (isMoeWeightTensor(moeSelectedCell.tensor)) {
      return normalizeMoeWeightCell(moeSelectedCell);
    }
    if (moeTokenCycling) {
      return remapMoeCellToken(moeSelectedCell, tok, sim);
    }
    return moeSelectedCell;
  }

  if (moeStepperActive) {
    return moeStepperCellForState(
      sim,
      moeStepperStep,
      moeStepperSubProgress,
      tok,
      exp,
    );
  }

  return null;
}

export interface MoeTokenStepResult {
  readonly nextToken: number;
  readonly nextSelectedCell: MoeCellRef | null;
}

/** Step the MoE focused token by `delta` (`-1` or `+1`, wrapping modulo `T`) and keep any pinned cell synced. */
export function stepMoeFocusToken(
  moeFocusToken: number | null,
  delta: -1 | 1,
  sim: MoeSimResult,
  moeSelectedCell: MoeCellRef | null,
): MoeTokenStepResult {
  const cur = resolveActiveMoeFocusToken(moeFocusToken, sim.tokens);
  const nextToken = (cur + delta + sim.tokens) % sim.tokens;
  const nextSelectedCell =
    moeSelectedCell !== null ? remapMoeCellToken(moeSelectedCell, nextToken, sim) : null;
  return { nextToken, nextSelectedCell };
}

export interface MoeTokenCycleTickResult {
  readonly nextToken: number;
  readonly nextSelectedCell: MoeCellRef | null;
  readonly nextHoveredCell: MoeCellRef | null;
}

/** Advance MoE token cycling by one token (`(cur + 1) % T`) and sync pinned/snapshot/weight hovered cells. */
export function advanceMoeTokenCycleTick(
  moeFocusToken: number | null,
  sim: MoeSimResult,
  moeSelectedCell: MoeCellRef | null,
  moeHoveredCell: MoeCellRef | null,
): MoeTokenCycleTickResult {
  const { nextToken, nextSelectedCell } = stepMoeFocusToken(moeFocusToken, 1, sim, moeSelectedCell);
  const nextHoveredCell =
    moeHoveredCell !== null
      ? isMoeSnapshotTensor(moeHoveredCell.tensor)
        ? remapMoeCellToken(moeHoveredCell, nextToken, sim)
        : isMoeWeightTensor(moeHoveredCell.tensor)
          ? normalizeMoeWeightCell(moeHoveredCell)
          : null
      : null;
  return { nextToken, nextSelectedCell, nextHoveredCell };
}
