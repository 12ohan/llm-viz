import {
  mambaStepperCellForState,
  type MambaCellRef,
  type MambaSimResult,
  type MambaTensorKey,
} from '../math/mambaSim';

/**
 * Tensors whose displayed matrix represents the fixed-size in-place state snapshot at `stateTokIdx`
 * (rather than a multi-row sequence strip where row index encodes a specific token).
 */
function isMambaFixedStateTensor(tensor: MambaTensorKey): boolean {
  return tensor === 'conv_buf' || tensor === 'h_prev' || tensor === 'h_curr';
}

/** Token-independent learned weight matrices in Mamba-2. */
function isMambaWeightTensor(tensor: MambaTensorKey): boolean {
  return tensor === 'w_in' || tensor === 'w_out';
}

/** Clamp the active Mamba recurrence token index to `[0..tokens-1]` (defaults to `tokens - 1`). */
export function resolveActiveMambaStateToken(
  mambaStateToken: number | null,
  tokens: number,
): number {
  const clampedT = Math.max(1, Math.round(tokens));
  const raw = mambaStateToken === null ? clampedT - 1 : mambaStateToken;
  return Math.max(0, Math.min(clampedT - 1, Math.round(raw)));
}

/** Remap a Mamba cell reference onto token `tok` without altering non-token coordinates. */
export function remapMambaCellToken(
  cell: MambaCellRef,
  tok: number,
  tokens: number,
): MambaCellRef {
  const clampedTok = Math.max(0, Math.min(Math.max(1, Math.round(tokens)) - 1, Math.round(tok)));
  if (isMambaFixedStateTensor(cell.tensor)) {
    return { ...cell, tokenIndex: clampedTok };
  }
  if (isMambaWeightTensor(cell.tensor)) {
    return cell;
  }
  return { ...cell, row: clampedTok, tokenIndex: clampedTok };
}

export interface MambaHoverPlaybackResolution {
  readonly hoveredCell: MambaCellRef | null;
  readonly shouldPauseTemporalPlayback: boolean;
}

/**
 * Explicit Interaction Policy between Hover Inspection and Temporal Recurrence Playback:
 *
 * 1. Hovering fixed-size state (`h_prev`, `h_curr`, `conv_buf`) while playback is running:
 *    playback CONTINUES and the hovered cell tracks the advancing `mambaStateToken` live so its
 *    equation, atlas values, and 3D dependency threads animate in place across `t = 0..T-1`.
 * 2. Hovering token-independent weights (`w_in`, `w_out`) or empty canvas while playing:
 *    playback CONTINUES uninterrupted.
 * 3. Hovering a specific sequence token row (`input`, `rmsnorm`, `xbc_pre`, `z`, `dt`, `x`,
 *    `b`, `c`, `y`, `y_norm`, `residual`) while playback is running:
 *    playback CLEANLY PAUSES so the user can inspect that exact token row under the cursor
 *    without cursor-vs-timer desynchronization, and all controls synchronize to that row's token.
 */
export function resolveMambaHoverDuringPlayback(
  hitCell: MambaCellRef | null,
  mambaTemporalPlaying: boolean,
  activeStateToken: number,
  tokens: number,
): MambaHoverPlaybackResolution {
  if (!hitCell) {
    return { hoveredCell: null, shouldPauseTemporalPlayback: false };
  }
  if (!mambaTemporalPlaying) {
    return { hoveredCell: hitCell, shouldPauseTemporalPlayback: false };
  }
  if (isMambaFixedStateTensor(hitCell.tensor)) {
    return {
      hoveredCell: remapMambaCellToken(hitCell, activeStateToken, tokens),
      shouldPauseTemporalPlayback: false,
    };
  }
  if (isMambaWeightTensor(hitCell.tensor)) {
    return { hoveredCell: hitCell, shouldPauseTemporalPlayback: false };
  }
  // Sequence-row tensor: pause playback so hover inspection owns the displayed token cleanly.
  return { hoveredCell: hitCell, shouldPauseTemporalPlayback: true };
}

export interface ResolveEffectiveMambaCellParams {
  readonly sim: MambaSimResult;
  readonly tokens: number;
  readonly activeHead: number;
  readonly mambaStateToken: number | null;
  readonly mambaTemporalPlaying: boolean;
  readonly mambaHoveredCell: MambaCellRef | null;
  readonly mambaSelectedCell: MambaCellRef | null;
  readonly mambaStepperActive: boolean;
  readonly mambaStepperStep: number;
  readonly mambaStepperSubProgress: number;
}

/**
 * Resolve the active Mamba-2 cell fed to `buildMambaCutaway`, guaranteeing that while temporal
 * playback is running, a stationary hover over a fixed-state sheet (`h_prev`, `h_curr`, `conv_buf`)
 * or a pinned cell never holds a stale token behind `mambaStateToken`.
 */
export function resolveEffectiveMambaCell(
  params: ResolveEffectiveMambaCellParams,
): MambaCellRef | null {
  const {
    sim,
    tokens,
    activeHead,
    mambaStateToken,
    mambaTemporalPlaying,
    mambaHoveredCell,
    mambaSelectedCell,
    mambaStepperActive,
    mambaStepperStep,
    mambaStepperSubProgress,
  } = params;
  const activeTok = resolveActiveMambaStateToken(mambaStateToken, tokens);

  if (mambaHoveredCell) {
    if (mambaTemporalPlaying && isMambaFixedStateTensor(mambaHoveredCell.tensor)) {
      return remapMambaCellToken(mambaHoveredCell, activeTok, tokens);
    }
    return mambaHoveredCell;
  }

  if (mambaSelectedCell) {
    if (mambaTemporalPlaying) {
      return remapMambaCellToken(mambaSelectedCell, activeTok, tokens);
    }
    return mambaSelectedCell;
  }

  if (mambaStepperActive) {
    const cell = mambaStepperCellForState(
      sim,
      activeHead,
      mambaStepperStep,
      mambaStepperSubProgress,
    );
    if (mambaStateToken !== null) {
      return remapMambaCellToken(cell, activeTok, tokens);
    }
    return cell;
  }

  return null;
}

export interface TemporalTickResult {
  readonly nextToken: number;
  readonly nextSelectedCell: MambaCellRef | null;
  readonly nextHoveredCell: MambaCellRef | null;
}

/** Advance temporal playback by one token (`(cur + 1) % T`) and synchronize pinned & fixed-state hovered cells. */
export function advanceMambaTemporalPlaybackTick(
  mambaStateToken: number | null,
  tokens: number,
  mambaSelectedCell: MambaCellRef | null,
  mambaHoveredCell: MambaCellRef | null,
): TemporalTickResult {
  const clampedT = Math.max(1, Math.round(tokens));
  const cur = resolveActiveMambaStateToken(mambaStateToken, clampedT);
  const nextToken = (cur + 1) % clampedT;

  const nextSelectedCell =
    mambaSelectedCell !== null
      ? remapMambaCellToken(mambaSelectedCell, nextToken, clampedT)
      : null;

  const nextHoveredCell =
    mambaHoveredCell !== null
      ? isMambaFixedStateTensor(mambaHoveredCell.tensor)
        ? remapMambaCellToken(mambaHoveredCell, nextToken, clampedT)
        : isMambaWeightTensor(mambaHoveredCell.tensor)
          ? mambaHoveredCell
          : null
      : null;

  return { nextToken, nextSelectedCell, nextHoveredCell };
}
