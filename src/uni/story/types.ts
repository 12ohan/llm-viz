/**
 * Pinned UNI6 Guided Pedagogical Tour Contracts (`src/uni/story/types.ts`)
 *
 * Defines the 5-Act, 28-Beat interactive tour contracts for Nemotron-Labs-Diffusion-Uni
 * (`UNI0_ARCHITECTURE_AUDIT.md` §9):
 *   - Act I:   Unified Multimodal Challenge & Elastic MoT (View 1 Macro)
 *   - Act II:  Dual Visual Pipelines — RADIO-H vs. Emu3 IBQ (View 2 Visual Input)
 *   - Act III: Joint vs. Split-Tower Attention & Coordinates (View 3 Joint Attn)
 *   - Act IV:  Ball-Tree Geometry, Progressive Self-Refinement & Dual-CFG Editing (Views 4–6)
 *   - Act V:   Understanding, Grounding & Cross-Paradigm OPD (Views 6–7)
 *
 * Isolation guarantee: the tour executes strictly against its own private deterministic
 * `UniSession`, ensuring entering, stepping, and exiting never mutates the user's
 * Free-Explore `UniSession`.
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, `src/ui/controls.ts`,
 * `src/twotower/`, or `src/diffusion/` imports.
 */

import type { UniSpatialFocusState, UniSpatialScene, UniSpatialViewMode } from '../scene/types';
import type { UniSession } from '../session';
import type {
  UniElasticPrefillMode,
  UniExecutionViewId,
  UniSessionTaskMode,
} from '../types';

export const UNI_TOUR_TOTAL_ACTS = 5 as const;
export const UNI_TOUR_TOTAL_BEATS = 28 as const;

export type UniTourActId = 'act1' | 'act2' | 'act3' | 'act4' | 'act5';

export interface UniTourActSummary {
  readonly id: UniTourActId;
  readonly actNumber: 1 | 2 | 3 | 4 | 5;
  readonly title: string;
  readonly shortTitle: string;
  readonly subtitle: string;
  readonly startBeatIndex: number;
  readonly beatCount: number;
}

export type UniTourDemoKey =
  | 't2i_elastic'
  | 't2i_hf_faithful'
  | 'edit_panorama_elastic'
  | 'edit_panorama_hf'
  | 'visual_understanding'
  | 'text_understanding'
  | 'object_grounding';

export interface UniTourBeatSpec {
  /** Zero-based global beat index (`0..27`). */
  readonly index: number;
  readonly id: string;
  readonly actId: UniTourActId;
  readonly actNumber: 1 | 2 | 3 | 4 | 5;
  readonly beatInAct: number;
  readonly title: string;
  readonly takeaway: string;
  readonly narrative: string;
  readonly equation?: string;
  readonly ledgerRefs: readonly string[];
  readonly viewMode: UniSpatialViewMode;
  readonly taskMode: UniSessionTaskMode;
  readonly executionViewId: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly demoKey: UniTourDemoKey;
  /** Target step offset inside the beat's private tour branch (`0` = prefill, `-1` = final step). */
  readonly targetStepOffset: number;
  /** Partial focus state merged over `DEFAULT_UNI_SPATIAL_FOCUS_STATE` when this beat activates. */
  readonly focusPatch: Partial<UniSpatialFocusState>;
  /** Optional partId highlighted on the 3D scene during this beat. */
  readonly highlightPartId?: string | null;
  /** Optional camera angle override (`azimuthDeg`, `elevationDeg`). */
  readonly cameraOverride?: {
    readonly azimuthDeg: number;
    readonly elevationDeg: number;
  };
}

export interface UniFreeExploreRestoreState {
  readonly viewMode: UniSpatialViewMode;
  readonly focus: UniSpatialFocusState;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
}

export interface UniTourState {
  readonly active: boolean;
  readonly beatIndex: number;
  readonly beatCount: number;
  readonly actIndex: number;
  readonly actCount: number;
  readonly currentAct: UniTourActSummary;
  readonly currentBeat: UniTourBeatSpec;
  readonly activeRecordId: string | null;
  readonly focus: UniSpatialFocusState;
}

export interface UniTourController {
  readonly acts: readonly UniTourActSummary[];
  readonly beats: readonly UniTourBeatSpec[];
  readonly totalBeats: number;
  /** Strictly isolated private `UniSession` owned exclusively by the tour. */
  readonly tourSession: UniSession;
  getState(): UniTourState;
  isActive(): boolean;
  start(
    restoreState?: UniFreeExploreRestoreState | null,
    initialBeatIndex?: number,
  ): UniTourState;
  goToBeat(beatIndex: number): UniTourState;
  nextBeat(): UniTourState;
  prevBeat(): UniTourState;
  goToAct(actId: UniTourActId): UniTourState;
  /** Build the live 3D `UniSpatialScene` for the current (or specified) beat using the isolated tour session. */
  buildBeatScene(beatIndex?: number): UniSpatialScene;
  /** Exit the tour and return the saved Free-Explore state untouched. */
  exit(): UniFreeExploreRestoreState | null;
}
