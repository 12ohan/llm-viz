import type { GqaCellRef } from '../math/gqaSim';
import type { MambaCellRef } from '../math/mambaSim';
import type { MoeCellRef } from '../math/moeSim';
import type { PrecisionMode, Vec3 } from './spec';

export interface CameraPose {
  target: Vec3;
  azimuthDeg: number;
  elevationDeg: number;
  distance: number;
  fovDeg: number;
}

export type ViewMode =
  | 'tower'
  | 'cutaway_mamba'
  | 'cutaway_moe'
  | 'cutaway_gqa';

export type GqaViewAngle = 'three_quarter' | 'near_plan';

export type TocSectionId =
  | 'overview'
  | 'embedding'
  | 'mamba'
  | 'moe'
  | 'gqa'
  | 'kv_fp8'
  | 'output';

export interface SceneState {
  mode: ViewMode;
  section: TocSectionId;
  precision: PrecisionMode;
  tokens: number;
  hoveredBay: number | null;
  selectedBay: number | null;
  hoveredPart: string | null;
  selectedPart: string | null;
  mambaBayId: 1 | 42;
  mambaViewAngle: GqaViewAngle;
  mambaActiveHead: number;
  /** null means follow latest token (T - 1); integer 0..T-1 pins temporal recurrence snapshot t. */
  mambaStateToken: number | null;
  mambaTemporalPlaying: boolean;
  mambaHoveredCell: MambaCellRef | null;
  mambaSelectedCell: MambaCellRef | null;
  mambaStepperActive: boolean;
  mambaStepperPlaying: boolean;
  mambaStepperStep: number;
  mambaStepperSubProgress: number;
  gqaBayId: 6 | 43;
  gqaViewAngle: GqaViewAngle;
  gqaActiveGroup0Head: number;
  gqaActiveGroup1Head: number;
  gqaHoveredCell: GqaCellRef | null;
  gqaSelectedCell: GqaCellRef | null;
  gqaStepperActive: boolean;
  gqaStepperPlaying: boolean;
  gqaStepperStep: number;
  gqaStepperSubProgress: number;
  moeBayId: 2 | 52;
  moeViewAngle: GqaViewAngle;
  /** null means follow latest token (T - 1); integer 0..T-1 pins a specific token in MoE cutaway. */
  moeFocusToken: number | null;
  /** null means follow rank-0 Top-6 winner for focusToken; integer 0..127 focuses that expert. */
  moeActiveExpert: number | null;
  moeTokenCycling: boolean;
  moeHoveredCell: MoeCellRef | null;
  moeSelectedCell: MoeCellRef | null;
  moeStepperActive: boolean;
  moeStepperPlaying: boolean;
  moeStepperStep: number;
  moeStepperSubProgress: number;
  stepperAutoFollow: boolean;
  camera: CameraPose;
  savedMacroCamera: CameraPose;
  reducedMotion: boolean;
  tourActive: boolean;
  tourChapter: number;
  memoryChartOpen: boolean;
  memoryProbeTokens: number;
  shortcutsOpen: boolean;
  beat: number | null;
  elapsedMs: number;
  playing: boolean;
}

export const MACRO_DEFAULT_AZIMUTH_DEG = 30;
export const MACRO_DEFAULT_ELEVATION_DEG = 14;
export const DEFAULT_FOV_DEG = 34;
export const MIN_CAMERA_DISTANCE = 3.2;
export const MAX_CAMERA_DISTANCE = 260;
export const MIN_ELEVATION_DEG = -85;
export const MAX_ELEVATION_DEG = 85;
export const CAMERA_FLY_DURATION_MS = 650;

export const DEFAULT_TOKENS = 8;
export const MIN_TOKENS = 1;
export const MAX_TOKENS = 16;

export function cloneCameraPose(pose: CameraPose): CameraPose {
  return {
    target: [pose.target[0], pose.target[1], pose.target[2]],
    azimuthDeg: pose.azimuthDeg,
    elevationDeg: pose.elevationDeg,
    distance: pose.distance,
    fovDeg: pose.fovDeg,
  };
}

export function createPerspectivePose(
  target: Vec3,
  distance: number,
  azimuthDeg = MACRO_DEFAULT_AZIMUTH_DEG,
  elevationDeg = MACRO_DEFAULT_ELEVATION_DEG,
  fovDeg = DEFAULT_FOV_DEG,
): CameraPose {
  return {
    target: [target[0], target[1], target[2]],
    azimuthDeg,
    elevationDeg: Math.max(MIN_ELEVATION_DEG, Math.min(MAX_ELEVATION_DEG, elevationDeg)),
    distance: Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, distance)),
    fovDeg,
  };
}

export function createInitialState(defaultPose: CameraPose, reducedMotion = false): SceneState {
  return {
    mode: 'tower',
    section: 'overview',
    precision: 'bf16',
    tokens: DEFAULT_TOKENS,
    hoveredBay: null,
    selectedBay: null,
    hoveredPart: null,
    selectedPart: null,
    mambaBayId: 1,
    mambaViewAngle: 'three_quarter',
    mambaActiveHead: 0,
    mambaStateToken: null,
    mambaTemporalPlaying: false,
    mambaHoveredCell: null,
    mambaSelectedCell: null,
    mambaStepperActive: false,
    mambaStepperPlaying: false,
    mambaStepperStep: 0,
    mambaStepperSubProgress: 0,
    gqaBayId: 6,
    gqaViewAngle: 'three_quarter',
    gqaActiveGroup0Head: 0,
    gqaActiveGroup1Head: 16,
    gqaHoveredCell: null,
    gqaSelectedCell: null,
    gqaStepperActive: false,
    gqaStepperPlaying: false,
    gqaStepperStep: 0,
    gqaStepperSubProgress: 0,
    moeBayId: 2,
    moeViewAngle: 'three_quarter',
    moeFocusToken: null,
    moeActiveExpert: null,
    moeTokenCycling: false,
    moeHoveredCell: null,
    moeSelectedCell: null,
    moeStepperActive: false,
    moeStepperPlaying: false,
    moeStepperStep: 0,
    moeStepperSubProgress: 0,
    stepperAutoFollow: true,
    camera: cloneCameraPose(defaultPose),
    savedMacroCamera: cloneCameraPose(defaultPose),
    reducedMotion,
    tourActive: true,
    tourChapter: 0,
    memoryChartOpen: false,
    memoryProbeTokens: 1_048_576,
    shortcutsOpen: false,
    beat: null,
    elapsedMs: 0,
    playing: false,
  };
}
