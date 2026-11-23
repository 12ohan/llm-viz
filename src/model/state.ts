import type { PrecisionMode, Vec3 } from './spec';

export interface CameraPose {
  target: Vec3;
  azimuthDeg: number;
  elevationDeg: number;
  distance: number;
  fovDeg: number;
}

export type TocSectionId =
  | 'overview'
  | 'embedding'
  | 'mamba'
  | 'moe'
  | 'gqa'
  | 'kv_fp8'
  | 'output';

export interface SceneState {
  view: 'macro' | 'detail';
  section: TocSectionId;
  precision: PrecisionMode;
  tokens: number;
  hoveredBay: number | null;
  selectedBay: number | null;
  selectedPart: string | null;
  camera: CameraPose;
  savedMacroCamera: CameraPose;
  reducedMotion: boolean;
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
    view: 'macro',
    section: 'overview',
    precision: 'bf16',
    tokens: DEFAULT_TOKENS,
    hoveredBay: null,
    selectedBay: null,
    selectedPart: null,
    camera: cloneCameraPose(defaultPose),
    savedMacroCamera: cloneCameraPose(defaultPose),
    reducedMotion,
    beat: null,
    elapsedMs: 0,
    playing: false,
  };
}
