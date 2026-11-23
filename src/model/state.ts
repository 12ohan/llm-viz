import type { PrecisionMode, Vec3 } from './spec';

export interface CameraPose {
  target: Vec3;
  azimuthDeg: number;
  elevationDeg: number;
  zoom: number;
}

export interface SceneState {
  view: 'macro' | 'detail';
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

export const MACRO_DEFAULT_AZIMUTH_DEG = 24;
export const MACRO_DEFAULT_ELEVATION_DEG = 27;
export const DETAIL_DEFAULT_AZIMUTH_DEG = 22;
export const DETAIL_DEFAULT_ELEVATION_DEG = 28;
export const DEFAULT_TOKENS = 8;
export const MIN_TOKENS = 1;
export const MAX_TOKENS = 16;
export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;

export function cloneCameraPose(pose: CameraPose): CameraPose {
  return {
    target: [pose.target[0], pose.target[1], pose.target[2]],
    azimuthDeg: pose.azimuthDeg,
    elevationDeg: pose.elevationDeg,
    zoom: pose.zoom,
  };
}

export function createMacroCameraPose(target: Vec3): CameraPose {
  return {
    target: [target[0], target[1], target[2]],
    azimuthDeg: MACRO_DEFAULT_AZIMUTH_DEG,
    elevationDeg: MACRO_DEFAULT_ELEVATION_DEG,
    zoom: 1,
  };
}

export function createInitialState(defaultTarget: Vec3, reducedMotion = false): SceneState {
  const macroPose = createMacroCameraPose(defaultTarget);
  return {
    view: 'macro',
    precision: 'bf16',
    tokens: DEFAULT_TOKENS,
    hoveredBay: null,
    selectedBay: null,
    selectedPart: null,
    camera: cloneCameraPose(macroPose),
    savedMacroCamera: cloneCameraPose(macroPose),
    reducedMotion,
    beat: null,
    elapsedMs: 0,
    playing: false,
  };
}
