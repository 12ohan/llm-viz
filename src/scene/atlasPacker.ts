/**
 * Shared RG32F WebGL2 Value Atlas packing primitives (Stage A+1 · Increment 3).
 * Used by both GQA and Mamba-2 live sheet cutaways.
 */

export type TexRect = readonly [number, number, number, number];

export interface AtlasSlice {
  readonly rect: TexRect;
  /** Normalization scale `1 / maxAbs` so shader maps `raw * scale` into [-1, +1] (or [0, 1] for mode=1). */
  readonly scale: number;
  /** 0 = diverging signed palette, 1 = probability/positive [0..1] palette */
  readonly mode: 0 | 1;
}

export interface Rg32fValueAtlas {
  readonly width: number;
  readonly height: number;
  /** Packed RG32F Float32Array (length = width * height * 2): R = value, G = 1 if masked else 0 */
  readonly data: Float32Array;
}

export function computeMaxAbs(arr: Float32Array, minFloor = 0.25): number {
  let m = minFloor;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
  }
  return m;
}

export interface Rg32fAtlasWriter {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  setTexel: (x: number, y: number, val: number, forceMasked?: boolean) => void;
  makeRect: (x0: number, y0: number, cols: number, rows: number) => TexRect;
}

export function createRg32fAtlas(width: number, height: number): Rg32fAtlasWriter {
  const data = new Float32Array(width * height * 2);

  const setTexel = (x: number, y: number, val: number, forceMasked = false): void => {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const idx = (y * width + x) * 2;
    if (!forceMasked && Number.isFinite(val)) {
      data[idx] = val;
      data[idx + 1] = 0.0;
    } else {
      data[idx] = 0.0;
      data[idx + 1] = 1.0;
    }
  };

  const makeRect = (x0: number, y0: number, cols: number, rows: number): TexRect => [
    x0 / width,
    y0 / height,
    cols / width,
    rows / height,
  ];

  return { width, height, data, setTexel, makeRect };
}
