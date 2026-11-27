/**
 * Dedicated 2D `<canvas>` RGB Inset Renderer (`src/uni/app/pixelCanvas.ts`)
 *
 * Enforces Locked Decision D4:
 *   - WebGL2 `SheetsPass` `RG32F` atlases store strictly 2-channel scalar/mask plates.
 *   - All 3-channel RGB image previews (`source`, `letterboxed`, `decoded`, `cropped`,
 *     `grounding`) render via this dedicated 2D `<canvas>` inset with crisp nearest-
 *     neighbor upscaling (`imageSmoothingEnabled = false`).
 *   - Overlays `U18` `paddingLockMask` diagonal amber/slate border-lock hatching and
 *     `U27` `<LOC_0..16>` `groundingBoxes` pixel rectangles deterministically.
 *
 * Pure `renderRgbDescriptorToRgba` executes cleanly in headless Node (Vitest without DOM).
 * Zero `Math.random()`, zero `src/ui/controls.ts`, `src/twotower/`, or `src/diffusion/` imports.
 */

import { UNI_SCENE_VISUAL_HONESTY_BADGE } from '../scene/provenance';
import type { UniRgbPreviewDescriptor } from '../scene/types';
import type {
  UniPixelCanvasMountOptions,
  UniPixelCanvasRgbaBuffer,
  UniRgbPreviewRendererApi,
} from './types';

const BOX_PALETTE: readonly (readonly [number, number, number])[] = Object.freeze([
  Object.freeze([16, 185, 129] as const), // emerald-500
  Object.freeze([225, 29, 72] as const),  // rose-600
  Object.freeze([37, 99, 235] as const),  // blue-600
  Object.freeze([217, 119, 6] as const),  // amber-600
]);

const HATCH_AMBER_R = 245;
const HATCH_AMBER_G = 158;
const HATCH_AMBER_B = 11;

const HATCH_SLATE_R = 51;
const HATCH_SLATE_G = 65;
const HATCH_SLATE_B = 85;

function clampUnitToByte(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v <= 0) return 0;
  if (v >= 1) return 255;
  return Math.round(v * 255);
}

function blendPixel(
  rgba: Uint8ClampedArray,
  offset: number,
  tr: number,
  tg: number,
  tb: number,
  alpha: number,
): void {
  const inv = 1 - alpha;
  rgba[offset] = Math.round(rgba[offset]! * inv + tr * alpha);
  rgba[offset + 1] = Math.round(rgba[offset + 1]! * inv + tg * alpha);
  rgba[offset + 2] = Math.round(rgba[offset + 2]! * inv + tb * alpha);
  rgba[offset + 3] = 255;
}

/**
 * Pure Node-testable RGBA upscaler converting row-major `[3, height, width]` Float32
 * `[0, 1]` pixels into crisp nearest-neighbor RGBA bytes (`Uint8ClampedArray`), with
 * optional `U18` `paddingLockMask` diagonal hatching and `U27` `groundingBoxes` borders.
 */
export function renderRgbDescriptorToRgba(
  descriptor: UniRgbPreviewDescriptor,
  options?: UniPixelCanvasMountOptions,
): UniPixelCanvasRgbaBuffer {
  const srcW = Math.max(1, Math.trunc(descriptor.width));
  const srcH = Math.max(1, Math.trunc(descriptor.height));
  const rawScale = options?.scale ?? 8;
  const scale = Number.isFinite(rawScale) ? Math.max(1, Math.trunc(rawScale)) : 8;
  const outW = srcW * scale;
  const outH = srcH * scale;

  const rgba = new Uint8ClampedArray(outW * outH * 4);
  const planeSize = srcW * srcH;
  const chw = descriptor.rgbChw;

  for (let sy = 0; sy < srcH; sy++) {
    const srcRowOffset = sy * srcW;
    const dstYStart = sy * scale;
    for (let sx = 0; sx < srcW; sx++) {
      const idx = srcRowOffset + sx;
      const r = idx < chw.length ? clampUnitToByte(chw[idx]!) : 0;
      const gIdx = planeSize + idx;
      const g = gIdx < chw.length ? clampUnitToByte(chw[gIdx]!) : 0;
      const bIdx = planeSize * 2 + idx;
      const b = bIdx < chw.length ? clampUnitToByte(chw[bIdx]!) : 0;

      const dstXStart = sx * scale;
      for (let dy = 0; dy < scale; dy++) {
        const rowBase = ((dstYStart + dy) * outW + dstXStart) * 4;
        for (let dx = 0; dx < scale; dx++) {
          const p = rowBase + dx * 4;
          rgba[p] = r;
          rgba[p + 1] = g;
          rgba[p + 2] = b;
          rgba[p + 3] = 255;
        }
      }
    }
  }

  let hatchedPixelCount = 0;
  const showPaddingLock = options?.showPaddingLock !== false;
  const lockInfo = descriptor.paddingLockMask;
  if (showPaddingLock && lockInfo && lockInfo.vqGridRows > 0 && lockInfo.vqGridCols > 0) {
    const gridRows = lockInfo.vqGridRows;
    const gridCols = lockInfo.vqGridCols;
    const hatchedMask = new Uint8Array(outW * outH);

    for (let row = 0; row < gridRows; row++) {
      const y0 = Math.floor((row * outH) / gridRows);
      const y1 = Math.floor(((row + 1) * outH) / gridRows);
      for (let col = 0; col < gridCols; col++) {
        const cellIdx = row * gridCols + col;
        if (cellIdx >= lockInfo.lockMask.length || lockInfo.lockMask[cellIdx] !== 1) {
          continue;
        }
        const x0 = Math.floor((col * outW) / gridCols);
        const x1 = Math.floor(((col + 1) * outW) / gridCols);

        for (let py = y0; py < y1; py++) {
          for (let px = x0; px < x1; px++) {
            const isCellEdge = px === x0 || px === x1 - 1 || py === y0 || py === y1 - 1;
            const diagPhase = (px - x0 + (py - y0)) % 6;
            const isAmberDiag = diagPhase === 0 || diagPhase === 1;
            const isSlateDiag = diagPhase === 2;

            if (!isCellEdge && !isAmberDiag && !isSlateDiag) {
              continue;
            }

            const pixelIndex = py * outW + px;
            const offset = pixelIndex * 4;
            if (isCellEdge || isAmberDiag) {
              blendPixel(rgba, offset, HATCH_AMBER_R, HATCH_AMBER_G, HATCH_AMBER_B, 0.85);
            } else {
              blendPixel(rgba, offset, HATCH_SLATE_R, HATCH_SLATE_G, HATCH_SLATE_B, 0.78);
            }

            if (hatchedMask[pixelIndex] === 0) {
              hatchedMask[pixelIndex] = 1;
              hatchedPixelCount += 1;
            }
          }
        }
      }
    }
  }

  let boxBorderPixelCount = 0;
  const showGroundingBoxes = options?.showGroundingBoxes !== false;
  const boxes = descriptor.groundingBoxes;
  if (showGroundingBoxes && boxes && boxes.length > 0) {
    const borderThickness = scale >= 4 ? 2 : 1;
    const boxMask = new Uint8Array(outW * outH);

    for (let b = 0; b < boxes.length; b++) {
      const box = boxes[b]!;
      const [rawX1, rawY1, rawX2, rawY2] = box.pixelBox;
      const fx1 = Number.isFinite(rawX1) ? rawX1 : 0;
      const fy1 = Number.isFinite(rawY1) ? rawY1 : 0;
      const fx2 = Number.isFinite(rawX2) ? rawX2 : fx1;
      const fy2 = Number.isFinite(rawY2) ? rawY2 : fy1;

      const minX = Math.min(fx1, fx2);
      const maxX = Math.max(fx1, fx2);
      const minY = Math.min(fy1, fy2);
      const maxY = Math.max(fy1, fy2);

      const x0 = Math.max(0, Math.min(outW - 1, Math.round(minX * scale)));
      const y0 = Math.max(0, Math.min(outH - 1, Math.round(minY * scale)));
      const x1 = Math.max(x0, Math.min(outW - 1, Math.round(maxX * scale) - 1));
      const y1 = Math.max(y0, Math.min(outH - 1, Math.round(maxY * scale) - 1));

      const [cr, cg, cb] = BOX_PALETTE[b % BOX_PALETTE.length]!;

      for (let py = y0; py <= y1; py++) {
        const onHorizontalEdge = py - y0 < borderThickness || y1 - py < borderThickness;
        for (let px = x0; px <= x1; px++) {
          const onVerticalEdge = px - x0 < borderThickness || x1 - px < borderThickness;
          if (!onHorizontalEdge && !onVerticalEdge) {
            continue;
          }
          const pixelIndex = py * outW + px;
          const offset = pixelIndex * 4;
          rgba[offset] = cr;
          rgba[offset + 1] = cg;
          rgba[offset + 2] = cb;
          rgba[offset + 3] = 255;
          if (boxMask[pixelIndex] === 0) {
            boxMask[pixelIndex] = 1;
            boxBorderPixelCount += 1;
          }
        }
      }
    }
  }

  return Object.freeze({
    width: outW,
    height: outH,
    rgba,
    hatchedPixelCount,
    boxBorderPixelCount,
  });
}

function paintBufferOntoCanvas(
  canvas: HTMLCanvasElement,
  buffer: UniPixelCanvasRgbaBuffer,
): void {
  canvas.width = buffer.width;
  canvas.height = buffer.height;
  canvas.style.imageRendering = 'pixelated';

  const ctx = typeof canvas.getContext === 'function' ? canvas.getContext('2d') : null;
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;

  if (typeof ImageData === 'function' && typeof ctx.putImageData === 'function') {
    const clamped = new Uint8ClampedArray(buffer.rgba);
    const imgData = new ImageData(clamped, buffer.width, buffer.height);
    ctx.putImageData(imgData, 0, 0);
    return;
  }

  if (typeof ctx.createImageData === 'function' && typeof ctx.putImageData === 'function') {
    const imgData = ctx.createImageData(buffer.width, buffer.height);
    imgData.data.set(buffer.rgba);
    ctx.putImageData(imgData, 0, 0);
  }
}

/**
 * Mount a responsive editorial strip of 2D `<canvas>` RGB preview `<figure>` cards
 * inside `container`, showing crisp upscaled pixels, `U18` border-lock badges,
 * `<LOC_*>` grounding pills, and the prominent `UNI_SCENE_VISUAL_HONESTY_BADGE`.
 */
export function createUniPixelCanvasPanel(container: HTMLElement): UniRgbPreviewRendererApi {
  const strip = document.createElement('div');
  strip.className = 'uni-pixel-strip';
  strip.dataset.testid = 'uni-pixel-strip';
  container.appendChild(strip);

  let disposed = false;

  function clear(): void {
    strip.innerHTML = '';
  }

  function renderDescriptors(
    descriptors: readonly UniRgbPreviewDescriptor[],
    options?: UniPixelCanvasMountOptions,
  ): void {
    if (disposed) return;
    clear();

    for (let i = 0; i < descriptors.length; i++) {
      const desc = descriptors[i]!;
      const buffer = renderRgbDescriptorToRgba(desc, options);

      const figure = document.createElement('figure');
      figure.className = 'uni-pixel-card';
      figure.dataset.testid = `uni-pixel-card-${desc.id}`;

      const header = document.createElement('div');
      header.className = 'uni-pixel-card-header';

      const title = document.createElement('span');
      title.className = 'uni-pixel-card-title';
      title.textContent = desc.label;

      const honesty = document.createElement('span');
      honesty.className = 'uni-pixel-honesty-badge';
      honesty.dataset.testid = `uni-pixel-honesty-${desc.id}`;
      honesty.textContent = desc.honestyBadge || UNI_SCENE_VISUAL_HONESTY_BADGE;

      header.append(title, honesty);

      const canvasWrap = document.createElement('div');
      canvasWrap.className = 'uni-pixel-canvas-wrap';

      const canvas = document.createElement('canvas');
      canvas.className = 'uni-pixel-canvas';
      canvas.dataset.testid = `uni-pixel-canvas-${desc.id}`;
      canvas.setAttribute('aria-label', `${desc.label} (${desc.width}x${desc.height} RGB)`);
      paintBufferOntoCanvas(canvas, buffer);
      canvasWrap.appendChild(canvas);

      const figcaption = document.createElement('figcaption');
      figcaption.className = 'uni-pixel-caption';

      const captionText = document.createElement('div');
      captionText.className = 'uni-pixel-caption-text';
      captionText.textContent = desc.caption;
      figcaption.appendChild(captionText);

      const pillsRow = document.createElement('div');
      pillsRow.className = 'uni-pixel-pills';

      if (desc.paddingLockMask && options?.showPaddingLock !== false) {
        const lockTotal = desc.paddingLockMask.vqGridRows * desc.paddingLockMask.vqGridCols;
        const lockPill = document.createElement('span');
        lockPill.className = 'uni-pixel-pill uni-pixel-pill--lock';
        lockPill.dataset.testid = `uni-pixel-lock-pill-${desc.id}`;
        lockPill.textContent = `U18 Lock ${desc.paddingLockMask.lockedCellCount}/${lockTotal}`;
        pillsRow.appendChild(lockPill);
      }

      if (desc.groundingBoxes && desc.groundingBoxes.length > 0 && options?.showGroundingBoxes !== false) {
        for (let b = 0; b < desc.groundingBoxes.length; b++) {
          const box = desc.groundingBoxes[b]!;
          const boxPill = document.createElement('span');
          boxPill.className = 'uni-pixel-pill uni-pixel-pill--loc';
          boxPill.dataset.testid = `uni-pixel-loc-pill-${desc.id}-${b}`;
          const [x1, y1, x2, y2] = box.pixelBox;
          const locRange = box.locTokenStrings.join('');
          boxPill.textContent = `${box.tag}: ${locRange} [${x1.toFixed(0)},${y1.toFixed(0)},${x2.toFixed(0)},${y2.toFixed(0)}]`;
          pillsRow.appendChild(boxPill);
        }
      }

      if (pillsRow.childNodes.length > 0) {
        figcaption.appendChild(pillsRow);
      }

      figure.append(header, canvasWrap, figcaption);
      strip.appendChild(figure);
    }
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    clear();
    if (strip.parentNode === container) {
      container.removeChild(strip);
    }
  }

  return {
    renderDescriptors,
    clear,
    dispose,
  };
}
