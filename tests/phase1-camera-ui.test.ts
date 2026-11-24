import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  createInitialState,
  createPerspectivePose,
  MAX_CAMERA_DISTANCE,
  MIN_CAMERA_DISTANCE,
} from '../src/model/state';
import {
  aabbCorners,
  buildCameraSnapshot,
  fitOpBoundsCamera,
  makeViewportMetrics,
  projectWorldToInnerCss,
  type AABB,
} from '../src/render/camera';

describe('Phase 1 · Camera Choreography & UI Safeguards', () => {
  describe('stepperAutoFollow initial state', () => {
    it('defaults stepperAutoFollow to true in createInitialState', () => {
      const pose = createPerspectivePose([0, 0, 0], 18);
      const normalState = createInitialState(pose, false);
      const reducedState = createInitialState(pose, true);

      expect(normalState.stepperAutoFollow).toBe(true);
      expect(reducedState.stepperAutoFollow).toBe(true);
    });
  });

  describe('fitOpBoundsCamera', () => {
    const sampleOpCluster: AABB = {
      min: [-2.4, 0.0, -2.2],
      max: [3.1, 2.1, 2.2],
    };

    const fullCutawayBounds: AABB = {
      min: [-8.4, -0.06, -3.78],
      max: [8.9, 3.24, 3.82],
    };

    it('frames all 8 corners inside the unobstructed viewport band across desktop and narrow viewports with ~75–85% fill', () => {
      const viewports = [
        { name: 'desktop-1440x900', w: 1076, h: 858, top: 82, bottom: 132, isNarrow: false },
        { name: 'desktop-1280x800', w: 952, h: 758, top: 76, bottom: 136, isNarrow: false },
        { name: 'desktop-1100x800', w: 772, h: 758, top: 72, bottom: 158, isNarrow: false },
        { name: 'desktop-1024x768', w: 696, h: 726, top: 72, bottom: 158, isNarrow: false },
        { name: 'narrow-mobile', w: 390, h: 326, top: 12, bottom: 12, isNarrow: true },
      ];

      const corners = aabbCorners(sampleOpCluster);

      for (const vp of viewports) {
        const metrics = makeViewportMetrics(0, 0, vp.w, vp.h, 1);
        const pose = fitOpBoundsCamera(sampleOpCluster, metrics, {
          topInsetPx: vp.top,
          bottomInsetPx: vp.bottom,
          isNarrow: vp.isNarrow,
          fillFraction: 0.80,
        });

        expect(Number.isFinite(pose.distance)).toBe(true);
        expect(pose.distance).toBeGreaterThanOrEqual(MIN_CAMERA_DISTANCE);
        expect(pose.distance).toBeLessThanOrEqual(MAX_CAMERA_DISTANCE);
        for (const coord of pose.target) {
          expect(Number.isFinite(coord)).toBe(true);
        }

        const snap = buildCameraSnapshot(pose, fullCutawayBounds, metrics);

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (const c of corners) {
          const pt = projectWorldToInnerCss(c, snap);
          expect(pt.visible).toBe(true);
          expect(pt.x).toBeGreaterThanOrEqual(-0.5);
          expect(pt.x).toBeLessThanOrEqual(vp.w + 0.5);
          expect(pt.y).toBeGreaterThanOrEqual(vp.top - 0.5);
          expect(pt.y).toBeLessThanOrEqual(vp.h - vp.bottom + 0.5);
          if (pt.x < minX) minX = pt.x;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.y > maxY) maxY = pt.y;
        }

        const usableH = vp.h - vp.top - vp.bottom;
        const fillX = (maxX - minX) / vp.w;
        const fillY = (maxY - minY) / usableH;
        const limitingFill = Math.max(fillX, fillY);
        expect(limitingFill).toBeGreaterThan(0.68);
        expect(limitingFill).toBeLessThanOrEqual(0.88);
      }
    });

    it('shifts projection center vertically to stay centered in asymmetric top/bottom inset bands', () => {
      const metrics = makeViewportMetrics(0, 0, 1076, 858, 1);
      const topInset = 76;
      const bottomInset = 188; // taller Math Dock when long formula wraps

      const pose = fitOpBoundsCamera(sampleOpCluster, metrics, {
        topInsetPx: topInset,
        bottomInsetPx: bottomInset,
        azimuthDeg: 20,
        elevationDeg: 23.5,
      });

      const snap = buildCameraSnapshot(pose, fullCutawayBounds, metrics);
      const corners = aabbCorners(sampleOpCluster);
      let minY = Infinity, maxY = -Infinity;
      for (const c of corners) {
        const pt = projectWorldToInnerCss(c, snap);
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
      }

      const expectedBandCenterY = topInset + (metrics.innerHeight - topInset - bottomInset) * 0.5;
      const projectedMidY = 0.5 * (minY + maxY);
      // Should sit near the unobstructed band center (above raw canvas midpoint 429px)
      expect(projectedMidY).toBeLessThan(metrics.innerHeight * 0.5 - 20);
      expect(Math.abs(projectedMidY - expectedBandCenterY)).toBeLessThan(40);
    });

    it('zooms significantly closer for an operation sub-cluster than for the full cutaway', () => {
      const metrics = makeViewportMetrics(0, 0, 1076, 858, 1);
      const fullPose = fitOpBoundsCamera(fullCutawayBounds, metrics, {
        azimuthDeg: 9,
        elevationDeg: 50,
        topInsetPx: 84,
        bottomInsetPx: 120,
      });
      const opPose = fitOpBoundsCamera(sampleOpCluster, metrics, {
        azimuthDeg: 9,
        elevationDeg: 50,
        topInsetPx: 84,
        bottomInsetPx: 120,
      });

      expect(opPose.distance).toBeLessThan(fullPose.distance);
    });

    it('supports near-plan and drafting-table angles without NaN or corner clipping', () => {
      const metrics = makeViewportMetrics(0, 0, 952, 758, 1);
      const angles = [
        { azimuthDeg: 1.5, elevationDeg: 78 },
        { azimuthDeg: 9, elevationDeg: 50 },
        { azimuthDeg: 20, elevationDeg: 23.5 },
      ];

      for (const ang of angles) {
        const pose = fitOpBoundsCamera(sampleOpCluster, metrics, {
          ...ang,
          topInsetPx: 80,
          bottomInsetPx: 132,
        });
        const snap = buildCameraSnapshot(pose, fullCutawayBounds, metrics);
        for (const c of aabbCorners(sampleOpCluster)) {
          const pt = projectWorldToInnerCss(c, snap);
          expect(pt.visible).toBe(true);
          expect(pt.x).toBeGreaterThanOrEqual(-0.5);
          expect(pt.x).toBeLessThanOrEqual(metrics.innerWidth + 0.5);
          expect(pt.y).toBeGreaterThanOrEqual(80 - 0.5);
          expect(pt.y).toBeLessThanOrEqual(metrics.innerHeight - 132 + 0.5);
        }
      }
    });

    it('never produces NaN or Infinity on degenerate, inverted, or extreme viewport inputs', () => {
      const degenerate: AABB = { min: [2, 1, -3], max: [2, 1, -3] };
      const inverted: AABB = { min: [5, 3, 4], max: [-2, -1, -4] };
      const tinyMetrics = makeViewportMetrics(0, 0, 12, 12, 1);
      const desktopMetrics = makeViewportMetrics(0, 0, 1024, 720, 1);

      const cases = [
        fitOpBoundsCamera(degenerate, desktopMetrics, { topInsetPx: 84, bottomInsetPx: 140 }),
        fitOpBoundsCamera(inverted, desktopMetrics, { topInsetPx: 500, bottomInsetPx: 500 }),
        fitOpBoundsCamera(sampleOpCluster, tinyMetrics, { topInsetPx: 200, bottomInsetPx: 200 }),
        fitOpBoundsCamera(
          { min: [Number.NaN, 0, 0], max: [1, Number.POSITIVE_INFINITY, 1] },
          desktopMetrics,
          { azimuthDeg: Number.NaN, elevationDeg: Number.NaN, fillFraction: Number.NaN },
        ),
      ];

      for (const pose of cases) {
        expect(Number.isFinite(pose.distance)).toBe(true);
        expect(pose.distance).toBeGreaterThanOrEqual(MIN_CAMERA_DISTANCE);
        expect(pose.distance).toBeLessThanOrEqual(MAX_CAMERA_DISTANCE);
        expect(Number.isFinite(pose.azimuthDeg)).toBe(true);
        expect(Number.isFinite(pose.elevationDeg)).toBe(true);
        expect(Number.isFinite(pose.fovDeg)).toBe(true);
        for (const v of pose.target) {
          expect(Number.isFinite(v)).toBe(true);
        }
      }
    });
  });

  describe('Math Dock HTML & CSS safeguards', () => {
    const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8');
    const css = readFileSync(resolve(process.cwd(), 'src/ui/theme.css'), 'utf8');

    it('exposes Auto-Follow toggle, all 6 step pills, and accessible precision disclaimer popover in index.html', () => {
      expect(html).toContain('id="gqa-step-autofollow"');
      expect(html).toContain('id="gqa-formula-note-btn"');
      expect(html).toContain('id="gqa-formula-note"');
      for (let i = 0; i < 6; i++) {
        expect(html).toContain(`data-gqa-step="${i}"`);
      }
    });

    it('wraps formulas cleanly without ellipsis truncation and keeps all step pills visible without horizontal clipping', () => {
      const eqMatch = css.match(/\.gqa-formula-eq\s*\{[^}]*\}/);
      const subMatch = css.match(/\.gqa-formula-sub\s*\{[^}]*\}/);
      const pillsMatch = css.match(/\.gqa-step-pills\s*\{[^}]*\}/);

      expect(eqMatch?.[0]).toContain('white-space: normal');
      expect(eqMatch?.[0]).toContain('overflow-wrap: anywhere');
      expect(eqMatch?.[0]).not.toContain('text-overflow: ellipsis');

      expect(subMatch?.[0]).toContain('white-space: normal');
      expect(subMatch?.[0]).toContain('overflow-wrap: anywhere');
      expect(subMatch?.[0]).not.toContain('text-overflow: ellipsis');

      expect(pillsMatch?.[0]).toContain('flex-wrap: wrap');
      expect(pillsMatch?.[0]).not.toContain('overflow-x: auto');
    });
  });
});
