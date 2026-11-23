import { describe, expect, it } from 'vitest';
import { NEMOTRON_SPEC, type Vec3 } from '../src/model/spec';
import { createMacroCameraPose } from '../src/model/state';
import {
  buildCameraSnapshot,
  fitCameraToBounds,
  makeViewportMetrics,
  projectWorldToClientCss,
} from '../src/render/camera';
import {
  clientPointToRay,
  pickMacroBay,
  pointToRayDistance,
  zoomCameraAtClientPoint,
} from '../src/render/pick';
import { buildMacroSpine, MACRO_FIT_BOUNDS } from '../src/scene/spine';

describe('Fixture 5: Projection round trip and ortho ray picking', () => {
  it('round-trips bay 42 register front at (15.5, 1.55, 0.20) across DPR 1, DPR 2, and 2x cursor zoom', () => {
    const probe: Vec3 = [15.5, 1.55, 0.20];
    const scene = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
    });

    // Host 768x600 with nonzero client origin (37, 53) and standard overlay inset (56 top / 72 bottom)
    const metricsDpr1 = makeViewportMetrics(37, 53, 768, 600, 1);
    const fit = fitCameraToBounds(
      MACRO_FIT_BOUNDS,
      24,
      27,
      metricsDpr1.innerWidth,
      metricsDpr1.innerHeight,
    );
    const defaultPose = createMacroCameraPose(fit.target);
    const snapDpr1 = buildCameraSnapshot(defaultPose, MACRO_FIT_BOUNDS, metricsDpr1);

    // 1. Project probe into client CSS coordinates & cast §4 ray back at DPR 1
    const clientPt1 = projectWorldToClientCss(probe, snapDpr1);
    const ray1 = clientPointToRay(clientPt1.x, clientPt1.y, snapDpr1);
    const hit1 = pickMacroBay(ray1, scene.solidPickBoxes, scene.fasciaPickBoxes);

    expect(hit1).not.toBeNull();
    expect(hit1!.bayId).toBe(42);
    expect(hit1!.isFascia).toBe(false);
    expect(pointToRayDistance(probe, ray1)).toBeLessThan(1e-4);

    // 2. Unchanged hit at DPR 2
    const metricsDpr2 = makeViewportMetrics(37, 53, 768, 600, 2);
    const snapDpr2 = buildCameraSnapshot(defaultPose, MACRO_FIT_BOUNDS, metricsDpr2);
    const clientPt2 = projectWorldToClientCss(probe, snapDpr2);
    expect(clientPt2.x).toBeCloseTo(clientPt1.x, 6);
    expect(clientPt2.y).toBeCloseTo(clientPt1.y, 6);

    const ray2 = clientPointToRay(clientPt2.x, clientPt2.y, snapDpr2);
    const hit2 = pickMacroBay(ray2, scene.solidPickBoxes, scene.fasciaPickBoxes);
    expect(hit2).not.toBeNull();
    expect(hit2!.bayId).toBe(42);
    expect(pointToRayDistance(probe, ray2)).toBeLessThan(1e-4);

    // 3. Repeat after 2x cursor zoom anchored at the probe's client position
    const zoomedPose = zoomCameraAtClientPoint(
      defaultPose,
      MACRO_FIT_BOUNDS,
      metricsDpr1,
      clientPt1.x,
      clientPt1.y,
      2,
    );
    expect(zoomedPose.zoom).toBeCloseTo(2, 6);

    const zoomedSnap = buildCameraSnapshot(zoomedPose, MACRO_FIT_BOUNDS, metricsDpr1);
    const zoomedClientPt = projectWorldToClientCss(probe, zoomedSnap);
    const zoomedRay = clientPointToRay(zoomedClientPt.x, zoomedClientPt.y, zoomedSnap);
    const zoomedHit = pickMacroBay(zoomedRay, scene.solidPickBoxes, scene.fasciaPickBoxes);

    expect(zoomedHit).not.toBeNull();
    expect(zoomedHit!.bayId).toBe(42);
    expect(pointToRayDistance(probe, zoomedRay)).toBeLessThan(1e-4);

    // 4. Project a second screen point and verify orthographic parallel rays (directions agree to 1e-6, origins differ)
    const secondProbe: Vec3 = [-20.5, 3.60, 0.29]; // bay 06 attention tine front
    const secondClientPt = projectWorldToClientCss(secondProbe, snapDpr1);
    const secondRay = clientPointToRay(secondClientPt.x, secondClientPt.y, snapDpr1);

    expect(Math.abs(ray1.direction[0] - secondRay.direction[0])).toBeLessThan(1e-6);
    expect(Math.abs(ray1.direction[1] - secondRay.direction[1])).toBeLessThan(1e-6);
    expect(Math.abs(ray1.direction[2] - secondRay.direction[2])).toBeLessThan(1e-6);

    const originDist = Math.hypot(
      ray1.origin[0] - secondRay.origin[0],
      ray1.origin[1] - secondRay.origin[1],
      ray1.origin[2] - secondRay.origin[2],
    );
    expect(originDist).toBeGreaterThan(1.0);
  });
});
