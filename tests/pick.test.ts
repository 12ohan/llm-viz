import { describe, expect, it } from 'vitest';
import { NEMOTRON_SPEC } from '../src/model/spec';
import {
  buildCameraSnapshot,
  fitPerspectiveToBounds,
  makeViewportMetrics,
  projectWorldToClientCss,
  zoomPerspectiveDistance,
} from '../src/render/camera';
import { clientPointToRay, pickMacroBay, pointToRayDistance } from '../src/render/pick';
import {
  buildMacroSpine,
  mambaRegisterFrontWorld,
  TOWER_FIT_BOUNDS,
} from '../src/scene/spine';

describe('Fixture 5: Projection round trip and perspective ray picking', () => {
  it('round-trips bay 42 register front across DPR 1, DPR 2, and 2x perspective zoom', () => {
    const probe = mambaRegisterFrontWorld(42);
    const scene = buildMacroSpine({
      spec: NEMOTRON_SPEC,
      tokens: 8,
      precision: 'bf16',
      selectedBay: null,
      hoveredBay: null,
    });

    // Host 768x600 with nonzero client origin (37, 53)
    const metricsDpr1 = makeViewportMetrics(37, 53, 768, 600, 1);
    const defaultPose = fitPerspectiveToBounds(
      TOWER_FIT_BOUNDS,
      30,
      14,
      metricsDpr1.innerWidth,
      metricsDpr1.innerHeight,
    );
    const snapDpr1 = buildCameraSnapshot(defaultPose, TOWER_FIT_BOUNDS, metricsDpr1);

    // 1. Project probe into client CSS coordinates & cast invViewProj ray back at DPR 1
    const clientPt1 = projectWorldToClientCss(probe, snapDpr1);
    expect(clientPt1.visible).toBe(true);
    const ray1 = clientPointToRay(clientPt1.x, clientPt1.y, snapDpr1);
    const hit1 = pickMacroBay(ray1, scene.solidPickBoxes, scene.fasciaPickBoxes);

    expect(hit1).not.toBeNull();
    expect(hit1!.bayId).toBe(42);
    expect(hit1!.isFascia).toBe(false);
    expect(pointToRayDistance(probe, ray1)).toBeLessThan(1e-4);

    // 2. Unchanged hit at DPR 2
    const metricsDpr2 = makeViewportMetrics(37, 53, 768, 600, 2);
    const snapDpr2 = buildCameraSnapshot(defaultPose, TOWER_FIT_BOUNDS, metricsDpr2);
    const clientPt2 = projectWorldToClientCss(probe, snapDpr2);
    expect(clientPt2.x).toBeCloseTo(clientPt1.x, 6);
    expect(clientPt2.y).toBeCloseTo(clientPt1.y, 6);

    const ray2 = clientPointToRay(clientPt2.x, clientPt2.y, snapDpr2);
    const hit2 = pickMacroBay(ray2, scene.solidPickBoxes, scene.fasciaPickBoxes);
    expect(hit2).not.toBeNull();
    expect(hit2!.bayId).toBe(42);
    expect(pointToRayDistance(probe, ray2)).toBeLessThan(1e-4);

    // 3. Repeat after 2x perspective zoom (half camera distance)
    const zoomedPose = zoomPerspectiveDistance(defaultPose, 0.5);
    expect(zoomedPose.distance).toBeCloseTo(defaultPose.distance * 0.5, 6);

    const zoomedSnap = buildCameraSnapshot(zoomedPose, TOWER_FIT_BOUNDS, metricsDpr1);
    const zoomedClientPt = projectWorldToClientCss(probe, zoomedSnap);
    expect(zoomedClientPt.visible).toBe(true);
    const zoomedRay = clientPointToRay(zoomedClientPt.x, zoomedClientPt.y, zoomedSnap);
    const zoomedHit = pickMacroBay(zoomedRay, scene.solidPickBoxes, scene.fasciaPickBoxes);

    expect(zoomedHit).not.toBeNull();
    expect(zoomedHit!.bayId).toBe(42);
    expect(pointToRayDistance(probe, zoomedRay)).toBeLessThan(1e-4);

    // 4. Project a second screen point (bay 01 register front) and verify perspective rays diverge
    const secondProbe = mambaRegisterFrontWorld(1);
    const secondClientPt = projectWorldToClientCss(secondProbe, snapDpr1);
    expect(secondClientPt.visible).toBe(true);
    const secondRay = clientPointToRay(secondClientPt.x, secondClientPt.y, snapDpr1);
    const secondHit = pickMacroBay(secondRay, scene.solidPickBoxes, scene.fasciaPickBoxes);

    expect(secondHit).not.toBeNull();
    expect(secondHit!.bayId).toBe(1);
    expect(pointToRayDistance(secondProbe, secondRay)).toBeLessThan(1e-4);

    // Perspective rays diverge with distinct directions from the camera frustum
    const dirDiff = Math.hypot(
      ray1.direction[0] - secondRay.direction[0],
      ray1.direction[1] - secondRay.direction[1],
      ray1.direction[2] - secondRay.direction[2],
    );
    expect(dirDiff).toBeGreaterThan(0.15);
  });
});
