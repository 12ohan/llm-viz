import { describe, expect, it } from 'vitest';
import { cross3, dot3, normalize3 } from '../src/render/camera';
import { pickMacroBay, type PickBox } from '../src/render/pick';
import {
  FLOW_RIBBON_VERTEX_FLOATS,
  FlowRibbonBuilder,
} from '../src/render/ribbons';
import {
  FRAME_XY_AXES,
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
  intersectRaySheetFrame,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  worldToLocalOnFrame,
} from '../src/render/sheetFrame';
import { createRg32fAtlas } from '../src/scene/atlasPacker';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  drawCellHighlight3D,
  drawOperandDependency3D,
  RIBBON_ACTIVE,
  spanFromSheetGrid,
  type SheetGeomRecord,
} from '../src/scene/sheetInspector';

describe('Spatial Increment 1 Checkpoint: 3D Sheet Primitives, OBB Picking & Cross-Plane Ribbons', () => {
  it('enforces right-handed orthonormal basis (det = +1) and exact local<->world roundtrip across XZ, XY, YZ, and tilted frames', () => {
    const frames = [
      makeSheetFrame3D([1.5, 0.2, -2.0], [3.2, 0.06, 1.6], FRAME_XZ_AXES.axisU, FRAME_XZ_AXES.axisN),
      makeSheetFrame3D([-2.4, 1.8, 0.5], [2.4, 0.06, 1.4], FRAME_XY_AXES.axisU, FRAME_XY_AXES.axisN),
      makeSheetFrame3D([0.8, 1.1, 2.2], [1.2, 0.05, 1.8], FRAME_YZ_AXES.axisU, FRAME_YZ_AXES.axisN),
      // Tilted 35-deg frame with non-orthogonal raw input normal (must be Gram-Schmidt orthonormalized)
      makeSheetFrame3D([0, 2.0, 0], [2.0, 0.07, 1.0], [0.8, 0.3, 0.5], [0.1, 0.9, 0.4]),
    ];

    for (const f of frames) {
      expect(Math.hypot(...f.axisU)).toBeCloseTo(1, 6);
      expect(Math.hypot(...f.axisN)).toBeCloseTo(1, 6);
      expect(Math.hypot(...f.axisV)).toBeCloseTo(1, 6);
      expect(Math.abs(dot3(f.axisU, f.axisN))).toBeLessThan(1e-6);
      expect(Math.abs(dot3(f.axisU, f.axisV))).toBeLessThan(1e-6);
      expect(Math.abs(dot3(f.axisN, f.axisV))).toBeLessThan(1e-6);
      // Right-handed: dot(U x N, V) === +1 preserves CCW mesh winding
      const detSign = dot3(cross3(f.axisU, f.axisN), f.axisV);
      expect(detSign).toBeCloseTo(1, 6);

      const sampleLocals: readonly [number, number, number][] = [
        [-0.5, 0.5, -0.5],
        [0.37, -0.5, 0.21],
        [0, 0, 0],
        [0.5, 0.5, 0.5],
      ];
      for (const [lx, ly, lz] of sampleLocals) {
        const w = localToWorldOnFrame(f, lx, ly, lz);
        const back = worldToLocalOnFrame(f, w);
        expect(back[0]).toBeCloseTo(lx, 5);
        expect(back[1]).toBeCloseTo(ly, 5);
        expect(back[2]).toBeCloseTo(lz, 5);
      }
    }
  });

  it('validates minimal 3-sheet spatial fixture: front/back RG32F consistency, exact nonuniform-extent OBB picking, nearest-hit metric, oriented highlights, vertical ribbon, and cross-plane threads', () => {
    // 3 differently oriented numerical tensor sheets with distinct non-uniform extents & distinct cell values
    const rowsXZ = 4;
    const colsXZ = 6;
    const dataXZ = new Float32Array(rowsXZ * colsXZ);
    for (let r = 0; r < rowsXZ; r++) {
      for (let c = 0; c < colsXZ; c++) dataXZ[r * colsXZ + c] = (r + 1) * 10 + (c + 1);
    }

    const rowsXY = 5;
    const colsXY = 4;
    const dataXY = new Float32Array(rowsXY * colsXY);
    for (let r = 0; r < rowsXY; r++) {
      for (let c = 0; c < colsXY; c++) dataXY[r * colsXY + c] = -((r + 1) * 10 + (c + 1));
    }

    const rowsYZ = 3;
    const colsYZ = 5;
    const dataYZ = new Float32Array(rowsYZ * colsYZ);
    for (let r = 0; r < rowsYZ; r++) {
      for (let c = 0; c < colsYZ; c++) dataYZ[r * colsYZ + c] = 0.25 * (r * colsYZ + c + 1);
    }

    // Pack RG32F atlas for all 3 sheets and verify exact texel storage + front/back UV lookup parity
    const atlas = createRg32fAtlas(16, 16);
    for (let r = 0; r < rowsXZ; r++) {
      for (let c = 0; c < colsXZ; c++) atlas.setTexel(c, r, dataXZ[r * colsXZ + c]);
    }
    const rectXZ = atlas.makeRect(0, 0, colsXZ, rowsXZ);
    for (let r = 0; r < rowsXY; r++) {
      for (let c = 0; c < colsXY; c++) atlas.setTexel(8 + c, r, dataXY[r * colsXY + c]);
    }
    const rectXY = atlas.makeRect(8, 0, colsXY, rowsXY);
    for (let r = 0; r < rowsYZ; r++) {
      for (let c = 0; c < colsYZ; c++) atlas.setTexel(c, 8 + r, dataYZ[r * colsYZ + c]);
    }
    const rectYZ = atlas.makeRect(0, 8, colsYZ, rowsYZ);
    expect(rectXZ[2]).toBeCloseTo(colsXZ / 16, 6);
    expect(rectXY[3]).toBeCloseTo(rowsXY / 16, 6);
    expect(rectYZ[0]).toBeCloseTo(0, 6);

    const sampleAtlasCell = (rect: readonly [number, number, number, number], cols: number, rows: number, u: number, vRow: number) => {
      const col = Math.min(cols - 1, Math.max(0, Math.floor(u * cols)));
      const row = Math.min(rows - 1, Math.max(0, Math.floor(vRow * rows)));
      const su = rect[0] + ((col + 0.5) / cols) * rect[2];
      const sv = rect[1] + ((row + 0.5) / rows) * rect[3];
      const tx = Math.floor(su * atlas.width);
      const ty = Math.floor(sv * atlas.height);
      return atlas.data[(ty * atlas.width + tx) * 2];
    };

    const fXZ = makeSheetFrame3D([0, 0, 0], [3.6, 0.06, 1.8], FRAME_XZ_AXES.axisU, FRAME_XZ_AXES.axisN);
    const fXY = makeSheetFrame3D([-2.6, 1.6, -1.8], [2.2, 0.06, 1.5], FRAME_XY_AXES.axisU, FRAME_XY_AXES.axisN);
    const fYZ = makeSheetFrame3D([2.4, 1.2, 0.6], [1.5, 0.05, 1.2], FRAME_YZ_AXES.axisU, FRAME_YZ_AXES.axisN);

    const boxes: PickBox[] = [
      {
        bayId: 6,
        partId: 'sheet-xz',
        tensorKey: 'scores',
        gridCols: colsXZ,
        gridRows: rowsXZ,
        ...sheetFrameWorldAabb(fXZ),
        frame: fXZ,
      },
      {
        bayId: 6,
        partId: 'sheet-xy',
        tensorKey: 'w_q',
        gridCols: colsXY,
        gridRows: rowsXY,
        ...sheetFrameWorldAabb(fXY),
        frame: fXY,
      },
      {
        bayId: 6,
        partId: 'sheet-yz',
        tensorKey: 'q',
        gridCols: colsYZ,
        gridRows: rowsYZ,
        ...sheetFrameWorldAabb(fYZ),
        frame: fYZ,
      },
    ];

    // 1. Front-face AND back-face rays to the exact same cell (r, c) on every oriented sheet
    //    must pick the exact same (hitRow, hitCol) and therefore identical numerical tensor entry.
    const testCases = [
      { frame: fXZ, partId: 'sheet-xz', rect: rectXZ, rows: rowsXZ, cols: colsXZ, data: dataXZ, r: 2, c: 4 },
      { frame: fXY, partId: 'sheet-xy', rect: rectXY, rows: rowsXY, cols: colsXY, data: dataXY, r: 3, c: 1 },
      { frame: fYZ, partId: 'sheet-yz', rect: rectYZ, rows: rowsYZ, cols: colsYZ, data: dataYZ, r: 1, c: 3 },
    ];

    for (const tc of testCases) {
      const lx = -0.5 + (tc.c + 0.5) / tc.cols;
      const lz = -0.5 + (tc.r + 0.5) / tc.rows;
      const cellCenter = localToWorldOnFrame(tc.frame, lx, 0, lz);

      // Front ray approaching along -axisN from distance 4.0
      const frontOrigin: [number, number, number] = [
        cellCenter[0] + tc.frame.axisN[0] * 4.0,
        cellCenter[1] + tc.frame.axisN[1] * 4.0,
        cellCenter[2] + tc.frame.axisN[2] * 4.0,
      ];
      const frontDir: [number, number, number] = [
        -tc.frame.axisN[0],
        -tc.frame.axisN[1],
        -tc.frame.axisN[2],
      ];
      const frontHit = pickMacroBay({ origin: frontOrigin, direction: frontDir }, boxes, []);
      expect(frontHit).not.toBeNull();
      expect(frontHit!.partId).toBe(tc.partId);
      expect(frontHit!.hitRow).toBe(tc.r);
      expect(frontHit!.hitCol).toBe(tc.c);
      // Exact metric distance = 4.0 - 0.5 * ey
      expect(frontHit!.t).toBeCloseTo(4.0 - 0.5 * tc.frame.extent[1], 4);

      // Back ray approaching along +axisN from distance 4.0
      const backOrigin: [number, number, number] = [
        cellCenter[0] - tc.frame.axisN[0] * 4.0,
        cellCenter[1] - tc.frame.axisN[1] * 4.0,
        cellCenter[2] - tc.frame.axisN[2] * 4.0,
      ];
      const backDir: [number, number, number] = [
        tc.frame.axisN[0],
        tc.frame.axisN[1],
        tc.frame.axisN[2],
      ];
      const backHit = pickMacroBay({ origin: backOrigin, direction: backDir }, boxes, []);
      expect(backHit).not.toBeNull();
      expect(backHit!.partId).toBe(tc.partId);
      expect(backHit!.hitRow).toBe(tc.r);
      expect(backHit!.hitCol).toBe(tc.c);
      expect(backHit!.t).toBeCloseTo(4.0 - 0.5 * tc.frame.extent[1], 4);

      // Same numerical tensor value returned & sampled from RG32F atlas from both faces
      const frontFrameHit = intersectRaySheetFrame(frontOrigin, frontDir, tc.frame)!;
      const backFrameHit = intersectRaySheetFrame(backOrigin, backDir, tc.frame)!;
      expect(frontFrameHit.faceSign).toBe(1);
      expect(backFrameHit.faceSign).toBe(-1);
      const frontAtlasVal = sampleAtlasCell(tc.rect, tc.cols, tc.rows, frontFrameHit.u, frontFrameHit.vRow);
      const backAtlasVal = sampleAtlasCell(tc.rect, tc.cols, tc.rows, backFrameHit.u, backFrameHit.vRow);
      const expectedVal = tc.data[tc.r * tc.cols + tc.c];
      expect(frontAtlasVal).toBeCloseTo(expectedVal, 5);
      expect(backAtlasVal).toBeCloseTo(expectedVal, 5);
    }

    // 2. Non-renormalized ray parameter t gives exact front-most hit ordering between stacked sheets
    //    with different nonuniform extents and oblique ray angles!
    const nearCard = makeSheetFrame3D([0, 2.0, 2.0], [1.2, 0.05, 0.8], FRAME_XY_AXES.axisU, FRAME_XY_AXES.axisN);
    const farCard = makeSheetFrame3D([0, 2.0, 1.0], [4.0, 0.05, 3.0], FRAME_XY_AXES.axisU, FRAME_XY_AXES.axisN);
    const stackedBoxes: PickBox[] = [
      { bayId: 1, partId: 'far', gridCols: 4, gridRows: 4, ...sheetFrameWorldAabb(farCard), frame: farCard },
      { bayId: 2, partId: 'near', gridCols: 4, gridRows: 4, ...sheetFrameWorldAabb(nearCard), frame: nearCard },
    ];
    const obliqueDir = normalize3([-0.12, -0.08, -1.0]);
    const stackedHit = pickMacroBay({ origin: [0.3, 2.2, 5.0], direction: obliqueDir }, stackedBoxes, []);
    expect(stackedHit?.partId).toBe('near');

    // Grazing edge hit clamps cleanly to boundary cell without NaN or OOB index
    const edgePt = localToWorldOnFrame(fXY, 0.5, 0, -0.5);
    const edgeHit = intersectRaySheetFrame(
      [edgePt[0], edgePt[1], edgePt[2] + 2.0],
      [0, 0, -1],
      fXY,
    );
    expect(edgeHit).not.toBeNull();
    expect(edgeHit!.u).toBe(1);
    expect(edgeHit!.vRow).toBe(0);

    // 3. Oriented highlights, vertical 3D ribbon, and cross-plane dependency threads produce finite non-degenerate geometry
    const flow = new FlowRibbonBuilder();
    const geomXZ: SheetGeomRecord = {
      cx: fXZ.center[0], cy: fXZ.center[1], cz: fXZ.center[2],
      ex: fXZ.extent[0], ey: fXZ.extent[1], ez: fXZ.extent[2],
      gridCols: colsXZ, gridRows: rowsXZ, frame: fXZ,
    };
    const geomXY: SheetGeomRecord = {
      cx: fXY.center[0], cy: fXY.center[1], cz: fXY.center[2],
      ex: fXY.extent[0], ey: fXY.extent[1], ez: fXY.extent[2],
      gridCols: colsXY, gridRows: rowsXY, frame: fXY,
    };
    const geomYZ: SheetGeomRecord = {
      cx: fYZ.center[0], cy: fYZ.center[1], cz: fYZ.center[2],
      ex: fYZ.extent[0], ey: fYZ.extent[1], ez: fYZ.extent[2],
      gridCols: colsYZ, gridRows: rowsYZ, frame: fYZ,
    };

    const spanTarget = spanFromSheetGrid(geomXZ, 2, 2, 4, 4);
    const spanUpA = spanFromSheetGrid(geomXY, 3, 3, 1, 1);
    const spanUpB = spanFromSheetGrid(geomYZ, 1, 1, 3, 3);

    drawCellHighlight3D(flow, spanTarget, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.022, true);
    drawOperandDependency3D(flow, spanUpA, spanTarget, true);
    drawOperandDependency3D(flow, spanUpB, spanTarget, false);

    // Pure vertical 3D ribbon (+Y climb) must not degenerate when tangent is parallel to upHint=[0,1,0]
    flow.addPolyline3D([[0, 0, 0], [0, 3.0, 0]], [0, 1, 0], 0.14, RIBBON_ACTIVE, true, 0.22, 0.22, true);

    const packed = flow.pack();
    expect(packed.vertices.length).toBeGreaterThan(0);
    expect(packed.vertices.length % FLOW_RIBBON_VERTEX_FLOATS).toBe(0);
    expect(packed.indices.length).toBeGreaterThan(0);
    expect(packed.indices.length % 3).toBe(0);
    for (let i = 0; i < packed.vertices.length; i++) {
      expect(Number.isFinite(packed.vertices[i])).toBe(true);
    }
    const vCount = packed.vertices.length / FLOW_RIBBON_VERTEX_FLOATS;
    for (let i = 0; i < packed.indices.length; i++) {
      expect(packed.indices[i]).toBeLessThan(vCount);
    }
  });
});
