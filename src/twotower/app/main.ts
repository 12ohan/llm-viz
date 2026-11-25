/**
 * Nemotron-Labs-TwoTower Topology A Browser Entry (`src/twotower/app/main.ts`)
 *
 * Clean, zero-dependency MPA entry root isolated from Nano v1.0.0 (`index.html` / `src/main.ts`).
 * Mounts the Stage TT3 Interactive WebGL2 Spatial Application + synced Developer Workbench drawer.
 */

import './spatial.css';
import './diagnostic.css';
import { mountTwoTowerDiagnosticWorkbench } from './diagnosticUi';
import { mountTwoTowerSpatialApp } from './spatialApp';

export { mountTwoTowerDiagnosticWorkbench, mountTwoTowerSpatialApp };

if (typeof document !== 'undefined') {
  const existingRoot = document.getElementById('twotower-app');
  if (existingRoot) {
    mountTwoTowerSpatialApp(existingRoot);
  } else if (document.body) {
    const fallbackRoot = document.createElement('div');
    fallbackRoot.id = 'twotower-app';
    fallbackRoot.dataset.testid = 'twotower-app';
    document.body.append(fallbackRoot);
    mountTwoTowerSpatialApp(fallbackRoot);
  }
}
