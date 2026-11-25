/**
 * Nemotron-Labs-TwoTower Topology A Browser Entry (`src/twotower/app/main.ts`)
 *
 * Clean, zero-dependency MPA entry root isolated from Nano v1.0.0 (`index.html` / `src/main.ts`).
 * Mounts the Stage TT2 Minimal Functional Diagnostic Session Workbench.
 */

import './diagnostic.css';
import { mountTwoTowerDiagnosticWorkbench } from './diagnosticUi';

export { mountTwoTowerDiagnosticWorkbench };

if (typeof document !== 'undefined') {
  const existingRoot = document.getElementById('twotower-app');
  if (existingRoot) {
    mountTwoTowerDiagnosticWorkbench(existingRoot);
  } else if (document.body) {
    const fallbackRoot = document.createElement('div');
    fallbackRoot.id = 'twotower-app';
    fallbackRoot.dataset.testid = 'twotower-app';
    document.body.append(fallbackRoot);
    mountTwoTowerDiagnosticWorkbench(fallbackRoot);
  }
}
