/**
 * Nemotron-Labs-Diffusion-14B Browser Entry (`src/diffusion/app/main.ts`)
 *
 * Clean, zero-dependency MPA entry isolated from sibling visualizers.
 * Mounts the editorial WebGL2 spatial application onto `#diffusion-app`.
 */

import './spatial.css';
import { mountDiffusionSpatialApp } from './spatialApp';

export { mountDiffusionSpatialApp };

if (typeof document !== 'undefined') {
  const existingRoot = document.getElementById('diffusion-app');
  if (existingRoot) {
    mountDiffusionSpatialApp(existingRoot);
  } else if (document.body) {
    const fallbackRoot = document.createElement('div');
    fallbackRoot.id = 'diffusion-app';
    fallbackRoot.dataset.testid = 'diffusion-app';
    document.body.append(fallbackRoot);
    mountDiffusionSpatialApp(fallbackRoot);
  }
}
