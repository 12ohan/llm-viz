/**
 * Standalone Boot Entry for Nemotron-Labs-Diffusion-Uni Spatial App (`src/uni/app/main.ts`)
 *
 * Mounts `createUniSpatialApp` onto `#app` with the editorial drafting-table stylesheet
 * (`./spatial.css`). Ready for `uni/index.html` wiring in UNI7.
 *
 * Strictly zero imports from `src/ui/controls.ts` (D4), `src/twotower/`, or `src/diffusion/`.
 */

import './spatial.css';
import { createUniSpatialApp, type UniSpatialAppInstance } from './spatialApp';

let mountedApp: UniSpatialAppInstance | null = null;

export function mountUniSpatialApp(): UniSpatialAppInstance | null {
  if (typeof document === 'undefined') {
    return null;
  }
  if (mountedApp) {
    return mountedApp;
  }

  let root = document.getElementById('app');
  if (!root) {
    root = document.createElement('div');
    root.id = 'app';
    document.body.appendChild(root);
  }

  mountedApp = createUniSpatialApp(root);

  if (typeof window !== 'undefined') {
    window.addEventListener(
      'beforeunload',
      () => {
        mountedApp?.dispose();
        mountedApp = null;
      },
      { once: true },
    );
  }

  return mountedApp;
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener(
      'DOMContentLoaded',
      () => {
        mountUniSpatialApp();
      },
      { once: true },
    );
  } else {
    mountUniSpatialApp();
  }
}
