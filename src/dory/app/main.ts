/**
 * Standalone Isolated Browser Entry for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/app/main.ts`).
 *
 * Mounts `createDorySpatialApp` onto `#dory-app` (or `#app`) with the `#F7F8FA` editorial
 * drafting-table stylesheet (`./spatial.css`). Ready for `DORY6` dev-only Rollup/WebGL2
 * dry-run and `DORY7` `/dory/` 5-way MPA wiring.
 *
 * Strictly `D4`-compliant: ZERO imports from Gate 4b/4c Nano modules, sibling apps, or
 * ambient RNG.
 */

import './spatial.css';
import { createDorySpatialApp } from './spatialApp';
import type { DoryAppMountOptions, DorySpatialAppHandle } from './types';

export { createDorySpatialApp };

let mountedApp: DorySpatialAppHandle | null = null;

/**
 * Idempotently mounts the standalone Dory1.0-4B WebGL2 spatial app onto `#dory-app` / `#app`
 * (creating `#dory-app` on `document.body` if absent). Headless-safe when `document` is undefined.
 */
export function mountDorySpatialApp(
  options?: Omit<DoryAppMountOptions, 'host'> & { readonly host?: HTMLElement },
): DorySpatialAppHandle | null {
  if (typeof document === 'undefined') {
    return null;
  }
  if (mountedApp) {
    return mountedApp;
  }

  let host: HTMLElement | null =
    options?.host ??
    document.getElementById('dory-app') ??
    document.getElementById('app');

  if (!host) {
    host = document.createElement('div');
    host.id = 'dory-app';
    document.body.appendChild(host);
  }
  if (!host.dataset.testid) {
    host.dataset.testid = 'dory-app';
  }

  const baseUrl = (
    options?.baseUrl ??
    import.meta.env?.BASE_URL ??
    '../'
  ).replace(/\/?$/, '/');

  mountedApp = createDorySpatialApp({
    ...options,
    host,
    baseUrl,
  });

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
        mountDorySpatialApp();
      },
      { once: true },
    );
  } else {
    mountDorySpatialApp();
  }
}
