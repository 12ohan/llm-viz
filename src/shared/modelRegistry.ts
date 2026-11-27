/**
 * Canonical Multi-Page Application (MPA) Model Switcher Registry
 * (`src/shared/modelRegistry.ts`)
 *
 * Single source of truth for top-bar cross-visualizer navigation across all
 * Nemotron interactive 3D architecture apps. Adding future visualizers (Models
 * #5, #6, …) requires adding one entry here — zero cross-app JS imports and
 * zero forbidden model-internal markers.
 */

export type ModelEntryId = 'nano' | 'twotower' | 'diffusion' | 'uni';

export interface ModelEntryMeta {
  readonly id: ModelEntryId;
  readonly shortLabel: string;
  readonly title: string;
  readonly subtitle: string;
  readonly pathSuffix: string;
  readonly testId: `nav-to-${ModelEntryId}`;
}

export const MODEL_REGISTRY: readonly ModelEntryMeta[] = Object.freeze([
  Object.freeze({
    id: 'nano' as const,
    shortLabel: 'Nano',
    title: 'Nemotron 3 Nano 30B-A3B',
    subtitle: '52L Hybrid · Mamba-2 + MoE + GQA',
    pathSuffix: '',
    testId: 'nav-to-nano' as const,
  }),
  Object.freeze({
    id: 'twotower' as const,
    shortLabel: 'TwoTower',
    title: 'Nemotron-Labs-TwoTower 30B-A3B',
    subtitle: '2×52L Dual-Tower · Context + Denoiser',
    pathSuffix: 'twotower/',
    testId: 'nav-to-twotower' as const,
  }),
  Object.freeze({
    id: 'diffusion' as const,
    shortLabel: 'Diffusion-14B',
    title: 'Nemotron-Labs-Diffusion-14B',
    subtitle: '40L Dense · YaRN RoPE + Spec Verify',
    pathSuffix: 'diffusion/',
    testId: 'nav-to-diffusion' as const,
  }),
  Object.freeze({
    id: 'uni' as const,
    shortLabel: 'NLD-Uni',
    title: 'Nemotron-Labs-Diffusion-Uni',
    subtitle: '34L Elastic-MoT · Multimodal Diffusion',
    pathSuffix: 'uni/',
    testId: 'nav-to-uni' as const,
  }),
]);

/**
 * Normalize Vite's `import.meta.env.BASE_URL` (root `/` or GitHub Pages subpath
 * `/llm-viz/`, or relative fallback `../`) and resolve the target model URL.
 */
export function resolveModelHref(
  baseUrl: string | undefined,
  entry: Pick<ModelEntryMeta, 'pathSuffix'>,
): string {
  const cleanBase = (baseUrl && baseUrl.length > 0 ? baseUrl : '/').replace(/\/?$/, '/');
  return entry.pathSuffix ? `${cleanBase}${entry.pathSuffix}` : cleanBase;
}
