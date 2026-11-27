/**
 * Pure layer-agnostic formula formatting helpers and readout/stepper contracts
 * shared across mathematical simulators (`src/math/*`) and scene/UI inspectors.
 *
 * Intentionally has ZERO imports from `src/scene/` or `src/render/` so the math
 * layer remains strictly headless and free of rendering dependencies.
 */

export { fmtSigned } from '../shared/format';

/** Layer-agnostic readout contract consumed by the docked Formula Inspector bar. */
export interface LayerFormulaReadout {
  readonly tensorLabel: string;
  readonly coordLabel: string;
  readonly headBadge: string | null;
  readonly value: number;
  readonly isMasked: boolean;
  readonly valueText: string;
  readonly equation: string;
  readonly subEquation: string;
}

/** Layer-agnostic stepper stage definition for the 6-step Math Stepper bar. */
export interface StepperStepDef {
  readonly index: number;
  readonly shortLabel: string;
  readonly title: string;
  readonly summary: string;
}
