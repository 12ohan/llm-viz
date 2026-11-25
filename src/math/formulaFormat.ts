/**
 * Pure layer-agnostic formula formatting helpers and readout/stepper contracts
 * shared across mathematical simulators (`src/math/*`) and scene/UI inspectors.
 *
 * Intentionally has ZERO imports from `src/scene/` or `src/render/` so the math
 * layer remains strictly headless and free of rendering dependencies.
 */

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

/**
 * Format a numeric value with an explicit sign prefix (`+` for non-negative
 * values including rounded zero, typographic minus `−` for negative values).
 */
export function fmtSigned(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v < 0 ? '−∞' : '+∞';
  const r = Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v;
  const sign = r < 0 ? '−' : '+';
  return `${sign}${Math.abs(r).toFixed(digits)}`;
}
