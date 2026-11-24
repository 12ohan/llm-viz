/**
 * Strict Numerical Parity & Finite-Value Assertion Helpers
 *
 * Explicitly rejects unexpected `NaN`, `+Infinity`, and `-Infinity` (preventing IEEE-754
 * silent passes where `Math.abs(NaN - x) > maxErr` evaluates to `false`), while preserving
 * intentional `-Infinity` masks in causal attention score matrices.
 */

export function assertFiniteNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`Expected finite number at ${label}, got ${String(value)}`);
  }
}

/**
 * Compute exact maximum absolute difference between two arrays of finite numbers.
 * Throws immediately on length mismatch or any unexpected `NaN` / `±Infinity` in either array.
 */
export function computeStrictMaxAbsDiff(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
): number {
  if (actual.length !== expected.length) {
    throw new Error(
      `Length mismatch for ${label}: actual=${actual.length}, expected=${expected.length}`,
    );
  }
  let maxErr = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const b = expected[i];
    if (!Number.isFinite(a)) {
      throw new Error(`Non-finite actual value at ${label}[${i}]: ${String(a)}`);
    }
    if (!Number.isFinite(b)) {
      throw new Error(`Non-finite expected value at ${label}[${i}]: ${String(b)}`);
    }
    const err = Math.abs(a - b);
    if (!Number.isFinite(err)) {
      throw new Error(`Non-finite diff at ${label}[${i}]: |${a} - ${b}| = ${String(err)}`);
    }
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

/** Assert every element is finite and within `tolerance` of `expected`; returns observed max abs error. */
export function assertFiniteArrayParity(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string,
): number {
  assertFiniteNumber(tolerance, `${label}.tolerance`);
  const maxErr = computeStrictMaxAbsDiff(actual, expected, label);
  if (maxErr > tolerance) {
    throw new Error(
      `Parity tolerance exceeded for ${label}: maxAbsDiff=${maxErr} > tolerance=${tolerance}`,
    );
  }
  return maxErr;
}

/** Assert elementwise bit-exact equality for finite Float32 values (rejects NaN and ±Infinity). */
export function assertBitExactFiniteArray(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
): void {
  if (actual.length !== expected.length) {
    throw new Error(
      `Length mismatch for ${label}: actual=${actual.length}, expected=${expected.length}`,
    );
  }
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const b = expected[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new Error(
        `Non-finite value in bit-exact check at ${label}[${i}]: actual=${String(a)}, expected=${String(b)}`,
      );
    }
    if (a !== b) {
      throw new Error(
        `Bit-exact mismatch at ${label}[${i}]: actual=${a}, expected=${b} (diff=${Math.abs(a - b)})`,
      );
    }
  }
}

/**
 * Validate causal attention score matrix where `null` in `expectedNullable` denotes an
 * intentional `-Infinity` causal mask (`k > q`), and all unmasked slots must be strictly finite.
 */
export function assertCausalScoresParity(
  actual: ArrayLike<number>,
  expectedNullable: readonly (number | null)[],
  tolerance: number,
  label: string,
): number {
  assertFiniteNumber(tolerance, `${label}.tolerance`);
  if (actual.length !== expectedNullable.length) {
    throw new Error(
      `Length mismatch for ${label}: actual=${actual.length}, expected=${expectedNullable.length}`,
    );
  }
  let maxErr = 0;
  for (let i = 0; i < expectedNullable.length; i++) {
    const a = actual[i];
    const refVal = expectedNullable[i];
    if (refVal === null) {
      if (a !== -Infinity) {
        throw new Error(
          `Expected intentional causal mask -Infinity at ${label}[${i}], got ${String(a)}`,
        );
      }
      continue;
    }
    if (!Number.isFinite(a)) {
      throw new Error(`Unexpected non-finite unmasked score at ${label}[${i}]: ${String(a)}`);
    }
    if (!Number.isFinite(refVal)) {
      throw new Error(`Unexpected non-finite reference score at ${label}[${i}]: ${String(refVal)}`);
    }
    const err = Math.abs(a - refVal);
    if (err > tolerance) {
      throw new Error(
        `Unmasked causal score parity exceeded at ${label}[${i}]: |${a} - ${refVal}| = ${err} > ${tolerance}`,
      );
    }
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}
