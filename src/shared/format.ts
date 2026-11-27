/**
 * Pure Domain-Agnostic Numeric & IEC Byte Formatting Helpers (`src/shared/format.ts`)
 *
 * Strictly zero dependencies — shared cleanly across all MPA visualizers without
 * pulling any model spec or memory calculator into `shared(allN)`.
 */

const BYTES_PER_KIB = 1024;
const BYTES_PER_MIB = 1048576;
const BYTES_PER_GIB = 1024 * 1024 * 1024;

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

export function formatMiB(bytes: number, decimals = 1): string {
  return `${(bytes / BYTES_PER_MIB).toFixed(decimals)} MiB`;
}

export function formatGiB(bytes: number, decimals = 2): string {
  return `${(bytes / BYTES_PER_GIB).toFixed(decimals)} GiB`;
}

export function formatKiB(bytes: number): string {
  const kib = bytes / BYTES_PER_KIB;
  return Number.isInteger(kib) ? `${kib} KiB` : `${kib.toFixed(1)} KiB`;
}

/** Format any byte quantity adaptively as B, KiB, MiB, or GiB. */
export function formatBytesAdaptive(bytes: number): string {
  if (bytes >= BYTES_PER_GIB) {
    const gib = bytes / BYTES_PER_GIB;
    return `${gib >= 10 ? gib.toFixed(1) : gib.toFixed(2)} GiB`;
  }
  if (bytes >= BYTES_PER_MIB) {
    const mib = bytes / BYTES_PER_MIB;
    return `${mib >= 100 ? mib.toFixed(0) : mib.toFixed(1)} MiB`;
  }
  if (bytes >= BYTES_PER_KIB) {
    return formatKiB(bytes);
  }
  return `${bytes} B`;
}

export function formatTokenCount(tokens: number): string {
  if (tokens === 1_048_576) return '1M (1,048,576)';
  if (tokens >= 1024 && tokens % 1024 === 0) {
    return `${tokens / 1024}K (${tokens.toLocaleString('en-US')})`;
  }
  return tokens.toLocaleString('en-US');
}
