import type { NemotronSpec, PrecisionMode } from './spec';

export interface MemoryReadout {
  readonly tokens: number;
  readonly precision: PrecisionMode;
  readonly fixedMambaBytes: number;
  readonly kvBytesPerToken: number;
  readonly referenceBytesPerToken: number;
  readonly kvBytes: number;
  readonly referenceBytes: number;
  readonly hybridTotalBytes: number;
}

const BYTES_PER_KIB = 1024;
const BYTES_PER_MIB = 1048576;

/**
 * Compute exact logical tensor sizes in bytes for one sequence at `tokens` cached positions.
 * Accepts any nonnegative integer `tokens` independently of the 1–16 scene scrubber range.
 */
export function memoryFor(
  spec: NemotronSpec,
  tokens: number,
  mode: PrecisionMode,
): MemoryReadout {
  if (!Number.isInteger(tokens) || tokens < 0) {
    throw new RangeError(`tokens must be a nonnegative integer, received ${tokens}`);
  }

  let mambaLayers = 0;
  let attentionLayers = 0;
  for (const bay of spec.bays) {
    if (bay.kind === 'M') mambaLayers++;
    else if (bay.kind === '*') attentionLayers++;
  }

  const totalLayers = spec.bays.length;

  // Fixed per-sequence Mamba-2 state:
  //   SSM state (FP32, 4 B): heads * channels * state * 4
  //   Conv1D state (BF16, 2 B): (heads * channels + 2 * groups * state) * kernel * 2
  const innerDim = spec.mamba.heads * spec.mamba.channels;
  const ssmBytesPerLayer = innerDim * spec.mamba.state * 4;
  const convChannels = innerDim + 2 * spec.mamba.groups * spec.mamba.state;
  const convBytesPerLayer = convChannels * spec.mamba.kernel * 2;
  const fixedMambaBytes = mambaLayers * (ssmBytesPerLayer + convBytesPerLayer);

  // Per-token KV cache payload: L * 2 (K & V) * kvHeads * headDim * bytesPerScalar
  const bytesPerKvScalar = mode === 'fp8_selective' ? 1 : 2;
  const perLayerKvBytesPerToken = 2 * spec.gqa.kv * spec.gqa.headDim * bytesPerKvScalar;
  const kvBytesPerToken = attentionLayers * perLayerKvBytesPerToken;
  const referenceBytesPerToken = totalLayers * perLayerKvBytesPerToken;

  const kvBytes = kvBytesPerToken * tokens;
  const referenceBytes = referenceBytesPerToken * tokens;
  const hybridTotalBytes = fixedMambaBytes + kvBytes;

  return {
    tokens,
    precision: mode,
    fixedMambaBytes,
    kvBytesPerToken,
    referenceBytesPerToken,
    kvBytes,
    referenceBytes,
    hybridTotalBytes,
  };
}

export function formatMiB(bytes: number, decimals = 1): string {
  return `${(bytes / BYTES_PER_MIB).toFixed(decimals)} MiB`;
}

export function formatKiB(bytes: number): string {
  const kib = bytes / BYTES_PER_KIB;
  return Number.isInteger(kib) ? `${kib} KiB` : `${kib.toFixed(1)} KiB`;
}

export interface FormattedMemoryReadout {
  readonly mambaText: string;
  readonly kvText: string;
  readonly referenceText: string;
}

/**
 * Format readout with units attached to values:
 * e.g. "Mamba state 47.1 MiB fixed" · "KV 48 KiB (+6 KiB/token)" · "52-layer GQA reference 416 KiB (+52 KiB/token)"
 */
export function formatMemoryReadout(readout: MemoryReadout): FormattedMemoryReadout {
  return {
    mambaText: `Mamba state ${formatMiB(readout.fixedMambaBytes, 1)} fixed`,
    kvText: `KV ${formatKiB(readout.kvBytes)} (+${formatKiB(readout.kvBytesPerToken)}/token)`,
    referenceText: `52-layer GQA reference ${formatKiB(readout.referenceBytes)} (+${formatKiB(readout.referenceBytesPerToken)}/token)`,
  };
}
