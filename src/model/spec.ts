export type PrecisionMode = 'bf16' | 'fp8_selective';
export type Kind = 'M' | 'E' | '*';
export type Vec3 = readonly [number, number, number];
export type Bracket = readonly [first: number, last: number, motif: string, repeats: number];
export type EffectivePrecision = 'bf16' | 'fp8';
export type StorageDtype = 'fp32' | 'bf16' | 'fp8';

export interface BayDescriptor {
  readonly id: number; // one-based (1..52)
  readonly kind: Kind;
  readonly feeder: boolean; // M immediately followed by *
  readonly sensitive: boolean; // * or feeder
  readonly bracket: 0 | 1 | 2 | 3;
}

export interface NemotronSpec {
  readonly revision: string;
  readonly pattern: string;
  readonly bays: readonly BayDescriptor[];
  readonly brackets: readonly Bracket[];
  readonly hidden: 2688;
  readonly vocabSize: 131072;
  readonly tieWordEmbeddings: false;
  readonly positionalEncoding: 'NoPE';
  readonly mamba: {
    readonly heads: 64;
    readonly channels: 64;
    readonly state: 128;
    readonly groups: 8;
    readonly kernel: 4;
  };
  readonly moe: {
    readonly routed: 128;
    readonly active: 6;
    readonly shared: 1;
    readonly width: 1856;
    readonly sharedWidth: 3712;
  };
  readonly gqa: {
    readonly q: 32;
    readonly kv: 2;
    readonly headDim: 128;
  };
}

export const CHECKPOINT_REVISION = 'bf77c3174f68ad409e1c2aa60daeb46e32d1c606';
export const HYBRID_OVERRIDE_PATTERN = 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME';

export const REPORT_BRACKETS: readonly Bracket[] = [
  [1, 35, 'MEMEM*E', 5],
  [36, 41, 'ME', 3],
  [42, 44, 'M*E', 1],
  [45, 52, 'ME', 4],
] as const;

/** Eight mandatory bay IDs displayed statically in full-spine view. */
export const STATIC_MACRO_BAYS: readonly number[] = [1, 6, 13, 20, 27, 34, 43, 52] as const;

export const PARAMETER_CAPTION =
  '31.6B total · 3.2B active per token excluding embeddings · 3.6B active including embeddings.';

function bracketIndexForBay(id: number, brackets: readonly Bracket[]): 0 | 1 | 2 | 3 {
  for (let i = 0; i < brackets.length; i++) {
    const [first, last] = brackets[i];
    if (id >= first && id <= last) {
      return i as 0 | 1 | 2 | 3;
    }
  }
  throw new Error(`Bay ${id} falls outside declared Figure 2 brackets`);
}

function buildBays(pattern: string, brackets: readonly Bracket[]): readonly BayDescriptor[] {
  const bays: BayDescriptor[] = [];
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch !== 'M' && ch !== 'E' && ch !== '*') {
      throw new Error(`Unexpected pattern character "${ch}" at index ${i}`);
    }
    const kind: Kind = ch;
    const id = i + 1;
    const nextChar = i + 1 < pattern.length ? pattern[i + 1] : null;
    const feeder = kind === 'M' && nextChar === '*';
    const sensitive = kind === '*' || feeder;
    const bracket = bracketIndexForBay(id, brackets);
    bays.push({ id, kind, feeder, sensitive, bracket });
  }
  return Object.freeze(bays);
}

export const NEMOTRON_SPEC: NemotronSpec = Object.freeze({
  revision: CHECKPOINT_REVISION,
  pattern: HYBRID_OVERRIDE_PATTERN,
  bays: buildBays(HYBRID_OVERRIDE_PATTERN, REPORT_BRACKETS),
  brackets: REPORT_BRACKETS,
  hidden: 2688,
  vocabSize: 131072,
  tieWordEmbeddings: false,
  positionalEncoding: 'NoPE',
  mamba: Object.freeze({
    heads: 64,
    channels: 64,
    state: 128,
    groups: 8,
    kernel: 4,
  }),
  moe: Object.freeze({
    routed: 128,
    active: 6,
    shared: 1,
    width: 1856,
    sharedWidth: 3712,
  }),
  gqa: Object.freeze({
    q: 32,
    kv: 2,
    headDim: 128,
  }),
});

/** Center X coordinate of 1-based bay `id` in drawing units. */
export function bayCenterX(id: number): number {
  return id - 26.5;
}

/** Operating compute/weight precision of a bay's main body. */
export function bodyPrecision(bay: BayDescriptor, mode: PrecisionMode): EffectivePrecision {
  if (mode === 'bf16') return 'bf16';
  return bay.sensitive ? 'bf16' : 'fp8';
}

/** All 23 Mamba-2 Conv1D strips remain BF16 in both precision modes. */
export function conv1dPrecision(bay: BayDescriptor, _mode: PrecisionMode): EffectivePrecision | null {
  if (bay.kind !== 'M') return null;
  return 'bf16';
}

/** All 23 Mamba-2 SSM state registers are always stored in FP32. */
export function ssmStateDtype(bay: BayDescriptor): 'fp32' | null {
  if (bay.kind !== 'M') return null;
  return 'fp32';
}

/** All 6 attention KV caches switch to FP8 in Selective mode, even beside BF16 attention bodies. */
export function kvCachePrecision(bay: BayDescriptor, mode: PrecisionMode): EffectivePrecision | null {
  if (bay.kind !== '*') return null;
  return mode === 'fp8_selective' ? 'fp8' : 'bf16';
}

export function formatBayId(id: number): string {
  return String(id).padStart(2, '0');
}

export function formatBracketCaption(bracket: Bracket): string {
  const [first, last, motif, repeats] = bracket;
  const dotted = motif.split('').join('·');
  return `${formatBayId(first)}–${formatBayId(last)} [${dotted}] ×${repeats}`;
}

export function bayKindLabel(kind: Kind): string {
  switch (kind) {
    case 'M':
      return 'Mamba-2';
    case 'E':
      return 'MoE';
    case '*':
      return 'GQA Attention';
  }
}

export function countByKind(spec: NemotronSpec): { M: number; E: number; '*': number } {
  let M = 0;
  let E = 0;
  let attn = 0;
  for (const bay of spec.bays) {
    if (bay.kind === 'M') M++;
    else if (bay.kind === 'E') E++;
    else attn++;
  }
  return { M, E, '*': attn };
}
