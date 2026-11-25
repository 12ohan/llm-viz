import { formatMemoryReadout, memoryFor } from '../model/memory';
import {
  bayKindLabel,
  bodyPrecision,
  conv1dPrecision,
  formatBayId,
  formatBracketCaption,
  kvCachePrecision,
  NEMOTRON_SPEC,
  PARAMETER_CAPTION,
  ssmStateDtype,
  type BayDescriptor,
  type NemotronSpec,
  type PrecisionMode,
} from '../model/spec';

type BayKind = BayDescriptor['kind'];
import {
  MAX_TOKENS,
  MIN_TOKENS,
  type ExecutionMode,
  type GqaViewAngle,
  type SceneState,
  type SpineWaveMode,
  type TocSectionId,
  type ViewMode,
} from '../model/state';
import { fmtSigned, formatEpsLabel, GQA_STEPPER_STEPS } from '../math/gqaSim';
import type { HybridExecutionResult } from '../math/hybridFlow';
import {
  getActiveDecodeBranch,
  getActiveDecodeStep,
  type HybridDecodeSession,
} from '../math/incrementalDecode';
import { MAMBA_STEPPER_STEPS } from '../math/mambaSim';
import { formatExpertId, MOE_STEPPER_STEPS } from '../math/moeSim';
import {
  EMBEDDING_STEPPER_STEPS,
  formatMambaHeadId,
  formatQueryHeadId,
  mambaGroupForHead,
  OUTPUT_STEPPER_STEPS,
  parseMambaHeadPartId,
  parseQueryHeadPartId,
  type CutawayPartInfo,
  type CutawaySceneGeometry,
} from '../scene/cutaways';
import { describeSpineWaveStage } from '../scene/spine';
import {
  beatsForChapter,
  getWalkthroughBeat,
  getWalkthroughChapter,
  resolveBeatDisplayCopy,
  WALKTHROUGH_BEAT_COUNT,
  WALKTHROUGH_BEATS,
  type LiveDecodeBeatSummary,
} from '../story/walkthrough';

export type CrossBayNodeKind = 'embedding' | 'output' | BayKind;

export interface CrossBayNeighbor {
  readonly kind: CrossBayNodeKind;
  /** 0 for Embedding, 1..52 for tower bays, 53 for Output LM Head. */
  readonly bayId: number;
  readonly mode: Exclude<ViewMode, 'tower'>;
  /** Concise label for clickable breadcrumb buttons, e.g. "Embed X₀", "Bay 01 M", "Bay 02 MoE". */
  readonly shortLabel: string;
  /** Full descriptive label for active breadcrumb pill & tooltip, e.g. "Bay 01 Mamba-2". */
  readonly label: string;
}

export interface CrossBayNeighbors {
  readonly prev: CrossBayNeighbor | null;
  readonly current: CrossBayNeighbor;
  readonly next: CrossBayNeighbor | null;
}

function makeEmbeddingCrossBayNode(): CrossBayNeighbor {
  return {
    kind: 'embedding',
    bayId: 0,
    mode: 'cutaway_embedding',
    shortLabel: 'Embed X₀',
    label: 'Embed X₀',
  };
}

function makeOutputCrossBayNode(totalBays: number): CrossBayNeighbor {
  return {
    kind: 'output',
    bayId: totalBays + 1,
    mode: 'cutaway_output',
    shortLabel: 'Output',
    label: 'Output LM Head',
  };
}

function makeBayCrossBayNode(bay: BayDescriptor): CrossBayNeighbor {
  const idStr = formatBayId(bay.id);
  const mode: Exclude<ViewMode, 'tower'> =
    bay.kind === 'M' ? 'cutaway_mamba' : bay.kind === 'E' ? 'cutaway_moe' : 'cutaway_gqa';
  const shortKind = bay.kind === 'M' ? 'M' : bay.kind === 'E' ? 'MoE' : 'GQA';
  const fullKind = bay.kind === 'M' ? 'Mamba-2' : bay.kind === 'E' ? 'MoE' : 'GQA';
  return {
    kind: bay.kind,
    bayId: bay.id,
    mode,
    shortLabel: `Bay ${idStr} ${shortKind}`,
    label: `Bay ${idStr} ${fullKind}`,
  };
}

/**
 * Resolve upstream (prev), current, and downstream (next) architectural stages for any cutaway
 * mode and bay ID along the 52-layer Nemotron 3 Nano spine (Embed X₀ → Bay 01..52 → Output LM Head).
 */
export function buildCrossBayNeighbors(
  mode: ViewMode,
  bayId?: number,
  spec: NemotronSpec = NEMOTRON_SPEC,
): CrossBayNeighbors | null {
  const totalBays = spec.bays.length;
  if (totalBays === 0) return null;

  if (mode === 'cutaway_embedding') {
    return {
      prev: null,
      current: makeEmbeddingCrossBayNode(),
      next: makeBayCrossBayNode(spec.bays[0]),
    };
  }
  if (mode === 'cutaway_output') {
    return {
      prev: makeBayCrossBayNode(spec.bays[totalBays - 1]),
      current: makeOutputCrossBayNode(totalBays),
      next: null,
    };
  }

  let resolvedBayId: number | null = null;
  if (bayId !== undefined && Number.isFinite(bayId)) {
    const truncated = Math.trunc(bayId);
    if (truncated === 0) {
      return {
        prev: null,
        current: makeEmbeddingCrossBayNode(),
        next: makeBayCrossBayNode(spec.bays[0]),
      };
    }
    if (truncated > totalBays) {
      return {
        prev: makeBayCrossBayNode(spec.bays[totalBays - 1]),
        current: makeOutputCrossBayNode(totalBays),
        next: null,
      };
    }
    resolvedBayId = Math.max(1, Math.min(totalBays, truncated));
  } else if (mode === 'cutaway_mamba') {
    resolvedBayId = 1;
  } else if (mode === 'cutaway_moe') {
    resolvedBayId = 2;
  } else if (mode === 'cutaway_gqa') {
    resolvedBayId = 6;
  }

  if (resolvedBayId === null) return null;
  const currentBay = spec.bays[resolvedBayId - 1];
  if (!currentBay) return null;

  return {
    prev:
      resolvedBayId === 1
        ? makeEmbeddingCrossBayNode()
        : makeBayCrossBayNode(spec.bays[resolvedBayId - 2]),
    current: makeBayCrossBayNode(currentBay),
    next:
      resolvedBayId === totalBays
        ? makeOutputCrossBayNode(totalBays)
        : makeBayCrossBayNode(spec.bays[resolvedBayId]),
  };
}

export interface ControlsCallbacks {
  onSelectMode: (mode: ViewMode) => void;
  onSelectSection: (section: TocSectionId) => void;
  onSelectMambaBay: (bayId: 1 | 42) => void;
  onSelectMambaView: (view: GqaViewAngle) => void;
  onSelectMambaStateToken: (tokenIdx: number | null) => void;
  onStepMambaStateToken: (delta: -1 | 1) => void;
  onToggleMambaTemporalPlay: () => void;
  onSelectMoeBay: (bayId: 2 | 52) => void;
  onSelectMoeView: (view: GqaViewAngle) => void;
  onSelectMoeFocusToken: (tokenIdx: number) => void;
  onStepMoeFocusToken: (delta: -1 | 1) => void;
  onToggleMoeTokenCycle: () => void;
  onSelectMoeExpert: (expertId: number) => void;
  onSelectGqaBay: (bayId: 6 | 43) => void;
  onSelectGqaView: (view: GqaViewAngle) => void;
  onSelectPart: (partId: string | null) => void;
  onInspectInternals: (kind: BayKind, bayId?: number) => void;
  onTokensChange: (tokens: number) => void;
  onPrecisionChange: (mode: PrecisionMode) => void;
  onFitAll: () => void;
  onZoomSelectedBay: () => void;
  onReset: () => void;
  onStepBay: (delta: -1 | 1) => void;
  onClearSelection: () => void;
  onTourToggle: () => void;
  onTourPrev: () => void;
  onTourNext: () => void;
  onTourRestart: () => void;
  onTourExit: () => void;
  onToggleMemoryChart: () => void;
  onToggleShortcuts: () => void;
  onCloseShortcuts: () => void;
  onGqaStepPrev: () => void;
  onGqaStepNext: () => void;
  onGqaStepTogglePlay: () => void;
  onGqaStepSelect: (step: number) => void;
  onGqaStepScrub: (progress01: number) => void;
  onGqaClearCell: () => void;
  onToggleAutoFollow: () => void;
  onSelectExecutionMode: (mode: ExecutionMode) => void;
  onDecodeStepToken: (tokenId: number) => void;
  onDecodeStepGreedy: () => void;
  onDecodeNavStep: (stepIndex: number) => void;
  onDecodeSwitchBranch: (branchId: string) => void;
  onDecodeReset: () => void;
  // V4 / V5 Optional Callbacks (backward-compatible prior to Wave 3 main.ts wiring)
  onTourBeatPrev?: () => void;
  onTourBeatNext?: () => void;
  onTourJumpBeat?: (beatIndex: number) => void;
  onTourResumeBeat?: () => void;
  onTourRefitBeat?: () => void;
  onTourToggleBeatPlay?: () => void;
  onSelectEmbeddingToken?: (tokenIdx: number) => void;
  onSelectOutputToken?: (tokenIdx: number) => void;
  onNavigateAdjacentBay?: (neighbor: CrossBayNeighbor) => void;
  onReplaySpineWave?: () => void;
  // RC3 Optional Spine Replay Mode / HUD Callbacks (backward-compatible)
  readonly onReplaySpineWaveMode?: (mode: SpineWaveMode) => void;
  readonly onToggleSpineWavePause?: () => void;
  readonly onDismissSpineWaveHud?: () => void;
}

export type UiCallbacks = ControlsCallbacks;

interface SectionCopy {
  readonly title: string;
  readonly badge: string;
  readonly body: string;
  readonly specs: readonly string[];
  readonly inspectKind?: BayKind;
}

const WEIGHTS_SUMMARY_LINE = 'Weights: 31.6B (~63 GB, ~94% MoE) · 3.2B active';
const SHARED_EXPERT_NOTE =
  'Shared expert width 3,712 (= 2 × 1,856 in NVIDIA paper, fused in checkpoint)';

const SECTION_SHORT_LABELS: Record<TocSectionId, string> = {
  overview: 'Overview',
  embedding: 'Embedding',
  mamba: 'Mamba-2 (×23)',
  moe: 'MoE 128+1 (×23)',
  gqa: 'GQA Attention (×6)',
  kv_fp8: 'KV Cache & FP8',
  output: 'Output',
};

const COMPACT_VIEWPORT_HEIGHT_PX = 840;

const SECTION_COPY: Record<TocSectionId, SectionCopy> = {
  overview: {
    title: '52-Layer Hybrid Architecture',
    badge: '23 Mamba-2 · 23 MoE · 6 GQA',
    body:
      'Nemotron 3 Nano replaces most quadratic attention blocks with linear-time Mamba-2 state-space layers and sparse Mixture-of-Experts banks. Only six bays (06, 13, 20, 27, 34, 43) accumulate K/V token history, while a continuous 2,688-channel emerald residual stream carries hidden state top to bottom.',
    specs: [
      PARAMETER_CAPTION,
      'Figure 2 layout: 01–35 [M·E·M·E·M·*·E] ×5 · 36–41 [M·E] ×3 · 42–44 [M·*·E] ×1 · 45–52 [M·E] ×4',
      'Pre-RMSNorm · NoPE (no rotary position embedding) · Untied 131,072-token output head',
    ],
  },
  embedding: {
    title: 'Token Embedding Matrix',
    badge: '131,072 × 2,688',
    body:
      'Each input token ID looks up one 2,688-dimensional row from the embedding table and injects it directly into the emerald residual stream on the right. Because embeddings are untied from the final LM head, input and output projections maintain independent parameters.',
    specs: [
      'Vocab size: 131,072 · Hidden width: 2,688 · Tie word embeddings: false',
      PARAMETER_CAPTION,
    ],
  },
  mamba: {
    title: 'Mamba-2 Recurrent Layers',
    badge: '23 bays · Fixed-size state',
    body:
      'Each Mamba-2 layer updates a four-slot causal Conv1D buffer and a square cyan-speckled state-space register in place. Scrubbing the token count shifts the register cell pattern without growing memory — all 23 Mamba-2 layers together stay fixed at 47.1 MiB.',
    specs: [
      'SSM state: 64 heads × 64 channels × 128 state values (FP32 storage)',
      'Conv1D state: 6,144 channels × 4 slots (always BF16 even in FP8 Selective)',
      'Output projection: 4,096 → 2,688 after gated RMSNorm',
    ],
    inspectKind: 'M',
  },
  moe: {
    title: 'Mixture of Experts (128 + 1)',
    badge: '23 bays · 6 routed + 1 shared',
    body:
      'A 128-score sigmoid router activates exactly six routed experts per token in parallel (amber-lit tiles, width 1,856, non-gated ReLU²), leaving 122 experts idle (95.3% skipped). Beside the bank sits one always-active parallel shared expert.',
    specs: [
      'Routed experts: 6 active of 128 · width 2,688 → 1,856 → 2,688 (non-gated ReLU²)',
      SHARED_EXPERT_NOTE,
      '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B token cache',
    ],
    inspectKind: 'E',
  },
  gqa: {
    title: 'Grouped-Query Attention',
    badge: '6 bays · 06, 13, 20, 27, 34, 43',
    body:
      'Only six layers use full attention, standing out along the tower with periwinkle Q/K/V projection plates and two paired emerald/cyan KV-cache tracks extending forward. Thirty-two query heads share just two KV heads (16 query heads per KV pair) with NoPE.',
    specs: [
      'Q projection: 32 heads × 128 = 4,096 · K & V projections: 2 heads × 128 = 256 each',
      'Logical KV payload: +6 KiB/token in BF16 · +3 KiB/token in FP8 Selective',
      '52-layer pure GQA reference would cost +52 KiB/token (BF16) or +26 KiB/token (FP8)',
    ],
    inspectKind: '*',
  },
  kv_fp8: {
    title: 'KV Cache & FP8 Selective Mode',
    badge: 'NVIDIA §4.2 · 12 BF16 / 40 FP8',
    body:
      'Toggle FP8 Selective below to see NVIDIA’s sensitivity boundary pop: the 40 quantized layers fade to frosted slate, while the six GQA bays (06, 13, 20, 27, 34, 43) and their six immediately preceding Mamba-2 feeders (05, 12, 19, 26, 33, 42) stay vivid BF16 with warm gold borders.',
    specs: [
      'All 23 Mamba Conv1D strips remain BF16; all 23 SSM registers remain FP32 (47.1 MiB fixed)',
      'All 6 GQA KV caches switch to FP8 even beside BF16 attention bodies (+3 KiB/token)',
      'Crossover vs 52-layer GQA reference: 1,048 tokens (BF16) · 2,096 tokens (FP8)',
    ],
    inspectKind: '*',
  },
  output: {
    title: 'Final RMSNorm & Untied LM Head',
    badge: '2,688 → 131,072 logits',
    body:
      'After Bay 52 (MoE), the emerald residual stream passes through a final amber RMSNorm wafer and an untied periwinkle projection plate that maps the 2,688-channel vector into 131,072 vocabulary logits.',
    specs: [
      'Final RMSNorm (2,688) → Untied LM Head (2,688 × 131,072) → Output Logits',
      PARAMETER_CAPTION,
    ],
  },
};

export interface ControlsElements {
  sidebarTop: HTMLElement;
  sidebarTopBody: HTMLDivElement;
  sidebarTocToggle: HTMLButtonElement;
  sidebarCurrentSection: HTMLSpanElement;
  modeButtons: HTMLButtonElement[];
  tocButtons: HTMLButtonElement[];
  presetButtons: HTMLButtonElement[];
  mambaBayButtons: HTMLButtonElement[];
  mambaViewButtons: HTMLButtonElement[];
  moeBayButtons: HTMLButtonElement[];
  moeViewButtons: HTMLButtonElement[];
  gqaBayButtons: HTMLButtonElement[];
  gqaViewButtons: HTMLButtonElement[];
  miniNodes: SVGGElement[];
  tourToggleBtn: HTMLButtonElement;
  floatingToolbar: HTMLDivElement;
  towerSubbar: HTMLDivElement;
  mambaSubbar: HTMLDivElement;
  moeSubbar: HTMLDivElement;
  gqaSubbar: HTMLDivElement;
  floatingBadge: HTMLDivElement;
  orbitHint: HTMLDivElement;
  gqaMathDock: HTMLElement;
  gqaFormulaTensor: HTMLSpanElement;
  gqaFormulaCoord: HTMLSpanElement;
  gqaFormulaHead: HTMLSpanElement;
  gqaFormulaValue: HTMLSpanElement;
  gqaFormulaNoteBtn: HTMLButtonElement;
  gqaFormulaNote: HTMLDivElement;
  gqaCellClear: HTMLButtonElement;
  gqaFormulaEq: HTMLDivElement;
  gqaFormulaSub: HTMLDivElement;
  gqaStepPrev: HTMLButtonElement;
  gqaStepPlay: HTMLButtonElement;
  gqaStepNext: HTMLButtonElement;
  gqaStepPills: HTMLButtonElement[];
  gqaStepAutoFollow: HTMLButtonElement;
  gqaStepScrub: HTMLInputElement;
  editorialPanel: HTMLDivElement;
  execModeFull: HTMLButtonElement;
  execModeDecode: HTMLButtonElement;
  tokenSlider: HTMLInputElement;
  tokenDec: HTMLButtonElement;
  tokenInc: HTMLButtonElement;
  tokenValue: HTMLOutputElement;
  precisionBf16: HTMLButtonElement;
  precisionFp8: HTMLButtonElement;
  memoryReadout: HTMLDivElement;
  memoryReadoutLines: HTMLDivElement;
  memoryChartBtn: HTMLButtonElement;
  bayStepRow: HTMLDivElement;
  bayPrev: HTMLButtonElement;
  bayNext: HTMLButtonElement;
  bayClear: HTMLButtonElement;
  zoomBayBtn: HTMLButtonElement;
  fitBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  shortcutsBtn: HTMLButtonElement;
  shortcutsPopover: HTMLDivElement;
  shortcutsCloseBtn: HTMLButtonElement;
  spineWaveHud?: HTMLDivElement | null;
}

export function queryControlsElements(): ControlsElements {
  const get = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing required DOM element #${id}`);
    return el as T;
  };
  return {
    sidebarTop: get<HTMLElement>('sidebar-top'),
    sidebarTopBody: get<HTMLDivElement>('sidebar-top-body'),
    sidebarTocToggle: get<HTMLButtonElement>('sidebar-toc-toggle'),
    sidebarCurrentSection: get<HTMLSpanElement>('sidebar-current-section'),
    modeButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.mode-btn[data-mode]')),
    tocButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.toc-btn[data-section]')),
    presetButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('.pill-btn[data-preset]')),
    mambaBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-mamba-bay]')),
    mambaViewButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-mamba-view]')),
    moeBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-moe-bay]')),
    moeViewButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-moe-view]')),
    gqaBayButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-bay]')),
    gqaViewButtons: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-view]')),
    miniNodes: Array.from(document.querySelectorAll<SVGGElement>('.mini-node[data-mini-section]')),
    tourToggleBtn: get<HTMLButtonElement>('tour-toggle-btn'),
    floatingToolbar: get<HTMLDivElement>('floating-toolbar'),
    towerSubbar: get<HTMLDivElement>('tower-subbar'),
    mambaSubbar: get<HTMLDivElement>('mamba-subbar'),
    moeSubbar: get<HTMLDivElement>('moe-subbar'),
    gqaSubbar: get<HTMLDivElement>('gqa-subbar'),
    floatingBadge: get<HTMLDivElement>('floating-badge'),
    orbitHint: get<HTMLDivElement>('orbit-hint'),
    gqaMathDock: get<HTMLElement>('gqa-math-dock'),
    gqaFormulaTensor: get<HTMLSpanElement>('gqa-formula-tensor'),
    gqaFormulaCoord: get<HTMLSpanElement>('gqa-formula-coord'),
    gqaFormulaHead: get<HTMLSpanElement>('gqa-formula-head'),
    gqaFormulaValue: get<HTMLSpanElement>('gqa-formula-value'),
    gqaFormulaNoteBtn: get<HTMLButtonElement>('gqa-formula-note-btn'),
    gqaFormulaNote: get<HTMLDivElement>('gqa-formula-note'),
    gqaCellClear: get<HTMLButtonElement>('gqa-cell-clear'),
    gqaFormulaEq: get<HTMLDivElement>('gqa-formula-eq'),
    gqaFormulaSub: get<HTMLDivElement>('gqa-formula-sub'),
    gqaStepPrev: get<HTMLButtonElement>('gqa-step-prev'),
    gqaStepPlay: get<HTMLButtonElement>('gqa-step-play'),
    gqaStepNext: get<HTMLButtonElement>('gqa-step-next'),
    gqaStepPills: Array.from(document.querySelectorAll<HTMLButtonElement>('[data-gqa-step]')),
    gqaStepAutoFollow: get<HTMLButtonElement>('gqa-step-autofollow'),
    gqaStepScrub: get<HTMLInputElement>('gqa-step-scrub'),
    editorialPanel: get<HTMLDivElement>('editorial-panel'),
    execModeFull: get<HTMLButtonElement>('exec-mode-full'),
    execModeDecode: get<HTMLButtonElement>('exec-mode-decode'),
    tokenSlider: get<HTMLInputElement>('token-slider'),
    tokenDec: get<HTMLButtonElement>('token-dec'),
    tokenInc: get<HTMLButtonElement>('token-inc'),
    tokenValue: get<HTMLOutputElement>('token-value'),
    precisionBf16: get<HTMLButtonElement>('precision-bf16'),
    precisionFp8: get<HTMLButtonElement>('precision-fp8'),
    memoryReadout: get<HTMLDivElement>('memory-readout'),
    memoryReadoutLines: get<HTMLDivElement>('memory-readout-lines'),
    memoryChartBtn: get<HTMLButtonElement>('memory-chart-btn'),
    bayStepRow: get<HTMLDivElement>('bay-step-row'),
    bayPrev: get<HTMLButtonElement>('bay-prev'),
    bayNext: get<HTMLButtonElement>('bay-next'),
    bayClear: get<HTMLButtonElement>('bay-clear'),
    zoomBayBtn: get<HTMLButtonElement>('zoom-bay-btn'),
    fitBtn: get<HTMLButtonElement>('fit-btn'),
    resetBtn: get<HTMLButtonElement>('reset-btn'),
    shortcutsBtn: get<HTMLButtonElement>('shortcuts-btn'),
    shortcutsPopover: get<HTMLDivElement>('shortcuts-popover'),
    shortcutsCloseBtn: get<HTMLButtonElement>('shortcuts-close'),
    spineWaveHud: document.getElementById('spine-wave-hud') as HTMLDivElement | null,
  };
}

/**
 * Build the sidebar editorial panel cache key. Includes effective cutaway token, head, and expert
 * from `activeCutaway` so hovering or clicking across different tokens/heads within the same tensor
 * or part always refreshes the sidebar Recurrence controls, head grid, and `partCatalog` specs.
 */
export function computeEditorialCacheKey(
  state: SceneState,
  activeCutaway: CutawaySceneGeometry | null,
  activeExecution: HybridExecutionResult | null = null,
  boundaryInspectToken: number | null = null,
  decodeSession: HybridDecodeSession | null = null,
  selectedCandidateTokenId: number | null = null,
): string {
  const isTower = state.mode === 'tower';
  const activeBayId = isTower ? state.hoveredBay ?? state.selectedBay : null;
  const activePart = !isTower ? state.hoveredPart ?? state.selectedPart : null;
  const effMambaHead = activeCutaway?.mambaStats?.activeHead ?? state.mambaActiveHead;
  const effMambaTok =
    activeCutaway?.mambaStats?.stateTokenIndex ?? state.mambaStateToken ?? 'last';
  const effMoeTok = activeCutaway?.moeStats?.focusToken ?? state.moeFocusToken ?? 'last';
  const effMoeExp = activeCutaway?.moeStats?.focusExpert ?? state.moeActiveExpert ?? 'top0';
  const effGqaG0 = activeCutaway?.gqaStats?.activeGroup0Head ?? state.gqaActiveGroup0Head;
  const effGqaG1 = activeCutaway?.gqaStats?.activeGroup1Head ?? state.gqaActiveGroup1Head;
  const execKey = activeExecution ? activeExecution.executionId : 'standalone';
  const decodeKey = decodeSession
    ? (() => {
        const activeBranch = getActiveDecodeBranch(decodeSession);
        const activeStep = getActiveDecodeStep(decodeSession);
        return `${decodeSession.activeBranchId}:${decodeSession.activeStepIndex}/${activeBranch.steps.length}:${decodeSession.branches.size}:${activeStep.stepExecutionId}:${activeStep.provenance.nextStateDigest}:cand${selectedCandidateTokenId ?? 'g'}`;
      })()
    : 'full';
  const boundTok =
    isTower || state.mode === 'cutaway_embedding' || state.mode === 'cutaway_output'
      ? (boundaryInspectToken ?? 'last')
      : 'na';
  const tourKey = state.tourActive
    ? `tour:${state.tourChapter}:b${state.tourBeat}:${state.tourExploring ? 'exp' : 'guided'}`
    : 'free';
  const waveKey = `${state.spineWaveMode}:${state.spineWaveProgress !== null ? 'wave' : 'idle'}`;
  const stepPlayKey =
    state.mode === 'cutaway_mamba'
      ? (state.mambaStepperPlaying ? 'sp1' : 'sp0')
      : state.mode === 'cutaway_moe'
        ? (state.moeStepperPlaying ? 'sp1' : 'sp0')
        : state.mode === 'cutaway_gqa'
          ? (state.gqaStepperPlaying ? 'sp1' : 'sp0')
          : 'sp0';
  return `${state.executionMode}:${decodeKey}:${tourKey}:${waveKey}:${stepPlayKey}:${execKey}:${boundTok}:${state.mode}:et${state.embeddingFocusToken}:ot${state.outputFocusToken}:${state.mambaBayId}:${effMambaHead}:${effMambaTok}:${state.mambaStateToken ?? 'last'}:${state.mambaTemporalPlaying ? 'tplay' : 'tpause'}:${state.moeBayId}:${effMoeTok}:${effMoeExp}:${state.moeTokenCycling ? 'mplay' : 'mpause'}:${state.gqaBayId}:${effGqaG0}:${effGqaG1}:${state.section}:${activeBayId ?? 'none'}:${state.selectedBay ?? 'none'}:${activePart ?? 'none'}:${state.selectedPart ?? 'none'}:${state.precision}:${state.tokens}`;
}

interface SpineWaveHudNodes {
  readonly dot: HTMLSpanElement;
  readonly title: HTMLSpanElement;
  readonly rms: HTMLSpanElement;
  readonly pauseBtn: HTMLButtonElement;
  readonly closeBtn: HTMLButtonElement;
  readonly detail: HTMLDivElement;
  readonly fill: HTMLDivElement;
  readonly disclaimer: HTMLSpanElement;
  readonly pct: HTMLSpanElement;
}

const SPINE_WAVE_STAGE_CLASSES: readonly string[] = [
  'is-embedding',
  'is-mamba',
  'is-moe',
  'is-gqa',
  'is-final_norm',
  'is-final-norm',
  'is-output',
];

export class MacroControls {
  private readonly spec: NemotronSpec;
  private readonly els: ControlsElements;
  private readonly cb: ControlsCallbacks;
  private readonly cleanupFns: (() => void)[] = [];

  private lastReadoutKey = '';
  private lastEditorialKey = '';
  private boundaryInspectToken: number | null = null;
  private selectedCandidateTokenId: number | null = null;
  private lastDecodeStepId = '';
  private tocCompactUserOverride: boolean | null = null;
  private isSidebarTopCompact = false;
  private lastSyncedState: SceneState | null = null;
  private lastSyncedCutaway: CutawaySceneGeometry | null = null;
  private lastSyncedExecution: HybridExecutionResult | null = null;
  private lastSyncedDecodeSession: HybridDecodeSession | null = null;
  private spineWaveHudNodes: SpineWaveHudNodes | null = null;

  constructor(spec: NemotronSpec, els: ControlsElements, cb: ControlsCallbacks) {
    this.spec = spec;
    this.els = els;
    this.cb = cb;
    this.bindEvents();
    this.applySidebarCompactState('full_sequence');
  }

  private on<K extends keyof HTMLElementEventMap>(
    el: Element,
    type: K,
    handler: (ev: HTMLElementEventMap[K]) => void,
  ): void {
    el.addEventListener(type, handler as EventListener);
    this.cleanupFns.push(() => el.removeEventListener(type, handler as EventListener));
  }

  private computeDesiredCompact(executionMode: ExecutionMode): boolean {
    if (this.tocCompactUserOverride !== null) {
      return this.tocCompactUserOverride;
    }
    const viewportH = typeof window !== 'undefined' ? window.innerHeight : 900;
    return viewportH <= COMPACT_VIEWPORT_HEIGHT_PX || executionMode === 'ar_decode';
  }

  private applySidebarCompactState(executionMode: ExecutionMode): void {
    const compact = this.computeDesiredCompact(executionMode);
    this.isSidebarTopCompact = compact;
    this.els.sidebarTop.classList.toggle('is-compact', compact);
    this.els.sidebarTopBody.hidden = compact;
    this.els.sidebarTocToggle.setAttribute('aria-expanded', String(!compact));
  }

  private bindEvents(): void {
    this.on(this.els.sidebarTocToggle, 'click', () => {
      this.tocCompactUserOverride = !this.isSidebarTopCompact;
      this.applySidebarCompactState(this.lastSyncedState?.executionMode ?? 'full_sequence');
    });

    if (typeof window !== 'undefined') {
      const onResize = () => {
        if (this.tocCompactUserOverride === null) {
          this.applySidebarCompactState(this.lastSyncedState?.executionMode ?? 'full_sequence');
        }
      };
      window.addEventListener('resize', onResize);
      this.cleanupFns.push(() => window.removeEventListener('resize', onResize));
    }

    for (const btn of this.els.modeButtons) {
      this.on(btn, 'click', () => {
        const rawMode = btn.dataset.mode ?? btn.dataset.view;
        if (
          rawMode === 'tower' ||
          rawMode === 'cutaway_embedding' ||
          rawMode === 'cutaway_mamba' ||
          rawMode === 'cutaway_moe' ||
          rawMode === 'cutaway_gqa' ||
          rawMode === 'cutaway_output'
        ) {
          this.cb.onSelectMode(rawMode);
        }
      });
    }

    for (const btn of this.els.tocButtons) {
      this.on(btn, 'click', () => {
        const sec = btn.dataset.section as TocSectionId | undefined;
        if (sec) this.cb.onSelectSection(sec);
      });
    }

    for (const btn of this.els.presetButtons) {
      this.on(btn, 'click', () => {
        const preset = btn.dataset.preset as TocSectionId | undefined;
        if (preset) this.cb.onSelectSection(preset);
      });
    }

    for (const btn of this.els.mambaBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.mambaBay ?? '1', 10);
        if (id === 1 || id === 42) this.cb.onSelectMambaBay(id);
      });
    }

    for (const btn of this.els.mambaViewButtons) {
      this.on(btn, 'click', () => {
        const view = btn.dataset.mambaView as GqaViewAngle | undefined;
        if (view === 'three_quarter' || view === 'near_plan') {
          this.cb.onSelectMambaView(view);
        }
      });
    }

    for (const btn of this.els.moeBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.moeBay ?? '2', 10);
        if (id === 2 || id === 52) this.cb.onSelectMoeBay(id);
      });
    }

    for (const btn of this.els.moeViewButtons) {
      this.on(btn, 'click', () => {
        const view = btn.dataset.moeView as GqaViewAngle | undefined;
        if (view === 'three_quarter' || view === 'near_plan') {
          this.cb.onSelectMoeView(view);
        }
      });
    }

    for (const btn of this.els.gqaBayButtons) {
      this.on(btn, 'click', () => {
        const id = Number.parseInt(btn.dataset.gqaBay ?? '6', 10);
        if (id === 6 || id === 43) this.cb.onSelectGqaBay(id);
      });
    }

    for (const btn of this.els.gqaViewButtons) {
      this.on(btn, 'click', () => {
        const view = btn.dataset.gqaView as GqaViewAngle | undefined;
        if (view === 'three_quarter' || view === 'near_plan') {
          this.cb.onSelectGqaView(view);
        }
      });
    }

    for (const node of this.els.miniNodes) {
      this.on(node, 'click', () => {
        const sec = node.dataset.miniSection as TocSectionId | undefined;
        if (sec) this.cb.onSelectSection(sec);
      });
    }

    this.on(this.els.tokenSlider, 'input', () => {
      const val = Number.parseInt(this.els.tokenSlider.value, 10);
      if (Number.isFinite(val)) {
        this.cb.onTokensChange(Math.max(MIN_TOKENS, Math.min(MAX_TOKENS, val)));
      }
    });

    this.on(this.els.tokenDec, 'click', () => {
      const cur = Number.parseInt(this.els.tokenSlider.value, 10) || 8;
      this.cb.onTokensChange(Math.max(MIN_TOKENS, cur - 1));
    });

    this.on(this.els.tokenInc, 'click', () => {
      const cur = Number.parseInt(this.els.tokenSlider.value, 10) || 8;
      this.cb.onTokensChange(Math.min(MAX_TOKENS, cur + 1));
    });

    this.on(this.els.execModeFull, 'click', () => this.cb.onSelectExecutionMode('full_sequence'));
    this.on(this.els.execModeDecode, 'click', () => this.cb.onSelectExecutionMode('ar_decode'));

    this.on(this.els.precisionBf16, 'click', () => this.cb.onPrecisionChange('bf16'));
    this.on(this.els.precisionFp8, 'click', () => this.cb.onPrecisionChange('fp8_selective'));

    this.on(this.els.fitBtn, 'click', () => this.cb.onFitAll());
    this.on(this.els.zoomBayBtn, 'click', () => this.cb.onZoomSelectedBay());
    this.on(this.els.resetBtn, 'click', () => this.cb.onReset());

    this.on(this.els.bayPrev, 'click', () => this.cb.onStepBay(-1));
    this.on(this.els.bayNext, 'click', () => this.cb.onStepBay(1));
    this.on(this.els.bayClear, 'click', () => this.cb.onClearSelection());

    this.on(this.els.tourToggleBtn, 'click', () => this.cb.onTourToggle());
    this.on(this.els.memoryChartBtn, 'click', () => this.cb.onToggleMemoryChart());
    this.on(this.els.shortcutsBtn, 'click', () => this.cb.onToggleShortcuts());
    this.on(this.els.shortcutsCloseBtn, 'click', () => this.cb.onCloseShortcuts());

    this.on(this.els.gqaStepPrev, 'click', () => this.cb.onGqaStepPrev());
    this.on(this.els.gqaStepPlay, 'click', () => this.cb.onGqaStepTogglePlay());
    this.on(this.els.gqaStepNext, 'click', () => this.cb.onGqaStepNext());
    this.on(this.els.gqaStepAutoFollow, 'click', () => this.cb.onToggleAutoFollow());
    this.on(this.els.gqaCellClear, 'click', () => this.cb.onGqaClearCell());
    this.on(this.els.gqaFormulaNoteBtn, 'click', (ev) => {
      ev.stopPropagation();
      const willOpen = this.els.gqaFormulaNote.hidden;
      this.els.gqaFormulaNote.hidden = !willOpen;
      this.els.gqaFormulaNoteBtn.setAttribute('aria-expanded', String(willOpen));
      this.els.gqaFormulaNoteBtn.classList.toggle('is-open', willOpen);
    });
    const onDocPointerDown = (ev: PointerEvent) => {
      if (this.els.gqaFormulaNote.hidden) return;
      const tgt = ev.target as Node | null;
      if (
        tgt &&
        (this.els.gqaFormulaNote.contains(tgt) || this.els.gqaFormulaNoteBtn.contains(tgt))
      ) {
        return;
      }
      this.closeFormulaNotePopover();
    };
    document.addEventListener('pointerdown', onDocPointerDown);
    this.cleanupFns.push(() => document.removeEventListener('pointerdown', onDocPointerDown));
    for (const btn of this.els.gqaStepPills) {
      this.on(btn, 'click', () => {
        const step = Number.parseInt(btn.dataset.gqaStep ?? '0', 10);
        if (Number.isFinite(step)) this.cb.onGqaStepSelect(step);
      });
    }
    this.on(this.els.gqaStepScrub, 'input', () => {
      const val = Number.parseInt(this.els.gqaStepScrub.value, 10);
      if (Number.isFinite(val)) {
        this.cb.onGqaStepScrub(Math.max(0, Math.min(100, val)) / 100);
      }
    });
  }

  closeFormulaNotePopover(): boolean {
    if (this.els.gqaFormulaNote.hidden) return false;
    this.els.gqaFormulaNote.hidden = true;
    this.els.gqaFormulaNoteBtn.setAttribute('aria-expanded', 'false');
    this.els.gqaFormulaNoteBtn.classList.remove('is-open');
    return true;
  }

  sync(
    state: SceneState,
    activeCutaway: CutawaySceneGeometry | null = null,
    autoFollowManualPaused = false,
    activeExecution: HybridExecutionResult | null = null,
    decodeSession: HybridDecodeSession | null = null,
  ): void {
    this.lastSyncedState = state;
    this.lastSyncedCutaway = activeCutaway;
    this.lastSyncedExecution = activeExecution;
    this.lastSyncedDecodeSession = decodeSession;
    if (decodeSession) {
      const activeStep = getActiveDecodeStep(decodeSession);
      if (activeStep.stepExecutionId !== this.lastDecodeStepId) {
        this.lastDecodeStepId = activeStep.stepExecutionId;
        this.selectedCandidateTokenId = activeStep.greedyNextToken.tokenId;
      }
    } else {
      this.lastDecodeStepId = '';
      this.selectedCandidateTokenId = null;
    }
    if (this.boundaryInspectToken !== null && this.boundaryInspectToken >= state.tokens) {
      this.boundaryInspectToken = state.tokens - 1;
    }
    // Active View Mode Switcher (all 6 ViewModes: tower + 5 cutaways)
    for (const btn of this.els.modeButtons) {
      const btnMode = btn.dataset.mode ?? btn.dataset.view;
      const isAct = btnMode === state.mode;
      btn.classList.toggle('is-active', isAct);
      btn.setAttribute('aria-selected', String(isAct));
    }

    // Contextual sub-bars
    this.els.towerSubbar.hidden = state.mode !== 'tower';
    this.els.mambaSubbar.hidden = state.mode !== 'cutaway_mamba';
    this.els.moeSubbar.hidden = state.mode !== 'cutaway_moe';
    this.els.gqaSubbar.hidden = state.mode !== 'cutaway_gqa';

    // Mamba, MoE & GQA representative bay & camera view toggles
    for (const btn of this.els.mambaBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.mambaBay) === state.mambaBayId);
    }
    for (const btn of this.els.mambaViewButtons) {
      btn.classList.toggle('is-active', btn.dataset.mambaView === state.mambaViewAngle);
    }
    for (const btn of this.els.moeBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.moeBay) === state.moeBayId);
    }
    for (const btn of this.els.moeViewButtons) {
      btn.classList.toggle('is-active', btn.dataset.moeView === state.moeViewAngle);
    }
    for (const btn of this.els.gqaBayButtons) {
      btn.classList.toggle('is-active', Number(btn.dataset.gqaBay) === state.gqaBayId);
    }
    for (const btn of this.els.gqaViewButtons) {
      btn.classList.toggle('is-active', btn.dataset.gqaView === state.gqaViewAngle);
    }

    // Floating corner badge & orbit hint (hidden when shared live Math Dock occupies bottom)
    const usesLiveMathDock = state.mode !== 'tower';
    this.els.floatingBadge.hidden = usesLiveMathDock;
    this.els.orbitHint.hidden = usesLiveMathDock;
    this.els.gqaMathDock.hidden = !usesLiveMathDock;

    if (!usesLiveMathDock) {
      this.closeFormulaNotePopover();
    }

    if (state.mode === 'tower') {
      this.els.floatingBadge.textContent = 'Nemotron 3 Nano 30B-A3B · 3.2B active';
    } else if (state.mode === 'cutaway_embedding') {
      this.els.floatingBadge.textContent = 'Cutaway: Embedding (NoPE) · 131,072 × 2,688 Untied';
      this.syncMathDock(state, activeCutaway, autoFollowManualPaused, decodeSession, activeExecution);
    } else if (state.mode === 'cutaway_mamba') {
      this.els.floatingBadge.textContent =
        state.mambaBayId === 42
          ? 'Cutaway: Mamba-2 · Bay 42 Feeder (BF16 §4.2)'
          : 'Cutaway: Mamba-2 · Bay 01 (47.1 MiB fixed state)';
      this.syncMathDock(state, activeCutaway, autoFollowManualPaused, decodeSession, activeExecution);
    } else if (state.mode === 'cutaway_moe') {
      this.els.floatingBadge.textContent =
        state.moeBayId === 52
          ? 'Cutaway: MoE 128+1 · Bay 52 Final MoE (6 Active + Shared 3,712)'
          : 'Cutaway: MoE 128+1 · Bay 02 (6 Active + Shared 3,712)';
      this.syncMathDock(state, activeCutaway, autoFollowManualPaused, decodeSession, activeExecution);
    } else if (state.mode === 'cutaway_output') {
      this.els.floatingBadge.textContent = 'Cutaway: Output & Untied LM Head · 2,688 → 131,072';
      this.syncMathDock(state, activeCutaway, autoFollowManualPaused, decodeSession, activeExecution);
    } else {
      this.els.floatingBadge.textContent = `Cutaway: GQA Attention · Bay ${formatBayId(state.gqaBayId)} (32Q / 2KV, NoPE)`;
      this.syncMathDock(state, activeCutaway, autoFollowManualPaused, decodeSession, activeExecution);
    }

    // Sync active TOC & mini-diagram highlight
    const effectiveSection: TocSectionId =
      state.mode === 'cutaway_embedding'
        ? 'embedding'
        : state.mode === 'cutaway_mamba'
          ? 'mamba'
          : state.mode === 'cutaway_moe'
            ? 'moe'
            : state.mode === 'cutaway_gqa'
              ? 'gqa'
              : state.mode === 'cutaway_output'
                ? 'output'
                : state.section;

    for (const btn of this.els.tocButtons) {
      btn.classList.toggle('is-active', btn.dataset.section === effectiveSection);
    }
    this.els.sidebarCurrentSection.textContent = SECTION_SHORT_LABELS[effectiveSection];
    this.applySidebarCompactState(state.executionMode);
    for (const btn of this.els.presetButtons) {
      btn.classList.toggle('is-active', btn.dataset.preset === state.section);
    }
    for (const node of this.els.miniNodes) {
      node.classList.toggle('is-active', node.dataset.miniSection === effectiveSection);
    }

    // Execution mode toggle (Full Sequence vs Cached Incremental AR Decode)
    const isArDecode = state.executionMode === 'ar_decode';
    this.els.execModeFull.classList.toggle('is-active', !isArDecode);
    this.els.execModeFull.setAttribute('aria-checked', String(!isArDecode));
    this.els.execModeDecode.classList.toggle('is-active', isArDecode);
    this.els.execModeDecode.setAttribute('aria-checked', String(isArDecode));

    // Scrubber & precision (scrubber follows active decode sequence length in AR Decode mode)
    const tokenStr = String(state.tokens);
    if (this.els.tokenSlider.value !== tokenStr) {
      this.els.tokenSlider.value = tokenStr;
    }
    this.els.tokenValue.textContent = isArDecode
      ? `${state.tokens} / ${MAX_TOKENS} (AR)`
      : `${state.tokens} / ${MAX_TOKENS}`;
    this.els.tokenSlider.disabled = isArDecode;
    this.els.tokenDec.disabled = isArDecode || state.tokens <= MIN_TOKENS;
    this.els.tokenInc.disabled = isArDecode || state.tokens >= MAX_TOKENS;

    const isBf16 = state.precision === 'bf16';
    this.els.precisionBf16.classList.toggle('is-active', isBf16);
    this.els.precisionBf16.setAttribute('aria-checked', String(isBf16));
    this.els.precisionFp8.classList.toggle('is-active', !isBf16);
    this.els.precisionFp8.setAttribute('aria-checked', String(!isBf16));

    // Zoom bay button (hidden when disabled or no bay selected) & bay stepping row (Tower only)
    const isTower = state.mode === 'tower';
    const hasPinnedBay = isTower && state.selectedBay !== null;

    this.els.zoomBayBtn.hidden = !hasPinnedBay;
    this.els.zoomBayBtn.disabled = !hasPinnedBay;
    this.els.zoomBayBtn.textContent = hasPinnedBay
      ? `Zoom Bay ${formatBayId(state.selectedBay!)}`
      : 'Zoom Bay';

    this.els.bayPrev.hidden = !isTower;
    this.els.bayNext.hidden = !isTower;
    this.els.bayClear.hidden = !hasPinnedBay;
    this.els.bayClear.textContent = 'Clear';
    this.els.bayStepRow.hidden = !isTower;

    // Guided Tour header toggle button (V5 22-beat readout)
    const activeBeat = getWalkthroughBeat(state.tourBeat);
    this.els.tourToggleBtn.classList.toggle('is-active', state.tourActive);
    this.els.tourToggleBtn.textContent = state.tourActive
      ? `Beat ${activeBeat.id} · ${activeBeat.index + 1}/${WALKTHROUGH_BEAT_COUNT}`
      : 'Guided Tour';
    this.els.tourToggleBtn.title = state.tourActive
      ? `Guided Tour · Beat ${activeBeat.id} (Tour ${activeBeat.index + 1}/${WALKTHROUGH_BEAT_COUNT}) — click to toggle`
      : 'Start 22-beat guided walkthrough';

    // Memory Scaling chart trigger highlight (Beat 7.4 + active open state)
    const highlightChartBtn =
      state.tourActive && Boolean(activeBeat.highlightMemoryChart) && !state.memoryChartOpen;
    this.els.memoryChartBtn.classList.toggle('is-open', state.memoryChartOpen);
    this.els.memoryChartBtn.classList.toggle('is-highlighted', highlightChartBtn);
    this.els.memoryChartBtn.setAttribute('aria-expanded', String(state.memoryChartOpen));

    // Shortcuts popover
    this.els.shortcutsBtn.classList.toggle('is-active', state.shortcutsOpen);
    this.els.shortcutsBtn.setAttribute('aria-expanded', String(state.shortcutsOpen));
    this.els.shortcutsPopover.hidden = !state.shortcutsOpen;

    // Memory readout (with Part A #4 static weights vs dynamic sequence cache clarification!)
    const readoutKey = `${state.tokens}:${state.precision}`;
    if (readoutKey !== this.lastReadoutKey) {
      this.lastReadoutKey = readoutKey;
      const mem = memoryFor(this.spec, state.tokens, state.precision);
      const fmt = formatMemoryReadout(mem);

      this.els.memoryReadoutLines.innerHTML = '';
      const rows = [
        { text: fmt.mambaText, cls: 'readout-line readout-fixed' },
        { text: fmt.kvText, cls: 'readout-line readout-kv' },
        { text: fmt.referenceText, cls: 'readout-line readout-ref' },
        { text: WEIGHTS_SUMMARY_LINE, cls: 'readout-line readout-weights' },
      ];
      for (const r of rows) {
        const div = document.createElement('div');
        div.className = r.cls;
        div.textContent = r.text;
        this.els.memoryReadoutLines.appendChild(div);
      }
    }

    // Middle editorial panel
    const activeBayId = isTower ? state.hoveredBay ?? state.selectedBay : null;
    const activePart = !isTower ? state.hoveredPart ?? state.selectedPart : null;
    const editorialKey = computeEditorialCacheKey(
      state,
      activeCutaway,
      activeExecution,
      this.boundaryInspectToken,
      decodeSession,
      this.selectedCandidateTokenId,
    );

    if (editorialKey !== this.lastEditorialKey) {
      this.lastEditorialKey = editorialKey;
      if (state.tourActive) {
        this.renderWalkthroughPanel(state, activeCutaway, activeExecution, decodeSession);
      } else if (isTower) {
        this.renderTowerEditorial(
          state.section,
          activeBayId,
          state.selectedBay !== null,
          state.precision,
          state.tokens,
          activeExecution,
          decodeSession,
          state.spineWaveMode,
        );
      } else {
        this.renderCutawayEditorial(state, activePart, activeCutaway, activeExecution, decodeSession);
      }
    }

    // Keep live in-canvas Tower spine replay HUD synced smoothly (also callable directly during RAF)
    this.syncSpineWaveHud(state, activeExecution, decodeSession);

    // Keep inline beat playback progress bar smooth without rebuilding the editorial DOM
    if (state.tourActive && typeof this.els.editorialPanel.querySelector === 'function') {
      const beatFill = this.els.editorialPanel.querySelector<HTMLElement>('.tour-beat-fill');
      if (beatFill) {
        const sub = this.activeCutawayStepperSubProgress(state, activeBeat);
        beatFill.style.width = `${Math.round(Math.max(0, Math.min(1, sub)) * 100)}%`;
      }
    }
  }

  private ensureSpineWaveHudNodes(hud: HTMLDivElement): SpineWaveHudNodes {
    if (this.spineWaveHudNodes) return this.spineWaveHudNodes;
    hud.innerHTML = '';

    const head = document.createElement('div');
    head.className = 'spine-wave-hud-head';

    const dot = document.createElement('span');
    dot.className = 'spine-wave-hud-dot';
    dot.dataset.testid = 'spine-wave-hud-dot';
    dot.setAttribute('aria-hidden', 'true');

    const title = document.createElement('span');
    title.className = 'spine-wave-hud-title';
    title.dataset.testid = 'spine-wave-hud-title';

    const rms = document.createElement('span');
    rms.className = 'spine-wave-hud-rms';
    rms.dataset.testid = 'spine-wave-hud-rms';
    rms.hidden = true;

    const actions = document.createElement('div');
    actions.className = 'spine-wave-hud-actions';

    const pauseBtn = document.createElement('button');
    pauseBtn.type = 'button';
    pauseBtn.className = 'spine-wave-hud-btn';
    pauseBtn.dataset.testid = 'spine-wave-hud-pause';
    pauseBtn.hidden = true;
    this.on(pauseBtn, 'click', () => this.cb.onToggleSpineWavePause?.());

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'spine-wave-hud-btn';
    closeBtn.dataset.testid = 'spine-wave-hud-close';
    closeBtn.setAttribute('aria-label', 'Close spine replay HUD');
    closeBtn.title = 'Dismiss replay HUD and settle propagation wave';
    closeBtn.textContent = '×';
    this.on(closeBtn, 'click', () => this.cb.onDismissSpineWaveHud?.());

    actions.append(pauseBtn, closeBtn);
    head.append(dot, title, rms, actions);

    const detail = document.createElement('div');
    detail.className = 'spine-wave-hud-detail';
    detail.dataset.testid = 'spine-wave-hud-detail';

    const track = document.createElement('div');
    track.className = 'spine-wave-hud-track';
    const fill = document.createElement('div');
    fill.className = 'spine-wave-hud-fill';
    fill.dataset.testid = 'spine-wave-hud-fill';
    track.appendChild(fill);

    const foot = document.createElement('div');
    foot.className = 'spine-wave-hud-foot';

    const disclaimer = document.createElement('span');
    disclaimer.className = 'spine-wave-hud-disclaimer';
    disclaimer.dataset.testid = 'spine-wave-hud-disclaimer';

    const pct = document.createElement('span');
    pct.className = 'spine-wave-hud-pct';
    pct.dataset.testid = 'spine-wave-hud-pct';

    foot.append(disclaimer, pct);
    hud.append(head, detail, track, foot);

    this.spineWaveHudNodes = {
      dot,
      title,
      rms,
      pauseBtn,
      closeBtn,
      detail,
      fill,
      disclaimer,
      pct,
    };
    return this.spineWaveHudNodes;
  }

  /**
   * Update the floating in-canvas `.spine-wave-hud` directly (also callable from `main.ts`
   * during RAF animation frames without rebuilding the sidebar editorial DOM).
   */
  syncSpineWaveHud(
    state: SceneState,
    activeExecution: HybridExecutionResult | null = this.lastSyncedExecution,
    decodeSession: HybridDecodeSession | null = this.lastSyncedDecodeSession,
  ): void {
    const hud = this.els.spineWaveHud;
    if (!hud) return;

    const waveProgress = state.spineWaveProgress;
    if (state.mode !== 'tower' || waveProgress === null || !Number.isFinite(waveProgress)) {
      hud.hidden = true;
      return;
    }

    const activeStep = decodeSession ? getActiveDecodeStep(decodeSession) : null;
    const baySummaries = activeStep?.baySummaries ?? activeExecution?.baySummaries ?? null;
    const activeTokIdx = activeStep
      ? Math.max(0, activeStep.tokens - 1)
      : activeExecution
        ? Math.max(0, activeExecution.tokens - 1)
        : Math.max(0, state.tokens - 1);

    const stage = describeSpineWaveStage(
      waveProgress,
      this.spec,
      state.precision,
      baySummaries,
      activeTokIdx,
    );
    const nodes = this.ensureSpineWaveHudNodes(hud);

    const stageClass = `is-${stage.stageKind}`;
    for (const cls of SPINE_WAVE_STAGE_CLASSES) {
      hud.classList.remove(cls);
      nodes.dot.classList.remove(cls);
    }
    hud.classList.add(stageClass);
    nodes.dot.classList.add(stageClass);
    if (stage.stageKind === 'final_norm') {
      hud.classList.add('is-final-norm');
      nodes.dot.classList.add('is-final-norm');
    }
    hud.dataset.stageKind = stage.stageKind;
    hud.classList.toggle('is-paused', state.spineWavePaused);
    hud.classList.toggle('is-reduced-motion', state.reducedMotion);

    nodes.title.textContent = stage.title;
    nodes.detail.textContent = stage.actionDetail;
    if (stage.rmsText) {
      nodes.rms.textContent = stage.rmsText;
      nodes.rms.hidden = false;
    } else {
      nodes.rms.textContent = '';
      nodes.rms.hidden = true;
    }

    nodes.fill.style.width = `${stage.progressPct}%`;
    nodes.pct.textContent = `${stage.progressPct}%`;
    nodes.disclaimer.textContent = state.reducedMotion
      ? `${stage.disclaimer} · Static replay summary (reduced motion)`
      : stage.disclaimer;

    const showPause = state.spineWaveMode === 'instructional' && !state.reducedMotion;
    nodes.pauseBtn.hidden = !showPause;
    if (showPause) {
      nodes.pauseBtn.textContent = state.spineWavePaused ? '▶ Resume' : '❚❚ Pause';
      nodes.pauseBtn.classList.toggle('is-active', state.spineWavePaused);
      nodes.pauseBtn.setAttribute('aria-pressed', String(state.spineWavePaused));
      nodes.pauseBtn.title = state.spineWavePaused
        ? 'Resume slow instructional replay'
        : 'Pause slow instructional replay';
    } else {
      nodes.pauseBtn.textContent = '';
      nodes.pauseBtn.classList.remove('is-active');
      nodes.pauseBtn.setAttribute('aria-pressed', 'false');
    }

    hud.hidden = false;
  }

  private activeCutawayStepperPlaying(state: SceneState): boolean {
    if (state.mode === 'cutaway_mamba') {
      return state.mambaStepperActive && state.mambaStepperPlaying;
    }
    if (state.mode === 'cutaway_moe') {
      return state.moeStepperActive && state.moeStepperPlaying;
    }
    if (state.mode === 'cutaway_gqa') {
      return state.gqaStepperActive && state.gqaStepperPlaying;
    }
    return false;
  }

  private activeCutawayStepperSubProgress(
    state: SceneState,
    beat = getWalkthroughBeat(state.tourBeat),
  ): number {
    if (state.mode === 'cutaway_mamba' && state.mambaStepperActive) {
      return state.mambaStepperSubProgress;
    }
    if (state.mode === 'cutaway_moe' && state.moeStepperActive) {
      return state.moeStepperSubProgress;
    }
    if (state.mode === 'cutaway_gqa' && state.gqaStepperActive) {
      return state.gqaStepperSubProgress;
    }
    return beat.chapterBeatCount > 0 ? beat.beatInChapter / beat.chapterBeatCount : 0;
  }

  private syncMathDock(
    state: SceneState,
    cutaway: CutawaySceneGeometry | null,
    autoFollowManualPaused: boolean,
    decodeSession: HybridDecodeSession | null = null,
    activeExecution: HybridExecutionResult | null = null,
  ): void {
    const isEmbed = state.mode === 'cutaway_embedding';
    const isOutput = state.mode === 'cutaway_output';
    const isMamba = state.mode === 'cutaway_mamba';
    const isMoe = state.mode === 'cutaway_moe';
    const formula = isEmbed
      ? (cutaway?.embeddingFormula ?? cutaway?.activeFormula ?? null)
      : isOutput
        ? (cutaway?.outputFormula ?? cutaway?.activeFormula ?? null)
        : (cutaway?.activeFormula ?? null);
    const activeDecodeStep = decodeSession ? getActiveDecodeStep(decodeSession) : null;
    const isPinnedCell = isEmbed
      ? state.embeddingActiveCell !== null
      : isOutput
        ? state.outputActiveCell !== null
        : isMamba
          ? state.mambaSelectedCell !== null
          : isMoe
            ? state.moeSelectedCell !== null
            : state.gqaSelectedCell !== null;
    const steps = isEmbed
      ? EMBEDDING_STEPPER_STEPS
      : isOutput
        ? OUTPUT_STEPPER_STEPS
        : isMamba
          ? MAMBA_STEPPER_STEPS
          : isMoe
            ? MOE_STEPPER_STEPS
            : GQA_STEPPER_STEPS;

    // Update honest miniature simulator disclaimer note inside accessible popover & tooltip
    const gqaEpsLabel = formatEpsLabel(cutaway?.gqaSim?.rmsEps ?? 1e-5);
    const activeBayForTrace = isMamba ? state.mambaBayId : isMoe ? state.moeBayId : state.gqaBayId;
    const traceOriginTag =
      activeDecodeStep && !isEmbed && !isOutput
        ? (() => {
            const orig = activeDecodeStep.getLayerTraceOrigin(activeBayForTrace);
            return orig === 'reconstructed_from_checkpoint'
              ? ' · Reconstructed from checkpoint'
              : ' · Computed this step';
          })()
        : '';
    const modePrefix = activeDecodeStep
      ? `Cached AR ${activeDecodeStep.provenance.stepKind} (${activeDecodeStep.provenance.branchId}, pos ${activeDecodeStep.provenance.tokenRange[0]}..${activeDecodeStep.provenance.tokenRange[1]}${traceOriginTag})`
      : 'Unified 52-layer miniature';
    const noteText = isEmbed
      ? `${modePrefix} Embedding sim (d=8, V=32, untied W_embed ≠ W_lm_head, NoPE; synthetic Float32, rounded) · 131,072 × 2,688 = production metadata`
      : isOutput
        ? `${modePrefix} Output & Untied LM Head sim (d=8, V=32, ε=1e−5, untied W_lm_head ≠ W_embed; synthetic Float32, rounded) · 2,688 → 131,072 = production metadata`
        : isMamba
          ? `${modePrefix} Mamba-2 sim (d=8, 16H/8G, dₕ=4, dₛ=4, ε=1e−5; synthetic Float32, rounded) · BF16/FP8 = production metadata`
          : isMoe
            ? `${modePrefix} MoE sim (d=8, 128E top-6, d_exp=4, d_shared=8, ×2.5, ε=1e−5; synthetic Float32, rounded) · BF16/FP8/FP32 = production metadata`
            : `${modePrefix} GQA sim (d=8, 32Q/2KV, dₕ=4, NoPE, ε=${gqaEpsLabel}; synthetic Float32, rounded) · BF16/FP8 = production metadata`;
    this.els.gqaFormulaNote.textContent = noteText;
    this.els.gqaFormulaNoteBtn.title = noteText;

    const accum = cutaway?.focusedAccumulation;

    if (formula) {
      this.els.gqaFormulaTensor.textContent = formula.tensorLabel;
      this.els.gqaFormulaCoord.textContent = formula.coordLabel;
      if (accum?.active) {
        this.els.gqaFormulaHead.textContent = accum.badgeText;
        this.els.gqaFormulaHead.hidden = false;
        this.els.gqaFormulaHead.classList.add('is-accumulating');
        this.els.gqaFormulaValue.textContent = accum.valueBadgeText;
        this.els.gqaFormulaValue.classList.remove('is-masked');
        this.els.gqaFormulaValue.classList.add('is-accumulating');
        this.els.gqaFormulaValue.hidden = false;
        this.els.gqaCellClear.hidden = !isPinnedCell;
        this.els.gqaFormulaEq.textContent = accum.partialEquation;
        this.els.gqaFormulaSub.textContent = accum.partialSubEquation;
      } else {
        this.els.gqaFormulaHead.classList.remove('is-accumulating');
        if (formula.headBadge) {
          this.els.gqaFormulaHead.textContent = formula.headBadge;
          this.els.gqaFormulaHead.hidden = false;
        } else {
          this.els.gqaFormulaHead.textContent = '';
          this.els.gqaFormulaHead.hidden = true;
        }
        this.els.gqaFormulaValue.textContent = formula.isMasked
          ? formula.valueText
          : `= ${formula.valueText}`;
        this.els.gqaFormulaValue.classList.toggle('is-masked', formula.isMasked);
        this.els.gqaFormulaValue.classList.remove('is-accumulating');
        this.els.gqaFormulaValue.hidden = false;
        this.els.gqaCellClear.hidden = !isPinnedCell;
        this.els.gqaFormulaEq.textContent = formula.equation;
        this.els.gqaFormulaSub.textContent = formula.subEquation;
      }
    } else {
      // Idle state: keep head badge and numeric value badge completely empty & hidden (Part 0 #2)
      this.els.gqaFormulaHead.textContent = '';
      this.els.gqaFormulaHead.classList.remove('is-accumulating');
      this.els.gqaFormulaHead.hidden = true;
      this.els.gqaFormulaValue.textContent = '';
      this.els.gqaFormulaValue.classList.remove('is-masked');
      this.els.gqaFormulaValue.classList.remove('is-accumulating');
      this.els.gqaFormulaValue.hidden = true;
      this.els.gqaCellClear.hidden = true;

      if (isEmbed) {
        const tokLabels = activeDecodeStep?.newTokenLabels ?? activeExecution?.tokenLabels ?? [];
        const totalT = tokLabels.length || state.tokens;
        const localTok = Math.max(0, Math.min(totalT - 1, state.embeddingFocusToken));
        const absTok = (activeDecodeStep?.prefixTokens ?? 0) + localTok;
        const tokLbl = tokLabels[localTok] ?? `t${absTok}`;
        this.els.gqaFormulaTensor.textContent = 'Miniature Embedding (NoPE)';
        this.els.gqaFormulaCoord.textContent = `Focus: t${absTok} "${tokLbl}" · Untied 32×8 mini (131K×2,688 prod)`;
        this.els.gqaFormulaEq.textContent =
          'X₀[:, t] = W_embed[tokenId(t), :] — direct row lookup with zero positional embeddings (NoPE).';
        this.els.gqaFormulaSub.textContent =
          'Hover or click any cell in Input Tokens, W_embed, or X₀ to trace bit-exact hand-off into Bay 01 Mamba-2.';
      } else if (isOutput) {
        const tokLabels = activeDecodeStep?.newTokenLabels ?? activeExecution?.tokenLabels ?? [];
        const totalT = tokLabels.length || state.tokens;
        const localTok = Math.max(0, Math.min(totalT - 1, state.outputFocusToken));
        const absTok = (activeDecodeStep?.prefixTokens ?? 0) + localTok;
        const tokLbl = tokLabels[localTok] ?? `t${absTok}`;
        this.els.gqaFormulaTensor.textContent = 'Miniature Output & Untied LM Head';
        this.els.gqaFormulaCoord.textContent = `Focus: t${absTok} "${tokLbl}" · ε=1e−5 · V=32 mini (131K prod)`;
        this.els.gqaFormulaEq.textContent =
          'Bay 52 Y → Final RMSNorm (ε=1e−5) → Untied W_lm_head (≠ W_embed) → Logits → Softmax → Top-5 Next Tokens.';
        this.els.gqaFormulaSub.textContent =
          'Hover or click any cell in Bay 52 Y, Final Norm, W_lm_head, Logits, or Probs to inspect next-token probabilities.';
      } else if (isMamba) {
        const sim = cutaway?.mambaSim;
        const simT = sim?.tokens ?? state.tokens;
        const activeH = cutaway?.mambaStats?.activeHead ?? state.mambaActiveHead;
        const activeG = mambaGroupForHead(activeH);
        const activeT = Math.min(
          simT - 1,
          cutaway?.mambaStats?.stateTokenIndex ?? state.mambaStateToken ?? Math.max(0, simT - 1),
        );
        const absT = (sim?.startTokenOffset ?? 0) + activeT;
        const tokLbl = sim?.tokenLabels[activeT] ?? `t${absT}`;
        const zeroPads = Math.max(0, 3 - absT);
        this.els.gqaFormulaTensor.textContent = `Miniature Mamba-2 · Bay ${formatBayId(state.mambaBayId)}`;
        this.els.gqaFormulaCoord.textContent = `Active: ${formatMambaHeadId(activeH)} (G${activeG}) · t${absT} "${tokLbl}"`;
        this.els.gqaFormulaEq.textContent =
          sim?.hasInitialState
            ? `Cached AR step at t${absT} "${tokLbl}": carried Hₜ₋₁ & 4-tap Conv1D update in place inside fixed 69 KiB mini / 47.1 MiB prod state.`
            : zeroPads > 0
              ? `Recurrence at t${absT} "${tokLbl}": Conv1D left-pads ${zeroPads} tap${zeroPads === 1 ? '' : 's'} with 0; Hₜ updates in place inside fixed 47.1 MiB state.`
              : 'Hover or click any matrix cell for rounded numerical explanations, step Recurrence t in the sidebar, or press Play.';
        const tokPreview = sim?.tokenLabels
          ? sim.tokenLabels.map((tok, i) => `${(sim.startTokenOffset ?? 0) + i}:"${tok}"`).join(' ')
          : '';
        this.els.gqaFormulaSub.textContent = tokPreview
          ? sim?.hasInitialState
            ? `Incremental step token (S=${simT}, carried prefix P=${sim.startTokenOffset}): ${tokPreview}`
            : `Educational token labels (T=${simT}): ${tokPreview}`
          : '';
      } else if (isMoe) {
        const sim = cutaway?.moeSim;
        const simT = sim?.tokens ?? state.tokens;
        const fTok = Math.min(simT - 1, cutaway?.moeStats?.focusToken ?? Math.max(0, simT - 1));
        const absTok = (sim?.startTokenOffset ?? 0) + fTok;
        const fExp = cutaway?.moeStats?.focusExpert ?? cutaway?.moeStats?.activeExpertIds[0] ?? 0;
        const tokLabel = sim?.tokenLabels[fTok] ?? `t${absTok}`;
        this.els.gqaFormulaTensor.textContent = `Miniature MoE 128+1 · Bay ${formatBayId(state.moeBayId)}`;
        this.els.gqaFormulaCoord.textContent = `Focus: t${absTok} "${tokLabel}" · ${formatExpertId(fExp)}`;
        this.els.gqaFormulaEq.textContent =
          'Hover or click any router tile, Top-6 slot, routed ReLU² cell, or 2× shared cell, or press Play to step through parallel MoE.';
        const topList = cutaway?.moeStats?.activeExpertIds?.map(formatExpertId).join(', ') ?? '';
        this.els.gqaFormulaSub.textContent = topList
          ? `Top-6 active for t${absTok} "${tokLabel}": ${topList} (122 idle = 0 contrib) · 0 B token cache`
          : '';
      } else {
        const sim = cutaway?.gqaSim;
        const g0 = formatQueryHeadId(cutaway?.gqaStats?.activeGroup0Head ?? state.gqaActiveGroup0Head);
        const g1 = formatQueryHeadId(cutaway?.gqaStats?.activeGroup1Head ?? state.gqaActiveGroup1Head);
        this.els.gqaFormulaTensor.textContent = `Miniature GQA · Bay ${formatBayId(state.gqaBayId)}`;
        this.els.gqaFormulaCoord.textContent =
          sim && sim.prefixTokens > 0
            ? `Rect [${sim.tokens}×${sim.keyTokens}] · ${g0}→KV₀ · ${g1}→KV₁`
            : `Displayed: ${g0}→KV₀ · ${g1}→KV₁`;
        this.els.gqaFormulaEq.textContent =
          sim && sim.prefixTokens > 0
            ? `Cached AR step: ${sim.tokens} query row attends ${sim.prefixTokens} reused prefix KV slots + ${sim.tokens} new KV slot (no prefix recompute).`
            : 'Hover or click any matrix cell for rounded numerical explanations, or press Play to step through causal attention.';
        const tokPreview = sim?.keyTokenLabels
          ? sim.keyTokenLabels
              .map((tok, i) => (sim.prefixTokens > 0 && i >= sim.prefixTokens ? `[new ${i}:"${tok}"]` : `${i}:"${tok}"`))
              .join(' ')
          : '';
        this.els.gqaFormulaSub.textContent = tokPreview
          ? sim && sim.prefixTokens > 0
            ? `KV context (${sim.prefixTokens} cached + ${sim.tokens} new): ${tokPreview}`
            : `Educational token labels (T=${state.tokens}): ${tokPreview}`
          : '';
      }
    }

    const activeTourBeat = state.tourActive ? getWalkthroughBeat(state.tourBeat) : null;
    const bookendTourStep =
      (isEmbed || isOutput) && activeTourBeat && activeTourBeat.mode === state.mode
        ? (activeTourBeat.stepperStep ?? null)
        : null;
    const stepperActive = isEmbed || isOutput
      ? bookendTourStep !== null
      : isMamba
        ? state.mambaStepperActive
        : isMoe
          ? state.moeStepperActive
          : state.gqaStepperActive;
    const stepperPlaying = isEmbed || isOutput
      ? false
      : isMamba
        ? state.mambaStepperPlaying
        : isMoe
          ? state.moeStepperPlaying
          : state.gqaStepperPlaying;
    const stepperStep = isEmbed || isOutput
      ? (bookendTourStep ?? 0)
      : isMamba
        ? state.mambaStepperStep
        : isMoe
          ? state.moeStepperStep
          : state.gqaStepperStep;
    const stepperSub = isEmbed || isOutput
      ? 0
      : isMamba
        ? state.mambaStepperSubProgress
        : isMoe
          ? state.moeStepperSubProgress
          : state.gqaStepperSubProgress;

    const playing = stepperActive && stepperPlaying;
    this.els.gqaStepPlay.textContent = playing ? '❚❚ Pause' : '▶ Play';
    this.els.gqaStepPlay.classList.toggle('is-playing', playing);
    this.els.gqaStepPlay.setAttribute('aria-pressed', String(playing));

    const autoFollowOn = state.stepperAutoFollow;
    const showManualPaused = autoFollowOn && stepperActive && autoFollowManualPaused;
    this.els.gqaStepAutoFollow.classList.toggle('is-active', autoFollowOn && !showManualPaused);
    this.els.gqaStepAutoFollow.classList.toggle('is-paused', showManualPaused);
    this.els.gqaStepAutoFollow.setAttribute('aria-pressed', String(autoFollowOn && !showManualPaused));
    this.els.gqaStepAutoFollow.title = showManualPaused
      ? 'Manual camera active — click to resume Auto-Follow on current math operation'
      : autoFollowOn
        ? 'Auto-Follow ON: camera smoothly focuses each math step (manual orbit/pan/zoom always overrides)'
        : 'Auto-Follow OFF: click to automatically focus camera on each math step';

    for (let i = 0; i < this.els.gqaStepPills.length; i++) {
      const btn = this.els.gqaStepPills[i];
      const stepDef = steps[i];
      if (stepDef) {
        btn.hidden = false;
        btn.textContent = stepDef.shortLabel;
        btn.title = stepDef.summary;
        const isAct = stepperActive && i === stepperStep;
        btn.classList.toggle('is-active', isAct);
        btn.setAttribute('aria-selected', String(isAct));
      } else {
        btn.hidden = true;
        btn.classList.remove('is-active');
        btn.setAttribute('aria-selected', 'false');
      }
    }

    const pct = String(Math.round(stepperSub * 100));
    if (this.els.gqaStepScrub.value !== pct) {
      this.els.gqaStepScrub.value = pct;
    }
  }

  private renderWalkthroughPanel(
    state: SceneState,
    cutaway: CutawaySceneGeometry | null = null,
    activeExecution: HybridExecutionResult | null = null,
    decodeSession: HybridDecodeSession | null = null,
  ): void {
    const beat = getWalkthroughBeat(state.tourBeat);
    const chapter = getWalkthroughChapter(beat.chapterIndex);
    const chapterBeats = beatsForChapter(beat.chapterIndex);
    const actRoman = beat.act === 1 ? 'Act I' : beat.act === 2 ? 'Act II' : 'Act III';

    const panel = this.els.editorialPanel;
    panel.classList.add('is-tour-mode');
    panel.innerHTML = '';

    // 1. Top < Act I/II/III · Beat X.Y (N / 22) > navigation bar
    const navBar = document.createElement('div');
    navBar.className = 'tour-nav-bar';

    const prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.className = 'tour-arrow-btn';
    prevBtn.dataset.testid = 'tour-beat-prev';
    prevBtn.setAttribute('aria-label', 'Previous beat');
    prevBtn.textContent = '‹';
    prevBtn.disabled = beat.index <= 0;
    prevBtn.addEventListener('click', () => {
      if (this.cb.onTourBeatPrev) this.cb.onTourBeatPrev();
      else this.cb.onTourPrev();
    });

    const counter = document.createElement('span');
    counter.className = 'tour-counter';
    counter.setAttribute('data-testid', 'tour-chapter-counter');
    counter.dataset.beatId = beat.id;
    counter.textContent = `${actRoman} · Beat ${beat.id} (${beat.index + 1} / ${WALKTHROUGH_BEAT_COUNT})`;
    counter.title = `${beat.actLabel} · ${chapter.shortTitle}`;

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'tour-arrow-btn';
    nextBtn.dataset.testid = 'tour-beat-next';
    nextBtn.setAttribute('aria-label', 'Next beat');
    nextBtn.textContent = '›';
    nextBtn.disabled = beat.index >= WALKTHROUGH_BEAT_COUNT - 1;
    nextBtn.addEventListener('click', () => {
      if (this.cb.onTourBeatNext) this.cb.onTourBeatNext();
      else this.cb.onTourNext();
    });

    navBar.append(prevBtn, counter, nextBtn);

    // Chapter & Act subtitle
    const subRow = document.createElement('div');
    subRow.className = 'ed-sub';
    subRow.dataset.testid = 'tour-chapter-subtitle';
    subRow.textContent = `${beat.actLabel} · ${chapter.shortTitle}`;

    // 2. Non-destructive "Exploring freely · Resume Beat X.Y" pill (keeps narrative intact!)
    let exploringPill: HTMLDivElement | null = null;
    if (state.tourExploring) {
      exploringPill = document.createElement('div');
      exploringPill.className = 'tour-exploring-pill';
      exploringPill.dataset.testid = 'tour-exploring-pill';

      const expLabel = document.createElement('span');
      expLabel.textContent = 'Exploring freely';

      const resumeBtn = document.createElement('button');
      resumeBtn.type = 'button';
      resumeBtn.className = 'tour-exploring-resume';
      resumeBtn.dataset.testid = 'tour-resume-beat';
      resumeBtn.textContent = `Resume Beat ${beat.id}`;
      resumeBtn.title = `Restore camera and stage focus to Beat ${beat.id} (${beat.title})`;
      resumeBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.cb.onTourResumeBeat?.();
      });

      exploringPill.addEventListener('click', () => {
        this.cb.onTourResumeBeat?.();
      });
      exploringPill.append(expLabel, resumeBtn);
    }

    // 3. Body row: slim left 22-beat progress rail + scrollable narrative column
    const bodyRow = document.createElement('div');
    bodyRow.style.display = 'flex';
    bodyRow.style.alignItems = 'stretch';
    bodyRow.style.gap = '9px';
    bodyRow.style.flex = '1 1 auto';
    bodyRow.style.minHeight = '0';

    const progressRail = document.createElement('div');
    progressRail.className = 'tour-progress-rail';
    progressRail.dataset.testid = 'tour-progress-rail';
    progressRail.setAttribute('role', 'navigation');
    progressRail.setAttribute('aria-label', '22-beat guided walkthrough progress');

    const railFill = document.createElement('div');
    railFill.className = 'tour-progress-rail-fill';
    const fillPct =
      WALKTHROUGH_BEAT_COUNT > 1
        ? (beat.index / (WALKTHROUGH_BEAT_COUNT - 1)) * 100
        : 100;
    railFill.style.height = `${fillPct.toFixed(1)}%`;
    progressRail.appendChild(railFill);

    for (let i = 0; i < WALKTHROUGH_BEAT_COUNT; i++) {
      const b = WALKTHROUGH_BEATS[i];
      const topPct =
        WALKTHROUGH_BEAT_COUNT > 1 ? (i / (WALKTHROUGH_BEAT_COUNT - 1)) * 100 : 0;
      const pip = document.createElement('button');
      pip.type = 'button';
      pip.dataset.tourBeat = String(i);
      pip.setAttribute('aria-label', `Jump to Beat ${b.id}: ${b.title}`);
      pip.title = `Beat ${b.id} (${i + 1}/${WALKTHROUGH_BEAT_COUNT}) · ${b.title}`;

      if (i === beat.index) {
        pip.className = 'tour-progress-rail-dot';
        pip.style.top = `${topPct.toFixed(2)}%`;
        pip.style.padding = '0';
        pip.style.cursor = 'pointer';
        pip.setAttribute('aria-current', 'step');
      } else {
        pip.className = 'tour-progress-rail-pip';
        pip.style.position = 'absolute';
        pip.style.left = '50%';
        pip.style.top = `${topPct.toFixed(2)}%`;
        pip.style.transform = 'translate(-50%, -50%)';
        pip.style.width = '10px';
        pip.style.height = '8px';
        pip.style.padding = '0';
        pip.style.border = '0';
        pip.style.background = 'transparent';
        pip.style.cursor = 'pointer';
        pip.style.display = 'flex';
        pip.style.alignItems = 'center';
        pip.style.justifyContent = 'center';
        const dotInner = document.createElement('span');
        dotInner.style.display = 'block';
        dotInner.style.width = '4px';
        dotInner.style.height = '4px';
        dotInner.style.borderRadius = '999px';
        dotInner.style.backgroundColor = i < beat.index ? '#93A5F6' : '#CBD5E1';
        pip.appendChild(dotInner);
      }

      pip.addEventListener('click', () => {
        this.cb.onTourJumpBeat?.(i);
      });
      progressRail.appendChild(pip);
    }

    const liveDecodeSummary: LiveDecodeBeatSummary | null =
      beat.executionMode === 'ar_decode' && decodeSession
        ? (() => {
            const branch = getActiveDecodeBranch(decodeSession);
            const step = getActiveDecodeStep(decodeSession);
            const lastPos = Math.max(0, step.totalTokens - 1);
            const lastLabel =
              step.sequenceTokenLabels[lastPos] ??
              step.newTokenLabels[step.newTokenLabels.length - 1] ??
              `t${lastPos}`;
            return {
              stepIndex: decodeSession.activeStepIndex,
              totalSteps: branch.steps.length,
              branchId: decodeSession.activeBranchId,
              prefixTokens: step.prefixTokens,
              newTokens: step.tokens,
              totalTokens: step.totalTokens,
              lastTokenPos: lastPos,
              lastTokenLabel: lastLabel,
              greedyNextLabel: step.greedyNextToken.label,
            };
          })()
        : null;
    const displayCopy = resolveBeatDisplayCopy(beat, liveDecodeSummary);

    const scrollBody = document.createElement('div');
    scrollBody.className = 'tour-scroll-body';

    const header = document.createElement('div');
    header.className = 'ed-header tour-ed-header';

    const title = document.createElement('h2');
    title.className = 'ed-title';
    title.textContent = `${beat.id} ${beat.title}`;

    const badge = document.createElement('span');
    badge.className = 'ed-badge';
    if (beat.precision === 'fp8_selective') badge.classList.add('ed-badge-gold');
    badge.textContent = displayCopy.badge;

    header.append(title, badge);
    scrollBody.appendChild(header);

    // 4. Inline Beat Playback Strip on cutaway beats with stepperStep / stepperAutoPlay
    if (
      beat.mode !== 'tower' &&
      (beat.stepperStep !== undefined || beat.stepperAutoPlay)
    ) {
      const beatBar = document.createElement('div');
      beatBar.className = 'tour-beat-bar';
      beatBar.dataset.testid = 'tour-beat-bar';

      const refitBtn = document.createElement('button');
      refitBtn.type = 'button';
      refitBtn.className = 'tour-beat-btn';
      refitBtn.dataset.testid = 'tour-beat-refit';
      refitBtn.textContent = '↓ Refit';
      refitBtn.title = `Refit camera onto Beat ${beat.id}'s active stage`;
      refitBtn.addEventListener('click', () => {
        if (this.cb.onTourRefitBeat) this.cb.onTourRefitBeat();
        else this.cb.onFitAll();
      });

      const isPlaying = this.activeCutawayStepperPlaying(state);
      const playBtn = document.createElement('button');
      playBtn.type = 'button';
      playBtn.className = isPlaying ? 'tour-beat-btn is-playing' : 'tour-beat-btn';
      playBtn.dataset.testid = 'tour-beat-play';
      playBtn.textContent = isPlaying ? '❚❚ Pause' : '▶ Play';
      playBtn.setAttribute('aria-pressed', String(isPlaying));
      playBtn.title = isPlaying ? 'Pause beat stepper animation' : 'Play beat stepper animation';
      playBtn.addEventListener('click', () => {
        if (this.cb.onTourToggleBeatPlay) this.cb.onTourToggleBeatPlay();
        else this.cb.onGqaStepTogglePlay();
      });

      const beatTrack = document.createElement('div');
      beatTrack.className = 'tour-beat-track';
      const beatFill = document.createElement('div');
      beatFill.className = 'tour-beat-fill';
      const initSub = this.activeCutawayStepperSubProgress(state, beat);
      beatFill.style.width = `${Math.round(Math.max(0, Math.min(1, initSub)) * 100)}%`;
      beatTrack.appendChild(beatFill);

      const dotsWrap = document.createElement('div');
      dotsWrap.className = 'tour-beat-dots';
      dotsWrap.setAttribute('aria-label', 'Chapter beats');
      for (const chBeat of chapterBeats) {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className =
          chBeat.index === beat.index
            ? 'tour-beat-dot is-active'
            : chBeat.index < beat.index
              ? 'tour-beat-dot is-done'
              : 'tour-beat-dot';
        dot.dataset.tourBeat = String(chBeat.index);
        dot.style.border = '0';
        dot.style.padding = '0';
        dot.style.cursor = 'pointer';
        dot.setAttribute('aria-label', `Beat ${chBeat.id}: ${chBeat.title}`);
        dot.title = `Beat ${chBeat.id}: ${chBeat.title}`;
        dot.addEventListener('click', () => {
          this.cb.onTourJumpBeat?.(chBeat.index);
        });
        dotsWrap.appendChild(dot);
      }

      beatBar.append(refitBtn, playBtn, beatTrack, dotsWrap);
      scrollBody.appendChild(beatBar);
    }

    for (const para of displayCopy.paragraphs) {
      const p = document.createElement('p');
      p.className = 'ed-body';
      p.textContent = para;
      scrollBody.appendChild(p);
    }

    const list = document.createElement('ul');
    list.className = 'ed-specs';
    for (const item of displayCopy.takeaways) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }
    scrollBody.appendChild(list);

    // 5. Embedded live AR Decode controls inside Act III (Beats 7.2–7.4) without exiting tour
    if (beat.executionMode === 'ar_decode' && decodeSession) {
      const arCard = this.buildArDecodeSessionCard(
        decodeSession,
        state.precision,
        state.mode,
        state.spineWaveMode,
      );
      arCard.style.marginTop = '7px';
      scrollBody.appendChild(arCard);
    }

    // 6. Interactive Space cue hint
    if (beat.interactivePrompt) {
      const hint = document.createElement('div');
      hint.className = 'tour-space-hint';
      hint.dataset.testid = 'tour-space-hint';
      hint.style.whiteSpace = 'normal';
      hint.style.marginTop = '6px';
      hint.textContent = beat.interactivePrompt;
      scrollBody.appendChild(hint);
    }

    bodyRow.append(progressRail, scrollBody);

    // 7. Bottom Continue -> / Exit Tour bar (always pinned visible!)
    const actions = document.createElement('div');
    actions.className = 'ed-actions tour-actions';

    const isLast = beat.index >= WALKTHROUGH_BEAT_COUNT - 1;
    if (!isLast) {
      const contBtn = document.createElement('button');
      contBtn.type = 'button';
      contBtn.id = 'tour-continue-btn';
      contBtn.className = 'ed-cta-btn tour-continue-btn';
      contBtn.textContent = 'Continue →';
      contBtn.addEventListener('click', () => {
        if (this.cb.onTourBeatNext) this.cb.onTourBeatNext();
        else this.cb.onTourNext();
      });
      actions.appendChild(contBtn);
    } else {
      const chartBtn = document.createElement('button');
      chartBtn.type = 'button';
      chartBtn.id = 'tour-chart-btn';
      chartBtn.className = 'ed-cta-btn tour-continue-btn';
      chartBtn.textContent = 'Memory Scaling 1 → 1M ↗';
      chartBtn.addEventListener('click', () => this.cb.onToggleMemoryChart());
      actions.appendChild(chartBtn);

      const restartBtn = document.createElement('button');
      restartBtn.type = 'button';
      restartBtn.className = 'ed-cta-btn ed-cta-secondary';
      restartBtn.textContent = 'Restart';
      restartBtn.addEventListener('click', () => this.cb.onTourRestart());
      actions.appendChild(restartBtn);
    }

    const exitBtn = document.createElement('button');
    exitBtn.type = 'button';
    exitBtn.id = 'tour-exit-btn';
    exitBtn.className = 'ed-cta-btn ed-cta-secondary';
    exitBtn.textContent = isLast ? 'Free Explore' : 'Exit Tour';
    exitBtn.addEventListener('click', () => this.cb.onTourExit());
    actions.appendChild(exitBtn);

    const breadcrumb =
      state.mode !== 'tower'
        ? this.buildCrossBayBreadcrumb(state, cutaway, activeExecution, decodeSession)
        : null;

    panel.append(navBar, subRow);
    if (breadcrumb) panel.appendChild(breadcrumb);
    if (exploringPill) panel.appendChild(exploringPill);
    panel.append(bodyRow, actions);
  }

  private setBoundaryInspectToken(tok: number): void {
    this.boundaryInspectToken = tok;
    if (this.lastSyncedState) {
      this.sync(
        this.lastSyncedState,
        this.lastSyncedCutaway,
        false,
        this.lastSyncedExecution,
        this.lastSyncedDecodeSession,
      );
    }
  }

  private setSelectedCandidateToken(tokenId: number): void {
    this.selectedCandidateTokenId = tokenId;
    if (this.lastSyncedState) {
      this.sync(
        this.lastSyncedState,
        this.lastSyncedCutaway,
        false,
        this.lastSyncedExecution,
        this.lastSyncedDecodeSession,
      );
    }
  }

  /**
   * Lightweight interactive Cached Incremental AR Decode session panel (Gate B).
   * Explicitly separates next-token prediction selection from inserting & executing the next step,
   * contrasts growing GQA KV vs. fixed Mamba state vs. stateless MoE, and carries synthetic disclaimers.
   */
  private buildArDecodeSessionCard(
    session: HybridDecodeSession,
    precision: PrecisionMode,
    mode: ViewMode,
    spineWaveMode: SpineWaveMode = this.lastSyncedState?.spineWaveMode ?? 'fast',
  ): HTMLDivElement {
    const branch = getActiveDecodeBranch(session);
    const step = getActiveDecodeStep(session);
    const prov = step.provenance;
    const acct = step.cacheAccounting;
    const isFp8 = precision === 'fp8_selective';
    const isHistoricalStep = session.activeStepIndex < branch.steps.length - 1;
    const canStepMore = step.totalTokens < session.maxTokens;

    const card = document.createElement('div');
    card.className = 'ar-decode-card';
    card.dataset.testid = 'ar-decode-panel';

    // 1. Header with step kind, provenance & branch tag
    const head = document.createElement('div');
    head.className = 'ar-decode-head';
    const title = document.createElement('span');
    title.className = 'ar-decode-title';
    const rangeStr =
      prov.tokenRange[0] === prov.tokenRange[1]
        ? `t${prov.tokenRange[0]}`
        : `t${prov.tokenRange[0]}..t${prov.tokenRange[1]}`;
    title.textContent = `AR Decode · Step ${session.activeStepIndex + 1}/${branch.steps.length} (${prov.stepKind} ${rangeStr})`;
    const branchPill = document.createElement('span');
    branchPill.className = 'ed-flow-tag ar-branch-tag';
    branchPill.textContent = prov.branchId;
    branchPill.title = `Step ID: ${prov.stepExecutionId}\nState digest: ${prov.nextStateDigest}`;
    head.append(title, branchPill);
    card.appendChild(head);

    // Optional branch switcher when the user has forked alternative continuations
    if (session.branches.size > 1) {
      const brRow = document.createElement('div');
      brRow.className = 'ar-branch-strip';
      brRow.dataset.testid = 'ar-branch-strip';
      const brLbl = document.createElement('span');
      brLbl.className = 'ar-mini-label';
      brLbl.textContent = 'Branches:';
      brRow.appendChild(brLbl);
      for (const [brId, brObj] of session.branches.entries()) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `ed-flow-tok-btn${brId === session.activeBranchId ? ' is-active' : ''}`;
        btn.dataset.decodeBranch = brId;
        const tipStep = brObj.steps[brObj.steps.length - 1];
        const tipWord = tipStep.newTokenLabels[tipStep.newTokenLabels.length - 1] ?? '';
        btn.textContent = `${brId} (${brObj.steps.length}s · "${tipWord}")`;
        btn.addEventListener('click', () => this.cb.onDecodeSwitchBranch(brId));
        brRow.appendChild(btn);
      }
      card.appendChild(brRow);
    }

    // 2. Cumulative sequence strip distinguishing cached prefix vs newly processed token(s)
    const seqStrip = document.createElement('div');
    seqStrip.className = 'ar-seq-strip';
    seqStrip.dataset.testid = 'ar-seq-strip';
    for (let i = 0; i < step.sequenceTokenLabels.length; i++) {
      const chip = document.createElement('span');
      const isNew = i >= step.prefixTokens;
      chip.className = isNew ? 'ar-seq-chip is-new' : 'ar-seq-chip is-cached';
      chip.textContent = `t${i} ${step.sequenceTokenLabels[i]}`;
      chip.title = isNew
        ? `Processed in this step (${prov.stepKind}) · vocab #${step.sequenceTokenIds[i]}`
        : `Reused from cached prefix (0 KV recompute) · vocab #${step.sequenceTokenIds[i]}`;
      seqStrip.appendChild(chip);
    }
    card.appendChild(seqStrip);

    // 3. 3-Kind Cache State Comparison (Growing GQA KV vs Fixed Mamba vs Stateless MoE)
    const cacheGrid = document.createElement('div');
    cacheGrid.className = 'ar-cache-grid';
    cacheGrid.dataset.testid = 'ar-cache-grid';

    const makeCacheCell = (label: string, val: string, sub: string, cls: string, tooltip: string) => {
      const el = document.createElement('div');
      el.className = `ar-cache-cell ${cls}`;
      el.title = tooltip;
      const l = document.createElement('span');
      l.className = 'ar-cache-k';
      l.textContent = label;
      const v = document.createElement('span');
      v.className = 'ar-cache-v';
      v.textContent = val;
      const s = document.createElement('span');
      s.className = 'ar-cache-s';
      s.textContent = sub;
      el.append(l, v, s);
      return el;
    };

    cacheGrid.append(
      makeCacheCell(
        'GQA KV (×6)',
        `${step.totalTokens}/${session.maxTokens} · ${(acct.gqaPopulatedBytes / 1024).toFixed(2)} KiB`,
        `prod +${isFp8 ? '3' : '6'} KiB/t`,
        'is-gqa',
        `Grows +${acct.bytesPerTokenGqaPopulated} B/token in mini sim · Production: +${isFp8 ? '3' : '6'} KiB/token (${isFp8 ? 'FP8' : 'BF16'})`,
      ),
      makeCacheCell(
        'Mamba (×23)',
        `FIXED ${(acct.mambaFixedBytes / 1024).toFixed(0)} KiB`,
        'prod 47.1 MiB',
        'is-mamba',
        'All 23 Mamba-2 bays update in place — 69 KiB miniature / 47.1 MiB production fixed from 1 to 1M tokens',
      ),
      makeCacheCell(
        'MoE (×23)',
        '0 B cache',
        'stateless',
        'is-moe',
        'All 23 MoE 128+1 bays are strictly per-token stateless (0 B dynamic sequence cache)',
      ),
    );
    card.appendChild(cacheGrid);

    // 4. History Navigation & Reset
    const navRow = document.createElement('div');
    navRow.className = 'ar-nav-row';

    const prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.className = 'mtemp-btn';
    prevBtn.dataset.testid = 'ar-step-prev';
    prevBtn.textContent = '◀ Prev';
    prevBtn.disabled = session.activeStepIndex <= 0;
    prevBtn.addEventListener('click', () => this.cb.onDecodeNavStep(session.activeStepIndex - 1));

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'mtemp-btn';
    nextBtn.dataset.testid = 'ar-step-next';
    nextBtn.textContent = 'Next ▶';
    nextBtn.disabled = session.activeStepIndex >= branch.steps.length - 1;
    nextBtn.addEventListener('click', () => this.cb.onDecodeNavStep(session.activeStepIndex + 1));

    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'mtemp-btn ar-reset-btn';
    resetBtn.dataset.testid = 'ar-decode-reset';
    resetBtn.textContent = 'Reset';
    resetBtn.title = 'Reset cached AR decode session back to 4-token prefill';
    resetBtn.addEventListener('click', () => this.cb.onDecodeReset());

    const snapInfo = document.createElement('span');
    snapInfo.className = 'ar-snap-meta';
    snapInfo.textContent = `${(acct.totalAllocatedBytes / 1024).toFixed(0)} KiB snap · ${step.prefixTokens}P+${step.tokens}S`;
    snapInfo.title = `Allocated miniature snapshot: ${acct.totalAllocatedBytes.toLocaleString()} B (75 KiB) · ${step.prefixTokens} reused prefix + ${step.tokens} new`;

    navRow.append(prevBtn, nextBtn, resetBtn, snapInfo);
    card.appendChild(navRow);

    // 5. Next-Token Prediction (Explicitly Separated from Subsequent Insertion & Processing!)
    const preds = step.predictedNextTokens;
    const stagedPred =
      preds.find((p) => p.tokenId === this.selectedCandidateTokenId) ?? preds[0];
    const maxProb = Math.max(0.05, preds[0]?.prob ?? 0.25);

    const predHead = document.createElement('div');
    predHead.className = 'ar-pred-head';
    predHead.textContent = `Predicted Next Token at t${prov.tokenRange[1]} (Not Yet Inserted):`;
    card.appendChild(predHead);

    const candList = document.createElement('div');
    candList.className = 'ar-cand-list';
    candList.dataset.testid = 'ar-cand-list';
    for (let i = 0; i < preds.length; i++) {
      const pred = preds[i];
      const isSelected = pred.tokenId === stagedPred.tokenId;
      const row = document.createElement('button');
      row.type = 'button';
      row.className = `ar-cand-row${isSelected ? ' is-selected' : ''}`;
      row.dataset.decodeCandidate = String(pred.tokenId);
      row.setAttribute('aria-pressed', String(isSelected));
      row.title = `Select synthetic candidate "${pred.label}" (vocab #${pred.tokenId}) for insertion`;
      row.addEventListener('click', () => this.setSelectedCandidateToken(pred.tokenId));

      const rankEl = document.createElement('span');
      rankEl.className = i === 0 ? 'ar-cand-rank is-greedy' : 'ar-cand-rank';
      rankEl.textContent = i === 0 ? '★1' : `#${i + 1}`;

      const word = document.createElement('span');
      word.className = 'ed-flow-pred-word';
      word.textContent = pred.label;

      const track = document.createElement('div');
      track.className = 'ed-flow-pred-track';
      const bar = document.createElement('div');
      bar.className = 'ed-flow-pred-bar';
      bar.style.width = `${Math.max(4, Math.min(100, (pred.prob / maxProb) * 100)).toFixed(1)}%`;
      track.appendChild(bar);

      const prob = document.createElement('span');
      prob.className = 'ed-flow-pred-prob';
      prob.textContent = `${(pred.prob * 100).toFixed(1)}%`;

      row.append(rankEl, word, track, prob);
      candList.appendChild(row);
    }
    card.appendChild(candList);

    // Explicit Separate Insertion & Execution Buttons
    const actRow = document.createElement('div');
    actRow.className = 'ar-actions-row';

    const insertBtn = document.createElement('button');
    insertBtn.type = 'button';
    insertBtn.className = 'ed-cta-btn ar-insert-btn';
    insertBtn.dataset.testid = 'ar-insert-selected';
    insertBtn.disabled = !canStepMore;
    insertBtn.textContent = !canStepMore
      ? `Max ${session.maxTokens} tokens reached`
      : isHistoricalStep
        ? `⑂ Branch & Step "${stagedPred.label}" →`
        : `+ Insert & Step "${stagedPred.label}" →`;
    insertBtn.title = isHistoricalStep
      ? `Fork a new branch from step ${session.activeStepIndex + 1} with "${stagedPred.label}" (preserves prior branch)`
      : `Insert "${stagedPred.label}" at position t${step.totalTokens} and run 1 cached incremental step`;
    insertBtn.addEventListener('click', () => this.cb.onDecodeStepToken(stagedPred.tokenId));

    const greedyBtn = document.createElement('button');
    greedyBtn.type = 'button';
    greedyBtn.className = 'ed-cta-btn ed-cta-secondary ar-greedy-btn';
    greedyBtn.dataset.testid = 'ar-step-greedy';
    greedyBtn.disabled = !canStepMore;
    greedyBtn.textContent = `Greedy "${step.greedyNextToken.label}"`;
    greedyBtn.title = `Insert & process Greedy argmax candidate "${step.greedyNextToken.label}"`;
    greedyBtn.addEventListener('click', () => this.cb.onDecodeStepGreedy());

    actRow.append(insertBtn, greedyBtn);
    card.appendChild(actRow);

    // Fast vs. Slow Instructional Spine Replay controls + cutaway links when viewing the 52-Layer Tower
    if (mode === 'tower') {
      const waveControls = document.createElement('div');
      waveControls.className = 'ar-wave-controls';
      waveControls.dataset.testid = 'ar-wave-controls';

      const fastBtn = document.createElement('button');
      fastBtn.type = 'button';
      fastBtn.className =
        spineWaveMode === 'fast'
          ? 'ed-flow-tok-btn ar-wave-speed-btn is-active'
          : 'ed-flow-tok-btn ar-wave-speed-btn';
      fastBtn.dataset.testid = 'ar-replay-wave';
      fastBtn.textContent = '⟳ Fast Wave (0.9s)';
      fastBtn.setAttribute('aria-pressed', String(spineWaveMode === 'fast'));
      fastBtn.title =
        'Replay the 52-layer forward propagation wave at fast macro pace (~0.9s educational replay)';
      fastBtn.addEventListener('click', () => {
        if (this.cb.onReplaySpineWaveMode) {
          this.cb.onReplaySpineWaveMode('fast');
        } else {
          this.cb.onReplaySpineWave?.();
        }
      });

      const slowBtn = document.createElement('button');
      slowBtn.type = 'button';
      slowBtn.className =
        spineWaveMode === 'instructional'
          ? 'ed-flow-tok-btn ar-wave-speed-btn is-active'
          : 'ed-flow-tok-btn ar-wave-speed-btn';
      slowBtn.dataset.testid = 'ar-replay-wave-slow';
      slowBtn.textContent = '▶ Slow Replay (5.5s)';
      slowBtn.setAttribute('aria-pressed', String(spineWaveMode === 'instructional'));
      slowBtn.title =
        'Replay the 52-layer forward propagation wave at slow instructional pace (~5.5s educational replay with live stage HUD)';
      slowBtn.addEventListener('click', () => {
        this.cb.onReplaySpineWaveMode?.('instructional');
      });

      const waveNote = document.createElement('span');
      waveNote.className = 'ar-wave-controls-label';
      waveNote.dataset.testid = 'ar-wave-disclaimer';
      waveNote.textContent = 'Educational replay (not measured runtime)';

      waveControls.append(fastBtn, slowBtn, waveNote);
      card.appendChild(waveControls);

      const cutRow = document.createElement('div');
      cutRow.className = 'ar-cutaway-links';
      const mkCutBtn = (label: string, kind: BayKind, bayId: number, testid: string) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'ed-flow-tok-btn';
        b.dataset.testid = testid;
        b.textContent = label;
        b.addEventListener('click', () => this.cb.onInspectInternals(kind, bayId));
        return b;
      };
      cutRow.append(
        mkCutBtn(`GQA Bay 43 [${step.tokens}×${step.totalTokens}]`, '*', 43, 'ar-inspect-gqa'),
        mkCutBtn(`Mamba Bay 01 (Hₜ)`, 'M', 1, 'ar-inspect-mamba'),
        mkCutBtn(`MoE Bay 02 (0 B)`, 'E', 2, 'ar-inspect-moe'),
      );
      card.appendChild(cutRow);
    }

    // Explicit synthetic disclaimer
    const disclaimer = document.createElement('div');
    disclaimer.className = 'ed-flow-note ar-disclaimer';
    disclaimer.textContent =
      'Synthetic miniature AR output (d=8, V=32, Float32) · Not 31.6B checkpoint text. Timing/sizes reflect miniature demo, not production throughput.';
    card.appendChild(disclaimer);

    return card;
  }

  private buildEmbeddingNumericalCard(exec: HybridExecutionResult): HTMLDivElement {
    const T = exec.tokens;
    const tok = Math.max(0, Math.min(T - 1, this.boundaryInspectToken ?? T - 1));
    const tokId = exec.tokenIds[tok];
    const tokLabel = exec.tokenLabels[tok] ?? `t${tok}`;

    const card = document.createElement('div');
    card.className = 'ed-flow-card';
    card.dataset.testid = 'embedding-flow-card';

    const head = document.createElement('div');
    head.className = 'ed-flow-head';
    const title = document.createElement('span');
    title.className = 'ed-flow-title';
    title.textContent = `Live Mini Embedding · t${tok} "${tokLabel}" (id ${tokId})`;
    const tag = document.createElement('span');
    tag.className = 'ed-flow-tag';
    tag.textContent = 'NoPE · X₀';
    head.append(title, tag);

    const tokStrip = document.createElement('div');
    tokStrip.className = 'ed-flow-tok-strip';
    for (let i = 0; i < T; i++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ed-flow-tok-btn${i === tok ? ' is-active' : ''}`;
      btn.textContent = `t${i} ${exec.tokenLabels[i] ?? ''}`.trim();
      btn.addEventListener('click', () => this.setBoundaryInspectToken(i));
      tokStrip.appendChild(btn);
    }

    const vecGrid = document.createElement('div');
    vecGrid.className = 'ed-flow-vec-grid';
    let sumSq = 0;
    for (let ch = 0; ch < exec.d; ch++) {
      const v = exec.X_embed[ch * T + tok];
      sumSq += v * v;
      const cell = document.createElement('div');
      cell.className = 'ed-flow-vec-cell';
      const idx = document.createElement('span');
      idx.className = 'ed-flow-vec-idx';
      idx.textContent = `c${ch}`;
      const val = document.createElement('span');
      val.className = 'ed-flow-vec-val';
      val.textContent = fmtSigned(v, 2);
      cell.append(idx, val);
      vecGrid.appendChild(cell);
    }
    const rms = Math.sqrt(sumSq / exec.d);

    const meta = document.createElement('div');
    meta.className = 'ed-flow-meta';
    meta.textContent = `RMS(X₀[:, t${tok}]) = ${rms.toFixed(3)} · → Bay 01 Mamba-2 Input X (bit-exact)`;

    const note = document.createElement('div');
    note.className = 'ed-flow-note';
    note.textContent = 'Synthetic Float32 mini (d=8, V=32, untied W_embed ≠ W_lm_head). Order enters via Bay 01 Mamba-2.';

    const actions = document.createElement('div');
    actions.className = 'ed-actions';
    const inspectBtn = document.createElement('button');
    inspectBtn.type = 'button';
    inspectBtn.className = 'ed-cta-btn';
    inspectBtn.textContent = 'Inspect Bay 01 Mamba-2 →';
    inspectBtn.addEventListener('click', () => this.cb.onInspectInternals('M', 1));
    actions.appendChild(inspectBtn);

    card.append(head, tokStrip, vecGrid, meta, note, actions);
    return card;
  }

  private buildOutputNumericalCard(exec: HybridExecutionResult): HTMLDivElement {
    const T = exec.tokens;
    const tok = Math.max(0, Math.min(T - 1, this.boundaryInspectToken ?? T - 1));
    const tokLabel = exec.tokenLabels[tok] ?? `t${tok}`;
    const fn = exec.finalNorm;
    const preds = exec.lmHead.topKByToken[tok] ?? [];
    const maxProb = Math.max(0.05, preds[0]?.prob ?? 0.25);

    const card = document.createElement('div');
    card.className = 'ed-flow-card';
    card.dataset.testid = 'output-flow-card';

    const head = document.createElement('div');
    head.className = 'ed-flow-head';
    const title = document.createElement('span');
    title.className = 'ed-flow-title';
    title.textContent = `Final RMSNorm & LM Head · t${tok} "${tokLabel}"`;
    const tag = document.createElement('span');
    tag.className = 'ed-flow-tag';
    tag.textContent = `ε=${formatEpsLabel(fn.eps)}`;
    head.append(title, tag);

    const tokStrip = document.createElement('div');
    tokStrip.className = 'ed-flow-tok-strip';
    for (let i = 0; i < T; i++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ed-flow-tok-btn${i === tok ? ' is-active' : ''}`;
      btn.textContent = `t${i} ${exec.tokenLabels[i] ?? ''}`.trim();
      btn.addEventListener('click', () => this.setBoundaryInspectToken(i));
      tokStrip.appendChild(btn);
    }

    const meta = document.createElement('div');
    meta.className = 'ed-flow-meta';
    meta.textContent = `← Bay 52 Final MoE Y (bit-exact) · rms_final = ${fn.rms[tok].toFixed(3)} · Top-5 Next Tokens:`;

    const predList = document.createElement('div');
    predList.className = 'ed-flow-pred-list';
    for (const pred of preds) {
      const row = document.createElement('div');
      row.className = 'ed-flow-pred-row';

      const word = document.createElement('span');
      word.className = 'ed-flow-pred-word';
      word.textContent = pred.label;
      word.title = `vocab #${pred.tokenId}: ${pred.label}`;

      const track = document.createElement('div');
      track.className = 'ed-flow-pred-track';
      const bar = document.createElement('div');
      bar.className = 'ed-flow-pred-bar';
      bar.style.width = `${Math.max(4, Math.min(100, (pred.prob / maxProb) * 100)).toFixed(1)}%`;
      track.appendChild(bar);

      const logit = document.createElement('span');
      logit.className = 'ed-flow-pred-logit';
      logit.textContent = fmtSigned(pred.logit, 2);

      const prob = document.createElement('span');
      prob.className = 'ed-flow-pred-prob';
      prob.textContent = `${(pred.prob * 100).toFixed(1)}%`;

      row.append(word, track, logit, prob);
      predList.appendChild(row);
    }

    const note = document.createElement('div');
    note.className = 'ed-flow-note';
    note.textContent = 'Synthetic V=32 mini vocab & untied W_lm_head (Float32) — educational demo, not 31.6B checkpoint logits.';

    const actions = document.createElement('div');
    actions.className = 'ed-actions';
    const inspectBtn = document.createElement('button');
    inspectBtn.type = 'button';
    inspectBtn.className = 'ed-cta-btn ed-cta-secondary';
    inspectBtn.textContent = 'Inspect Bay 52 Final MoE →';
    inspectBtn.addEventListener('click', () => this.cb.onInspectInternals('E', 52));
    actions.appendChild(inspectBtn);

    card.append(head, tokStrip, meta, predList, note, actions);
    return card;
  }

  private renderTowerEditorial(
    section: TocSectionId,
    bayId: number | null,
    isPinned: boolean,
    precision: PrecisionMode,
    tokens: number,
    activeExecution: HybridExecutionResult | null = null,
    decodeSession: HybridDecodeSession | null = null,
    spineWaveMode: SpineWaveMode = 'fast',
  ): void {
    const panel = this.els.editorialPanel;
    panel.classList.remove('is-tour-mode');
    panel.innerHTML = '';

    if (decodeSession) {
      panel.appendChild(
        this.buildArDecodeSessionCard(decodeSession, precision, 'tower', spineWaveMode),
      );
    }

    if (bayId !== null) {
      const bay = this.spec.bays[bayId - 1];
      if (bay) {
        const bracket = this.spec.brackets[bay.bracket];
        const computePrec = bodyPrecision(bay, precision);
        const computeLine =
          precision === 'bf16'
            ? 'Compute: BF16 weights & activations'
            : computePrec === 'bf16'
              ? 'Compute: BF16 preserved (sensitive bay in FP8 Selective)'
              : 'Compute: FP8 weights & activations (quantized)';

        let storageLine: string;
        let narrative: string;
        const extraSpecs: string[] = [];

        if (bay.kind === 'M') {
          const ssm = ssmStateDtype(bay)!.toUpperCase();
          const conv = conv1dPrecision(bay, precision)!.toUpperCase();
          storageLine = `Memory: SSM ${ssm} (64×64×128) · Conv1D ${conv} (6,144×4) — fixed`;
          narrative = bay.feeder
            ? `Bay ${formatBayId(bay.id)} is a Mamba-2 feeder immediately preceding GQA attention bay ${formatBayId(bay.id + 1)}. NVIDIA §4.2 retains BF16 compute here to protect attention input fidelity, while its cyan SSM register updates in place.`
            : `Bay ${formatBayId(bay.id)} runs a linear-time Mamba-2 state-space update. Its four-slot Conv1D strip stays BF16 in all modes, and its cyan-speckled FP32 state matrix updates in place without growing with sequence length.`;
        } else if (bay.kind === '*') {
          const kvPrec = kvCachePrecision(bay, precision)!;
          const perLayerTotalKiB = ((kvPrec === 'fp8' ? 512 : 1024) * tokens) / 1024;
          const perLayerSlope = kvPrec === 'fp8' ? '+0.5 KiB/token' : '+1 KiB/token';
          storageLine = `KV Cache: ${kvPrec.toUpperCase()} (${perLayerTotalKiB} KiB at T=${tokens}, ${perLayerSlope})`;
          narrative = `Bay ${formatBayId(bay.id)} is one of six Grouped-Query Attention layers (32Q / 2KV, NoPE). In FP8 Selective mode, the attention body stays BF16 while its KV cache switches to FP8.`;
        } else {
          storageLine = 'Memory: 0 B token history (stateless MoE layer)';
          narrative = `Bay ${formatBayId(bay.id)} routes each token through a 128-score sigmoid gate to activate six routed experts (width 1,856, non-gated ReLU²) in parallel alongside one always-active shared expert.`;
          extraSpecs.push(SHARED_EXPERT_NOTE);
        }

        if (activeExecution) {
          const sum = activeExecution.baySummaries[bay.id - 1];
          const lastT = activeExecution.tokens - 1;
          if (sum) {
            const tokLabel = activeExecution.tokenLabels[lastT] ?? `t${lastT}`;
            extraSpecs.push(
              `Unified mini stream (t${lastT} "${tokLabel}"): RMS ${sum.rmsIn[lastT].toFixed(2)} → Δ ${sum.rmsDelta[lastT].toFixed(2)} → ${sum.rmsOut[lastT].toFixed(2)}`,
            );
          }
        }

        const header = document.createElement('div');
        header.className = 'ed-header';

        const title = document.createElement('h2');
        title.className = 'ed-title';
        title.textContent = `Bay ${formatBayId(bay.id)} / 52 · ${bay.kind} ${bayKindLabel(bay.kind)}`;

        const badge = document.createElement('span');
        badge.className = 'ed-badge ed-badge-active';
        badge.textContent = isPinned ? 'Pinned' : 'Hover';

        header.append(title, badge);

        const sub = document.createElement('div');
        sub.className = 'ed-sub';
        sub.textContent = formatBracketCaption(bracket);

        const p = document.createElement('p');
        p.className = 'ed-body';
        p.textContent = narrative;

        const list = document.createElement('ul');
        list.className = 'ed-specs';
        for (const item of [computeLine, storageLine, ...extraSpecs, PARAMETER_CAPTION]) {
          const li = document.createElement('li');
          li.textContent = item;
          list.appendChild(li);
        }

        const actions = document.createElement('div');
        actions.className = 'ed-actions';
        const inspectBtn = document.createElement('button');
        inspectBtn.type = 'button';
        inspectBtn.className = 'ed-cta-btn';
        inspectBtn.dataset.testid = 'inspect-internals-btn';
        inspectBtn.textContent = 'Inspect Internals →';
        inspectBtn.addEventListener('click', () => {
          this.cb.onInspectInternals(bay.kind, bay.id);
        });
        actions.appendChild(inspectBtn);

        panel.append(header, sub, p, list, actions);
        return;
      }
    }

    const copy = SECTION_COPY[section];
    const header = document.createElement('div');
    header.className = 'ed-header';

    const title = document.createElement('h2');
    title.className = 'ed-title';
    title.textContent = copy.title;

    const badge = document.createElement('span');
    badge.className = 'ed-badge';
    badge.textContent = copy.badge;

    header.append(title, badge);

    const p = document.createElement('p');
    p.className = 'ed-body';
    p.textContent = copy.body;

    const list = document.createElement('ul');
    list.className = 'ed-specs';
    for (const item of copy.specs) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }

    panel.append(header, p);
    if (activeExecution && section === 'embedding') {
      panel.appendChild(this.buildEmbeddingNumericalCard(activeExecution));
    } else if (activeExecution && section === 'output') {
      panel.appendChild(this.buildOutputNumericalCard(activeExecution));
    }
    panel.appendChild(list);

    if (copy.inspectKind) {
      const actions = document.createElement('div');
      actions.className = 'ed-actions';
      const inspectBtn = document.createElement('button');
      inspectBtn.type = 'button';
      inspectBtn.className = 'ed-cta-btn';
      inspectBtn.textContent = 'Inspect Internals →';
      inspectBtn.addEventListener('click', () => {
        this.cb.onInspectInternals(copy.inspectKind!);
      });
      actions.appendChild(inspectBtn);
      panel.appendChild(actions);
    }
  }

  private navigateToCrossBayNeighbor(neighbor: CrossBayNeighbor): void {
    if (this.cb.onNavigateAdjacentBay) {
      this.cb.onNavigateAdjacentBay(neighbor);
      return;
    }
    if (neighbor.kind === 'embedding') {
      this.cb.onSelectMode('cutaway_embedding');
    } else if (neighbor.kind === 'output') {
      this.cb.onSelectMode('cutaway_output');
    } else {
      this.cb.onInspectInternals(neighbor.kind, neighbor.bayId);
    }
  }

  /**
   * Build the clickable V4 cross-bay continuity breadcrumb bar derived from live
   * `activeExecution.baySummaries` / `activeDecodeStep.baySummaries`.
   */
  private buildCrossBayBreadcrumb(
    state: SceneState,
    cutaway: CutawaySceneGeometry | null,
    activeExecution: HybridExecutionResult | null,
    decodeSession: HybridDecodeSession | null,
  ): HTMLElement | null {
    const activeBayId =
      state.mode === 'cutaway_embedding'
        ? 0
        : state.mode === 'cutaway_output'
          ? this.spec.bays.length + 1
          : state.mode === 'cutaway_mamba'
            ? state.mambaBayId
            : state.mode === 'cutaway_moe'
              ? state.moeBayId
              : state.mode === 'cutaway_gqa'
                ? state.gqaBayId
                : undefined;

    const neighbors = buildCrossBayNeighbors(state.mode, activeBayId, this.spec);
    if (!neighbors) return null;

    const activeDecodeStep = decodeSession ? getActiveDecodeStep(decodeSession) : null;
    const summaries = activeDecodeStep?.baySummaries ?? activeExecution?.baySummaries ?? null;
    const finalNormRms = activeDecodeStep?.finalNorm.rms ?? activeExecution?.finalNorm.rms ?? null;
    const summaryTokens = activeDecodeStep?.tokens ?? activeExecution?.tokens ?? state.tokens;

    const rawTok =
      state.mode === 'cutaway_embedding'
        ? state.embeddingFocusToken
        : state.mode === 'cutaway_output'
          ? state.outputFocusToken
          : state.mode === 'cutaway_mamba'
            ? (cutaway?.mambaStats?.stateTokenIndex ?? state.mambaStateToken ?? summaryTokens - 1)
            : state.mode === 'cutaway_moe'
              ? (cutaway?.moeStats?.focusToken ?? state.moeFocusToken ?? summaryTokens - 1)
              : summaryTokens - 1;
    const focusTok = Math.max(0, Math.min(Math.max(0, summaryTokens - 1), rawTok));

    const rmsForNode = (node: CrossBayNeighbor): number | null => {
      if (!summaries || summaries.length === 0) return null;
      if (node.kind === 'embedding') {
        const val = summaries[0]?.rmsIn[focusTok];
        return val !== undefined && Number.isFinite(val) ? val : null;
      }
      if (node.kind === 'output') {
        const val = finalNormRms?.[focusTok] ?? summaries[summaries.length - 1]?.rmsOut[focusTok];
        return val !== undefined && Number.isFinite(val) ? val : null;
      }
      const sum = summaries[node.bayId - 1];
      const val = sum?.rmsOut[focusTok];
      return val !== undefined && Number.isFinite(val) ? val : null;
    };

    const fmtRmsSuffix = (node: CrossBayNeighbor): string => {
      const rms = rmsForNode(node);
      return rms !== null ? ` (${rms.toFixed(2)})` : '';
    };

    const nav = document.createElement('nav');
    nav.className = 'cutaway-breadcrumb';
    nav.dataset.testid = 'cutaway-breadcrumb';
    nav.setAttribute('aria-label', 'Cross-bay continuity');

    if (neighbors.prev) {
      const prevNode = neighbors.prev;
      const prevBtn = document.createElement('button');
      prevBtn.type = 'button';
      prevBtn.className = 'cutaway-breadcrumb-btn';
      prevBtn.dataset.breadcrumbDir = 'prev';
      prevBtn.dataset.breadcrumbBay = String(prevNode.bayId);
      const rmsSuffix = fmtRmsSuffix(prevNode);
      prevBtn.textContent = `← ${prevNode.shortLabel}${rmsSuffix}`;
      prevBtn.title = `Jump upstream to ${prevNode.label}${rmsSuffix}`;
      prevBtn.addEventListener('click', () => this.navigateToCrossBayNeighbor(prevNode));
      nav.appendChild(prevBtn);

      const sep = document.createElement('span');
      sep.className = 'cutaway-breadcrumb-sep';
      sep.setAttribute('aria-hidden', 'true');
      sep.textContent = '·';
      nav.appendChild(sep);
    }

    const currSpan = document.createElement('span');
    currSpan.className = 'cutaway-breadcrumb-current';
    currSpan.dataset.testid = 'cutaway-breadcrumb-current';
    currSpan.textContent = neighbors.current.label;
    const currRms = rmsForNode(neighbors.current);
    if (currRms !== null) {
      currSpan.title = `${neighbors.current.label} · stream RMS = ${currRms.toFixed(3)}`;
    }
    nav.appendChild(currSpan);

    if (neighbors.next) {
      const nextNode = neighbors.next;
      const sep = document.createElement('span');
      sep.className = 'cutaway-breadcrumb-sep';
      sep.setAttribute('aria-hidden', 'true');
      sep.textContent = '·';
      nav.appendChild(sep);

      const nextBtn = document.createElement('button');
      nextBtn.type = 'button';
      nextBtn.className = 'cutaway-breadcrumb-btn';
      nextBtn.dataset.breadcrumbDir = 'next';
      nextBtn.dataset.breadcrumbBay = String(nextNode.bayId);
      const rmsSuffix = fmtRmsSuffix(nextNode);
      nextBtn.textContent = `${nextNode.shortLabel}${rmsSuffix} →`;
      nextBtn.title = `Jump downstream to ${nextNode.label}${rmsSuffix}`;
      nextBtn.addEventListener('click', () => this.navigateToCrossBayNeighbor(nextNode));
      nav.appendChild(nextBtn);
    }

    return nav;
  }

  /** Interactive token chips & live X₀ RMS readout inside dedicated Embedding Cutaway panel. */
  private buildEmbeddingCutawayInspector(
    state: SceneState,
    activeExecution: HybridExecutionResult | null,
    decodeSession: HybridDecodeSession | null,
  ): HTMLDivElement | null {
    const activeDecodeStep = decodeSession ? getActiveDecodeStep(decodeSession) : null;
    const T = activeDecodeStep?.tokens ?? activeExecution?.tokens ?? 0;
    const d = activeDecodeStep?.d ?? activeExecution?.d ?? 8;
    const X_embed = activeDecodeStep?.X_embed ?? activeExecution?.X_embed ?? null;
    const tokenIds = activeDecodeStep?.newTokenIds ?? activeExecution?.tokenIds ?? null;
    const tokenLabels = activeDecodeStep?.newTokenLabels ?? activeExecution?.tokenLabels ?? null;
    const startOffset = activeDecodeStep?.prefixTokens ?? 0;

    if (!X_embed || !tokenIds || !tokenLabels || T <= 0) return null;

    const rawFocus = this.cb.onSelectEmbeddingToken
      ? state.embeddingFocusToken
      : (this.boundaryInspectToken ?? state.embeddingFocusToken);
    const tok = Math.max(0, Math.min(T - 1, rawFocus));
    const absTok = startOffset + tok;
    const tokId = tokenIds[tok];
    const tokLabel = tokenLabels[tok] ?? `t${absTok}`;

    const card = document.createElement('div');
    card.className = 'ed-flow-card';
    card.dataset.testid = 'embedding-cutaway-inspector';

    const head = document.createElement('div');
    head.className = 'ed-flow-head';
    const title = document.createElement('span');
    title.className = 'ed-flow-title';
    title.textContent = `Active Token · t${absTok} "${tokLabel}" (id ${tokId})`;
    const tag = document.createElement('span');
    tag.className = 'ed-flow-tag';
    tag.textContent = 'NoPE · X₀';
    head.append(title, tag);

    const tokStrip = document.createElement('div');
    tokStrip.className = 'ed-flow-tok-strip';
    tokStrip.dataset.testid = 'embedding-token-strip';
    for (let i = 0; i < T; i++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ed-flow-tok-btn${i === tok ? ' is-active' : ''}`;
      btn.dataset.embeddingToken = String(i);
      const absI = startOffset + i;
      btn.textContent = `t${absI} ${tokenLabels[i] ?? ''}`.trim();
      btn.setAttribute('aria-pressed', String(i === tok));
      btn.addEventListener('click', () => {
        if (this.cb.onSelectEmbeddingToken) this.cb.onSelectEmbeddingToken(i);
        else this.setBoundaryInspectToken(i);
      });
      tokStrip.appendChild(btn);
    }

    const vecGrid = document.createElement('div');
    vecGrid.className = 'ed-flow-vec-grid';
    let sumSq = 0;
    for (let ch = 0; ch < d; ch++) {
      const v = X_embed[ch * T + tok];
      sumSq += v * v;
      const cell = document.createElement('div');
      cell.className = 'ed-flow-vec-cell';
      const idx = document.createElement('span');
      idx.className = 'ed-flow-vec-idx';
      idx.textContent = `c${ch}`;
      const val = document.createElement('span');
      val.className = 'ed-flow-vec-val';
      val.textContent = fmtSigned(v, 2);
      cell.append(idx, val);
      vecGrid.appendChild(cell);
    }
    const rms = Math.sqrt(sumSq / d);

    const meta = document.createElement('div');
    meta.className = 'ed-flow-meta';
    meta.dataset.testid = 'embedding-rms-readout';
    meta.textContent = `RMS(X₀[:, t${absTok}]) = ${rms.toFixed(3)} · → Bay 01 Mamba-2 Input X (bit-exact)`;

    const note = document.createElement('div');
    note.className = 'ed-flow-note';
    note.textContent =
      'Synthetic Float32 mini (d=8, V=32, untied W_embed ≠ W_lm_head). Zero positional embeddings (NoPE); order enters in Bay 01 Mamba-2.';

    card.append(head, tokStrip, vecGrid, meta, note);
    return card;
  }

  /** Interactive token selector & live Top-5 next-token bars inside dedicated Output Cutaway panel. */
  private buildOutputCutawayInspector(
    state: SceneState,
    activeExecution: HybridExecutionResult | null,
    decodeSession: HybridDecodeSession | null,
  ): HTMLDivElement | null {
    const activeDecodeStep = decodeSession ? getActiveDecodeStep(decodeSession) : null;
    const T = activeDecodeStep?.tokens ?? activeExecution?.tokens ?? 0;
    const fn = activeDecodeStep?.finalNorm ?? activeExecution?.finalNorm ?? null;
    const lmHead = activeDecodeStep?.lmHead ?? activeExecution?.lmHead ?? null;
    const tokenLabels = activeDecodeStep?.newTokenLabels ?? activeExecution?.tokenLabels ?? null;
    const startOffset = activeDecodeStep?.prefixTokens ?? 0;

    if (!fn || !lmHead || !tokenLabels || T <= 0) return null;

    const rawFocus = this.cb.onSelectOutputToken
      ? state.outputFocusToken
      : (this.boundaryInspectToken ?? state.outputFocusToken);
    const tok = Math.max(0, Math.min(T - 1, rawFocus));
    const absTok = startOffset + tok;
    const tokLabel = tokenLabels[tok] ?? `t${absTok}`;
    const preds = lmHead.topKByToken[tok] ?? [];
    const maxProb = Math.max(0.05, preds[0]?.prob ?? 0.25);

    const card = document.createElement('div');
    card.className = 'ed-flow-card';
    card.dataset.testid = 'output-cutaway-inspector';

    const head = document.createElement('div');
    head.className = 'ed-flow-head';
    const title = document.createElement('span');
    title.className = 'ed-flow-title';
    title.textContent = `Final RMSNorm & LM Head · t${absTok} "${tokLabel}"`;
    const tag = document.createElement('span');
    tag.className = 'ed-flow-tag';
    tag.textContent = `ε=${formatEpsLabel(fn.eps)}`;
    head.append(title, tag);

    const tokStrip = document.createElement('div');
    tokStrip.className = 'ed-flow-tok-strip';
    tokStrip.dataset.testid = 'output-token-strip';
    for (let i = 0; i < T; i++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ed-flow-tok-btn${i === tok ? ' is-active' : ''}`;
      btn.dataset.outputToken = String(i);
      const absI = startOffset + i;
      btn.textContent = `t${absI} ${tokenLabels[i] ?? ''}`.trim();
      btn.setAttribute('aria-pressed', String(i === tok));
      btn.addEventListener('click', () => {
        if (this.cb.onSelectOutputToken) this.cb.onSelectOutputToken(i);
        else this.setBoundaryInspectToken(i);
      });
      tokStrip.appendChild(btn);
    }

    const meta = document.createElement('div');
    meta.className = 'ed-flow-meta';
    meta.textContent = `← Bay 52 Final MoE Y (bit-exact) · rms_final = ${fn.rms[tok].toFixed(3)} · Top-5 Next Tokens:`;

    const predList = document.createElement('div');
    predList.className = 'ed-flow-pred-list';
    predList.dataset.testid = 'output-top5-bars';
    for (const pred of preds) {
      const row = document.createElement('div');
      row.className = 'ed-flow-pred-row';

      const word = document.createElement('span');
      word.className = 'ed-flow-pred-word';
      word.textContent = pred.label;
      word.title = `vocab #${pred.tokenId}: ${pred.label}`;

      const track = document.createElement('div');
      track.className = 'ed-flow-pred-track';
      const bar = document.createElement('div');
      bar.className = 'ed-flow-pred-bar';
      bar.style.width = `${Math.max(4, Math.min(100, (pred.prob / maxProb) * 100)).toFixed(1)}%`;
      track.appendChild(bar);

      const logit = document.createElement('span');
      logit.className = 'ed-flow-pred-logit';
      logit.textContent = fmtSigned(pred.logit, 2);

      const prob = document.createElement('span');
      prob.className = 'ed-flow-pred-prob';
      prob.textContent = `${(pred.prob * 100).toFixed(1)}%`;

      row.append(word, track, logit, prob);
      predList.appendChild(row);
    }

    const note = document.createElement('div');
    note.className = 'ed-flow-note';
    note.dataset.testid = 'output-synthetic-disclaimer';
    note.textContent =
      'Synthetic miniature demo (d=8, V=32, untied W_lm_head, Float32) — educational next-token probabilities, not 31.6B checkpoint logits.';

    card.append(head, tokStrip, meta, predList, note);
    return card;
  }

  private renderCutawayEditorial(
    state: SceneState,
    activePart: string | null,
    cutaway: CutawaySceneGeometry | null,
    activeExecution: HybridExecutionResult | null = null,
    decodeSession: HybridDecodeSession | null = null,
  ): void {
    const panel = this.els.editorialPanel;
    panel.classList.remove('is-tour-mode');
    panel.innerHTML = '';

    // 0. Clickable Cross-Bay Continuity Breadcrumb at top of every cutaway panel
    const breadcrumb = this.buildCrossBayBreadcrumb(
      state,
      cutaway,
      activeExecution,
      decodeSession,
    );
    if (breadcrumb) {
      panel.appendChild(breadcrumb);
    }

    if (decodeSession) {
      panel.appendChild(
        this.buildArDecodeSessionCard(decodeSession, state.precision, state.mode, state.spineWaveMode),
      );
    }

    // 1. If user clicked/hovered a specific tensor slab inside the cutaway, show its details!
    const partInfo: CutawayPartInfo | undefined =
      activePart && cutaway ? cutaway.partCatalog[activePart] : undefined;

    if (partInfo) {
      const header = document.createElement('div');
      header.className = 'ed-header';

      const title = document.createElement('h2');
      title.className = 'ed-title';
      title.textContent = partInfo.title;

      const badge = document.createElement('span');
      badge.className = 'ed-badge ed-badge-active';
      badge.textContent = partInfo.badge;
      header.append(title, badge);

      const p = document.createElement('p');
      p.className = 'ed-body';
      p.textContent = partInfo.summary;

      const list = document.createElement('ul');
      list.className = 'ed-specs';
      for (const item of partInfo.specs) {
        const li = document.createElement('li');
        li.textContent = item;
        list.appendChild(li);
      }

      const actions = document.createElement('div');
      actions.className = 'ed-actions';
      if (state.selectedPart !== null) {
        const clearBtn = document.createElement('button');
        clearBtn.type = 'button';
        clearBtn.className = 'ed-cta-btn ed-cta-secondary';
        clearBtn.textContent = 'All Layer Internals';
        clearBtn.addEventListener('click', () => this.cb.onClearSelection());
        actions.appendChild(clearBtn);
      }
      if (state.mode === 'cutaway_embedding') {
        const bay01Btn = document.createElement('button');
        bay01Btn.type = 'button';
        bay01Btn.className = 'ed-cta-btn';
        bay01Btn.dataset.testid = 'embed-cta-bay01';
        bay01Btn.textContent = '→ Bay 01 Mamba-2';
        bay01Btn.addEventListener('click', () => this.cb.onInspectInternals('M', 1));
        actions.appendChild(bay01Btn);
      } else if (state.mode === 'cutaway_output') {
        const bay52Btn = document.createElement('button');
        bay52Btn.type = 'button';
        bay52Btn.className = 'ed-cta-btn';
        bay52Btn.dataset.testid = 'output-cta-bay52';
        bay52Btn.textContent = '← Bay 52 Final MoE';
        bay52Btn.addEventListener('click', () => this.cb.onInspectInternals('E', 52));
        actions.appendChild(bay52Btn);
      }
      const backBtn = document.createElement('button');
      backBtn.type = 'button';
      backBtn.className = 'ed-cta-btn ed-cta-secondary';
      backBtn.textContent = '← 52-Layer Tower';
      backBtn.addEventListener('click', () => this.cb.onSelectMode('tower'));
      actions.appendChild(backBtn);

      panel.append(header, p, list);
      if (state.mode === 'cutaway_embedding') {
        const embCard = this.buildEmbeddingCutawayInspector(state, activeExecution, decodeSession);
        if (embCard) panel.appendChild(embCard);
      } else if (state.mode === 'cutaway_output' && !decodeSession) {
        const outCard = this.buildOutputCutawayInspector(state, activeExecution, decodeSession);
        if (outCard) panel.appendChild(outCard);
      } else if (state.mode === 'cutaway_gqa') {
        panel.appendChild(this.buildGqaHeadSelector(state.selectedPart, state.hoveredPart));
      } else if (state.mode === 'cutaway_mamba') {
        const effHead = cutaway?.mambaStats?.activeHead ?? state.mambaActiveHead;
        panel.appendChild(this.buildMambaTemporalControls(state, cutaway));
        panel.appendChild(
          this.buildMambaHeadSelector(effHead, state.selectedPart, state.hoveredPart),
        );
      } else if (state.mode === 'cutaway_moe') {
        panel.appendChild(this.buildMoeInspectorControls(state, cutaway));
      }
      panel.appendChild(actions);
      return;
    }

    // 2. Otherwise show the rich Cutaway Overview card
    const isFp8 = state.precision === 'fp8_selective';
    const header = document.createElement('div');
    header.className = 'ed-header';

    const title = document.createElement('h2');
    title.className = 'ed-title';

    const badge = document.createElement('span');
    badge.className = 'ed-badge';

    let bodyText = '';
    let specs: string[] = [];

    if (state.mode === 'cutaway_embedding') {
      title.textContent = 'Embedding Cutaway (NoPE)';
      badge.textContent = '131,072 × 2,688 · Untied';

      bodyText =
        'Untied token lookup gathers one 2,688-channel row of W_embed per token ID directly into X₀ with zero matrix multiply. Nemotron 3 Nano uses NoPE — no positional embeddings are added here; causal order arrives downstream in Bay 01 Mamba-2.';

      specs = [
        'Untied table: 131,072 vocab × 2,688 hidden (~352M params · tie_word_embeddings: false)',
        'Absent positional embeddings (NoPE): zero W_pos or RoPE — clean extrapolation toward 1M context',
        'Bit-exact hand-off: Embedding output X₀ feeds directly onto the emerald spine into Bay 01 Mamba-2',
      ];
    } else if (state.mode === 'cutaway_output') {
      title.textContent = 'Output & Untied LM Head';
      badge.textContent = '2,688 → 131,072 · ε=1e−5';
      if (isFp8) badge.classList.add('ed-badge-gold');

      bodyText =
        'After Bay 52 Final MoE, the emerald residual stream Y passes through Final RMSNorm (ε = 1e−5) and an independent untied LM Head (W_lm_head ≠ W_embed) to emit 131,072 vocabulary logits and next-token probabilities.';

      specs = [
        'Bay 52 MoE Y → Final RMSNorm (2,688, ε=1e−5) → Untied W_lm_head (2,688 × 131,072)',
        'Untied output projection (~352M params) decouples next-token logits from input token embeddings',
        'Synthetic miniature readout (d=8, V=32, Float32) demonstrates exact math — not 31.6B checkpoint logits',
      ];
    } else if (state.mode === 'cutaway_mamba') {
      const isFeeder = state.mambaBayId === 42;
      title.textContent = isFeeder
        ? 'Mamba-2 Cutaway · Bay 42 (Feeder)'
        : 'Mamba-2 Cutaway · Bay 01';
      badge.textContent = isFeeder && isFp8 ? 'BF16 Preserved §4.2' : '47.1 MiB Fixed';
      if (isFeeder && isFp8) badge.classList.add('ed-badge-gold');

      bodyText = isFeeder
        ? 'Feeder before Bay 43 GQA — preserved in BF16 (§4.2). Deterministic miniature sim (d=8, 16H/8G, dₕ=4, dₛ=4; Float32, rounded): sequence decks grow with T while Conv1D & Hₜ stay fixed.'
        : 'Deterministic miniature Mamba-2 sim (d=8, 16H/8G, dₕ=4, dₛ=4; Float32, rounded). Sequence decks grow with T while the 4-tap Conv1D buffer and Hₜ₋₁ ⇄ Hₜ pedestal stay fixed.';

      specs = [
        ...(activeExecution
          ? [
              isFeeder
                ? 'Unified 52-layer stream: Residual Y feeds Bay 43 GQA Input X bit-exact'
                : 'Unified 52-layer stream: Input X = Untied Embedding X₀ (NoPE) · Residual Y → Bay 02 MoE X',
            ]
          : []),
        'Persistent storage (23 bays): 46.0 MiB SSM (FP32) + ~1.08 MiB Conv1D (BF16) ≈ 47.1 MiB fixed · H_history is browser-only playback',
        isFeeder
          ? 'Bay 42 Feeder metadata (§4.2): W_in, W_out, Conv1D preserved in BF16 (gold rim); live sim runs Float32'
          : `Bay 01 metadata (§4.2): W_in & W_out ${isFp8 ? 'FP8' : 'BF16'} · Conv1D ALWAYS BF16 · SSM FP32 (live sim runs Float32)`,
        'Gate FIRST (y·SiLU(z)) → 8-Group RMSNorm (ε=1e−5, norm_before_gate=False) · Select H00–H15 below.',
      ];
    } else if (state.mode === 'cutaway_moe') {
      const fTok = cutaway?.moeStats?.focusToken ?? Math.max(0, state.tokens - 1);
      const absFTok = fTok + (cutaway?.moeSim?.startTokenOffset ?? 0);
      const tokLabel = cutaway?.moeSim?.tokenLabels[fTok] ?? `t${absFTok}`;
      title.textContent =
        state.moeBayId === 52
          ? 'MoE 128+1 Cutaway · Bay 52 (Final)'
          : 'MoE 128+1 Cutaway · Bay 02';
      badge.textContent = isFp8 ? 'FP8 Experts · FP32 Router' : `6 / 128 Active · t${absFTok}`;
      if (isFp8) badge.classList.add('ed-badge-gold');

      bodyText =
        `Runs a deterministic miniature MoE simulator (d=8, 128 routed top-6, d_exp=4, d_shared=8 = 2×, ×2.5; Float32 storage, rounded outputs). For token t${absFTok} "${tokLabel}", RMSNorm forks in parallel into the sparse 128-expert bank (non-gated ReLU², no gate_proj) and the always-on 2× shared expert.`;

      specs = [
        ...(activeExecution
          ? [
              state.moeBayId === 52
                ? 'Unified 52-layer stream: Final Residual Y → Final RMSNorm (ε=1e−5) → Untied LM Head'
                : 'Unified 52-layer stream: Input X = Bay 01 Mamba-2 Residual Y bit-exact',
            ]
          : []),
        'Router: sigmoid(W_router·U) + e_score_correction_bias selects Top-6; unbiased σ gathers weights → norm × 2.5 (Router stays FP32 in FP8 Selective)',
        SHARED_EXPERT_NOTE + ' (added unscaled at 1.0× alongside Top-6 routed sum)',
        '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B dynamic token cache (strictly per-token independent)',
      ];
    } else {
      const perBayKiB = ((isFp8 ? 512 : 1024) * state.tokens) / 1024;
      const gqaEps = formatEpsLabel(cutaway?.gqaSim?.rmsEps ?? 1e-5);
      title.textContent = `GQA Cutaway · Bay ${formatBayId(state.gqaBayId)}`;
      badge.textContent = isFp8 ? 'BF16 Compute · FP8 KV' : '32Q / 2KV · NoPE';
      if (isFp8) badge.classList.add('ed-badge-gold');

      bodyText =
        `Thirty-two query heads (Group 0: Q00–Q15 → KV₀; Group 1: Q16–Q31 → KV₁) drive a deterministic miniature simulator with Float32 tensor storage and rounded outputs (d=8, dₕ=4, NoPE, ε=${gqaEps}). Hover or click any cell for rounded numerical explanations.`;

      specs = [
        ...(activeExecution
          ? [
              state.gqaBayId === 43
                ? `Unified 52-layer stream: Input X = Bay 42 Mamba-2 Feeder Residual Y bit-exact (ε=${gqaEps})`
                : `Unified 52-layer stream: Input X = Bay 05 Mamba-2 Feeder Residual Y bit-exact (ε=${gqaEps})`,
            ]
          : []),
        `Shared KV cache: 2 KV heads × ${state.tokens} tokens = ${perBayKiB} KiB (${isFp8 ? '+0.5 KiB/tok FP8 storage' : '+1 KiB/tok BF16 storage'})`,
        isFp8
          ? 'Production precision metadata (§4.2): Q/K/V/O compute BF16; KV storage FP8 (miniature sim uses Float32 storage)'
          : 'Production precision metadata: BF16 weights/KV (toggle FP8 Selective for §4.2; miniature sim uses Float32 storage)',
        'Click any Q00–Q31 head below to switch its group’s live Scores & Softmax textures.',
      ];
    }

    header.append(title, badge);

    const p = document.createElement('p');
    p.className = 'ed-body';
    p.textContent = bodyText;

    const list = document.createElement('ul');
    list.className = 'ed-specs';
    for (const item of specs) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }

    const actions = document.createElement('div');
    actions.className = 'ed-actions';
    if (state.mode === 'cutaway_embedding') {
      const bay01Btn = document.createElement('button');
      bay01Btn.type = 'button';
      bay01Btn.className = 'ed-cta-btn';
      bay01Btn.dataset.testid = 'embed-cta-bay01';
      bay01Btn.textContent = '→ Bay 01 Mamba-2';
      bay01Btn.addEventListener('click', () => this.cb.onInspectInternals('M', 1));
      actions.appendChild(bay01Btn);
    } else if (state.mode === 'cutaway_output') {
      const bay52Btn = document.createElement('button');
      bay52Btn.type = 'button';
      bay52Btn.className = 'ed-cta-btn';
      bay52Btn.dataset.testid = 'output-cta-bay52';
      bay52Btn.textContent = '← Bay 52 Final MoE';
      bay52Btn.addEventListener('click', () => this.cb.onInspectInternals('E', 52));
      actions.appendChild(bay52Btn);
    }
    const backBtn = document.createElement('button');
    backBtn.type = 'button';
    backBtn.className = 'ed-cta-btn ed-cta-secondary';
    backBtn.textContent = '← Back to 52-Layer Tower';
    backBtn.addEventListener('click', () => this.cb.onSelectMode('tower'));
    actions.appendChild(backBtn);

    if (state.mode === 'cutaway_embedding') {
      panel.append(header, p);
      const embCard = this.buildEmbeddingCutawayInspector(state, activeExecution, decodeSession);
      if (embCard) panel.appendChild(embCard);
      panel.appendChild(list);
    } else if (state.mode === 'cutaway_output') {
      panel.append(header, p);
      if (state.executionMode !== 'ar_decode' || !decodeSession) {
        const outCard = this.buildOutputCutawayInspector(state, activeExecution, decodeSession);
        if (outCard) panel.appendChild(outCard);
      }
      panel.appendChild(list);
    } else if (state.mode === 'cutaway_moe') {
      panel.append(header, p, this.buildMoeInspectorControls(state, cutaway), list);
    } else if (state.mode === 'cutaway_mamba') {
      const effHead = cutaway?.mambaStats?.activeHead ?? state.mambaActiveHead;
      panel.append(
        header,
        p,
        this.buildMambaTemporalControls(state, cutaway),
        this.buildMambaHeadSelector(effHead, state.selectedPart, state.hoveredPart),
        list,
      );
    } else {
      panel.append(header, p, list);
      if (state.mode === 'cutaway_gqa') {
        panel.appendChild(this.buildGqaHeadSelector(state.selectedPart, state.hoveredPart));
      }
    }
    panel.appendChild(actions);
  }

  /** Compact Focused Token (t0..t_{T-1}) & Top-6 Winning Expert selector inside MoE inspector. */
  private buildMoeInspectorControls(
    state: SceneState,
    cutaway: CutawaySceneGeometry | null,
  ): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector';
    wrap.setAttribute('data-testid', 'moe-inspector-controls');

    const sim = cutaway?.moeSim ?? null;
    const simTokens = sim?.tokens ?? state.tokens;
    const focusTok = Math.min(
      simTokens - 1,
      cutaway?.moeStats?.focusToken ?? Math.max(0, simTokens - 1),
    );
    const absFocusTok = (sim?.startTokenOffset ?? 0) + focusTok;
    const focusExp = cutaway?.moeStats?.focusExpert ?? cutaway?.moeStats?.activeExpertIds[0] ?? 0;

    // 1. Focused Token Row + Lightweight Per-Token Cycling (0 B sequence cache!)
    const tokTop = document.createElement('div');
    tokTop.className = 'qhead-top-row';
    const tokHeading = document.createElement('span');
    tokHeading.className = 'qhead-heading';
    const tokLabel = sim?.tokenLabels[focusTok] ?? `t${absFocusTok}`;
    tokHeading.textContent = `Focused Token: t${absFocusTok} "${tokLabel}" (${focusTok + 1}/${simTokens})`;
    tokTop.appendChild(tokHeading);
    wrap.appendChild(tokTop);

    const cycleRow = document.createElement('div');
    cycleRow.className = 'mtemp-controls-row';

    const prevTokBtn = document.createElement('button');
    prevTokBtn.type = 'button';
    prevTokBtn.className = 'mtemp-btn';
    prevTokBtn.dataset.testid = 'moe-t-prev';
    prevTokBtn.textContent = '◀';
    prevTokBtn.title = 'Previous token (t − 1)';
    prevTokBtn.disabled = simTokens <= 1;
    prevTokBtn.addEventListener('click', () => this.cb.onStepMoeFocusToken(-1));

    const cycleBtn = document.createElement('button');
    cycleBtn.type = 'button';
    cycleBtn.className = state.moeTokenCycling
      ? 'mtemp-btn mtemp-play-btn is-playing'
      : 'mtemp-btn mtemp-play-btn';
    cycleBtn.dataset.testid = 'moe-t-cycle';
    cycleBtn.textContent = state.moeTokenCycling ? '❚❚ Pause t' : '▶ Cycle t';
    cycleBtn.title = 'Cycle tokens t = 0 → T−1 to watch Top-6 active experts switch independently (0 B cache)';
    cycleBtn.disabled = simTokens <= 1;
    cycleBtn.setAttribute('aria-pressed', String(state.moeTokenCycling));
    cycleBtn.addEventListener('click', () => this.cb.onToggleMoeTokenCycle());

    const nextTokBtn = document.createElement('button');
    nextTokBtn.type = 'button';
    nextTokBtn.className = 'mtemp-btn';
    nextTokBtn.dataset.testid = 'moe-t-next';
    nextTokBtn.textContent = '▶';
    nextTokBtn.title = 'Next token (t + 1)';
    nextTokBtn.disabled = simTokens <= 1;
    nextTokBtn.addEventListener('click', () => this.cb.onStepMoeFocusToken(1));

    cycleRow.append(prevTokBtn, cycleBtn, nextTokBtn);
    wrap.appendChild(cycleRow);

    const tokGrid = document.createElement('div');
    tokGrid.className = 'moe-token-grid';
    for (let t = 0; t < simTokens; t++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'moe-tok-btn';
      btn.dataset.moeToken = String(t);
      const absT = (sim?.startTokenOffset ?? 0) + t;
      const lbl = sim?.tokenLabels[t] ?? '';
      btn.textContent = `t${absT} ${lbl}`;
      btn.title = `Inspect routing for token t${absT} "${lbl}"`;
      const isSel = t === focusTok;
      btn.classList.toggle('is-selected', isSel);
      btn.setAttribute('aria-pressed', String(isSel));
      btn.addEventListener('click', () => this.cb.onSelectMoeFocusToken(t));
      tokGrid.appendChild(btn);
    }
    wrap.appendChild(tokGrid);

    // 2. Top-6 Active Experts Row (clickable to sync detail chain)
    if (sim) {
      const expTop = document.createElement('div');
      expTop.className = 'qhead-group-label';
      expTop.textContent = `Top-6 Active Experts for t${absFocusTok} (×2.5 scaled, click to inspect):`;
      wrap.appendChild(expTop);

      const expGrid = document.createElement('div');
      expGrid.className = 'moe-topk-grid';
      for (let j = 0; j < sim.topK; j++) {
        const eid = sim.topk_indices[focusTok * sim.topK + j];
        const w = sim.topk_weights[focusTok * sim.topK + j];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'moe-exp-btn';
        btn.dataset.moeExpert = String(eid);
        const isSel = eid === focusExp;
        btn.classList.toggle('is-selected', isSel);
        btn.setAttribute('aria-pressed', String(isSel));
        btn.innerHTML = `<span>${formatExpertId(eid)}</span><span class="moe-exp-w">${w.toFixed(2)}</span>`;
        btn.addEventListener('click', () => this.cb.onSelectMoeExpert(eid));
        expGrid.appendChild(btn);
      }
      wrap.appendChild(expGrid);
    }

    return wrap;
  }

  /** Interactive Q00..Q31 head selector grouped into 16:1 KV banks inside GQA inspector. */
  private buildGqaHeadSelector(selectedPart: string | null, hoveredPart: string | null): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector';
    wrap.setAttribute('data-testid', 'gqa-head-selector');

    const activeHead = parseQueryHeadPartId(hoveredPart ?? selectedPart);
    const pinnedHead = parseQueryHeadPartId(selectedPart);

    const topRow = document.createElement('div');
    topRow.className = 'qhead-top-row';
    const label = document.createElement('span');
    label.className = 'qhead-heading';
    label.textContent =
      activeHead !== null
        ? `Query Head ${formatQueryHeadId(activeHead)} → KV${activeHead < 16 ? 0 : 1}`
        : 'Select Query Head (Q00–Q31)';
    topRow.appendChild(label);

    if (pinnedHead !== null) {
      const resetHead = document.createElement('button');
      resetHead.type = 'button';
      resetHead.className = 'qhead-clear-btn';
      resetHead.textContent = 'All 32';
      resetHead.addEventListener('click', () => this.cb.onSelectPart(null));
      topRow.appendChild(resetHead);
    }
    wrap.appendChild(topRow);

    const groups: readonly { title: string; start: number; end: number }[] = [
      { title: 'Group 0 → KV₀ (Q00–Q15)', start: 0, end: 15 },
      { title: 'Group 1 → KV₁ (Q16–Q31)', start: 16, end: 31 },
    ];

    for (const grp of groups) {
      const grpEl = document.createElement('div');
      grpEl.className = 'qhead-group';

      const grpTitle = document.createElement('div');
      grpTitle.className = 'qhead-group-label';
      grpTitle.textContent = grp.title;
      grpEl.appendChild(grpTitle);

      const grid = document.createElement('div');
      grid.className = 'qhead-grid';
      for (let i = grp.start; i <= grp.end; i++) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qhead-btn';
        btn.dataset.qhead = String(i);
        btn.textContent = formatQueryHeadId(i);
        const isSel = pinnedHead === i;
        const isActive = activeHead === i;
        btn.classList.toggle('is-selected', isSel);
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', String(isSel));
        btn.addEventListener('click', () => {
          this.cb.onSelectPart(isSel ? null : `q-head-${i}`);
        });
        grid.appendChild(btn);
      }
      grpEl.appendChild(grid);
      wrap.appendChild(grpEl);
    }

    return wrap;
  }

  /** Compact Recurrence Time (t = 0..T-1) playback & zero-padding control inside Mamba-2 inspector. */
  private buildMambaTemporalControls(
    state: SceneState,
    cutaway: CutawaySceneGeometry | null,
  ): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector mamba-temporal-box';
    wrap.setAttribute('data-testid', 'mamba-temporal-controls');

    const sim = cutaway?.mambaSim ?? null;
    const simTokens = sim?.tokens ?? state.tokens;
    const tok = Math.max(
      0,
      Math.min(
        simTokens - 1,
        cutaway?.mambaStats?.stateTokenIndex ?? state.mambaStateToken ?? simTokens - 1,
      ),
    );
    const absTok = (sim?.startTokenOffset ?? 0) + tok;
    const tokLabel = sim?.tokenLabels[tok] ?? `t${absTok}`;
    const zeroPads = Math.max(0, 3 - absTok);

    const topRow = document.createElement('div');
    topRow.className = 'qhead-top-row';
    const heading = document.createElement('span');
    heading.className = 'qhead-heading';
    heading.textContent = `Recurrence t${absTok} "${tokLabel}" (${tok + 1}/${simTokens})`;
    topRow.appendChild(heading);

    const padPill = document.createElement('span');
    padPill.className = zeroPads > 0 ? 'mtemp-pad-pill is-padded' : 'mtemp-pad-pill';
    padPill.textContent =
      zeroPads > 0
        ? `${zeroPads} zero-pad`
        : sim?.hasInitialState
          ? 'carried 4-tap'
          : 'full 4-tap';
    padPill.title =
      zeroPads > 0
        ? `Causal Conv1D left-pads ${zeroPads} slot${zeroPads === 1 ? '' : 's'} with 0 at t=${absTok}`
        : sim?.hasInitialState
          ? '3 taps carried from prior Conv1D boundary state + 1 newly computed tap'
          : 'All 4 causal Conv1D taps hold live token inputs (t−3..t)';
    topRow.appendChild(padPill);
    wrap.appendChild(topRow);

    const ctrlRow = document.createElement('div');
    ctrlRow.className = 'mtemp-controls-row';

    const prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.className = 'mtemp-btn';
    prevBtn.dataset.testid = 'mamba-t-prev';
    prevBtn.textContent = '◀';
    prevBtn.title = 'Previous recurrence token (t − 1)';
    prevBtn.disabled = simTokens <= 1;
    prevBtn.addEventListener('click', () => this.cb.onStepMambaStateToken(-1));

    const playBtn = document.createElement('button');
    playBtn.type = 'button';
    playBtn.className = state.mambaTemporalPlaying
      ? 'mtemp-btn mtemp-play-btn is-playing'
      : 'mtemp-btn mtemp-play-btn';
    playBtn.dataset.testid = 'mamba-t-play';
    playBtn.textContent = state.mambaTemporalPlaying ? '❚❚ Pause t' : '▶ Step t';
    playBtn.title =
      'Play token recurrence t = 0 → T−1 to watch Conv1D shift & Hₜ overwrite in place';
    playBtn.disabled = simTokens <= 1;
    playBtn.setAttribute('aria-pressed', String(state.mambaTemporalPlaying));
    playBtn.addEventListener('click', () => this.cb.onToggleMambaTemporalPlay());

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'mtemp-btn';
    nextBtn.dataset.testid = 'mamba-t-next';
    nextBtn.textContent = '▶';
    nextBtn.title = 'Next recurrence token (t + 1)';
    nextBtn.disabled = simTokens <= 1;
    nextBtn.addEventListener('click', () => this.cb.onStepMambaStateToken(1));

    ctrlRow.append(prevBtn, playBtn, nextBtn);

    if (state.mambaStateToken !== null) {
      const latestBtn = document.createElement('button');
      latestBtn.type = 'button';
      latestBtn.className = 'qhead-clear-btn';
      latestBtn.dataset.testid = 'mamba-t-latest';
      latestBtn.textContent = 'Latest';
      latestBtn.title = 'Follow latest sequence token (T − 1)';
      latestBtn.addEventListener('click', () => this.cb.onSelectMambaStateToken(null));
      ctrlRow.appendChild(latestBtn);
    }
    wrap.appendChild(ctrlRow);

    const tokGrid = document.createElement('div');
    tokGrid.className = 'mtemp-token-grid';
    for (let t = 0; t < simTokens; t++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'mtemp-tok-btn';
      btn.dataset.mambaToken = String(t);
      const absT = (sim?.startTokenOffset ?? 0) + t;
      const lbl = sim?.tokenLabels[t] ?? '';
      btn.textContent = `t${absT}`;
      btn.title = `Inspect recurrence at t${absT} "${lbl}"`;
      const isSel = t === tok;
      btn.classList.toggle('is-selected', isSel);
      btn.setAttribute('aria-pressed', String(isSel));
      btn.addEventListener('click', () => this.cb.onSelectMambaStateToken(t));
      tokGrid.appendChild(btn);
    }
    wrap.appendChild(tokGrid);

    const note = document.createElement('div');
    note.className = 'mtemp-note';
    note.textContent = 'Browser-only H_history playback · Prod keeps 1 fixed buffer (47.1 MiB)';
    wrap.appendChild(note);

    return wrap;
  }

  /** Interactive H00..H15 head selector grouped into 8 B/C groups (G0..G7, 2 mini heads/group). */
  private buildMambaHeadSelector(
    activeMambaHead: number,
    selectedPart: string | null,
    hoveredPart: string | null,
  ): HTMLDivElement {
    const wrap = document.createElement('div');
    wrap.className = 'qhead-selector';
    wrap.setAttribute('data-testid', 'mamba-head-selector');

    const partHead = parseMambaHeadPartId(hoveredPart ?? selectedPart);
    const activeHead = partHead ?? activeMambaHead;
    const pinnedHead = parseMambaHeadPartId(selectedPart);

    const topRow = document.createElement('div');
    topRow.className = 'qhead-top-row';
    const label = document.createElement('span');
    label.className = 'qhead-heading';
    label.textContent = `SSM Head ${formatMambaHeadId(activeHead)} · Group G${mambaGroupForHead(activeHead)} (16 mini / 64 prod)`;
    topRow.appendChild(label);

    if (pinnedHead !== null) {
      const resetHead = document.createElement('button');
      resetHead.type = 'button';
      resetHead.className = 'qhead-clear-btn';
      resetHead.textContent = 'Clear';
      resetHead.addEventListener('click', () => this.cb.onSelectPart(null));
      topRow.appendChild(resetHead);
    }
    wrap.appendChild(topRow);

    const groupsGrid = document.createElement('div');
    groupsGrid.className = 'mhead-groups-grid';

    for (let g = 0; g < 8; g++) {
      const pair = document.createElement('div');
      pair.className = 'mhead-pair';

      const pairLabel = document.createElement('div');
      pairLabel.className = 'mhead-pair-label';
      pairLabel.textContent = `G${g}`;
      pair.appendChild(pairLabel);

      const row = document.createElement('div');
      row.className = 'mhead-pair-row';
      for (let local = 0; local < 2; local++) {
        const h = g * 2 + local;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qhead-btn';
        btn.dataset.mhead = String(h);
        btn.textContent = formatMambaHeadId(h);
        const isSel = pinnedHead === h || (pinnedHead === null && activeMambaHead === h);
        const isActive = activeHead === h;
        btn.classList.toggle('is-selected', isSel);
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', String(isSel));
        btn.addEventListener('click', () => {
          this.cb.onSelectPart(pinnedHead === h ? null : `mamba-head-${h}`);
        });
        row.appendChild(btn);
      }
      pair.appendChild(row);
      groupsGrid.appendChild(pair);
    }

    wrap.appendChild(groupsGrid);
    return wrap;
  }

  dispose(): void {
    for (const fn of this.cleanupFns) fn();
    this.cleanupFns.length = 0;
  }
}

export type ControlsUI = MacroControls;
