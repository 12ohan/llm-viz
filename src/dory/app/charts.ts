/**
 * Pure Headless 2D Analytical `<svg>` Charts & Editorial Panel for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/app/charts.ts`).
 *
 * Implements `DoryChartsModuleContract` (`DORY6` Slice 6A, Locked Decisions `D2`, `D3`, `D4`, `D7`;
 * Guards `DR6`, `DR7`, `DR13`, `DR17`, `DR19`, `DR23`, Finding 9):
 *   1. `renderSpanningSubspaceSvg`    — Exact 2D Spanning-Subspace NLERP (`span(a, b)`) + 8-channel
 *                                       effective `|alpha_hat|` bar strip (highlighting `> 1` extrapolation).
 *   2. `renderExploratoryPcaSvg`      — Exploratory 2D PCA trajectory of actual 8D states, explicitly
 *                                       badged `PROJECTED NORMS <= 1; ALL METRICS COMPUTED IN 8D`,
 *                                       paired with 8D numeric table & local `120°` anchor-map caveat.
 *   3. `renderDualRopePhasorsSvg`     — 4-plane NeoX Dual-RoPE phasor dials contrasting SWA `theta_S=16`
 *                                       (`inv_freq=[1, 1/2, 1/4, 1/8]`) vs. Full `theta_F=1024` (`DR6`).
 *   4. `renderProductionKvScalingSvg` — Production `1 -> 256K` BF16 logical KV scaling chart calling out
 *                                       both the `S=1,024` `4x` SWA slope knee and exact `S`-invariant
 *                                       `(4 + 5k) / 9` ratio across `last_loop` vs. `per_loop` (`k=2,4,8`).
 *   5. `renderSelfSpecProbsSvg`       — Reconstructed Paper §6.2 Contract A (`104` banks) `(q_j, p_j)`
 *                                       division-free binary64 acceptance, exact residual correction bars,
 *                                       and `cap_swa=16` SWA ring rollback strip (`DR19`, `DR23`).
 *   6. `renderTbptlSchematicSvg`      — Strictly `READ-ONLY · NO GRADIENTS` TBPTL depth-1 stop-gradient
 *                                       wall (`sg[s_{k-1}]`), Prelude `+x` gradient path, and 5-stage timeline.
 *
 * Strict `D4` Architectural Boundary Isolation:
 *   - ZERO imports from Gate 4b/4c Nano modules or sibling visualizer apps.
 *   - Pure SVG renderers require ZERO DOM and work cleanly in headless Vitest Node runs.
 *   - 100% clean-room original TypeScript — zero runtime npm deps, zero ambient RNG.
 */

import { fmtSigned, formatBytesAdaptive } from '../../shared/format';
import {
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_TBPTL_READONLY_BADGE,
  type DoryGqaRopeViewMetadata,
  type DoryHypersphereViewMetadata,
  type DorySelfSpecViewMetadata,
  type DorySpanningSubspaceDiagramData,
  type DorySpatialScene,
} from '../scene/types';
import type {
  DoryArchReadoutsBundle,
  DoryProductionKvScalingRow,
  DoryTbptlSchematicReadout,
} from '../story/types';
import { DORY_NLERP_MARKER } from '../types';
import type {
  DoryAnalyticalChartsPanelHandle,
  DoryChartsModuleContract,
  DoryKvScalingChartInput,
  DorySvgChartDimensions,
  DorySvgChartKind,
  DorySvgChartRenderResult,
} from './types';

// ============================================================================
// EDITORIAL #F7F8FA DRAFTING-TABLE PALETTE & PURE SVG UTILITIES
// ============================================================================

export const DORY_PCA_COMPACT_BADGE =
  'PROJECTED NORMS <= 1; ALL METRICS COMPUTED IN 8D' as const;

const PALETTE = Object.freeze({
  bgSurface: '#F7F8FA',
  bgCard: '#FFFFFF',
  bgSubtle: '#EFF2F6',
  grid: '#E2E8F0',
  axis: '#94A3B8',
  border: '#CBD5E1',
  ink: '#18202C',
  inkSecondary: '#475569',
  inkMuted: '#64748B',
  preludeIndigo: '#3861BD',
  preludeBg: '#EEF3FC',
  recurrentTeal: '#1A8A7A',
  recurrentBg: '#EAF6F4',
  codaAmber: '#B56E06',
  codaBg: '#FEF6E4',
  swaCyan: '#156C96',
  swaBg: '#E8F5FC',
  fullIndigo: '#3B49B2',
  nlerpEmerald: '#157A4F',
  nlerpBg: '#E8F7F0',
  specOxide: '#C63A24',
  specBg: '#FDF0ED',
  readonlySlate: '#475569',
  readonlyBg: '#EEF1F5',
});

const FONT_MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

function clampDim(val: number | undefined, fallback: number, min: number, max: number): number {
  if (val === undefined || !Number.isFinite(val)) return fallback;
  return Math.max(min, Math.min(max, Math.round(val)));
}

function safeNum(v: number, fallback = 0): number {
  return Number.isFinite(v) ? v : fallback;
}

function f1(v: number): string {
  return safeNum(v).toFixed(1);
}

function f2(v: number): string {
  return safeNum(v).toFixed(2);
}

function escXml(raw: string): string {
  return raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function truncateText(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  if (maxLen <= 1) return text.slice(0, maxLen);
  return `${text.slice(0, maxLen - 1)}…`;
}

function wrapSvg(
  width: number,
  height: number,
  ariaLabel: string,
  kind: DorySvgChartKind,
  body: string,
): string {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" ` +
    `width="${width}" height="${height}" role="img" aria-label="${escXml(ariaLabel)}" ` +
    `class="dory-chart-svg dory-chart-svg--${kind}">` +
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="6" fill="${PALETTE.bgCard}" stroke="${PALETTE.border}" stroke-width="1"/>` +
    body +
    `</svg>`
  );
}

function buildArrowHeadPath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  headLen = 6,
  halfSpreadRad = 0.44,
): string {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy);
  if (dist < 2) return '';
  const angle = Math.atan2(dy, dx);
  const ax = x2 - headLen * Math.cos(angle - halfSpreadRad);
  const ay = y2 - headLen * Math.sin(angle - halfSpreadRad);
  const bx = x2 - headLen * Math.cos(angle + halfSpreadRad);
  const by = y2 - headLen * Math.sin(angle + halfSpreadRad);
  return `M${f1(ax)},${f1(ay)}L${f1(x2)},${f1(y2)}L${f1(bx)},${f1(by)}`;
}

// ============================================================================
// CHART 1: EXACT 2D SPANNING-SUBSPACE NLERP DIAGRAM (`span(a, b)`)
// ============================================================================

/**
 * Renders View 2 Diagram (a): the **Exact 2D Spanning-Subspace NLERP Diagram** (`span(a, b)`)
 * paired with an 8-channel effective `|alpha_hat_c|` strip highlighting unclamped `> 1` channels.
 */
export function renderSpanningSubspaceSvg(
  data: DorySpanningSubspaceDiagramData,
  dims?: DorySvgChartDimensions,
): DorySvgChartRenderResult {
  const width = clampDim(dims?.width, 392, 260, 1200);
  const height = clampDim(dims?.height, 236, 170, 900);

  const leftPanelW = Math.round(width * 0.56);
  const cx = Math.round(leftPanelW * 0.52);
  const cy = Math.round(height * 0.55);
  const radius = Math.max(42, Math.min(leftPanelW * 0.36, height * 0.33));

  const toPx = (ux: number, uy: number): readonly [number, number] => [
    cx + safeNum(ux) * radius,
    cy - safeNum(uy) * radius,
  ];

  const [ax, ay] = toPx(data.a2d[0], data.a2d[1]);
  const [bx, by] = toPx(data.b2d[0], data.b2d[1]);
  const [preX, preY] = toPx(data.projectedBlendedPreNorm2d[0], data.projectedBlendedPreNorm2d[1]);
  const [outX, outY] = toPx(data.projectedOutput2d[0], data.projectedOutput2d[1]);

  // Split exact uniform-scalar NLERP curve into interpolation [0..1] vs. extrapolation (> 1)
  const interpPts: string[] = [];
  const extrapPts: string[] = [];
  for (let i = 0; i < data.uniformAlphaCurve.length; i += 1) {
    const pt = data.uniformAlphaCurve[i]!;
    const [px, py] = toPx(pt.x, pt.y);
    const coord = `${f1(px)},${f1(py)}`;
    if (pt.isExtrapolation) {
      if (extrapPts.length === 0 && interpPts.length > 0) {
        extrapPts.push(interpPts[interpPts.length - 1]!);
      }
      extrapPts.push(coord);
    } else {
      interpPts.push(coord);
    }
  }

  // Angle arc omega between a2d=(1,0) and b2d=(cos omega, sin omega)
  const omegaRad = safeNum(data.omegaRadians, 0);
  const arcR = radius * 0.32;
  const arcEndX = cx + arcR * Math.cos(omegaRad);
  const arcEndY = cy - arcR * Math.sin(omegaRad);
  const largeArc = Math.abs(omegaRad) > Math.PI ? 1 : 0;
  const sweepFlag = omegaRad >= 0 ? 0 : 1;
  const arcPath =
    Math.abs(omegaRad) > 0.04
      ? `M${f1(cx + arcR)},${f1(cy)}A${f1(arcR)},${f1(arcR)} 0 ${largeArc} ${sweepFlag} ${f1(arcEndX)},${f1(arcEndY)}`
      : '';

  const parts: string[] = [];

  // Header strip inside SVG
  parts.push(
    `<rect x="8" y="8" width="${width - 16}" height="20" rx="4" fill="${PALETTE.nlerpBg}" stroke="#A3DEC2" stroke-width="0.8"/>`,
    `<text x="14" y="21" font-family="${FONT_MONO}" font-size="9" font-weight="700" fill="${PALETTE.nlerpEmerald}">EXACT 2D SPAN(a, b) · nGPT NLERP (NOT TRIG SLERP)</text>`,
  );

  // Unit circle & orthonormal basis crosshairs
  parts.push(
    `<circle cx="${cx}" cy="${cy}" r="${f1(radius)}" fill="${PALETTE.bgSurface}" stroke="${PALETTE.axis}" stroke-width="1" stroke-dasharray="3 2.5"/>`,
    `<line x1="${f1(cx - radius - 8)}" y1="${cy}" x2="${f1(cx + radius + 10)}" y2="${cy}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<line x1="${cx}" y1="${f1(cy - radius - 8)}" x2="${cx}" y2="${f1(cy + radius + 8)}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<text x="${f1(cx + radius + 3)}" y="${f1(cy + 11)}" font-family="${FONT_MONO}" font-size="8" fill="${PALETTE.inkMuted}">e1</text>`,
    `<text x="${f1(cx + 4)}" y="${f1(cy - radius - 2)}" font-family="${FONT_MONO}" font-size="8" fill="${PALETTE.inkMuted}">e2</text>`,
  );

  // Linear chord a -> b before unit-sphere projection
  parts.push(
    `<line x1="${f1(ax)}" y1="${f1(ay)}" x2="${f1(bx)}" y2="${f1(by)}" stroke="${PALETTE.codaAmber}" stroke-width="1.2" stroke-dasharray="3 2"/>`,
  );

  // Uniform-scalar NLERP arcs
  if (interpPts.length >= 2) {
    parts.push(
      `<polyline points="${interpPts.join(' ')}" fill="none" stroke="${PALETTE.nlerpEmerald}" stroke-width="2.4" stroke-linecap="round"/>`,
    );
  }
  if (extrapPts.length >= 2) {
    parts.push(
      `<polyline points="${extrapPts.join(' ')}" fill="none" stroke="${PALETTE.specOxide}" stroke-width="2.4" stroke-dasharray="4 2" stroke-linecap="round"/>`,
    );
  }
  if (arcPath) {
    parts.push(`<path d="${arcPath}" fill="none" stroke="${PALETTE.inkSecondary}" stroke-width="1"/>`);
  }

  // Vectors a, b, pre-norm blend, and projected actual 8D output
  parts.push(
    `<line x1="${cx}" y1="${cy}" x2="${f1(ax)}" y2="${f1(ay)}" stroke="${PALETTE.preludeIndigo}" stroke-width="1.8"/>`,
    `<path d="${buildArrowHeadPath(cx, cy, ax, ay, 5.5)}" fill="none" stroke="${PALETTE.preludeIndigo}" stroke-width="1.5"/>`,
    `<circle cx="${f1(ax)}" cy="${f1(ay)}" r="3.4" fill="${PALETTE.preludeIndigo}" stroke="#FFFFFF" stroke-width="1"/>`,
    `<text x="${f1(ax - 12)}" y="${f1(ay + 12)}" font-family="${FONT_MONO}" font-size="8.5" font-weight="700" fill="${PALETTE.preludeIndigo}">a=N(x)</text>`,

    `<line x1="${cx}" y1="${cy}" x2="${f1(bx)}" y2="${f1(by)}" stroke="${PALETTE.recurrentTeal}" stroke-width="1.8"/>`,
    `<path d="${buildArrowHeadPath(cx, cy, bx, by, 5.5)}" fill="none" stroke="${PALETTE.recurrentTeal}" stroke-width="1.5"/>`,
    `<circle cx="${f1(bx)}" cy="${f1(by)}" r="3.4" fill="${PALETTE.recurrentTeal}" stroke="#FFFFFF" stroke-width="1"/>`,
    `<text x="${f1(clampDim(bx - 16, 12, 10, leftPanelW - 44))}" y="${f1(clampDim(by - 8, 42, 38, height - 24))}" font-family="${FONT_MONO}" font-size="8.5" font-weight="700" fill="${PALETTE.recurrentTeal}">b=N(o)</text>`,

    // Pre-norm per-channel blend projection
    `<line x1="${cx}" y1="${cy}" x2="${f1(preX)}" y2="${f1(preY)}" stroke="${PALETTE.codaAmber}" stroke-width="1" stroke-dasharray="2 2"/>`,
    `<circle cx="${f1(preX)}" cy="${f1(preY)}" r="3" fill="${PALETTE.bgCard}" stroke="${PALETTE.codaAmber}" stroke-width="1.4"/>`,

    // Actual 8D NLERP output orthogonal projection onto span(a, b)
    `<line x1="${cx}" y1="${cy}" x2="${f1(outX)}" y2="${f1(outY)}" stroke="${PALETTE.specOxide}" stroke-width="2"/>`,
    `<circle cx="${f1(outX)}" cy="${f1(outY)}" r="3.8" fill="${PALETTE.specOxide}" stroke="#FFFFFF" stroke-width="1.1"/>`,
  );

  // Left-panel footer summary metrics
  parts.push(
    `<text x="12" y="${height - 19}" font-family="${FONT_MONO}" font-size="8.5" fill="${PALETTE.inkSecondary}">ω=${safeNum(data.omegaDegrees).toFixed(1)}° · cos=${safeNum(data.cosineAB).toFixed(3)}</text>`,
    `<text x="12" y="${height - 8}" font-family="${FONT_MONO}" font-size="8.5" fill="${PALETTE.specOxide}">‖x_next − Π‖=${safeNum(data.outOfPlaneNorm).toFixed(4)} (per-ch)</text>`,
  );

  // Right panel: 8-channel effective |alpha_hat_c| bar strip (highlighting > 1.0 extrapolation)
  const rightX = leftPanelW + 6;
  const rightW = Math.max(90, width - rightX - 10);
  const barTop = 48;
  const chCount = Math.min(8, data.effectiveAlpha.length);
  const rowPitch = Math.max(15, Math.floor((height - barTop - 38) / Math.max(1, chCount)));
  const maxLam = Math.max(1.25, safeNum(data.maxEffectiveAlpha, 1.0) * 1.08);
  const labelColW = 20;
  const valColW = 32;
  const trackX = rightX + labelColW;
  const trackW = Math.max(36, rightW - labelColW - valColW);
  const refOneX = trackX + (1.0 / maxLam) * trackW;

  parts.push(
    `<line x1="${leftPanelW}" y1="34" x2="${leftPanelW}" y2="${height - 10}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<text x="${rightX}" y="42" font-family="${FONT_MONO}" font-size="8.5" font-weight="700" fill="${PALETTE.ink}">Per-Ch |α̂_c| (unclamped)</text>`,
    `<line x1="${f1(refOneX)}" y1="${barTop - 3}" x2="${f1(refOneX)}" y2="${barTop + chCount * rowPitch + 2}" stroke="${PALETTE.specOxide}" stroke-width="1" stroke-dasharray="2.5 2"/>`,
    `<text x="${f1(refOneX - 8)}" y="${barTop + chCount * rowPitch + 12}" font-family="${FONT_MONO}" font-size="7.5" fill="${PALETTE.specOxide}">λ=1</text>`,
  );

  for (let c = 0; c < chCount; c += 1) {
    const lam = Math.max(0, safeNum(data.effectiveAlpha[c] ?? 0));
    const isExtrap = lam > 1.0;
    const ry = barTop + c * rowPitch;
    const barW = Math.max(1, Math.min(trackW, (lam / maxLam) * trackW));
    const fill = isExtrap ? PALETTE.specOxide : PALETTE.recurrentTeal;

    parts.push(
      `<text x="${rightX}" y="${ry + 9}" font-family="${FONT_MONO}" font-size="8" fill="${isExtrap ? PALETTE.specOxide : PALETTE.inkSecondary}">d${c}</text>`,
      `<rect x="${trackX}" y="${ry + 1}" width="${f1(trackW)}" height="10" rx="2" fill="${PALETTE.bgSubtle}"/>`,
      `<rect x="${trackX}" y="${ry + 1}" width="${f1(barW)}" height="10" rx="2" fill="${fill}"/>`,
      `<text x="${f1(trackX + trackW + 3)}" y="${ry + 9}" font-family="${FONT_MONO}" font-size="8" font-weight="${isExtrap ? '700' : '500'}" fill="${isExtrap ? PALETTE.specOxide : PALETTE.ink}">${f2(lam)}</text>`,
    );
  }

  parts.push(
    `<text x="${rightX}" y="${height - 8}" font-family="${FONT_MONO}" font-size="8" fill="${PALETTE.inkMuted}">mean ${f2(data.meanEffectiveAlpha)} · max ${f2(data.maxEffectiveAlpha)}</text>`,
  );

  const title = data.diagramTitle || 'Exact 2D Spanning-Subspace NLERP — span(a, b)';
  const ariaLabel =
    `${title}. Exact angle omega ${safeNum(data.omegaDegrees).toFixed(1)} degrees, ` +
    `out-of-plane norm ${safeNum(data.outOfPlaneNorm).toFixed(4)}, ` +
    `max effective alpha ${f2(data.maxEffectiveAlpha)}.`;

  const extrapNote =
    data.extrapolatedChannels.length > 0
      ? `Unclamped extrapolation |alpha_hat| > 1 active on channels [${data.extrapolatedChannels.join(', ')}] (max ${f2(data.maxEffectiveAlpha)}).`
      : `All 8 channels currently inside [0, 1] (max ${f2(data.maxEffectiveAlpha)}); NLERP remains unclamped (>= 0).`;

  return Object.freeze({
    kind: 'spanning_subspace',
    title,
    badge: 'EXACT 2D SPAN(a, b) · nGPT NLERP (NOT SLERP)',
    ariaLabel,
    width,
    height,
    svgMarkup: wrapSvg(width, height, ariaLabel, 'spanning_subspace', parts.join('')),
    epistemicCaveats: Object.freeze([
      `${DORY_ANTI_SLERP_PLAQUE} (${DORY_NLERP_MARKER}).`,
      `Exact 2D orthonormal span(a, b): per-channel rates induce out-of-plane component ||x_next - Pi|| = ${safeNum(data.outOfPlaneNorm).toFixed(4)}. ${extrapNote}`,
    ]),
  });
}

// ============================================================================
// CHART 2: EXPLORATORY 2D PCA TRAJECTORY PROJECTION (8D METRICS ONLY)
// ============================================================================

/**
 * Renders View 2 Diagram (b): the **Exploratory 2D PCA Trajectory Projection** of actual 8D
 * latent states `{x, u_1, s_1, ..., u_k, s_k}`, explicitly badged
 * `'PROJECTED NORMS <= 1; ALL METRICS COMPUTED IN 8D'` with an 8D numeric table and local `120°`
 * anchor-map Jacobian caveat (Finding 9).
 */
export function renderExploratoryPcaSvg(
  metadata: DoryHypersphereViewMetadata,
  dims?: DorySvgChartDimensions,
): DorySvgChartRenderResult {
  const width = clampDim(dims?.width, 392, 260, 1200);
  const height = clampDim(dims?.height, 246, 180, 900);

  const pca = metadata.exploratoryPca;
  const leftPanelW = Math.round(width * 0.50);
  const cx = Math.round(leftPanelW * 0.52);
  const cy = Math.round(height * 0.57);
  const radius = Math.max(40, Math.min(leftPanelW * 0.37, height * 0.30));

  const toPx = (px: number, py: number): readonly [number, number] => [
    cx + safeNum(px) * radius,
    cy - safeNum(py) * radius,
  ];

  const ev1Pct = (safeNum(pca.explainedVarianceRatio[0]) * 100).toFixed(1);
  const ev2Pct = (safeNum(pca.explainedVarianceRatio[1]) * 100).toFixed(1);

  const parts: string[] = [];

  // Mandatory top epistemic disclosure banner inside SVG
  parts.push(
    `<rect x="8" y="8" width="${width - 16}" height="20" rx="4" fill="${PALETTE.codaBg}" stroke="#F0C975" stroke-width="0.9"/>`,
    `<text x="14" y="21" font-family="${FONT_MONO}" font-size="8.5" font-weight="700" fill="${PALETTE.codaAmber}">${escXml(DORY_PCA_COMPACT_BADGE)}</text>`,
  );

  // Unit disk D^2 (projected norm <= 1.0 visual disclosure)
  parts.push(
    `<circle cx="${cx}" cy="${cy}" r="${f1(radius)}" fill="${PALETTE.bgSurface}" stroke="${PALETTE.axis}" stroke-width="1" stroke-dasharray="3 2.5"/>`,
    `<circle cx="${cx}" cy="${cy}" r="${f1(radius * 0.5)}" fill="none" stroke="${PALETTE.grid}" stroke-width="0.8" stroke-dasharray="2 2"/>`,
    `<line x1="${f1(cx - radius - 6)}" y1="${cy}" x2="${f1(cx + radius + 8)}" y2="${cy}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<line x1="${cx}" y1="${f1(cy - radius - 6)}" x2="${cx}" y2="${f1(cy + radius + 6)}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<text x="${f1(cx + radius - 26)}" y="${f1(cy + 11)}" font-family="${FONT_MONO}" font-size="7.5" fill="${PALETTE.inkMuted}">PC1 ${ev1Pct}%</text>`,
    `<text x="${f1(cx + 4)}" y="${f1(cy - radius + 8)}" font-family="${FONT_MONO}" font-size="7.5" fill="${PALETTE.inkMuted}">PC2 ${ev2Pct}%</text>`,
  );

  // Trajectory polyline through {x, u_1, s_1, ..., u_k, s_k}
  if (pca.points.length >= 2) {
    const polyPts: string[] = [];
    for (let i = 0; i < pca.points.length; i += 1) {
      const pt = pca.points[i]!;
      const [sx, sy] = toPx(pt.x, pt.y);
      polyPts.push(`${f1(sx)},${f1(sy)}`);
    }
    parts.push(
      `<polyline points="${polyPts.join(' ')}" fill="none" stroke="${PALETTE.recurrentTeal}" stroke-width="1.6" stroke-linejoin="round" opacity="0.85"/>`,
    );
  }

  // Individual state points
  for (let i = 0; i < pca.points.length; i += 1) {
    const pt = pca.points[i]!;
    const [sx, sy] = toPx(pt.x, pt.y);
    const isSelectedExit =
      pt.kind === 'loop_exit' && pt.loopIteration === metadata.selectedLoopIteration;

    if (pt.kind === 'anchor') {
      const s = 4.2;
      parts.push(
        `<polygon points="${f1(sx)},${f1(sy - s)} ${f1(sx + s)},${f1(sy)} ${f1(sx)},${f1(sy + s)} ${f1(sx - s)},${f1(sy)}" fill="${PALETTE.codaAmber}" stroke="#FFFFFF" stroke-width="1"/>`,
        `<text x="${f1(sx + 6)}" y="${f1(sy + 3)}" font-family="${FONT_MONO}" font-size="8" font-weight="700" fill="${PALETTE.codaAmber}">x</text>`,
      );
    } else if (pt.kind === 'loop_entry') {
      parts.push(
        `<circle cx="${f1(sx)}" cy="${f1(sy)}" r="2.8" fill="${PALETTE.bgCard}" stroke="${PALETTE.preludeIndigo}" stroke-width="1.3"/>`,
      );
      if (pca.points.length <= 9 || pt.loopIteration === 1) {
        parts.push(
          `<text x="${f1(sx - 13)}" y="${f1(sy - 5)}" font-family="${FONT_MONO}" font-size="7.5" fill="${PALETTE.preludeIndigo}">u${pt.loopIteration}</text>`,
        );
      }
    } else {
      const fill = isSelectedExit ? PALETTE.specOxide : PALETTE.recurrentTeal;
      const r = isSelectedExit ? 4.2 : 3.2;
      parts.push(
        `<circle cx="${f1(sx)}" cy="${f1(sy)}" r="${f1(r)}" fill="${fill}" stroke="#FFFFFF" stroke-width="1"/>`,
        `<text x="${f1(sx + 5)}" y="${f1(sy - 4)}" font-family="${FONT_MONO}" font-size="8" font-weight="700" fill="${fill}">s${pt.loopIteration}</text>`,
      );
    }
  }

  parts.push(
    `<text x="12" y="${height - 10}" font-family="${FONT_MONO}" font-size="7.8" fill="${PALETTE.inkMuted}">Disk r=1.0 · pos ${metadata.selectedAbsTokenPos} "${escXml(truncateText(metadata.selectedTokenText, 10))}"</text>`,
  );

  // Right panel: Strictly 8D numeric trajectory table + local 120-deg Jacobian caveat
  const rightX = leftPanelW + 6;
  const rightW = Math.max(100, width - rightX - 10);
  parts.push(
    `<line x1="${leftPanelW}" y1="34" x2="${leftPanelW}" y2="${height - 10}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<text x="${rightX}" y="42" font-family="${FONT_MONO}" font-size="8.3" font-weight="700" fill="${PALETTE.ink}">Exact 8D Loop Metrics</text>`,
    `<rect x="${rightX}" y="47" width="${rightW}" height="15" rx="2" fill="${PALETTE.bgSubtle}"/>`,
    `<text x="${rightX + 3}" y="58" font-family="${FONT_MONO}" font-size="7.8" font-weight="700" fill="${PALETTE.inkSecondary}">t  ‖s‖   cos(x)  θ°    J_N</text>`,
  );

  const maxRows = Math.min(metadata.loopMetrics8d.length, 6);
  const rowStartY = 73;
  const rowStepY = 17;
  for (let i = 0; i < maxRows; i += 1) {
    const m = metadata.loopMetrics8d[i]!;
    const ry = rowStartY + i * rowStepY;
    const isSel = m.loopIteration === metadata.selectedLoopIteration;
    const isLocalContract = m.entryAnchorJacobian.isLocalContractiveRegime;
    const thetaDeg = safeNum(m.entryAnchorJacobian.angleDegrees);
    const jac = safeNum(m.entryAnchorJacobian.epsStabilizedJacobianNorm);
    const jacStr = jac >= 100 ? jac.toFixed(0) : jac.toFixed(2);

    if (isSel) {
      parts.push(
        `<rect x="${rightX}" y="${ry - 10}" width="${rightW}" height="14" rx="2" fill="${PALETTE.specBg}" stroke="#F1B5A8" stroke-width="0.6"/>`,
      );
    }
    const rowColor = isSel ? PALETTE.specOxide : PALETTE.ink;
    const rowLine =
      `t${m.loopIteration} ${safeNum(m.sNorm).toFixed(3)} ${fmtSigned(m.anchorCosine, 2)} ` +
      `${thetaDeg.toFixed(0).padStart(3, ' ')}° ${jacStr.padStart(5, ' ')}`;

    parts.push(
      `<text x="${rightX + 3}" y="${ry}" font-family="${FONT_MONO}" font-size="7.8" font-weight="${isSel ? '700' : '500'}" fill="${rowColor}">${escXml(rowLine)}</text>`,
      `<circle cx="${rightX + rightW - 6}" cy="${ry - 3}" r="2.5" fill="${isLocalContract ? PALETTE.nlerpEmerald : PALETTE.codaAmber}"/>`,
    );
  }

  // Local 120-deg anchor-map Jacobian caveat box
  const caveatY = Math.max(rowStartY + maxRows * rowStepY + 4, height - 56);
  parts.push(
    `<rect x="${rightX}" y="${caveatY}" width="${rightW}" height="44" rx="4" fill="${PALETTE.bgSurface}" stroke="${PALETTE.border}" stroke-width="0.8"/>`,
    `<text x="${rightX + 5}" y="${caveatY + 12}" font-family="${FONT_MONO}" font-size="7.5" font-weight="700" fill="${PALETTE.inkSecondary}">Anchor Jacobian Caveat:</text>`,
    `<text x="${rightX + 5}" y="${caveatY + 23}" font-family="${FONT_SANS}" font-size="7.8" fill="${PALETTE.inkSecondary}">θ &lt; 120° contracts N(s+x)</text>`,
    `<text x="${rightX + 5}" y="${caveatY + 34}" font-family="${FONT_SANS}" font-size="7.8" fill="${PALETTE.inkSecondary}">locally ONLY — NOT core F_θ.</text>`,
  );

  const title = 'Exploratory 2D PCA Latent Trajectory (8D Metrics)';
  const ariaLabel =
    `${title}. ${DORY_PCA_COMPACT_BADGE}. PC1 ${ev1Pct}%, PC2 ${ev2Pct}%, ` +
    `${metadata.loopMetrics8d.length} recurrent loops plotted.`;

  const caveats: string[] = [
    DORY_EXPLORATORY_PCA_BADGE,
    DORY_ANCHOR_MAP_CAVEAT_BADGE,
  ];
  if (metadata.counterfactualCoda !== null) {
    caveats.push(
      `${metadata.counterfactualCoda.badge} (read-only historical Deep Coda KV; KL=${safeNum(metadata.counterfactualCoda.klFromFinalLoopNats).toFixed(4)} nats).`,
    );
  }

  return Object.freeze({
    kind: 'exploratory_pca',
    title,
    badge: DORY_PCA_COMPACT_BADGE,
    ariaLabel,
    width,
    height,
    svgMarkup: wrapSvg(width, height, ariaLabel, 'exploratory_pca', parts.join('')),
    epistemicCaveats: Object.freeze(caveats),
  });
}

// ============================================================================
// CHART 3: DUAL-ROPE 4-PLANE NEOX PHASORS (`SWA theta_S=16` vs. `Full theta_F=1024`)
// ============================================================================

/**
 * Renders View 3's **4-Plane NeoX Dual-RoPE Phasor Diagram** (`D2`, `DR6`), showing pre-RoPE vs.
 * post-RoPE query & key rotations and relative phase angle `Delta phi_j = (q_pos - k_pos) * omega_j`.
 */
export function renderDualRopePhasorsSvg(
  metadata: DoryGqaRopeViewMetadata,
  dims?: DorySvgChartDimensions,
): DorySvgChartRenderResult {
  const width = clampDim(dims?.width, 392, 260, 1200);
  const height = clampDim(dims?.height, 224, 160, 900);

  const isSwa = metadata.ropeRegime === 'sliding';
  const accentColor = isSwa ? PALETTE.swaCyan : PALETTE.fullIndigo;
  const accentBg = isSwa ? PALETTE.swaBg : PALETTE.preludeBg;
  const regimeBanner = isSwa
    ? `SWA · θ_S=${metadata.ropeThetaMini} (prod 10K) · W=${metadata.slidingWindowMini ?? 8} · RoPE -> L2Norm (DR6)`
    : `FULL · θ_F=${metadata.ropeThetaMini} (prod 10M) · Causal · RoPE -> L2Norm (DR6)`;

  const planes = metadata.ropePhasors;
  const count = Math.max(1, Math.min(4, planes.length));
  const padX = 10;
  const gapX = 6;
  const cardW = (width - padX * 2 - gapX * (count - 1)) / count;
  const cardTop = 34;
  const cardH = height - cardTop - 42;
  const dialR = Math.max(20, Math.min(cardW * 0.34, cardH * 0.30));

  const parts: string[] = [];

  parts.push(
    `<rect x="8" y="8" width="${width - 16}" height="20" rx="4" fill="${accentBg}" stroke="${accentColor}" stroke-width="0.8"/>`,
    `<text x="14" y="21" font-family="${FONT_MONO}" font-size="8.4" font-weight="700" fill="${accentColor}">${escXml(regimeBanner)}</text>`,
  );

  for (let i = 0; i < count; i += 1) {
    const p = planes[i]!;
    const x0 = padX + i * (cardW + gapX);
    const cx = x0 + cardW * 0.5;
    const cy = cardTop + 26 + dialR;
    const isSel = p.planeIdx === metadata.selectedRotaryPlane;

    parts.push(
      `<rect x="${f1(x0)}" y="${cardTop}" width="${f1(cardW)}" height="${f1(cardH)}" rx="5" ` +
        `fill="${isSel ? accentBg : PALETTE.bgSurface}" stroke="${isSel ? accentColor : PALETTE.border}" stroke-width="${isSel ? '1.5' : '0.9'}"/>`,
      `<text x="${f1(cx)}" y="${cardTop + 13}" text-anchor="middle" font-family="${FONT_MONO}" font-size="8.2" font-weight="700" fill="${isSel ? accentColor : PALETTE.ink}">P${p.planeIdx} (c${p.planeIdx},c${p.planeIdx + 4})</text>`,
      `<text x="${f1(cx)}" y="${cardTop + 23}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.4" fill="${PALETTE.inkMuted}">ω=${safeNum(p.invFreq).toFixed(3)}</text>`,
      `<circle cx="${f1(cx)}" cy="${f1(cy)}" r="${f1(dialR)}" fill="${PALETTE.bgCard}" stroke="${PALETTE.axis}" stroke-width="0.9"/>`,
      `<line x1="${f1(cx - dialR)}" y1="${f1(cy)}" x2="${f1(cx + dialR)}" y2="${f1(cy)}" stroke="${PALETTE.grid}" stroke-width="0.7"/>`,
      `<line x1="${f1(cx)}" y1="${f1(cy - dialR)}" x2="${f1(cx)}" y2="${f1(cy + dialR)}" stroke="${PALETTE.grid}" stroke-width="0.7"/>`,
    );

    const drawPairVec = (
      pair: readonly [number, number],
      color: string,
      strokeW: number,
      dashed: boolean,
    ): void => {
      const vx = safeNum(pair[0]);
      const vy = safeNum(pair[1]);
      const mag = Math.hypot(vx, vy);
      if (mag < 1e-8) return;
      const nx = vx / mag;
      const ny = vy / mag;
      const ex = cx + nx * (dialR * 0.86);
      const ey = cy - ny * (dialR * 0.86);
      parts.push(
        `<line x1="${f1(cx)}" y1="${f1(cy)}" x2="${f1(ex)}" y2="${f1(ey)}" stroke="${color}" stroke-width="${strokeW}"` +
          `${dashed ? ' stroke-dasharray="2.5 2"' : ''}/>`,
        `<circle cx="${f1(ex)}" cy="${f1(ey)}" r="${dashed ? '2' : '2.7'}" fill="${color}"/>`,
      );
    };

    // Draw pre-RoPE Q (dashed), post-RoPE Q (indigo), and post-RoPE K (teal)
    drawPairVec(p.qPrePair, PALETTE.axis, 1.0, true);
    drawPairVec(p.qRotPair, PALETTE.preludeIndigo, 1.8, false);
    drawPairVec(p.kRotPair, PALETTE.recurrentTeal, 1.8, false);

    // Relative phase angle wedge Delta phi = (qPos - kPos) * invFreq
    const dPhiDeg = ((safeNum(p.deltaAngleRad) * 180) / Math.PI) % 360;
    const periodStr =
      p.periodTokens >= 100 ? `${p.periodTokens.toFixed(0)}t` : `${p.periodTokens.toFixed(1)}t`;
    const footY = cardTop + cardH - 18;
    parts.push(
      `<text x="${f1(cx)}" y="${f1(footY)}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.6" font-weight="600" fill="${PALETTE.codaAmber}">Δφ=${dPhiDeg.toFixed(0)}°</text>`,
      `<text x="${f1(cx)}" y="${f1(footY + 11)}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.inkMuted}">T=${periodStr}</text>`,
    );
  }

  // Bottom legend + dual-regime comparison
  const legY = height - 24;
  parts.push(
    `<line x1="12" y1="${legY}" x2="24" y2="${legY}" stroke="${PALETTE.preludeIndigo}" stroke-width="2"/>`,
    `<text x="27" y="${legY + 3}" font-family="${FONT_MONO}" font-size="7.8" fill="${PALETTE.ink}">Q_rot (pos ${metadata.selectedQueryAbsPos}, h${metadata.selectedQHead})</text>`,
    `<line x1="158" y1="${legY}" x2="170" y2="${legY}" stroke="${PALETTE.recurrentTeal}" stroke-width="2"/>`,
    `<text x="173" y="${legY + 3}" font-family="${FONT_MONO}" font-size="7.8" fill="${PALETTE.ink}">K_rot (pos ${metadata.selectedKeyAbsPos}, kv${metadata.selectedKvHead})</text>`,
    `<line x1="308" y1="${legY}" x2="320" y2="${legY}" stroke="${PALETTE.axis}" stroke-width="1.2" stroke-dasharray="2.5 2"/>`,
    `<text x="323" y="${legY + 3}" font-family="${FONT_MONO}" font-size="7.8" fill="${PALETTE.inkMuted}">Q_pre</text>`,
    `<text x="12" y="${height - 9}" font-family="${FONT_MONO}" font-size="7.6" fill="${PALETTE.inkSecondary}">Phys L${metadata.physicalLayerIdx} (Pair L${metadata.pairedLayerIdx}) · 100% NeoX rotate_half · sqk² Q-only · prod 10K vs 10M (1000x)</text>`,
  );

  const title = `Dual-RoPE 4-Plane NeoX Phasors — Phys L${metadata.physicalLayerIdx} (${isSwa ? 'SWA' : 'Full'})`;
  const ariaLabel =
    `${title}. Regime ${metadata.ropeRegime}, theta mini ${metadata.ropeThetaMini} ` +
    `(prod ${metadata.ropeThetaProd}), qPos ${metadata.selectedQueryAbsPos}, kPos ${metadata.selectedKeyAbsPos}.`;

  return Object.freeze({
    kind: 'dual_rope_phasors',
    title,
    badge: isSwa
      ? `SWA θ_S=16 (prod 10,000) · W=8`
      : `FULL θ_F=1024 (prod 10,000,000)`,
    ariaLabel,
    width,
    height,
    svgMarkup: wrapSvg(width, height, ariaLabel, 'dual_rope_phasors', parts.join('')),
    epistemicCaveats: Object.freeze([
      'DR6 Canonical Order: 100% NeoX rotate_half applied BEFORE per-head Q/K L2Norm (matching vLLM, llama.cpp & Paper TikZ); sqk^2 applied on Q only.',
      `D2 Scaling: miniature dual-RoPE (theta_S=16 dyadic [1, 1/2, 1/4, 1/8], theta_F=1024; 64x) reflects production (10,000 vs. 10,000,000; 1,000x; no YaRN/NTK).`,
    ]),
  });
}

// ============================================================================
// CHART 4: PRODUCTION `1 -> 256K` BF16 LOGICAL KV MEMORY SCALING CHART
// ============================================================================

/**
 * Renders View 5's **Production `1 -> 256K` BF16 Logical KV Memory Scaling Chart** (`Audit §7.2`),
 * highlighting the exact `S = 1,024` `4x` marginal slope knee and the strictly `S`-invariant
 * `(4 + 5k) / 9` ratio between `per_loop` and `last_loop`.
 */
export function renderProductionKvScalingSvg(
  input: DoryKvScalingChartInput,
  dims?: DorySvgChartDimensions,
): DorySvgChartRenderResult {
  const width = clampDim(dims?.width, 392, 260, 1200);
  const height = clampDim(dims?.height, 242, 175, 900);

  const rows: readonly DoryProductionKvScalingRow[] = input.rows;
  const activeK = Math.max(1, Math.trunc(safeNum(input.activeKLoops, 2)));
  const activeRatio = (4 + 5 * activeK) / 9;

  const padL = 48;
  const padR = 14;
  const padT = 36;
  const padB = 42;
  const plotW = Math.max(120, width - padL - padR);
  const plotH = Math.max(80, height - padT - padB);

  // Log2-X over [1 .. 262,144] (log2 in [0 .. 18]) and Log10-Y over [0.1 MiB .. 64 GiB]
  const minLog2S = 0;
  const maxLog2S = 18;
  const BYTES_PER_MIB = 1_048_576;
  const minLog10MiB = Math.log10(0.1);
  const maxLog10MiB = Math.log10(65_536);

  const xOfS = (s: number): number => {
    const l2 = Math.log2(Math.max(1, safeNum(s, 1)));
    return padL + ((l2 - minLog2S) / (maxLog2S - minLog2S)) * plotW;
  };
  const yOfBytes = (bytes: number): number => {
    const mib = Math.max(0.1, safeNum(bytes, BYTES_PER_MIB * 0.1) / BYTES_PER_MIB);
    const t = (Math.log10(mib) - minLog10MiB) / (maxLog10MiB - minLog10MiB);
    return padT + (1 - Math.max(0, Math.min(1, t))) * plotH;
  };

  const parts: string[] = [];

  // Top summary banner
  const ratioLabel = `1 -> 256K BF16 KV · Ratio (4+5k)/9 = ${activeRatio.toFixed(3)}x (S-invariant) · 4x Knee @ 1K`;
  parts.push(
    `<rect x="8" y="8" width="${width - 16}" height="20" rx="4" fill="${PALETTE.preludeBg}" stroke="#B8CBF0" stroke-width="0.8"/>`,
    `<text x="14" y="21" font-family="${FONT_MONO}" font-size="8.2" font-weight="700" fill="${PALETTE.preludeIndigo}">${escXml(ratioLabel)}</text>`,
    `<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="${PALETTE.bgSurface}" stroke="${PALETTE.border}" stroke-width="0.8"/>`,
  );

  // Horizontal Y-grid ticks (1 MiB, 10 MiB, 100 MiB, 1 GiB, 10 GiB)
  const yTicks: readonly { readonly bytes: number; readonly label: string }[] = [
    { bytes: 1 * BYTES_PER_MIB, label: '1 MiB' },
    { bytes: 10 * BYTES_PER_MIB, label: '10 MiB' },
    { bytes: 100 * BYTES_PER_MIB, label: '100 MiB' },
    { bytes: 1024 * BYTES_PER_MIB, label: '1 GiB' },
    { bytes: 10_240 * BYTES_PER_MIB, label: '10 GiB' },
  ];
  for (let i = 0; i < yTicks.length; i += 1) {
    const tk = yTicks[i]!;
    const py = yOfBytes(tk.bytes);
    parts.push(
      `<line x1="${padL}" y1="${f1(py)}" x2="${padL + plotW}" y2="${f1(py)}" stroke="${PALETTE.grid}" stroke-width="0.8"/>`,
      `<text x="${padL - 5}" y="${f1(py + 3)}" text-anchor="end" font-family="${FONT_MONO}" font-size="7.5" fill="${PALETTE.inkMuted}">${tk.label}</text>`,
    );
  }

  // Vertical X-grid ticks
  const xTicks: readonly { readonly s: number; readonly label: string }[] = [
    { s: 1, label: '1' },
    { s: 64, label: '64' },
    { s: 1024, label: '1K (W)' },
    { s: 8192, label: '8K' },
    { s: 32768, label: '32K' },
    { s: 262144, label: '256K' },
  ];
  for (let i = 0; i < xTicks.length; i += 1) {
    const tk = xTicks[i]!;
    const px = xOfS(tk.s);
    const isKnee = tk.s === 1024;
    parts.push(
      `<line x1="${f1(px)}" y1="${padT}" x2="${f1(px)}" y2="${padT + plotH}" stroke="${isKnee ? PALETTE.codaAmber : PALETTE.grid}" stroke-width="${isKnee ? '1.3' : '0.8'}"${isKnee ? ' stroke-dasharray="3.5 2"' : ''}/>`,
      `<text x="${f1(px)}" y="${padT + plotH + 12}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.5" font-weight="${isKnee ? '700' : '500'}" fill="${isKnee ? PALETTE.codaAmber : PALETTE.inkMuted}">${tk.label}</text>`,
    );
  }

  // SWA saturation shading under last_loop SWA bytes (shows SWA flattening completely after S=1,024)
  if (rows.length >= 2) {
    const swaAreaPts: string[] = [];
    for (let i = 0; i < rows.length; i += 1) {
      const r = rows[i]!;
      swaAreaPts.push(`${f1(xOfS(r.seqLen))},${f1(yOfBytes(r.lastLoopPoint.swaBytes))}`);
    }
    swaAreaPts.push(
      `${f1(xOfS(rows[rows.length - 1]!.seqLen))},${f1(padT + plotH)}`,
      `${f1(xOfS(rows[0]!.seqLen))},${f1(padT + plotH)}`,
    );
    parts.push(
      `<polygon points="${swaAreaPts.join(' ')}" fill="${PALETTE.swaCyan}" fill-opacity="0.11"/>`,
    );

    const buildCurve = (
      getter: (row: DoryProductionKvScalingRow) => number,
      color: string,
      strokeWidth: number,
      dashed: boolean,
    ): void => {
      const pts: string[] = [];
      for (let i = 0; i < rows.length; i += 1) {
        const r = rows[i]!;
        pts.push(`${f1(xOfS(r.seqLen))},${f1(yOfBytes(getter(r)))}`);
      }
      parts.push(
        `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="${strokeWidth}"` +
          `${dashed ? ' stroke-dasharray="4 2.5"' : ''} stroke-linecap="round" stroke-linejoin="round"/>`,
      );
      const lastRow = rows[rows.length - 1]!;
      const lx = xOfS(lastRow.seqLen);
      const ly = yOfBytes(getter(lastRow));
      parts.push(`<circle cx="${f1(lx)}" cy="${f1(ly)}" r="2.8" fill="${color}" stroke="#FFFFFF" stroke-width="0.8"/>`);
    };

    const isLastActive = input.activeKvCacheMode === 'last_loop' || activeK === 1;
    buildCurve((r) => r.lastLoopPoint.totalBytes, PALETTE.swaCyan, isLastActive ? 2.5 : 1.7, false);
    buildCurve(
      (r) => r.perLoopK2Point.totalBytes,
      PALETTE.recurrentTeal,
      !isLastActive && activeK === 2 ? 2.5 : 1.6,
      false,
    );
    buildCurve(
      (r) => r.perLoopK4Point.totalBytes,
      PALETTE.preludeIndigo,
      !isLastActive && activeK === 4 ? 2.6 : 1.7,
      false,
    );
    buildCurve(
      (r) => r.perLoopK8Point.totalBytes,
      PALETTE.codaAmber,
      !isLastActive && activeK >= 6 ? 2.4 : 1.3,
      true,
    );
  }

  // Knee annotation pill at S=1,024
  const kneeX = xOfS(1024);
  parts.push(
    `<rect x="${f1(kneeX + 4)}" y="${padT + 6}" width="112" height="25" rx="3" fill="${PALETTE.codaBg}" stroke="#F0C975" stroke-width="0.7"/>`,
    `<text x="${f1(kneeX + 8)}" y="${padT + 16}" font-family="${FONT_MONO}" font-size="7.4" font-weight="700" fill="${PALETTE.codaAmber}">S=1,024 SWA Knee</text>`,
    `<text x="${f1(kneeX + 8)}" y="${padT + 26}" font-family="${FONT_MONO}" font-size="7.2" fill="${PALETTE.inkSecondary}">4x slope drop · 75%→1.16%</text>`,
  );

  // Selected seqLen marker if supplied
  if (input.selectedSeqLen !== undefined && Number.isFinite(input.selectedSeqLen) && input.selectedSeqLen >= 1) {
    const selX = xOfS(input.selectedSeqLen);
    parts.push(
      `<line x1="${f1(selX)}" y1="${padT}" x2="${f1(selX)}" y2="${padT + plotH}" stroke="${PALETTE.specOxide}" stroke-width="1.1" stroke-dasharray="2 2"/>`,
    );
  }

  // Bottom legend with exact 256K figures
  const lastRow = rows.length > 0 ? rows[rows.length - 1]! : null;
  const llEnd = lastRow ? formatBytesAdaptive(lastRow.lastLoopPoint.totalBytes) : '9.11 GiB';
  const k2End = lastRow ? formatBytesAdaptive(lastRow.perLoopK2Point.totalBytes) : '14.16 GiB';
  const k4End = lastRow ? formatBytesAdaptive(lastRow.perLoopK4Point.totalBytes) : '24.28 GiB';
  const k8End = lastRow ? formatBytesAdaptive(lastRow.perLoopK8Point.totalBytes) : '44.52 GiB';

  const legY = height - 13;
  parts.push(
    `<rect x="10" y="${legY - 6}" width="8" height="3" fill="${PALETTE.swaCyan}"/>`,
    `<text x="21" y="${legY - 2}" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.ink}">last/k1 ${llEnd}</text>`,
    `<rect x="106" y="${legY - 6}" width="8" height="3" fill="${PALETTE.recurrentTeal}"/>`,
    `<text x="117" y="${legY - 2}" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.ink}">k2 ${k2End}</text>`,
    `<rect x="194" y="${legY - 6}" width="8" height="3" fill="${PALETTE.preludeIndigo}"/>`,
    `<text x="205" y="${legY - 2}" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.ink}">k4 ${k4End}</text>`,
    `<rect x="286" y="${legY - 6}" width="8" height="3" fill="${PALETTE.codaAmber}"/>`,
    `<text x="297" y="${legY - 2}" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.ink}">k8 ${k8End}</text>`,
  );

  const title = 'Production BF16 Logical KV Scaling (1 -> 256K)';
  const ariaLabel =
    `${title}. Active mode ${input.activeKvCacheMode}, k=${activeK}, ` +
    `S-invariant per_loop/last_loop ratio ${activeRatio.toFixed(3)}x, S=1,024 4x SWA knee.`;

  return Object.freeze({
    kind: 'kv_scaling_256k',
    title,
    badge: `S-INVARIANT (4+5k)/9 = ${activeRatio.toFixed(3)}x · 4x KNEE @ 1K`,
    ariaLabel,
    width,
    height,
    svgMarkup: wrapSvg(width, height, ariaLabel, 'kv_scaling_256k', parts.join('')),
    epistemicCaveats: Object.freeze([
      'Production B=1 BF16 logical retained KV (4,096 B/tok/layer, visible min(S, W) slots); excludes paged-attention fragmentation and transient workspace.',
      `Exact S-Invariance Theorem: 3:1 SWA:Full in every stage locks KV_per_loop / KV_last_loop = (4+5k)/9 = ${activeRatio.toFixed(3)}x across every S in [1, 256K].`,
      'DR17: For k >= 2, last_loop trades (4+5k)/9 memory savings for prefill-partition dependence (chunk boundaries alter historical KV provenance).',
    ]),
  });
}

// ============================================================================
// CHART 5: CONTRACT A §6.2 SELF-SPECULATIVE ACCEPTANCE & RESIDUAL PROBS
// ============================================================================

/**
 * Renders View 6's **Contract A Self-Speculative `(q_j, p_j)` Acceptance, Binary64 Residual &
 * `cap_swa=16` Rollback Chart** (`D3`, `DR19`, `DR23`), explicitly badged as reconstructed from
 * Paper §6.2.
 */
export function renderSelfSpecProbsSvg(
  metadata: DorySelfSpecViewMetadata,
  dims?: DorySvgChartDimensions,
): DorySvgChartRenderResult {
  const width = clampDim(dims?.width, 392, 260, 1200);
  const height = clampDim(dims?.height, 252, 185, 900);

  const summary = metadata.cycleSummary;
  const leftW = Math.round(width * 0.56);
  const parts: string[] = [];

  // Mandatory Paper §6.2 Contract A reconstruction banner
  parts.push(
    `<rect x="8" y="8" width="${width - 16}" height="20" rx="4" fill="${PALETTE.specBg}" stroke="#F1B5A8" stroke-width="0.9"/>`,
    `<text x="14" y="21" font-family="${FONT_MONO}" font-size="7.6" font-weight="700" fill="${PALETTE.specOxide}">${escXml(truncateText(metadata.contractBadge, 66))}</text>`,
  );

  // 104-Bank Contract A Ownership Strip (8 Input + 20 L1 + 60 L2-4 + 8 Deep + 8 Shallow)
  const bankStripY = 33;
  const bankStripW = width - 16;
  const segments: readonly { readonly n: number; readonly label: string; readonly color: string }[] = [
    { n: metadata.inputBankCount, label: 'In 8', color: PALETTE.preludeIndigo },
    { n: metadata.loop1SharedBankCount, label: 'L1 20 (Reused)', color: PALETTE.recurrentTeal },
    { n: metadata.loops2To4VerifierBankCount, label: 'L2-4 60 (Deep)', color: PALETTE.fullIndigo },
    { n: metadata.outputDeepBankCount, label: 'Deep 8', color: PALETTE.codaAmber },
    { n: metadata.outputShallowBankCount, label: 'Shal 8', color: PALETTE.specOxide },
  ];
  let curX = 8;
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i]!;
    const segW = (seg.n / metadata.totalBankCount) * bankStripW;
    parts.push(
      `<rect x="${f1(curX)}" y="${bankStripY}" width="${f1(Math.max(1, segW - 1))}" height="14" rx="2" fill="${seg.color}"/>`,
    );
    if (segW >= 30) {
      parts.push(
        `<text x="${f1(curX + segW * 0.5)}" y="${bankStripY + 10}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.2" font-weight="700" fill="#FFFFFF">${seg.label}</text>`,
      );
    }
    curX += segW;
  }

  // Left panel: per-position grouped (q_j, p_j) bars
  const barAreaX = 14;
  const barAreaY = 64;
  const barAreaW = leftW - 20;
  const barAreaH = height - barAreaY - 56;
  const bars = metadata.positionBars;
  const nBars = Math.max(1, bars.length);
  const groupW = barAreaW / nBars;

  parts.push(
    `<text x="${barAreaX}" y="58" font-family="${FONT_MONO}" font-size="8" font-weight="700" fill="${PALETTE.ink}">Draft q_j (k=1) vs. Target p_j (k=4)</text>`,
    `<line x1="${barAreaX}" y1="${barAreaY + barAreaH}" x2="${barAreaX + barAreaW}" y2="${barAreaY + barAreaH}" stroke="${PALETTE.axis}" stroke-width="1"/>`,
  );

  let maxProb = 0.25;
  for (let i = 0; i < bars.length; i += 1) {
    const b = bars[i]!;
    if (b.proposalProb !== null && b.proposalProb > maxProb) maxProb = b.proposalProb;
    if (b.targetProb !== null && b.targetProb > maxProb) maxProb = b.targetProb;
  }
  maxProb = Math.min(1.0, maxProb * 1.15);

  for (let i = 0; i < bars.length; i += 1) {
    const b = bars[i]!;
    const gx = barAreaX + i * groupW;
    const isSel = b.verifyOffset === metadata.selectedVerifyOffset;
    const bw = Math.max(6, Math.min(14, (groupW - 10) * 0.44));
    const qVal = Math.max(0, safeNum(b.proposalProb ?? 0));
    const pVal = Math.max(0, safeNum(b.targetProb ?? 0));
    const qH = (qVal / maxProb) * barAreaH;
    const pH = (pVal / maxProb) * barAreaH;

    if (isSel) {
      parts.push(
        `<rect x="${f1(gx + 1)}" y="${barAreaY - 2}" width="${f1(groupW - 2)}" height="${f1(barAreaH + 18)}" rx="3" fill="${PALETTE.bgSubtle}" stroke="${PALETTE.border}" stroke-width="0.7"/>`,
      );
    }

    const qx = gx + groupW * 0.5 - bw - 1;
    const px = gx + groupW * 0.5 + 1;
    if (b.proposalProb !== null) {
      parts.push(
        `<rect x="${f1(qx)}" y="${f1(barAreaY + barAreaH - qH)}" width="${f1(bw)}" height="${f1(Math.max(1, qH))}" rx="1.5" fill="${PALETTE.recurrentTeal}"/>`,
      );
    }
    if (b.targetProb !== null) {
      parts.push(
        `<rect x="${f1(px)}" y="${f1(barAreaY + barAreaH - pH)}" width="${f1(bw)}" height="${f1(Math.max(1, pH))}" rx="1.5" fill="${PALETTE.preludeIndigo}"/>`,
      );
    }

    // Threshold marker r * q[d] when stochastic test ran
    if (b.uniformVariate !== null && b.proposalProb !== null) {
      const rqH = ((b.uniformVariate * qVal) / maxProb) * barAreaH;
      const ty = barAreaY + barAreaH - rqH;
      parts.push(
        `<line x1="${f1(qx - 1)}" y1="${f1(ty)}" x2="${f1(px + bw + 1)}" y2="${f1(ty)}" stroke="${PALETTE.specOxide}" stroke-width="1.2" stroke-dasharray="2 1.5"/>`,
      );
    }

    const statusTag = b.isBonusSlot
      ? 'BON'
      : b.accepted === true
        ? 'ACC'
        : b.accepted === false
          ? 'REJ'
          : b.rolledBack
            ? 'RB'
            : '—';
    const statusColor = b.isBonusSlot
      ? PALETTE.codaAmber
      : b.accepted === true
        ? PALETTE.nlerpEmerald
        : b.accepted === false
          ? PALETTE.specOxide
          : PALETTE.inkMuted;

    parts.push(
      `<text x="${f1(gx + groupW * 0.5)}" y="${barAreaY + 8}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.3" font-weight="700" fill="${statusColor}">${statusTag}</text>`,
      `<text x="${f1(gx + groupW * 0.5)}" y="${barAreaY + barAreaH + 11}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.inkSecondary}">p${b.absPos}${b.reusedPreludeAndLoop1 ? '·R' : ''}</text>`,
    );
  }

  // Right panel: Binary64 Residual Correction `p_res = max(0, p - q) / Z`
  const rightX = leftW + 6;
  const rightW = Math.max(96, width - rightX - 10);
  parts.push(
    `<line x1="${leftW}" y1="52" x2="${leftW}" y2="${height - 40}" stroke="${PALETTE.grid}" stroke-width="1"/>`,
    `<text x="${rightX}" y="58" font-family="${FONT_MONO}" font-size="8" font-weight="700" fill="${PALETTE.ink}">Binary64 Residual p_res</text>`,
  );

  if (metadata.residualTopTokens.length > 0) {
    const showN = Math.min(5, metadata.residualTopTokens.length);
    const maxRes = Math.max(0.05, metadata.residualTopTokens[0]!.residualProb);
    const trackX = rightX + 44;
    const trackW = Math.max(30, rightW - 76);

    for (let i = 0; i < showN; i += 1) {
      const rt = metadata.residualTopTokens[i]!;
      const ry = 67 + i * 19;
      const bw = Math.max(1, (safeNum(rt.residualProb) / maxRes) * trackW);
      parts.push(
        `<text x="${rightX}" y="${ry + 9}" font-family="${FONT_MONO}" font-size="7.6" fill="${PALETTE.ink}">${escXml(truncateText(rt.tokenText, 7))}</text>`,
        `<rect x="${trackX}" y="${ry + 1}" width="${f1(trackW)}" height="10" rx="2" fill="${PALETTE.bgSubtle}"/>`,
        `<rect x="${trackX}" y="${ry + 1}" width="${f1(bw)}" height="10" rx="2" fill="${PALETTE.specOxide}"/>`,
        `<text x="${f1(trackX + trackW + 3)}" y="${ry + 9}" font-family="${FONT_MONO}" font-size="7.5" fill="${PALETTE.specOxide}">${f2(rt.residualProb)}</text>`,
      );
    }
    const zStr = summary.residualMassZ !== null ? summary.residualMassZ.toFixed(4) : '—';
    parts.push(
      `<text x="${rightX}" y="${height - 46}" font-family="${FONT_MONO}" font-size="7.4" fill="${PALETTE.inkMuted}">Z=${zStr} · 0 extra Shal Coda</text>`,
    );
  } else {
    parts.push(
      `<rect x="${rightX}" y="70" width="${rightW}" height="62" rx="4" fill="${PALETTE.nlerpBg}" stroke="#A3DEC2" stroke-width="0.8"/>`,
      `<text x="${rightX + 6}" y="86" font-family="${FONT_MONO}" font-size="7.8" font-weight="700" fill="${PALETTE.nlerpEmerald}">No Residual Needed</text>`,
      `<text x="${rightX + 6}" y="99" font-family="${FONT_SANS}" font-size="8" fill="${PALETTE.inkSecondary}">Accepted ${summary.acceptedDraftCount}/${summary.draftCount} drafts.</text>`,
      `<text x="${rightX + 6}" y="112" font-family="${FONT_SANS}" font-size="8" fill="${PALETTE.inkSecondary}">Shal Coda catch-up: ${metadata.shallowCodaCatchupExecuted ? '1-tok' : '0'}</text>`,
      `<text x="${rightX + 6}" y="124" font-family="${FONT_MONO}" font-size="7.2" fill="${PALETTE.inkMuted}">${escXml(metadata.divisionFreeBinary64Rule)}</text>`,
    );
  }

  // Bottom strip: Transaction-local cap_swa = 16 SWA ring slots (DR19)
  const ringY = height - 32;
  const ringSlots = metadata.swaRingSlots;
  const slotCount = Math.max(1, ringSlots.length);
  const slotTrackX = 112;
  const slotTrackW = Math.max(80, width - slotTrackX - 10);
  const slotW = slotTrackW / slotCount;

  parts.push(
    `<text x="12" y="${ringY + 9}" font-family="${FONT_MONO}" font-size="7.6" font-weight="700" fill="${PALETTE.inkSecondary}">cap_swa=${metadata.swaRingCapacity} Ring:</text>`,
  );
  for (let i = 0; i < ringSlots.length; i += 1) {
    const sl = ringSlots[i]!;
    const sx = slotTrackX + i * slotW;
    const fill =
      sl.status === 'committed_new'
        ? PALETTE.nlerpEmerald
        : sl.status === 'committed_past'
          ? PALETTE.preludeIndigo
          : sl.status === 'rolled_back'
            ? PALETTE.specOxide
            : sl.status === 'evicted_from_window'
              ? PALETTE.codaAmber
              : PALETTE.bgSubtle;
    parts.push(
      `<rect x="${f1(sx)}" y="${ringY}" width="${f1(Math.max(2, slotW - 1.5))}" height="12" rx="1.5" fill="${fill}"/>`,
    );
  }
  parts.push(
    `<text x="12" y="${height - 8}" font-family="${FONT_MONO}" font-size="7.4" fill="${PALETTE.inkMuted}">Acc ${summary.acceptedDraftCount}/${summary.draftCount} · Reused ${summary.reusedActivationPositions} · Rolled back ${metadata.rolledBackPositions} · Rule: ${escXml(metadata.divisionFreeBinary64Rule)}</text>`,
  );

  const title = 'Self-Speculative Acceptance (q_j vs. p_j) & Binary64 Residual';
  const ariaLabel =
    `${title}. ${metadata.contractBadge}. Accepted ${summary.acceptedDraftCount} of ${summary.draftCount} drafts, ` +
    `rolled back ${metadata.rolledBackPositions} positions across 104 Contract A banks.`;

  return Object.freeze({
    kind: 'self_spec_probs',
    title,
    badge: metadata.contractBadge,
    ariaLabel,
    width,
    height,
    svgMarkup: wrapSvg(width, height, ariaLabel, 'self_spec_probs', parts.join('')),
    epistemicCaveats: Object.freeze([
      metadata.contractBadge,
      `Division-free binary64 acceptance (${metadata.divisionFreeBinary64Rule}) with exact residual p_res = max(0, p - q) / Z (zero 1e-38 clamp, zero 1e-12 fallback).`,
      `Contract A allocates 8 persistent output_shallow Coda banks (+${formatBytesAdaptive(metadata.shallowCodaExtraBytes)}; 104 banks total = ${formatBytesAdaptive(metadata.totalKvPayloadBytes)}) with transaction-local cap_swa=${metadata.swaRingCapacity} rollback (DR19, DR23).`,
    ]),
  });
}

// ============================================================================
// CHART 6: STRICTLY `READ-ONLY · NO GRADIENTS` TBPTL DEPTH-1 SCHEMATIC
// ============================================================================

/**
 * Renders View 7's strictly **`READ-ONLY · NO GRADIENTS` TBPTL Depth-1 Stop-Gradient & 5-Stage
 * Training Schematic** (`Audit §1.4 & §12.1 View 7`).
 */
export function renderTbptlSchematicSvg(
  schematic: DoryTbptlSchematicReadout,
  dims?: DorySvgChartDimensions,
): DorySvgChartRenderResult {
  const width = clampDim(dims?.width, 392, 260, 1200);
  const height = clampDim(dims?.height, 246, 180, 900);

  const parts: string[] = [];

  // Mandatory READ-ONLY · NO GRADIENTS quarantine banner
  parts.push(
    `<rect x="8" y="8" width="${width - 16}" height="20" rx="4" fill="${PALETTE.readonlyBg}" stroke="${PALETTE.readonlySlate}" stroke-width="1"/>`,
    `<text x="14" y="21" font-family="${FONT_MONO}" font-size="8.4" font-weight="700" fill="${PALETTE.readonlySlate}">${escXml(schematic.badge)} · TBPTL DEPTH=${schematic.tbptlDepth} · EARLY-EXIT LOSSES=${schematic.earlyExitLosses}</text>`,
  );

  // Upper schematic: Forward flow + TBPTL depth-1 stop-gradient wall + Prelude +x gradient path
  const boxY = 58;
  const boxH = 34;
  const innerW = width - 24;
  const xPre = 12;
  const wPre = Math.round(innerW * 0.19);
  const xPrev = xPre + wPre + 14;
  const wPrev = Math.round(innerW * 0.18);
  const xWall = xPrev + wPrev + 12;
  const xUk = xWall + 14;
  const wUk = Math.round(innerW * 0.19);
  const xCoda = xUk + wUk + 14;
  const wCoda = Math.max(52, width - 12 - xCoda);

  const midY = boxY + boxH * 0.5;
  const anchorBusY = 40;
  const gradBusY = 108;

  // Persistent anchor bus +x (forward + SOLE gradient path back to Prelude!)
  const preCenterX = xPre + wPre * 0.5;
  const ukCenterX = xUk + wUk * 0.5;
  parts.push(
    `<path d="M${f1(preCenterX)},${boxY} L${f1(preCenterX)},${anchorBusY} L${f1(ukCenterX)},${anchorBusY} L${f1(ukCenterX)},${boxY}" fill="none" stroke="${PALETTE.codaAmber}" stroke-width="1.8"/>`,
    `<path d="${buildArrowHeadPath(ukCenterX, anchorBusY, ukCenterX, boxY, 5)}" fill="none" stroke="${PALETTE.codaAmber}" stroke-width="1.6"/>`,
    `<rect x="${f1((preCenterX + ukCenterX) * 0.5 - 56)}" y="${anchorBusY - 8}" width="112" height="14" rx="3" fill="${PALETTE.codaBg}" stroke="#F0C975" stroke-width="0.7"/>`,
    `<text x="${f1((preCenterX + ukCenterX) * 0.5)}" y="${anchorBusY + 2}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.3" font-weight="700" fill="${PALETTE.codaAmber}">+x Anchor (SOLE dL/dx)</text>`,
  );

  // Forward blocks
  parts.push(
    `<rect x="${xPre}" y="${boxY}" width="${wPre}" height="${boxH}" rx="4" fill="${PALETTE.preludeBg}" stroke="${PALETTE.preludeIndigo}" stroke-width="1.1"/>`,
    `<text x="${f1(preCenterX)}" y="${boxY + 14}" text-anchor="middle" font-family="${FONT_MONO}" font-size="8" font-weight="700" fill="${PALETTE.preludeIndigo}">Prelude</text>`,
    `<text x="${f1(preCenterX)}" y="${boxY + 26}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.3" fill="${PALETTE.inkSecondary}">0..15 (1x)</text>`,

    `<line x1="${xPre + wPre}" y1="${f1(midY)}" x2="${xPrev}" y2="${f1(midY)}" stroke="${PALETTE.inkSecondary}" stroke-width="1.2"/>`,

    `<rect x="${xPrev}" y="${boxY}" width="${wPrev}" height="${boxH}" rx="4" fill="${PALETTE.bgSubtle}" stroke="${PALETTE.axis}" stroke-width="1" stroke-dasharray="3 2"/>`,
    `<text x="${f1(xPrev + wPrev * 0.5)}" y="${boxY + 14}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.8" font-weight="700" fill="${PALETTE.inkSecondary}">Loops 1..k-1</text>`,
    `<text x="${f1(xPrev + wPrev * 0.5)}" y="${boxY + 26}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.2" fill="${PALETTE.inkMuted}">s_{k-1} (no grad)</text>`,

    `<line x1="${xPrev + wPrev}" y1="${f1(midY)}" x2="${xUk}" y2="${f1(midY)}" stroke="${PALETTE.inkSecondary}" stroke-width="1.2"/>`,

    // Red TBPTL depth-1 stop-gradient barrier wall
    `<line x1="${xWall}" y1="${boxY - 6}" x2="${xWall}" y2="${boxY + boxH + 10}" stroke="${PALETTE.specOxide}" stroke-width="2.2" stroke-dasharray="3 2"/>`,
    `<circle cx="${xWall}" cy="${f1(midY)}" r="7" fill="${PALETTE.specBg}" stroke="${PALETTE.specOxide}" stroke-width="1.3"/>`,
    `<text x="${xWall}" y="${f1(midY + 2.8)}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.5" font-weight="800" fill="${PALETTE.specOxide}">sg</text>`,

    `<rect x="${xUk}" y="${boxY}" width="${wUk}" height="${boxH}" rx="4" fill="${PALETTE.recurrentBg}" stroke="${PALETTE.recurrentTeal}" stroke-width="1.2"/>`,
    `<text x="${f1(ukCenterX)}" y="${boxY + 14}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.8" font-weight="700" fill="${PALETTE.recurrentTeal}">Final Loop k</text>`,
    `<text x="${f1(ukCenterX)}" y="${boxY + 26}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.1" fill="${PALETTE.inkSecondary}">u_k=N(sg+x)</text>`,

    `<line x1="${xUk + wUk}" y1="${f1(midY)}" x2="${xCoda}" y2="${f1(midY)}" stroke="${PALETTE.inkSecondary}" stroke-width="1.2"/>`,

    `<rect x="${xCoda}" y="${boxY}" width="${wCoda}" height="${boxH}" rx="4" fill="${PALETTE.codaBg}" stroke="${PALETTE.codaAmber}" stroke-width="1.1"/>`,
    `<text x="${f1(xCoda + wCoda * 0.5)}" y="${boxY + 14}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.8" font-weight="700" fill="${PALETTE.codaAmber}">Coda+Loss</text>`,
    `<text x="${f1(xCoda + wCoda * 0.5)}" y="${boxY + 26}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.1" fill="${PALETTE.inkSecondary}">56..71 · L_LM</text>`,
  );

  // Reverse gradient dashed rail blocked at sg wall
  parts.push(
    `<path d="M${f1(xCoda + wCoda * 0.5)},${boxY + boxH} L${f1(xCoda + wCoda * 0.5)},${gradBusY} L${f1(xWall + 4)},${gradBusY}" fill="none" stroke="${PALETTE.specOxide}" stroke-width="1.3" stroke-dasharray="3.5 2"/>`,
    `<path d="${buildArrowHeadPath(xUk + wUk, gradBusY, xWall + 4, gradBusY, 5)}" fill="none" stroke="${PALETTE.specOxide}" stroke-width="1.3"/>`,
    `<text x="${f1((xWall + xCoda + wCoda * 0.5) * 0.5)}" y="${gradBusY + 11}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.4" fill="${PALETTE.specOxide}">Active depth-1 grad stops at sg[s_{k-1}]</text>`,
  );

  // Spec chips row
  const chipY = 128;
  parts.push(
    `<rect x="12" y="${chipY}" width="${width - 24}" height="17" rx="3" fill="${PALETTE.bgSurface}" stroke="${PALETTE.border}" stroke-width="0.7"/>`,
    `<text x="18" y="${chipY + 11}" font-family="${FONT_MONO}" font-size="7.6" fill="${PALETTE.inkSecondary}">ELR rank override: ${schematic.elrRankOverride} · GRPO RLVR curriculum: ${schematic.rlvrDifficultyRatio} (α=${schematic.rlvrAlpha}) · 0 fake quant toggles</text>`,
  );

  // Lower strip: 5-Stage Training Timeline
  const stageTop = 154;
  const stageCount = Math.max(1, schematic.stages.length);
  const stageGap = 5;
  const stageW = (width - 24 - stageGap * (stageCount - 1)) / stageCount;
  const stageH = height - stageTop - 16;

  for (let i = 0; i < schematic.stages.length; i += 1) {
    const st = schematic.stages[i]!;
    const sx = 12 + i * (stageW + stageGap);
    const isPretrain = st.stageNumber <= 3;
    const stroke = isPretrain ? PALETTE.preludeIndigo : PALETTE.recurrentTeal;
    const fill = isPretrain ? PALETTE.preludeBg : PALETTE.recurrentBg;

    parts.push(
      `<rect x="${f1(sx)}" y="${stageTop}" width="${f1(stageW)}" height="${f1(stageH)}" rx="4" fill="${fill}" stroke="${stroke}" stroke-width="0.8"/>`,
      `<text x="${f1(sx + stageW * 0.5)}" y="${stageTop + 13}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.6" font-weight="700" fill="${stroke}">S${st.stageNumber}</text>`,
      `<text x="${f1(sx + stageW * 0.5)}" y="${stageTop + 26}" text-anchor="middle" font-family="${FONT_MONO}" font-size="7.2" font-weight="700" fill="${PALETTE.ink}">${escXml(truncateText(st.tokenBudgetLabel, 11))}</text>`,
      `<text x="${f1(sx + stageW * 0.5)}" y="${stageTop + 39}" text-anchor="middle" font-family="${FONT_SANS}" font-size="7.2" fill="${PALETTE.inkSecondary}">${escXml(truncateText(st.precisionAndDepthLabel, 13))}</text>`,
      `<text x="${f1(sx + stageW * 0.5)}" y="${stageTop + 52}" text-anchor="middle" font-family="${FONT_MONO}" font-size="6.9" fill="${PALETTE.inkMuted}">${escXml(truncateText(st.maxContextLabel, 13))}</text>`,
    );
  }

  parts.push(
    `<text x="12" y="${height - 5}" font-family="${FONT_MONO}" font-size="7.2" fill="${PALETTE.readonlySlate}">Strictly conceptual metadata from arXiv:2611.03998 — zero training loops or optimizer state in browser.</text>`,
  );

  const title = 'TBPTL Depth-1 Stop-Gradient & 5-Stage Training Schematic';
  const ariaLabel =
    `${title}. ${schematic.badge}. TBPTL depth ${schematic.tbptlDepth}, ` +
    `${schematic.earlyExitLosses} early-exit losses, ${schematic.stages.length} training stages.`;

  return Object.freeze({
    kind: 'tbptl_schematic',
    title,
    badge: DORY_TBPTL_READONLY_BADGE,
    ariaLabel,
    width,
    height,
    svgMarkup: wrapSvg(width, height, ariaLabel, 'tbptl_schematic', parts.join('')),
    epistemicCaveats: Object.freeze([
      schematic.badge,
      schematic.preludeGradientPath,
      schematic.epistemicCaveat,
    ]),
  });
}

// ============================================================================
// SCENE DISPATCHER & EDITORIAL DOM PANEL
// ============================================================================

/**
 * Builds the view-appropriate analytical `<svg>` chart payloads for any of the 7 Dory views
 * (`Audit §12.1`):
 *   - View 2 (`hypersphere`): `spanning_subspace` + `exploratory_pca`
 *   - View 3 (`gqa_rope`):    `dual_rope_phasors`
 *   - View 5 (`kv_compare`):  `kv_scaling_256k`
 *   - View 6 (`self_spec`):   `self_spec_probs`
 *   - View 7 (`bookends`):    `tbptl_schematic`
 *   - Views 1 (`macro`) & 4 (`swiglu`): `[]`
 */
export function buildChartsForDoryScene(
  scene: DorySpatialScene,
  readouts: DoryArchReadoutsBundle,
): readonly DorySvgChartRenderResult[] {
  const out: DorySvgChartRenderResult[] = [];

  switch (scene.viewMode) {
    case 'hypersphere': {
      const meta = scene.hypersphereMetadata;
      if (meta) {
        out.push(renderSpanningSubspaceSvg(meta.spanningSubspace));
        out.push(renderExploratoryPcaSvg(meta));
      }
      break;
    }
    case 'gqa_rope': {
      const meta = scene.gqaRopeMetadata;
      if (meta) {
        out.push(renderDualRopePhasorsSvg(meta));
      }
      break;
    }
    case 'kv_compare': {
      const meta = scene.kvCompareMetadata;
      out.push(
        renderProductionKvScalingSvg({
          rows: readouts.productionKvScaling,
          activeKLoops: meta?.kLoops ?? scene.kLoops,
          activeKvCacheMode: scene.kvCacheMode,
        }),
      );
      break;
    }
    case 'self_spec': {
      const meta = scene.selfSpecMetadata;
      if (meta) {
        out.push(renderSelfSpecProbsSvg(meta));
      }
      break;
    }
    case 'bookends': {
      out.push(renderTbptlSchematicSvg(readouts.tbptlSchematic));
      break;
    }
    case 'macro':
    case 'swiglu':
      break;
  }

  return Object.freeze(out);
}

/**
 * Mounts the responsive editorial 2D `<svg>` analytical charts panel into `host`.
 */
export function createDoryChartsPanel(host: HTMLElement): DoryAnalyticalChartsPanelHandle {
  const root = document.createElement('section');
  root.className = 'dory-charts-panel';
  root.dataset.testid = 'dory-charts-panel';
  root.setAttribute('aria-label', 'Dory Analytical SVG Insets');
  root.hidden = true;
  host.appendChild(root);

  let renderedKinds: readonly DorySvgChartKind[] = Object.freeze([]);
  let lastRenderSignature = '';
  let disposed = false;

  const clear = (): void => {
    root.innerHTML = '';
    root.hidden = true;
    renderedKinds = Object.freeze([]);
    lastRenderSignature = '';
  };

  return {
    root,
    renderForScene(
      scene: DorySpatialScene,
      readouts: DoryArchReadoutsBundle,
    ): readonly DorySvgChartKind[] {
      if (disposed) return renderedKinds;
      const charts = buildChartsForDoryScene(scene, readouts);
      if (charts.length === 0) {
        clear();
        return renderedKinds;
      }

      const signature = charts
        .map((c) => `${c.kind}|${c.title}|${c.badge ?? ''}|${c.svgMarkup.length}|${c.svgMarkup}`)
        .join('||');
      if (signature === lastRenderSignature) {
        root.hidden = false;
        return renderedKinds;
      }

      lastRenderSignature = signature;
      root.innerHTML = '';
      root.hidden = false;

      const kinds: DorySvgChartKind[] = [];
      for (let i = 0; i < charts.length; i += 1) {
        const chart = charts[i]!;
        kinds.push(chart.kind);

        const card = document.createElement('figure');
        card.className = `dory-chart-card dory-chart-card--${chart.kind}`;
        card.dataset.chartKind = chart.kind;
        card.dataset.testid = `dory-chart-${chart.kind}`;

        const header = document.createElement('div');
        header.className = 'dory-chart-header';

        const titleEl = document.createElement('h3');
        titleEl.className = 'dory-chart-title';
        titleEl.textContent = chart.title;
        header.appendChild(titleEl);

        if (chart.badge) {
          const badgeEl = document.createElement('span');
          badgeEl.className = `dory-chart-badge dory-chart-badge--${chart.kind}`;
          badgeEl.textContent = chart.badge;
          header.appendChild(badgeEl);
        }

        const svgWrap = document.createElement('div');
        svgWrap.className = 'dory-chart-svg-wrap';
        svgWrap.innerHTML = chart.svgMarkup;

        card.append(header, svgWrap);

        if (chart.epistemicCaveats.length > 0) {
          const caption = document.createElement('figcaption');
          caption.className = 'dory-chart-caveats';
          for (let j = 0; j < chart.epistemicCaveats.length; j += 1) {
            const p = document.createElement('p');
            p.className = 'dory-chart-caveat';
            p.textContent = chart.epistemicCaveats[j]!;
            caption.appendChild(p);
          }
          card.appendChild(caption);
        }

        root.appendChild(card);
      }

      renderedKinds = Object.freeze(kinds);
      return renderedKinds;
    },
    getRenderedChartKinds(): readonly DorySvgChartKind[] {
      return renderedKinds;
    },
    clear,
    dispose(): void {
      if (disposed) return;
      disposed = true;
      clear();
      if (root.parentNode === host) {
        host.removeChild(root);
      }
    },
  };
}

// Compile-time verification against frozen Step-0 contract (`src/dory/app/types.ts`)
const _CHARTS_CONTRACT_CHECK: DoryChartsModuleContract = {
  renderSpanningSubspaceSvg,
  renderExploratoryPcaSvg,
  renderDualRopePhasorsSvg,
  renderProductionKvScalingSvg,
  renderSelfSpecProbsSvg,
  renderTbptlSchematicSvg,
  buildChartsForDoryScene,
  createDoryChartsPanel,
};
void _CHARTS_CONTRACT_CHECK;
