/**
 * Nemotron-Labs-TwoTower Topology A Browser Entry Stub (`src/twotower/app/main.ts`)
 *
 * Clean, zero-dependency MPA entry root isolated from Nano v1.0.0 (`index.html` / `src/main.ts`).
 * Displays pinned provenance and miniature simulator specification until Wave 3 wires the full
 * interactive diagnostic session UI.
 */

import { TWOTOWER_MINI_CONFIG, TWOTOWER_SPEC } from '../spec';

interface MetricItem {
  readonly label: string;
  readonly value: string;
}

function createMetricList(items: readonly MetricItem[]): HTMLDListElement {
  const dl = document.createElement('dl');
  dl.className = 'tt-grid';
  for (const item of items) {
    const wrap = document.createElement('div');
    wrap.className = 'tt-metric';
    const dt = document.createElement('dt');
    dt.textContent = item.label;
    const dd = document.createElement('dd');
    dd.textContent = item.value;
    wrap.append(dt, dd);
    dl.append(wrap);
  }
  return dl;
}

export function mountTwoTowerShell(root: HTMLElement): void {
  const { provenance, layerAlignment, derivedInventoryCalculations, productionDefaults } =
    TWOTOWER_SPEC;
  const shortRevision = provenance.hfRevision.slice(0, 8);
  const totalBillions = (derivedInventoryCalculations.grandTotalParams / 1e9).toFixed(2);

  root.textContent = '';

  const main = document.createElement('main');
  main.className = 'tt-shell';
  main.setAttribute('aria-label', 'Nemotron-Labs-TwoTower diagnostic shell');

  const header = document.createElement('header');
  header.className = 'tt-header';

  const titleWrap = document.createElement('div');
  const h1 = document.createElement('h1');
  h1.className = 'tt-title';
  h1.textContent = 'Nemotron-Labs-TwoTower 30B-A3B · Diagnostic Shell';
  const badge = document.createElement('span');
  badge.className = 'tt-pill';
  badge.textContent = `Topology A Isolated Entry · HF ${shortRevision} · arXiv:${provenance.paperArxivId}`;
  titleWrap.append(h1, badge);

  const backLink = document.createElement('a');
  backLink.className = 'tt-back-link';
  backLink.href = '../';
  backLink.textContent = '← Return to Nemotron 3 Nano 30B-A3B';

  header.append(titleWrap, backLink);

  const prodCard = document.createElement('section');
  prodCard.className = 'tt-card';
  const prodHeading = document.createElement('h2');
  prodHeading.textContent = 'Pinned Production Provenance & Dual-Tower Topology';
  prodCard.append(
    prodHeading,
    createMetricList([
      { label: 'Hugging Face Repo', value: provenance.hfRepo },
      { label: 'Pinned Revision', value: `${shortRevision} (${provenance.hfRevision})` },
      {
        label: 'Dual-Tower Backbone',
        value: `2 × ${layerAlignment.totalLayers}L (${layerAlignment.mambaTransferBayCount}M · ${layerAlignment.statelessMoeBayCount}E · ${layerAlignment.gqaTransferBayCount}*)`,
      },
      {
        label: 'Aligned Transfer Bays',
        value: `${layerAlignment.alignedTransferBayCount} stateful (${layerAlignment.mambaTransferBayCount} Mamba + ${layerAlignment.gqaTransferBayCount} GQA)`,
      },
      {
        label: 'Exact Checkpoint Params',
        value: `${derivedInventoryCalculations.grandTotalParams.toLocaleString('en-US')} (${totalBillions}B)`,
      },
      {
        label: 'Execution Modes',
        value: 'AR · Mock-AR · Mask-Diffusion',
      },
    ]),
  );

  const miniCard = document.createElement('section');
  miniCard.className = 'tt-card';
  const miniHeading = document.createElement('h2');
  miniHeading.textContent = 'Educational Miniature Simulator Configuration';
  miniCard.append(
    miniHeading,
    createMetricList([
      {
        label: 'Miniature Dimensions',
        value: `d=${TWOTOWER_MINI_CONFIG.d}, V=${TWOTOWER_MINI_CONFIG.vocabSize}, kvCapacity=${TWOTOWER_MINI_CONFIG.kvCapacity}`,
      },
      {
        label: 'Special Tokens',
        value: `MASK=${TWOTOWER_MINI_CONFIG.maskTokenId}, EOS=${TWOTOWER_MINI_CONFIG.eosTokenId}`,
      },
      {
        label: 'Prefix & Block Window',
        value: `P=${TWOTOWER_MINI_CONFIG.defaultPromptLen}, B∈{${TWOTOWER_MINI_CONFIG.supportedBlockSizes.join(', ')}}, K=${TWOTOWER_MINI_CONFIG.defaultStepsPerBlock} (independent)`,
      },
      {
        label: 'Confidence Threshold γ',
        value: `γ=${TWOTOWER_MINI_CONFIG.provisionalDemoThreshold.toFixed(2)} mini (paper ${TWOTOWER_MINI_CONFIG.presetThresholds.paper.toFixed(2)} / code ${TWOTOWER_MINI_CONFIG.presetThresholds.referenceCode.toFixed(2)})`,
      },
    ]),
  );

  const note = document.createElement('p');
  note.className = 'tt-note';
  note.textContent = `${productionDefaults.epistemicNote} Interactive dual-tower diagnostic controls and live session trajectory views will mount in this Topology A shell in Stage TT2 Wave 3.`;
  miniCard.append(note);

  main.append(header, prodCard, miniCard);
  root.append(main);
}

if (typeof document !== 'undefined') {
  const existingRoot = document.getElementById('twotower-app');
  if (existingRoot) {
    mountTwoTowerShell(existingRoot);
  } else if (document.body) {
    const fallbackRoot = document.createElement('div');
    fallbackRoot.id = 'twotower-app';
    document.body.append(fallbackRoot);
    mountTwoTowerShell(fallbackRoot);
  }
}
