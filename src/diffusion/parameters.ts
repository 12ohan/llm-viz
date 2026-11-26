/**
 * Deterministic Synthetic Parameter Generator & Canonical Weight Collector
 * (`src/diffusion/parameters.ts`)
 *
 * Generates all 443 Float32 tensors (363 backbone = 229,064 floats + 80 LoRA =
 * 42,240 floats = 271,304 total floats) for `DIFFUSION_MINI_CONFIG` (`L=40,
 * d=8, nQ=32, nKv=8, dHead=8, dFf=24, V=36, rLora=4, alpha=16`) using isolated
 * per-tensor `mulberry32` + Box-Muller Gaussian streams seeded via FNV-1a.
 *
 * Clean-room deterministic synthetic weights only — pure headless TypeScript,
 * zero copied checkpoints, isolated from sibling modules.
 */

import {
  DIFFUSION_BACKBONE_SEED,
  DIFFUSION_EMBED_TENSOR_NAME,
  DIFFUSION_FINAL_NORM_TENSOR_NAME,
  DIFFUSION_HEAD_TENSOR_NAME,
  DIFFUSION_LORA_SEED,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_VOCAB,
  layerAttnWeightName,
  layerLoraWeightName,
  layerMlpWeightName,
  layerNormWeightName,
  type CreateDiffusionModelOptions,
  type DiffusionCanonicalWeightTensor,
  type DiffusionLayerWeights,
  type DiffusionModelParameters,
} from './types';

// ============================================================================
// ISOLATED LOCAL PRNG (FNV-1a + mulberry32 + Box-Muller, deterministic)
// ============================================================================

const FNV1A_32_OFFSET = 0x811c9dc5;
const FNV1A_32_PRIME = 0x01000193;

const FNV1A_64_OFFSET = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x100000001b3n;
const MASK_64 = 0xffffffffffffffffn;

function fnv1a32Seed(baseSeed: number, label: string): number {
  let h = (FNV1A_32_OFFSET ^ (baseSeed >>> 0)) >>> 0;
  h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i) & 0xff;
    h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createGaussianGenerator(seed: number): () => number {
  const nextUniform = mulberry32(seed);
  let spare: number | null = null;

  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u1 = nextUniform();
    while (u1 <= 1e-12) {
      u1 = nextUniform();
    }
    const u2 = nextUniform();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const angle = 2.0 * Math.PI * u2;
    spare = mag * Math.sin(angle);
    return mag * Math.cos(angle);
  };
}

function sampleGaussianTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
  std: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.fround(gauss() * std);
  }
  return out;
}

function sampleRmsNormTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const v = 1.0 + 0.05 * gauss();
    out[i] = Math.fround(v < 0.5 ? 0.5 : v);
  }
  return out;
}

// ============================================================================
// CANONICAL WEIGHT COLLECTORS & FINGERPRINTING
// ============================================================================

/**
 * Collect the 363 backbone weight tensors (229,064 floats) in locked canonical order:
 *   1. `encoder.embed_tokens.weight` (`[36, 8]`)
 *   2. Layers `0..39`: `input_layernorm`, `q_proj`, `k_proj`, `v_proj`, `o_proj`,
 *      `post_attention_layernorm`, `gate_proj`, `up_proj`, `down_proj` (360 tensors)
 *   3. `encoder.norm.weight` (`[8]`)
 *   4. `diffusion_head.weight` (`[36, 8]`)
 */
export function collectDiffusionBackboneWeights(
  model: DiffusionModelParameters,
): DiffusionCanonicalWeightTensor[] {
  const { config } = model;
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;
  const vSize = config.vocabSize;

  const tensors: DiffusionCanonicalWeightTensor[] = [
    {
      name: DIFFUSION_EMBED_TENSOR_NAME,
      role: 'embed_tokens',
      layerIndex: null,
      isLora: false,
      shape: Object.freeze([vSize, d]),
      data: model.embedTokens,
    },
  ];

  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    tensors.push(
      {
        name: layerNormWeightName(l, 'input'),
        role: 'input_layernorm',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([d]),
        data: layer.inputLayerNorm,
      },
      {
        name: layerAttnWeightName(l, 'q_proj'),
        role: 'q_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([qDim, d]),
        data: layer.selfAttn.qProj,
      },
      {
        name: layerAttnWeightName(l, 'k_proj'),
        role: 'k_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([kvDim, d]),
        data: layer.selfAttn.kProj,
      },
      {
        name: layerAttnWeightName(l, 'v_proj'),
        role: 'v_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([kvDim, d]),
        data: layer.selfAttn.vProj,
      },
      {
        name: layerAttnWeightName(l, 'o_proj'),
        role: 'o_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([d, qDim]),
        data: layer.selfAttn.oProj,
      },
      {
        name: layerNormWeightName(l, 'post_attention'),
        role: 'post_attention_layernorm',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([d]),
        data: layer.postAttentionLayerNorm,
      },
      {
        name: layerMlpWeightName(l, 'gate_proj'),
        role: 'gate_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([dFf, d]),
        data: layer.mlp.gateProj,
      },
      {
        name: layerMlpWeightName(l, 'up_proj'),
        role: 'up_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([dFf, d]),
        data: layer.mlp.upProj,
      },
      {
        name: layerMlpWeightName(l, 'down_proj'),
        role: 'down_proj',
        layerIndex: l,
        isLora: false,
        shape: Object.freeze([d, dFf]),
        data: layer.mlp.downProj,
      },
    );
  }

  tensors.push(
    {
      name: DIFFUSION_FINAL_NORM_TENSOR_NAME,
      role: 'final_norm',
      layerIndex: null,
      isLora: false,
      shape: Object.freeze([d]),
      data: model.finalNorm,
    },
    {
      name: DIFFUSION_HEAD_TENSOR_NAME,
      role: 'diffusion_head',
      layerIndex: null,
      isLora: false,
      shape: Object.freeze([vSize, d]),
      data: model.diffusionHead,
    },
  );

  return tensors;
}

/**
 * Collect the 80 draft-only `o_proj` LoRA tensors (42,240 floats) in canonical layer order:
 * for `l = 0..39`: `lora_A` (`[4, 256]`), `lora_B` (`[8, 4]`).
 */
export function collectDiffusionLoraWeights(
  model: DiffusionModelParameters,
): DiffusionCanonicalWeightTensor[] {
  const { config } = model;
  const d = config.hiddenSize;
  const qDim = config.qProjOutDim;
  const r = config.loraRank;

  const tensors: DiffusionCanonicalWeightTensor[] = [];
  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    tensors.push(
      {
        name: layerLoraWeightName(l, 'lora_A'),
        role: 'lora_A',
        layerIndex: l,
        isLora: true,
        shape: Object.freeze([r, qDim]),
        data: layer.lora.loraA,
      },
      {
        name: layerLoraWeightName(l, 'lora_B'),
        role: 'lora_B',
        layerIndex: l,
        isLora: true,
        shape: Object.freeze([d, r]),
        data: layer.lora.loraB,
      },
    );
  }
  return tensors;
}

/**
 * Collect all 443 canonical tensors (363 backbone followed by 80 LoRA; 271,304 floats total).
 */
export function collectDiffusionCanonicalWeights(
  model: DiffusionModelParameters,
): DiffusionCanonicalWeightTensor[] {
  return [
    ...collectDiffusionBackboneWeights(model),
    ...collectDiffusionLoraWeights(model),
  ];
}

/**
 * Compute a deterministic 64-bit FNV-1a hex fingerprint over tensor names, shapes,
 * and exact little-endian IEEE-754 Float32 bit patterns.
 */
export function computeDiffusionWeightsFingerprint(
  tensors: readonly DiffusionCanonicalWeightTensor[],
): string {
  let hash = FNV1A_64_OFFSET;

  const mixByte = (b: number): void => {
    hash ^= BigInt(b & 0xff);
    hash = (hash * FNV1A_64_PRIME) & MASK_64;
  };

  const mixUint32 = (u: number): void => {
    mixByte(u & 0xff);
    mixByte((u >>> 8) & 0xff);
    mixByte((u >>> 16) & 0xff);
    mixByte((u >>> 24) & 0xff);
  };

  for (let tIdx = 0; tIdx < tensors.length; tIdx++) {
    const tensor = tensors[tIdx];

    for (let i = 0; i < tensor.name.length; i++) {
      mixByte(tensor.name.charCodeAt(i));
    }
    mixByte(0x00);

    mixUint32(tensor.shape.length);
    for (let s = 0; s < tensor.shape.length; s++) {
      mixUint32(tensor.shape[s]);
    }

    const view = new DataView(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength,
    );
    for (let i = 0; i < tensor.data.length; i++) {
      mixUint32(view.getUint32(i * 4, true));
    }
  }

  return hash.toString(16).padStart(16, '0');
}

// ============================================================================
// MODEL FACTORY & DEFAULT SINGLETON
// ============================================================================

/**
 * Create a deterministic synthetic miniature `DiffusionModelParameters` bundle.
 */
export function createDiffusionModelParameters(
  options: CreateDiffusionModelOptions = {},
): DiffusionModelParameters {
  const config = options.config ?? DIFFUSION_MINI_CONFIG;
  const vocab = options.vocab ?? DIFFUSION_MINI_VOCAB;
  const backboneSeed = (options.backboneSeed ?? config.backboneSeed ?? DIFFUSION_BACKBONE_SEED) >>> 0;
  const loraSeed = (options.loraSeed ?? config.loraSeed ?? DIFFUSION_LORA_SEED) >>> 0;
  const modelId = options.modelId ?? 'nemotron-labs-diffusion-14b-mini';

  const L = config.numLayers;
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;
  const vSize = config.vocabSize;
  const r = config.loraRank;

  const invSqrtD = 1.0 / Math.sqrt(d);
  const residualScale = 1.0 / Math.sqrt(4 * L);
  const oProjStd = (1.0 / Math.sqrt(qDim)) * residualScale;
  const downProjStd = (1.0 / Math.sqrt(dFf)) * residualScale;
  const loraAStd = (1.0 / Math.sqrt(qDim)) * 0.25;
  const loraBStd = 0.05;

  const embedTokens = sampleGaussianTensor(
    backboneSeed,
    DIFFUSION_EMBED_TENSOR_NAME,
    vSize * d,
    invSqrtD,
  );
  const finalNorm = sampleRmsNormTensor(
    backboneSeed,
    DIFFUSION_FINAL_NORM_TENSOR_NAME,
    d,
  );
  const diffusionHead = sampleGaussianTensor(
    backboneSeed,
    DIFFUSION_HEAD_TENSOR_NAME,
    vSize * d,
    invSqrtD,
  );

  const layers: DiffusionLayerWeights[] = new Array<DiffusionLayerWeights>(L);
  for (let l = 0; l < L; l++) {
    const inputLayerNorm = sampleRmsNormTensor(
      backboneSeed,
      layerNormWeightName(l, 'input'),
      d,
    );
    const qProj = sampleGaussianTensor(
      backboneSeed,
      layerAttnWeightName(l, 'q_proj'),
      qDim * d,
      invSqrtD,
    );
    const kProj = sampleGaussianTensor(
      backboneSeed,
      layerAttnWeightName(l, 'k_proj'),
      kvDim * d,
      invSqrtD,
    );
    const vProj = sampleGaussianTensor(
      backboneSeed,
      layerAttnWeightName(l, 'v_proj'),
      kvDim * d,
      invSqrtD,
    );
    const oProj = sampleGaussianTensor(
      backboneSeed,
      layerAttnWeightName(l, 'o_proj'),
      d * qDim,
      oProjStd,
    );
    const postAttentionLayerNorm = sampleRmsNormTensor(
      backboneSeed,
      layerNormWeightName(l, 'post_attention'),
      d,
    );
    const gateProj = sampleGaussianTensor(
      backboneSeed,
      layerMlpWeightName(l, 'gate_proj'),
      dFf * d,
      invSqrtD,
    );
    const upProj = sampleGaussianTensor(
      backboneSeed,
      layerMlpWeightName(l, 'up_proj'),
      dFf * d,
      invSqrtD,
    );
    const downProj = sampleGaussianTensor(
      backboneSeed,
      layerMlpWeightName(l, 'down_proj'),
      d * dFf,
      downProjStd,
    );

    const loraA = sampleGaussianTensor(
      loraSeed,
      layerLoraWeightName(l, 'lora_A'),
      r * qDim,
      loraAStd,
    );
    const loraB = sampleGaussianTensor(
      loraSeed,
      layerLoraWeightName(l, 'lora_B'),
      d * r,
      loraBStd,
    );

    layers[l] = Object.freeze({
      layerIndex: l,
      inputLayerNorm,
      selfAttn: Object.freeze({ qProj, kProj, vProj, oProj }),
      postAttentionLayerNorm,
      mlp: Object.freeze({ gateProj, upProj, downProj }),
      lora: Object.freeze({ loraA, loraB }),
    });
  }

  const draftModel: DiffusionModelParameters = {
    modelId,
    config,
    vocab: Object.freeze([...vocab]),
    backboneSeed,
    loraSeed,
    backboneFingerprint: '',
    loraFingerprint: '',
    combinedFingerprint: '',
    embedTokens,
    layers: Object.freeze(layers),
    finalNorm,
    diffusionHead,
  };

  const backboneTensors = collectDiffusionBackboneWeights(draftModel);
  const loraTensors = collectDiffusionLoraWeights(draftModel);
  const combinedTensors = [...backboneTensors, ...loraTensors];

  const backboneFingerprint = computeDiffusionWeightsFingerprint(backboneTensors);
  const loraFingerprint = computeDiffusionWeightsFingerprint(loraTensors);
  const combinedFingerprint = computeDiffusionWeightsFingerprint(combinedTensors);

  return Object.freeze({
    ...draftModel,
    backboneFingerprint,
    loraFingerprint,
    combinedFingerprint,
  });
}

let cachedDefaultModel: DiffusionModelParameters | null = null;

/**
 * Return the memoized immutable default 40-layer miniature `DiffusionModelParameters`.
 */
export function createDefaultDiffusionModel(): DiffusionModelParameters {
  if (cachedDefaultModel === null) {
    cachedDefaultModel = createDiffusionModelParameters();
  }
  return cachedDefaultModel;
}
