/**
 * Deterministic Synthetic Parameter Generator & Canonical Weight Collector
 * (`src/uni/parameters.ts`)
 *
 * Generates all 618 core Elastic-MoT backbone + U/G IO Float32 tensors
 * (`139,024` floats = `556,096 B`) for `UNI_MINI_CONFIG`
 * (`L=34`: `L0..17` joint + `L18..33` split-tower, `d=8`, `d_ff=28`, `8Q/2KV`,
 * `headDim=8`, `V_txt=64`, `V_vq=256`, `V_gen_emb=264`) using isolated
 * per-tensor FNV-1a(seed, tensorName) + mulberry32 + Box-Muller Gaussian streams.
 *
 * Clean-room deterministic synthetic weights only — pure headless TypeScript,
 * zero ambient RNG calls, zero copied checkpoints, isolated from sibling apps.
 */

import {
  UNI_BACKBONE_SEED,
  UNI_DIFFUSION_HEAD_TENSOR_NAME,
  UNI_EMBED_TENSOR_NAME,
  UNI_FINAL_NORM_GEN_TENSOR_NAME,
  UNI_FINAL_NORM_TENSOR_NAME,
  UNI_GEN_EMBEDDING_TENSOR_NAME,
  UNI_GEN_PREDICTOR_TENSOR_NAME,
  UNI_MINI_CONFIG,
  UNI_MINI_CORE_FLOAT_COUNT,
  UNI_MINI_CORE_TENSOR_COUNT,
  UNI_MINI_VOCAB,
  uniLayerAttnWeightName,
  uniLayerMlpWeightName,
  uniLayerNormWeightName,
  type CreateUniCoreModelOptions,
  type UniBranchId,
  type UniBranchLayerWeights,
  type UniCanonicalWeightTensor,
  type UniCoreModelParameters,
  type UniElasticLayerWeights,
} from './types';

// ============================================================================
// ISOLATED PER-TENSOR PRNG (FNV-1a + mulberry32 + Box-Muller, deterministic)
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
// CANONICAL WEIGHT COLLECTOR & DETERMINISTIC FINGERPRINT
// ============================================================================

function pushBranchLayerTensors(
  tensors: UniCanonicalWeightTensor[],
  layerIndex: number,
  splitTower: boolean,
  branch: UniBranchId,
  weights: UniBranchLayerWeights,
  d: number,
  qOut: number,
  kvOut: number,
  dFf: number,
): void {
  tensors.push(
    {
      name: uniLayerNormWeightName(layerIndex, 'input', branch),
      role: 'input_layernorm',
      branch,
      layerIndex,
      splitTower,
      shape: [d],
      data: weights.inputLayerNorm,
    },
    {
      name: uniLayerAttnWeightName(layerIndex, 'q_proj', branch),
      role: 'q_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [qOut, d],
      data: weights.selfAttn.qProj,
    },
    {
      name: uniLayerAttnWeightName(layerIndex, 'k_proj', branch),
      role: 'k_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [kvOut, d],
      data: weights.selfAttn.kProj,
    },
    {
      name: uniLayerAttnWeightName(layerIndex, 'v_proj', branch),
      role: 'v_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [kvOut, d],
      data: weights.selfAttn.vProj,
    },
    {
      name: uniLayerAttnWeightName(layerIndex, 'o_proj', branch),
      role: 'o_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [d, qOut],
      data: weights.selfAttn.oProj,
    },
    {
      name: uniLayerNormWeightName(layerIndex, 'post_attention', branch),
      role: 'post_attention_layernorm',
      branch,
      layerIndex,
      splitTower,
      shape: [d],
      data: weights.postAttentionLayerNorm,
    },
    {
      name: uniLayerMlpWeightName(layerIndex, 'gate_proj', branch),
      role: 'gate_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [dFf, d],
      data: weights.mlp.gateProj,
    },
    {
      name: uniLayerMlpWeightName(layerIndex, 'up_proj', branch),
      role: 'up_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [dFf, d],
      data: weights.mlp.upProj,
    },
    {
      name: uniLayerMlpWeightName(layerIndex, 'down_proj', branch),
      role: 'down_proj',
      branch,
      layerIndex,
      splitTower,
      shape: [d, dFf],
      data: weights.mlp.downProj,
    },
  );
}

/**
 * Collect all 618 core Elastic-MoT backbone + U/G IO tensors (`139,024` floats)
 * in locked canonical order:
 *   1. `encoder.embed_tokens.weight` (`[64, 8]`, `u`)
 *   2. For `l = 0..33`:
 *      - 9 `U` tensors: `input_layernorm`, `q_proj`, `k_proj`, `v_proj`, `o_proj`,
 *        `post_attention_layernorm`, `gate_proj`, `up_proj`, `down_proj`
 *      - 9 `G` tensors (`_gen` suffix): same 9 shapes
 *   3. `encoder.norm.weight` (`[8]`, `u`)
 *   4. `diffusion_head.weight` (`[64, 8]`, `u`)
 *   5. `encoder.norm_gen.weight` (`[8]`, `g`)
 *   6. `encoder.gen_embedding.weight` (`[264, 8]`, `g`)
 *   7. `encoder.gen_predictor.weight` (`[256, 8]`, `g`)
 */
export function collectUniBackboneWeights(
  model: UniCoreModelParameters,
): readonly UniCanonicalWeightTensor[] {
  const cfg = model.config;
  const d = cfg.hiddenSize;
  const qOut = cfg.qProjOutDim;
  const kvOut = cfg.kvProjOutDim;
  const dFf = cfg.intermediateSize;

  const tensors: UniCanonicalWeightTensor[] = [
    {
      name: UNI_EMBED_TENSOR_NAME,
      role: 'embed_tokens',
      branch: 'u',
      layerIndex: null,
      splitTower: false,
      shape: [cfg.vocabSize, d],
      data: model.embedTokens,
    },
  ];

  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    pushBranchLayerTensors(tensors, layer.layerIndex, layer.splitTower, 'u', layer.u, d, qOut, kvOut, dFf);
    pushBranchLayerTensors(tensors, layer.layerIndex, layer.splitTower, 'g', layer.g, d, qOut, kvOut, dFf);
  }

  tensors.push(
    {
      name: UNI_FINAL_NORM_TENSOR_NAME,
      role: 'final_norm',
      branch: 'u',
      layerIndex: null,
      splitTower: false,
      shape: [d],
      data: model.finalNorm,
    },
    {
      name: UNI_DIFFUSION_HEAD_TENSOR_NAME,
      role: 'diffusion_head',
      branch: 'u',
      layerIndex: null,
      splitTower: false,
      shape: [cfg.vocabSize, d],
      data: model.diffusionHead,
    },
    {
      name: UNI_FINAL_NORM_GEN_TENSOR_NAME,
      role: 'final_norm_gen',
      branch: 'g',
      layerIndex: null,
      splitTower: false,
      shape: [d],
      data: model.finalNormGen,
    },
    {
      name: UNI_GEN_EMBEDDING_TENSOR_NAME,
      role: 'gen_embedding',
      branch: 'g',
      layerIndex: null,
      splitTower: false,
      shape: [cfg.genEmbeddingRows, d],
      data: model.genEmbedding,
    },
    {
      name: UNI_GEN_PREDICTOR_TENSOR_NAME,
      role: 'gen_predictor',
      branch: 'g',
      layerIndex: null,
      splitTower: false,
      shape: [cfg.vqCodebookSize, d],
      data: model.genPredictor,
    },
  );

  return tensors;
}

/**
 * Deterministic 64-bit FNV-1a fingerprint (16 lowercase hex chars) over canonical
 * tensor names, shapes, and exact IEEE-754 Float32 bit patterns.
 */
export function computeUniWeightsFingerprint(
  tensors: readonly UniCanonicalWeightTensor[],
): string {
  let h = FNV1A_64_OFFSET;
  const mixByte = (b: number): void => {
    h ^= BigInt(b & 0xff);
    h = (h * FNV1A_64_PRIME) & MASK_64;
  };
  const mixUint32 = (v: number): void => {
    const u = v >>> 0;
    mixByte(u & 0xff);
    mixByte((u >>> 8) & 0xff);
    mixByte((u >>> 16) & 0xff);
    mixByte((u >>> 24) & 0xff);
  };

  for (let i = 0; i < tensors.length; i++) {
    const t = tensors[i];
    for (let j = 0; j < t.name.length; j++) {
      mixByte(t.name.charCodeAt(j));
    }
    mixByte(0);
    mixUint32(t.shape.length);
    for (let s = 0; s < t.shape.length; s++) {
      mixUint32(t.shape[s]);
    }
    const bytes = new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    for (let b = 0; b < bytes.length; b++) {
      mixByte(bytes[b]);
    }
  }

  return h.toString(16).padStart(16, '0');
}

// ============================================================================
// CORE MODEL FACTORY
// ============================================================================

function createBranchLayerWeights(
  seed: number,
  layerIndex: number,
  branch: UniBranchId,
): UniBranchLayerWeights {
  const cfg = UNI_MINI_CONFIG;
  const d = cfg.hiddenSize;
  const qOut = cfg.qProjOutDim;
  const kvOut = cfg.kvProjOutDim;
  const dFf = cfg.intermediateSize;
  const residualBranchFactor = 2 * cfg.numLayers;

  const projInStd = 0.25 / Math.sqrt(d);
  const oProjStd = 0.25 / Math.sqrt(qOut * residualBranchFactor);
  const downProjStd = 0.25 / Math.sqrt(dFf * residualBranchFactor);

  return Object.freeze({
    inputLayerNorm: sampleRmsNormTensor(
      seed,
      uniLayerNormWeightName(layerIndex, 'input', branch),
      d,
    ),
    selfAttn: Object.freeze({
      qProj: sampleGaussianTensor(
        seed,
        uniLayerAttnWeightName(layerIndex, 'q_proj', branch),
        qOut * d,
        projInStd,
      ),
      kProj: sampleGaussianTensor(
        seed,
        uniLayerAttnWeightName(layerIndex, 'k_proj', branch),
        kvOut * d,
        projInStd,
      ),
      vProj: sampleGaussianTensor(
        seed,
        uniLayerAttnWeightName(layerIndex, 'v_proj', branch),
        kvOut * d,
        projInStd,
      ),
      oProj: sampleGaussianTensor(
        seed,
        uniLayerAttnWeightName(layerIndex, 'o_proj', branch),
        d * qOut,
        oProjStd,
      ),
    }),
    postAttentionLayerNorm: sampleRmsNormTensor(
      seed,
      uniLayerNormWeightName(layerIndex, 'post_attention', branch),
      d,
    ),
    mlp: Object.freeze({
      gateProj: sampleGaussianTensor(
        seed,
        uniLayerMlpWeightName(layerIndex, 'gate_proj', branch),
        dFf * d,
        projInStd,
      ),
      upProj: sampleGaussianTensor(
        seed,
        uniLayerMlpWeightName(layerIndex, 'up_proj', branch),
        dFf * d,
        projInStd,
      ),
      downProj: sampleGaussianTensor(
        seed,
        uniLayerMlpWeightName(layerIndex, 'down_proj', branch),
        d * dFf,
        downProjStd,
      ),
    }),
  });
}

export function createUniCoreModel(
  options: CreateUniCoreModelOptions = {},
): UniCoreModelParameters {
  const cfg = UNI_MINI_CONFIG;
  const seed = (options.backboneSeed ?? cfg.backboneSeed) >>> 0;
  const modelId = options.modelId ?? 'nld-uni-mini-34l-core-v1';

  const d = cfg.hiddenSize;
  const embedStd = 0.25;
  const headStd = 0.25 / Math.sqrt(d);

  const embedTokens = sampleGaussianTensor(seed, UNI_EMBED_TENSOR_NAME, cfg.vocabSize * d, embedStd);

  const layers: UniElasticLayerWeights[] = [];
  for (let l = 0; l < cfg.numLayers; l++) {
    const splitTower = l >= cfg.dualTowerLayers;
    layers.push(
      Object.freeze({
        layerIndex: l,
        splitTower,
        u: createBranchLayerWeights(seed, l, 'u'),
        g: createBranchLayerWeights(seed, l, 'g'),
      }),
    );
  }

  const finalNorm = sampleRmsNormTensor(seed, UNI_FINAL_NORM_TENSOR_NAME, d);
  const diffusionHead = sampleGaussianTensor(
    seed,
    UNI_DIFFUSION_HEAD_TENSOR_NAME,
    cfg.vocabSize * d,
    headStd,
  );
  const finalNormGen = sampleRmsNormTensor(seed, UNI_FINAL_NORM_GEN_TENSOR_NAME, d);
  const genEmbedding = sampleGaussianTensor(
    seed,
    UNI_GEN_EMBEDDING_TENSOR_NAME,
    cfg.genEmbeddingRows * d,
    embedStd,
  );
  const genPredictor = sampleGaussianTensor(
    seed,
    UNI_GEN_PREDICTOR_TENSOR_NAME,
    cfg.vqCodebookSize * d,
    headStd,
  );

  const draft: UniCoreModelParameters = {
    modelId,
    backboneSeed: seed,
    backboneFingerprint: '',
    config: cfg,
    vocab: UNI_MINI_VOCAB,
    embedTokens,
    layers: Object.freeze(layers),
    finalNorm,
    diffusionHead,
    finalNormGen,
    genEmbedding,
    genPredictor,
  };

  const collected = collectUniBackboneWeights(draft);
  if (collected.length !== UNI_MINI_CORE_TENSOR_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_CORE_TENSOR_COUNT} core tensors, collected ${collected.length}`,
    );
  }
  let totalFloats = 0;
  for (let i = 0; i < collected.length; i++) {
    totalFloats += collected[i].data.length;
  }
  if (totalFloats !== UNI_MINI_CORE_FLOAT_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_CORE_FLOAT_COUNT} core floats, collected ${totalFloats}`,
    );
  }

  const backboneFingerprint = computeUniWeightsFingerprint(collected);
  return Object.freeze({
    ...draft,
    backboneFingerprint,
  });
}

export function createDefaultUniCoreModel(): UniCoreModelParameters {
  return createUniCoreModel({
    modelId: 'nld-uni-mini-34l-core-v1',
    backboneSeed: UNI_BACKBONE_SEED,
  });
}
