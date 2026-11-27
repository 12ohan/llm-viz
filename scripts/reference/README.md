# Independent PyTorch Reference Parity Workflow

Development-only verification scripts that validate all four TypeScript miniature simulators (**Nemotron 3 Nano**, **Nemotron-Labs-TwoTower**, **Nemotron-Labs-Diffusion-14B**, and **Nemotron-Labs-Diffusion-Uni**) against clean-room PyTorch (`torch==2.15.0`) implementations of NVIDIA's published architectural equations. Python/PyTorch is **never** a browser runtime dependency; Vitest consumes the checked-in JSON fixtures directly in Node.

## Pinned Environment
- Python: `3.12.x` (`>=3.12,<3.13`, PEP 723 inline metadata in every `generate_*.py` script)
- PyTorch: `torch==2.15.0` (pinned in both PEP 723 metadata and `requirements-ref.txt`)

## Reproducing All Reference Fixtures

### 1. Nemotron 3 Nano `v1.0.0` (Single-Tower Causal Hybrid)

```bash
# 1. Export all 52 layers of createDefaultHybridModel() as exact little-endian IEEE-754 Float32 + SHA-256
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params

# 2. Run pinned PyTorch 2.15.0 over single-layer components AND the full 52-layer chain
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json
```

Outputs written to `tests/fixtures/`:
- `tests/fixtures/pytorch-reference-parity.json` (Single-layer components)
- `tests/fixtures/pytorch-52layer-parity.json` (Full 52-layer chain)
- `tests/fixtures/pytorch-incremental-52layer-parity.json` (52-layer cached incremental AR decode)

### 2. Nemotron-Labs-TwoTower `twotower-v1.0.0` (Dual-Tower Context + Block-Diffusion Denoiser)

```bash
# 1. Export all 860 canonical Float32 tensors (context.*, denoiser.*, time_cond.*)
#    for BOTH default (perturbationScale=0.15) and identity (perturbationScale=0) models
node scripts/reference/export_twotower_params.mjs

# 2. Run pinned Python 3.12 / PyTorch 2.15.0 clean-room reference generator
#    (verifies both weightsSha256 digests and writes Categories 1–8 fixtures)
uv run scripts/reference/generate_twotower_pytorch_fixtures.py
```

Outputs written to `tests/fixtures/`:
- `tests/fixtures/pytorch-twotower-blocks-parity.json` (Categories 1–4)
- `tests/fixtures/pytorch-twotower-handoff-parity.json` (Category 5)
- `tests/fixtures/pytorch-twotower-trajectory-parity.json` (Categories 6–8)

### 3. Nemotron-Labs-Diffusion-14B `diffusion-v1.0.0` (40L Ministral 3 Dense Decoder + Draft LoRA)

```bash
# 1. Export all 40L backbone + draft-only o_proj LoRA Float32 tensors + SHA-256 manifest
node scripts/reference/export_diffusion_params.mjs

# 2. Run pinned PyTorch 2.15.0 over Categories 1–6 (blocks/masks/YaRN/GQA/SwiGLU/40L)
uv run scripts/reference/generate_diffusion_pytorch_fixtures.py

# 3. Run pinned PyTorch 2.15.0 over Categories 7–11 (cached AR, Samplers A/B, Spec Verify & LoRA)
uv run scripts/reference/generate_diffusion_trajectory_fixtures.py
```

Outputs written to `tests/fixtures/`:
- `tests/fixtures/pytorch-diffusion-blocks-parity.json` (Categories 1–6)
- `tests/fixtures/pytorch-diffusion-trajectories-parity.json` (Categories 7–11)

### 4. Nemotron-Labs-Diffusion-Uni `uni-v1.0.0` (18.27B Elastic-MoT Multimodal Backbone, Understanding & Diffusion)

```bash
# 1. Export 34L Elastic-MoT backbone tensors and generate backbone parity fixture
node scripts/reference/export_uni_params.mjs
uv run scripts/reference/generate_uni_backbone_pytorch_fixtures.py

# 2. Export full multimodal tensors (RADIO-H, Emu3 IBQ, 2x2 Codec, D=8 ball-tree, U/G towers)
node scripts/reference/export_uni_multimodal_params.mjs

# 3. Generate multimodal understanding & hierarchical discrete diffusion parity fixtures
uv run scripts/reference/generate_uni_multimodal_pytorch_fixtures.py
uv run scripts/reference/generate_uni_diffusion_pytorch_fixtures.py
```

Outputs written to `tests/fixtures/`:
- `tests/fixtures/pytorch-uni-backbone.json` (34L Joint/Split Elastic-MoT backbone, YaRN + 2D SinCos-G, `D5` prefill equivalence)
- `tests/fixtures/pytorch-uni-multimodal.json` (RADIO-H, Emu3 IBQ, `2×2` Codec, FLARE understanding & `<LOC_*>` grounding)
- `tests/fixtures/pytorch-uni-diffusion.json` (`BalancedSemanticTree`, stratified quadtree refinement, Dual-CFG edit & read-only OPD)

### Verify All Fixtures in Node

```bash
npm run typecheck && npm test
```

## Reference Attribution & Licensing
- All four reference math suites are implemented clean-room from first principles in standard PyTorch (`torch`, `torch.nn.functional`) matching the verified mathematical equations of Nemotron 3 Nano, Nemotron-Labs-TwoTower, Nemotron-Labs-Diffusion-14B, and Nemotron-Labs-Diffusion-Uni.
- No third-party source files, NVIDIA production weights, LoRA adapters, or `pca_tree.pt` binaries are copied into this repository; all parameters are deterministic synthetic miniature Float32 tensors verified by cryptographic SHA-256 digests.
