# Independent PyTorch Reference Parity Workflow

Development-only verification scripts that validate TypeScript miniature simulators against clean-room PyTorch (`torch==2.15.0`) implementations of NVIDIA's `NemotronH` equations. Python/PyTorch is **never** a browser runtime dependency; Vitest consumes the checked-in JSON fixtures directly in Node.

## Pinned Environment
- Python: `3.12.x` (`>=3.12,<3.13`, PEP 723 inline metadata in `generate_pytorch_parity_fixtures.py`)
- PyTorch: `torch==2.15.0` (pinned in both PEP 723 metadata and `requirements-ref.txt`)

## Reproducing All Reference Fixtures

### Nemotron 3 Nano v1.0 (Single-Tower Causal Hybrid)

```bash
# 1. Export all 52 layers of createDefaultHybridModel() as exact little-endian IEEE-754 Float32 + SHA-256
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params

# 2. Run pinned PyTorch 2.15.0 over single-layer components AND the full 52-layer chain
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json

# 3. Verify parity in Vitest (zero Python runtime required during npm test)
npm test
```

### Nemotron-Labs-TwoTower (Dual-Tower Context + Block-Diffusion Denoiser)

Generates 3 committed JSON parity fixtures covering all 8 mandatory verification categories (timestep & split adaLN-single modulation + order-inversion sensitivity, bidirectional rectangular `[B x (P+B)]` GQA + future-token influence, seeded Mamba-2 conv/recurrence + prefix-stitching + `O(1e3)` FP32 SSM stability, FP64 MoE routing + near-tie vs. exact-tie fixtures, complete 52-layer Context-to-Denoiser handoff with 0-ULP Context immutability, full two-block diffusion trajectory with Context commits, AR vs. Mock-AR alignment under both perturbed and zero-perturbation identity weights, and MDLM/SUBS confidence-unmasking + deterministic Gumbel-max sampler edge cases):

```bash
# 1. Export all 860 canonical Float32 tensors (context.*, denoiser.*, time_cond.*)
#    for BOTH default (perturbationScale=0.15) and identity (perturbationScale=0) models
node scripts/reference/export_twotower_params.mjs

# 2. Run pinned Python 3.12 / PyTorch 2.15.0 clean-room reference generator
#    (verifies both weightsSha256 digests and writes tests/fixtures/pytorch-twotower-*-parity.json)
uv run scripts/reference/generate_twotower_pytorch_fixtures.py

# 3. Verify all Nano + TwoTower parity tests in Vitest (zero Python runtime required during npm test)
npm run typecheck && npm test
```

Outputs written to `tests/fixtures/`:
- `tests/fixtures/pytorch-twotower-blocks-parity.json` (Categories 1–4)
- `tests/fixtures/pytorch-twotower-handoff-parity.json` (Category 5)
- `tests/fixtures/pytorch-twotower-trajectory-parity.json` (Categories 6–8)

## Reference Attribution & Licensing
- Nano architecture equations follow Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`modeling_nemotron_h.py`, Apache License 2.0).
- TwoTower reference math is implemented clean-room from first principles in standard PyTorch (`torch`, `torch.nn.functional`) matching the verified mathematical equations of Nemotron-Labs-TwoTower.
- No third-party source files or NVIDIA production model weights are copied into this repository; all parameters are deterministic synthetic miniature Float32 tensors verified by cryptographic SHA-256 digests.
