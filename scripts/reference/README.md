# Independent PyTorch Reference Parity Workflow

Development-only verification scripts that validate TypeScript miniature simulators against clean-room PyTorch (`torch==2.15.0`) implementations of NVIDIA's `NemotronH` equations. Python/PyTorch is **never** a browser runtime dependency; Vitest consumes the checked-in JSON fixtures directly in Node.

## Pinned Environment
- Python: `3.12.x` (`>=3.12,<3.13`, PEP 723 inline metadata in `generate_pytorch_parity_fixtures.py`)
- PyTorch: `torch==2.15.0` (pinned in both PEP 723 metadata and `requirements-ref.txt`)

## Reproducing All Reference Fixtures

```bash
# 1. Export all 52 layers of createDefaultHybridModel() as exact little-endian IEEE-754 Float32 + SHA-256
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params

# 2. Run pinned PyTorch 2.15.0 over single-layer components AND the full 52-layer chain
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json

# 3. Verify parity in Vitest (zero Python runtime required during npm test)
npm test
```

## Reference Attribution & Licensing
- Architecture equations follow Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`modeling_nemotron_h.py`, Apache License 2.0).
- No third-party source files or NVIDIA model weights are copied into this repository; all parameters are deterministic synthetic miniature Float32 tensors.
