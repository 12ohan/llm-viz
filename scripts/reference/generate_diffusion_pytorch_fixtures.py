#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator for
Nemotron-Labs-Diffusion Miniature Simulator (DF1 Categories 1-6).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-diffusion-blocks-parity.json`:
  node scripts/reference/export_diffusion_params.mjs
  uv run scripts/reference/generate_diffusion_pytorch_fixtures.py

Clean-Room Mathematical Reference & Licensing Note:
  Implements Nemotron-Labs-Diffusion's verified mathematical equations from first principles
  using standard PyTorch (`torch`, `torch.nn.functional`):
    - Exact YaRN RoPE (`truncate=True`) with dyadic `inv_freq = [1.0, 0.3125, 0.0625, 0.03125]`
    - Half-split rotary embedding (`rotate_half`) & Llama-4 query temperature scaling
    - 5-mode attention masks + 3-coordinate separation (`seq_idx`, `rope_position_ids`, `cache_position`)
    - 40-layer Ministral-3 dense decoder with `32Q/8KV` GQA, transient `use_cache=False` concat,
      draft-only `o_proj` LoRA (`4.0 * B(A * a)`), FP32-variance RMSNorm, and dense SwiGLU
  Zero NVIDIA Python source code or production checkpoint weights are copied or touched;
  all parameters are deterministic miniature IEEE-754 Float32 tensors verified by SHA-256.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PINNED_TORCH_VERSION = "2.15.0"
DEFAULT_PARAMS_DIR = Path("/tmp/nemotron-viz-diffusion-ref")
EXPECTED_DYADIC_INV_FREQ = [1.0, 0.3125, 0.0625, 0.03125]


# ============================================================================
# STRICT FINITE-VALUE GUARDS & SERIALIZERS
# ============================================================================

def assert_all_finite(t: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(t).all().item()):
        raise RuntimeError(f"Unexpected non-finite value detected in {label}")


def to_f32_list(t: torch.Tensor, label: str = "tensor") -> list[float]:
    t32 = t.detach().to(torch.float32).contiguous()
    assert_all_finite(t32, label)
    return [float(x) for x in t32.reshape(-1).tolist()]


def to_i32_list(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.detach().to(torch.int64).reshape(-1).tolist()]


def to_nullable_scores(scores: torch.Tensor, label: str) -> list[float | None]:
    flat = scores.detach().to(torch.float32).reshape(-1).tolist()
    out: list[float | None] = []
    for idx, v in enumerate(flat):
        if math.isnan(v) or (math.isinf(v) and v > 0):
            raise RuntimeError(f"Unexpected non-finite score {v} at {label}[{idx}]")
        out.append(None if math.isinf(v) else float(v))
    return out


# ============================================================================
# EXPORTED WEIGHT LOADING WITH CRYPTOGRAPHIC SHA-256 VERIFICATION
# ============================================================================

def load_exported_diffusion_weights(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing exported Diffusion manifest: {manifest_path}. "
            "Run `node scripts/reference/export_diffusion_params.mjs` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = Path(manifest["weightsFile"])
    raw_bytes = weights_path.read_bytes()

    actual_weights_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_weights_sha256 != manifest["weightsSha256"]:
        raise RuntimeError(
            f"Diffusion weightsSha256 mismatch: {actual_weights_sha256} != {manifest['weightsSha256']}"
        )

    backbone_hash = hashlib.sha256()
    lora_hash = hashlib.sha256()
    backbone_count = 0
    lora_count = 0
    backbone_floats = 0
    lora_floats = 0

    floats = array.array("f")
    floats.frombytes(raw_bytes)
    if sys.byteorder != "little":
        floats.byteswap()
    if floats.itemsize != 4:
        raise RuntimeError(f"Expected 4-byte float32 items, got {floats.itemsize}")

    flat_tensor = torch.tensor(floats, dtype=torch.float32)
    assert_all_finite(flat_tensor, "exported_diffusion_weights")

    tensor_map: dict[str, torch.Tensor] = {}
    for entry in manifest["tensors"]:
        name = entry["name"]
        off = int(entry["offsetFloats"])
        length = int(entry["length"])
        shape = [int(s) for s in entry["shape"]]
        byte_slice = raw_bytes[off * 4 : (off + length) * 4]
        if hashlib.sha256(byte_slice).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Per-tensor SHA-256 mismatch for {name}")
        if entry["isLora"]:
            lora_hash.update(byte_slice)
            lora_count += 1
            lora_floats += length
        else:
            backbone_hash.update(byte_slice)
            backbone_count += 1
            backbone_floats += length
        tensor_map[name] = flat_tensor[off : off + length].view(*shape).contiguous()

    if backbone_hash.hexdigest() != manifest["backboneSha256"]:
        raise RuntimeError("Diffusion backboneSha256 mismatch")
    if lora_hash.hexdigest() != manifest["loraSha256"]:
        raise RuntimeError("Diffusion loraSha256 mismatch")
    if (
        len(tensor_map) != 443
        or backbone_count != 363
        or lora_count != 80
        or backbone_floats != 229_064
        or lora_floats != 42_240
    ):
        raise RuntimeError(
            f"Unexpected manifest counts: total={len(tensor_map)}, backbone={backbone_count}/{backbone_floats}, "
            f"lora={lora_count}/{lora_floats}"
        )
    return manifest, tensor_map


# ============================================================================
# CATEGORY 1: PRODUCTION 0 B SAFETENSORS + LFS INVENTORY (DF0 §2.2)
# ============================================================================

def build_verified_lfs_inventory() -> list[dict[str, Any]]:
    entries = [
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-14B",
            "revision": "f8c3e2c078e193599b8882d965b1001c456ba738",
            "file": "model.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 363,
            "parameters": 13_506_073_600,
            "headerBytes": 43_512,
            "payloadBytes": 27_012_147_200,
            "totalBytes": 27_012_190_712,
            "hfLfsSizeBytes": 27_012_190_712,
            "diffBytes": 0,
            "lfsSha256Prefix": "272e78c39809",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-14B",
            "revision": "f8c3e2c078e193599b8882d965b1001c456ba738",
            "file": "linear_spec_lora",
            "dtype": "F32",
            "isLora": True,
            "tensors": 80,
            "parameters": 47_185_920,
            "headerBytes": 11_120,
            "payloadBytes": 188_743_680,
            "totalBytes": 188_754_800,
            "hfLfsSizeBytes": 188_754_800,
            "diffBytes": 0,
            "lfsSha256Prefix": "cb29e2595fe0",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-14B-Base",
            "revision": "ca30a6db6274c0b01d4eb21e6f1cb28222ea0c63",
            "file": "model.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 363,
            "parameters": 13_506_073_600,
            "headerBytes": 43_512,
            "payloadBytes": 27_012_147_200,
            "totalBytes": 27_012_190_712,
            "hfLfsSizeBytes": 27_012_190_712,
            "diffBytes": 0,
            "lfsSha256Prefix": "f43537460484",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-8B",
            "revision": "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc",
            "file": "model.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 309,
            "parameters": 8_489_553_920,
            "headerBytes": 36_880,
            "payloadBytes": 16_979_107_840,
            "totalBytes": 16_979_144_720,
            "hfLfsSizeBytes": 16_979_144_720,
            "diffBytes": 0,
            "lfsSha256Prefix": "14de6f8fee56",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-8B",
            "revision": "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc",
            "file": "linear_spec_lora",
            "dtype": "F32",
            "isLora": True,
            "tensors": 68,
            "parameters": 35_651_584,
            "headerBytes": 9_432,
            "payloadBytes": 142_606_336,
            "totalBytes": 142_615_768,
            "hfLfsSizeBytes": 142_615_768,
            "diffBytes": 0,
            "lfsSha256Prefix": "5893f4aec2f4",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-8B-Base",
            "revision": "59ff0ffee284112fc6ccf37493ced41a11031434",
            "file": "model.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 309,
            "parameters": 8_489_553_920,
            "headerBytes": 36_880,
            "payloadBytes": 16_979_107_840,
            "totalBytes": 16_979_144_720,
            "hfLfsSizeBytes": 16_979_144_720,
            "diffBytes": 0,
            "lfsSha256Prefix": "73af2cd1c982",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-3B",
            "revision": "0d51902da1f8869f83413ce642fab402fa5641e0",
            "file": "model.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 237,
            "parameters": 3_831_659_520,
            "headerBytes": 27_960,
            "payloadBytes": 7_663_319_040,
            "totalBytes": 7_663_347_000,
            "hfLfsSizeBytes": 7_663_347_000,
            "diffBytes": 0,
            "lfsSha256Prefix": "8c4beff3a8a2",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-3B",
            "revision": "0d51902da1f8869f83413ce642fab402fa5641e0",
            "file": "linear_spec_lora",
            "dtype": "F32",
            "isLora": True,
            "tensors": 52,
            "parameters": 23_855_104,
            "headerBytes": 7_184,
            "payloadBytes": 95_420_416,
            "totalBytes": 95_427_600,
            "hfLfsSizeBytes": 95_427_600,
            "diffBytes": 0,
            "lfsSha256Prefix": "897ef67dff8a",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-3B-Base",
            "revision": "4bd78816ba988ca8b71b8f432b00b682572b236c",
            "file": "model.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 237,
            "parameters": 3_831_659_520,
            "headerBytes": 27_960,
            "payloadBytes": 7_663_319_040,
            "totalBytes": 7_663_347_000,
            "hfLfsSizeBytes": 7_663_347_000,
            "diffBytes": 0,
            "lfsSha256Prefix": "cf9ef7826f4d",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-VLM-8B",
            "revision": "adca93d16471c1e07d594ae444d23e1876f6b365",
            "file": "model-00001-of-00004.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 286,
            "parameters": 2_492_131_328,
            "headerBytes": 39_432,
            "payloadBytes": 4_984_262_656,
            "totalBytes": 4_984_302_088,
            "hfLfsSizeBytes": 4_984_302_088,
            "diffBytes": 0,
            "lfsSha256Prefix": "d6bdc8ac3f1b",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-VLM-8B",
            "revision": "adca93d16471c1e07d594ae444d23e1876f6b365",
            "file": "model-00002-of-00004.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 104,
            "parameters": 2_499_895_296,
            "headerBytes": 12_312,
            "payloadBytes": 4_999_790_592,
            "totalBytes": 4_999_802_904,
            "hfLfsSizeBytes": 4_999_802_904,
            "diffBytes": 0,
            "lfsSha256Prefix": "6e4c3f9cac91",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-VLM-8B",
            "revision": "adca93d16471c1e07d594ae444d23e1876f6b365",
            "file": "model-00003-of-00004.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 100,
            "parameters": 2_457_952_256,
            "headerBytes": 11_864,
            "payloadBytes": 4_915_904_512,
            "totalBytes": 4_915_916_376,
            "hfLfsSizeBytes": 4_915_916_376,
            "diffBytes": 0,
            "lfsSha256Prefix": "562159c86be4",
        },
        {
            "repoId": "nvidia/Nemotron-Labs-Diffusion-VLM-8B",
            "revision": "adca93d16471c1e07d594ae444d23e1876f6b365",
            "file": "model-00004-of-00004.safetensors",
            "dtype": "BF16",
            "isLora": False,
            "tensors": 41,
            "parameters": 1_468_055_552,
            "headerBytes": 4_864,
            "payloadBytes": 2_936_111_104,
            "totalBytes": 2_936_115_968,
            "hfLfsSizeBytes": 2_936_115_968,
            "diffBytes": 0,
            "lfsSha256Prefix": "7323f1fdacbf",
        },
    ]
    for e in entries:
        bpe = 2 if e["dtype"] == "BF16" else 4
        assert e["payloadBytes"] == e["parameters"] * bpe
        assert e["headerBytes"] + e["payloadBytes"] == e["totalBytes"]
        assert e["totalBytes"] == e["hfLfsSizeBytes"]
        assert e["diffBytes"] == 0
    return entries


# ============================================================================
# CLEAN-ROOM PYTORCH 2.15.0 MINISTRAL-3 DIFFUSION EQUATIONS
# ============================================================================

def get_yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_yarn_inv_freq_ref(rope_cfg: dict[str, Any]) -> dict[str, Any]:
    dim = int(rope_cfg["headDim"])
    half_dim = dim // 2
    base = float(rope_cfg["ropeTheta"])
    orig_max_pos = int(rope_cfg["originalMaxPositionEmbeddings"])
    factor = float(rope_cfg["factor"])
    beta_fast = float(rope_cfg["betaFast"])
    beta_slow = float(rope_cfg["betaSlow"])
    mscale = float(rope_cfg["mscale"])
    mscale_all_dim = float(rope_cfg["mscaleAllDim"])
    truncate = bool(rope_cfg["truncate"])

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(orig_max_pos / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))

    raw_low = find_correction_dim(beta_fast)
    raw_high = find_correction_dim(beta_slow)
    if truncate:
        low = max(math.floor(raw_low), 0)
        high = min(math.ceil(raw_high), half_dim - 1)
    else:
        low = max(raw_low, 0.0)
        high = min(raw_high, float(half_dim - 1))

    denom = 0.001 if high == low else (high - low)
    linear = (torch.arange(half_dim, dtype=torch.float32) - float(low)) / float(denom)
    extrap_factor = 1.0 - torch.clamp(linear, 0.0, 1.0)
    inv_freq = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor

    attention_scaling = get_yarn_mscale(factor, mscale) / get_yarn_mscale(factor, mscale_all_dim)
    zones = [
        "original" if float(ef) == 1.0 else ("interpolated" if float(ef) == 0.0 else "blended")
        for ef in extrap_factor.tolist()
    ]
    assert_all_finite(inv_freq, "yarn.inv_freq")
    return {
        "invFreq": inv_freq,
        "attentionScaling": float(attention_scaling),
        "low": low,
        "high": high,
        "extrapolationFactor": extrap_factor,
        "zones": zones,
    }


def compute_rope_cos_sin_ref(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.outer(position_ids.to(torch.float32), inv_freq.to(torch.float32))
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb) * float(attention_scaling)
    sin = torch.sin(emb) * float(attention_scaling)
    assert_all_finite(cos, "rope.cos")
    assert_all_finite(sin, "rope.sin")
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_ref(
    x: torch.Tensor,  # [seqLen, numHeads, headDim]
    cos: torch.Tensor,  # [seqLen, headDim]
    sin: torch.Tensor,  # [seqLen, headDim]
) -> torch.Tensor:
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    out = (x * c) + (rotate_half(x) * s)
    assert_all_finite(out, "rope.rotated")
    return out


def compute_llama4_attn_scales_ref(
    cache_positions: torch.Tensor,
    beta: float,
    orig_max_pos: int,
) -> torch.Tensor:
    pos_f = cache_positions.to(torch.float32)
    scales = 1.0 + float(beta) * torch.log(1.0 + torch.floor(pos_f / float(orig_max_pos)))
    assert_all_finite(scales, "llama4.scales")
    return scales


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Ministral3RMSNorm: variance in FP32 -> normalize -> cast back -> multiply weight."""
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_f32 * torch.rsqrt(variance + float(eps))).to(input_dtype)
    out = x_norm * weight
    assert_all_finite(out, "rmsnorm.out")
    return out


def attention_ref(
    hidden_states: torch.Tensor,  # [S, d]
    layer_idx: int,
    tensor_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    position_ids: torch.Tensor,  # [S]
    cache_positions: torch.Tensor,  # [S]
    mask_mode: str,
    lora_enabled: bool,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    custom_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    s_len = hidden_states.shape[0]
    n_q = int(cfg["numAttentionHeads"])
    n_kv = int(cfg["numKeyValueHeads"])
    d_head = int(cfg["headDim"])
    n_rep = n_q // n_kv
    q_dim = n_q * d_head
    kv_dim = n_kv * d_head

    wq = tensor_map[f"encoder.layers.{layer_idx}.self_attn.q_proj.weight"]
    wk = tensor_map[f"encoder.layers.{layer_idx}.self_attn.k_proj.weight"]
    wv = tensor_map[f"encoder.layers.{layer_idx}.self_attn.v_proj.weight"]
    wo = tensor_map[f"encoder.layers.{layer_idx}.self_attn.o_proj.weight"]

    q_pre = F.linear(hidden_states, wq)  # [S, 256]
    k_pre = F.linear(hidden_states, wk)  # [S, 64]
    v_new = F.linear(hidden_states, wv)  # [S, 64]

    cos, sin = compute_rope_cos_sin_ref(inv_freq, position_ids, attention_scaling)
    q_rot_h = apply_rotary_pos_emb_ref(q_pre.view(s_len, n_q, d_head), cos, sin)
    k_rot_h = apply_rotary_pos_emb_ref(k_pre.view(s_len, n_kv, d_head), cos, sin)

    llama4_scales = compute_llama4_attn_scales_ref(
        cache_positions,
        float(cfg["rope"]["llama4Beta"]),
        int(cfg["rope"]["originalMaxPositionEmbeddings"]),
    )
    q_scaled_h = q_rot_h * llama4_scales.view(s_len, 1, 1)

    q_rot = q_rot_h.reshape(s_len, q_dim)
    k_rot = k_rot_h.reshape(s_len, kv_dim)
    q_scaled = q_scaled_h.reshape(s_len, q_dim)

    if past_kv is not None and past_kv[0].shape[0] > 0:
        past_k, past_v = past_kv
        past_len = past_k.shape[0]
        k_full = torch.cat([past_k, k_rot], dim=0).contiguous()
        v_full = torch.cat([past_v, v_new], dim=0).contiguous()
    else:
        past_len = 0
        k_full = k_rot
        v_full = v_new

    kv_len = k_full.shape[0]
    q_h = q_scaled_h.transpose(0, 1).contiguous()  # [nQ, S, dHead]
    k_kv_h = k_full.view(kv_len, n_kv, d_head).transpose(0, 1).contiguous()  # [nKV, kvLen, dHead]
    v_kv_h = v_full.view(kv_len, n_kv, d_head).transpose(0, 1).contiguous()  # [nKV, kvLen, dHead]
    k_rep = k_kv_h.repeat_interleave(n_rep, dim=0)  # [nQ, kvLen, dHead]
    v_rep = v_kv_h.repeat_interleave(n_rep, dim=0)  # [nQ, kvLen, dHead]

    scale = 1.0 / math.sqrt(d_head)
    raw_scores = torch.matmul(q_h, k_rep.transpose(-1, -2)) * scale  # [nQ, S, kvLen]

    if mask_mode == "causal":
        q_idx = torch.arange(past_len, past_len + s_len, dtype=torch.long).unsqueeze(1)
        k_idx = torch.arange(0, kv_len, dtype=torch.long).unsqueeze(0)
        visible_mask = k_idx <= q_idx
        scores = raw_scores.masked_fill(~visible_mask.unsqueeze(0), float("-inf"))
        sdpa_mask = visible_mask
    elif mask_mode == "bidirectional":
        scores = raw_scores
        sdpa_mask = None
    elif mask_mode == "custom":
        if custom_mask is None:
            raise ValueError("custom_mask required when mask_mode == 'custom'")
        visible_mask = custom_mask.to(torch.bool).view(s_len, kv_len)
        scores = raw_scores.masked_fill(~visible_mask.unsqueeze(0), float("-inf"))
        sdpa_mask = visible_mask
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode!r}")

    probs = F.softmax(scores.to(torch.float32), dim=-1)

    # Cross-check against PyTorch scaled_dot_product_attention (enable_gqa=True when unmasked/masked)
    try:
        v_out_sdpa = F.scaled_dot_product_attention(
            q_h.unsqueeze(0),
            k_kv_h.unsqueeze(0),
            v_kv_h.unsqueeze(0),
            attn_mask=sdpa_mask,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        ).squeeze(0)
    except TypeError:
        v_out_sdpa = F.scaled_dot_product_attention(
            q_h.unsqueeze(0),
            k_rep.unsqueeze(0),
            v_rep.unsqueeze(0),
            attn_mask=sdpa_mask,
            is_causal=False,
            scale=scale,
        ).squeeze(0)

    v_out_explicit = torch.matmul(probs, v_rep)
    sdpa_diff = float((v_out_sdpa - v_out_explicit).abs().max().item())
    if sdpa_diff > 1e-5:
        raise RuntimeError(f"SDPA vs explicit softmax@V mismatch at layer {layer_idx}: {sdpa_diff}")

    attn_concat = v_out_sdpa.transpose(0, 1).contiguous().view(s_len, q_dim)
    o_proj_base = F.linear(attn_concat, wo)

    if lora_enabled:
        lora_a = tensor_map[f"base_model.model.encoder.layers.{layer_idx}.self_attn.o_proj.lora_A.weight"]
        lora_b = tensor_map[f"base_model.model.encoder.layers.{layer_idx}.self_attn.o_proj.lora_B.weight"]
        lora_hidden = F.linear(attn_concat, lora_a)
        lora_raw = F.linear(lora_hidden, lora_b)
        lora_delta = float(cfg["loraScale"]) * lora_raw
        output = o_proj_base + lora_delta
    else:
        lora_hidden = None
        lora_delta = None
        output = o_proj_base.clone()

    assert_all_finite(output, f"layer{layer_idx}.attn.output")
    return {
        "qPre": q_pre,
        "kPre": k_pre,
        "vNew": v_new,
        "qRot": q_rot,
        "kRot": k_rot,
        "llama4Scales": llama4_scales,
        "qScaled": q_scaled,
        "kFull": k_full,
        "vFull": v_full,
        "scores": scores,
        "probs": probs,
        "attnConcat": attn_concat,
        "oProjBase": o_proj_base,
        "loraHidden": lora_hidden,
        "loraDelta": lora_delta,
        "output": output,
        "pastLen": past_len,
        "kvLen": kv_len,
        "newKv": (k_rot, v_new),
    }


def swiglu_ref(
    hidden_states: torch.Tensor,
    layer_idx: int,
    tensor_map: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    w_gate = tensor_map[f"encoder.layers.{layer_idx}.mlp.gate_proj.weight"]
    w_up = tensor_map[f"encoder.layers.{layer_idx}.mlp.up_proj.weight"]
    w_down = tensor_map[f"encoder.layers.{layer_idx}.mlp.down_proj.weight"]

    gate_pre = F.linear(hidden_states, w_gate)
    gate_silu = F.silu(gate_pre)
    up_val = F.linear(hidden_states, w_up)
    gated = gate_silu * up_val
    output = F.linear(gated, w_down)
    assert_all_finite(output, f"layer{layer_idx}.swiglu.output")
    return {
        "gatePre": gate_pre,
        "gateSilu": gate_silu,
        "upVal": up_val,
        "gated": gated,
        "output": output,
    }


def forward_40_layers_ref(
    token_ids: list[int],
    tensor_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    mask_mode: str,
    lora_enabled: bool,
    use_cache: bool,
    past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    s_len = len(token_ids)
    past_len = past_kvs[0][0].shape[0] if (past_kvs is not None and len(past_kvs) > 0) else 0
    pos_ids = torch.arange(past_len, past_len + s_len, dtype=torch.long)
    cache_pos = pos_ids.clone()

    embed_w = tensor_map["encoder.embed_tokens.weight"]
    tok_t = torch.tensor(token_ids, dtype=torch.long)
    embeddings = embed_w[tok_t].contiguous()  # [S, 8]
    h = embeddings.clone()

    residual_checkpoints: list[torch.Tensor] = [h.clone()]
    new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    eps = float(cfg["rmsNormEps"])

    for layer_idx in range(int(cfg["numLayers"])):
        in_norm_w = tensor_map[f"encoder.layers.{layer_idx}.input_layernorm.weight"]
        post_norm_w = tensor_map[f"encoder.layers.{layer_idx}.post_attention_layernorm.weight"]

        in_norm = rmsnorm_ref(h, in_norm_w, eps)
        attn_res = attention_ref(
            in_norm,
            layer_idx,
            tensor_map,
            cfg,
            inv_freq,
            attention_scaling,
            pos_ids,
            cache_pos,
            mask_mode=mask_mode,
            lora_enabled=lora_enabled,
            past_kv=past_kvs[layer_idx] if past_kvs is not None else None,
        )
        post_attn_res = h + attn_res["output"]
        post_norm = rmsnorm_ref(post_attn_res, post_norm_w, eps)
        mlp_res = swiglu_ref(post_norm, layer_idx, tensor_map)
        h = post_attn_res + mlp_res["output"]
        residual_checkpoints.append(h.clone())
        new_kvs.append(attn_res["newKv"])

    final_norm = rmsnorm_ref(h, tensor_map["encoder.norm.weight"], eps)
    logits = F.linear(final_norm, tensor_map["diffusion_head.weight"])
    probs = F.softmax(logits.to(torch.float32), dim=-1)
    top5 = torch.topk(logits, k=5, dim=-1).indices.tolist()

    return {
        "tokenIds": token_ids,
        "seqLen": s_len,
        "pastLen": past_len,
        "kvLen": past_len + s_len,
        "maskMode": mask_mode,
        "loraEnabled": lora_enabled,
        "useCache": use_cache,
        "positionIds": to_i32_list(pos_ids),
        "cachePositions": to_i32_list(cache_pos),
        "embeddings": embeddings,
        "residualCheckpoints": residual_checkpoints,
        "finalNorm": final_norm,
        "logits": logits,
        "probs": probs,
        "top5ByPosition": top5,
        "newKvs": new_kvs,
    }


# ============================================================================
# CATEGORY 3: 5-MODE MASKS & 3-COORDINATE FIXTURES
# ============================================================================

def build_mask_fixtures() -> dict[str, Any]:
    # 1. Joint Training [x_t || x_0] (L=6, B=2 -> 12x12)
    l_seq, b_size = 6, 2
    q_len = 2 * l_seq
    jt_vis: list[int] = []
    jt_regions: list[str] = []
    jt_rope = list(range(l_seq)) + list(range(l_seq))
    jt_seq = list(range(q_len))
    jt_cache = list(range(q_len))
    min_delta_p: int | None = None
    zero_twin_violations = 0

    for q in range(q_len):
        fq = 1 if q >= l_seq else 0
        bq = (q - l_seq * fq) // b_size
        for k in range(q_len):
            fk = 1 if k >= l_seq else 0
            bk = (k - l_seq * fk) // b_size
            if bq == bk and fk == 0 and fq == 0:
                reg = "M_BD"
                vis = 1
            elif bq > bk and fk == 1 and fq == 0:
                reg = "M_OBC"
                vis = 1
                dp = jt_rope[q] - jt_rope[k]
                if min_delta_p is None or dp < min_delta_p:
                    min_delta_p = dp
                if dp == 0:
                    zero_twin_violations += 1
            elif q >= k and fk == 1 and fq == 1:
                reg = "M_BC"
                vis = 1
            else:
                reg = "none"
                vis = 0
            jt_vis.append(vis)
            jt_regions.append(reg)

    assert min_delta_p is not None and min_delta_p >= 1 and zero_twin_violations == 0

    # 2A. AR Prefill (P=0, Q=4 -> 4x4)
    ar_prefill_vis = [1 if k <= q else 0 for q in range(4) for k in range(4)]

    # 2B. AR Decode (P=4, Q=1 -> 1x5)
    ar_decode_vis = [1, 1, 1, 1, 1]

    # 3. Block-Diff Bidir (P=4, B=4 -> 4x8)
    bidir_vis = [1] * (4 * 8)

    # 4. Causal Commit / Verify (P=4, B=4 -> 4x8)
    commit_vis = [1 if k <= 4 + q else 0 for q in range(4) for k in range(8)]

    # 5. VLM-8B Quadratic SS (P=2, k=2 -> draft_len=6, kv_len=8)
    p_vlm, k_vlm = 2, 2
    draft_len = k_vlm * (k_vlm + 1)
    kv_len_vlm = draft_len + p_vlm
    vlm_q_rope = [p_vlm + b + j for b in range(k_vlm) for j in range(k_vlm + 1)]
    vlm_q_cache = list(range(p_vlm, p_vlm + draft_len))
    vlm_kv_rope = vlm_q_rope + list(range(p_vlm))
    vlm_kv_cache = vlm_q_cache + list(range(p_vlm))
    vlm_vis: list[int] = []
    for q in range(draft_len):
        for kv in range(kv_len_vlm):
            clean_part = kv >= draft_len
            first_clean = (kv % (k_vlm + 1) == 0) and (kv < draft_len) and (q >= kv)
            same_blk_except_first = (
                (q // (k_vlm + 1) == kv // (k_vlm + 1))
                and (q < draft_len)
                and (q % (k_vlm + 1) != 0)
            )
            vlm_vis.append(1 if (clean_part or first_clean or same_blk_except_first) else 0)

    return {
        "jointTraining": {
            "L": l_seq,
            "B": b_size,
            "qLen": q_len,
            "kvLen": q_len,
            "visible": jt_vis,
            "regions": jt_regions,
            "seqIdx": jt_seq,
            "ropePositionIds": jt_rope,
            "cachePositions": jt_cache,
            "kvRopePositionIds": jt_rope,
            "kvCachePositions": jt_cache,
            "mobcMinDeltaP": min_delta_p,
            "mobcZeroTwinViolations": zero_twin_violations,
        },
        "arPrefill": {
            "P": 0,
            "Q": 4,
            "qLen": 4,
            "kvLen": 4,
            "visible": ar_prefill_vis,
            "seqIdx": [0, 1, 2, 3],
            "ropePositionIds": [0, 1, 2, 3],
            "cachePositions": [0, 1, 2, 3],
            "kvRopePositionIds": [0, 1, 2, 3],
            "kvCachePositions": [0, 1, 2, 3],
        },
        "arDecode": {
            "P": 4,
            "Q": 1,
            "qLen": 1,
            "kvLen": 5,
            "visible": ar_decode_vis,
            "seqIdx": [0],
            "ropePositionIds": [4],
            "cachePositions": [4],
            "kvRopePositionIds": [0, 1, 2, 3, 4],
            "kvCachePositions": [0, 1, 2, 3, 4],
        },
        "blockBidir": {
            "P": 4,
            "B": 4,
            "qLen": 4,
            "kvLen": 8,
            "visible": bidir_vis,
            "seqIdx": [0, 1, 2, 3],
            "ropePositionIds": [4, 5, 6, 7],
            "cachePositions": [4, 5, 6, 7],
            "kvRopePositionIds": [0, 1, 2, 3, 4, 5, 6, 7],
            "kvCachePositions": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        "causalCommitVerify": {
            "P": 4,
            "B": 4,
            "qLen": 4,
            "kvLen": 8,
            "visible": commit_vis,
            "seqIdx": [0, 1, 2, 3],
            "ropePositionIds": [4, 5, 6, 7],
            "cachePositions": [4, 5, 6, 7],
            "kvRopePositionIds": [0, 1, 2, 3, 4, 5, 6, 7],
            "kvCachePositions": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        "vlmQuadraticSs": {
            "P": p_vlm,
            "k": k_vlm,
            "qLen": draft_len,
            "kvLen": kv_len_vlm,
            "visible": vlm_vis,
            "seqIdx": list(range(draft_len)),
            "ropePositionIds": vlm_q_rope,
            "cachePositions": vlm_q_cache,
            "kvRopePositionIds": vlm_kv_rope,
            "kvCachePositions": vlm_kv_cache,
        },
    }


# ============================================================================
# FIXTURE BUILDER (CATEGORIES 1-6)
# ============================================================================

def build_blocks_parity_fixture(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    gen: torch.Generator,
) -> dict[str, Any]:
    cfg = manifest["config"]
    d = int(cfg["hiddenSize"])
    n_q = int(cfg["numAttentionHeads"])
    n_kv = int(cfg["numKeyValueHeads"])
    d_head = int(cfg["headDim"])
    orig_max_pos = int(cfg["rope"]["originalMaxPositionEmbeddings"])
    llama4_beta = float(cfg["rope"]["llama4Beta"])

    # --- Category 1 ---
    cat1 = {
        "tensorCount": manifest["tensorCount"],
        "backboneTensorCount": manifest["backboneTensorCount"],
        "loraTensorCount": manifest["loraTensorCount"],
        "totalFloats": manifest["totalFloats"],
        "backboneFloats": manifest["backboneFloats"],
        "loraFloats": manifest["loraFloats"],
        "weightsSha256": manifest["weightsSha256"],
        "backboneSha256": manifest["backboneSha256"],
        "loraSha256": manifest["loraSha256"],
        "backboneFingerprint": manifest["backboneFingerprint"],
        "loraFingerprint": manifest["loraFingerprint"],
        "combinedFingerprint": manifest["combinedFingerprint"],
        "tensors": manifest["tensors"],
        "lfsInventory": build_verified_lfs_inventory(),
    }

    # --- Category 2 ---
    yarn_res = compute_yarn_inv_freq_ref(cfg["rope"])
    inv_freq = yarn_res["invFreq"]
    attn_scale = yarn_res["attentionScaling"]
    if inv_freq.tolist() != EXPECTED_DYADIC_INV_FREQ:
        raise RuntimeError(f"Expected exact dyadic inv_freq {EXPECTED_DYADIC_INV_FREQ}, got {inv_freq.tolist()}")

    all_pos = torch.arange(0, 32, dtype=torch.long)
    cos32, sin32 = compute_rope_cos_sin_ref(inv_freq, all_pos, attn_scale)
    llama4_all = compute_llama4_attn_scales_ref(all_pos, llama4_beta, orig_max_pos)

    sample_pos = torch.tensor([0, 5, 15, 16, 23, 31], dtype=torch.long)
    cos_s, sin_s = compute_rope_cos_sin_ref(inv_freq, sample_pos, attn_scale)
    q_in = torch.randn((6, n_q, d_head), generator=gen, dtype=torch.float32) * 0.5
    k_in = torch.randn((6, n_kv, d_head), generator=gen, dtype=torch.float32) * 0.5
    q_rot = apply_rotary_pos_emb_ref(q_in, cos_s, sin_s)
    k_rot = apply_rotary_pos_emb_ref(k_in, cos_s, sin_s)
    llama4_s = compute_llama4_attn_scales_ref(sample_pos, llama4_beta, orig_max_pos)
    q_scaled = q_rot * llama4_s.view(6, 1, 1)

    # Relative-position inner product invariance for p, k < 16
    rel_q = torch.randn((1, 1, d_head), generator=gen, dtype=torch.float32)
    rel_k = torch.randn((1, 1, d_head), generator=gen, dtype=torch.float32)
    def rotated_dot(p_pos: int, k_pos: int) -> float:
        cp, sp = cos32[p_pos : p_pos + 1], sin32[p_pos : p_pos + 1]
        ck, sk = cos32[k_pos : k_pos + 1], sin32[k_pos : k_pos + 1]
        qp = apply_rotary_pos_emb_ref(rel_q, cp, sp)
        kk = apply_rotary_pos_emb_ref(rel_k, ck, sk)
        return float((qp * kk).sum().item())

    dot_2_5 = rotated_dot(2, 5)
    dot_9_12 = rotated_dot(9, 12)
    dot_14_3 = rotated_dot(14, 3)
    dot_15_4 = rotated_dot(15, 4)
    if abs(dot_2_5 - dot_9_12) > 1e-5 or abs(dot_14_3 - dot_15_4) > 1e-5:
        raise RuntimeError("Relative-position RoPE inner-product invariance failed in PyTorch!")

    cat2 = {
        "invFreq": to_f32_list(inv_freq, "cat2.invFreq"),
        "attentionScaling": attn_scale,
        "low": yarn_res["low"],
        "high": yarn_res["high"],
        "extrapolationFactor": to_f32_list(yarn_res["extrapolationFactor"], "cat2.extrap"),
        "zones": yarn_res["zones"],
        "cos32": to_f32_list(cos32, "cat2.cos32"),
        "sin32": to_f32_list(sin32, "cat2.sin32"),
        "llama4Scales32": to_f32_list(llama4_all, "cat2.llama4Scales32"),
        "rotationTest": {
            "positionIds": to_i32_list(sample_pos),
            "qPre": to_f32_list(q_in, "cat2.qPre"),
            "kPre": to_f32_list(k_in, "cat2.kPre"),
            "qRot": to_f32_list(q_rot, "cat2.qRot"),
            "kRot": to_f32_list(k_rot, "cat2.kRot"),
            "llama4Scales": to_f32_list(llama4_s, "cat2.llama4Scales"),
            "qScaled": to_f32_list(q_scaled, "cat2.qScaled"),
        },
        "relativePositionInvariance": {
            "qVec": to_f32_list(rel_q, "cat2.relQ"),
            "kVec": to_f32_list(rel_k, "cat2.relK"),
            "pairA": {"p1": 2, "k1": 5, "p2": 9, "k2": 12, "dot1": dot_2_5, "dot2": dot_9_12},
            "pairB": {"p1": 14, "k1": 3, "p2": 15, "k2": 4, "dot1": dot_14_3, "dot2": dot_15_4},
        },
    }

    # --- Category 3 ---
    cat3 = build_mask_fixtures()

    # --- Category 4: GQA (32Q/8KV) + o_proj LoRA on Layers 0 and 20 ---
    def pack_attn(res: dict[str, Any], tag: str) -> dict[str, Any]:
        return {
            "pastLen": res["pastLen"],
            "kvLen": res["kvLen"],
            "qPre": to_f32_list(res["qPre"], f"{tag}.qPre"),
            "kPre": to_f32_list(res["kPre"], f"{tag}.kPre"),
            "vNew": to_f32_list(res["vNew"], f"{tag}.vNew"),
            "qRot": to_f32_list(res["qRot"], f"{tag}.qRot"),
            "kRot": to_f32_list(res["kRot"], f"{tag}.kRot"),
            "llama4Scales": to_f32_list(res["llama4Scales"], f"{tag}.llama4Scales"),
            "qScaled": to_f32_list(res["qScaled"], f"{tag}.qScaled"),
            "kFull": to_f32_list(res["kFull"], f"{tag}.kFull"),
            "vFull": to_f32_list(res["vFull"], f"{tag}.vFull"),
            "scores": to_nullable_scores(res["scores"], f"{tag}.scores"),
            "probs": to_f32_list(res["probs"], f"{tag}.probs"),
            "attnConcat": to_f32_list(res["attnConcat"], f"{tag}.attnConcat"),
            "oProjBase": to_f32_list(res["oProjBase"], f"{tag}.oProjBase"),
            "loraHidden": (
                to_f32_list(res["loraHidden"], f"{tag}.loraHidden")
                if res["loraHidden"] is not None
                else None
            ),
            "loraDelta": (
                to_f32_list(res["loraDelta"], f"{tag}.loraDelta")
                if res["loraDelta"] is not None
                else None
            ),
            "output": to_f32_list(res["output"], f"{tag}.output"),
        }

    x_causal_4 = torch.randn((4, d), generator=gen, dtype=torch.float32) * 0.75
    past_k_14 = torch.randn((14, n_kv * d_head), generator=gen, dtype=torch.float32) * 0.5
    past_v_14 = torch.randn((14, n_kv * d_head), generator=gen, dtype=torch.float32) * 0.5
    pos_14_17 = torch.tensor([14, 15, 16, 17], dtype=torch.long)

    x_bidir_4 = torch.randn((4, d), generator=gen, dtype=torch.float32) * 0.75
    past_k_8 = torch.randn((8, n_kv * d_head), generator=gen, dtype=torch.float32) * 0.5
    past_v_8 = torch.randn((8, n_kv * d_head), generator=gen, dtype=torch.float32) * 0.5
    pos_8_11 = torch.tensor([8, 9, 10, 11], dtype=torch.long)

    cat4_layers: dict[str, Any] = {}
    for layer_idx in (0, 20):
        causal_res = attention_ref(
            x_causal_4,
            layer_idx,
            tensor_map,
            cfg,
            inv_freq,
            attn_scale,
            pos_14_17,
            pos_14_17,
            mask_mode="causal",
            lora_enabled=False,
            past_kv=(past_k_14, past_v_14),
        )
        bidir_off = attention_ref(
            x_bidir_4,
            layer_idx,
            tensor_map,
            cfg,
            inv_freq,
            attn_scale,
            pos_8_11,
            pos_8_11,
            mask_mode="bidirectional",
            lora_enabled=False,
            past_kv=(past_k_8, past_v_8),
        )
        bidir_on = attention_ref(
            x_bidir_4,
            layer_idx,
            tensor_map,
            cfg,
            inv_freq,
            attn_scale,
            pos_8_11,
            pos_8_11,
            mask_mode="bidirectional",
            lora_enabled=True,
            past_kv=(past_k_8, past_v_8),
        )
        cat4_layers[str(layer_idx)] = {
            "layerIndex": layer_idx,
            "causalCrossBoundary": pack_attn(causal_res, f"cat4.L{layer_idx}.causal"),
            "bidirLoraOff": pack_attn(bidir_off, f"cat4.L{layer_idx}.bidirOff"),
            "bidirLoraOn": pack_attn(bidir_on, f"cat4.L{layer_idx}.bidirOn"),
        }

    cat4 = {
        "causalCrossBoundaryInputs": {
            "seqLen": 4,
            "pastLen": 14,
            "positionIds": to_i32_list(pos_14_17),
            "cachePositions": to_i32_list(pos_14_17),
            "hiddenStates": to_f32_list(x_causal_4, "cat4.xCausal"),
            "pastK": to_f32_list(past_k_14, "cat4.pastK14"),
            "pastV": to_f32_list(past_v_14, "cat4.pastV14"),
        },
        "bidirDraftInputs": {
            "seqLen": 4,
            "pastLen": 8,
            "positionIds": to_i32_list(pos_8_11),
            "cachePositions": to_i32_list(pos_8_11),
            "hiddenStates": to_f32_list(x_bidir_4, "cat4.xBidir"),
            "pastK": to_f32_list(past_k_8, "cat4.pastK8"),
            "pastV": to_f32_list(past_v_8, "cat4.pastV8"),
        },
        "layers": cat4_layers,
    }

    # --- Category 5: Layer 0 Dense SwiGLU + RMSNorm + Residual Add ---
    x_mlp_in = torch.randn((4, d), generator=gen, dtype=torch.float32) * 0.75
    eps = float(cfg["rmsNormEps"])
    normed_mlp_in = rmsnorm_ref(x_mlp_in, tensor_map["encoder.layers.0.post_attention_layernorm.weight"], eps)
    swiglu_l0 = swiglu_ref(normed_mlp_in, 0, tensor_map)
    residual_out = x_mlp_in + swiglu_l0["output"]

    cat5 = {
        "layerIndex": 0,
        "seqLen": 4,
        "rawHiddenStates": to_f32_list(x_mlp_in, "cat5.rawHidden"),
        "postAttentionNorm": to_f32_list(normed_mlp_in, "cat5.postAttnNorm"),
        "gatePre": to_f32_list(swiglu_l0["gatePre"], "cat5.gatePre"),
        "gateSilu": to_f32_list(swiglu_l0["gateSilu"], "cat5.gateSilu"),
        "upVal": to_f32_list(swiglu_l0["upVal"], "cat5.upVal"),
        "gated": to_f32_list(swiglu_l0["gated"], "cat5.gated"),
        "output": to_f32_list(swiglu_l0["output"], "cat5.output"),
        "residualOut": to_f32_list(residual_out, "cat5.residualOut"),
    }

    # --- Category 6: Full 40-Layer Forward Pass ---
    def pack_forward(fwd: dict[str, Any], tag: str) -> dict[str, Any]:
        ckpt_layers = [0, 1, 20, 40]
        return {
            "tokenIds": fwd["tokenIds"],
            "seqLen": fwd["seqLen"],
            "pastLen": fwd["pastLen"],
            "kvLen": fwd["kvLen"],
            "maskMode": fwd["maskMode"],
            "loraEnabled": fwd["loraEnabled"],
            "useCache": fwd["useCache"],
            "positionIds": fwd["positionIds"],
            "cachePositions": fwd["cachePositions"],
            "embeddings": to_f32_list(fwd["embeddings"], f"{tag}.embeddings"),
            "residualCheckpoints": {
                str(idx): to_f32_list(fwd["residualCheckpoints"][idx], f"{tag}.resid{idx}")
                for idx in ckpt_layers
            },
            "finalNorm": to_f32_list(fwd["finalNorm"], f"{tag}.finalNorm"),
            "logits": to_f32_list(fwd["logits"], f"{tag}.logits"),
            "probs": to_f32_list(fwd["probs"], f"{tag}.probs"),
            "top5ByPosition": fwd["top5ByPosition"],
        }

    prefill_tokens = [34, 0, 1, 2, 3, 4, 5, 6]
    fwd_prefill = forward_40_layers_ref(
        prefill_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )

    draft_tokens = [16, 32, 32, 32]
    fwd_draft_off = forward_40_layers_ref(
        draft_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="bidirectional",
        lora_enabled=False,
        use_cache=False,
        past_kvs=fwd_prefill["newKvs"],
    )
    fwd_draft_on = forward_40_layers_ref(
        draft_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="bidirectional",
        lora_enabled=True,
        use_cache=False,
        past_kvs=fwd_prefill["newKvs"],
    )

    prefill_14_tokens = [34, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    fwd_prefill_14 = forward_40_layers_ref(
        prefill_14_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    verify_14_17_tokens = [13, 14, 15, 16]
    fwd_verify_boundary = forward_40_layers_ref(
        verify_14_17_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=fwd_prefill_14["newKvs"],
    )

    cat6 = {
        "causalPrefill8": pack_forward(fwd_prefill, "cat6.prefill8"),
        "bidirDraftLoraOff": pack_forward(fwd_draft_off, "cat6.draftOff"),
        "bidirDraftLoraOn": pack_forward(fwd_draft_on, "cat6.draftOn"),
        "causalPrefill14ForBoundary": {
            "tokenIds": prefill_14_tokens,
            "seqLen": 14,
        },
        "causalVerifyCrossLlama4": pack_forward(fwd_verify_boundary, "cat6.verifyBoundary"),
    }

    return {
        "metadata": {
            "generator": "scripts/reference/generate_diffusion_pytorch_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "referenceArchitecture": "Nemotron-Labs-Diffusion-14B Miniature (40L, 32Q/8KV, Dyadic YaRN, SwiGLU, o_proj LoRA)",
            "tolerances": {
                "invFreqUlp": 0,
                "ropeCosSinLlama4MaxAbs": 1e-6,
                "ropeRotatedQkMaxAbs": 1e-5,
                "singleLayerGqaSwigluMaxAbs": 1e-5,
                "embeddingUlp": 0,
                "residualFinalNormProbsMaxAbs": 2e-5,
                "logitsMaxAbs": 5e-5,
            },
        },
        "cat1ParamsAndLfsInventory": cat1,
        "cat2YarnRopeAndLlama4": cat2,
        "cat3MasksAndThreeCoords": cat3,
        "cat4GqaAndLora": cat4,
        "cat5Swiglu": cat5,
        "cat6Full40LayerForward": cat6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate independent PyTorch 2.15.0 reference parity fixtures for Nemotron-Labs-Diffusion (Categories 1-6)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PARAMS_DIR / "diffusion_mini_manifest.json",
        help="Path to exported Diffusion manifest JSON.",
    )
    args = parser.parse_args()

    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Expected pinned PyTorch {PINNED_TORCH_VERSION}, got {torch.__version__}. "
            "Run via `uv run scripts/reference/generate_diffusion_pytorch_fixtures.py`."
        )

    torch.manual_seed(20261126)
    gen = torch.Generator().manual_seed(20261126)

    manifest, tensor_map = load_exported_diffusion_weights(args.manifest)

    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    payload = build_blocks_parity_fixture(manifest, tensor_map, gen)
    out_path = fixtures_dir / "pytorch-diffusion-blocks-parity.json"
    out_path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote Diffusion blocks parity fixture ({out_path.stat().st_size:,} bytes) -> {out_path}"
    )


if __name__ == "__main__":
    main()
