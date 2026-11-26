#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator
for Nemotron-Labs-Diffusion-Uni Core Elastic-MoT Backbone (UNI1 Slice 1C).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-uni-backbone.json`:
  node scripts/reference/export_uni_params.mjs
  uv run scripts/reference/generate_uni_backbone_pytorch_fixtures.py

Clean-Room Mathematical Reference & Licensing Note:
  Implements Nemotron-Labs-Diffusion-Uni's verified mathematical equations from
  first principles using standard PyTorch (`torch`, `torch.nn.functional`):
    1. Dyadic YaRN RoPE (`factor=8.0`, `truncate=True`, `inv_freq=[1, 9/32, 1/32, 1/64]`),
       half-split `rotate_half`, and Llama-4 query temperature scaling (`pos=0..35`)
    2. Additive 2D SinCos spatial embeddings (`marker_column=True` vs `False`)
    3. Single-layer Joint (`L0, L17`) vs Split-Tower (`L18, L33`) routed RMSNorm,
       8Q/2KV GQA, and 3.5x SwiGLU on a mixed `U+G` sequence
    4. Full 34L `U`-only Understanding forward (causal prefill 14 + cached decode 4)
    5. Full 34L `G`-only Generation forward (len 10, bidirectional)
    6. Full 34L T2I 2-Stage (`D5`: `hf_faithful` 34L vs `elastic_pruned` 18L -> 0-ULP Stage 2)
    7. Full 34L Mixed `U+G` Editing 2-Stage (`D5`: `hf_faithful` vs `elastic_pruned` -> 0-ULP Stage 2)
  Zero NVIDIA or Adobe Python source code or checkpoint weights are copied.
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
DEFAULT_PARAMS_DIR = Path("/tmp/nemotron-viz-uni-ref")
DEFAULT_OUT_PATH = Path("tests/fixtures/pytorch-uni-backbone.json")

EXPECTED_CORE_TENSORS = 618
EXPECTED_CORE_FLOATS = 139_024
EXPECTED_DYADIC_INV_FREQ = [1.0, 0.28125, 0.03125, 0.015625]
KEY_CHECKPOINT_LAYERS = [0, 1, 17, 18, 19, 33, 34]


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

def load_exported_uni_weights(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing exported UNI manifest: {manifest_path}. "
            "Run `node scripts/reference/export_uni_params.mjs` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = Path(manifest["weightsFile"])
    raw_bytes = weights_path.read_bytes()

    actual_weights_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_weights_sha256 != manifest["weightsSha256"]:
        raise RuntimeError(
            f"UNI weightsSha256 mismatch: {actual_weights_sha256} != {manifest['weightsSha256']}"
        )

    floats = array.array("f")
    floats.frombytes(raw_bytes)
    if sys.byteorder != "little":
        floats.byteswap()
    if floats.itemsize != 4:
        raise RuntimeError(f"Expected 4-byte float32 items, got {floats.itemsize}")

    flat_tensor = torch.tensor(floats, dtype=torch.float32)
    assert_all_finite(flat_tensor, "exported_uni_core_weights")

    tensor_map: dict[str, torch.Tensor] = {}
    total_floats = 0
    for entry in manifest["tensors"]:
        name = entry["name"]
        off = int(entry["offsetFloats"])
        length = int(entry["length"])
        shape = [int(s) for s in entry["shape"]]
        byte_slice = raw_bytes[off * 4 : (off + length) * 4]
        if hashlib.sha256(byte_slice).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Per-tensor SHA-256 mismatch for {name}")
        tensor_map[name] = flat_tensor[off : off + length].view(*shape).contiguous()
        total_floats += length

    if len(tensor_map) != EXPECTED_CORE_TENSORS or total_floats != EXPECTED_CORE_FLOATS:
        raise RuntimeError(
            f"Unexpected UNI manifest counts: tensors={len(tensor_map)}, floats={total_floats}"
        )
    return manifest, tensor_map


# ============================================================================
# CLEAN-ROOM PYTORCH 2.15.0 MATHEMATICAL OPERATORS
# ============================================================================

def get_yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_yarn_inv_freq(rope_cfg: dict[str, Any]) -> dict[str, Any]:
    dim = int(rope_cfg["headDim"])
    base = float(rope_cfg["ropeTheta"])
    orig_max_pos = int(rope_cfg["originalMaxPositionEmbeddings"])
    factor = float(rope_cfg["factor"])
    beta_fast = float(rope_cfg["betaFast"])
    beta_slow = float(rope_cfg["betaSlow"])
    mscale = float(rope_cfg["mscale"])
    mscale_all_dim = float(rope_cfg["mscaleAllDim"])
    truncate = bool(rope_cfg["truncate"])

    attention_scaling = get_yarn_mscale(factor, mscale) / get_yarn_mscale(factor, mscale_all_dim)

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(orig_max_pos / (num_rotations * 2.0 * math.pi))) / (
            2.0 * math.log(base)
        )

    low_raw = find_correction_dim(beta_fast)
    high_raw = find_correction_dim(beta_slow)
    if truncate:
        low_raw = math.floor(low_raw)
        high_raw = math.ceil(high_raw)
    half = dim // 2
    low = max(low_raw, 0)
    # Clamp to rotary plane upper index `half - 1` matching `src/uni/rope.ts`
    # (upstream HF clamps to `dim - 1`, which is identical here since high_raw == 2 <= 3).
    high = min(high_raw, half - 1)
    denom = float(high - low) if high != low else 0.001

    idx = torch.arange(0, half, dtype=torch.float32)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim))
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)

    linear_ramp = torch.clamp((idx - float(low)) / denom, 0.0, 1.0)
    extrap_factor = 1.0 - linear_ramp
    inv_freq = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor

    return {
        "inv_freq": inv_freq,
        "attention_scaling": attention_scaling,
        "low": int(low),
        "high": int(high),
        "extrap_factor": extrap_factor,
    }


def compute_rope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = position_ids.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [S, nHeads, headDim]; cos/sin: [S, headDim]
    return (x * cos[:, None, :]) + (rotate_half(x) * sin[:, None, :])


def compute_llama4_scales(
    cache_positions: torch.Tensor, beta: float, orig_max_pos: int
) -> torch.Tensor:
    return 1.0 + beta * torch.log(
        1.0 + torch.floor(cache_positions.to(torch.float32) / float(orig_max_pos))
    )


def compute_2d_sincos_pos_embed(
    height: int,
    width: int,
    hidden_size: int,
    marker_column: bool,
    canonical_content_size: float = 32.0,
    scale: float = 0.0018,
    base_theta: float = 10_000.0,
) -> dict[str, Any]:
    if hidden_size % 4 != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by 4")
    content_width = width - 1 if marker_column else width
    if content_width < 1:
        raise ValueError(f"content_width must be >= 1, got {content_width}")

    quarter = hidden_size // 4
    f_idx = torch.arange(quarter, dtype=torch.float32)
    omega = base_theta ** (-f_idx / float(quarter))

    col_stride = canonical_content_size / float(content_width)
    row_stride = canonical_content_size / float(height)

    cols = torch.arange(width, dtype=torch.float32) * col_stride
    rows = torch.arange(height, dtype=torch.float32) * row_stride

    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    flat_r = grid_r.reshape(-1)
    flat_c = grid_c.reshape(-1)

    c_angles = flat_c[:, None] * omega[None, :]
    r_angles = flat_r[:, None] * omega[None, :]

    emb = torch.cat(
        [torch.sin(c_angles), torch.cos(c_angles), torch.sin(r_angles), torch.cos(r_angles)],
        dim=-1,
    ) * float(scale)

    return {
        "embeddings": emb.to(torch.float32).contiguous(),
        "height": height,
        "width": width,
        "contentWidth": content_width,
        "hiddenSize": hidden_size,
        "markerColumn": marker_column,
        "columnStride": col_stride,
        "rowStride": row_stride,
        "frequencies": omega.to(torch.float32).contiguous(),
    }


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)
    return (normed * weight.to(torch.float32)).to(torch.float32)


def routed_rms_norm(
    x: torch.Tensor,
    routing: torch.Tensor,
    w_u: torch.Tensor,
    w_g: torch.Tensor,
    eps: float,
    skip_u: bool = False,
) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float32)
    u_mask = routing == 0
    g_mask = routing == 1
    if not skip_u and bool(u_mask.any().item()):
        out[u_mask] = rms_norm(x[u_mask], w_u, eps)
    if bool(g_mask.any().item()):
        out[g_mask] = rms_norm(x[g_mask], w_g, eps)
    return out


def routed_linear(
    x: torch.Tensor,
    routing: torch.Tensor,
    w_u: torch.Tensor,
    w_g: torch.Tensor,
    skip_u: bool = False,
) -> torch.Tensor:
    out_dim = w_u.shape[0]
    out = torch.zeros((x.shape[0], out_dim), dtype=torch.float32)
    u_mask = routing == 0
    g_mask = routing == 1
    if not skip_u and bool(u_mask.any().item()):
        out[u_mask] = F.linear(x[u_mask], w_u)
    if bool(g_mask.any().item()):
        out[g_mask] = F.linear(x[g_mask], w_g)
    return out


def build_base_mask(
    mask_mode: str,
    seq_len: int,
    past_len: int,
) -> torch.Tensor:
    kv_len = past_len + seq_len
    if mask_mode == "causal":
        q_idx = torch.arange(seq_len, dtype=torch.int64)[:, None] + past_len
        k_idx = torch.arange(kv_len, dtype=torch.int64)[None, :]
        return k_idx <= q_idx
    if mask_mode == "bidirectional":
        return torch.ones((seq_len, kv_len), dtype=torch.bool)
    raise ValueError(f"Unsupported mask_mode: {mask_mode}")


def layer_name(layer_idx: int, sub: str, branch: str) -> str:
    suffix = "_gen" if branch == "g" else ""
    if sub in ("input_layernorm", "post_attention_layernorm"):
        return f"encoder.layers.{layer_idx}.{sub}{suffix}.weight"
    if sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"encoder.layers.{layer_idx}.self_attn.{sub}{suffix}.weight"
    if sub in ("gate_proj", "up_proj", "down_proj"):
        return f"encoder.layers.{layer_idx}.mlp.{sub}{suffix}.weight"
    raise ValueError(f"Unknown layer sub-tensor: {sub}")


def run_elastic_layer(
    hidden_states: torch.Tensor,
    routing: torch.Tensor,
    layer_idx: int,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    position_ids: torch.Tensor,
    cache_positions: torch.Tensor,
    mask_mode: str,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    past_routing: torch.Tensor | None = None,
    prune_u: bool = False,
) -> dict[str, Any]:
    seq_len = hidden_states.shape[0]
    d = int(cfg["hiddenSize"])
    n_q = int(cfg["numAttentionHeads"])
    n_kv = int(cfg["numKeyValueHeads"])
    kv_group = int(cfg["kvGroupSize"])
    head_dim = int(cfg["headDim"])
    eps = float(cfg["rmsNormEps"])
    dual_tower_layers = int(cfg["dualTowerLayers"])
    split_tower = layer_idx >= dual_tower_layers
    understanding_pruned = split_tower and prune_u

    # 1. Pre-attention routed RMSNorm
    input_norm = routed_rms_norm(
        hidden_states,
        routing,
        weights[layer_name(layer_idx, "input_layernorm", "u")],
        weights[layer_name(layer_idx, "input_layernorm", "g")],
        eps,
        skip_u=understanding_pruned,
    )

    # 2. Routed Q/K/V projections
    q_pre = routed_linear(
        input_norm,
        routing,
        weights[layer_name(layer_idx, "q_proj", "u")],
        weights[layer_name(layer_idx, "q_proj", "g")],
        skip_u=understanding_pruned,
    )
    k_pre = routed_linear(
        input_norm,
        routing,
        weights[layer_name(layer_idx, "k_proj", "u")],
        weights[layer_name(layer_idx, "k_proj", "g")],
        skip_u=understanding_pruned,
    )
    v_new = routed_linear(
        input_norm,
        routing,
        weights[layer_name(layer_idx, "v_proj", "u")],
        weights[layer_name(layer_idx, "v_proj", "g")],
        skip_u=understanding_pruned,
    )

    # 3. 1D YaRN RoPE + Llama-4 query temperature scale
    cos, sin = compute_rope_cos_sin(inv_freq, position_ids, attention_scaling)
    q_rot_h = apply_rotary_pos_emb(q_pre.view(seq_len, n_q, head_dim), cos, sin)
    k_rot_h = apply_rotary_pos_emb(k_pre.view(seq_len, n_kv, head_dim), cos, sin)
    if understanding_pruned:
        u_mask = routing == 0
        q_rot_h[u_mask] = 0.0
        k_rot_h[u_mask] = 0.0

    llama4_scales = compute_llama4_scales(
        cache_positions,
        float(cfg["rope"]["llama4Beta"]),
        int(cfg["rope"]["originalMaxPositionEmbeddings"]),
    )
    q_scaled_h = q_rot_h * llama4_scales[:, None, None]

    q_rot = q_rot_h.reshape(seq_len, n_q * head_dim)
    k_rot = k_rot_h.reshape(seq_len, n_kv * head_dim)
    q_scaled = q_scaled_h.reshape(seq_len, n_q * head_dim)

    # 4. Canonical routing & prefix KV handling (D5 exact contract)
    has_u = bool((routing == 0).any().item())
    has_g = bool((routing == 1).any().item())
    canonical = "mixed" if (has_u and has_g) else ("g_only" if has_g else "u_only")

    past_len = int(past_routing.shape[0]) if past_routing is not None else (
        int(past_kv[0].shape[0]) if past_kv is not None else 0
    )
    kv_len = past_len + seq_len
    cached_has_gen = bool((past_routing == 1).any().item()) if past_routing is not None else False
    drop_cached_u = split_tower and canonical == "g_only" and past_len > 0 and not cached_has_gen

    base_mask = build_base_mask(mask_mode, seq_len, past_len)
    if past_len > 0:
        r_past = past_routing if past_routing is not None else torch.zeros(past_len, dtype=torch.int64)
        r_kv = torch.cat([r_past, routing], dim=0)
    else:
        r_kv = routing

    if split_tower:
        same_tower = routing[:, None] == r_kv[None, :]
        effective_mask = base_mask & same_tower
    else:
        effective_mask = base_mask.clone()

    # Concatenate K/V ONLY on active prefix slices; when drop_cached_u is True,
    # never touch past_kv so hf_faithful and elastic_pruned are guaranteed 0-ULP identical!
    scale = 1.0 / math.sqrt(float(head_dim))
    scores = torch.full((n_q, seq_len, kv_len), float("-inf"), dtype=torch.float32)
    probs = torch.zeros((n_q, seq_len, kv_len), dtype=torch.float32)
    attn_concat_h = torch.zeros((seq_len, n_q, head_dim), dtype=torch.float32)

    if drop_cached_u or past_len == 0:
        k_active = k_rot_h  # [seq_len, n_kv, head_dim]
        v_active = v_new.view(seq_len, n_kv, head_dim)
        col_offset = past_len
        active_mask = effective_mask[:, col_offset:]  # [seq_len, seq_len]
    else:
        if past_kv is None:
            raise RuntimeError(f"Missing past_kv at layer {layer_idx} when past_len={past_len}")
        k_past_h = past_kv[0].view(past_len, n_kv, head_dim)
        v_past_h = past_kv[1].view(past_len, n_kv, head_dim)
        k_active = torch.cat([k_past_h, k_rot_h], dim=0)
        v_active = torch.cat([v_past_h, v_new.view(seq_len, n_kv, head_dim)], dim=0)
        col_offset = 0
        active_mask = effective_mask

    # Compute GQA strictly over unmasked columns (avoiding 0 * NaN/unused prefix columns)
    for hq in range(n_q):
        hkv = hq // kv_group
        q_h = q_scaled_h[:, hq, :]  # [S, head_dim]
        k_h = k_active[:, hkv, :]  # [A, head_dim]
        v_h = v_active[:, hkv, :]  # [A, head_dim]

        for qi in range(seq_len):
            if understanding_pruned and int(routing[qi].item()) == 0:
                continue
            valid_cols = torch.nonzero(active_mask[qi], as_tuple=False).reshape(-1)
            if valid_cols.numel() == 0:
                continue
            # Dot product strictly over visible keys only (0-ULP independent of masked prefix KVs)
            dots = torch.matmul(k_h[valid_cols], q_h[qi]) * scale  # [V]
            p_valid = torch.softmax(dots, dim=-1)
            scores[hq, qi, valid_cols + col_offset] = dots
            probs[hq, qi, valid_cols + col_offset] = p_valid
            attn_concat_h[qi, hq, :] = torch.matmul(p_valid, v_h[valid_cols])

    attn_concat = attn_concat_h.reshape(seq_len, n_q * head_dim)
    attn_output = routed_linear(
        attn_concat,
        routing,
        weights[layer_name(layer_idx, "o_proj", "u")],
        weights[layer_name(layer_idx, "o_proj", "g")],
        skip_u=understanding_pruned,
    )

    # 5. First residual (U preserved verbatim when pruned)
    post_attn_res = hidden_states.clone()
    if understanding_pruned:
        g_mask = routing == 1
        post_attn_res[g_mask] = hidden_states[g_mask] + attn_output[g_mask]
    else:
        post_attn_res = hidden_states + attn_output

    # 6. Pre-MLP routed RMSNorm + 3.5x SwiGLU
    post_attn_norm = routed_rms_norm(
        post_attn_res,
        routing,
        weights[layer_name(layer_idx, "post_attention_layernorm", "u")],
        weights[layer_name(layer_idx, "post_attention_layernorm", "g")],
        eps,
        skip_u=understanding_pruned,
    )

    gate_pre = routed_linear(
        post_attn_norm,
        routing,
        weights[layer_name(layer_idx, "gate_proj", "u")],
        weights[layer_name(layer_idx, "gate_proj", "g")],
        skip_u=understanding_pruned,
    )
    gate_silu = F.silu(gate_pre)
    if understanding_pruned:
        gate_silu[routing == 0] = 0.0
    up_val = routed_linear(
        post_attn_norm,
        routing,
        weights[layer_name(layer_idx, "up_proj", "u")],
        weights[layer_name(layer_idx, "up_proj", "g")],
        skip_u=understanding_pruned,
    )
    gated = gate_silu * up_val
    mlp_out = routed_linear(
        gated,
        routing,
        weights[layer_name(layer_idx, "down_proj", "u")],
        weights[layer_name(layer_idx, "down_proj", "g")],
        skip_u=understanding_pruned,
    )

    output = post_attn_res.clone()
    if understanding_pruned:
        g_mask = routing == 1
        output[g_mask] = post_attn_res[g_mask] + mlp_out[g_mask]
    else:
        output = post_attn_res + mlp_out

    return {
        "layerIndex": layer_idx,
        "splitTower": split_tower,
        "understandingPruned": understanding_pruned,
        "dropCachedUnderstanding": drop_cached_u,
        "inputNorm": input_norm,
        "qPre": q_pre,
        "kPre": k_pre,
        "vNew": v_new,
        "qRot": q_rot,
        "kRot": k_rot,
        "llama4Scales": llama4_scales,
        "qScaled": q_scaled,
        "effectiveMask": effective_mask,
        "scores": scores,
        "probs": probs,
        "attnConcat": attn_concat,
        "attnOutput": attn_output,
        "postAttentionResidual": post_attn_res,
        "postAttentionNorm": post_attn_norm,
        "swigluGatePre": gate_pre,
        "swigluGateSilu": gate_silu,
        "swigluUpVal": up_val,
        "swigluGated": gated,
        "swigluOutput": mlp_out,
        "output": output,
    }


def run_forward_pass(
    inputs_embeds: torch.Tensor,
    routing: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    mask_mode: str,
    use_cache: bool,
    prefill_mode: str = "hf_faithful",
    past_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    past_routing: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    cache_positions: torch.Tensor | None = None,
) -> dict[str, Any]:
    seq_len = inputs_embeds.shape[0]
    num_layers = int(cfg["numLayers"])
    dual_tower_layers = int(cfg["dualTowerLayers"])
    eps = float(cfg["rmsNormEps"])

    has_u = bool((routing == 0).any().item())
    has_g = bool((routing == 1).any().item())
    canonical = "mixed" if (has_u and has_g) else ("g_only" if has_g else "u_only")

    past_len = int(past_routing.shape[0]) if past_routing is not None else 0
    if position_ids is None:
        position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.int64)
    if cache_positions is None:
        cache_positions = position_ids.clone()

    skip_split = prefill_mode == "elastic_pruned" and canonical == "u_only"
    prune_u_split = prefill_mode == "elastic_pruned" and canonical == "mixed"

    checkpoints: list[torch.Tensor | None] = [inputs_embeds.clone()] + [None] * num_layers
    new_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
    traces: dict[int, dict[str, Any]] = {}
    skipped_layers: list[int] = []
    dropped_u_layers: list[int] = []

    h = checkpoints[0]
    assert h is not None
    last_executed = -1

    for l in range(num_layers):
        is_split = l >= dual_tower_layers
        if skip_split and is_split:
            skipped_layers.append(l)
            continue

        past_kv_l = past_kvs[l] if past_kvs is not None else None
        trace = run_elastic_layer(
            hidden_states=h,
            routing=routing,
            layer_idx=l,
            weights=weights,
            cfg=cfg,
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
            position_ids=position_ids,
            cache_positions=cache_positions,
            mask_mode=mask_mode,
            past_kv=past_kv_l,
            past_routing=past_routing,
            prune_u=(prune_u_split and is_split),
        )
        h = trace["output"]
        checkpoints[l + 1] = h
        new_kvs[l] = (trace["kRot"], trace["vNew"])
        traces[l] = trace
        if trace["dropCachedUnderstanding"]:
            dropped_u_layers.append(l)
        last_executed = l

    final_norm: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    text_probs: torch.Tensor | None = None
    gen_logits: torch.Tensor | None = None
    gen_probs: torch.Tensor | None = None

    if last_executed == num_layers - 1:
        final_norm = routed_rms_norm(
            h,
            routing,
            weights["encoder.norm.weight"],
            weights["encoder.norm_gen.weight"],
            eps,
            skip_u=prune_u_split,
        )
        if has_u and not prune_u_split:
            text_logits = F.linear(final_norm, weights["diffusion_head.weight"])
            text_probs = torch.softmax(text_logits, dim=-1)
        if has_g:
            gen_logits = F.linear(final_norm, weights["encoder.gen_predictor.weight"])
            gen_probs = torch.softmax(gen_logits, dim=-1)

    return {
        "seqLen": seq_len,
        "pastLen": past_len,
        "kvLen": past_len + seq_len,
        "canonicalRouting": canonical,
        "prefillMode": prefill_mode,
        "positionIds": position_ids,
        "cachePositions": cache_positions,
        "checkpoints": checkpoints,
        "newKvs": new_kvs,
        "traces": traces,
        "skippedLayers": skipped_layers,
        "droppedCachedUnderstandingLayers": dropped_u_layers,
        "finalNorm": final_norm,
        "textLogits": text_logits,
        "textProbs": text_probs,
        "genLogits": gen_logits,
        "genProbs": gen_probs,
    }


def serialize_key_checkpoints(
    checkpoints: list[torch.Tensor | None], label: str
) -> dict[str, list[float] | None]:
    out: dict[str, list[float] | None] = {}
    for k in KEY_CHECKPOINT_LAYERS:
        t = checkpoints[k]
        out[str(k)] = None if t is None else to_f32_list(t, f"{label}.ckpt[{k}]")
    return out


# ============================================================================
# MAIN FIXTURE GENERATION (7 CATEGORIES)
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch 2.15.0 reference fixtures for NLD-Uni 34L Elastic-MoT backbone"
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=DEFAULT_PARAMS_DIR,
        help="Directory containing uni_mini_core_manifest.json and uni_mini_core_weights.f32",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output JSON fixture path",
    )
    args = parser.parse_args()

    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Expected torch=={PINNED_TORCH_VERSION}, got {torch.__version__}"
        )

    torch.set_num_threads(1)
    manifest_path = args.params_dir / "uni_mini_core_manifest.json"
    manifest, weights = load_exported_uni_weights(manifest_path)
    cfg = manifest["config"]

    # ------------------------------------------------------------------------
    # CATEGORY 1: Dyadic YaRN RoPE + rotate_half + Llama-4 Scales (pos 0..35)
    # ------------------------------------------------------------------------
    yarn = compute_yarn_inv_freq(cfg["rope"])
    inv_freq = yarn["inv_freq"]
    attention_scaling = float(yarn["attention_scaling"])

    # Assert exact 0-ULP dyadic inv_freq
    expected_inv = torch.tensor(EXPECTED_DYADIC_INV_FREQ, dtype=torch.float32)
    if not torch.equal(inv_freq, expected_inv):
        raise RuntimeError(f"Dyadic YaRN inv_freq mismatch: {inv_freq.tolist()} != {EXPECTED_DYADIC_INV_FREQ}")

    pos_0_35 = torch.arange(0, 36, dtype=torch.int64)
    cos_0_35, sin_0_35 = compute_rope_cos_sin(inv_freq, pos_0_35, attention_scaling)
    llama4_0_35 = compute_llama4_scales(
        pos_0_35,
        float(cfg["rope"]["llama4Beta"]),
        int(cfg["rope"]["originalMaxPositionEmbeddings"]),
    )

    sample_pos = torch.tensor([0, 1, 15, 16, 17, 32], dtype=torch.int64)
    sample_cos, sample_sin = compute_rope_cos_sin(inv_freq, sample_pos, attention_scaling)
    sample_embeds = weights["encoder.embed_tokens.weight"][torch.tensor([1, 4, 6, 26, 29, 42])]
    sample_q_in = F.linear(sample_embeds, weights[layer_name(0, "q_proj", "u")]).view(6, 8, 8)
    sample_k_in = F.linear(sample_embeds, weights[layer_name(0, "k_proj", "u")]).view(6, 2, 8)
    sample_q_rot = apply_rotary_pos_emb(sample_q_in, sample_cos, sample_sin)
    sample_k_rot = apply_rotary_pos_emb(sample_k_in, sample_cos, sample_sin)

    cat1_rope = {
        "invFreq": to_f32_list(inv_freq, "yarn.invFreq"),
        "attentionScaling": attention_scaling,
        "low": yarn["low"],
        "high": yarn["high"],
        "extrapolationFactor": to_f32_list(yarn["extrap_factor"], "yarn.extrap"),
        "positions": to_i32_list(pos_0_35),
        "cos": to_f32_list(cos_0_35, "rope.cos"),
        "sin": to_f32_list(sin_0_35, "rope.sin"),
        "llama4Scales": to_f32_list(llama4_0_35, "rope.llama4Scales"),
        "sampleRotate": {
            "positionIds": to_i32_list(sample_pos),
            "tokenIds": [1, 4, 6, 26, 29, 42],
            "qIn": to_f32_list(sample_q_in, "sample.qIn"),
            "kIn": to_f32_list(sample_k_in, "sample.kIn"),
            "qRot": to_f32_list(sample_q_rot, "sample.qRot"),
            "kRot": to_f32_list(sample_k_rot, "sample.kRot"),
        },
    }

    # ------------------------------------------------------------------------
    # CATEGORY 2: Additive 2D SinCos (marker_column=True vs False)
    # ------------------------------------------------------------------------
    sc_marked = compute_2d_sincos_pos_embed(
        height=2,
        width=3,
        hidden_size=8,
        marker_column=True,
        canonical_content_size=float(cfg["sinCos2d"]["canonicalContentSize"]),
        scale=float(cfg["sinCos2d"]["scale"]),
        base_theta=float(cfg["sinCos2d"]["baseTheta"]),
    )
    sc_unmarked = compute_2d_sincos_pos_embed(
        height=2,
        width=2,
        hidden_size=8,
        marker_column=False,
        canonical_content_size=float(cfg["sinCos2d"]["canonicalContentSize"]),
        scale=float(cfg["sinCos2d"]["scale"]),
        base_theta=float(cfg["sinCos2d"]["baseTheta"]),
    )

    # Assert 0-ULP content-column match in Python
    emb_m_grid = sc_marked["embeddings"].view(2, 3, 8)
    emb_u_grid = sc_unmarked["embeddings"].view(2, 2, 8)
    if not torch.equal(emb_m_grid[:, :2, :], emb_u_grid):
        raise RuntimeError("2D SinCos content columns (c=0..1) did not match bit-exact (0-ULP)")

    cat2_sincos = {
        "marked2x3": {
            "height": sc_marked["height"],
            "width": sc_marked["width"],
            "contentWidth": sc_marked["contentWidth"],
            "hiddenSize": sc_marked["hiddenSize"],
            "markerColumn": sc_marked["markerColumn"],
            "columnStride": sc_marked["columnStride"],
            "rowStride": sc_marked["rowStride"],
            "frequencies": to_f32_list(sc_marked["frequencies"], "sc_marked.freq"),
            "embeddings": to_f32_list(sc_marked["embeddings"], "sc_marked.emb"),
        },
        "unmarked2x2": {
            "height": sc_unmarked["height"],
            "width": sc_unmarked["width"],
            "contentWidth": sc_unmarked["contentWidth"],
            "hiddenSize": sc_unmarked["hiddenSize"],
            "markerColumn": sc_unmarked["markerColumn"],
            "columnStride": sc_unmarked["columnStride"],
            "rowStride": sc_unmarked["rowStride"],
            "frequencies": to_f32_list(sc_unmarked["frequencies"], "sc_unmarked.freq"),
            "embeddings": to_f32_list(sc_unmarked["embeddings"], "sc_unmarked.emb"),
        },
    }

    # ------------------------------------------------------------------------
    # CATEGORY 3: Single-Layer Joint (L0, L17) vs Split-Tower (L18, L33) Mixed
    # ------------------------------------------------------------------------
    sl_routing_list = [0, 0, 1, 0, 1, 1, 0, 1]
    sl_token_ids = [1, 4, 17, 26, 259, 42, 32, 256]  # U IDs on r=0, G IDs on r=1
    sl_routing = torch.tensor(sl_routing_list, dtype=torch.int64)
    sl_pos = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int64)
    sl_embeds = torch.zeros((len(sl_routing_list), 8), dtype=torch.float32)
    for i, (r_val, tid) in enumerate(zip(sl_routing_list, sl_token_ids)):
        if r_val == 0:
            sl_embeds[i] = weights["encoder.embed_tokens.weight"][tid]
        else:
            sl_embeds[i] = weights["encoder.gen_embedding.weight"][tid]

    single_layers: dict[str, Any] = {}
    for l_idx in (0, 17, 18, 33):
        tr = run_elastic_layer(
            hidden_states=sl_embeds,
            routing=sl_routing,
            layer_idx=l_idx,
            weights=weights,
            cfg=cfg,
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
            position_ids=sl_pos,
            cache_positions=sl_pos,
            mask_mode="causal",
        )
        # Verify split-tower zero cross-modal visibility for L18 and L33
        if l_idx >= 18:
            for qi in range(len(sl_routing_list)):
                for ki in range(len(sl_routing_list)):
                    if sl_routing_list[qi] != sl_routing_list[ki] and bool(tr["effectiveMask"][qi, ki].item()):
                        raise RuntimeError(f"Split-tower leak at L{l_idx} ({qi}, {ki})")

        single_layers[str(l_idx)] = {
            "layerIndex": l_idx,
            "splitTower": tr["splitTower"],
            "inputNorm": to_f32_list(tr["inputNorm"], f"L{l_idx}.inputNorm"),
            "qPre": to_f32_list(tr["qPre"], f"L{l_idx}.qPre"),
            "kPre": to_f32_list(tr["kPre"], f"L{l_idx}.kPre"),
            "vNew": to_f32_list(tr["vNew"], f"L{l_idx}.vNew"),
            "qRot": to_f32_list(tr["qRot"], f"L{l_idx}.qRot"),
            "kRot": to_f32_list(tr["kRot"], f"L{l_idx}.kRot"),
            "llama4Scales": to_f32_list(tr["llama4Scales"], f"L{l_idx}.llama4"),
            "qScaled": to_f32_list(tr["qScaled"], f"L{l_idx}.qScaled"),
            "effectiveMask": [int(x) for x in tr["effectiveMask"].to(torch.int64).reshape(-1).tolist()],
            "scores": to_nullable_scores(tr["scores"], f"L{l_idx}.scores"),
            "probs": to_f32_list(tr["probs"], f"L{l_idx}.probs"),
            "attnConcat": to_f32_list(tr["attnConcat"], f"L{l_idx}.attnConcat"),
            "attnOutput": to_f32_list(tr["attnOutput"], f"L{l_idx}.attnOutput"),
            "postAttentionResidual": to_f32_list(tr["postAttentionResidual"], f"L{l_idx}.postAttnRes"),
            "postAttentionNorm": to_f32_list(tr["postAttentionNorm"], f"L{l_idx}.postAttnNorm"),
            "swigluGatePre": to_f32_list(tr["swigluGatePre"], f"L{l_idx}.gatePre"),
            "swigluGateSilu": to_f32_list(tr["swigluGateSilu"], f"L{l_idx}.gateSilu"),
            "swigluUpVal": to_f32_list(tr["swigluUpVal"], f"L{l_idx}.upVal"),
            "swigluGated": to_f32_list(tr["swigluGated"], f"L{l_idx}.gated"),
            "swigluOutput": to_f32_list(tr["swigluOutput"], f"L{l_idx}.swigluOut"),
            "output": to_f32_list(tr["output"], f"L{l_idx}.output"),
        }

    cat3_single_layers = {
        "seqLen": len(sl_routing_list),
        "routing": sl_routing_list,
        "tokenIds": sl_token_ids,
        "positionIds": to_i32_list(sl_pos),
        "maskMode": "causal",
        "inputsEmbeds": to_f32_list(sl_embeds, "singleLayers.inputsEmbeds"),
        "layers": single_layers,
    }

    # ------------------------------------------------------------------------
    # CATEGORY 4: Full 34L U-Only Understanding (Prefill 14 + Decode 4)
    # ------------------------------------------------------------------------
    u_prefill_ids = [1, 10, 26, 27, 28, 3, 4, 6, 14, 26, 35, 32, 11, 10]
    u_decode_ids = [42, 3, 5, 7]
    u_prefill_emb = weights["encoder.embed_tokens.weight"][torch.tensor(u_prefill_ids)]
    u_decode_emb = weights["encoder.embed_tokens.weight"][torch.tensor(u_decode_ids)]
    u_prefill_r = torch.zeros(len(u_prefill_ids), dtype=torch.int64)
    u_decode_r = torch.zeros(len(u_decode_ids), dtype=torch.int64)

    u_prefill = run_forward_pass(
        inputs_embeds=u_prefill_emb,
        routing=u_prefill_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="causal",
        use_cache=True,
        prefill_mode="hf_faithful",
    )
    u_decode = run_forward_pass(
        inputs_embeds=u_decode_emb,
        routing=u_decode_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="causal",
        use_cache=True,
        prefill_mode="hf_faithful",
        past_kvs=u_prefill["newKvs"],
        past_routing=u_prefill_r,
    )

    cat4_understanding = {
        "prefill": {
            "tokenIds": u_prefill_ids,
            "positionIds": to_i32_list(u_prefill["positionIds"]),
            "checkpoints": serialize_key_checkpoints(u_prefill["checkpoints"], "u_prefill"),
            "finalNorm": to_f32_list(u_prefill["finalNorm"], "u_prefill.finalNorm"),
            "textLogits": to_f32_list(u_prefill["textLogits"], "u_prefill.textLogits"),
            "textProbs": to_f32_list(u_prefill["textProbs"], "u_prefill.textProbs"),
        },
        "decode": {
            "tokenIds": u_decode_ids,
            "positionIds": to_i32_list(u_decode["positionIds"]),
            "checkpoints": serialize_key_checkpoints(u_decode["checkpoints"], "u_decode"),
            "finalNorm": to_f32_list(u_decode["finalNorm"], "u_decode.finalNorm"),
            "textLogits": to_f32_list(u_decode["textLogits"], "u_decode.textLogits"),
            "textProbs": to_f32_list(u_decode["textProbs"], "u_decode.textProbs"),
        },
    }

    # ------------------------------------------------------------------------
    # CATEGORY 5: Full 34L G-Only Generation Forward (len 10, bidirectional)
    # ------------------------------------------------------------------------
    g_only_ids = [259, 260, 261, 262, 17, 42, 256, 128, 257, 258]
    g_only_emb = weights["encoder.gen_embedding.weight"][torch.tensor(g_only_ids)]
    g_only_r = torch.ones(len(g_only_ids), dtype=torch.int64)

    g_only = run_forward_pass(
        inputs_embeds=g_only_emb,
        routing=g_only_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        prefill_mode="hf_faithful",
    )

    cat5_gen_only = {
        "genTokenIds": g_only_ids,
        "positionIds": to_i32_list(g_only["positionIds"]),
        "checkpoints": serialize_key_checkpoints(g_only["checkpoints"], "g_only"),
        "finalNorm": to_f32_list(g_only["finalNorm"], "g_only.finalNorm"),
        "genLogits": to_f32_list(g_only["genLogits"], "g_only.genLogits"),
        "genProbs": to_f32_list(g_only["genProbs"], "g_only.genProbs"),
    }

    # ------------------------------------------------------------------------
    # CATEGORY 6: Full 34L T2I 2-Stage (D5: hf_faithful vs elastic_pruned)
    # ------------------------------------------------------------------------
    t2i_prompt_ids = [1, 10, 29, 3, 4, 6, 15, 36, 9, 11]  # len 10
    t2i_gen_ids = [259, 260, 261, 262, 12, 85, 256, 144, 211, 258]  # len 10, pos 10..19
    t2i_u_emb = weights["encoder.embed_tokens.weight"][torch.tensor(t2i_prompt_ids)]
    t2i_u_r = torch.zeros(len(t2i_prompt_ids), dtype=torch.int64)
    t2i_g_emb = weights["encoder.gen_embedding.weight"][torch.tensor(t2i_gen_ids)]
    t2i_g_r = torch.ones(len(t2i_gen_ids), dtype=torch.int64)

    t2i_s1_hf = run_forward_pass(
        inputs_embeds=t2i_u_emb,
        routing=t2i_u_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="causal",
        use_cache=True,
        prefill_mode="hf_faithful",
    )
    t2i_s1_pruned = run_forward_pass(
        inputs_embeds=t2i_u_emb,
        routing=t2i_u_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="causal",
        use_cache=True,
        prefill_mode="elastic_pruned",
    )

    if len(t2i_s1_pruned["skippedLayers"]) != 16 or t2i_s1_pruned["skippedLayers"] != list(range(18, 34)):
        raise RuntimeError(f"Expected T2I pruned skipped layers 18..33, got {t2i_s1_pruned['skippedLayers']}")

    t2i_s2_from_hf = run_forward_pass(
        inputs_embeds=t2i_g_emb,
        routing=t2i_g_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        past_kvs=t2i_s1_hf["newKvs"],
        past_routing=t2i_u_r,
    )
    t2i_s2_from_pruned = run_forward_pass(
        inputs_embeds=t2i_g_emb,
        routing=t2i_g_r,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        past_kvs=t2i_s1_pruned["newKvs"],
        past_routing=t2i_u_r,
    )

    # Assert 0-ULP bit-exact equivalence in PyTorch between hf_faithful and elastic_pruned Stage 2!
    for l in range(35):
        c_hf = t2i_s2_from_hf["checkpoints"][l]
        c_pr = t2i_s2_from_pruned["checkpoints"][l]
        assert c_hf is not None and c_pr is not None
        if not torch.equal(c_hf, c_pr):
            raise RuntimeError(f"T2I D5 0-ULP violation at Stage 2 checkpoint {l}")
    if not torch.equal(t2i_s2_from_hf["finalNorm"], t2i_s2_from_pruned["finalNorm"]):
        raise RuntimeError("T2I D5 0-ULP violation at Stage 2 finalNorm")
    if not torch.equal(t2i_s2_from_hf["genLogits"], t2i_s2_from_pruned["genLogits"]):
        raise RuntimeError("T2I D5 0-ULP violation at Stage 2 genLogits")
    if not torch.equal(t2i_s2_from_hf["genProbs"], t2i_s2_from_pruned["genProbs"]):
        raise RuntimeError("T2I D5 0-ULP violation at Stage 2 genProbs")

    cat6_t2i = {
        "promptTokenIds": t2i_prompt_ids,
        "genTokenIds": t2i_gen_ids,
        "stage1Hf": {
            "checkpoints": serialize_key_checkpoints(t2i_s1_hf["checkpoints"], "t2i_s1_hf"),
            "finalNorm": to_f32_list(t2i_s1_hf["finalNorm"], "t2i_s1_hf.finalNorm"),
            "textLogits": to_f32_list(t2i_s1_hf["textLogits"], "t2i_s1_hf.textLogits"),
        },
        "stage1Pruned": {
            "skippedLayers": t2i_s1_pruned["skippedLayers"],
            "checkpoints": serialize_key_checkpoints(t2i_s1_pruned["checkpoints"], "t2i_s1_pruned"),
        },
        "stage2": {
            "positionIds": to_i32_list(t2i_s2_from_hf["positionIds"]),
            "droppedCachedUnderstandingLayers": t2i_s2_from_hf["droppedCachedUnderstandingLayers"],
            "l17EffectiveMask": [
                int(x) for x in t2i_s2_from_hf["traces"][17]["effectiveMask"].to(torch.int64).reshape(-1).tolist()
            ],
            "l18EffectiveMask": [
                int(x) for x in t2i_s2_from_hf["traces"][18]["effectiveMask"].to(torch.int64).reshape(-1).tolist()
            ],
            "checkpoints": serialize_key_checkpoints(t2i_s2_from_hf["checkpoints"], "t2i_s2"),
            "finalNorm": to_f32_list(t2i_s2_from_hf["finalNorm"], "t2i_s2.finalNorm"),
            "genLogits": to_f32_list(t2i_s2_from_hf["genLogits"], "t2i_s2.genLogits"),
            "genProbs": to_f32_list(t2i_s2_from_hf["genProbs"], "t2i_s2.genProbs"),
            "pytorchBitExactPrunedVsHf": True,
        },
    }

    # ------------------------------------------------------------------------
    # CATEGORY 7: Full 34L Mixed U+G Editing 2-Stage (D5)
    # ------------------------------------------------------------------------
    edit_u_ids = [1, 10, 23, 23, 30, 26, 4, 6]  # 8 U tokens
    edit_src_g_ids = [19, 73, 150, 224]          # 4 G source VQ tokens
    edit_s1_routing_list = [0] * 8 + [1] * 4
    edit_s1_routing = torch.tensor(edit_s1_routing_list, dtype=torch.int64)
    edit_s1_emb = torch.cat(
        [
            weights["encoder.embed_tokens.weight"][torch.tensor(edit_u_ids)],
            weights["encoder.gen_embedding.weight"][torch.tensor(edit_src_g_ids)],
        ],
        dim=0,
    )

    edit_target_g_ids = [259, 260, 261, 262, 31, 99, 256, 177, 203, 258]  # len 10, pos 12..21
    edit_s2_emb = weights["encoder.gen_embedding.weight"][torch.tensor(edit_target_g_ids)]
    edit_s2_routing = torch.ones(len(edit_target_g_ids), dtype=torch.int64)

    edit_s1_hf = run_forward_pass(
        inputs_embeds=edit_s1_emb,
        routing=edit_s1_routing,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="causal",
        use_cache=True,
        prefill_mode="hf_faithful",
    )
    edit_s1_pruned = run_forward_pass(
        inputs_embeds=edit_s1_emb,
        routing=edit_s1_routing,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="causal",
        use_cache=True,
        prefill_mode="elastic_pruned",
    )

    # Assert U tokens stay frozen at L17 residual in elastic_pruned while G rows match 0-ULP!
    l17_u_ref = edit_s1_pruned["checkpoints"][18][:8]
    assert l17_u_ref is not None
    for l in range(18, 34):
        c_pr = edit_s1_pruned["checkpoints"][l + 1]
        c_hf = edit_s1_hf["checkpoints"][l + 1]
        assert c_pr is not None and c_hf is not None
        if not torch.equal(c_pr[:8], l17_u_ref):
            raise RuntimeError(f"Editing pruned U residual drifted at L{l}")
        if not torch.equal(c_pr[8:], c_hf[8:]):
            raise RuntimeError(f"Editing pruned G residual mismatch vs hf_faithful at L{l}")

    edit_s2_from_hf = run_forward_pass(
        inputs_embeds=edit_s2_emb,
        routing=edit_s2_routing,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        past_kvs=edit_s1_hf["newKvs"],
        past_routing=edit_s1_routing,
    )
    edit_s2_from_pruned = run_forward_pass(
        inputs_embeds=edit_s2_emb,
        routing=edit_s2_routing,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        past_kvs=edit_s1_pruned["newKvs"],
        past_routing=edit_s1_routing,
    )

    # Assert 0-ULP bit-exact equivalence in PyTorch between hf_faithful and elastic_pruned Stage 2!
    for l in range(35):
        c_hf = edit_s2_from_hf["checkpoints"][l]
        c_pr = edit_s2_from_pruned["checkpoints"][l]
        assert c_hf is not None and c_pr is not None
        if not torch.equal(c_hf, c_pr):
            raise RuntimeError(f"Editing D5 0-ULP violation at Stage 2 checkpoint {l}")
    if not torch.equal(edit_s2_from_hf["finalNorm"], edit_s2_from_pruned["finalNorm"]):
        raise RuntimeError("Editing D5 0-ULP violation at Stage 2 finalNorm")
    if not torch.equal(edit_s2_from_hf["genLogits"], edit_s2_from_pruned["genLogits"]):
        raise RuntimeError("Editing D5 0-ULP violation at Stage 2 genLogits")
    if not torch.equal(edit_s2_from_hf["genProbs"], edit_s2_from_pruned["genProbs"]):
        raise RuntimeError("Editing D5 0-ULP violation at Stage 2 genProbs")

    # Prove L18..33 target G attends ONLY to cached G prefix (8..11) + current G (12..21), NOT cached U (0..7)!
    for l in range(18, 34):
        m = edit_s2_from_hf["traces"][l]["effectiveMask"]
        if bool(m[:, :8].any().item()):
            raise RuntimeError(f"Editing Stage 2 L{l} unexpectedly attended to cached U prefix!")
        if not bool(m[:, 8:].all().item()):
            raise RuntimeError(f"Editing Stage 2 L{l} failed to attend to cached G + current G!")

    # Perturbation causal proof: perturbing cached U at L18..33 produces 0-ULP change,
    # whereas perturbing cached G at L18..33 strictly changes Stage 2 genLogits.
    perturbed_u_kvs = []
    perturbed_g_kvs = []
    for l in range(34):
        k_l, v_l = edit_s1_hf["newKvs"][l]
        k_u = k_l.clone()
        v_u = v_l.clone()
        k_g = k_l.clone()
        v_g = v_l.clone()
        if l >= 18:
            k_u[:8] = k_u[:8] * 100.0 + 7.5
            v_u[:8] = v_u[:8] * 100.0 - 3.25
            k_g[8:] = k_g[8:] * 2.0 + 1.0
            v_g[8:] = v_g[8:] * 2.0 - 1.0
        perturbed_u_kvs.append((k_u, v_u))
        perturbed_g_kvs.append((k_g, v_g))

    edit_s2_perturb_u = run_forward_pass(
        inputs_embeds=edit_s2_emb,
        routing=edit_s2_routing,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        past_kvs=perturbed_u_kvs,
        past_routing=edit_s1_routing,
    )
    edit_s2_perturb_g = run_forward_pass(
        inputs_embeds=edit_s2_emb,
        routing=edit_s2_routing,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        mask_mode="bidirectional",
        use_cache=False,
        past_kvs=perturbed_g_kvs,
        past_routing=edit_s1_routing,
    )
    if not torch.equal(edit_s2_from_hf["genLogits"], edit_s2_perturb_u["genLogits"]):
        raise RuntimeError("Editing L18..33 cached U perturbation leaked into Stage 2!")
    g_perturb_diff = float((edit_s2_from_hf["genLogits"] - edit_s2_perturb_g["genLogits"]).abs().max().item())
    if not (g_perturb_diff > 1e-4):
        raise RuntimeError(f"Editing L18..33 cached G perturbation had negligible effect: {g_perturb_diff}")

    cat7_editing = {
        "stage1UnderstandingTokenIds": edit_u_ids,
        "stage1SourceGenTokenIds": edit_src_g_ids,
        "stage1Routing": edit_s1_routing_list,
        "stage1InputsEmbeds": to_f32_list(edit_s1_emb, "edit_s1.inputsEmbeds"),
        "stage2GenTokenIds": edit_target_g_ids,
        "stage1Hf": {
            "checkpoints": serialize_key_checkpoints(edit_s1_hf["checkpoints"], "edit_s1_hf"),
            "finalNorm": to_f32_list(edit_s1_hf["finalNorm"], "edit_s1_hf.finalNorm"),
            "textLogits": to_f32_list(edit_s1_hf["textLogits"], "edit_s1_hf.textLogits"),
            "genLogits": to_f32_list(edit_s1_hf["genLogits"], "edit_s1_hf.genLogits"),
        },
        "stage1Pruned": {
            "checkpoints": serialize_key_checkpoints(edit_s1_pruned["checkpoints"], "edit_s1_pruned"),
            "finalNorm": to_f32_list(edit_s1_pruned["finalNorm"], "edit_s1_pruned.finalNorm"),
        },
        "stage2": {
            "positionIds": to_i32_list(edit_s2_from_hf["positionIds"]),
            "droppedCachedUnderstandingLayers": edit_s2_from_hf["droppedCachedUnderstandingLayers"],
            "l17EffectiveMask": [
                int(x) for x in edit_s2_from_hf["traces"][17]["effectiveMask"].to(torch.int64).reshape(-1).tolist()
            ],
            "l18EffectiveMask": [
                int(x) for x in edit_s2_from_hf["traces"][18]["effectiveMask"].to(torch.int64).reshape(-1).tolist()
            ],
            "checkpoints": serialize_key_checkpoints(edit_s2_from_hf["checkpoints"], "edit_s2"),
            "finalNorm": to_f32_list(edit_s2_from_hf["finalNorm"], "edit_s2.finalNorm"),
            "genLogits": to_f32_list(edit_s2_from_hf["genLogits"], "edit_s2.genLogits"),
            "genProbs": to_f32_list(edit_s2_from_hf["genProbs"], "edit_s2.genProbs"),
            "pytorchBitExactPrunedVsHf": True,
            "pytorchCachedUPerturbZeroUlp": True,
            "pytorchCachedGPerturbMaxAbsDiff": g_perturb_diff,
        },
    }

    fixture = {
        "schemaVersion": "uni1-backbone-pytorch-parity-v1",
        "modelId": manifest["modelId"],
        "backboneSeed": manifest["backboneSeed"],
        "backboneFingerprint": manifest["backboneFingerprint"],
        "tensorCount": manifest["tensorCount"],
        "totalFloats": manifest["totalFloats"],
        "totalBytes": manifest["totalBytes"],
        "weightsSha256": manifest["weightsSha256"],
        "torchVersion": torch.__version__,
        "pythonVersion": platform.python_version(),
        "keyCheckpointLayers": KEY_CHECKPOINT_LAYERS,
        "rope": cat1_rope,
        "sinCos2d": cat2_sincos,
        "singleLayers": cat3_single_layers,
        "understanding34L": cat4_understanding,
        "genOnly34L": cat5_gen_only,
        "t2iTwoStageD5": cat6_t2i,
        "editingTwoStageD5": cat7_editing,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(fixture, allow_nan=False, separators=(",", ":")) + "\n"
    args.out.write_text(serialized, encoding="utf-8")
    print(
        f"Generated PyTorch {torch.__version__} UNI backbone fixture -> {args.out} "
        f"({len(serialized):,} bytes, weightsSha256={manifest['weightsSha256'][:16]}..., "
        f"editG_perturb_diff={g_perturb_diff:.6f})"
    )


if __name__ == "__main__":
    main()
