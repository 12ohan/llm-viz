#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator
for Nemotron-Labs-Diffusion-Uni Full Multimodal Parity (UNI2 Slice 2C / Gate G2).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-uni-multimodal.json`:
  node scripts/reference/export_uni_multimodal_params.mjs
  uv run scripts/reference/generate_uni_multimodal_pytorch_fixtures.py

Clean-Room Mathematical Reference (zero NVIDIA/Adobe source code or checkpoints):
  Verifies all 766 tensors (618 core + 148 multimodal = 191,235 floats = 764,940 B)
  via cryptographic SHA-256 and generates FP32 PyTorch 2.15.0 fixtures for:
    1. MiniRadioVisionTower + MiniRadioHProjector on `sunlit_cat_16x16` (4x4 -> 2x2)
       and `harbor_panorama_16x8` (2x4 via bilinear CPE -> 1x2) across BOTH D3 gain
       presets (`mini_default = 0.25` and `prod_exact = 0.01`).
    2. MiniEmu3VQ Encoder + IBQ `argmax(z @ E^T)` (0-ULP indices) + Decoder on BOTH
       `sunlit_cat_16x16` and letterboxed `harbor_panorama_16x8 -> 16x16`.
    3. PackedLinearImageCodec compress + expand (with `gen_predictor` logits) for BOTH
       target gen (`4x6 -> 2x3`, `isSourceEdit=False`) and source edit (`4x4 -> 2x2`,
       `isSourceEdit=True`) under BOTH D3 gain presets, plus `+2` row markers &
       `_source_padding_vq_mask`.
    4. Multimodal Visual Understanding Block-Diffusion (`B=4`, `shift_logits=True`,
       spliced RADIO-H embeddings into `fixOnePrompt`, BOTH `causalContext=True` and
       `causalContext=False`, greedy `T=0` AND stochastic `T=0.7` with shared explicit
       uniform variates `U23`).
    5. `<LOC_0..16>` Object Grounding (`steps=2` on `sunlit_cat_16x16` single-tag &
       `harbor_panorama_16x8` multi-tag, `pixel_box` un-letterboxing, direct ID decode
       without `U27` stripping).
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
DEFAULT_PARAMS_DIR = Path("/tmp/nemotron-viz-uni-mm-ref")
DEFAULT_OUT_PATH = Path("tests/fixtures/pytorch-uni-multimodal.json")

EXPECTED_FULL_TENSORS = 766
EXPECTED_FULL_FLOATS = 191_235
EXPECTED_FULL_BYTES = 764_940

UNI_EOS_ID = 11
UNI_UND_PAD_ID = 23
UNI_NEWLINE_ID = 24
UNI_MASK_ID = 25
UNI_LOC_START_ID = 43
UNI_LOC_END_ID = 59
UNI_BOX_OPEN_ID = 60
UNI_BOX_CLOSE_ID = 61
UNI_LOC_BEGIN_ID = 62
UNI_LOC_END_TOKEN_ID = 63

UNI_GEN_ROW_MARKER_ID = 256
UNI_GEN_IMAGE_END_ID = 258
UNI_GEN_FIXED_PREFIX_IDS = [259, 260, 261, 262]


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


# ============================================================================
# EXPORTED 766-TENSOR LOADING WITH CRYPTOGRAPHIC SHA-256 VERIFICATION
# ============================================================================

def load_exported_full_weights(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing exported full UNI manifest: {manifest_path}. "
            "Run `node scripts/reference/export_uni_multimodal_params.mjs` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = Path(manifest["weightsFile"])
    raw_bytes = weights_path.read_bytes()

    if len(raw_bytes) != EXPECTED_FULL_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_FULL_BYTES} weight bytes, got {len(raw_bytes)}"
        )

    actual_weights_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_weights_sha256 != manifest["weightsSha256"]:
        raise RuntimeError(
            f"weightsSha256 mismatch: {actual_weights_sha256} != {manifest['weightsSha256']}"
        )

    core_byte_len = int(manifest["counts"]["coreFloats"]) * 4
    actual_core_sha256 = hashlib.sha256(raw_bytes[:core_byte_len]).hexdigest()
    actual_mm_sha256 = hashlib.sha256(raw_bytes[core_byte_len:]).hexdigest()
    if actual_core_sha256 != manifest["coreSha256"]:
        raise RuntimeError(f"coreSha256 mismatch: {actual_core_sha256} != {manifest['coreSha256']}")
    if actual_mm_sha256 != manifest["multimodalSha256"]:
        raise RuntimeError(
            f"multimodalSha256 mismatch: {actual_mm_sha256} != {manifest['multimodalSha256']}"
        )

    floats = array.array("f")
    floats.frombytes(raw_bytes)
    if sys.byteorder != "little":
        floats.byteswap()
    if floats.itemsize != 4:
        raise RuntimeError(f"Expected 4-byte float32 items, got {floats.itemsize}")

    flat_tensor = torch.tensor(floats, dtype=torch.float32)
    assert_all_finite(flat_tensor, "exported_uni_full_weights")

    tensor_map: dict[str, torch.Tensor] = {}
    total_floats = 0
    for entry in manifest["tensors"]:
        name = entry["name"]
        off = int(entry["offsetFloats"])
        length = int(entry["length"])
        shape = [int(s) for s in entry["shape"]]
        byte_slice = raw_bytes[off * 4 : (off + length) * 4]
        actual_t_sha = hashlib.sha256(byte_slice).hexdigest()
        if actual_t_sha != entry["sha256"]:
            raise RuntimeError(
                f"Per-tensor SHA-256 mismatch for {name}: {actual_t_sha} != {entry['sha256']}"
            )
        tensor_map[name] = flat_tensor[off : off + length].view(*shape).contiguous()
        total_floats += length

    if len(tensor_map) != EXPECTED_FULL_TENSORS or total_floats != EXPECTED_FULL_FLOATS:
        raise RuntimeError(
            f"Unexpected full manifest counts: tensors={len(tensor_map)}, floats={total_floats}"
        )
    return manifest, tensor_map


# ============================================================================
# SHARED CORE ELASTIC-MOT 34L OPERATORS (CLEAN-ROOM PYTORCH 2.15.0)
# ============================================================================

def get_yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_yarn_inv_freq(rope_cfg: dict[str, Any]) -> tuple[torch.Tensor, float]:
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
    high = min(high_raw, half - 1)
    denom = float(high - low) if high != low else 0.001

    idx = torch.arange(0, half, dtype=torch.float32)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim))
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)

    linear_ramp = torch.clamp((idx - float(low)) / denom, 0.0, 1.0)
    extrap_factor = 1.0 - linear_ramp
    inv_freq = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor
    return inv_freq, attention_scaling


def compute_rope_cos_sin(
    inv_freq: torch.Tensor, position_ids: torch.Tensor, attention_scaling: float
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = position_ids.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
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
) -> torch.Tensor:
    content_width = width - 1 if marker_column else width
    quarter = hidden_size // 4
    f_idx = torch.arange(quarter, dtype=torch.float32)
    omega = base_theta ** (-f_idx / float(quarter))

    col_stride = canonical_content_size / float(content_width)
    row_stride = canonical_content_size / float(height)
    cols = torch.arange(width, dtype=torch.float32) * col_stride
    rows = torch.arange(height, dtype=torch.float32) * row_stride
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")

    c_angles = grid_c.reshape(-1)[:, None] * omega[None, :]
    r_angles = grid_r.reshape(-1)[:, None] * omega[None, :]
    emb = torch.cat(
        [torch.sin(c_angles), torch.cos(c_angles), torch.sin(r_angles), torch.cos(r_angles)],
        dim=-1,
    ) * float(scale)
    return emb.to(torch.float32).contiguous()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(torch.float32)


def layer_name(layer_idx: int, sub: str, branch: str) -> str:
    suffix = "_gen" if branch == "g" else ""
    if sub in ("input_layernorm", "post_attention_layernorm"):
        return f"encoder.layers.{layer_idx}.{sub}{suffix}.weight"
    if sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"encoder.layers.{layer_idx}.self_attn.{sub}{suffix}.weight"
    if sub in ("gate_proj", "up_proj", "down_proj"):
        return f"encoder.layers.{layer_idx}.mlp.{sub}{suffix}.weight"
    raise ValueError(f"Unknown layer sub-tensor: {sub}")


def run_u_only_forward_34l(
    inputs_embeds: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    mask_mode: str,
    past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Run full 34L Understanding (`U`-only) forward pass with optional prefix KV cache."""
    seq_len = inputs_embeds.shape[0]
    n_q = int(cfg["numAttentionHeads"])
    n_kv = int(cfg["numKeyValueHeads"])
    kv_group = int(cfg["kvGroupSize"])
    head_dim = int(cfg["headDim"])
    eps = float(cfg["rmsNormEps"])
    num_layers = int(cfg["numLayers"])
    past_len = int(past_kvs[0][0].shape[0]) if past_kvs is not None and len(past_kvs) > 0 else 0
    kv_len = past_len + seq_len

    position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.int64)
    cos, sin = compute_rope_cos_sin(inv_freq, position_ids, attention_scaling)
    llama4_scales = compute_llama4_scales(
        position_ids,
        float(cfg["rope"]["llama4Beta"]),
        int(cfg["rope"]["originalMaxPositionEmbeddings"]),
    )

    if mask_mode == "causal":
        q_idx = torch.arange(seq_len, dtype=torch.int64)[:, None] + past_len
        k_idx = torch.arange(kv_len, dtype=torch.int64)[None, :]
        eff_mask = k_idx <= q_idx
    elif mask_mode == "bidirectional":
        eff_mask = torch.ones((seq_len, kv_len), dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    scale = 1.0 / math.sqrt(float(head_dim))
    h = inputs_embeds.to(torch.float32).contiguous()
    new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for l in range(num_layers):
        x_norm = rms_norm(h, weights[layer_name(l, "input_layernorm", "u")], eps)
        q_pre = F.linear(x_norm, weights[layer_name(l, "q_proj", "u")])
        k_pre = F.linear(x_norm, weights[layer_name(l, "k_proj", "u")])
        v_new = F.linear(x_norm, weights[layer_name(l, "v_proj", "u")])

        q_rot = apply_rotary_pos_emb(q_pre.view(seq_len, n_q, head_dim), cos, sin)
        k_rot = apply_rotary_pos_emb(k_pre.view(seq_len, n_kv, head_dim), cos, sin)
        q_scaled = q_rot * llama4_scales[:, None, None]

        k_rot_flat = k_rot.reshape(seq_len, n_kv * head_dim)
        new_kvs.append((k_rot_flat, v_new))

        if past_len > 0 and past_kvs is not None:
            k_full = torch.cat([past_kvs[l][0].view(past_len, n_kv, head_dim), k_rot], dim=0)
            v_full = torch.cat([past_kvs[l][1].view(past_len, n_kv, head_dim), v_new.view(seq_len, n_kv, head_dim)], dim=0)
        else:
            k_full = k_rot
            v_full = v_new.view(seq_len, n_kv, head_dim)

        attn_concat_h = torch.zeros((seq_len, n_q, head_dim), dtype=torch.float32)
        for hq in range(n_q):
            hkv = hq // kv_group
            q_h = q_scaled[:, hq, :]
            k_h = k_full[:, hkv, :]
            v_h = v_full[:, hkv, :]
            for qi in range(seq_len):
                valid_cols = torch.nonzero(eff_mask[qi], as_tuple=False).reshape(-1)
                dots = torch.matmul(k_h[valid_cols], q_h[qi]) * scale
                p_valid = torch.softmax(dots, dim=-1)
                attn_concat_h[qi, hq, :] = torch.matmul(p_valid, v_h[valid_cols])

        attn_out = F.linear(attn_concat_h.reshape(seq_len, n_q * head_dim), weights[layer_name(l, "o_proj", "u")])
        h = h + attn_out

        h_norm2 = rms_norm(h, weights[layer_name(l, "post_attention_layernorm", "u")], eps)
        gate = F.silu(F.linear(h_norm2, weights[layer_name(l, "gate_proj", "u")]))
        up = F.linear(h_norm2, weights[layer_name(l, "up_proj", "u")])
        mlp_out = F.linear(gate * up, weights[layer_name(l, "down_proj", "u")])
        h = h + mlp_out

    final_norm = rms_norm(h, weights["encoder.norm.weight"], eps)
    text_logits = F.linear(final_norm, weights["diffusion_head.weight"])
    return text_logits, new_kvs


# ============================================================================
# CATEGORY 1: MINI RADIO-H VISION TOWER + RADIO-H PROJECTOR (D3, U8)
# ============================================================================

def run_mini_radio_vision_pipeline(
    rgb_chw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    clip_mean: list[float],
    clip_std: list[float],
    radio_multiply: float,
) -> dict[str, Any]:
    _, h_px, w_px = rgb_chw.shape
    patch_size = 4
    pr = h_px // patch_size
    pc = w_px // patch_size
    n_patches = pr * pc

    mean_t = torch.tensor(clip_mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(clip_std, dtype=torch.float32).view(3, 1, 1)
    clip_norm = (rgb_chw.to(torch.float32) - mean_t) / std_t

    patches_list: list[torch.Tensor] = []
    for r in range(pr):
        for c in range(pc):
            patch = clip_norm[:, r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size]
            patches_list.append(patch.reshape(-1))
    raw_patches = torch.stack(patches_list, dim=0)  # [N_p, 48]

    patch_embed = F.linear(raw_patches, weights["encoder.vision_tower.patch_generator.embedder.weight"])

    base_pos = weights["encoder.vision_tower.pos_embed"]  # [16, 16]
    if pr == 4 and pc == 4:
        interp_pos = base_pos.clone()
    else:
        pos_4d = base_pos.view(4, 4, 16).permute(2, 0, 1).unsqueeze(0)  # [1, 16, 4, 4]
        interp_4d = F.interpolate(pos_4d, size=(pr, pc), mode="bilinear", align_corners=False)
        interp_pos = interp_4d.squeeze(0).permute(1, 2, 0).reshape(n_patches, 16).contiguous()

    cls_tokens = weights["encoder.vision_tower.cls_token"]  # [4, 16]
    vit_in = torch.cat([cls_tokens, patch_embed + interp_pos], dim=0)  # [4 + N_p, 16]
    seq_len = vit_in.shape[0]
    x = vit_in

    num_heads = 4
    head_dim = 4
    scale = 1.0 / math.sqrt(float(head_dim))

    for b in range(2):
        n1 = F.layer_norm(
            x,
            (16,),
            weights[f"encoder.vision_tower.blocks.{b}.norm1.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.norm1.bias"],
            eps=1e-6,
        )
        qkv = F.linear(
            n1,
            weights[f"encoder.vision_tower.blocks.{b}.attn.qkv.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.attn.qkv.bias"],
        )
        q, k, v = qkv.chunk(3, dim=-1)
        qh = q.view(seq_len, num_heads, head_dim).transpose(0, 1)  # [4, T, 4]
        kh = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
        vh = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, vh).transpose(0, 1).reshape(seq_len, 16)

        proj = F.linear(
            attn_out,
            weights[f"encoder.vision_tower.blocks.{b}.attn.proj.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.attn.proj.bias"],
        )
        post_attn = x + proj

        n2 = F.layer_norm(
            post_attn,
            (16,),
            weights[f"encoder.vision_tower.blocks.{b}.norm2.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.norm2.bias"],
            eps=1e-6,
        )
        fc1 = F.linear(
            n2,
            weights[f"encoder.vision_tower.blocks.{b}.mlp.fc1.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.mlp.fc1.bias"],
        )
        gelu_out = F.gelu(fc1, approximate="none")
        fc2 = F.linear(
            gelu_out,
            weights[f"encoder.vision_tower.blocks.{b}.mlp.fc2.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.mlp.fc2.bias"],
        )
        x = post_attn + fc2

    patch_features = x[4:, :].contiguous()  # [N_p, 16]

    # RadioHProjector (2x2 ps_version='v2' row-major gather -> RMSNorm -> fc1 -> SquaredReLU -> fc2 -> *gain)
    tr = pr // 2
    tc = pc // 2
    pf_grid = patch_features.view(pr, pc, 16)
    gathered_rows: list[torch.Tensor] = []
    for br in range(tr):
        for bc in range(tc):
            gathered_rows.append(
                torch.cat(
                    [
                        pf_grid[2 * br, 2 * bc],
                        pf_grid[2 * br, 2 * bc + 1],
                        pf_grid[2 * br + 1, 2 * bc],
                        pf_grid[2 * br + 1, 2 * bc + 1],
                    ],
                    dim=-1,
                )
            )
    gathered = torch.stack(gathered_rows, dim=0)  # [N_tok, 64]
    normed = rms_norm(gathered, weights["encoder.mm_projector.norm.weight"], eps=1e-5)
    fc1_out = F.linear(normed, weights["encoder.mm_projector.fc1.weight"])
    sq_relu_out = F.relu(fc1_out).pow(2)
    fc2_out = F.linear(sq_relu_out, weights["encoder.mm_projector.fc2.weight"])
    visual_embeddings = (fc2_out * float(radio_multiply)).to(torch.float32)

    return {
        "patchGridRows": pr,
        "patchGridCols": pc,
        "tokenGridRows": tr,
        "tokenGridCols": tc,
        "radioMultiply": float(radio_multiply),
        "clipNormalized": to_f32_list(clip_norm, "clipNormalized"),
        "rawPatches": to_f32_list(raw_patches, "rawPatches"),
        "interpolatedPosEmbed": to_f32_list(interp_pos, "interpolatedPosEmbed"),
        "vitInputTokens": to_f32_list(vit_in, "vitInputTokens"),
        "patchFeatures": to_f32_list(patch_features, "patchFeatures"),
        "gathered2x2": to_f32_list(gathered, "gathered2x2"),
        "normed": to_f32_list(normed, "normed"),
        "fc1Out": to_f32_list(fc1_out, "fc1Out"),
        "squaredReluOut": to_f32_list(sq_relu_out, "squaredReluOut"),
        "fc2Out": to_f32_list(fc2_out, "fc2Out"),
        "visualEmbeddings": to_f32_list(visual_embeddings, "visualEmbeddings"),
        "_visualEmbeddingsTensor": visual_embeddings,
    }


# ============================================================================
# CATEGORY 2: MINI EMU3 VQ-VAE (ENCODER + IBQ + DECODER)
# ============================================================================

def emu3_gn(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return F.group_norm(
        x,
        num_groups=4,
        weight=weights[f"{prefix}.weight"],
        bias=weights[f"{prefix}.bias"],
        eps=1e-6,
    )


def emu3_conv(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    stride: int = 1,
    padding: int = 1,
) -> torch.Tensor:
    return F.conv2d(
        x,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
        stride=stride,
        padding=padding,
    )


def emu3_resblock(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    h = F.silu(emu3_gn(x, weights, f"{prefix}.norm1"))
    h = emu3_conv(h, weights, f"{prefix}.conv1", stride=1, padding=1)
    h = F.silu(emu3_gn(h, weights, f"{prefix}.norm2"))
    h = emu3_conv(h, weights, f"{prefix}.conv2", stride=1, padding=1)
    shortcut_key = f"{prefix}.nin_shortcut.weight"
    if shortcut_key in weights:
        skip = emu3_conv(x, weights, f"{prefix}.nin_shortcut", stride=1, padding=0)
    else:
        skip = x
    return skip + h


def emu3_attnblock(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    _, c, h_px, w_px = x.shape
    hw = h_px * w_px
    h = emu3_gn(x, weights, f"{prefix}.norm")
    q = emu3_conv(h, weights, f"{prefix}.q", stride=1, padding=0).reshape(c, hw)
    k = emu3_conv(h, weights, f"{prefix}.k", stride=1, padding=0).reshape(c, hw)
    v = emu3_conv(h, weights, f"{prefix}.v", stride=1, padding=0).reshape(c, hw)

    scores = torch.matmul(q.transpose(0, 1), k) / math.sqrt(float(c))  # [HW, HW]
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(1, c, h_px, w_px)
    proj = emu3_conv(out, weights, f"{prefix}.proj_out", stride=1, padding=0)
    return x + proj


def run_mini_emu3_vq_roundtrip(
    rgb_chw: torch.Tensor, weights: dict[str, torch.Tensor]
) -> dict[str, Any]:
    _, h_px, w_px = rgb_chw.shape
    norm_in = (rgb_chw.to(torch.float32) * 2.0 - 1.0).unsqueeze(0)  # [1, 3, H, W]

    # Encoder
    h = emu3_conv(norm_in, weights, "encoder.vqvae.encoder.conv_in", stride=1, padding=1)
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.down.0.block.0")
    h = emu3_conv(
        F.pad(h, (0, 1, 0, 1), mode="constant", value=0.0),
        weights,
        "encoder.vqvae.encoder.down.0.downsample.conv",
        stride=2,
        padding=0,
    )
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.down.1.block.0")
    h = emu3_conv(
        F.pad(h, (0, 1, 0, 1), mode="constant", value=0.0),
        weights,
        "encoder.vqvae.encoder.down.1.downsample.conv",
        stride=2,
        padding=0,
    )
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.mid.block_1")
    h = emu3_attnblock(h, weights, "encoder.vqvae.encoder.mid.attn_1")
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.mid.block_2")
    h = F.silu(emu3_gn(h, weights, "encoder.vqvae.encoder.norm_out"))
    h = emu3_conv(h, weights, "encoder.vqvae.encoder.conv_out", stride=1, padding=1)
    z_cont = emu3_conv(h, weights, "encoder.vqvae.quant_conv", stride=1, padding=0)  # [1, 8, gr, gc]

    _, z_ch, gr, gc = z_cont.shape
    z_flat = z_cont.squeeze(0).permute(1, 2, 0).reshape(gr * gc, z_ch).contiguous()  # [N, 8]
    codebook = weights["encoder.vqvae.quantize.embedding.weight"]  # [256, 8]

    # Exact IBQ inner-product quantization (U14): argmax(z @ E^T)
    dot_scores = torch.matmul(z_flat, codebook.transpose(0, 1))  # [N, 256]
    codes = dot_scores.argmax(dim=-1)  # [N]
    z_quant = codebook[codes].reshape(gr, gc, z_ch).permute(2, 0, 1).unsqueeze(0).contiguous()

    # Decoder
    d = emu3_conv(z_quant, weights, "encoder.vqvae.post_quant_conv", stride=1, padding=0)
    d = emu3_conv(d, weights, "encoder.vqvae.decoder.conv_in", stride=1, padding=1)
    d = emu3_resblock(d, weights, "encoder.vqvae.decoder.mid.block_1")
    d = emu3_attnblock(d, weights, "encoder.vqvae.decoder.mid.attn_1")
    d = emu3_resblock(d, weights, "encoder.vqvae.decoder.mid.block_2")
    d = emu3_resblock(d, weights, "encoder.vqvae.decoder.up.1.block.0")
    d = F.interpolate(d, scale_factor=2.0, mode="nearest")
    d = emu3_conv(d, weights, "encoder.vqvae.decoder.up.1.upsample.conv", stride=1, padding=1)
    d = emu3_resblock(d, weights, "encoder.vqvae.decoder.up.0.block.0")
    d = F.interpolate(d, scale_factor=2.0, mode="nearest")
    d = emu3_conv(d, weights, "encoder.vqvae.decoder.up.0.upsample.conv", stride=1, padding=1)
    d = F.silu(emu3_gn(d, weights, "encoder.vqvae.decoder.norm_out"))
    raw_out = emu3_conv(d, weights, "encoder.vqvae.decoder.conv_out", stride=1, padding=1).squeeze(0)
    rgb_out = torch.clamp((raw_out + 1.0) * 0.5, 0.0, 1.0)

    return {
        "inputWidth": w_px,
        "inputHeight": h_px,
        "vqGridRows": gr,
        "vqGridCols": gc,
        "normalizedInput": to_f32_list(norm_in.squeeze(0), "vq.normalizedInput"),
        "zContinuous": to_f32_list(z_cont.squeeze(0), "vq.zContinuous"),
        "dotScores": to_f32_list(dot_scores, "vq.dotScores"),
        "codes": to_i32_list(codes),
        "zQuantized": to_f32_list(z_quant.squeeze(0), "vq.zQuantized"),
        "rawOutput": to_f32_list(raw_out, "vq.rawOutput"),
        "rgbOutput": to_f32_list(rgb_out, "vq.rgbOutput"),
        "_codesList": to_i32_list(codes),
    }


# ============================================================================
# CATEGORY 3: PACKED LINEAR IMAGE CODEC + MARKERS + PADDING MASK (D3, U13)
# ============================================================================

def add_generation_row_markers(codes: list[int], rows: int = 4, cols: int = 4) -> list[int]:
    out: list[int] = []
    for r in range(rows):
        out.extend(codes[r * cols : (r + 1) * cols])
        marker = UNI_GEN_IMAGE_END_ID if r == rows - 1 else UNI_GEN_ROW_MARKER_ID
        out.extend([marker, marker])
    return out


def compute_source_padding_vq_mask(info: dict[str, Any], vq_stride: int = 4) -> list[int]:
    target = int(info["targetSize"])
    gr = target // vq_stride
    gc = target // vq_stride
    pad_top = int(info["padTop"])
    pad_left = int(info["padLeft"])
    res_h = int(info["resizedHeight"])
    res_w = int(info["resizedWidth"])

    mask: list[int] = []
    for r in range(gr):
        y0 = r * vq_stride
        y1 = (r + 1) * vq_stride
        for c in range(gc):
            x0 = c * vq_stride
            x1 = (c + 1) * vq_stride
            outside = (
                y0 < pad_top
                or y1 > pad_top + res_h
                or x0 < pad_left
                or x1 > pad_left + res_w
            )
            mask.append(1 if outside else 0)
    return mask


def run_packed_codec_case(
    token_ids: list[int],
    child_rows: int,
    child_cols: int,
    is_source_edit: bool,
    gain_preset_id: str,
    combined_scale: float,
    weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    gen_emb = weights["encoder.gen_embedding.weight"]  # [264, 8]
    idx = torch.tensor(token_ids, dtype=torch.int64)
    child_embs = gen_emb[idx]  # [child_rows * child_cols, 8]

    normed_children = rms_norm(
        child_embs, weights["encoder.packed_image_codec.down_norm.weight"], eps=1e-6
    )

    pr = child_rows // 2
    pc = child_cols // 2
    grid = normed_children.view(child_rows, child_cols, 8)
    packed_rows: list[torch.Tensor] = []
    for r in range(pr):
        for c in range(pc):
            packed_rows.append(
                torch.cat(
                    [
                        grid[2 * r, 2 * c],
                        grid[2 * r, 2 * c + 1],
                        grid[2 * r + 1, 2 * c],
                        grid[2 * r + 1, 2 * c + 1],
                    ],
                    dim=-1,
                )
            )
    packed_concat = torch.stack(packed_rows, dim=0)  # [pr * pc, 32]
    projected = F.linear(packed_concat, weights["encoder.packed_image_codec.down_projection.weight"])
    scaled = (projected * float(combined_scale)).to(torch.float32)
    sincos = compute_2d_sincos_pos_embed(pr, pc, 8, marker_column=(not is_source_edit))
    compressed_out = (scaled + sincos).to(torch.float32)

    # Expand back to children + gen_predictor logits
    normed_parents = rms_norm(
        compressed_out, weights["encoder.packed_image_codec.up_norm.weight"], eps=1e-6
    )
    up_projected = F.linear(normed_parents, weights["encoder.packed_image_codec.up_projection.weight"])
    unpacked_grid = torch.zeros((child_rows, child_cols, 8), dtype=torch.float32)
    for r in range(pr):
        for c in range(pc):
            vec = up_projected[r * pc + c]
            unpacked_grid[2 * r, 2 * c] = vec[0:8]
            unpacked_grid[2 * r, 2 * c + 1] = vec[8:16]
            unpacked_grid[2 * r + 1, 2 * c] = vec[16:24]
            unpacked_grid[2 * r + 1, 2 * c + 1] = vec[24:32]
    unpacked_children = unpacked_grid.reshape(child_rows * child_cols, 8).contiguous()
    logits = F.linear(unpacked_children, weights["encoder.gen_predictor.weight"])

    return {
        "tokenIds": token_ids,
        "childRows": child_rows,
        "childCols": child_cols,
        "parentRows": pr,
        "parentCols": pc,
        "isSourceEdit": is_source_edit,
        "markerColumn": not is_source_edit,
        "gainPresetId": gain_preset_id,
        "combinedScale": float(combined_scale),
        "compress": {
            "normedChildren": to_f32_list(normed_children, "codec.normedChildren"),
            "packedConcat": to_f32_list(packed_concat, "codec.packedConcat"),
            "projected": to_f32_list(projected, "codec.projected"),
            "scaled": to_f32_list(scaled, "codec.scaled"),
            "sinCos2d": to_f32_list(sincos, "codec.sinCos2d"),
            "output": to_f32_list(compressed_out, "codec.output"),
        },
        "expand": {
            "normedParents": to_f32_list(normed_parents, "codec.normedParents"),
            "upProjected": to_f32_list(up_projected, "codec.upProjected"),
            "unpackedChildren": to_f32_list(unpacked_children, "codec.unpackedChildren"),
            "logits": to_f32_list(logits, "codec.logits"),
        },
    }


# ============================================================================
# CATEGORY 4: MULTIMODAL BLOCK-DIFFUSION TEXT DECODER (U21, U22, U23, U30)
# ============================================================================

def fix_one_prompt(
    prompt_ids: list[int], token_rows: int, token_cols: int
) -> tuple[list[int], list[int]]:
    expanded: list[int] = []
    visual_slots: list[int] = []
    replaced = False
    for tid in prompt_ids:
        if tid == UNI_UND_PAD_ID and not replaced:
            replaced = True
            for r in range(token_rows):
                if r > 0:
                    expanded.append(UNI_NEWLINE_ID)
                for _ in range(token_cols):
                    visual_slots.append(len(expanded))
                    expanded.append(UNI_UND_PAD_ID)
        else:
            expanded.append(tid)
    if not replaced:
        raise ValueError("Prompt must contain <SPECIAL_23> placeholder")
    return expanded, visual_slots


def splice_visual_embeddings(
    expanded_ids: list[int],
    visual_slots: list[int],
    visual_embeddings: torch.Tensor,
    embed_table: torch.Tensor,
) -> torch.Tensor:
    out = embed_table[torch.tensor(expanded_ids, dtype=torch.int64)].clone()
    for i, slot_idx in enumerate(visual_slots):
        out[slot_idx] = visual_embeddings[i]
    return out


def decode_text_row(
    raw_logits: torch.Tensor,
    mask_id: int,
    suppress_mask: bool,
    temperature: float,
    uniforms: list[float] | None,
    cursor: int,
    consume_stochastic: bool,
) -> tuple[int, bool, torch.Tensor, int, float, int]:
    vocab_size = raw_logits.shape[0]
    raw_top1 = int(raw_logits.argmax(dim=-1).item())
    guard_intervened = suppress_mask and raw_top1 == mask_id

    masked = raw_logits.clone()
    if suppress_mask:
        masked[mask_id] = float("-inf")
    probs = torch.softmax(masked, dim=-1)

    if consume_stochastic and temperature > 0.0:
        assert uniforms is not None
        finite_mask = torch.isfinite(masked)
        max_logit = float(masked[finite_mask].max().item())
        best_idx = -1
        best_score = float("-inf")
        for v in range(vocab_size):
            u = float(uniforms[cursor])
            cursor += 1
            u = min(max(u, 1e-15), 1.0 - 1e-15)
            lv = float(masked[v].item())
            if not math.isfinite(lv):
                continue
            score = (lv - max_logit) / float(temperature) - math.log(-math.log(u))
            if score > best_score:
                best_score = score
                best_idx = v
        proposed = best_idx
    else:
        proposed = int(masked.argmax(dim=-1).item())

    return raw_top1, guard_intervened, probs, proposed, float(probs[proposed].item()), cursor


def run_block_diffusion_case(
    prompt_ids: list[int],
    prompt_embeds: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    block_length: int,
    max_new_tokens: int,
    steps_per_block: int,
    confidence_threshold: float,
    temperature: float,
    causal_context: bool,
    suppress_mask: bool = True,
    uniforms_pool: list[float] | None = None,
) -> dict[str, Any]:
    vocab_size = int(cfg["vocabSize"])
    embed_table = weights["encoder.embed_tokens.weight"]
    prompt_len = len(prompt_ids)

    prefill_logits, cache_kvs = run_u_only_forward_34l(
        prompt_embeds, weights, cfg, inv_freq, attention_scaling, mask_mode="causal", past_kvs=None
    )
    nfe_count = 1
    next_logits_ctx = prefill_logits[-1].clone()

    generated: list[int] = []
    step_traces: list[dict[str, Any]] = []
    stopped_by_eos = False
    eos_offset: int | None = None
    stop_reason = "max_new_tokens"
    total_guard = 0
    block_idx = 0
    cursor = 0

    while len(generated) < max_new_tokens:
        cur_seq_len = int(cache_kvs[0][0].shape[0])
        if cur_seq_len + block_length > int(cfg["kvCapacity"]):
            stop_reason = "capacity_exhausted"
            break

        block_start_pos = cur_seq_len
        block_tokens = [UNI_MASK_ID] * block_length
        trace_step_idx = 0

        if causal_context:
            before = list(block_tokens)
            raw_top1, guard_hit, probs0, prop0, conf0, cursor = decode_text_row(
                next_logits_ctx, UNI_MASK_ID, suppress_mask, temperature, uniforms_pool, cursor, True
            )
            block_tokens[0] = prop0
            shifted_logits = torch.zeros((block_length, vocab_size), dtype=torch.float32)
            shifted_logits[0] = next_logits_ctx
            shifted_probs = torch.zeros((block_length, vocab_size), dtype=torch.float32)
            shifted_probs[0] = probs0
            slot_confs = [0.0] * block_length
            slot_confs[0] = conf0
            raw_top1s = [0] * block_length
            raw_top1s[0] = raw_top1
            guard_offs = [0] if guard_hit else []
            total_guard += len(guard_offs)
            step_traces.append(
                {
                    "blockIndex": block_idx,
                    "stepIndex": trace_step_idx,
                    "blockStartPos": block_start_pos,
                    "blockTokensBefore": before,
                    "blockTokensAfter": list(block_tokens),
                    "unmaskedOffsets": [0],
                    "slotConfidences": slot_confs,
                    "rawTop1TokenIds": raw_top1s,
                    "guardIntervenedOffsets": guard_offs,
                    "shiftedLogits": to_f32_list(shifted_logits, "bd.preseed.logits"),
                    "shiftedProbs": to_f32_list(shifted_probs, "bd.preseed.probs"),
                    "usedCausalPreSeed": True,
                }
            )
            trace_step_idx += 1

        for s in range(steps_per_block):
            masked_offs = [j for j in range(block_length) if block_tokens[j] == UNI_MASK_ID]
            if not masked_offs:
                break

            before = list(block_tokens)
            blk_embeds = embed_table[torch.tensor(block_tokens, dtype=torch.int64)]
            fwd_logits, _ = run_u_only_forward_34l(
                blk_embeds,
                weights,
                cfg,
                inv_freq,
                attention_scaling,
                mask_mode="bidirectional",
                past_kvs=cache_kvs,
            )
            nfe_count += 1

            shifted_logits = torch.zeros((block_length, vocab_size), dtype=torch.float32)
            shifted_logits[0] = next_logits_ctx
            for j in range(1, block_length):
                shifted_logits[j] = fwd_logits[j - 1]

            shifted_probs = torch.zeros((block_length, vocab_size), dtype=torch.float32)
            slot_confs = [0.0] * block_length
            raw_top1s = [0] * block_length
            proposals = [0] * block_length
            guard_offs: list[int] = []

            for j in range(block_length):
                is_masked = block_tokens[j] == UNI_MASK_ID
                r_top1, g_hit, p_row, prop, conf, cursor = decode_text_row(
                    shifted_logits[j],
                    UNI_MASK_ID,
                    suppress_mask,
                    temperature,
                    uniforms_pool,
                    cursor,
                    is_masked,
                )
                shifted_probs[j] = p_row
                raw_top1s[j] = r_top1
                slot_confs[j] = conf
                proposals[j] = prop
                if is_masked and g_hit:
                    guard_offs.append(j)

            total_guard += len(guard_offs)
            quota = math.ceil(len(masked_offs) / (steps_per_block - s))
            above = [off for off in masked_offs if slot_confs[off] >= confidence_threshold]
            if len(above) >= quota:
                selected = above
            else:
                ranked = sorted(masked_offs, key=lambda off: (-slot_confs[off], off))
                selected = sorted(ranked[:quota])

            for off in selected:
                block_tokens[off] = proposals[off]

            step_traces.append(
                {
                    "blockIndex": block_idx,
                    "stepIndex": trace_step_idx,
                    "blockStartPos": block_start_pos,
                    "blockTokensBefore": before,
                    "blockTokensAfter": list(block_tokens),
                    "unmaskedOffsets": selected,
                    "slotConfidences": slot_confs,
                    "rawTop1TokenIds": raw_top1s,
                    "guardIntervenedOffsets": guard_offs,
                    "shiftedLogits": to_f32_list(shifted_logits, "bd.step.logits"),
                    "shiftedProbs": to_f32_list(shifted_probs, "bd.step.probs"),
                    "usedCausalPreSeed": False,
                }
            )
            trace_step_idx += 1

        # Causal commit
        commit_embeds = embed_table[torch.tensor(block_tokens, dtype=torch.int64)]
        commit_logits, commit_kvs = run_u_only_forward_34l(
            commit_embeds,
            weights,
            cfg,
            inv_freq,
            attention_scaling,
            mask_mode="causal",
            past_kvs=cache_kvs,
        )
        nfe_count += 1
        cache_kvs = [
            (torch.cat([cache_kvs[l][0], commit_kvs[l][0]], dim=0), torch.cat([cache_kvs[l][1], commit_kvs[l][1]], dim=0))
            for l in range(len(cache_kvs))
        ]
        next_logits_ctx = commit_logits[-1].clone()

        remaining_budget = max_new_tokens - len(generated)
        limit = min(block_length, remaining_budget)
        eos_in_block = -1
        for i in range(limit):
            if block_tokens[i] == UNI_EOS_ID:
                eos_in_block = i
                break

        if eos_in_block >= 0:
            generated.extend(block_tokens[: eos_in_block + 1])
            stopped_by_eos = True
            eos_offset = len(generated) - 1
            stop_reason = "eos"
            break

        generated.extend(block_tokens[:limit])
        if limit < block_length:
            stop_reason = "max_new_tokens"
            break
        block_idx += 1

    return {
        "promptTokenIds": prompt_ids,
        "blockLength": block_length,
        "maxNewTokens": max_new_tokens,
        "stepsPerBlock": steps_per_block,
        "confidenceThreshold": confidence_threshold,
        "temperature": temperature,
        "causalContext": causal_context,
        "suppressMaskLogit": suppress_mask,
        "explicitUniformVariates": uniforms_pool[:cursor] if (uniforms_pool is not None and temperature > 0.0) else None,
        "generatedTokenIds": generated,
        "stoppedByEos": stopped_by_eos,
        "eosOffset": eos_offset,
        "stopReason": stop_reason,
        "nfeCount": nfe_count,
        "totalGuardInterventions": total_guard,
        "steps": step_traces,
    }


# ============================================================================
# CATEGORY 5: <LOC_0..16> OBJECT GROUNDING (U21, U27, PIXEL_BOX)
# ============================================================================

def loc_bins_to_pixel_box(
    bins: list[int], width: int, height: int, loc_bins: int = 16
) -> list[float]:
    bx1, by1, bx2, by2 = bins
    side = float(max(width, height))
    scale = side / float(loc_bins)
    pad_x = (side - float(width)) / 2.0
    pad_y = (side - float(height)) / 2.0
    clamp = lambda v, lo, hi: lo if v < lo else (hi if v > hi else v)
    return [
        clamp(bx1 * scale - pad_x, 0.0, float(width)),
        clamp(by1 * scale - pad_y, 0.0, float(height)),
        clamp(bx2 * scale - pad_x, 0.0, float(width)),
        clamp(by2 * scale - pad_y, 0.0, float(height)),
    ]


def run_grounding_case(
    prompt_ids: list[int],
    prompt_embeds: torch.Tensor,
    image_width: int,
    image_height: int,
    tags: list[str],
    steps: int,
    vocab: list[str],
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> dict[str, Any]:
    vocab_size = int(cfg["vocabSize"])
    embed_table = weights["encoder.embed_tokens.weight"]
    prompt_len = len(prompt_ids)

    def resolve_tag_id(tag: str) -> int:
        norm = tag.strip().lower()
        for idx, tok in enumerate(vocab):
            if tok.lower() == norm:
                return idx
        return 0

    template: list[int] = []
    slot_template_indices: list[int] = []
    if len(tags) == 1:
        template.append(UNI_LOC_BEGIN_ID)
        for _ in range(4):
            slot_template_indices.append(len(template))
            template.append(UNI_MASK_ID)
        template.extend([UNI_LOC_END_TOKEN_ID, UNI_EOS_ID])
    else:
        for tag in tags:
            template.extend([UNI_BOX_OPEN_ID, resolve_tag_id(tag), UNI_BOX_CLOSE_ID, UNI_LOC_BEGIN_ID])
            for _ in range(4):
                slot_template_indices.append(len(template))
                template.append(UNI_MASK_ID)
            template.append(UNI_LOC_END_TOKEN_ID)
        template.append(UNI_EOS_ID)

    first_slot_tpl_idx = slot_template_indices[0]
    first_slot = prompt_len + first_slot_tpl_idx
    total_slots = len(slot_template_indices)
    slot_offsets_full = [prompt_len + i for i in slot_template_indices]
    slot_suffix_indices = [i - first_slot_tpl_idx for i in slot_template_indices]

    tpl_prefix_embeds = embed_table[torch.tensor(template[:first_slot_tpl_idx], dtype=torch.int64)]
    prefill_embeds = torch.cat([prompt_embeds, tpl_prefix_embeds], dim=0)

    prefill_logits, cache_kvs = run_u_only_forward_34l(
        prefill_embeds, weights, cfg, inv_freq, attention_scaling, mask_mode="causal", past_kvs=None
    )
    prev_logits = prefill_logits[-1].clone()

    suffix_tokens = list(template[first_slot_tpl_idx:])
    step_traces: list[dict[str, Any]] = []
    final_raw_top1 = [0] * total_slots
    final_guard_hit = [False] * total_slots

    for s in range(steps):
        remaining = [k for k in range(total_slots) if suffix_tokens[slot_suffix_indices[k]] == UNI_MASK_ID]
        if not remaining:
            break

        before = [suffix_tokens[slot_suffix_indices[k]] for k in range(total_slots)]
        suffix_embeds = embed_table[torch.tensor(suffix_tokens, dtype=torch.int64)]
        fwd_logits, _ = run_u_only_forward_34l(
            suffix_embeds,
            weights,
            cfg,
            inv_freq,
            attention_scaling,
            mask_mode="bidirectional",
            past_kvs=cache_kvs,
        )

        slot_confs = [0.0] * total_slots
        raw_top1s = [0] * total_slots
        proposals = [0] * total_slots
        guard_slots: list[int] = []

        for k in range(total_slots):
            off = slot_suffix_indices[k]
            row_logits = prev_logits if off == 0 else fwd_logits[off - 1]
            r_top1 = int(row_logits.argmax(dim=-1).item())
            raw_top1s[k] = r_top1

            masked = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
            masked[UNI_LOC_START_ID : UNI_LOC_END_ID + 1] = row_logits[UNI_LOC_START_ID : UNI_LOC_END_ID + 1]
            probs = torch.softmax(masked, dim=-1)
            prop = int(masked.argmax(dim=-1).item())
            slot_confs[k] = float(probs[prop].item())
            proposals[k] = prop

            if suffix_tokens[off] == UNI_MASK_ID and (r_top1 < UNI_LOC_START_ID or r_top1 > UNI_LOC_END_ID):
                guard_slots.append(k)

        quota = math.ceil(len(remaining) / (steps - s))
        ranked = sorted(remaining, key=lambda k: (-slot_confs[k], k))
        selected = sorted(ranked[:quota])
        for k in selected:
            suffix_tokens[slot_suffix_indices[k]] = proposals[k]
            final_raw_top1[k] = raw_top1s[k]
            final_guard_hit[k] = k in guard_slots

        after = [suffix_tokens[slot_suffix_indices[k]] for k in range(total_slots)]
        step_traces.append(
            {
                "stepIndex": s,
                "slotTokenIdsBefore": before,
                "slotTokenIdsAfter": after,
                "unmaskedSlotIndices": selected,
                "slotConfidences": slot_confs,
                "rawTop1TokenIds": raw_top1s,
                "guardIntervenedSlotIndices": guard_slots,
            }
        )

    boxes: list[dict[str, Any]] = []
    total_guard = 0
    for i, tag in enumerate(tags):
        base = i * 4
        ids = [suffix_tokens[slot_suffix_indices[base + c]] for c in range(4)]
        tok_strs = [vocab[tid] for tid in ids]
        bins = [tid - UNI_LOC_START_ID for tid in ids]
        px_box = loc_bins_to_pixel_box(bins, image_width, image_height, 16)
        raw_ids = [final_raw_top1[base + c] for c in range(4)]
        g_cnt = sum(1 for c in range(4) if final_guard_hit[base + c])
        total_guard += g_cnt
        boxes.append(
            {
                "tag": tag,
                "locTokenIds": ids,
                "locTokenStrings": tok_strs,
                "normalizedBins": bins,
                "pixelBox": px_box,
                "rawTop1Ids": raw_ids,
                "guardIntervenedCount": g_cnt,
            }
        )

    filled_tpl = list(template[:first_slot_tpl_idx]) + suffix_tokens
    strip_count = sum(1 for tid in filled_tpl if UNI_LOC_START_ID <= tid <= UNI_LOC_END_TOKEN_ID)

    return {
        "promptTokenIds": prompt_ids,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "tags": tags,
        "stepsRequested": steps,
        "assembledTemplateIds": template,
        "slotOffsetsInFullSequence": slot_offsets_full,
        "boxes": boxes,
        "stepTraces": step_traces,
        "skipSpecialTokensWouldStripCount": strip_count,
        "totalGuardInterventions": total_guard,
    }


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def scene_to_tensor(scene: dict[str, Any]) -> torch.Tensor:
    h = int(scene["height"])
    w = int(scene["width"])
    return torch.tensor(scene["data"], dtype=torch.float32).view(3, h, w).contiguous()


def main() -> None:
    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Pinned PyTorch {PINNED_TORCH_VERSION} required, found {torch.__version__}"
        )

    parser = argparse.ArgumentParser(description="Generate PyTorch 2.15.0 UNI multimodal fixtures")
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    manifest_path = args.params_dir / "uni_mini_full_manifest.json"
    manifest, weights = load_exported_full_weights(manifest_path)

    cfg = manifest["config"]
    vocab: list[str] = manifest["vocab"]
    clip_mean: list[float] = manifest["clipMean"]
    clip_std: list[float] = manifest["clipStd"]
    gain_presets = manifest["gainPresets"]

    inv_freq, attention_scaling = compute_yarn_inv_freq(cfg["rope"])
    torch.set_grad_enabled(False)

    cat_rgb = scene_to_tensor(manifest["scenes"]["sunlit_cat_16x16"])
    harbor_rgb = scene_to_tensor(manifest["scenes"]["harbor_panorama_16x8"])
    harbor_lb_entry = manifest["scenes"]["harbor_panorama_letterboxed_16x16"]
    harbor_lb_rgb = scene_to_tensor(harbor_lb_entry["image"])
    harbor_lb_info = harbor_lb_entry["info"]

    # 1. Vision + Projector (both scenes x both D3 gain presets)
    vision_cat_mini = run_mini_radio_vision_pipeline(
        cat_rgb, weights, clip_mean, clip_std, gain_presets["mini_default"]["radioProjectorMultiply"]
    )
    vision_cat_prod = run_mini_radio_vision_pipeline(
        cat_rgb, weights, clip_mean, clip_std, gain_presets["prod_exact"]["radioProjectorMultiply"]
    )
    vision_harbor_mini = run_mini_radio_vision_pipeline(
        harbor_rgb, weights, clip_mean, clip_std, gain_presets["mini_default"]["radioProjectorMultiply"]
    )
    vision_harbor_prod = run_mini_radio_vision_pipeline(
        harbor_rgb, weights, clip_mean, clip_std, gain_presets["prod_exact"]["radioProjectorMultiply"]
    )

    # 2. Mini Emu3VQ round-trip (sunlit_cat_16x16 & letterboxed harbor_panorama_16x8 -> 16x16)
    vq_cat = run_mini_emu3_vq_roundtrip(cat_rgb, weights)
    vq_harbor_lb = run_mini_emu3_vq_roundtrip(harbor_lb_rgb, weights)

    # 3. PackedLinearImageCodec (target gen 4x6->2x3 & source edit 4x4->2x2 x both D3 presets)
    cat_marked_codes = add_generation_row_markers(vq_cat["_codesList"], rows=4, cols=4)
    harbor_src_codes = vq_harbor_lb["_codesList"]
    harbor_padding_mask = compute_source_padding_vq_mask(harbor_lb_info, vq_stride=4)

    codec_target_mini = run_packed_codec_case(
        cat_marked_codes, 4, 6, False, "mini_default", gain_presets["mini_default"]["combinedGenerationScale"], weights
    )
    codec_target_prod = run_packed_codec_case(
        cat_marked_codes, 4, 6, False, "prod_exact", gain_presets["prod_exact"]["combinedGenerationScale"], weights
    )
    codec_edit_mini = run_packed_codec_case(
        harbor_src_codes, 4, 4, True, "mini_default", gain_presets["mini_default"]["combinedGenerationScale"], weights
    )
    codec_edit_prod = run_packed_codec_case(
        harbor_src_codes, 4, 4, True, "prod_exact", gain_presets["prod_exact"]["combinedGenerationScale"], weights
    )

    # 4. Multimodal Visual Understanding Block-Diffusion (B=4, shift_logits=True)
    base_und_prompt = [1, 10, 23, 26, 27, 28]  # <s> <|im_start|> <SPECIAL_23> the image shows
    cat_expanded_ids, cat_visual_slots = fix_one_prompt(base_und_prompt, 2, 2)
    cat_spliced_embeds = splice_visual_embeddings(
        cat_expanded_ids,
        cat_visual_slots,
        vision_cat_mini["_visualEmbeddingsTensor"],
        weights["encoder.embed_tokens.weight"],
    )

    gen_rng = torch.Generator(device="cpu").manual_seed(0x554E4954)
    uniform_pool = [
        float(min(max(x, 1e-15), 1.0 - 1e-15))
        for x in torch.rand(2048, generator=gen_rng, dtype=torch.float64).tolist()
    ]

    bd_causal_greedy = run_block_diffusion_case(
        cat_expanded_ids,
        cat_spliced_embeds,
        weights,
        cfg,
        inv_freq,
        attention_scaling,
        block_length=4,
        max_new_tokens=8,
        steps_per_block=4,
        confidence_threshold=0.9,
        temperature=0.0,
        causal_context=True,
    )
    bd_noncausal_greedy = run_block_diffusion_case(
        cat_expanded_ids,
        cat_spliced_embeds,
        weights,
        cfg,
        inv_freq,
        attention_scaling,
        block_length=4,
        max_new_tokens=8,
        steps_per_block=4,
        confidence_threshold=0.9,
        temperature=0.0,
        causal_context=False,
    )
    bd_causal_stochastic = run_block_diffusion_case(
        cat_expanded_ids,
        cat_spliced_embeds,
        weights,
        cfg,
        inv_freq,
        attention_scaling,
        block_length=4,
        max_new_tokens=8,
        steps_per_block=4,
        confidence_threshold=0.9,
        temperature=0.7,
        causal_context=True,
        uniforms_pool=uniform_pool,
    )

    # 5. <LOC_0..16> Object Grounding (steps=2, single-tag cat 16x16 & multi-tag harbor 16x8)
    grounding_cat_single = run_grounding_case(
        cat_expanded_ids,
        cat_spliced_embeds,
        16,
        16,
        ["cat"],
        steps=2,
        vocab=vocab,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
    )

    base_harbor_prompt = [1, 10, 23, 26, 31]  # <s> <|im_start|> <SPECIAL_23> the photo
    harbor_expanded_ids, harbor_visual_slots = fix_one_prompt(base_harbor_prompt, 1, 2)
    harbor_spliced_embeds = splice_visual_embeddings(
        harbor_expanded_ids,
        harbor_visual_slots,
        vision_harbor_mini["_visualEmbeddingsTensor"],
        weights["encoder.embed_tokens.weight"],
    )
    grounding_harbor_multi = run_grounding_case(
        harbor_expanded_ids,
        harbor_spliced_embeds,
        16,
        8,
        ["house", "water"],
        steps=2,
        vocab=vocab,
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
    )

    # Strip internal tensor keys before JSON serialization
    for v_dict in (vision_cat_mini, vision_cat_prod, vision_harbor_mini, vision_harbor_prod):
        v_dict.pop("_visualEmbeddingsTensor", None)
    vq_cat.pop("_codesList", None)
    vq_harbor_lb.pop("_codesList", None)

    fixture = {
        "schemaVersion": "uni-multimodal-parity-v1",
        "modelId": manifest["modelId"],
        "backboneSeed": manifest["backboneSeed"],
        "multimodalSeed": manifest["multimodalSeed"],
        "backboneFingerprint": manifest["backboneFingerprint"],
        "counts": manifest["counts"],
        "weightsSha256": manifest["weightsSha256"],
        "coreSha256": manifest["coreSha256"],
        "multimodalSha256": manifest["multimodalSha256"],
        "torchVersion": torch.__version__,
        "pythonVersion": platform.python_version(),
        "visionProjector": {
            "sunlitCatMiniDefault": vision_cat_mini,
            "sunlitCatProdExact": vision_cat_prod,
            "harborPanoramaMiniDefault": vision_harbor_mini,
            "harborPanoramaProdExact": vision_harbor_prod,
        },
        "vqvae": {
            "sunlitCat16x16": vq_cat,
            "harborPanoramaLetterboxed16x16": vq_harbor_lb,
        },
        "packedCodec": {
            "markedSunlitCatCodes": cat_marked_codes,
            "fixedPrefixTokenIds": UNI_GEN_FIXED_PREFIX_IDS,
            "harborSourcePaddingVqMask": harbor_padding_mask,
            "targetGenMiniDefault": codec_target_mini,
            "targetGenProdExact": codec_target_prod,
            "sourceEditMiniDefault": codec_edit_mini,
            "sourceEditProdExact": codec_edit_prod,
        },
        "multimodalBlockDiffusion": {
            "basePromptTokenIds": base_und_prompt,
            "expandedPromptTokenIds": cat_expanded_ids,
            "visualSlotIndices": cat_visual_slots,
            "causalGreedy": bd_causal_greedy,
            "nonCausalGreedy": bd_noncausal_greedy,
            "causalStochastic": bd_causal_stochastic,
        },
        "objectGrounding": {
            "sunlitCatSingleTag": {
                "basePromptTokenIds": base_und_prompt,
                "expandedPromptTokenIds": cat_expanded_ids,
                "visualSlotIndices": cat_visual_slots,
                "result": grounding_cat_single,
            },
            "harborPanoramaMultiTag": {
                "basePromptTokenIds": base_harbor_prompt,
                "expandedPromptTokenIds": harbor_expanded_ids,
                "visualSlotIndices": harbor_visual_slots,
                "result": grounding_harbor_multi,
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Generated {args.out} (torch=={torch.__version__}, "
        f"766 tensors / 191,235 floats, sha256={manifest['weightsSha256'][:16]}...)"
    )


if __name__ == "__main__":
    main()
