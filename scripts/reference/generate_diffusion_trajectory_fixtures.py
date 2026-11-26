#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Trajectory Parity Fixture Generator
for Nemotron-Labs-Diffusion (DF2 Categories 7-11).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-diffusion-trajectories-parity.json`:
  node scripts/reference/export_diffusion_params.mjs
  uv run scripts/reference/generate_diffusion_trajectory_fixtures.py

Clean-Room Mathematical Reference & Licensing Note:
  Implements Nemotron-Labs-Diffusion's verified generation equations from first principles
  using standard PyTorch (`torch`, `torch.nn.functional`):
    - Category 7: Incremental vs full-context causal AR crossing origMaxPos = 16
    - Category 8: Strictly separated Sampler A (Float64 Gumbel-max + UNSCALED softmax,
      causal_context = True | False) & Sampler B (temp-scaled multinomial + confidence)
    - Category 9: KV cache lifecycle, kvMaskProvenance, speculative crop vs pure AR KV,
      and D19 mid-accepted-prefix EOS crop provenance
    - Category 10: Greedy T = 0 linear self-speculation (±LoRA) token parity & verify invariance
    - Category 11: Stochastic T = 0.7 explicit-variate replay across AR, DLM, and Linear Spec
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
OPEN_UNIT_MIN = 1e-15
OPEN_UNIT_MAX = 1.0 - 1e-15


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
# CLEAN-ROOM PYTORCH 2.15.0 MINISTRAL-3 DIFFUSION FORWARD KERNEL
# ============================================================================

def get_yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_yarn_inv_freq_ref(rope_cfg: dict[str, Any]) -> tuple[torch.Tensor, float]:
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
    if inv_freq.tolist() != EXPECTED_DYADIC_INV_FREQ:
        raise RuntimeError(f"Expected dyadic inv_freq {EXPECTED_DYADIC_INV_FREQ}, got {inv_freq.tolist()}")
    return inv_freq, float(attention_scaling)


def compute_rope_cos_sin_ref(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.outer(position_ids.to(torch.float32), inv_freq.to(torch.float32))
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb) * float(attention_scaling)
    sin = torch.sin(emb) * float(attention_scaling)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))


def compute_llama4_attn_scales_ref(
    cache_positions: torch.Tensor,
    beta: float,
    orig_max_pos: int,
) -> torch.Tensor:
    pos_f = cache_positions.to(torch.float32)
    return 1.0 + float(beta) * torch.log(1.0 + torch.floor(pos_f / float(orig_max_pos)))


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_f32 * torch.rsqrt(variance + float(eps))).to(x.dtype)
    return x_norm * weight


def attention_ref(
    hidden_states: torch.Tensor,
    layer_idx: int,
    tensor_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    position_ids: torch.Tensor,
    cache_positions: torch.Tensor,
    mask_mode: str,
    lora_enabled: bool,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    q_pre = F.linear(hidden_states, wq)
    k_pre = F.linear(hidden_states, wk)
    v_new = F.linear(hidden_states, wv)

    cos, sin = compute_rope_cos_sin_ref(inv_freq, position_ids, attention_scaling)
    q_rot_h = apply_rotary_pos_emb_ref(q_pre.view(s_len, n_q, d_head), cos, sin)
    k_rot_h = apply_rotary_pos_emb_ref(k_pre.view(s_len, n_kv, d_head), cos, sin)

    llama4_scales = compute_llama4_attn_scales_ref(
        cache_positions,
        float(cfg["rope"]["llama4Beta"]),
        int(cfg["rope"]["originalMaxPositionEmbeddings"]),
    )
    q_scaled_h = q_rot_h * llama4_scales.view(s_len, 1, 1)
    k_rot = k_rot_h.reshape(s_len, kv_dim)

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
    q_h = q_scaled_h.transpose(0, 1).contiguous()
    k_kv_h = k_full.view(kv_len, n_kv, d_head).transpose(0, 1).contiguous()
    v_kv_h = v_full.view(kv_len, n_kv, d_head).transpose(0, 1).contiguous()
    k_rep = k_kv_h.repeat_interleave(n_rep, dim=0)
    v_rep = v_kv_h.repeat_interleave(n_rep, dim=0)

    scale = 1.0 / math.sqrt(d_head)
    raw_scores = torch.matmul(q_h, k_rep.transpose(-1, -2)) * scale

    if mask_mode == "causal":
        q_idx = torch.arange(past_len, past_len + s_len, dtype=torch.long).unsqueeze(1)
        k_idx = torch.arange(0, kv_len, dtype=torch.long).unsqueeze(0)
        visible_mask = k_idx <= q_idx
        scores = raw_scores.masked_fill(~visible_mask.unsqueeze(0), float("-inf"))
    elif mask_mode == "bidirectional":
        scores = raw_scores
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode!r}")

    probs = F.softmax(scores.to(torch.float32), dim=-1)
    v_out = torch.matmul(probs, v_rep)
    attn_concat = v_out.transpose(0, 1).contiguous().view(s_len, q_dim)
    o_proj_base = F.linear(attn_concat, wo)

    if lora_enabled:
        lora_a = tensor_map[f"base_model.model.encoder.layers.{layer_idx}.self_attn.o_proj.lora_A.weight"]
        lora_b = tensor_map[f"base_model.model.encoder.layers.{layer_idx}.self_attn.o_proj.lora_B.weight"]
        output = o_proj_base + float(cfg["loraScale"]) * F.linear(F.linear(attn_concat, lora_a), lora_b)
    else:
        output = o_proj_base

    return output, (k_rot, v_new)


def swiglu_ref(hidden_states: torch.Tensor, layer_idx: int, tensor_map: dict[str, torch.Tensor]) -> torch.Tensor:
    w_gate = tensor_map[f"encoder.layers.{layer_idx}.mlp.gate_proj.weight"]
    w_up = tensor_map[f"encoder.layers.{layer_idx}.mlp.up_proj.weight"]
    w_down = tensor_map[f"encoder.layers.{layer_idx}.mlp.down_proj.weight"]
    gated = F.silu(F.linear(hidden_states, w_gate)) * F.linear(hidden_states, w_up)
    return F.linear(gated, w_down)


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
    h = embed_w[tok_t].contiguous()
    new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    eps = float(cfg["rmsNormEps"])

    for layer_idx in range(int(cfg["numLayers"])):
        in_norm = rmsnorm_ref(h, tensor_map[f"encoder.layers.{layer_idx}.input_layernorm.weight"], eps)
        attn_out, new_kv = attention_ref(
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
        post_attn_res = h + attn_out
        post_norm = rmsnorm_ref(
            post_attn_res,
            tensor_map[f"encoder.layers.{layer_idx}.post_attention_layernorm.weight"],
            eps,
        )
        h = post_attn_res + swiglu_ref(post_norm, layer_idx, tensor_map)
        new_kvs.append(new_kv)

    final_norm = rmsnorm_ref(h, tensor_map["encoder.norm.weight"], eps)
    logits = F.linear(final_norm, tensor_map["diffusion_head.weight"])
    probs = F.softmax(logits.to(torch.float32), dim=-1)

    return {
        "tokenIds": token_ids,
        "seqLen": s_len,
        "pastLen": past_len,
        "kvLen": past_len + s_len,
        "maskMode": mask_mode,
        "loraEnabled": lora_enabled,
        "useCache": use_cache,
        "logits": logits,
        "probs": probs,
        "newKvs": new_kvs,
    }


def append_kvs(
    past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None,
    new_kvs: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if past_kvs is None:
        return [(k.clone(), v.clone()) for k, v in new_kvs]
    return [
        (torch.cat([pk, nk], dim=0).contiguous(), torch.cat([pv, nv], dim=0).contiguous())
        for (pk, pv), (nk, nv) in zip(past_kvs, new_kvs)
    ]


def crop_kvs(
    kvs: list[tuple[torch.Tensor, torch.Tensor]],
    target_len: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k[:target_len].contiguous(), v[:target_len].contiguous()) for k, v in kvs]


# ============================================================================
# CLEAN-ROOM SAMPLERS (STRICTLY SEPARATED: CAUSAL, SAMPLER A, SAMPLER B)
# ============================================================================

def draw_open_unit_uniforms(gen: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    u = torch.rand(shape, generator=gen, dtype=torch.float64)
    return torch.clamp(u * (1.0 - 2e-6) + 1e-6, OPEN_UNIT_MIN, OPEN_UNIT_MAX)


def sample_categorical_from_probs(probs_row: torch.Tensor, u: float) -> int:
    probs_list = probs_row.detach().to(torch.float32).tolist()
    total = sum(p for p in probs_list if p > 0.0)
    if total <= 0.0:
        return 0
    u_clamped = min(max(float(u), 0.0), OPEN_UNIT_MAX)
    target = u_clamped * total
    cdf = 0.0
    last_pos = 0
    for v, p in enumerate(probs_list):
        if p > 0.0:
            last_pos = v
            cdf += p
            if target < cdf:
                return v
    return last_pos


def sample_causal_token_ref(
    logits_row: torch.Tensor,
    temperature: float,
    uniform: float | None = None,
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    if uniform is None:
        raise ValueError("sample_causal_token_ref requires uniform when temperature > 0")
    scaled = (logits_row.to(torch.float32) / float(temperature)).to(torch.float32)
    probs = F.softmax(scaled, dim=-1)
    return sample_categorical_from_probs(probs, uniform)


def compute_static_transfer_quota_ref(num_masked: int, steps: int) -> list[int]:
    base = num_masked // steps
    rem = num_masked % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]


def run_sampler_a_step_ref(
    logits: torch.Tensor,  # [B, V]
    block_tokens: list[int],
    num_transfer_tokens: int,
    temperature: float,
    threshold: float | None,
    mask_token_id: int,
    explicit_gumbel_uniforms: list[float] | None = None,
) -> dict[str, Any]:
    seq_len, vocab_size = logits.shape
    unscaled_probs = F.softmax(logits.to(torch.float32), dim=-1)

    if temperature <= 0.0:
        candidate_tokens = [int(torch.argmax(logits[t]).item()) for t in range(seq_len)]
    else:
        if explicit_gumbel_uniforms is None or len(explicit_gumbel_uniforms) < seq_len * vocab_size:
            raise ValueError("Sampler A requires seq_len * vocab_size gumbel uniforms when temperature > 0")
        u64 = torch.tensor(explicit_gumbel_uniforms[: seq_len * vocab_size], dtype=torch.float64).view(seq_len, vocab_size)
        u64 = torch.clamp(u64, OPEN_UNIT_MIN, OPEN_UNIT_MAX)
        logits64 = logits.to(torch.float64)
        max_l = logits64.max(dim=-1, keepdim=True).values
        score = torch.exp(logits64 - max_l) / ((-torch.log(u64)) ** float(temperature))
        candidate_tokens = [int(x) for x in torch.argmax(score, dim=-1).tolist()]

    confidences: list[float] = []
    masked_indices: list[int] = []
    for t in range(seq_len):
        if block_tokens[t] == mask_token_id:
            confidences.append(float(unscaled_probs[t, candidate_tokens[t]].item()))
            masked_indices.append(t)
        else:
            confidences.append(float("-inf"))

    masked_indices.sort(key=lambda idx: (-confidences[idx], idx))
    transfer_index = [0] * seq_len
    if masked_indices:
        if threshold is None:
            quota = max(0, min(num_transfer_tokens, len(masked_indices)))
            for k in range(quota):
                transfer_index[masked_indices[k]] = 1
        else:
            transfer_index[masked_indices[0]] = 1
            for k in range(1, len(masked_indices)):
                pos = masked_indices[k]
                if confidences[pos] >= threshold:
                    transfer_index[pos] = 1
                else:
                    break

    updated_block: list[int] = []
    selected_count = 0
    unmasked_mask_count = 0
    remaining_mask_count = 0
    for t in range(seq_len):
        if transfer_index[t] == 1:
            selected_count += 1
            tok = candidate_tokens[t]
            updated_block.append(tok)
            if tok == mask_token_id:
                unmasked_mask_count += 1
        else:
            updated_block.append(block_tokens[t])
        if updated_block[-1] == mask_token_id:
            remaining_mask_count += 1

    return {
        "candidateTokens": candidate_tokens,
        "confidences": confidences,
        "transferIndex": transfer_index,
        "updatedBlock": updated_block,
        "selectedCount": selected_count,
        "remainingMaskCount": remaining_mask_count,
        "unmaskedMaskTokenCount": unmasked_mask_count,
    }


def run_sampler_b_step_ref(
    draft_logits: torch.Tensor,  # [B, V]
    block_tokens: list[int],
    temperature: float,
    threshold: float,
    mask_token_id: int,
    explicit_uniforms: list[float] | None = None,
) -> dict[str, Any]:
    seq_len, _ = draft_logits.shape
    draft_tokens: list[int] = []
    confidences: list[float] = []

    if temperature <= 0.0:
        probs = F.softmax(draft_logits.to(torch.float32), dim=-1)
        for t in range(seq_len):
            tok = int(torch.argmax(draft_logits[t]).item())
            draft_tokens.append(tok)
            confidences.append(float(probs[t, tok].item()))
    else:
        if explicit_uniforms is None or len(explicit_uniforms) < seq_len:
            raise ValueError("Sampler B requires seq_len uniforms when temperature > 0")
        scaled = (draft_logits.to(torch.float32) / float(temperature)).to(torch.float32)
        scaled_probs = F.softmax(scaled, dim=-1)
        for t in range(seq_len):
            tok = sample_categorical_from_probs(scaled_probs[t], explicit_uniforms[t])
            draft_tokens.append(tok)
            confidences.append(float(scaled_probs[t, tok].item()))

    single_pass_commit_all = threshold <= 0.0
    unmask_mask = [0] * seq_len
    masked_count = 0
    passed_count = 0
    best_masked_idx = -1
    best_masked_conf = float("-inf")

    for t in range(seq_len):
        if block_tokens[t] == mask_token_id:
            masked_count += 1
            c = confidences[t]
            if c > best_masked_conf:
                best_masked_conf = c
                best_masked_idx = t
            if single_pass_commit_all or c >= threshold:
                unmask_mask[t] = 1
                passed_count += 1

    if not single_pass_commit_all and masked_count > 0 and passed_count == 0 and best_masked_idx >= 0:
        unmask_mask[best_masked_idx] = 1

    updated_block: list[int] = []
    changed = False
    remaining_mask_count = 0
    for t in range(seq_len):
        nxt = draft_tokens[t] if unmask_mask[t] == 1 else block_tokens[t]
        updated_block.append(nxt)
        if nxt != block_tokens[t]:
            changed = True
        if nxt == mask_token_id:
            remaining_mask_count += 1

    return {
        "draftTokens": draft_tokens,
        "confidences": confidences,
        "unmaskMask": unmask_mask,
        "updatedBlock": updated_block,
        "remainingMaskCount": remaining_mask_count,
        "singlePassCommitAll": single_pass_commit_all,
        "noProgress": not changed,
    }


# ============================================================================
# CLEAN-ROOM FULL TRAJECTORY RUNNERS (AR, DLM, LINEAR_SPEC ± LORA)
# ============================================================================

def run_ar_ref(
    prompt_tokens: list[int],
    max_new_tokens: int,
    tensor_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attn_scale: float,
    temperature: float = 0.0,
    gen: torch.Generator | None = None,
) -> dict[str, Any]:
    eos_id = int(cfg["eosTokenId"])
    prefill = forward_40_layers_ref(
        prompt_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    kvs = append_kvs(None, prefill["newKvs"])
    prefill_u = float(draw_open_unit_uniforms(gen, (1,)).item()) if temperature > 0.0 and gen is not None else None
    first_tok = sample_causal_token_ref(prefill["logits"][-1], temperature, prefill_u)
    generated = [first_tok]
    step_last_logits: list[list[float]] = []
    step_last_probs: list[list[float]] = []
    step_uniforms: list[float] = []

    actual_passes = 1
    decode_passes = 0
    ref_nfe = 1

    if first_tok != eos_id:
        while len(generated) < max_new_tokens:
            prev_tok = generated[-1]
            step_fwd = forward_40_layers_ref(
                [prev_tok],
                tensor_map,
                cfg,
                inv_freq,
                attn_scale,
                mask_mode="causal",
                lora_enabled=False,
                use_cache=True,
                past_kvs=kvs,
            )
            kvs = append_kvs(kvs, step_fwd["newKvs"])
            actual_passes += 1
            decode_passes += 1
            ref_nfe += 1
            step_last_logits.append(to_f32_list(step_fwd["logits"][-1], "ar.stepLogit"))
            step_last_probs.append(to_f32_list(step_fwd["probs"][-1], "ar.stepProb"))

            u = float(draw_open_unit_uniforms(gen, (1,)).item()) if temperature > 0.0 and gen is not None else None
            if u is not None:
                step_uniforms.append(u)
            nxt = sample_causal_token_ref(step_fwd["logits"][-1], temperature, u)
            generated.append(nxt)
            if nxt == eos_id:
                break

    return {
        "generatedTokenIds": generated,
        "prefillLastLogits": to_f32_list(prefill["logits"][-1], "ar.prefillLogit"),
        "prefillLastProbs": to_f32_list(prefill["probs"][-1], "ar.prefillProb"),
        "stepLastLogits": step_last_logits,
        "stepLastProbs": step_last_probs,
        "kvs": kvs,
        "counters": {
            "actualForwardPasses": actual_passes,
            "decodeForwardPasses": decode_passes,
            "referenceMethodNfe": ref_nfe,
        },
        "explicitVariates": {
            "prefillUniform": prefill_u,
            "arStepUniforms": step_uniforms,
        },
    }


def has_mask_before_first_eos(block: list[int], mask_id: int, eos_id: int) -> bool:
    limit = block.index(eos_id) if eos_id in block else len(block)
    return any(block[i] == mask_id for i in range(limit))


def run_dlm_ref(
    prompt_tokens: list[int],
    block_length: int,
    max_new_tokens: int,
    causal_context: bool,
    temperature: float,
    threshold: float | None,
    tensor_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attn_scale: float,
    gen: torch.Generator | None = None,
) -> dict[str, Any]:
    mask_id = int(cfg["maskTokenId"])
    eos_id = int(cfg["eosTokenId"])
    vocab_size = int(cfg["vocabSize"])

    prefill = forward_40_layers_ref(
        prompt_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal" if causal_context else "bidirectional",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    kvs = append_kvs(None, prefill["newKvs"])
    actual_passes = 1
    decode_passes = 0
    ref_nfe = 0

    prefill_u: float | None = None
    pending_seed: int | None = None
    if causal_context:
        prefill_u = float(draw_open_unit_uniforms(gen, (1,)).item()) if temperature > 0.0 and gen is not None else None
        pending_seed = sample_causal_token_ref(prefill["logits"][-1], temperature, prefill_u)

    generated: list[int] = []
    blocks_out: list[dict[str, Any]] = []
    gumbel_uniforms_by_block: list[list[list[float]]] = []
    commit_uniforms: list[float] = []

    while len(generated) < max_new_tokens:
        block = [mask_id] * block_length
        if causal_context and pending_seed is not None:
            block[0] = pending_seed

        init_masked = sum(1 for tok in block if tok == mask_id)
        quotas = compute_static_transfer_quota_ref(init_masked, block_length) if threshold is None else None
        denoise_steps_out: list[dict[str, Any]] = []
        block_gumbel_uniforms: list[list[float]] = []

        for step in range(block_length):
            curr_masked = sum(1 for tok in block if tok == mask_id)
            if curr_masked == 0:
                break
            denoise_fwd = forward_40_layers_ref(
                block,
                tensor_map,
                cfg,
                inv_freq,
                attn_scale,
                mask_mode="bidirectional",
                lora_enabled=False,
                use_cache=False,
                past_kvs=kvs,
            )
            actual_passes += 1
            decode_passes += 1
            ref_nfe += 1

            quota = quotas[step] if quotas is not None else curr_masked
            step_u: list[float] | None = None
            if temperature > 0.0 and gen is not None:
                step_u = [float(x) for x in draw_open_unit_uniforms(gen, (block_length * vocab_size,)).tolist()]
                block_gumbel_uniforms.append(step_u)

            step_res = run_sampler_a_step_ref(
                denoise_fwd["logits"],
                block,
                quota,
                temperature,
                threshold,
                mask_id,
                step_u,
            )
            block = step_res["updatedBlock"]
            denoise_steps_out.append({
                "stepIndex": step,
                "quota": quota,
                "blockAfter": list(block),
                "candidateTokens": step_res["candidateTokens"],
                "confidences": [None if math.isinf(c) else c for c in step_res["confidences"]],
                "transferIndex": step_res["transferIndex"],
                "remainingMaskCount": step_res["remainingMaskCount"],
            })

            if not has_mask_before_first_eos(block, mask_id, eos_id):
                break

        commit_fwd = forward_40_layers_ref(
            block,
            tensor_map,
            cfg,
            inv_freq,
            attn_scale,
            mask_mode="causal" if causal_context else "bidirectional",
            lora_enabled=False,
            use_cache=True,
            past_kvs=kvs,
        )
        kvs = append_kvs(kvs, commit_fwd["newKvs"])
        actual_passes += 1
        decode_passes += 1
        ref_nfe += 1

        first_eos = block.index(eos_id) if eos_id in block else None
        emitted = block[: first_eos + 1] if first_eos is not None else list(block)
        generated.extend(emitted)

        if causal_context and first_eos is None:
            cu = float(draw_open_unit_uniforms(gen, (1,)).item()) if temperature > 0.0 and gen is not None else None
            if cu is not None:
                commit_uniforms.append(cu)
            pending_seed = sample_causal_token_ref(commit_fwd["logits"][-1], temperature, cu)
        else:
            pending_seed = None

        gumbel_uniforms_by_block.append(block_gumbel_uniforms)
        blocks_out.append({
            "denoisePassCount": len(denoise_steps_out),
            "denoiseSteps": denoise_steps_out,
            "finalBlockTokens": list(block),
            "emittedTokens": emitted,
            "commitLastLogits": to_f32_list(commit_fwd["logits"][-1], "dlm.commitLastLogit"),
        })
        if first_eos is not None:
            break

    return {
        "generatedTokenIds": generated,
        "kvMaskProvenance": "token_causal" if causal_context else "block_bidirectional",
        "blocks": blocks_out,
        "counters": {
            "actualForwardPasses": actual_passes,
            "decodeForwardPasses": decode_passes,
            "referenceMethodNfe": ref_nfe,
        },
        "explicitVariates": {
            "prefillUniform": prefill_u,
            "dlmBlockGumbelUniforms": gumbel_uniforms_by_block,
            "dlmCommitUniforms": commit_uniforms,
        },
    }


def run_linear_spec_ref(
    prompt_tokens: list[int],
    block_length: int,
    max_new_tokens: int,
    lora_enabled: bool,
    temperature: float,
    threshold: float,
    tensor_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attn_scale: float,
    gen: torch.Generator | None = None,
) -> dict[str, Any]:
    mask_id = int(cfg["maskTokenId"])
    eos_id = int(cfg["eosTokenId"])

    prefill = forward_40_layers_ref(
        prompt_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    kvs = append_kvs(None, prefill["newKvs"])
    prefill_u = float(draw_open_unit_uniforms(gen, (1,)).item()) if temperature > 0.0 and gen is not None else None
    pending_seed: int | None = sample_causal_token_ref(prefill["logits"][-1], temperature, prefill_u)
    generated = [pending_seed]

    actual_passes = 1
    decode_passes = 0
    ref_nfe = 1

    cycles_out: list[dict[str, Any]] = []
    draft_uniforms_by_cycle: list[list[list[float]]] = []
    verify_uniforms_by_cycle: list[list[float]] = []

    if pending_seed != eos_id:
        while len(generated) < max_new_tokens and pending_seed is not None:
            cache_len = kvs[0][0].shape[0]
            block = [pending_seed] + [mask_id] * (block_length - 1)
            cycle_draft_uniforms: list[list[float]] = []
            draft_pass_count = 0
            first_draft_logits: torch.Tensor | None = None

            for _ in range(block_length):
                if mask_id not in block:
                    break
                draft_fwd = forward_40_layers_ref(
                    block,
                    tensor_map,
                    cfg,
                    inv_freq,
                    attn_scale,
                    mask_mode="bidirectional",
                    lora_enabled=lora_enabled,
                    use_cache=False,
                    past_kvs=kvs,
                )
                if first_draft_logits is None:
                    first_draft_logits = draft_fwd["logits"]
                actual_passes += 1
                decode_passes += 1
                ref_nfe += 1
                draft_pass_count += 1

                du: list[float] | None = None
                if temperature > 0.0 and gen is not None:
                    du = [float(x) for x in draw_open_unit_uniforms(gen, (block_length,)).tolist()]
                    cycle_draft_uniforms.append(du)

                draft_res = run_sampler_b_step_ref(
                    draft_fwd["logits"],
                    block,
                    temperature,
                    threshold,
                    mask_id,
                    du,
                )
                block = draft_res["updatedBlock"]

                if threshold > 0.0 and temperature <= 0.0 and draft_res["noProgress"]:
                    break
                if draft_res["remainingMaskCount"] == 0 or threshold <= 0.0:
                    break

            verify_fwd = forward_40_layers_ref(
                block,
                tensor_map,
                cfg,
                inv_freq,
                attn_scale,
                mask_mode="causal",
                lora_enabled=False,
                use_cache=True,
                past_kvs=kvs,
            )
            pre_crop_kvs = append_kvs(kvs, verify_fwd["newKvs"])
            actual_passes += 1
            decode_passes += 1
            ref_nfe += 1

            vu: list[float] | None = None
            if temperature > 0.0 and gen is not None:
                vu = [float(x) for x in draw_open_unit_uniforms(gen, (block_length,)).tolist()]
                verify_uniforms_by_cycle.append(vu)

            verify_ar_tokens = [
                sample_causal_token_ref(
                    verify_fwd["logits"][i],
                    temperature,
                    vu[i] if vu is not None else None,
                )
                for i in range(block_length)
            ]

            matched = 0
            while matched < block_length - 1 and verify_ar_tokens[matched] == block[matched + 1]:
                matched += 1
            accepted = matched + 1

            first_eos: int | None = None
            for j in range(accepted):
                if block[j] == eos_id:
                    first_eos = j
                    break

            if first_eos is not None:
                emitted = list(block[1 : first_eos + 1])
                eos_hit = True
                pending_seed = None
                browser_crop_len = cache_len + first_eos + 1
            else:
                corr = verify_ar_tokens[matched]
                emitted = list(block[1 : matched + 1]) + [corr]
                eos_hit = corr == eos_id
                pending_seed = None if eos_hit else corr
                browser_crop_len = cache_len + accepted

            kvs = crop_kvs(pre_crop_kvs, browser_crop_len)
            generated.extend(emitted)
            draft_uniforms_by_cycle.append(cycle_draft_uniforms)

            cycles_out.append({
                "cacheLenBeforeVerify": cache_len,
                "draftedBlockTokens": list(block),
                "draftPassCount": draft_pass_count,
                "verifyArTokens": verify_ar_tokens,
                "matchedDraftCount": matched,
                "acceptedCount": accepted,
                "firstEosOffset": first_eos,
                "browserPostEosCropLen": browser_crop_len,
                "upstreamPreEosCropLen": cache_len + accepted,
                "emittedTokens": emitted,
                "firstDraftLogits": to_f32_list(first_draft_logits, "spec.draftLogits") if first_draft_logits is not None else [],
                "verifyLogits": to_f32_list(verify_fwd["logits"], "spec.verifyLogits"),
                "postCropKvs": kvs,
            })

            if eos_hit:
                break

    return {
        "generatedTokenIds": generated,
        "cycles": cycles_out,
        "kvs": kvs,
        "counters": {
            "actualForwardPasses": actual_passes,
            "decodeForwardPasses": decode_passes,
            "referenceMethodNfe": ref_nfe,
        },
        "explicitVariates": {
            "prefillUniform": prefill_u,
            "specCycleDraftUniforms": draft_uniforms_by_cycle,
            "specCycleVerifyUniforms": verify_uniforms_by_cycle,
        },
    }


def pick_cdf_midpoint_uniform(logits_row: torch.Tensor, temperature: float, target_token: int) -> float:
    scaled = (logits_row.to(torch.float32) / float(temperature)).to(torch.float32)
    probs = F.softmax(scaled, dim=-1).tolist()
    total = sum(p for p in probs if p > 0.0)
    cdf_low = 0.0
    for v, p in enumerate(probs):
        if v == target_token:
            return float((cdf_low + 0.5 * p) / total)
        if p > 0.0:
            cdf_low += p
    raise RuntimeError(f"Target token {target_token} had non-positive probability")


# ============================================================================
# FIXTURE BUILDER (CATEGORIES 7-11)
# ============================================================================

def build_trajectories_parity_fixture(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    gen: torch.Generator,
) -> dict[str, Any]:
    cfg = manifest["config"]
    inv_freq, attn_scale = compute_yarn_inv_freq_ref(cfg["rope"])
    prompt = [0, 1, 2, 3, 4, 5, 6, 7]
    mask_id = int(cfg["maskTokenId"])
    eos_id = int(cfg["eosTokenId"])
    vocab_size = int(cfg["vocabSize"])

    # ------------------------------------------------------------------------
    # Category 7: Incremental vs Full-Context Causal AR crossing pos = 16
    # ------------------------------------------------------------------------
    ar_cat7 = run_ar_ref(prompt, 11, tensor_map, cfg, inv_freq, attn_scale, temperature=0.0)
    full_seq_18 = prompt + ar_cat7["generatedTokenIds"][:10]
    assert len(full_seq_18) == 18
    full_18_fwd = forward_40_layers_ref(
        full_seq_18,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )

    inc_prefill_logit_t = torch.tensor(ar_cat7["prefillLastLogits"], dtype=torch.float32)
    inc_steps_logits_t = torch.tensor(ar_cat7["stepLastLogits"], dtype=torch.float32)
    inc_logits_7_17 = torch.cat([inc_prefill_logit_t.unsqueeze(0), inc_steps_logits_t], dim=0)
    full_logits_7_17 = full_18_fwd["logits"][7:18]
    max_logit_diff = float((inc_logits_7_17 - full_logits_7_17).abs().max().item())

    inc_prefill_prob_t = torch.tensor(ar_cat7["prefillLastProbs"], dtype=torch.float32)
    inc_steps_probs_t = torch.tensor(ar_cat7["stepLastProbs"], dtype=torch.float32)
    inc_probs_7_17 = torch.cat([inc_prefill_prob_t.unsqueeze(0), inc_steps_probs_t], dim=0)
    full_probs_7_17 = full_18_fwd["probs"][7:18]
    max_prob_diff = float((inc_probs_7_17 - full_probs_7_17).abs().max().item())

    max_kv_diff = 0.0
    for l_idx in range(int(cfg["numLayers"])):
        dk = float((ar_cat7["kvs"][l_idx][0] - full_18_fwd["newKvs"][l_idx][0]).abs().max().item())
        dv = float((ar_cat7["kvs"][l_idx][1] - full_18_fwd["newKvs"][l_idx][1]).abs().max().item())
        max_kv_diff = max(max_kv_diff, dk, dv)

    if max_logit_diff > 5e-5 or max_prob_diff > 2e-5 or max_kv_diff > 2e-5:
        raise RuntimeError(
            f"Cat 7 PyTorch self-check failed: logit={max_logit_diff}, prob={max_prob_diff}, kv={max_kv_diff}"
        )

    cat7 = {
        "promptTokenIds": prompt,
        "maxNewTokens": 11,
        "generatedTokenIds": ar_cat7["generatedTokenIds"],
        "fullSequence18": full_seq_18,
        "prefillLastLogits": ar_cat7["prefillLastLogits"],
        "prefillLastProbs": ar_cat7["prefillLastProbs"],
        "stepLastLogits": ar_cat7["stepLastLogits"],
        "stepLastProbs": ar_cat7["stepLastProbs"],
        "full18Logits": to_f32_list(full_18_fwd["logits"], "cat7.full18Logits"),
        "full18Probs": to_f32_list(full_18_fwd["probs"], "cat7.full18Probs"),
        "full18LayerKvs": {
            str(l): {
                "k": to_f32_list(full_18_fwd["newKvs"][l][0], f"cat7.L{l}.k"),
                "v": to_f32_list(full_18_fwd["newKvs"][l][1], f"cat7.L{l}.v"),
            }
            for l in (0, 19, 39)
        },
        "incrementalVsFullMaxLogitDiff": max_logit_diff,
        "incrementalVsFullMaxProbDiff": max_prob_diff,
        "incrementalVsFullMaxKvDiff": max_kv_diff,
        "counters": ar_cat7["counters"],
    }

    # ------------------------------------------------------------------------
    # Category 8: Dual Samplers & causal_context = True | False
    # ------------------------------------------------------------------------
    dlm_causal_null = run_dlm_ref(prompt, 4, 8, True, 0.0, None, tensor_map, cfg, inv_freq, attn_scale)
    dlm_causal_022 = run_dlm_ref(prompt, 4, 8, True, 0.0, 0.22, tensor_map, cfg, inv_freq, attn_scale)
    dlm_causal_090 = run_dlm_ref(prompt, 4, 8, True, 0.0, 0.90, tensor_map, cfg, inv_freq, attn_scale)
    dlm_causal_t07 = run_dlm_ref(prompt, 4, 8, True, 0.7, 0.22, tensor_map, cfg, inv_freq, attn_scale, gen=gen)
    dlm_bidir_null = run_dlm_ref(prompt, 4, 8, False, 0.0, None, tensor_map, cfg, inv_freq, attn_scale)
    dlm_bidir_022 = run_dlm_ref(prompt, 4, 8, False, 0.0, 0.22, tensor_map, cfg, inv_freq, attn_scale)

    # Sampler A direct unit edge-case fixtures
    # 8.A.1: T=0.7 unscaled vs scaled softmax confidence check
    u_logits = torch.randn((4, vocab_size), generator=gen, dtype=torch.float32) * 1.1
    u_logits[1, 5] = 2.0
    u_logits[1, 6] = 1.4
    u_logits[2, 7] = 2.5
    u_logits[2, 8] = 1.5
    u_gumbel = [float(x) for x in draw_open_unit_uniforms(gen, (4 * vocab_size,)).tolist()]
    sa_unscaled = run_sampler_a_step_ref(u_logits, [9, mask_id, mask_id, mask_id], 3, 0.7, 0.22, mask_id, u_gumbel)
    scaled_probs_contrast = F.softmax((u_logits / 0.7).to(torch.float32), dim=-1)
    scaled_conf_contrast = [
        float(scaled_probs_contrast[t, sa_unscaled["candidateTokens"][t]].item()) if t > 0 else None
        for t in range(4)
    ]

    # 8.A.2: k = 0 unconditional floor when all confidences < threshold
    low_conf_logits = torch.zeros((4, vocab_size), dtype=torch.float32)
    low_conf_logits[2, 10] = 0.5
    sa_k0_floor = run_sampler_a_step_ref(low_conf_logits, [9, mask_id, mask_id, mask_id], 3, 0.0, 0.99, mask_id)
    assert sa_k0_floor["selectedCount"] == 1 and sa_k0_floor["transferIndex"][2] == 1

    # 8.A.3: Non-strict >= threshold (exact 0.25 probability on 4 equal-logit entries)
    ge_logits = torch.full((4, vocab_size), -1e9, dtype=torch.float32)
    ge_logits[1, 0:4] = 0.0  # exact 0.25
    ge_logits[2, 0:4] = 0.0  # exact 0.25
    ge_logits[3, 0] = 0.0    # exact 1.0 (rank-0 floor)
    sa_ge = run_sampler_a_step_ref(ge_logits, [9, mask_id, mask_id, mask_id], 3, 0.0, 0.25, mask_id)
    assert sa_ge["transferIndex"] == [0, 1, 1, 1]

    # 8.A.4: Raw un-suppressed [MASK] logits
    mask_logits = torch.full((4, vocab_size), -5.0, dtype=torch.float32)
    mask_logits[:, mask_id] = 4.0
    sa_raw_mask = run_sampler_a_step_ref(mask_logits, [9, mask_id, mask_id, mask_id], 3, 0.0, 0.22, mask_id)
    assert sa_raw_mask["unmaskedMaskTokenCount"] == 3 and sa_raw_mask["updatedBlock"] == [9, mask_id, mask_id, mask_id]

    # Sampler B trajectories & unit cases
    spec_t0_0 = run_linear_spec_ref(prompt, 4, 8, False, 0.0, 0.0, tensor_map, cfg, inv_freq, attn_scale)
    spec_t0_022 = run_linear_spec_ref(prompt, 4, 8, False, 0.0, 0.22, tensor_map, cfg, inv_freq, attn_scale)
    spec_t0_090 = run_linear_spec_ref(prompt, 4, 8, False, 0.0, 0.90, tensor_map, cfg, inv_freq, attn_scale)
    spec_t07_022 = run_linear_spec_ref(prompt, 4, 8, False, 0.7, 0.22, tensor_map, cfg, inv_freq, attn_scale, gen=gen)

    # Sampler B direct unit cases
    sb_logits = torch.full((4, vocab_size), -1e9, dtype=torch.float32)
    sb_logits[1, 0:2] = 0.0   # at T=1.0 -> 0.50; at T=0.5 -> still 0.50
    sb_logits[2, 0] = 1.4
    sb_logits[2, 1] = 0.0
    sb_logits[3, mask_id] = 3.0  # raw un-suppressed [MASK]
    sb_logits[3, 4] = -2.0
    sb_u = [0.25, 0.25, 0.1, 0.5]
    sb_unit_0 = run_sampler_b_step_ref(sb_logits, [9, mask_id, mask_id, mask_id], 0.0, 0.0, mask_id)
    sb_unit_t07 = run_sampler_b_step_ref(sb_logits, [9, mask_id, mask_id, mask_id], 0.7, 0.60, mask_id, sb_u)
    sb_fallback_logits = torch.zeros((4, vocab_size), dtype=torch.float32)
    sb_fallback_logits[2, 11] = 0.8
    sb_unit_fallback = run_sampler_b_step_ref(sb_fallback_logits, [9, mask_id, mask_id, mask_id], 0.0, 0.99, mask_id)
    assert sb_unit_fallback["unmaskMask"] == [0, 0, 1, 0]

    def compact_dlm_run(res: dict[str, Any]) -> dict[str, Any]:
        return {
            "generatedTokenIds": res["generatedTokenIds"],
            "kvMaskProvenance": res["kvMaskProvenance"],
            "counters": res["counters"],
            "blocks": res["blocks"],
            "explicitVariates": res["explicitVariates"],
        }

    def compact_spec_run(res: dict[str, Any]) -> dict[str, Any]:
        return {
            "generatedTokenIds": res["generatedTokenIds"],
            "counters": res["counters"],
            "explicitVariates": res["explicitVariates"],
            "cycles": [
                {
                    "cacheLenBeforeVerify": c["cacheLenBeforeVerify"],
                    "draftedBlockTokens": c["draftedBlockTokens"],
                    "draftPassCount": c["draftPassCount"],
                    "verifyArTokens": c["verifyArTokens"],
                    "matchedDraftCount": c["matchedDraftCount"],
                    "acceptedCount": c["acceptedCount"],
                    "emittedTokens": c["emittedTokens"],
                }
                for c in res["cycles"]
            ],
        }

    cat8 = {
        "promptTokenIds": prompt,
        "blockLength": 4,
        "maxNewTokens": 8,
        "dlmRuns": {
            "causalThresholdNull": compact_dlm_run(dlm_causal_null),
            "causalThreshold022": compact_dlm_run(dlm_causal_022),
            "causalThreshold090": compact_dlm_run(dlm_causal_090),
            "causalStochasticT07": {
                "temperature": 0.7,
                "threshold": 0.22,
                **compact_dlm_run(dlm_causal_t07),
            },
            "bidirThresholdNull": compact_dlm_run(dlm_bidir_null),
            "bidirThreshold022": compact_dlm_run(dlm_bidir_022),
        },
        "samplerAUnitCases": {
            "unscaledVsScaledConfidence": {
                "logits": to_f32_list(u_logits, "cat8.sa.unscaledLogits"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.7,
                "threshold": 0.22,
                "explicitGumbelUniforms": u_gumbel,
                "candidateTokens": sa_unscaled["candidateTokens"],
                "unscaledConfidences": [None if math.isinf(c) else c for c in sa_unscaled["confidences"]],
                "scaledConfidencesForContrast": scaled_conf_contrast,
                "transferIndex": sa_unscaled["transferIndex"],
                "updatedBlock": sa_unscaled["updatedBlock"],
            },
            "k0UnconditionalFloor": {
                "logits": to_f32_list(low_conf_logits, "cat8.sa.k0Logits"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.0,
                "threshold": 0.99,
                "candidateTokens": sa_k0_floor["candidateTokens"],
                "confidences": [None if math.isinf(c) else c for c in sa_k0_floor["confidences"]],
                "transferIndex": sa_k0_floor["transferIndex"],
                "updatedBlock": sa_k0_floor["updatedBlock"],
                "selectedCount": sa_k0_floor["selectedCount"],
            },
            "nonStrictThresholdGe": {
                "logits": to_f32_list(ge_logits, "cat8.sa.geLogits"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.0,
                "threshold": 0.25,
                "confidences": [None if math.isinf(c) else c for c in sa_ge["confidences"]],
                "transferIndex": sa_ge["transferIndex"],
                "updatedBlock": sa_ge["updatedBlock"],
            },
            "rawUnsuppressedMaskLogits": {
                "logits": to_f32_list(mask_logits, "cat8.sa.maskLogits"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.0,
                "threshold": 0.22,
                "candidateTokens": sa_raw_mask["candidateTokens"],
                "confidences": [None if math.isinf(c) else c for c in sa_raw_mask["confidences"]],
                "transferIndex": sa_raw_mask["transferIndex"],
                "updatedBlock": sa_raw_mask["updatedBlock"],
                "unmaskedMaskTokenCount": sa_raw_mask["unmaskedMaskTokenCount"],
            },
        },
        "samplerBRuns": {
            "linearSpecThreshold0T0": compact_spec_run(spec_t0_0),
            "linearSpecThreshold022T0": compact_spec_run(spec_t0_022),
            "linearSpecThreshold090T0": compact_spec_run(spec_t0_090),
            "linearSpecThreshold022T07": {
                "temperature": 0.7,
                "threshold": 0.22,
                **compact_spec_run(spec_t07_022),
            },
        },
        "samplerBUnitCases": {
            "threshold0SinglePass": {
                "draftLogits": to_f32_list(sb_logits, "cat8.sb.logits"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.0,
                "threshold": 0.0,
                **sb_unit_0,
            },
            "stochasticTempScaledConfAndMaskRaw": {
                "draftLogits": to_f32_list(sb_logits, "cat8.sb.logitsT07"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.7,
                "threshold": 0.60,
                "explicitUniforms": sb_u,
                **sb_unit_t07,
            },
            "argmaxFallback": {
                "draftLogits": to_f32_list(sb_fallback_logits, "cat8.sb.fallback"),
                "blockTokens": [9, mask_id, mask_id, mask_id],
                "temperature": 0.0,
                "threshold": 0.99,
                **sb_unit_fallback,
            },
        },
    }

    # ------------------------------------------------------------------------
    # Category 9: Cache Lifecycle, kvMaskProvenance & Rollback / D19 EOS Crop
    # ------------------------------------------------------------------------
    # 9A: Real speculative cycle rollback vs pure AR KV on accepted prefix
    c0 = spec_t0_0["cycles"][0]
    accepted_prefix_len = c0["cacheLenBeforeVerify"] + c0["acceptedCount"]
    # Note: seedToken (draftedBlockTokens[0]) was emitted at prefill; accepted prefix tokens in KV:
    accepted_prefix_tokens = prompt + c0["draftedBlockTokens"][: c0["acceptedCount"]]
    assert len(accepted_prefix_tokens) == accepted_prefix_len
    pure_ar_prefix_fwd = forward_40_layers_ref(
        accepted_prefix_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    for l_idx in range(int(cfg["numLayers"])):
        dk = float((c0["postCropKvs"][l_idx][0] - pure_ar_prefix_fwd["newKvs"][l_idx][0]).abs().max().item())
        dv = float((c0["postCropKvs"][l_idx][1] - pure_ar_prefix_fwd["newKvs"][l_idx][1]).abs().max().item())
        assert dk <= 2e-5 and dv <= 2e-5

    # 9B: Explicit D19 Mid-Accepted-Prefix EOS Crop Fixture (Clarification #1)
    prefill_8_fwd = forward_40_layers_ref(
        prompt,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    d19_block = [20, 9, eos_id, 5]  # EOS=33 at offset 2 < accepted-1 (=3)
    d19_verify_fwd = forward_40_layers_ref(
        d19_block,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=prefill_8_fwd["newKvs"],
    )
    d19_temp = 2.0
    d19_uniforms = [
        pick_cdf_midpoint_uniform(d19_verify_fwd["logits"][0], d19_temp, d19_block[1]),
        pick_cdf_midpoint_uniform(d19_verify_fwd["logits"][1], d19_temp, d19_block[2]),
        pick_cdf_midpoint_uniform(d19_verify_fwd["logits"][2], d19_temp, d19_block[3]),
        0.5,
    ]
    d19_verify_ar_tokens = [
        sample_causal_token_ref(d19_verify_fwd["logits"][i], d19_temp, d19_uniforms[i])
        for i in range(4)
    ]
    assert d19_verify_ar_tokens[0] == 9 and d19_verify_ar_tokens[1] == eos_id and d19_verify_ar_tokens[2] == 5
    d19_matched = 3
    d19_accepted = 4
    d19_first_eos = 2
    d19_browser_crop_len = 8 + d19_first_eos + 1  # 11
    d19_upstream_crop_len = 8 + d19_accepted      # 12
    d19_pre_crop_kvs = append_kvs(prefill_8_fwd["newKvs"], d19_verify_fwd["newKvs"])
    d19_post_crop_kvs = crop_kvs(d19_pre_crop_kvs, d19_browser_crop_len)
    d19_ar_prefix_tokens = prompt + d19_block[: d19_first_eos + 1]  # len = 11
    d19_pure_ar_fwd = forward_40_layers_ref(
        d19_ar_prefix_tokens,
        tensor_map,
        cfg,
        inv_freq,
        attn_scale,
        mask_mode="causal",
        lora_enabled=False,
        use_cache=True,
        past_kvs=None,
    )
    for l_idx in range(int(cfg["numLayers"])):
        dk = float((d19_post_crop_kvs[l_idx][0] - d19_pure_ar_fwd["newKvs"][l_idx][0]).abs().max().item())
        dv = float((d19_post_crop_kvs[l_idx][1] - d19_pure_ar_fwd["newKvs"][l_idx][1]).abs().max().item())
        assert dk <= 2e-5 and dv <= 2e-5

    cat9 = {
        "promptTokenIds": prompt,
        "specCycleRollback": {
            "blockLength": 4,
            "cycle0DraftedBlock": c0["draftedBlockTokens"],
            "cycle0AcceptedCount": c0["acceptedCount"],
            "cycle0PreCropLen": c0["cacheLenBeforeVerify"] + 4,
            "cycle0PostCropLen": accepted_prefix_len,
            "acceptedPrefixTokens": accepted_prefix_tokens,
            "croppedLayerKvs": {
                str(l): {
                    "k": to_f32_list(c0["postCropKvs"][l][0], f"cat9.crop.L{l}.k"),
                    "v": to_f32_list(c0["postCropKvs"][l][1], f"cat9.crop.L{l}.v"),
                }
                for l in (0, 19, 39)
            },
            "pureArLayerKvs": {
                str(l): {
                    "k": to_f32_list(pure_ar_prefix_fwd["newKvs"][l][0], f"cat9.ar.L{l}.k"),
                    "v": to_f32_list(pure_ar_prefix_fwd["newKvs"][l][1], f"cat9.ar.L{l}.v"),
                }
                for l in (0, 19, 39)
            },
        },
        "d19MidAcceptedPrefixEosCrop": {
            "promptTokenIds": prompt,
            "cacheLenBeforeVerify": 8,
            "draftedBlockTokens": d19_block,
            "verifyTemperature": d19_temp,
            "explicitVerifyUniforms": d19_uniforms,
            "verifyArTokens": d19_verify_ar_tokens,
            "matchedDraftCount": d19_matched,
            "acceptedCount": d19_accepted,
            "firstEosOffset": d19_first_eos,
            "browserPostEosCropLen": d19_browser_crop_len,
            "upstreamPreEosCropLen": d19_upstream_crop_len,
            "emittedTokens": [9, eos_id],
            "postEosArPrefixTokens": d19_ar_prefix_tokens,
            "croppedLayerKvs": {
                str(l): {
                    "k": to_f32_list(d19_post_crop_kvs[l][0], f"cat9.d19.crop.L{l}.k"),
                    "v": to_f32_list(d19_post_crop_kvs[l][1], f"cat9.d19.crop.L{l}.v"),
                }
                for l in (0, 39)
            },
            "pureArLayerKvs": {
                str(l): {
                    "k": to_f32_list(d19_pure_ar_fwd["newKvs"][l][0], f"cat9.d19.ar.L{l}.k"),
                    "v": to_f32_list(d19_pure_ar_fwd["newKvs"][l][1], f"cat9.d19.ar.L{l}.v"),
                }
                for l in (0, 39)
            },
        },
    }

    # ------------------------------------------------------------------------
    # Category 10: Linear Self-Speculation (±LoRA), Greedy T = 0 (maxNewTokens = 12)
    # ------------------------------------------------------------------------
    ar_12 = run_ar_ref(prompt, 12, tensor_map, cfg, inv_freq, attn_scale, temperature=0.0)
    spec_12 = run_linear_spec_ref(prompt, 4, 12, False, 0.0, 0.0, tensor_map, cfg, inv_freq, attn_scale)
    spec_lora_12 = run_linear_spec_ref(prompt, 4, 12, True, 0.0, 0.0, tensor_map, cfg, inv_freq, attn_scale)

    assert ar_12["generatedTokenIds"] == spec_12["generatedTokenIds"] == spec_lora_12["generatedTokenIds"], (
        f"Greedy T=0 token mismatch: ar={ar_12['generatedTokenIds']}, "
        f"spec={spec_12['generatedTokenIds']}, lora={spec_lora_12['generatedTokenIds']}"
    )
    c0_no_lora = spec_12["cycles"][0]
    c0_with_lora = spec_lora_12["cycles"][0]
    assert c0_no_lora["draftedBlockTokens"] == c0_with_lora["draftedBlockTokens"]
    draft_delta = max(
        abs(a - b) for a, b in zip(c0_no_lora["firstDraftLogits"], c0_with_lora["firstDraftLogits"])
    )
    verify_delta = max(
        abs(a - b) for a, b in zip(c0_no_lora["verifyLogits"], c0_with_lora["verifyLogits"])
    )
    assert draft_delta > 1e-4 and verify_delta == 0.0

    cat10 = {
        "promptTokenIds": prompt,
        "maxNewTokens": 12,
        "blockLength": 4,
        "arTokens": ar_12["generatedTokenIds"],
        "linearSpecTokens": spec_12["generatedTokenIds"],
        "linearSpecLoraTokens": spec_lora_12["generatedTokenIds"],
        "cycle0DraftLogitsNoLora": c0_no_lora["firstDraftLogits"],
        "cycle0DraftLogitsWithLora": c0_with_lora["firstDraftLogits"],
        "cycle0VerifyLogitsNoLora": c0_no_lora["verifyLogits"],
        "cycle0VerifyLogitsWithLora": c0_with_lora["verifyLogits"],
        "draftLogitMaxDiffLoraOnVsOff": draft_delta,
        "verifyLogitMaxDiffLoraOnVsOff": verify_delta,
        "arCounters": ar_12["counters"],
        "linearSpecCounters": spec_12["counters"],
        "linearSpecLoraCounters": spec_lora_12["counters"],
    }

    # ------------------------------------------------------------------------
    # Category 11: Stochastic T = 0.7 Explicit-Variate Replay
    # ------------------------------------------------------------------------
    ar_t07 = run_ar_ref(prompt, 8, tensor_map, cfg, inv_freq, attn_scale, temperature=0.7, gen=gen)
    dlm_t07 = run_dlm_ref(prompt, 4, 8, True, 0.7, 0.22, tensor_map, cfg, inv_freq, attn_scale, gen=gen)
    spec_t07 = run_linear_spec_ref(prompt, 4, 8, False, 0.7, 0.0, tensor_map, cfg, inv_freq, attn_scale, gen=gen)
    spec_lora_t07 = run_linear_spec_ref(prompt, 4, 8, True, 0.7, 0.0, tensor_map, cfg, inv_freq, attn_scale, gen=gen)

    cat11 = {
        "promptTokenIds": prompt,
        "temperature": 0.7,
        "blockLength": 4,
        "maxNewTokens": 8,
        "ar": {
            "explicitVariates": ar_t07["explicitVariates"],
            "generatedTokenIds": ar_t07["generatedTokenIds"],
            "lastStepLogits": ar_t07["stepLastLogits"][-1] if ar_t07["stepLastLogits"] else ar_t07["prefillLastLogits"],
            "counters": ar_t07["counters"],
        },
        "dlmCausal": {
            "threshold": 0.22,
            "explicitVariates": dlm_t07["explicitVariates"],
            "generatedTokenIds": dlm_t07["generatedTokenIds"],
            "lastBlockCommitLogits": dlm_t07["blocks"][-1]["commitLastLogits"],
            "counters": dlm_t07["counters"],
        },
        "linearSpec": {
            "threshold": 0.0,
            "explicitVariates": spec_t07["explicitVariates"],
            "generatedTokenIds": spec_t07["generatedTokenIds"],
            "lastCycleVerifyLogits": spec_t07["cycles"][-1]["verifyLogits"] if spec_t07["cycles"] else [],
            "counters": spec_t07["counters"],
        },
        "linearSpecLora": {
            "threshold": 0.0,
            "explicitVariates": spec_lora_t07["explicitVariates"],
            "generatedTokenIds": spec_lora_t07["generatedTokenIds"],
            "lastCycleVerifyLogits": spec_lora_t07["cycles"][-1]["verifyLogits"] if spec_lora_t07["cycles"] else [],
            "counters": spec_lora_t07["counters"],
        },
    }

    return {
        "metadata": {
            "generator": "scripts/reference/generate_diffusion_trajectory_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "weightsSha256": manifest["weightsSha256"],
            "backboneSha256": manifest["backboneSha256"],
            "loraSha256": manifest["loraSha256"],
            "tolerances": {
                "kvCacheProbsMaxAbs": 2e-5,
                "logitsMaxAbs": 5e-5,
                "tsInternalMaxAbs": 1e-6,
            },
        },
        "cat7IncrementalVsFullCausal": cat7,
        "cat8DualSamplersAndCausalContext": cat8,
        "cat9CacheLifecycleAndD19EosCrop": cat9,
        "cat10LinearSpecGreedyAndLora": cat10,
        "cat11StochasticExplicitVariates": cat11,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate independent PyTorch 2.15.0 trajectory parity fixtures for Nemotron-Labs-Diffusion (Categories 7-11)."
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
            "Run via `uv run scripts/reference/generate_diffusion_trajectory_fixtures.py`."
        )

    torch.manual_seed(20261127)
    gen = torch.Generator().manual_seed(20261127)

    manifest, tensor_map = load_exported_diffusion_weights(args.manifest)
    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    payload = build_trajectories_parity_fixture(manifest, tensor_map, gen)
    out_path = fixtures_dir / "pytorch-diffusion-trajectories-parity.json"
    out_path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote Diffusion trajectories parity fixture ({out_path.stat().st_size:,} bytes) -> {out_path}"
    )


if __name__ == "__main__":
    main()
