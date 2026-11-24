#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent PyTorch Reference Parity Fixture Generator for Nemotron 3 Nano Miniature Simulators.

Development/reference script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced automatically via PEP 723 metadata).

Reproduce both single-layer and complete 52-layer end-to-end fixtures:
  node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params
  uv run scripts/reference/generate_pytorch_parity_fixtures.py --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json

Mathematical Reference & Licensing Note:
  Implements NVIDIA NemotronH architecture equations (`NemotronHRMSNorm`, `NemotronHMamba2Mixer` +
  `MambaRMSNormGated` with `norm_before_gate=False`, `NemotronHMOE` + `NemotronHTopkRouter` +
  non-gated `ReLU^2` `NemotronHMLP`, and NoPE `32Q/2KV` GQA Attention, plus Untied Embeddings and
  Untied LM Head) using clean-room standard PyTorch (`torch`, `torch.nn.functional`).
  Reference specification: Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  (`modeling_nemotron_h.py`, Apache License 2.0; model weights separately under NVIDIA Open Model
  License — no external weights or third-party code are copied here).
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PINNED_TORCH_VERSION = "2.15.0"


def to_f32_list(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().to(torch.float32).reshape(-1).tolist()]


def to_i32_list(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.detach().to(torch.int64).reshape(-1).tolist()]


# ============================================================================
# SHARED PYTORCH REFERENCE LAYER OPERATIONS (Used for both single-layer & 52-layer parity)
# ============================================================================

def rmsnorm_ref(x_d_t: torch.Tensor, gamma: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    """x_d_t: [d, T], gamma: [d] -> matches NemotronHRMSNorm over channel dim 0."""
    x = x_d_t.transpose(0, 1)  # [T, d]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # [T, 1]
    x_norm = x / rms
    u = x_norm * gamma.unsqueeze(0)
    return {
        "rms": rms.squeeze(-1),           # [T]
        "X_norm": x_norm.transpose(0, 1), # [d, T]
        "U": u.transpose(0, 1),           # [d, T]
    }


def gqa_forward_ref(
    x: torch.Tensor,
    gamma: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    eps: float,
    n_q: int = 32,
    n_kv: int = 2,
    d_head: int = 4,
) -> dict[str, torch.Tensor]:
    """32Q / 2KV NoPE GQA Attention forward pass matching NemotronHAttention."""
    T = x.shape[1]
    q_concat = n_q * d_head  # 128

    rn = rmsnorm_ref(x, gamma, eps)
    u_t_d = rn["U"].transpose(0, 1)  # [T, d]

    q_flat = F.linear(u_t_d, w_q, bias=None).transpose(0, 1).contiguous()  # [128, T]
    k_flat = F.linear(u_t_d, w_k, bias=None).transpose(0, 1).contiguous()  # [8, T]
    v_flat = F.linear(u_t_d, w_v, bias=None).transpose(0, 1).contiguous()  # [8, T]

    q_h = q_flat.view(n_q, d_head, T).transpose(1, 2)  # [32, T, 4]
    k_h = k_flat.view(n_kv, d_head, T).transpose(1, 2).repeat_interleave(n_q // n_kv, dim=0)
    v_h = v_flat.view(n_kv, d_head, T).transpose(1, 2).repeat_interleave(n_q // n_kv, dim=0)

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(d_head)  # [32, T, T]
    causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    scores_masked = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    softmax = F.softmax(scores_masked, dim=-1)  # [32, T, T]

    v_out_h = torch.matmul(softmax, v_h)  # [32, T, 4]
    v_out_flat = v_out_h.transpose(1, 2).contiguous().view(q_concat, T)  # [128, T]
    attn_out = F.linear(v_out_flat.transpose(0, 1), w_o, bias=None).transpose(0, 1).contiguous()
    y = x + attn_out

    return {
        "U": rn["U"],
        "Q": q_flat,
        "K": k_flat,
        "V": v_flat,
        "Scores": scores_masked,
        "Softmax": softmax,
        "V_out": v_out_flat,
        "Attn_out": attn_out,
        "Y": y,
    }


def mamba2_forward_ref(
    x: torch.Tensor,
    gamma_in: torch.Tensor,
    w_in: torch.Tensor,
    w_conv: torch.Tensor,
    b_conv: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_skip: torch.Tensor,
    gamma_gate: torch.Tensor,
    w_out: torch.Tensor,
    eps: float,
    n_heads: int = 16,
    n_groups: int = 8,
    d_head: int = 4,
    d_state: int = 4,
    k_conv: int = 4,
) -> dict[str, torch.Tensor]:
    """Mamba-2 forward pass matching NemotronHMamba2Mixer + MambaRMSNormGated (norm_before_gate=False)."""
    T = x.shape[1]
    d_inner = n_heads * d_head                    # 64
    conv_dim = d_inner + 2 * n_groups * d_state   # 128
    group_size = d_inner // n_groups              # 8

    rn = rmsnorm_ref(x, gamma_in, eps)
    proj = F.linear(rn["U"].transpose(0, 1), w_in, bias=None).transpose(0, 1).contiguous()
    z = proj[:d_inner]
    xbc_pre = proj[d_inner : d_inner + conv_dim]
    dt_raw = proj[d_inner + conv_dim :]

    xbc_padded = F.pad(xbc_pre.unsqueeze(0), (k_conv - 1, 0))
    conv_out = F.conv1d(xbc_padded, w_conv.unsqueeze(1), b_conv, groups=conv_dim).squeeze(0)
    xbc = F.silu(conv_out)

    x_ssm = xbc[:d_inner].view(n_heads, d_head, T)
    b_ssm = xbc[d_inner : d_inner + n_groups * d_state].view(n_groups, d_state, T)
    c_ssm = xbc[d_inner + n_groups * d_state :].view(n_groups, d_state, T)

    a_decay = -torch.exp(a_log)
    dt = F.softplus(dt_raw + dt_bias.unsqueeze(1))
    da = torch.exp(dt * a_decay.unsqueeze(1))

    h_state = torch.zeros((n_heads, d_head, d_state), dtype=torch.float32)
    h_prev = torch.zeros_like(h_state)
    y_ssm = torch.zeros((n_heads, d_head, T), dtype=torch.float32)
    heads_per_group = n_heads // n_groups

    for t in range(T):
        if t == T - 1:
            h_prev = h_state.clone()
        b_rep = b_ssm[:, :, t].repeat_interleave(heads_per_group, dim=0)
        c_rep = c_ssm[:, :, t].repeat_interleave(heads_per_group, dim=0)
        da_t = da[:, t].view(n_heads, 1, 1)
        dt_t = dt[:, t].view(n_heads, 1, 1)
        x_t = x_ssm[:, :, t].unsqueeze(-1)
        h_state = da_t * h_state + dt_t * (x_t * b_rep.unsqueeze(1))
        y_ssm[:, :, t] = torch.einsum("hpn,hn->hp", h_state, c_rep)

    y_total = y_ssm + d_skip.view(n_heads, 1, 1) * x_ssm
    y_flat = y_total.view(d_inner, T)

    # MambaRMSNormGated (norm_before_gate = False): gate FIRST -> 8-group RMSNorm -> * gamma_gate
    gated = y_flat * F.silu(z)
    gated_grp = gated.view(n_groups, group_size, T)
    rms_gate = torch.sqrt(gated_grp.pow(2).mean(dim=1, keepdim=True) + eps)
    y_norm_pre = (gated_grp / rms_gate).view(d_inner, T)
    y_norm = y_norm_pre * gamma_gate.unsqueeze(1)

    mamba_out = F.linear(y_norm.transpose(0, 1), w_out, bias=None).transpose(0, 1).contiguous()
    y_res = x + mamba_out

    return {
        "U": rn["U"],
        "conv_out": conv_out,
        "dt": dt,
        "dA": da,
        "H_prev": h_prev,
        "H": h_state,
        "y": y_flat,
        "y_norm": y_norm,
        "Mamba_out": mamba_out,
        "Y": y_res,
    }


def moe_forward_ref(
    x: torch.Tensor,
    gamma_in: torch.Tensor,
    w_router: torch.Tensor,
    e_bias: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    w_up_s: torch.Tensor,
    w_down_s: torch.Tensor,
    eps: float,
    top_k: int = 6,
    routed_scaling: float = 2.5,
) -> dict[str, torch.Tensor]:
    """Granular MoE (128+1) forward pass matching NemotronHMOE + NemotronHTopkRouter + ReLU^2 NemotronHMLP."""
    d, T = x.shape
    rn = rmsnorm_ref(x, gamma_in, eps)
    u = rn["U"]  # [d, T]

    router_logits = F.linear(u.transpose(0, 1), w_router, bias=None).transpose(0, 1).contiguous()
    scores = torch.sigmoid(router_logits)
    scores_for_choice = scores + e_bias.unsqueeze(1)

    topk_indices = torch.zeros((T, top_k), dtype=torch.long)
    topk_raw_scores = torch.zeros((T, top_k), dtype=torch.float32)
    topk_weights = torch.zeros((T, top_k), dtype=torch.float32)
    y_routed = torch.zeros((d, T), dtype=torch.float32)

    for t in range(T):
        _, idx = torch.topk(scores_for_choice[:, t], k=top_k, largest=True, sorted=True)
        topk_indices[t] = idx
        unbiased = scores[idx, t]
        topk_raw_scores[t] = unbiased
        weights = (unbiased / (unbiased.sum() + 1e-20)) * routed_scaling
        topk_weights[t] = weights

        u_t = u[:, t]
        for j in range(top_k):
            e = int(idx[j].item())
            h = F.linear(u_t, w_up[e], bias=None)
            a = F.relu(h).pow(2)
            y_e = F.linear(a, w_down[e], bias=None)
            y_routed[:, t] = y_routed[:, t] + weights[j] * y_e

    shared_h = F.linear(u.transpose(0, 1), w_up_s, bias=None).transpose(0, 1).contiguous()
    shared_a = F.relu(shared_h).pow(2)
    y_shared = F.linear(shared_a.transpose(0, 1), w_down_s, bias=None).transpose(0, 1).contiguous()
    y_moe = y_routed + y_shared
    y_res = x + y_moe

    return {
        "U": u,
        "scores": scores,
        "scores_for_choice": scores_for_choice,
        "topk_indices": topk_indices,
        "topk_raw_scores": topk_raw_scores,
        "topk_weights": topk_weights,
        "shared_a": shared_a,
        "y_shared": y_shared,
        "y_routed": y_routed,
        "y_moe": y_moe,
        "Y": y_res,
    }


def lm_head_forward_ref(
    x_final: torch.Tensor,
    final_gamma: torch.Tensor,
    w_lm_head: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    fn = rmsnorm_ref(x_final, final_gamma, eps)
    logits = F.linear(fn["U"].transpose(0, 1), w_lm_head, bias=None).transpose(0, 1).contiguous()
    probs = F.softmax(logits, dim=0)
    return {
        "rms": fn["rms"],
        "X_norm": fn["X_norm"],
        "U": fn["U"],
        "logits": logits,
        "probs": probs,
    }


# ============================================================================
# SINGLE-LAYER COMPONENT FIXTURES
# ============================================================================

def build_rmsnorm_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, T = 8, 6
    x = torch.randn((d, T), generator=gen, dtype=torch.float32) * 0.9
    gamma = torch.tensor([0.91, 1.09, 0.95, 1.04, 0.88, 1.12, 0.97, 1.03], dtype=torch.float32)
    out_e5 = rmsnorm_ref(x, gamma, 1e-5)
    out_e6 = rmsnorm_ref(x, gamma, 1e-6)
    return {
        "d": d,
        "tokens": T,
        "X": to_f32_list(x),
        "gamma": to_f32_list(gamma),
        "eps_1e5": {
            "eps": 1e-5,
            "rms": to_f32_list(out_e5["rms"]),
            "X_norm": to_f32_list(out_e5["X_norm"]),
            "U": to_f32_list(out_e5["U"]),
        },
        "eps_1e6": {
            "eps": 1e-6,
            "rms": to_f32_list(out_e6["rms"]),
            "X_norm": to_f32_list(out_e6["X_norm"]),
            "U": to_f32_list(out_e6["U"]),
        },
    }


def build_embedding_and_lm_head_fixture(gen: torch.Generator) -> dict[str, Any]:
    vocab, d, T = 32, 8, 6
    eps = 1e-5
    w_embed = torch.randn((vocab, d), generator=gen, dtype=torch.float32)
    w_embed = w_embed / torch.sqrt(w_embed.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    token_ids = torch.tensor([2, 5, 2, 0, 11, 7], dtype=torch.long)
    x_embed = F.embedding(token_ids, w_embed).transpose(0, 1).contiguous()

    final_gamma = torch.tensor([0.93, 1.06, 0.98, 1.11, 0.89, 1.04, 0.96, 1.07], dtype=torch.float32)
    w_lm_head = torch.randn((vocab, d), generator=gen, dtype=torch.float32) * (1.15 / math.sqrt(d))
    x_final = torch.randn((d, T), generator=gen, dtype=torch.float32) * 1.1
    out = lm_head_forward_ref(x_final, final_gamma, w_lm_head, eps)

    return {
        "vocabSize": vocab,
        "d": d,
        "tokens": T,
        "eps": eps,
        "tokenIds": to_i32_list(token_ids),
        "W_embed": to_f32_list(w_embed),
        "X_embed": to_f32_list(x_embed),
        "X_final": to_f32_list(x_final),
        "finalNormGamma": to_f32_list(final_gamma),
        "finalNormRms": to_f32_list(out["rms"]),
        "finalNormU": to_f32_list(out["U"]),
        "W_lm_head": to_f32_list(w_lm_head),
        "logits": to_f32_list(out["logits"]),
        "probs": to_f32_list(out["probs"]),
    }


def build_gqa_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, n_q, n_kv, d_head, T = 8, 32, 2, 4, 6
    eps = 1e-5
    q_concat = n_q * d_head

    x = torch.randn((d, T), generator=gen, dtype=torch.float32) * 0.85
    gamma = torch.tensor([0.94, 1.07, 0.96, 1.03, 0.90, 1.09, 0.98, 1.05], dtype=torch.float32)
    w_q = torch.randn((q_concat, d), generator=gen, dtype=torch.float32) * (0.8 / math.sqrt(d))
    w_k = torch.randn((n_kv * d_head, d), generator=gen, dtype=torch.float32) * (0.8 / math.sqrt(d))
    w_v = torch.randn((n_kv * d_head, d), generator=gen, dtype=torch.float32) * (0.8 / math.sqrt(d))
    w_o = torch.randn((d, q_concat), generator=gen, dtype=torch.float32) * (0.5 / math.sqrt(q_concat))

    out = gqa_forward_ref(x, gamma, w_q, w_k, w_v, w_o, eps, n_q, n_kv, d_head)
    scores_json: list[float | None] = [
        None if math.isinf(v) else float(v)
        for v in out["Scores"].reshape(-1).tolist()
    ]

    return {
        "d": d,
        "nQ": n_q,
        "nKv": n_kv,
        "dHead": d_head,
        "tokens": T,
        "eps": eps,
        "X": to_f32_list(x),
        "gamma": to_f32_list(gamma),
        "W_Q": to_f32_list(w_q),
        "W_K": to_f32_list(w_k),
        "W_V": to_f32_list(w_v),
        "W_O": to_f32_list(w_o),
        "U": to_f32_list(out["U"]),
        "Q": to_f32_list(out["Q"]),
        "K": to_f32_list(out["K"]),
        "V": to_f32_list(out["V"]),
        "Scores": scores_json,
        "Softmax": to_f32_list(out["Softmax"]),
        "V_out": to_f32_list(out["V_out"]),
        "Attn_out": to_f32_list(out["Attn_out"]),
        "Y": to_f32_list(out["Y"]),
    }


def build_mamba2_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, n_heads, n_groups, d_head, d_state, k_conv, T = 8, 16, 8, 4, 4, 4, 6
    d_inner = n_heads * d_head
    conv_dim = d_inner + 2 * n_groups * d_state
    in_proj_dim = d_inner + conv_dim + n_heads
    eps = 1e-5

    x = torch.randn((d, T), generator=gen, dtype=torch.float32) * 0.8
    gamma_in = torch.tensor([0.92, 1.08, 0.95, 1.06, 0.90, 1.10, 0.97, 1.02], dtype=torch.float32)
    w_in = torch.randn((in_proj_dim, d), generator=gen, dtype=torch.float32) * (0.52 / math.sqrt(d))
    w_conv = torch.randn((conv_dim, k_conv), generator=gen, dtype=torch.float32) * 0.35
    b_conv = torch.randn((conv_dim,), generator=gen, dtype=torch.float32) * 0.1
    a_log = torch.linspace(-1.5, 0.3, n_heads, dtype=torch.float32)
    dt_bias = torch.linspace(-0.7, 0.2, n_heads, dtype=torch.float32)
    d_skip = torch.linspace(0.45, 0.90, n_heads, dtype=torch.float32)
    gamma_gate = torch.linspace(0.88, 1.12, d_inner, dtype=torch.float32)
    w_out = torch.randn((d, d_inner), generator=gen, dtype=torch.float32) * (0.3 / math.sqrt(d_inner))

    out = mamba2_forward_ref(
        x, gamma_in, w_in, w_conv, b_conv, a_log, dt_bias, d_skip, gamma_gate, w_out, eps,
    )

    return {
        "d": d,
        "tokens": T,
        "eps": eps,
        "X": to_f32_list(x),
        "gamma_in": to_f32_list(gamma_in),
        "W_in": to_f32_list(w_in),
        "w_conv": to_f32_list(w_conv),
        "b_conv": to_f32_list(b_conv),
        "A_log": to_f32_list(a_log),
        "dt_bias": to_f32_list(dt_bias),
        "D": to_f32_list(d_skip),
        "gamma_gate": to_f32_list(gamma_gate),
        "W_out": to_f32_list(w_out),
        "U": to_f32_list(out["U"]),
        "conv_out": to_f32_list(out["conv_out"]),
        "dt": to_f32_list(out["dt"]),
        "dA": to_f32_list(out["dA"]),
        "H_prev": to_f32_list(out["H_prev"]),
        "H": to_f32_list(out["H"]),
        "y": to_f32_list(out["y"]),
        "y_norm": to_f32_list(out["y_norm"]),
        "Mamba_out": to_f32_list(out["Mamba_out"]),
        "Y": to_f32_list(out["Y"]),
    }


def build_moe_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, n_exp, top_k, d_exp, d_shared, T = 8, 128, 6, 4, 8, 6
    eps = 1e-5

    x = torch.randn((d, T), generator=gen, dtype=torch.float32) * 0.85
    gamma_in = torch.tensor([0.91, 1.05, 0.97, 1.09, 0.88, 1.11, 0.96, 1.04], dtype=torch.float32)
    w_router = torch.randn((n_exp, d), generator=gen, dtype=torch.float32)
    w_router = w_router * (1.45 / torch.sqrt(w_router.pow(2).sum(dim=-1, keepdim=True) + 1e-6))
    e_bias = torch.randn((n_exp,), generator=gen, dtype=torch.float32) * 0.14

    w_up = torch.randn((n_exp, d_exp, d), generator=gen, dtype=torch.float32) * (0.72 / math.sqrt(d))
    w_down = torch.randn((n_exp, d, d_exp), generator=gen, dtype=torch.float32) * (0.42 / math.sqrt(d_exp))
    w_up_s = torch.randn((d_shared, d), generator=gen, dtype=torch.float32) * (0.72 / math.sqrt(d))
    w_down_s = torch.randn((d, d_shared), generator=gen, dtype=torch.float32) * (0.42 / math.sqrt(d_shared))

    out = moe_forward_ref(x, gamma_in, w_router, e_bias, w_up, w_down, w_up_s, w_down_s, eps, top_k)

    return {
        "d": d,
        "nRoutedExperts": n_exp,
        "topK": top_k,
        "dExp": d_exp,
        "dShared": d_shared,
        "tokens": T,
        "eps": eps,
        "X": to_f32_list(x),
        "gamma_in": to_f32_list(gamma_in),
        "W_router": to_f32_list(w_router),
        "e_score_correction_bias": to_f32_list(e_bias),
        "W_up": to_f32_list(w_up),
        "W_down": to_f32_list(w_down),
        "W_up_s": to_f32_list(w_up_s),
        "W_down_s": to_f32_list(w_down_s),
        "U": to_f32_list(out["U"]),
        "scores": to_f32_list(out["scores"]),
        "scores_for_choice": to_f32_list(out["scores_for_choice"]),
        "topk_indices": to_i32_list(out["topk_indices"]),
        "topk_raw_scores": to_f32_list(out["topk_raw_scores"]),
        "topk_weights": to_f32_list(out["topk_weights"]),
        "shared_a": to_f32_list(out["shared_a"]),
        "y_shared": to_f32_list(out["y_shared"]),
        "y_routed": to_f32_list(out["y_routed"]),
        "y_moe": to_f32_list(out["y_moe"]),
        "Y": to_f32_list(out["Y"]),
    }


# ============================================================================
# COMPLETE 52-LAYER END-TO-END PYTORCH REFERENCE EXECUTION
# ============================================================================

def build_52layer_end_to_end_fixture(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_bytes = Path(manifest["weightsFile"]).read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != manifest["weightsSha256"]:
        raise RuntimeError(
            f"Weight file SHA-256 mismatch: {actual_sha256} != {manifest['weightsSha256']}"
        )

    floats = array.array("f")
    floats.frombytes(raw_bytes)
    if floats.itemsize != 4:
        raise RuntimeError(f"Expected 4-byte float32 items, got {floats.itemsize}")
    flat_tensor = torch.tensor(floats, dtype=torch.float32)

    tensor_map: dict[str, torch.Tensor] = {}
    for entry in manifest["tensors"]:
        off = entry["offsetFloats"]
        length = entry["length"]
        tensor_map[entry["name"]] = flat_tensor[off : off + length]

    d = int(manifest["d"])
    vocab_size = int(manifest["vocabSize"])
    eps = float(manifest["rmsNormEps"])
    pattern: str = manifest["pattern"]
    num_layers = len(pattern)

    # Non-trivial token sequence with repeated token ID 2 at t=0 and t=2 (exercises NoPE + Mamba-2 order)
    token_ids_list = [2, 5, 2, 11, 0, 19]
    token_ids = torch.tensor(token_ids_list, dtype=torch.long)

    w_embed = tensor_map["W_embed"].view(vocab_size, d)
    final_gamma = tensor_map["finalNormGamma"].view(d)
    w_lm_head = tensor_map["W_lm_head"].view(vocab_size, d)

    # Stage 0: Untied Embedding (NoPE)
    x_curr = F.embedding(token_ids, w_embed).transpose(0, 1).contiguous()  # [d, T]
    residual_streams: list[list[float]] = [to_f32_list(x_curr)]
    moe_topk_by_bay: dict[str, list[int]] = {}

    # Stages 1..52: Exact HYBRID_OVERRIDE_PATTERN reusing the shared reference operations
    for bay_id in range(1, num_layers + 1):
        kind = pattern[bay_id - 1]
        p = f"bay_{bay_id:02d}"
        if kind == "M":
            out = mamba2_forward_ref(
                x_curr,
                gamma_in=tensor_map[f"{p}.gamma_in"].view(d),
                w_in=tensor_map[f"{p}.W_in"].view(208, d),
                w_conv=tensor_map[f"{p}.w_conv"].view(128, 4),
                b_conv=tensor_map[f"{p}.b_conv"].view(128),
                a_log=tensor_map[f"{p}.A_log"].view(16),
                dt_bias=tensor_map[f"{p}.dt_bias"].view(16),
                d_skip=tensor_map[f"{p}.D"].view(16),
                gamma_gate=tensor_map[f"{p}.gamma_gate"].view(64),
                w_out=tensor_map[f"{p}.W_out"].view(d, 64),
                eps=eps,
            )
            x_curr = out["Y"]
        elif kind == "E":
            out = moe_forward_ref(
                x_curr,
                gamma_in=tensor_map[f"{p}.gamma_in"].view(d),
                w_router=tensor_map[f"{p}.W_router"].view(128, d),
                e_bias=tensor_map[f"{p}.e_score_correction_bias"].view(128),
                w_up=tensor_map[f"{p}.W_up"].view(128, 4, d),
                w_down=tensor_map[f"{p}.W_down"].view(128, d, 4),
                w_up_s=tensor_map[f"{p}.W_up_s"].view(8, d),
                w_down_s=tensor_map[f"{p}.W_down_s"].view(d, 8),
                eps=eps,
            )
            moe_topk_by_bay[str(bay_id)] = to_i32_list(out["topk_indices"])
            x_curr = out["Y"]
        elif kind == "*":
            out = gqa_forward_ref(
                x_curr,
                gamma=tensor_map[f"{p}.gamma"].view(d),
                w_q=tensor_map[f"{p}.W_Q"].view(128, d),
                w_k=tensor_map[f"{p}.W_K"].view(8, d),
                w_v=tensor_map[f"{p}.W_V"].view(8, d),
                w_o=tensor_map[f"{p}.W_O"].view(d, 128),
                eps=eps,
            )
            x_curr = out["Y"]
        else:
            raise ValueError(f"Unexpected layer kind {kind!r} at bay {bay_id}")

        residual_streams.append(to_f32_list(x_curr))

    # Stage 53 + 54: Final RMSNorm (eps = 1e-5) + Untied LM Head + Softmax
    lm = lm_head_forward_ref(x_curr, final_gamma, w_lm_head, eps)
    top5_by_token: list[list[int]] = []
    for t in range(len(token_ids_list)):
        _, top5 = torch.topk(lm["logits"][:, t], k=5, largest=True, sorted=True)
        top5_by_token.append([int(v) for v in top5.tolist()])

    return {
        "metadata": {
            "generator": "scripts/reference/generate_pytorch_parity_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "referenceArchitecture": "NVIDIA NemotronH 52-Layer Miniature Hybrid (Apache-2.0 equations)",
            "precisionNote": (
                "Full 52-layer forward pass executed in PyTorch float32 over the exact exported IEEE-754 "
                "Float32 parameters of createDefaultHybridModel() (verified by SHA-256). TypeScript "
                "accumulates inner dot products in JS float64 prior to Math.fround writes while PyTorch "
                "accumulates in float32; cumulative drift across 52 sequential layers remains bounded <= 2e-5."
            ),
            "tolerances": {
                "embeddingMaxAbs": 0.0,
                "residualCheckpointMaxAbs": 2e-5,
                "finalNormMaxAbs": 2e-5,
                "lmHeadLogitsMaxAbs": 5e-5,
                "lmHeadProbsMaxAbs": 2e-5,
            },
        },
        "modelId": manifest["modelId"],
        "parameterFingerprint": manifest["parameterFingerprint"],
        "weightsSha256": actual_sha256,
        "totalWeightFloats": manifest["totalFloats"],
        "pattern": pattern,
        "d": d,
        "vocabSize": vocab_size,
        "rmsNormEps": eps,
        "tokenIds": token_ids_list,
        "residualStreams": residual_streams,
        "moeTopkByBay": moe_topk_by_bay,
        "finalNorm": {
            "rms": to_f32_list(lm["rms"]),
            "X_norm": to_f32_list(lm["X_norm"]),
            "U": to_f32_list(lm["U"]),
        },
        "lmHead": {
            "logits": to_f32_list(lm["logits"]),
            "probs": to_f32_list(lm["probs"]),
            "top5TokenIdsByToken": top5_by_token,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nemotron 3 Nano PyTorch parity fixtures.")
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=None,
        help="Path to exported 52-layer model manifest from scripts/reference/export_52layer_params.mjs",
    )
    args = parser.parse_args()

    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Expected pinned PyTorch {PINNED_TORCH_VERSION}, got {torch.__version__}. "
            "Run via `uv run scripts/reference/generate_pytorch_parity_fixtures.py`."
        )

    torch.manual_seed(20261124)
    gen = torch.Generator().manual_seed(20261124)

    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    single_layer_payload: dict[str, Any] = {
        "metadata": {
            "generator": "scripts/reference/generate_pytorch_parity_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "referenceArchitecture": "NVIDIA NemotronH (Nemotron 3 Nano 30B-A3B, Apache-2.0 reference equations)",
            "precisionNote": (
                "Parameters and inputs are identical IEEE-754 Float32 values across PyTorch and TypeScript. "
                "TypeScript accumulates inner dot products in JS float64 before Math.fround at tensor writes, "
                "whereas PyTorch torch.float32 accumulates in float32 BLAS/SIMD; expected max abs parity error <= 1e-5."
            ),
            "tolerances": {
                "rmsnormMaxAbs": 1e-5,
                "embeddingMaxAbs": 0.0,
                "lmHeadLogitsMaxAbs": 1e-5,
                "lmHeadProbsMaxAbs": 1e-5,
                "gqaMaxAbs": 1e-5,
                "mamba2MaxAbs": 2e-5,
                "moeMaxAbs": 2e-5,
            },
        },
        "rmsnorm": build_rmsnorm_fixture(gen),
        "embeddingAndLmHead": build_embedding_and_lm_head_fixture(gen),
        "gqa": build_gqa_fixture(gen),
        "mamba2": build_mamba2_fixture(gen),
        "moe": build_moe_fixture(gen),
    }

    single_out = fixtures_dir / "pytorch-reference-parity.json"
    single_out.write_text(json.dumps(single_layer_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote single-layer PyTorch parity fixtures ({single_out.stat().st_size:,} bytes) -> {single_out}")

    if args.model_manifest is not None:
        e2e_payload = build_52layer_end_to_end_fixture(args.model_manifest)
        e2e_out = fixtures_dir / "pytorch-52layer-parity.json"
        e2e_out.write_text(json.dumps(e2e_payload, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"Wrote 52-layer end-to-end PyTorch parity fixtures ({e2e_out.stat().st_size:,} bytes) -> {e2e_out}")


if __name__ == "__main__":
    main()
