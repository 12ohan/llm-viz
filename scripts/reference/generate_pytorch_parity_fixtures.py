#!/usr/bin/env python3
"""
Independent PyTorch Reference Parity Fixture Generator for Nemotron 3 Nano Miniature Simulators.

Development/reference script ONLY — NOT a browser runtime dependency.
Run via:
  uv run --python 3.12 --with torch python scripts/reference/generate_pytorch_parity_fixtures.py

Mathematical Reference & Licensing Note:
  Implements NVIDIA NemotronH architecture equations (`NemotronHRMSNorm`, `NemotronHMamba2Mixer` +
  `MambaRMSNormGated` with `norm_before_gate=False`, `NemotronHMOE` + `NemotronHTopkRouter` +
  non-gated `ReLU^2` `NemotronHMLP`, and NoPE `32Q/2KV` GQA Attention, plus Untied Embeddings and
  Untied LM Head) using clean-room standard PyTorch (`torch`, `torch.nn.functional`).
  Reference specification: Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  (`modeling_nemotron_h.py`, Apache License 2.0; model weights separately under NVIDIA Open Model
  License — no external weights or third-party code are copied here).

Exports identical IEEE-754 Float32 synthetic parameters, inputs, and PyTorch reference outputs to
`tests/fixtures/pytorch-reference-parity.json` so Vitest validates TypeScript numerical parity
independently with zero Python runtime requirement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def to_f32_list(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().to(torch.float32).reshape(-1).tolist()]


def to_i32_list(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.detach().to(torch.int64).reshape(-1).tolist()]


def rmsnorm_ref(x_d_t: torch.Tensor, gamma: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    """x_d_t: [d, T], gamma: [d] -> matches NemotronHRMSNorm over channel dim 0."""
    # Transpose to [T, d] to match standard PyTorch last-dim RMSNorm
    x = x_d_t.transpose(0, 1)  # [T, d]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # [T, 1]
    x_norm = x / rms
    u = x_norm * gamma.unsqueeze(0)
    return {
        "rms": rms.squeeze(-1),               # [T]
        "X_norm": x_norm.transpose(0, 1),     # [d, T]
        "U": u.transpose(0, 1),               # [d, T]
    }


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

    # F.embedding -> [T, d], transpose to [d, T]
    x_embed = F.embedding(token_ids, w_embed).transpose(0, 1).contiguous()

    # Final RMSNorm + strictly untied LM Head
    final_gamma = torch.tensor([0.93, 1.06, 0.98, 1.11, 0.89, 1.04, 0.96, 1.07], dtype=torch.float32)
    w_lm_head = torch.randn((vocab, d), generator=gen, dtype=torch.float32) * (1.15 / math.sqrt(d))
    x_final = torch.randn((d, T), generator=gen, dtype=torch.float32) * 1.1
    fn = rmsnorm_ref(x_final, final_gamma, eps)
    # F.linear(U_final^T [T, d], W_lm_head [V, d]) -> [T, V] -> transpose [V, T]
    logits = F.linear(fn["U"].transpose(0, 1), w_lm_head, bias=None).transpose(0, 1).contiguous()
    probs = F.softmax(logits, dim=0)

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
        "finalNormRms": to_f32_list(fn["rms"]),
        "finalNormU": to_f32_list(fn["U"]),
        "W_lm_head": to_f32_list(w_lm_head),
        "logits": to_f32_list(logits),
        "probs": to_f32_list(probs),
    }


def build_gqa_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, n_q, n_kv, d_head, T = 8, 32, 2, 4, 6
    eps = 1e-5
    q_concat = n_q * d_head  # 128

    x = torch.randn((d, T), generator=gen, dtype=torch.float32) * 0.85
    gamma = torch.tensor([0.94, 1.07, 0.96, 1.03, 0.90, 1.09, 0.98, 1.05], dtype=torch.float32)
    w_q = torch.randn((q_concat, d), generator=gen, dtype=torch.float32) * (0.8 / math.sqrt(d))
    w_k = torch.randn((n_kv * d_head, d), generator=gen, dtype=torch.float32) * (0.8 / math.sqrt(d))
    w_v = torch.randn((n_kv * d_head, d), generator=gen, dtype=torch.float32) * (0.8 / math.sqrt(d))
    w_o = torch.randn((d, q_concat), generator=gen, dtype=torch.float32) * (0.5 / math.sqrt(q_concat))

    rn = rmsnorm_ref(x, gamma, eps)
    u_t_d = rn["U"].transpose(0, 1)  # [T, d]

    q_flat = F.linear(u_t_d, w_q, bias=None).transpose(0, 1).contiguous()  # [128, T]
    k_flat = F.linear(u_t_d, w_k, bias=None).transpose(0, 1).contiguous()  # [8, T]
    v_flat = F.linear(u_t_d, w_v, bias=None).transpose(0, 1).contiguous()  # [8, T]

    # [n_q, T, d_head] and repeated KV [n_q, T, d_head] (16:1 GQA sharing, NoPE)
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

    # Replace -inf with null in JSON (JSON has no -Infinity)
    scores_json: list[float | None] = [
        None if math.isinf(v) else float(v)
        for v in scores_masked.reshape(-1).tolist()
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
        "U": to_f32_list(rn["U"]),
        "Q": to_f32_list(q_flat),
        "K": to_f32_list(k_flat),
        "V": to_f32_list(v_flat),
        "Scores": scores_json,
        "Softmax": to_f32_list(softmax),
        "V_out": to_f32_list(v_out_flat),
        "Attn_out": to_f32_list(attn_out),
        "Y": to_f32_list(y),
    }


def build_mamba2_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, n_heads, n_groups, d_head, d_state, k_conv, T = 8, 16, 8, 4, 4, 4, 6
    d_inner = n_heads * d_head          # 64
    conv_dim = d_inner + 2 * n_groups * d_state  # 128
    in_proj_dim = d_inner + conv_dim + n_heads   # 208
    group_size = d_inner // n_groups    # 8
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

    rn = rmsnorm_ref(x, gamma_in, eps)
    proj = F.linear(rn["U"].transpose(0, 1), w_in, bias=None).transpose(0, 1).contiguous()  # [208, T]
    z = proj[:d_inner]                           # [64, T]
    xbc_pre = proj[d_inner : d_inner + conv_dim] # [128, T]
    dt_raw = proj[d_inner + conv_dim :]          # [16, T]

    # Causal depthwise Conv1D (left-pad k-1 = 3 zeros, groups = conv_dim)
    xbc_padded = F.pad(xbc_pre.unsqueeze(0), (k_conv - 1, 0))  # [1, 128, T+3]
    conv_out = F.conv1d(xbc_padded, w_conv.unsqueeze(1), b_conv, groups=conv_dim).squeeze(0)  # [128, T]
    xbc = F.silu(conv_out)

    x_ssm = xbc[:d_inner].view(n_heads, d_head, T)
    b_ssm = xbc[d_inner : d_inner + n_groups * d_state].view(n_groups, d_state, T)
    c_ssm = xbc[d_inner + n_groups * d_state :].view(n_groups, d_state, T)

    a_decay = -torch.exp(a_log)                             # [16]
    dt = F.softplus(dt_raw + dt_bias.unsqueeze(1))          # [16, T]
    da = torch.exp(dt * a_decay.unsqueeze(1))               # [16, T]

    # Recurrent SSM state update matching NemotronHMamba2Mixer
    h_state = torch.zeros((n_heads, d_head, d_state), dtype=torch.float32)
    h_prev = torch.zeros_like(h_state)
    y_ssm = torch.zeros((n_heads, d_head, T), dtype=torch.float32)
    heads_per_group = n_heads // n_groups

    for t in range(T):
        if t == T - 1:
            h_prev = h_state.clone()
        b_rep = b_ssm[:, :, t].repeat_interleave(heads_per_group, dim=0)  # [16, 4]
        c_rep = c_ssm[:, :, t].repeat_interleave(heads_per_group, dim=0)  # [16, 4]
        da_t = da[:, t].view(n_heads, 1, 1)
        dt_t = dt[:, t].view(n_heads, 1, 1)
        x_t = x_ssm[:, :, t].unsqueeze(-1)  # [16, 4, 1]
        h_state = da_t * h_state + dt_t * (x_t * b_rep.unsqueeze(1))
        y_ssm[:, :, t] = torch.einsum("hpn,hn->hp", h_state, c_rep)

    y_total = y_ssm + d_skip.view(n_heads, 1, 1) * x_ssm  # [16, 4, T]
    y_flat = y_total.view(d_inner, T)

    # MambaRMSNormGated (norm_before_gate = False): gate FIRST -> 8-group RMSNorm -> * gamma_gate
    gated = y_flat * F.silu(z)  # [64, T]
    gated_grp = gated.view(n_groups, group_size, T)
    rms_gate = torch.sqrt(gated_grp.pow(2).mean(dim=1, keepdim=True) + eps)  # [8, 1, T]
    y_norm_pre = (gated_grp / rms_gate).view(d_inner, T)
    y_norm = y_norm_pre * gamma_gate.unsqueeze(1)

    mamba_out = F.linear(y_norm.transpose(0, 1), w_out, bias=None).transpose(0, 1).contiguous()
    y_res = x + mamba_out

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
        "U": to_f32_list(rn["U"]),
        "conv_out": to_f32_list(conv_out),
        "dt": to_f32_list(dt),
        "dA": to_f32_list(da),
        "H_prev": to_f32_list(h_prev),
        "H": to_f32_list(h_state),
        "y": to_f32_list(y_flat),
        "y_norm": to_f32_list(y_norm),
        "Mamba_out": to_f32_list(mamba_out),
        "Y": to_f32_list(y_res),
    }


def build_moe_fixture(gen: torch.Generator) -> dict[str, Any]:
    d, n_exp, top_k, d_exp, d_shared, T = 8, 128, 6, 4, 8, 6
    eps = 1e-5
    routed_scaling = 2.5

    x = torch.randn((d, T), generator=gen, dtype=torch.float32) * 0.85
    gamma_in = torch.tensor([0.91, 1.05, 0.97, 1.09, 0.88, 1.11, 0.96, 1.04], dtype=torch.float32)
    w_router = torch.randn((n_exp, d), generator=gen, dtype=torch.float32)
    w_router = w_router * (1.45 / torch.sqrt(w_router.pow(2).sum(dim=-1, keepdim=True) + 1e-6))
    e_bias = torch.randn((n_exp,), generator=gen, dtype=torch.float32) * 0.14

    w_up = torch.randn((n_exp, d_exp, d), generator=gen, dtype=torch.float32) * (0.72 / math.sqrt(d))
    w_down = torch.randn((n_exp, d, d_exp), generator=gen, dtype=torch.float32) * (0.42 / math.sqrt(d_exp))
    w_up_s = torch.randn((d_shared, d), generator=gen, dtype=torch.float32) * (0.72 / math.sqrt(d))
    w_down_s = torch.randn((d, d_shared), generator=gen, dtype=torch.float32) * (0.42 / math.sqrt(d_shared))

    rn = rmsnorm_ref(x, gamma_in, eps)
    u = rn["U"]  # [8, T]

    # NemotronHTopkRouter: sigmoid + e_score_correction_bias for SELECTION ONLY; unbiased sigmoid for WEIGHTS
    router_logits = F.linear(u.transpose(0, 1), w_router, bias=None).transpose(0, 1).contiguous()  # [128, T]
    scores = torch.sigmoid(router_logits)                           # [128, T]
    scores_for_choice = scores + e_bias.unsqueeze(1)                # [128, T]

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

    # Always-on 2x shared expert (non-gated ReLU^2, unscaled 1.0x)
    shared_h = F.linear(u.transpose(0, 1), w_up_s, bias=None).transpose(0, 1).contiguous()
    shared_a = F.relu(shared_h).pow(2)
    y_shared = F.linear(shared_a.transpose(0, 1), w_down_s, bias=None).transpose(0, 1).contiguous()
    y_moe = y_routed + y_shared
    y_res = x + y_moe

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
        "U": to_f32_list(u),
        "scores": to_f32_list(scores),
        "scores_for_choice": to_f32_list(scores_for_choice),
        "topk_indices": to_i32_list(topk_indices),
        "topk_raw_scores": to_f32_list(topk_raw_scores),
        "topk_weights": to_f32_list(topk_weights),
        "shared_a": to_f32_list(shared_a),
        "y_shared": to_f32_list(y_shared),
        "y_routed": to_f32_list(y_routed),
        "y_moe": to_f32_list(y_moe),
        "Y": to_f32_list(y_res),
    }


def main() -> None:
    torch.manual_seed(20261124)
    gen = torch.Generator().manual_seed(20261124)

    payload: dict[str, Any] = {
        "metadata": {
            "generator": "scripts/reference/generate_pytorch_parity_fixtures.py",
            "torchVersion": torch.__version__,
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

    out_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "pytorch-reference-parity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote PyTorch parity fixtures ({out_path.stat().st_size:,} bytes) -> {out_path}")


if __name__ == "__main__":
    main()
