#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator for
Nemotron-Labs-TwoTower Miniature Simulators.

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce all 3 committed TwoTower parity fixtures covering all 8 mandatory categories:
  node scripts/reference/export_twotower_params.mjs
  uv run scripts/reference/generate_twotower_pytorch_fixtures.py

Clean-Room Mathematical Reference & Licensing Note:
  Implements Nemotron-Labs-TwoTower's mathematical equations (sinusoidal timestep embedding,
  PixArt-alpha adaLN-single conditioning with split M/* modulate->RMSNorm vs. E RMSNorm->modulate
  ordering and residual gating, seeded causal Conv1D + FP32 Mamba-2 SSM scan + MambaRMSNormGated,
  bidirectional rectangular [B x (P+B)] and causal GQA via torch.nn.functional.scaled_dot_product_attention,
  FP64 MoE router selection & unbiased x2.5 weighting with ReLU^2 routed + 2x shared experts,
  unmodulated final RMSNorm + untied LM head, Context-to-Denoiser read-only handoff, AR vs. Mock-AR
  alignment, and MDLM/SUBS confidence-unmasking + Gumbel-max sampling) strictly from first principles
  using standard PyTorch (`torch`, `torch.nn.functional`).
  Zero third-party source code is copied and zero production checkpoint weights are touched;
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
DEFAULT_PARAMS_DIR = Path("/tmp/nemotron-viz-twotower-ref")
MDLM_MASK_SUPPRESS_LOGIT = -1e12
GUMBEL_UNIFORM_MIN = 1e-10


# ============================================================================
# STRICT FINITE-VALUE GUARDS & JSON SERIALIZERS
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


def to_nullable_causal_scores(scores: torch.Tensor, label: str) -> list[float | None]:
    flat = scores.detach().to(torch.float32).reshape(-1).tolist()
    out: list[float | None] = []
    for idx, v in enumerate(flat):
        if math.isnan(v) or (math.isinf(v) and v > 0):
            raise RuntimeError(f"Unexpected non-finite score {v} at {label}[{idx}]")
        out.append(None if math.isinf(v) else float(v))
    return out


def to_nullable_confidence(conf: list[float], label: str) -> list[float | None]:
    out: list[float | None] = []
    for idx, v in enumerate(conf):
        if math.isnan(v) or (math.isinf(v) and v < 0):
            raise RuntimeError(f"Unexpected invalid confidence {v} at {label}[{idx}]")
        out.append(None if math.isinf(v) else float(v))
    return out


# ============================================================================
# EXPORTED WEIGHT LOADING WITH CRYPTOGRAPHIC SHA-256 VERIFICATION
# ============================================================================

def load_exported_twotower_weights(manifest_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing exported TwoTower manifest: {manifest_path}. "
            "Run `node scripts/reference/export_twotower_params.mjs` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = Path(manifest["weightsFile"])
    raw_bytes = weights_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != manifest["weightsSha256"]:
        raise RuntimeError(
            f"TwoTower weight SHA-256 mismatch for {manifest_path}: "
            f"{actual_sha256} != {manifest['weightsSha256']}"
        )

    floats = array.array("f")
    floats.frombytes(raw_bytes)
    if sys.byteorder != "little":
        floats.byteswap()
    if floats.itemsize != 4:
        raise RuntimeError(f"Expected 4-byte float32 items, got {floats.itemsize}")
    flat_tensor = torch.tensor(floats, dtype=torch.float32)
    assert_all_finite(flat_tensor, f"exported_weights:{manifest['variant']}")

    tensor_map: dict[str, torch.Tensor] = {}
    for entry in manifest["tensors"]:
        off = int(entry["offsetFloats"])
        length = int(entry["length"])
        tensor_map[entry["name"]] = flat_tensor[off : off + length]

    if len(tensor_map) != int(manifest["tensorCount"]):
        raise RuntimeError(
            f"Tensor count mismatch: loaded {len(tensor_map)} != manifest {manifest['tensorCount']}"
        )
    return manifest, tensor_map


# ============================================================================
# CLEAN-ROOM PYTORCH 2.15.0 TWOTOWER REFERENCE EQUATIONS
# ============================================================================

def rmsnorm_ref(x_d_t: torch.Tensor, gamma: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    """NemotronHRMSNorm over channel dim 0 for channel-major [d, T] tensors."""
    x = x_d_t.transpose(0, 1)  # [T, d]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # [T, 1]
    x_norm = x / rms
    u = x_norm * gamma.unsqueeze(0)
    out = {
        "rms": rms.squeeze(-1),            # [T]
        "X_norm": x_norm.transpose(0, 1).contiguous(),  # [d, T]
        "U": u.transpose(0, 1).contiguous(),            # [d, T]
    }
    assert_all_finite(out["U"], "rmsnorm.U")
    return out


def timestep_embedding_ref(
    t: float,
    freq_dim: int = 16,
    max_period_scale: float = 1000.0,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (`t * 1000`, `max_period = 10000`, FP32 freqs,
    cos-first `[cos, sin]`).
    """
    half = freq_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32)
        / half
    )
    args = (torch.tensor(float(t), dtype=torch.float32) * float(max_period_scale)) * freqs
    phi = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).contiguous()
    assert_all_finite(phi, "timestep_embedding.phi")
    return phi


def t_embedder_ref(
    phi: torch.Tensor,
    mlp0_w: torch.Tensor,
    mlp0_b: torch.Tensor,
    mlp2_w: torch.Tensor,
    mlp2_b: torch.Tensor,
) -> torch.Tensor:
    h0 = F.silu(F.linear(phi, mlp0_w, mlp0_b))
    t_repr = F.linear(h0, mlp2_w, mlp2_b)
    assert_all_finite(t_repr, "t_embedder.t_repr")
    return t_repr


def t_block_ref(
    t_repr: torch.Tensor,
    t_block_w: torch.Tensor,
    t_block_b: torch.Tensor,
) -> torch.Tensor:
    t_emb = F.linear(F.silu(t_repr), t_block_w, t_block_b)
    assert_all_finite(t_emb, "t_block.t_emb")
    return t_emb


def layer_mod_params_ref(
    t_emb: torch.Tensor,
    scale_shift_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    combined = scale_shift_table + t_emb
    assert_all_finite(combined, "layer_mod_params.combined")
    shift, scale, gate = combined.chunk(3, dim=0)
    return shift.contiguous(), scale.contiguous(), gate.contiguous()


def modulate_ref(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Channel-wise AdaLN affine modulation on [d, T]: x * (1 + scale) + shift."""
    out = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    assert_all_finite(out, "modulate.out")
    return out


def compute_global_timestep_ref(
    t: float,
    tensor_map: dict[str, torch.Tensor],
    d: int,
    freq_dim: int,
    max_period_scale: float = 1000.0,
    max_period: float = 10000.0,
) -> dict[str, torch.Tensor]:
    phi = timestep_embedding_ref(t, freq_dim, max_period_scale, max_period)
    t_repr = t_embedder_ref(
        phi,
        mlp0_w=tensor_map["time_cond.t_embedder.mlp.0.weight"].view(d, freq_dim),
        mlp0_b=tensor_map["time_cond.t_embedder.mlp.0.bias"].view(d),
        mlp2_w=tensor_map["time_cond.t_embedder.mlp.2.weight"].view(d, d),
        mlp2_b=tensor_map["time_cond.t_embedder.mlp.2.bias"].view(d),
    )
    t_emb = t_block_ref(
        t_repr,
        t_block_w=tensor_map["time_cond.t_block.1.weight"].view(3 * d, d),
        t_block_b=tensor_map["time_cond.t_block.1.bias"].view(3 * d),
    )
    return {"phi": phi, "tRepr": t_repr, "tEmb": t_emb}


def mamba2_ref(
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
    conv_state_in: torch.Tensor | None = None,
    ssm_state_in: torch.Tensor | None = None,
    pre_norm_mod: tuple[torch.Tensor, torch.Tensor] | None = None,
    residual_gate: torch.Tensor | None = None,
    invert_mod_order: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Mamba-2 forward pass with verified TwoTower AdaLN ordering (modulate -> RMSNorm),
    seeded causal F.conv1d (d_conv - 1 = 3 taps), FP32 recurrent scan, MambaRMSNormGated
    (norm_before_gate = False), and gated residual X + gate * y_mamba.
    """
    T = x.shape[1]
    d_inner = n_heads * d_head                    # 64
    conv_dim = d_inner + 2 * n_groups * d_state   # 128
    group_size = d_inner // n_groups              # 8

    x_mod: torch.Tensor | None = None
    if pre_norm_mod is not None:
        shift, scale = pre_norm_mod
        if not invert_mod_order:
            # Authoritative TwoTower M/* ordering: modulate -> RMSNorm
            x_mod = modulate_ref(x, shift, scale)
            rn = rmsnorm_ref(x_mod, gamma_in, eps)
            u_in = rn["U"]
        else:
            # Inverted diagnostic ordering ONLY for Cat 1 order-sensitivity check
            rn = rmsnorm_ref(x, gamma_in, eps)
            u_in = modulate_ref(rn["U"], shift, scale)
    else:
        rn = rmsnorm_ref(x, gamma_in, eps)
        u_in = rn["U"]

    proj = F.linear(u_in.transpose(0, 1), w_in, bias=None).transpose(0, 1).contiguous()
    z = proj[:d_inner]
    xbc_pre = proj[d_inner : d_inner + conv_dim]
    dt_raw = proj[d_inner + conv_dim :]

    if conv_state_in is not None:
        prefix_taps = conv_state_in[:, 1:]  # last d_conv - 1 = 3 prior taps
    else:
        prefix_taps = torch.zeros((conv_dim, k_conv - 1), dtype=torch.float32)

    xbc_padded = torch.cat([prefix_taps, xbc_pre], dim=1)  # [128, 3 + T]
    conv_out = F.conv1d(
        xbc_padded.unsqueeze(0),
        w_conv.unsqueeze(1),
        b_conv,
        groups=conv_dim,
    ).squeeze(0)
    xbc = F.silu(conv_out)

    if conv_state_in is not None:
        conv_hist = torch.cat([conv_state_in, xbc_pre], dim=1)
    else:
        conv_hist = torch.cat([torch.zeros((conv_dim, k_conv), dtype=torch.float32), xbc_pre], dim=1)
    conv_state_out = conv_hist[:, -k_conv:].contiguous()

    x_ssm = xbc[:d_inner].view(n_heads, d_head, T)
    b_ssm = xbc[d_inner : d_inner + n_groups * d_state].view(n_groups, d_state, T)
    c_ssm = xbc[d_inner + n_groups * d_state :].view(n_groups, d_state, T)

    a_decay = -torch.exp(a_log.to(torch.float32))
    dt = F.softplus(dt_raw.to(torch.float32) + dt_bias.to(torch.float32).unsqueeze(1))
    da = torch.exp(dt * a_decay.unsqueeze(1))

    h_state = (
        ssm_state_in.clone().to(torch.float32)
        if ssm_state_in is not None
        else torch.zeros((n_heads, d_head, d_state), dtype=torch.float32)
    )
    h_prev = h_state.clone()
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
    y_norm = ((gated_grp / rms_gate).view(d_inner, T)) * gamma_gate.unsqueeze(1)

    mamba_out = F.linear(y_norm.transpose(0, 1), w_out, bias=None).transpose(0, 1).contiguous()
    delta = residual_gate.unsqueeze(1) * mamba_out if residual_gate is not None else mamba_out
    y_res = x + delta  # Always adds to UNMODULATED residual x

    for key, val in (
        ("U", u_in),
        ("conv_out", conv_out),
        ("conv_state", conv_state_out),
        ("dt", dt),
        ("dA", da),
        ("H", h_state),
        ("y", y_flat),
        ("y_norm", y_norm),
        ("Mamba_out", mamba_out),
        ("Y", y_res),
    ):
        assert_all_finite(val, f"mamba2.{key}")

    result: dict[str, torch.Tensor] = {
        "U": u_in,
        "conv_out": conv_out,
        "conv_state": conv_state_out,
        "dt": dt,
        "dA": da,
        "H_prev": h_prev,
        "H": h_state,
        "y": y_flat,
        "y_norm": y_norm,
        "Mamba_out": mamba_out,
        "Y": y_res,
    }
    if x_mod is not None:
        result["X_mod"] = x_mod
    return result


def gqa_ref(
    x: torch.Tensor,
    gamma: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    eps: float,
    attention_mask_mode: str = "bidirectional_block",
    n_q: int = 32,
    n_kv: int = 2,
    d_head: int = 4,
    kv_cache_in: tuple[torch.Tensor, torch.Tensor] | None = None,
    pre_norm_mod: tuple[torch.Tensor, torch.Tensor] | None = None,
    residual_gate: torch.Tensor | None = None,
    invert_mod_order: bool = False,
) -> dict[str, torch.Tensor]:
    """
    32Q/2KV NoPE GQA Attention supporting both `'bidirectional_block'`
    (`F.scaled_dot_product_attention(..., attn_mask=None, is_causal=False)`) and `'causal'`
    rectangular `[B x (P+B)]` attention, with verified TwoTower AdaLN ordering (modulate -> RMSNorm)
    and gated residual `X + gate * y_attn`.
    """
    S = x.shape[1]
    q_concat = n_q * d_head  # 128

    x_mod: torch.Tensor | None = None
    if pre_norm_mod is not None:
        shift, scale = pre_norm_mod
        if not invert_mod_order:
            # Authoritative TwoTower M/* ordering: modulate -> RMSNorm
            x_mod = modulate_ref(x, shift, scale)
            rn = rmsnorm_ref(x_mod, gamma, eps)
            u_in = rn["U"]
        else:
            # Inverted diagnostic ordering ONLY for Cat 1 order-sensitivity check
            rn = rmsnorm_ref(x, gamma, eps)
            u_in = modulate_ref(rn["U"], shift, scale)
    else:
        rn = rmsnorm_ref(x, gamma, eps)
        u_in = rn["U"]

    u_t_d = u_in.transpose(0, 1)  # [S, d]
    q_flat = F.linear(u_t_d, w_q, bias=None).transpose(0, 1).contiguous()
    k_new = F.linear(u_t_d, w_k, bias=None).transpose(0, 1).contiguous()
    v_new = F.linear(u_t_d, w_v, bias=None).transpose(0, 1).contiguous()

    if kv_cache_in is not None and kv_cache_in[0].shape[1] > 0:
        k_prev, v_prev = kv_cache_in
        P = k_prev.shape[1]
        k_flat = torch.cat([k_prev, k_new], dim=1).contiguous()
        v_flat = torch.cat([v_prev, v_new], dim=1).contiguous()
    else:
        P = 0
        k_flat = k_new
        v_flat = v_new

    K_len = P + S
    q_h = q_flat.view(n_q, d_head, S).transpose(1, 2).contiguous()  # [32, S, 4]
    k_h = k_flat.view(n_kv, d_head, K_len).transpose(1, 2).repeat_interleave(n_q // n_kv, dim=0).contiguous()
    v_h = v_flat.view(n_kv, d_head, K_len).transpose(1, 2).repeat_interleave(n_q // n_kv, dim=0).contiguous()

    raw_scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(d_head)  # [32, S, P+S]
    if attention_mask_mode == "bidirectional_block":
        scores = raw_scores
        assert_all_finite(scores, "gqa.bidirectional.scores")
        softmax = F.softmax(scores, dim=-1)
        v_out_sdpa = F.scaled_dot_product_attention(
            q_h.unsqueeze(0),
            k_h.unsqueeze(0),
            v_h.unsqueeze(0),
            attn_mask=None,
            is_causal=False,
        ).squeeze(0)
    elif attention_mask_mode == "causal":
        q_pos = torch.arange(P, P + S, dtype=torch.long).unsqueeze(1)
        k_pos = torch.arange(0, K_len, dtype=torch.long).unsqueeze(0)
        visible_mask = k_pos <= q_pos  # [S, P+S], True = attend
        scores = raw_scores.masked_fill(~visible_mask.unsqueeze(0), float("-inf"))
        softmax = F.softmax(scores, dim=-1)
        v_out_sdpa = F.scaled_dot_product_attention(
            q_h.unsqueeze(0),
            k_h.unsqueeze(0),
            v_h.unsqueeze(0),
            attn_mask=visible_mask,
            is_causal=False,
        ).squeeze(0)
    else:
        raise ValueError(f"Unsupported attention_mask_mode: {attention_mask_mode!r}")

    # Cross-check F.scaled_dot_product_attention against explicit softmax @ V
    v_out_explicit = torch.matmul(softmax, v_h)
    sdpa_diff = float((v_out_sdpa - v_out_explicit).abs().max().item())
    if sdpa_diff > 1e-6:
        raise RuntimeError(f"SDPA vs explicit softmax@V mismatch: {sdpa_diff}")

    v_out_flat = v_out_sdpa.transpose(1, 2).contiguous().view(q_concat, S)
    attn_out = F.linear(v_out_flat.transpose(0, 1), w_o, bias=None).transpose(0, 1).contiguous()
    delta = residual_gate.unsqueeze(1) * attn_out if residual_gate is not None else attn_out
    y = x + delta  # Always adds to UNMODULATED residual x

    for key, val in (
        ("U", u_in),
        ("Q", q_flat),
        ("K", k_flat),
        ("V", v_flat),
        ("Softmax", softmax),
        ("V_out", v_out_flat),
        ("Attn_out", attn_out),
        ("Y", y),
    ):
        assert_all_finite(val, f"gqa.{key}")

    result: dict[str, torch.Tensor] = {
        "U": u_in,
        "Q": q_flat,
        "K": k_flat,
        "V": v_flat,
        "Scores": scores,
        "Softmax": softmax,
        "V_out": v_out_flat,
        "Attn_out": attn_out,
        "Y": y,
    }
    if x_mod is not None:
        result["X_mod"] = x_mod
    return result


def moe_ref(
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
    router_precision_policy: str = "float64_reference",
    post_norm_mod: tuple[torch.Tensor, torch.Tensor] | None = None,
    residual_gate: torch.Tensor | None = None,
    invert_mod_order: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Granular MoE (128+1) forward pass with verified TwoTower AdaLN ordering (RMSNorm -> modulate),
    Float64 router projection/sigmoid/biased-choice/deterministic-Top-6/unbiased-x2.5 normalization
    (or optional `'float32_nano'` mode), non-gated ReLU^2 routed + 2x shared experts, and gated
    residual `X + gate * y_moe`.
    """
    d, T = x.shape

    u_mod: torch.Tensor | None = None
    if post_norm_mod is not None:
        shift, scale = post_norm_mod
        if not invert_mod_order:
            # Authoritative TwoTower E ordering: RMSNorm -> modulate
            rn = rmsnorm_ref(x, gamma_in, eps)
            u = rn["U"]
            u_mod = modulate_ref(u, shift, scale)
            u_in = u_mod
        else:
            # Inverted diagnostic ordering ONLY for Cat 1 order-sensitivity check
            x_mod = modulate_ref(x, shift, scale)
            rn = rmsnorm_ref(x_mod, gamma_in, eps)
            u = rn["U"]
            u_in = u
    else:
        rn = rmsnorm_ref(x, gamma_in, eps)
        u = rn["U"]
        u_in = u

    topk_indices = torch.zeros((T, top_k), dtype=torch.long)
    topk_raw_scores = torch.zeros((T, top_k), dtype=torch.float32)
    topk_weights = torch.zeros((T, top_k), dtype=torch.float32)
    y_routed = torch.zeros((d, T), dtype=torch.float32)

    if router_precision_policy == "float64_reference":
        u64 = u_in.to(torch.float64)
        w64 = w_router.to(torch.float64)
        b64 = e_bias.to(torch.float64)
        logits64 = F.linear(u64.transpose(0, 1), w64, bias=None).transpose(0, 1).contiguous()
        scores64 = torch.sigmoid(logits64)
        choice64 = scores64 + b64.unsqueeze(1)

        scores_f32 = scores64.to(torch.float32)
        choice_f32 = choice64.to(torch.float32)

        for t in range(T):
            # Stable descending sort guarantees deterministic lower-index tie-break on exact ties
            _, sorted_idx = torch.sort(choice64[:, t], descending=True, stable=True)
            idx = sorted_idx[:top_k]
            topk_indices[t] = idx
            unbiased64 = scores64[idx, t]
            topk_raw_scores[t] = unbiased64.to(torch.float32)
            weights64 = (unbiased64 / (unbiased64.sum() + 1e-20)) * routed_scaling
            weights = weights64.to(torch.float32)
            topk_weights[t] = weights

            u_t = u_in[:, t]
            for j in range(top_k):
                e = int(idx[j].item())
                h = F.linear(u_t, w_up[e], bias=None)
                a = F.relu(h).pow(2)
                y_e = F.linear(a, w_down[e], bias=None)
                y_routed[:, t] = y_routed[:, t] + weights[j] * y_e
    elif router_precision_policy == "float32_nano":
        logits32 = F.linear(u_in.transpose(0, 1), w_router, bias=None).transpose(0, 1).contiguous()
        scores_f32 = torch.sigmoid(logits32)
        choice_f32 = scores_f32 + e_bias.unsqueeze(1)

        for t in range(T):
            _, sorted_idx = torch.sort(choice_f32[:, t], descending=True, stable=True)
            idx = sorted_idx[:top_k]
            topk_indices[t] = idx
            unbiased32 = scores_f32[idx, t]
            topk_raw_scores[t] = unbiased32
            weights = (unbiased32 / (unbiased32.sum() + 1e-20)) * routed_scaling
            topk_weights[t] = weights

            u_t = u_in[:, t]
            for j in range(top_k):
                e = int(idx[j].item())
                h = F.linear(u_t, w_up[e], bias=None)
                a = F.relu(h).pow(2)
                y_e = F.linear(a, w_down[e], bias=None)
                y_routed[:, t] = y_routed[:, t] + weights[j] * y_e
    else:
        raise ValueError(f"Unsupported router_precision_policy: {router_precision_policy!r}")

    shared_h = F.linear(u_in.transpose(0, 1), w_up_s, bias=None).transpose(0, 1).contiguous()
    shared_a = F.relu(shared_h).pow(2)
    y_shared = F.linear(shared_a.transpose(0, 1), w_down_s, bias=None).transpose(0, 1).contiguous()
    y_moe = y_routed + y_shared
    delta = residual_gate.unsqueeze(1) * y_moe if residual_gate is not None else y_moe
    y_res = x + delta  # Always adds to UNMODULATED residual x

    for key, val in (
        ("U", u),
        ("scores", scores_f32),
        ("scores_for_choice", choice_f32),
        ("topk_weights", topk_weights),
        ("y_shared", y_shared),
        ("y_routed", y_routed),
        ("y_moe", y_moe),
        ("Y", y_res),
    ):
        assert_all_finite(val, f"moe.{key}")

    result: dict[str, torch.Tensor] = {
        "U": u,
        "U_in": u_in,
        "scores": scores_f32,
        "scores_for_choice": choice_f32,
        "topk_indices": topk_indices,
        "topk_raw_scores": topk_raw_scores,
        "topk_weights": topk_weights,
        "shared_a": shared_a,
        "y_shared": y_shared,
        "y_routed": y_routed,
        "y_moe": y_moe,
        "Y": y_res,
    }
    if u_mod is not None:
        result["U_mod"] = u_mod
    return result


def lm_head_ref(
    x_final: torch.Tensor,
    final_gamma: torch.Tensor,
    w_lm_head: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Strictly unmodulated Final RMSNorm + untied LM head returning row-major [S, V]."""
    fn = rmsnorm_ref(x_final, final_gamma, eps)
    # fn["U"] is [d, S] -> transpose [S, d] -> F.linear -> [S, V] (row-major, unshifted!)
    logits_s_v = F.linear(fn["U"].transpose(0, 1), w_lm_head, bias=None).contiguous()
    probs_s_v = F.softmax(logits_s_v, dim=-1)
    assert_all_finite(logits_s_v, "lm_head.logits")
    assert_all_finite(probs_s_v, "lm_head.probs")
    return {
        "rms": fn["rms"],
        "X_norm": fn["X_norm"],
        "U_final": fn["U"],
        "logits": logits_s_v,  # [S, V]
        "probs": probs_s_v,    # [S, V]
    }


# ============================================================================
# FULL 52-LAYER CONTEXT TOWER & DENOISER TOWER RUNNERS
# ============================================================================

def clone_mamba_states(
    states: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {b: (c.clone(), s.clone()) for b, (c, s) in states.items()}


def clone_gqa_caches(
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {b: (k.clone(), v.clone()) for b, (k, v) in caches.items()}


def run_context_step_ref(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    token_ids_list: list[int],
    mamba_in: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    gqa_in: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Causal 52-layer Context Tower forward step updating Mamba & GQA caches."""
    d = int(manifest["d"])
    vocab_size = int(manifest["vocabSize"])
    eps = float(manifest["rmsNormEps"])
    pattern: str = manifest["pattern"]
    router_policy: str = manifest["routerPrecisionPolicy"]

    w_embed = tensor_map["context.W_embed"].view(vocab_size, d)
    final_gamma = tensor_map["context.finalNormGamma"].view(d)
    w_lm_head = tensor_map["context.W_lm_head"].view(vocab_size, d)

    token_ids = torch.tensor(token_ids_list, dtype=torch.long)
    x_curr = F.embedding(token_ids, w_embed).transpose(0, 1).contiguous()
    residuals: list[torch.Tensor] = [x_curr.clone()]

    mamba_out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    gqa_out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for bay_id in range(1, len(pattern) + 1):
        kind = pattern[bay_id - 1]
        p = f"context.bay_{bay_id:02d}"
        if kind == "M":
            prev_m = mamba_in.get(bay_id) if mamba_in is not None else None
            out = mamba2_ref(
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
                conv_state_in=prev_m[0] if prev_m is not None else None,
                ssm_state_in=prev_m[1] if prev_m is not None else None,
            )
            mamba_out[bay_id] = (out["conv_state"], out["H"])
            x_curr = out["Y"]
        elif kind == "E":
            out = moe_ref(
                x_curr,
                gamma_in=tensor_map[f"{p}.gamma_in"].view(d),
                w_router=tensor_map[f"{p}.W_router"].view(128, d),
                e_bias=tensor_map[f"{p}.e_score_correction_bias"].view(128),
                w_up=tensor_map[f"{p}.W_up"].view(128, 4, d),
                w_down=tensor_map[f"{p}.W_down"].view(128, d, 4),
                w_up_s=tensor_map[f"{p}.W_up_s"].view(8, d),
                w_down_s=tensor_map[f"{p}.W_down_s"].view(d, 8),
                eps=eps,
                router_precision_policy=router_policy,
            )
            x_curr = out["Y"]
        elif kind == "*":
            prev_g = gqa_in.get(bay_id) if gqa_in is not None else None
            out = gqa_ref(
                x_curr,
                gamma=tensor_map[f"{p}.gamma"].view(d),
                w_q=tensor_map[f"{p}.W_Q"].view(128, d),
                w_k=tensor_map[f"{p}.W_K"].view(8, d),
                w_v=tensor_map[f"{p}.W_V"].view(8, d),
                w_o=tensor_map[f"{p}.W_O"].view(d, 128),
                eps=eps,
                attention_mask_mode="causal",
                kv_cache_in=prev_g,
            )
            gqa_out[bay_id] = (out["K"], out["V"])
            x_curr = out["Y"]
        else:
            raise ValueError(f"Unexpected layer kind {kind!r} at bay {bay_id}")
        residuals.append(x_curr.clone())

    lm = lm_head_ref(x_curr, final_gamma, w_lm_head, eps)
    return {
        "mamba": mamba_out,
        "gqa": gqa_out,
        "residuals": residuals,
        "finalNorm": lm,
        "logits": lm["logits"],
        "probs": lm["probs"],
    }


def context_prefill_ref(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    prompt_ids: list[int],
) -> dict[str, Any]:
    """Context prefill preserving exact mambaS2 (S-2) and ctxState (S-1 + full KV)."""
    S = len(prompt_ids)
    assert S >= 2, "Prompt length >= 2 expected for prefill helper"
    head = run_context_step_ref(manifest, tensor_map, prompt_ids[:-1])
    mamba_s2 = clone_mamba_states(head["mamba"])
    tail = run_context_step_ref(
        manifest,
        tensor_map,
        [prompt_ids[-1]],
        mamba_in=head["mamba"],
        gqa_in=head["gqa"],
    )
    # Self-check 1-pass vs 2-step split equivalence
    full = run_context_step_ref(manifest, tensor_map, prompt_ids)
    split_err = float((tail["logits"][0] - full["logits"][-1]).abs().max().item())
    if split_err > 1e-5:
        raise RuntimeError(f"PyTorch Context prefill split mismatch: {split_err}")

    return {
        "prefixIds": list(prompt_ids),
        "ctxMamba": tail["mamba"],
        "ctxGqa": tail["gqa"],
        "mambaS2": mamba_s2,
        "arLogits": tail["logits"][0].clone(),
        "arProbs": tail["probs"][0].clone(),
    }


def context_commit_ref(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    ctx_snapshot: dict[str, Any],
    block_ids: list[int],
) -> dict[str, Any]:
    """Causally commit a completed block (B >= 1) to Context Tower, preserving exact mambaS2."""
    B = len(block_ids)
    assert B >= 1
    if B == 1:
        mamba_s2 = clone_mamba_states(ctx_snapshot["ctxMamba"])
        step = run_context_step_ref(
            manifest,
            tensor_map,
            block_ids,
            mamba_in=ctx_snapshot["ctxMamba"],
            gqa_in=ctx_snapshot["ctxGqa"],
        )
        return {
            "prefixIds": ctx_snapshot["prefixIds"] + list(block_ids),
            "ctxMamba": step["mamba"],
            "ctxGqa": step["gqa"],
            "mambaS2": mamba_s2,
            "arLogits": step["logits"][0].clone(),
            "arProbs": step["probs"][0].clone(),
        }

    head = run_context_step_ref(
        manifest,
        tensor_map,
        block_ids[:-1],
        mamba_in=ctx_snapshot["ctxMamba"],
        gqa_in=ctx_snapshot["ctxGqa"],
    )
    mamba_s2 = clone_mamba_states(head["mamba"])
    tail = run_context_step_ref(
        manifest,
        tensor_map,
        [block_ids[-1]],
        mamba_in=head["mamba"],
        gqa_in=head["gqa"],
    )
    return {
        "prefixIds": ctx_snapshot["prefixIds"] + list(block_ids),
        "ctxMamba": tail["mamba"],
        "ctxGqa": tail["gqa"],
        "mambaS2": mamba_s2,
        "arLogits": tail["logits"][0].clone(),
        "arProbs": tail["probs"][0].clone(),
    }


def run_denoiser_pass_ref(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    ctx_snapshot: dict[str, Any],
    block_ids: list[int],
    mode: str = "mask_diffusion",
    t_override: float | None = None,
    capture_bays: set[int] | None = None,
) -> dict[str, Any]:
    """
    Read-only 52-layer Denoiser Tower pass in `'mask_diffusion'` or `'mock_ar'` mode.
    NEVER mutates `ctx_snapshot` and discards all Denoiser scratchpad states.
    """
    d = int(manifest["d"])
    vocab_size = int(manifest["vocabSize"])
    freq_dim = int(manifest["freqDim"])
    mask_token_id = int(manifest["maskTokenId"])
    eps = float(manifest["rmsNormEps"])
    pattern: str = manifest["pattern"]
    router_policy: str = manifest["routerPrecisionPolicy"]
    max_period_scale = float(manifest["maxPeriodScale"])
    max_period = float(manifest["maxPeriod"])

    B = len(block_ids)
    w_embed = tensor_map["denoiser.W_embed"].view(vocab_size, d)
    final_gamma = tensor_map["denoiser.finalNormGamma"].view(d)
    w_lm_head = tensor_map["denoiser.W_lm_head"].view(vocab_size, d)

    if mode == "mask_diffusion":
        n_masked = sum(1 for tok in block_ids if tok == mask_token_id)
        t_model: float | None = float(t_override) if t_override is not None else float(n_masked) / float(B)
        time_proj = compute_global_timestep_ref(
            t_model, tensor_map, d, freq_dim, max_period_scale, max_period
        )
        t_emb: torch.Tensor | None = time_proj["tEmb"]
        mamba_seeds = ctx_snapshot["ctxMamba"]
        gqa_prefix = ctx_snapshot["ctxGqa"]
        attn_mode = "bidirectional_block"
    elif mode == "mock_ar":
        assert B == 1, "mock_ar requires B == 1"
        t_model = None
        time_proj = None
        t_emb = None
        mamba_seeds = ctx_snapshot["mambaS2"]
        # Truncate Context KV to KV[:-1] (prefix len P - 1)
        P = len(ctx_snapshot["prefixIds"])
        gqa_prefix = {
            bay_id: (kv[0][:, : P - 1].contiguous(), kv[1][:, : P - 1].contiguous())
            for bay_id, kv in ctx_snapshot["ctxGqa"].items()
        }
        attn_mode = "causal"
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    token_ids = torch.tensor(block_ids, dtype=torch.long)
    x_curr = F.embedding(token_ids, w_embed).transpose(0, 1).contiguous()
    residuals: list[torch.Tensor] = [x_curr.clone()]
    moe_topk_by_bay: dict[str, list[int]] = {}
    traces: dict[int, dict[str, Any]] = {}

    for bay_id in range(1, len(pattern) + 1):
        layer_idx = bay_id - 1
        kind = pattern[layer_idx]
        p = f"denoiser.bay_{bay_id:02d}"

        mod_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if t_emb is not None:
            table = tensor_map[f"time_cond.scale_shift_tables.{layer_idx:02d}"].view(3 * d)
            mod_params = layer_mod_params_ref(t_emb, table)

        if kind == "M":
            seed = mamba_seeds[bay_id]
            has_prefix = (len(ctx_snapshot["prefixIds"]) - (1 if mode == "mock_ar" else 0)) > 0
            out = mamba2_ref(
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
                conv_state_in=seed[0] if has_prefix else None,
                ssm_state_in=seed[1] if has_prefix else None,
                pre_norm_mod=(mod_params[0], mod_params[1]) if mod_params is not None else None,
                residual_gate=mod_params[2] if mod_params is not None else None,
            )
            x_curr = out["Y"]
        elif kind == "E":
            out = moe_ref(
                x_curr,
                gamma_in=tensor_map[f"{p}.gamma_in"].view(d),
                w_router=tensor_map[f"{p}.W_router"].view(128, d),
                e_bias=tensor_map[f"{p}.e_score_correction_bias"].view(128),
                w_up=tensor_map[f"{p}.W_up"].view(128, 4, d),
                w_down=tensor_map[f"{p}.W_down"].view(128, d, 4),
                w_up_s=tensor_map[f"{p}.W_up_s"].view(8, d),
                w_down_s=tensor_map[f"{p}.W_down_s"].view(d, 8),
                eps=eps,
                router_precision_policy=router_policy,
                post_norm_mod=(mod_params[0], mod_params[1]) if mod_params is not None else None,
                residual_gate=mod_params[2] if mod_params is not None else None,
            )
            moe_topk_by_bay[str(bay_id)] = to_i32_list(out["topk_indices"])
            x_curr = out["Y"]
        elif kind == "*":
            kv_pref = gqa_prefix[bay_id]
            out = gqa_ref(
                x_curr,
                gamma=tensor_map[f"{p}.gamma"].view(d),
                w_q=tensor_map[f"{p}.W_Q"].view(128, d),
                w_k=tensor_map[f"{p}.W_K"].view(8, d),
                w_v=tensor_map[f"{p}.W_V"].view(8, d),
                w_o=tensor_map[f"{p}.W_O"].view(d, 128),
                eps=eps,
                attention_mask_mode=attn_mode,
                kv_cache_in=kv_pref if kv_pref[0].shape[1] > 0 else None,
                pre_norm_mod=(mod_params[0], mod_params[1]) if mod_params is not None else None,
                residual_gate=mod_params[2] if mod_params is not None else None,
            )
            x_curr = out["Y"]
        else:
            raise ValueError(f"Unexpected layer kind {kind!r} at bay {bay_id}")

        if capture_bays is not None and bay_id in capture_bays:
            traces[bay_id] = {"kind": kind, "mod": mod_params, "out": out}
        residuals.append(x_curr.clone())

    lm = lm_head_ref(x_curr, final_gamma, w_lm_head, eps)
    return {
        "tModel": t_model,
        "timestepProjection": time_proj,
        "residuals": residuals,
        "moeTopkByBay": moe_topk_by_bay,
        "traces": traces,
        "finalNorm": lm,
        "logits": lm["logits"],
        "probs": lm["probs"],
    }


# ============================================================================
# MDLM / SUBS CONFIDENCE-UNMASKING & GUMBEL-MAX SAMPLER REFERENCE
# ============================================================================

def mdlm_forward_ref(
    logits: torch.Tensor,
    xt: list[int],
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SUBS / MDLM log-probability constraint on unshifted logits [B, V]:
      - Locked unmasked positions carry exact delta (log=0 / -1e12, p=1 / 0).
      - Masked positions suppress mask_token_id with -1e12 pre-log_softmax (p[mask]=0).
    """
    assert_all_finite(logits, "mdlm.logits")
    B, V = logits.shape
    log_x_theta = torch.full((B, V), MDLM_MASK_SUPPRESS_LOGIT, dtype=torch.float32)
    x_theta = torch.zeros((B, V), dtype=torch.float32)

    for j in range(B):
        tok = xt[j]
        if tok != mask_token_id:
            log_x_theta[j, tok] = 0.0
            x_theta[j, tok] = 1.0
        else:
            row64 = logits[j].to(torch.float64).clone()
            row64[mask_token_id] = MDLM_MASK_SUPPRESS_LOGIT
            lp64 = F.log_softmax(row64, dim=-1)
            p64 = torch.exp(lp64)
            p64[mask_token_id] = 0.0
            log_x_theta[j] = lp64.to(torch.float32)
            x_theta[j] = p64.to(torch.float32)

    assert_all_finite(log_x_theta, "mdlm.log_x_theta")
    assert_all_finite(x_theta, "mdlm.x_theta")
    return log_x_theta, x_theta


def gumbel_sample_ref(
    logits: torch.Tensor,
    xt: list[int],
    mask_token_id: int,
    temperature: float,
    uniform_variates: list[float] | None = None,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Greedy (temperature <= 0) or deterministic Gumbel-max (explicit uniform array u)."""
    log_x_theta, x_theta = mdlm_forward_ref(logits, xt, mask_token_id)
    B, V = logits.shape
    predicted: list[int] = [0] * B

    if temperature <= 0.0:
        for j in range(B):
            if xt[j] != mask_token_id:
                predicted[j] = xt[j]
                continue
            row = log_x_theta[j].clone()
            row[mask_token_id] = float("-inf")
            predicted[j] = int(torch.argmax(row).item())
    else:
        if uniform_variates is None or len(uniform_variates) != B * V:
            raise ValueError(f"Expected uniform_variates of length {B * V}")
        u64 = torch.tensor(uniform_variates, dtype=torch.float64).view(B, V)
        inv_temp = 1.0 / float(temperature)
        for j in range(B):
            if xt[j] != mask_token_id:
                predicted[j] = xt[j]
                continue
            row64 = logits[j].to(torch.float64) * inv_temp
            row64[mask_token_id] = MDLM_MASK_SUPPRESS_LOGIT
            scaled_lp64 = F.log_softmax(row64, dim=-1)
            clamped_u = torch.clamp(u64[j], min=GUMBEL_UNIFORM_MIN)
            gumbel = -torch.log(-torch.log(clamped_u))
            score = scaled_lp64 + gumbel
            score[mask_token_id] = float("-inf")
            predicted[j] = int(torch.argmax(score).item())

    return predicted, log_x_theta, x_theta


def confidence_unmask_step_ref(
    xt: list[int],
    logits: torch.Tensor,
    mask_token_id: int,
    step_index: int,
    steps_per_block: int,
    confidence_threshold: float,
    temperature: float,
    eos_token_ids: list[int],
    uniform_variates: list[float] | None = None,
) -> dict[str, Any]:
    """Clean-room PyTorch confidence-unmasking step matching all verified TwoTower rules."""
    B, V = logits.shape
    masked_indices = [j for j in range(B) if xt[j] == mask_token_id]
    n_masked_before = len(masked_indices)
    remaining_steps = max(1, steps_per_block - step_index)
    t_model = float(n_masked_before) / float(B)

    if n_masked_before == 0:
        log_x_theta, x_theta = mdlm_forward_ref(logits, xt, mask_token_id)
        xt_next = list(xt)
        eos_set = set(eos_token_ids)
        return {
            "xtBefore": list(xt),
            "xtNext": xt_next,
            "predicted": list(xt),
            "confidence": [float("inf")] * B,
            "logXTheta": log_x_theta,
            "unscaledProbs": x_theta,
            "tModel": 0.0,
            "stepIndex": step_index,
            "stepsPerBlock": steps_per_block,
            "remainingSteps": remaining_steps,
            "nMaskedBefore": 0,
            "nMaskedAfter": 0,
            "numAboveThreshold": 0,
            "minCommitFloor": 0,
            "tokensCommittedCount": 0,
            "commitReason": "early_completed",
            "committedIndices": [],
            "thresholdCommittedIndices": [],
            "floorCommittedIndices": [],
            "remaskedIndices": [],
            "earlyCompleted": True,
            "isBlockComplete": True,
            "containsEos": any(tok in eos_set for tok in xt_next),
        }

    predicted, log_x_theta, x_theta = gumbel_sample_ref(
        logits, xt, mask_token_id, temperature, uniform_variates
    )

    # Confidence strictly from UNSCALED x_theta
    confidence: list[float] = []
    num_above = 0
    for j in range(B):
        if xt[j] == mask_token_id:
            c = float(x_theta[j, predicted[j]].item())
            confidence.append(c)
            if c > confidence_threshold:  # Strict '>' comparison!
                num_above += 1
        else:
            confidence.append(float("inf"))

    min_commit_floor = math.ceil(n_masked_before / remaining_steps)
    if step_index >= steps_per_block - 1:
        tokens_to_commit = n_masked_before
        commit_reason = "final_step"
    else:
        c1 = max(num_above, 1)
        tokens_to_commit = min(n_masked_before, max(c1, min_commit_floor))
        if num_above >= max(1, min_commit_floor):
            commit_reason = "threshold"
        elif min_commit_floor > 1:
            commit_reason = "budget_floor"
        else:
            commit_reason = "progress_floor"

    # Ascending confidence sort among CURRENTLY MASKED slots only (lower-index tie-break)
    sorted_masked_asc = sorted(masked_indices, key=lambda idx: (confidence[idx], idx))
    num_to_remask = n_masked_before - tokens_to_commit
    remasked = sorted(sorted_masked_asc[:num_to_remask])
    committed = sorted(sorted_masked_asc[num_to_remask:])

    xt_next = [predicted[j] if xt[j] == mask_token_id else xt[j] for j in range(B)]
    for idx in remasked:
        xt_next[idx] = mask_token_id

    threshold_committed = [idx for idx in committed if confidence[idx] > confidence_threshold]
    floor_committed = [idx for idx in committed if confidence[idx] <= confidence_threshold]
    n_masked_after = n_masked_before - tokens_to_commit
    is_block_complete = n_masked_after == 0
    early_completed = is_block_complete and (step_index < steps_per_block - 1)
    eos_set = set(eos_token_ids)

    return {
        "xtBefore": list(xt),
        "xtNext": xt_next,
        "predicted": predicted,
        "confidence": confidence,
        "logXTheta": log_x_theta,
        "unscaledProbs": x_theta,
        "tModel": t_model,
        "stepIndex": step_index,
        "stepsPerBlock": steps_per_block,
        "remainingSteps": remaining_steps,
        "nMaskedBefore": n_masked_before,
        "nMaskedAfter": n_masked_after,
        "numAboveThreshold": num_above,
        "minCommitFloor": min_commit_floor,
        "tokensCommittedCount": tokens_to_commit,
        "commitReason": commit_reason,
        "committedIndices": committed,
        "thresholdCommittedIndices": threshold_committed,
        "floorCommittedIndices": floor_committed,
        "remaskedIndices": remasked,
        "earlyCompleted": early_completed,
        "isBlockComplete": is_block_complete,
        "containsEos": any(tok in eos_set for tok in xt_next),
    }


# ============================================================================
# FIXTURE 1: BLOCKS PARITY (CATEGORIES 1, 2, 3, 4)
# ============================================================================

def build_blocks_parity_fixture(
    default_manifest: dict[str, Any],
    default_tensors: dict[str, torch.Tensor],
    gen: torch.Generator,
) -> dict[str, Any]:
    d = int(default_manifest["d"])
    freq_dim = int(default_manifest["freqDim"])
    eps = float(default_manifest["rmsNormEps"])
    pattern: str = default_manifest["pattern"]
    max_period_scale = float(default_manifest["maxPeriodScale"])
    max_period = float(default_manifest["maxPeriod"])

    # ------------------------------------------------------------------
    # CAT 1: Timestep & Asymmetric Modulation + Order-Inversion Sensitivity
    # ------------------------------------------------------------------
    timesteps = [1.0, 0.75, 0.50, 0.25]
    sample_layers = [0, 1, 5, 51]  # kinds: M (bay 1), E (bay 2), * (bay 6), E (bay 52)
    timestep_records: list[dict[str, Any]] = []

    for t_val in timesteps:
        proj = compute_global_timestep_ref(
            t_val, default_tensors, d, freq_dim, max_period_scale, max_period
        )
        layers_out: list[dict[str, Any]] = []
        for l_idx in sample_layers:
            table = default_tensors[f"time_cond.scale_shift_tables.{l_idx:02d}"].view(3 * d)
            shift, scale, gate = layer_mod_params_ref(proj["tEmb"], table)
            layers_out.append({
                "layerIndex": l_idx,
                "bayId": l_idx + 1,
                "kind": pattern[l_idx],
                "shift": to_f32_list(shift, f"cat1.t{t_val}.l{l_idx}.shift"),
                "scale": to_f32_list(scale, f"cat1.t{t_val}.l{l_idx}.scale"),
                "gate": to_f32_list(gate, f"cat1.t{t_val}.l{l_idx}.gate"),
            })
        timestep_records.append({
            "t": t_val,
            "phi": to_f32_list(proj["phi"], f"cat1.t{t_val}.phi"),
            "tRepr": to_f32_list(proj["tRepr"], f"cat1.t{t_val}.tRepr"),
            "tEmb": to_f32_list(proj["tEmb"], f"cat1.t{t_val}.tEmb"),
            "layers": layers_out,
        })

    # Order-inversion sensitivity deltas (must exceed > 1e-2!)
    proj_075 = compute_global_timestep_ref(
        0.75, default_tensors, d, freq_dim, max_period_scale, max_period
    )
    # Non-zero mean + spread ensures pre-norm vs post-norm shift/scale diverges strongly
    x_sens = torch.randn((d, 4), generator=gen, dtype=torch.float32) * 1.35 + 1.15

    # Bay 01 Mamba (layer 0, M): correct = modulate->RMSNorm vs inverted = RMSNorm->modulate
    s0, sc0, g0 = layer_mod_params_ref(
        proj_075["tEmb"], default_tensors["time_cond.scale_shift_tables.00"].view(3 * d)
    )
    p1 = "denoiser.bay_01"
    m_correct = mamba2_ref(
        x_sens,
        gamma_in=default_tensors[f"{p1}.gamma_in"].view(d),
        w_in=default_tensors[f"{p1}.W_in"].view(208, d),
        w_conv=default_tensors[f"{p1}.w_conv"].view(128, 4),
        b_conv=default_tensors[f"{p1}.b_conv"].view(128),
        a_log=default_tensors[f"{p1}.A_log"].view(16),
        dt_bias=default_tensors[f"{p1}.dt_bias"].view(16),
        d_skip=default_tensors[f"{p1}.D"].view(16),
        gamma_gate=default_tensors[f"{p1}.gamma_gate"].view(64),
        w_out=default_tensors[f"{p1}.W_out"].view(d, 64),
        eps=eps,
        pre_norm_mod=(s0, sc0),
        residual_gate=g0,
        invert_mod_order=False,
    )
    m_inverted = mamba2_ref(
        x_sens,
        gamma_in=default_tensors[f"{p1}.gamma_in"].view(d),
        w_in=default_tensors[f"{p1}.W_in"].view(208, d),
        w_conv=default_tensors[f"{p1}.w_conv"].view(128, 4),
        b_conv=default_tensors[f"{p1}.b_conv"].view(128),
        a_log=default_tensors[f"{p1}.A_log"].view(16),
        dt_bias=default_tensors[f"{p1}.dt_bias"].view(16),
        d_skip=default_tensors[f"{p1}.D"].view(16),
        gamma_gate=default_tensors[f"{p1}.gamma_gate"].view(64),
        w_out=default_tensors[f"{p1}.W_out"].view(d, 64),
        eps=eps,
        pre_norm_mod=(s0, sc0),
        residual_gate=g0,
        invert_mod_order=True,
    )
    m_u_delta = float((m_correct["U"] - m_inverted["U"]).abs().max().item())
    m_y_delta = float((m_correct["Y"] - m_inverted["Y"]).abs().max().item())

    # Bay 06 GQA (layer 5, *): correct = modulate->RMSNorm vs inverted = RMSNorm->modulate
    s5, sc5, g5 = layer_mod_params_ref(
        proj_075["tEmb"], default_tensors["time_cond.scale_shift_tables.05"].view(3 * d)
    )
    p6 = "denoiser.bay_06"
    g_correct = gqa_ref(
        x_sens,
        gamma=default_tensors[f"{p6}.gamma"].view(d),
        w_q=default_tensors[f"{p6}.W_Q"].view(128, d),
        w_k=default_tensors[f"{p6}.W_K"].view(8, d),
        w_v=default_tensors[f"{p6}.W_V"].view(8, d),
        w_o=default_tensors[f"{p6}.W_O"].view(d, 128),
        eps=eps,
        attention_mask_mode="bidirectional_block",
        pre_norm_mod=(s5, sc5),
        residual_gate=g5,
        invert_mod_order=False,
    )
    g_inverted = gqa_ref(
        x_sens,
        gamma=default_tensors[f"{p6}.gamma"].view(d),
        w_q=default_tensors[f"{p6}.W_Q"].view(128, d),
        w_k=default_tensors[f"{p6}.W_K"].view(8, d),
        w_v=default_tensors[f"{p6}.W_V"].view(8, d),
        w_o=default_tensors[f"{p6}.W_O"].view(d, 128),
        eps=eps,
        attention_mask_mode="bidirectional_block",
        pre_norm_mod=(s5, sc5),
        residual_gate=g5,
        invert_mod_order=True,
    )
    g_u_delta = float((g_correct["U"] - g_inverted["U"]).abs().max().item())
    g_y_delta = float((g_correct["Y"] - g_inverted["Y"]).abs().max().item())

    # Bay 02 MoE (layer 1, E): correct = RMSNorm->modulate vs inverted = modulate->RMSNorm
    s1, sc1, g1 = layer_mod_params_ref(
        proj_075["tEmb"], default_tensors["time_cond.scale_shift_tables.01"].view(3 * d)
    )
    p2 = "denoiser.bay_02"
    e_correct = moe_ref(
        x_sens,
        gamma_in=default_tensors[f"{p2}.gamma_in"].view(d),
        w_router=default_tensors[f"{p2}.W_router"].view(128, d),
        e_bias=default_tensors[f"{p2}.e_score_correction_bias"].view(128),
        w_up=default_tensors[f"{p2}.W_up"].view(128, 4, d),
        w_down=default_tensors[f"{p2}.W_down"].view(128, d, 4),
        w_up_s=default_tensors[f"{p2}.W_up_s"].view(8, d),
        w_down_s=default_tensors[f"{p2}.W_down_s"].view(d, 8),
        eps=eps,
        router_precision_policy="float64_reference",
        post_norm_mod=(s1, sc1),
        residual_gate=g1,
        invert_mod_order=False,
    )
    e_inverted = moe_ref(
        x_sens,
        gamma_in=default_tensors[f"{p2}.gamma_in"].view(d),
        w_router=default_tensors[f"{p2}.W_router"].view(128, d),
        e_bias=default_tensors[f"{p2}.e_score_correction_bias"].view(128),
        w_up=default_tensors[f"{p2}.W_up"].view(128, 4, d),
        w_down=default_tensors[f"{p2}.W_down"].view(128, d, 4),
        w_up_s=default_tensors[f"{p2}.W_up_s"].view(8, d),
        w_down_s=default_tensors[f"{p2}.W_down_s"].view(d, 8),
        eps=eps,
        router_precision_policy="float64_reference",
        post_norm_mod=(s1, sc1),
        residual_gate=g1,
        invert_mod_order=True,
    )
    e_u_delta = float((e_correct["U_in"] - e_inverted["U_in"]).abs().max().item())
    e_y_delta = float((e_correct["Y"] - e_inverted["Y"]).abs().max().item())

    for label, delta_u, delta_y in (
        ("bay01_mamba", m_u_delta, m_y_delta),
        ("bay06_gqa", g_u_delta, g_y_delta),
        ("bay02_moe", e_u_delta, e_y_delta),
    ):
        if delta_u <= 1e-2 or delta_y <= 1e-2:
            raise RuntimeError(
                f"Order-inversion sensitivity too small for {label}: "
                f"delta_u={delta_u:.4e}, delta_y={delta_y:.4e} (need > 1e-2)"
            )

    cat1 = {
        "timesteps": timestep_records,
        "orderInversionSensitivity": {
            "t": 0.75,
            "tokens": 4,
            "X": to_f32_list(x_sens, "cat1.sens.X"),
            "bay01Mamba": {
                "bayId": 1,
                "layerIndex": 0,
                "kind": "M",
                "correctOrder": "modulate_then_norm",
                "U_correct": to_f32_list(m_correct["U"], "cat1.m.U_c"),
                "U_inverted": to_f32_list(m_inverted["U"], "cat1.m.U_i"),
                "Y_correct": to_f32_list(m_correct["Y"], "cat1.m.Y_c"),
                "Y_inverted": to_f32_list(m_inverted["Y"], "cat1.m.Y_i"),
                "maxAbsDeltaU": m_u_delta,
                "maxAbsDeltaY": m_y_delta,
            },
            "bay06Gqa": {
                "bayId": 6,
                "layerIndex": 5,
                "kind": "*",
                "correctOrder": "modulate_then_norm",
                "U_correct": to_f32_list(g_correct["U"], "cat1.g.U_c"),
                "U_inverted": to_f32_list(g_inverted["U"], "cat1.g.U_i"),
                "Y_correct": to_f32_list(g_correct["Y"], "cat1.g.Y_c"),
                "Y_inverted": to_f32_list(g_inverted["Y"], "cat1.g.Y_i"),
                "maxAbsDeltaU": g_u_delta,
                "maxAbsDeltaY": g_y_delta,
            },
            "bay02Moe": {
                "bayId": 2,
                "layerIndex": 1,
                "kind": "E",
                "correctOrder": "norm_then_modulate",
                "U_correct": to_f32_list(e_correct["U_in"], "cat1.e.U_c"),
                "U_inverted": to_f32_list(e_inverted["U_in"], "cat1.e.U_i"),
                "Y_correct": to_f32_list(e_correct["Y"], "cat1.e.Y_c"),
                "Y_inverted": to_f32_list(e_inverted["Y"], "cat1.e.Y_i"),
                "maxAbsDeltaU": e_u_delta,
                "maxAbsDeltaY": e_y_delta,
            },
        },
    }

    # ------------------------------------------------------------------
    # CAT 2: Bidirectional Rectangular GQA [B=4 x (P+B=12)] + Future-Token Influence
    # ------------------------------------------------------------------
    P_gqa, B_gqa, kv_cap = 8, 4, 24
    p43 = "denoiser.bay_43"
    s42, sc42, g42 = layer_mod_params_ref(
        proj_075["tEmb"], default_tensors["time_cond.scale_shift_tables.42"].view(3 * d)
    )
    k_pref = torch.randn((8, P_gqa), generator=gen, dtype=torch.float32) * 0.75
    v_pref = torch.randn((8, P_gqa), generator=gen, dtype=torch.float32) * 0.75
    x_gqa_a = torch.randn((d, B_gqa), generator=gen, dtype=torch.float32) * 0.95
    x_gqa_b = x_gqa_a.clone()
    # Perturb ONLY future block token j = 3 (columns 0, 1, 2 remain bit-identical!)
    x_gqa_b[:, 3] = torch.randn((d,), generator=gen, dtype=torch.float32) * 1.8 + 1.2

    gqa_args = dict(
        gamma=default_tensors[f"{p43}.gamma"].view(d),
        w_q=default_tensors[f"{p43}.W_Q"].view(128, d),
        w_k=default_tensors[f"{p43}.W_K"].view(8, d),
        w_v=default_tensors[f"{p43}.W_V"].view(8, d),
        w_o=default_tensors[f"{p43}.W_O"].view(d, 128),
        eps=eps,
        kv_cache_in=(k_pref, v_pref),
        pre_norm_mod=(s42, sc42),
        residual_gate=g42,
    )
    gqa_bidir_a = gqa_ref(x_gqa_a, attention_mask_mode="bidirectional_block", **gqa_args)
    gqa_bidir_b = gqa_ref(x_gqa_b, attention_mask_mode="bidirectional_block", **gqa_args)
    gqa_causal_a = gqa_ref(x_gqa_a, attention_mask_mode="causal", **gqa_args)
    gqa_causal_b = gqa_ref(x_gqa_b, attention_mask_mode="causal", **gqa_args)

    bidir_tok0_diff = float((gqa_bidir_a["Y"][:, 0] - gqa_bidir_b["Y"][:, 0]).abs().max().item())
    causal_tok0_diff = float((gqa_causal_a["Y"][:, 0] - gqa_causal_b["Y"][:, 0]).abs().max().item())
    if bidir_tok0_diff <= 1e-3:
        raise RuntimeError(f"Bidirectional future-token influence too small: {bidir_tok0_diff}")
    if causal_tok0_diff != 0.0:
        raise RuntimeError(f"Causal mode leaked future token j=3 into i=0: {causal_tok0_diff}")

    cat2 = {
        "bayId": 43,
        "layerIndex": 42,
        "t": 0.75,
        "prefixLen": P_gqa,
        "blockSize": B_gqa,
        "kvCapacity": kv_cap,
        "eps": eps,
        "shift": to_f32_list(s42, "cat2.shift"),
        "scale": to_f32_list(sc42, "cat2.scale"),
        "gate": to_f32_list(g42, "cat2.gate"),
        "KPrefix": to_f32_list(k_pref, "cat2.KPrefix"),
        "VPrefix": to_f32_list(v_pref, "cat2.VPrefix"),
        "X": to_f32_list(x_gqa_a, "cat2.X"),
        "bidirectional": {
            "X_mod": to_f32_list(gqa_bidir_a["X_mod"], "cat2.bidir.X_mod"),
            "U": to_f32_list(gqa_bidir_a["U"], "cat2.bidir.U"),
            "Q": to_f32_list(gqa_bidir_a["Q"], "cat2.bidir.Q"),
            "K": to_f32_list(gqa_bidir_a["K"], "cat2.bidir.K"),
            "V": to_f32_list(gqa_bidir_a["V"], "cat2.bidir.V"),
            "Scores": to_f32_list(gqa_bidir_a["Scores"], "cat2.bidir.Scores"),
            "Softmax": to_f32_list(gqa_bidir_a["Softmax"], "cat2.bidir.Softmax"),
            "V_out": to_f32_list(gqa_bidir_a["V_out"], "cat2.bidir.V_out"),
            "Attn_out": to_f32_list(gqa_bidir_a["Attn_out"], "cat2.bidir.Attn_out"),
            "Y": to_f32_list(gqa_bidir_a["Y"], "cat2.bidir.Y"),
        },
        "futureTokenInfluence": {
            "perturbedSlot": 3,
            "observedEarlierSlot": 0,
            "X_a": to_f32_list(x_gqa_a, "cat2.inf.X_a"),
            "X_b": to_f32_list(x_gqa_b, "cat2.inf.X_b"),
            "Y_bidir_a": to_f32_list(gqa_bidir_a["Y"], "cat2.inf.Y_bidir_a"),
            "Y_bidir_b": to_f32_list(gqa_bidir_b["Y"], "cat2.inf.Y_bidir_b"),
            "Y_causal_a": to_f32_list(gqa_causal_a["Y"], "cat2.inf.Y_causal_a"),
            "Y_causal_b": to_f32_list(gqa_causal_b["Y"], "cat2.inf.Y_causal_b"),
            "causalScoresA": to_nullable_causal_scores(gqa_causal_a["Scores"], "cat2.inf.causalScoresA"),
            "bidirToken0MaxDiff": bidir_tok0_diff,
            "causalToken0MaxDiff": causal_tok0_diff,
        },
    }

    # ------------------------------------------------------------------
    # CAT 3: Seeded Mamba-2 Conv & Recurrence + Prefix-Stitching + O(1e3) Stability
    # ------------------------------------------------------------------
    mamba_weights_b01 = dict(
        gamma_in=default_tensors[f"{p1}.gamma_in"].view(d),
        w_in=default_tensors[f"{p1}.W_in"].view(208, d),
        w_conv=default_tensors[f"{p1}.w_conv"].view(128, 4),
        b_conv=default_tensors[f"{p1}.b_conv"].view(128),
        a_log=default_tensors[f"{p1}.A_log"].view(16),
        dt_bias=default_tensors[f"{p1}.dt_bias"].view(16),
        d_skip=default_tensors[f"{p1}.D"].view(16),
        gamma_gate=default_tensors[f"{p1}.gamma_gate"].view(64),
        w_out=default_tensors[f"{p1}.W_out"].view(d, 64),
        eps=eps,
    )

    # 3A: Modulated & seeded from non-zero (convState, ssmState)
    x_m_seed = torch.randn((d, 4), generator=gen, dtype=torch.float32) * 0.9
    conv_seed = torch.randn((128, 4), generator=gen, dtype=torch.float32) * 0.55
    ssm_seed = torch.randn((16, 4, 4), generator=gen, dtype=torch.float32) * 0.45
    m_seeded = mamba2_ref(
        x_m_seed,
        conv_state_in=conv_seed,
        ssm_state_in=ssm_seed,
        pre_norm_mod=(s0, sc0),
        residual_gate=g0,
        **mamba_weights_b01,
    )

    # 3B: Prefix-stitching parity ([0..11] 1-pass vs [0..7] then seeded [8..11])
    x_stitch_full = torch.randn((d, 12), generator=gen, dtype=torch.float32) * 0.85
    m_full12 = mamba2_ref(x_stitch_full, **mamba_weights_b01)
    m_pref8 = mamba2_ref(x_stitch_full[:, :8].contiguous(), **mamba_weights_b01)
    m_tail4 = mamba2_ref(
        x_stitch_full[:, 8:].contiguous(),
        conv_state_in=m_pref8["conv_state"],
        ssm_state_in=m_pref8["H"],
        **mamba_weights_b01,
    )
    stitch_y_err = float((m_full12["Y"][:, 8:] - m_tail4["Y"]).abs().max().item())
    stitch_h_err = float((m_full12["H"] - m_tail4["H"]).abs().max().item())
    stitch_c_err = float((m_full12["conv_state"] - m_tail4["conv_state"]).abs().max().item())
    if max(stitch_y_err, stitch_h_err, stitch_c_err) > 1e-5:
        raise RuntimeError(
            f"PyTorch Mamba prefix-stitching mismatch: Y={stitch_y_err}, H={stitch_h_err}, conv={stitch_c_err}"
        )

    # 3C: High-magnitude O(1e3) FP32 SSM stability (zero NaN/Inf!)
    x_large = torch.randn((d, 4), generator=gen, dtype=torch.float32) * 1200.0
    conv_large = torch.randn((128, 4), generator=gen, dtype=torch.float32) * 500.0
    ssm_large = torch.randn((16, 4, 4), generator=gen, dtype=torch.float32) * 1000.0
    m_large = mamba2_ref(
        x_large,
        conv_state_in=conv_large,
        ssm_state_in=ssm_large,
        pre_norm_mod=(s0, sc0),
        residual_gate=g0,
        **mamba_weights_b01,
    )

    cat3 = {
        "bayId": 1,
        "layerIndex": 0,
        "t": 0.75,
        "eps": eps,
        "shift": to_f32_list(s0, "cat3.shift"),
        "scale": to_f32_list(sc0, "cat3.scale"),
        "gate": to_f32_list(g0, "cat3.gate"),
        "modulatedSeeded": {
            "tokens": 4,
            "X": to_f32_list(x_m_seed, "cat3.seed.X"),
            "convStateIn": to_f32_list(conv_seed, "cat3.seed.convIn"),
            "ssmStateIn": to_f32_list(ssm_seed, "cat3.seed.ssmIn"),
            "X_mod": to_f32_list(m_seeded["X_mod"], "cat3.seed.X_mod"),
            "U": to_f32_list(m_seeded["U"], "cat3.seed.U"),
            "conv_out": to_f32_list(m_seeded["conv_out"], "cat3.seed.conv_out"),
            "convStateOut": to_f32_list(m_seeded["conv_state"], "cat3.seed.convOut"),
            "dt": to_f32_list(m_seeded["dt"], "cat3.seed.dt"),
            "dA": to_f32_list(m_seeded["dA"], "cat3.seed.dA"),
            "H_prev": to_f32_list(m_seeded["H_prev"], "cat3.seed.H_prev"),
            "H": to_f32_list(m_seeded["H"], "cat3.seed.H"),
            "y": to_f32_list(m_seeded["y"], "cat3.seed.y"),
            "y_norm": to_f32_list(m_seeded["y_norm"], "cat3.seed.y_norm"),
            "Mamba_out": to_f32_list(m_seeded["Mamba_out"], "cat3.seed.Mamba_out"),
            "Y": to_f32_list(m_seeded["Y"], "cat3.seed.Y"),
        },
        "prefixStitching": {
            "prefixLen": 8,
            "blockSize": 4,
            "X_full": to_f32_list(x_stitch_full, "cat3.stitch.X_full"),
            "Y_full_tail": to_f32_list(m_full12["Y"][:, 8:], "cat3.stitch.Y_full_tail"),
            "Y_stitched": to_f32_list(m_tail4["Y"], "cat3.stitch.Y_stitched"),
            "convStateFull": to_f32_list(m_full12["conv_state"], "cat3.stitch.conv_full"),
            "convStateStitched": to_f32_list(m_tail4["conv_state"], "cat3.stitch.conv_stitched"),
            "HFull": to_f32_list(m_full12["H"], "cat3.stitch.H_full"),
            "HStitched": to_f32_list(m_tail4["H"], "cat3.stitch.H_stitched"),
            "pytorchStitchMaxDiff": max(stitch_y_err, stitch_h_err, stitch_c_err),
        },
        "highMagnitudeStability": {
            "tokens": 4,
            "X": to_f32_list(x_large, "cat3.large.X"),
            "convStateIn": to_f32_list(conv_large, "cat3.large.convIn"),
            "ssmStateIn": to_f32_list(ssm_large, "cat3.large.ssmIn"),
            "conv_out": to_f32_list(m_large["conv_out"], "cat3.large.conv_out"),
            "dt": to_f32_list(m_large["dt"], "cat3.large.dt"),
            "dA": to_f32_list(m_large["dA"], "cat3.large.dA"),
            "H": to_f32_list(m_large["H"], "cat3.large.H"),
            "y": to_f32_list(m_large["y"], "cat3.large.y"),
            "y_norm": to_f32_list(m_large["y_norm"], "cat3.large.y_norm"),
            "Mamba_out": to_f32_list(m_large["Mamba_out"], "cat3.large.Mamba_out"),
            "Y": to_f32_list(m_large["Y"], "cat3.large.Y"),
        },
    }

    # ------------------------------------------------------------------
    # CAT 4: FP64 MoE Routing & Gated Residuals + Near-Tie vs Exact-Tie
    # ------------------------------------------------------------------
    moe_weights_b02 = dict(
        gamma_in=default_tensors[f"{p2}.gamma_in"].view(d),
        w_router=default_tensors[f"{p2}.W_router"].view(128, d),
        e_bias=default_tensors[f"{p2}.e_score_correction_bias"].view(128),
        w_up=default_tensors[f"{p2}.W_up"].view(128, 4, d),
        w_down=default_tensors[f"{p2}.W_down"].view(128, d, 4),
        w_up_s=default_tensors[f"{p2}.W_up_s"].view(8, d),
        w_down_s=default_tensors[f"{p2}.W_down_s"].view(d, 8),
        eps=eps,
    )
    x_moe = torch.randn((d, 4), generator=gen, dtype=torch.float32) * 0.9
    moe_mod = moe_ref(
        x_moe,
        router_precision_policy="float64_reference",
        post_norm_mod=(s1, sc1),
        residual_gate=g1,
        **moe_weights_b02,
    )

    # 4B: Near-tie router fixture (~2e-8 margin in Float64 -> flips in Float32!)
    # Single non-zero weight in row 7 guarantees bit-identical logit in PyTorch & TS.
    x_tie = torch.ones((d, 1), dtype=torch.float32)
    gamma_tie = torch.ones((d,), dtype=torch.float32)
    w_router_near = torch.zeros((128, d), dtype=torch.float32)
    bias_near = torch.full((128,), -0.5, dtype=torch.float32)
    # Clear Top-5 experts: indices [0, 1, 3, 4, 5]
    for rank_i, e_idx in enumerate([0, 1, 3, 4, 5]):
        w_router_near[e_idx, 0] = 1.5 - 0.1 * rank_i
        bias_near[e_idx] = 0.35
    # Rank #6 vs Rank #7 battle: expert 2 (choice64 = 1.0) vs expert 7 (choice64 = 1.0 + ~2.0e-8)
    w_router_near[2, 0] = 0.0
    bias_near[2] = 0.5
    w_router_near[7, 0] = 8.0e-8  # z64 ~= 8.0e-8 -> sigmoid(z64) ~= 0.5 + 2.0e-8 -> rounds to 0.5 in FP32!
    bias_near[7] = 0.5

    near_f64 = moe_ref(
        x_tie,
        gamma_in=gamma_tie,
        w_router=w_router_near,
        e_bias=bias_near,
        w_up=moe_weights_b02["w_up"],
        w_down=moe_weights_b02["w_down"],
        w_up_s=moe_weights_b02["w_up_s"],
        w_down_s=moe_weights_b02["w_down_s"],
        eps=eps,
        router_precision_policy="float64_reference",
    )
    near_f32 = moe_ref(
        x_tie,
        gamma_in=gamma_tie,
        w_router=w_router_near,
        e_bias=bias_near,
        w_up=moe_weights_b02["w_up"],
        w_down=moe_weights_b02["w_down"],
        w_up_s=moe_weights_b02["w_up_s"],
        w_down_s=moe_weights_b02["w_down_s"],
        eps=eps,
        router_precision_policy="float32_nano",
    )
    u_tie64 = rmsnorm_ref(x_tie, gamma_tie, eps)["U"].to(torch.float64)
    c64_e2 = float((torch.sigmoid(w_router_near[2].to(torch.float64) @ u_tie64[:, 0]) + bias_near[2].to(torch.float64)).item())
    c64_e7 = float((torch.sigmoid(w_router_near[7].to(torch.float64) @ u_tie64[:, 0]) + bias_near[7].to(torch.float64)).item())
    near_margin_f64 = c64_e7 - c64_e2

    topk_near_f64 = to_i32_list(near_f64["topk_indices"])
    topk_near_f32 = to_i32_list(near_f32["topk_indices"])
    if not (5e-9 < near_margin_f64 < 5e-8):
        raise RuntimeError(f"Unexpected near-tie margin: {near_margin_f64:.3e}")
    if topk_near_f64 != [0, 1, 3, 4, 5, 7] or topk_near_f32 != [0, 1, 3, 4, 5, 2]:
        raise RuntimeError(
            f"Near-tie fixture failed to produce expected FP64 vs FP32 flip: "
            f"f64={topk_near_f64}, f32={topk_near_f32}"
        )

    # 4C: Exact-tie router fixture (bit-identical Float64 choice scores -> deterministic lower-index tie-break)
    w_router_exact = torch.zeros((128, d), dtype=torch.float32)
    bias_exact = torch.full((128,), -0.5, dtype=torch.float32)
    for rank_i, e_idx in enumerate([10, 20, 30, 40]):
        w_router_exact[e_idx, 0] = 1.5 - 0.1 * rank_i
        bias_exact[e_idx] = 0.4
    # 4 experts with BIT-IDENTICAL choice scores fighting for remaining 2 slots (#5 and #6):
    # indices [5, 17, 64, 99] -> deterministic lower-index tie-break MUST pick [5, 17]!
    for e_idx in [5, 17, 64, 99]:
        w_router_exact[e_idx, 0] = 0.25
        bias_exact[e_idx] = 0.3

    exact_f64 = moe_ref(
        x_tie,
        gamma_in=gamma_tie,
        w_router=w_router_exact,
        e_bias=bias_exact,
        w_up=moe_weights_b02["w_up"],
        w_down=moe_weights_b02["w_down"],
        w_up_s=moe_weights_b02["w_up_s"],
        w_down_s=moe_weights_b02["w_down_s"],
        eps=eps,
        router_precision_policy="float64_reference",
    )
    topk_exact_f64 = to_i32_list(exact_f64["topk_indices"])
    if topk_exact_f64 != [10, 20, 30, 40, 5, 17]:
        raise RuntimeError(f"Exact-tie fixture unexpected topk: {topk_exact_f64}")

    cat4 = {
        "bayId": 2,
        "layerIndex": 1,
        "t": 0.75,
        "eps": eps,
        "shift": to_f32_list(s1, "cat4.shift"),
        "scale": to_f32_list(sc1, "cat4.scale"),
        "gate": to_f32_list(g1, "cat4.gate"),
        "modulatedFp64": {
            "tokens": 4,
            "X": to_f32_list(x_moe, "cat4.mod.X"),
            "U": to_f32_list(moe_mod["U"], "cat4.mod.U"),
            "U_mod": to_f32_list(moe_mod["U_mod"], "cat4.mod.U_mod"),
            "scores": to_f32_list(moe_mod["scores"], "cat4.mod.scores"),
            "scores_for_choice": to_f32_list(moe_mod["scores_for_choice"], "cat4.mod.choice"),
            "topk_indices": to_i32_list(moe_mod["topk_indices"]),
            "topk_raw_scores": to_f32_list(moe_mod["topk_raw_scores"], "cat4.mod.topk_raw"),
            "topk_weights": to_f32_list(moe_mod["topk_weights"], "cat4.mod.topk_w"),
            "y_shared": to_f32_list(moe_mod["y_shared"], "cat4.mod.y_shared"),
            "y_routed": to_f32_list(moe_mod["y_routed"], "cat4.mod.y_routed"),
            "y_moe": to_f32_list(moe_mod["y_moe"], "cat4.mod.y_moe"),
            "Y": to_f32_list(moe_mod["Y"], "cat4.mod.Y"),
        },
        "nearTieRouter": {
            "tokens": 1,
            "expertWeightsBayId": 2,
            "X": to_f32_list(x_tie, "cat4.near.X"),
            "gamma_in": to_f32_list(gamma_tie, "cat4.near.gamma"),
            "W_router": to_f32_list(w_router_near, "cat4.near.W_router"),
            "e_score_correction_bias": to_f32_list(bias_near, "cat4.near.bias"),
            "float64MarginRank6VsRank7": near_margin_f64,
            "float64TopkIndices": topk_near_f64,
            "float64TopkWeights": to_f32_list(near_f64["topk_weights"], "cat4.near.f64.w"),
            "float64Y": to_f32_list(near_f64["Y"], "cat4.near.f64.Y"),
            "float32TopkIndices": topk_near_f32,
            "float32TopkWeights": to_f32_list(near_f32["topk_weights"], "cat4.near.f32.w"),
            "float32Y": to_f32_list(near_f32["Y"], "cat4.near.f32.Y"),
        },
        "exactTieRouter": {
            "tokens": 1,
            "expertWeightsBayId": 2,
            "X": to_f32_list(x_tie, "cat4.exact.X"),
            "gamma_in": to_f32_list(gamma_tie, "cat4.exact.gamma"),
            "W_router": to_f32_list(w_router_exact, "cat4.exact.W_router"),
            "e_score_correction_bias": to_f32_list(bias_exact, "cat4.exact.bias"),
            "tiedExperts": [5, 17, 64, 99],
            "expectedTopkIndices": topk_exact_f64,
            "expectedTopkWeights": to_f32_list(exact_f64["topk_weights"], "cat4.exact.w"),
            "expectedY": to_f32_list(exact_f64["Y"], "cat4.exact.Y"),
        },
    }

    return {
        "metadata": {
            "generator": "scripts/reference/generate_twotower_pytorch_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "referenceArchitecture": "Nemotron-Labs-TwoTower Clean-Room Miniature Blocks (Cat 1-4)",
            "precisionNote": (
                "Generated in pinned Python 3.12 / PyTorch 2.15.0 using exact SHA-256-verified "
                "Float32 miniature parameters from createDefaultTwoTowerModel(). Uses PyTorch "
                "FP32 sinusoidal freqs, F.conv1d, FP32 Mamba-2 SSM scan, F.scaled_dot_product_attention, "
                "and FP64 MoE router selection."
            ),
            "tolerances": {
                "modulationMaxAbs": 1e-5,
                "gqaMaxAbs": 1e-5,
                "mamba2MaxAbs": 2e-5,
                "mamba2HighMagRel": 1e-5,
                "moeMaxAbs": 2e-5,
            },
        },
        "modelId": default_manifest["modelId"],
        "twoTowerFingerprint": default_manifest["twoTowerFingerprint"],
        "contextFingerprint": default_manifest["contextFingerprint"],
        "denoiserFingerprint": default_manifest["denoiserFingerprint"],
        "timeConditioningFingerprint": default_manifest["timeConditioningFingerprint"],
        "weightsSha256": default_manifest["weightsSha256"],
        "totalWeightFloats": default_manifest["totalFloats"],
        "cat1TimestepAndModulation": cat1,
        "cat2BidirectionalGqa": cat2,
        "cat3SeededMamba2": cat3,
        "cat4Fp64MoeRouting": cat4,
    }


# ============================================================================
# FIXTURE 2: COMPLETE 52-LAYER CONTEXT-TO-DENOISER HANDOFF (CATEGORY 5)
# ============================================================================

def build_handoff_parity_fixture(
    default_manifest: dict[str, Any],
    default_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    prompt_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    kv_capacity = 24
    mask_token_id = int(default_manifest["maskTokenId"])
    noisy_block = [mask_token_id, mask_token_id, mask_token_id, mask_token_id]

    ctx = context_prefill_ref(default_manifest, default_tensors, prompt_ids)
    # Snapshot pre-handoff checksums in Python to self-verify Context immutability
    pre_mamba_b01 = (ctx["ctxMamba"][1][0].clone(), ctx["ctxMamba"][1][1].clone())
    pre_gqa_b43 = (ctx["ctxGqa"][43][0].clone(), ctx["ctxGqa"][43][1].clone())

    den = run_denoiser_pass_ref(
        default_manifest,
        default_tensors,
        ctx,
        noisy_block,
        mode="mask_diffusion",
        capture_bays={1, 2, 43},
    )

    # Assert 0-ULP Context immutability in PyTorch reference
    assert torch.equal(ctx["ctxMamba"][1][0], pre_mamba_b01[0])
    assert torch.equal(ctx["ctxMamba"][1][1], pre_mamba_b01[1])
    assert torch.equal(ctx["ctxGqa"][43][0], pre_gqa_b43[0])
    assert torch.equal(ctx["ctxGqa"][43][1], pre_gqa_b43[1])

    rep_mamba_bays = [1, 12, 33, 51]
    ctx_mamba_json = {
        str(b): {
            "convState": to_f32_list(ctx["ctxMamba"][b][0], f"cat5.ctx.m{b}.conv"),
            "ssmState": to_f32_list(ctx["ctxMamba"][b][1], f"cat5.ctx.m{b}.ssm"),
        }
        for b in rep_mamba_bays
    }
    ctx_mamba_s2_json = {
        str(b): {
            "convState": to_f32_list(ctx["mambaS2"][b][0], f"cat5.s2.m{b}.conv"),
            "ssmState": to_f32_list(ctx["mambaS2"][b][1], f"cat5.s2.m{b}.ssm"),
        }
        for b in rep_mamba_bays
    }
    ctx_gqa_json = {
        str(b): {
            "K": to_f32_list(ctx["ctxGqa"][b][0], f"cat5.ctx.g{b}.K"),
            "V": to_f32_list(ctx["ctxGqa"][b][1], f"cat5.ctx.g{b}.V"),
        }
        for b in sorted(ctx["ctxGqa"].keys())
    }

    tr_m = den["traces"][1]
    tr_e = den["traces"][2]
    tr_g = den["traces"][43]

    top5_by_pos: list[list[int]] = []
    for j in range(len(noisy_block)):
        _, top5 = torch.topk(den["logits"][j], k=5, largest=True, sorted=True)
        top5_by_pos.append([int(v) for v in top5.tolist()])

    return {
        "metadata": {
            "generator": "scripts/reference/generate_twotower_pytorch_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "referenceArchitecture": "Nemotron-Labs-TwoTower 52-Layer Context-to-Denoiser Handoff (Cat 5)",
            "tolerances": {
                "embeddingMaxAbs": 0.0,
                "modulationMaxAbs": 1e-5,
                "cacheStateMaxAbs": 2e-5,
                "residualCheckpointMaxAbs": 2e-5,
                "traceMaxAbs": 2e-5,
                "finalNormMaxAbs": 2e-5,
                "lmHeadLogitsMaxAbs": 5e-5,
                "lmHeadProbsMaxAbs": 2e-5,
            },
        },
        "modelId": default_manifest["modelId"],
        "twoTowerFingerprint": default_manifest["twoTowerFingerprint"],
        "contextFingerprint": default_manifest["contextFingerprint"],
        "denoiserFingerprint": default_manifest["denoiserFingerprint"],
        "timeConditioningFingerprint": default_manifest["timeConditioningFingerprint"],
        "weightsSha256": default_manifest["weightsSha256"],
        "totalWeightFloats": default_manifest["totalFloats"],
        "promptTokenIds": prompt_ids,
        "kvCapacity": kv_capacity,
        "blockTokenIds": noisy_block,
        "blockId": 0,
        "refinementIndex": 0,
        "tModel": den["tModel"],
        "contextPrefill": {
            "prefixLen": len(prompt_ids),
            "arLogits": to_f32_list(ctx["arLogits"], "cat5.ctx.arLogits"),
            "arProbs": to_f32_list(ctx["arProbs"], "cat5.ctx.arProbs"),
            "mambaS1": ctx_mamba_json,
            "mambaS2": ctx_mamba_s2_json,
            "gqaKv": ctx_gqa_json,
        },
        "denoiserPass": {
            "timestepProjection": {
                "phi": to_f32_list(den["timestepProjection"]["phi"], "cat5.den.phi"),
                "tRepr": to_f32_list(den["timestepProjection"]["tRepr"], "cat5.den.tRepr"),
                "tEmb": to_f32_list(den["timestepProjection"]["tEmb"], "cat5.den.tEmb"),
            },
            "residualStreams": [
                to_f32_list(r, f"cat5.den.res[{idx}]") for idx, r in enumerate(den["residuals"])
            ],
            "moeTopkByBay": den["moeTopkByBay"],
            "representativeTraces": {
                "bay01Mamba": {
                    "bayId": 1,
                    "shift": to_f32_list(tr_m["mod"][0], "cat5.m1.shift"),
                    "scale": to_f32_list(tr_m["mod"][1], "cat5.m1.scale"),
                    "gate": to_f32_list(tr_m["mod"][2], "cat5.m1.gate"),
                    "U": to_f32_list(tr_m["out"]["U"], "cat5.m1.U"),
                    "conv_out": to_f32_list(tr_m["out"]["conv_out"], "cat5.m1.conv_out"),
                    "convState": to_f32_list(tr_m["out"]["conv_state"], "cat5.m1.convState"),
                    "H_prev": to_f32_list(tr_m["out"]["H_prev"], "cat5.m1.H_prev"),
                    "H": to_f32_list(tr_m["out"]["H"], "cat5.m1.H"),
                    "y_norm": to_f32_list(tr_m["out"]["y_norm"], "cat5.m1.y_norm"),
                    "Mamba_out": to_f32_list(tr_m["out"]["Mamba_out"], "cat5.m1.Mamba_out"),
                    "Y": to_f32_list(tr_m["out"]["Y"], "cat5.m1.Y"),
                },
                "bay02Moe": {
                    "bayId": 2,
                    "shift": to_f32_list(tr_e["mod"][0], "cat5.e2.shift"),
                    "scale": to_f32_list(tr_e["mod"][1], "cat5.e2.scale"),
                    "gate": to_f32_list(tr_e["mod"][2], "cat5.e2.gate"),
                    "U": to_f32_list(tr_e["out"]["U"], "cat5.e2.U"),
                    "U_mod": to_f32_list(tr_e["out"]["U_mod"], "cat5.e2.U_mod"),
                    "topk_indices": to_i32_list(tr_e["out"]["topk_indices"]),
                    "topk_weights": to_f32_list(tr_e["out"]["topk_weights"], "cat5.e2.topk_w"),
                    "y_shared": to_f32_list(tr_e["out"]["y_shared"], "cat5.e2.y_shared"),
                    "y_routed": to_f32_list(tr_e["out"]["y_routed"], "cat5.e2.y_routed"),
                    "y_moe": to_f32_list(tr_e["out"]["y_moe"], "cat5.e2.y_moe"),
                    "Y": to_f32_list(tr_e["out"]["Y"], "cat5.e2.Y"),
                },
                "bay43Gqa": {
                    "bayId": 43,
                    "shift": to_f32_list(tr_g["mod"][0], "cat5.g43.shift"),
                    "scale": to_f32_list(tr_g["mod"][1], "cat5.g43.scale"),
                    "gate": to_f32_list(tr_g["mod"][2], "cat5.g43.gate"),
                    "U": to_f32_list(tr_g["out"]["U"], "cat5.g43.U"),
                    "Q": to_f32_list(tr_g["out"]["Q"], "cat5.g43.Q"),
                    "K": to_f32_list(tr_g["out"]["K"], "cat5.g43.K"),
                    "V": to_f32_list(tr_g["out"]["V"], "cat5.g43.V"),
                    "Scores": to_f32_list(tr_g["out"]["Scores"], "cat5.g43.Scores"),
                    "Softmax": to_f32_list(tr_g["out"]["Softmax"], "cat5.g43.Softmax"),
                    "V_out": to_f32_list(tr_g["out"]["V_out"], "cat5.g43.V_out"),
                    "Attn_out": to_f32_list(tr_g["out"]["Attn_out"], "cat5.g43.Attn_out"),
                    "Y": to_f32_list(tr_g["out"]["Y"], "cat5.g43.Y"),
                },
            },
            "finalNorm": {
                "rms": to_f32_list(den["finalNorm"]["rms"], "cat5.fn.rms"),
                "X_norm": to_f32_list(den["finalNorm"]["X_norm"], "cat5.fn.X_norm"),
                "U_final": to_f32_list(den["finalNorm"]["U_final"], "cat5.fn.U_final"),
            },
            "logits": to_f32_list(den["logits"], "cat5.den.logits"),
            "probs": to_f32_list(den["probs"], "cat5.den.probs"),
            "top5ByPosition": top5_by_pos,
        },
    }


# ============================================================================
# FIXTURE 3: TRAJECTORY, AR vs MOCK-AR ALIGNMENT & SAMPLER EDGE CASES (CAT 6-8)
# ============================================================================

def serialize_ctx_checkpoint(ctx: dict[str, Any], tag: str) -> dict[str, Any]:
    rep_mamba = [1, 12, 33, 51]
    return {
        "prefixLen": len(ctx["prefixIds"]),
        "prefixTokenIds": list(ctx["prefixIds"]),
        "arLogits": to_f32_list(ctx["arLogits"], f"{tag}.arLogits"),
        "arProbs": to_f32_list(ctx["arProbs"], f"{tag}.arProbs"),
        "mambaS1": {
            str(b): {
                "convState": to_f32_list(ctx["ctxMamba"][b][0], f"{tag}.m{b}.conv"),
                "ssmState": to_f32_list(ctx["ctxMamba"][b][1], f"{tag}.m{b}.ssm"),
            }
            for b in rep_mamba
        },
        "mambaS2Bay01": {
            "convState": to_f32_list(ctx["mambaS2"][1][0], f"{tag}.s2.m1.conv"),
            "ssmState": to_f32_list(ctx["mambaS2"][1][1], f"{tag}.s2.m1.ssm"),
        },
        "gqaKv": {
            str(b): {
                "K": to_f32_list(ctx["ctxGqa"][b][0], f"{tag}.g{b}.K"),
                "V": to_f32_list(ctx["ctxGqa"][b][1], f"{tag}.g{b}.V"),
            }
            for b in sorted(ctx["ctxGqa"].keys())
        },
    }


def run_two_step_ar_and_mock_ar_ref(
    manifest: dict[str, Any],
    tensor_map: dict[str, torch.Tensor],
    prompt_ids: list[int],
    tag: str,
) -> dict[str, Any]:
    # 1. Context AR (2 steps: P=8 -> 9 -> 10)
    ctx_ar = context_prefill_ref(manifest, tensor_map, prompt_ids)
    ar_steps: list[dict[str, Any]] = []
    for s_idx in range(2):
        logits_before = ctx_ar["arLogits"].clone()
        probs_before = ctx_ar["arProbs"].clone()
        tok_id = int(torch.argmax(logits_before).item())
        ctx_ar = context_commit_ref(manifest, tensor_map, ctx_ar, [tok_id])
        ar_steps.append({
            "stepIndex": s_idx,
            "tokenId": tok_id,
            "logits": to_f32_list(logits_before, f"{tag}.ar[{s_idx}].logits"),
            "probs": to_f32_list(probs_before, f"{tag}.ar[{s_idx}].probs"),
            "postCommitArLogits": to_f32_list(ctx_ar["arLogits"], f"{tag}.ar[{s_idx}].postArLogits"),
            "postCommitBay01Ssm": to_f32_list(ctx_ar["ctxMamba"][1][1], f"{tag}.ar[{s_idx}].ssm"),
            "postCommitBay43K": to_f32_list(ctx_ar["ctxGqa"][43][0], f"{tag}.ar[{s_idx}].K"),
            "postCommitBay43V": to_f32_list(ctx_ar["ctxGqa"][43][1], f"{tag}.ar[{s_idx}].V"),
        })

    # 2. Mock-AR (2 steps: P=8 -> 9 -> 10, denoiser causal with KV[:-1] + mambaS2, tEmb=None)
    ctx_mock = context_prefill_ref(manifest, tensor_map, prompt_ids)
    mock_steps: list[dict[str, Any]] = []
    for s_idx in range(2):
        last_tok = ctx_mock["prefixIds"][-1]
        den = run_denoiser_pass_ref(
            manifest,
            tensor_map,
            ctx_mock,
            [last_tok],
            mode="mock_ar",
        )
        den_logits = den["logits"][0]
        den_probs = den["probs"][0]
        tok_id = int(torch.argmax(den_logits).item())
        ctx_mock = context_commit_ref(manifest, tensor_map, ctx_mock, [tok_id])
        mock_steps.append({
            "stepIndex": s_idx,
            "queryTokenId": last_tok,
            "committedPrefixLen": len(ctx_mock["prefixIds"]) - 2,
            "tokenId": tok_id,
            "denoiserLogits": to_f32_list(den_logits, f"{tag}.mock[{s_idx}].logits"),
            "denoiserProbs": to_f32_list(den_probs, f"{tag}.mock[{s_idx}].probs"),
            "residualStreams": [
                to_f32_list(r, f"{tag}.mock[{s_idx}].res[{i}]")
                for i, r in enumerate(den["residuals"])
            ],
            "postCommitArLogits": to_f32_list(ctx_mock["arLogits"], f"{tag}.mock[{s_idx}].postArLogits"),
            "postCommitBay01Ssm": to_f32_list(ctx_mock["ctxMamba"][1][1], f"{tag}.mock[{s_idx}].ssm"),
            "postCommitBay43K": to_f32_list(ctx_mock["ctxGqa"][43][0], f"{tag}.mock[{s_idx}].K"),
            "postCommitBay43V": to_f32_list(ctx_mock["ctxGqa"][43][1], f"{tag}.mock[{s_idx}].V"),
        })

    return {
        "arSteps": ar_steps,
        "mockArSteps": mock_steps,
    }


def build_trajectory_parity_fixture(
    default_manifest: dict[str, Any],
    default_tensors: dict[str, torch.Tensor],
    identity_manifest: dict[str, Any],
    identity_tensors: dict[str, torch.Tensor],
    gen: torch.Generator,
) -> dict[str, Any]:
    prompt_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    kv_capacity = 24
    mask_token_id = int(default_manifest["maskTokenId"])
    eos_token_id = int(default_manifest["eosTokenId"])
    vocab_size = int(default_manifest["vocabSize"])

    # ------------------------------------------------------------------
    # CAT 6: Full Two-Block Diffusion Trajectory (P_0=8 -> 12 -> 16)
    # Choose a clean threshold that exercises BOTH multi-token threshold commit AND floor commit!
    # First probe initial step confidences to select a well-separated threshold (with healthy margin
    # away from any confidence boundary so FP32/FP64 drift can never flip a decision!).
    # ------------------------------------------------------------------
    ctx_probe = context_prefill_ref(default_manifest, default_tensors, prompt_ids)
    den_probe = run_denoiser_pass_ref(
        default_manifest, default_tensors, ctx_probe, [mask_token_id] * 4, mode="mask_diffusion"
    )
    _, x_theta_probe = mdlm_forward_ref(den_probe["logits"], [mask_token_id] * 4, mask_token_id)
    probe_confs = sorted([float(x_theta_probe[j].max().item()) for j in range(4)])
    # Midpoint between 2nd-lowest and 2nd-highest confidence ensures ~2 tokens exceed threshold on step 0
    # while lower-confidence slots require subsequent steps / floor commits.
    # Let's check actual candidate thresholds cleanly:
    candidate_thresholds = [0.22, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05]
    chosen_threshold = 0.22
    for cand_thr in candidate_thresholds:
        # Simulate 2 blocks to see if both threshold multi-token AND floor commits are exercised
        # with comfortable margin >= 2e-3 away from cand_thr!
        ctx_sim = context_prefill_ref(default_manifest, default_tensors, prompt_ids)
        saw_multi_threshold = False
        saw_floor = False
        min_margin = float("inf")
        stopped_early_eos = False
        for _blk in range(2):
            xt_s = [mask_token_id] * 4
            for r in range(4):
                if all(tok != mask_token_id for tok in xt_s):
                    break
                d_s = run_denoiser_pass_ref(default_manifest, default_tensors, ctx_sim, xt_s, mode="mask_diffusion")
                u_s = confidence_unmask_step_ref(
                    xt_s, d_s["logits"], mask_token_id, r, 4, cand_thr, 0.0, [eos_token_id]
                )
                for j_idx, c_val in enumerate(u_s["confidence"]):
                    if not math.isinf(c_val):
                        min_margin = min(min_margin, abs(c_val - cand_thr))
                if u_s["commitReason"] == "threshold" and len(u_s["committedIndices"]) >= 2:
                    saw_multi_threshold = True
                if u_s["commitReason"] in ("progress_floor", "budget_floor"):
                    saw_floor = True
                xt_s = u_s["xtNext"]
                if u_s["isBlockComplete"]:
                    break
            ctx_sim = context_commit_ref(default_manifest, default_tensors, ctx_sim, xt_s)
            if eos_token_id in xt_s:
                stopped_early_eos = True
                break
        if saw_multi_threshold and saw_floor and min_margin >= 1e-3 and not stopped_early_eos:
            chosen_threshold = cand_thr
            break
    else:
        # Fallback: pick midpoint of probe_confs[1] and probe_confs[2] rounded to 4 decimals
        chosen_threshold = round(0.5 * (probe_confs[1] + probe_confs[2]), 4)

    # Now execute the authoritative 2-block diffusion trajectory with `chosen_threshold`
    ctx_traj = context_prefill_ref(default_manifest, default_tensors, prompt_ids)
    blocks_out: list[dict[str, Any]] = []
    generated_tokens: list[int] = []
    total_nfe = 0
    stopped_by_eos = False

    for block_id in range(2):
        xt = [mask_token_id] * 4
        refinements_json: list[dict[str, Any]] = []
        for r in range(4):
            if all(tok != mask_token_id for tok in xt):
                break
            den = run_denoiser_pass_ref(
                default_manifest, default_tensors, ctx_traj, xt, mode="mask_diffusion"
            )
            step = confidence_unmask_step_ref(
                xt=xt,
                logits=den["logits"],
                mask_token_id=mask_token_id,
                step_index=r,
                steps_per_block=4,
                confidence_threshold=chosen_threshold,
                temperature=0.0,
                eos_token_ids=[eos_token_id],
            )
            refinements_json.append({
                "refinementIndex": r,
                "tModel": den["tModel"],
                "xtBefore": step["xtBefore"],
                "logits": to_f32_list(den["logits"], f"cat6.b{block_id}.r{r}.logits"),
                "confidence": to_nullable_confidence(step["confidence"], f"cat6.b{block_id}.r{r}.conf"),
                "predicted": step["predicted"],
                "committedIndices": step["committedIndices"],
                "thresholdCommittedIndices": step["thresholdCommittedIndices"],
                "floorCommittedIndices": step["floorCommittedIndices"],
                "remaskedIndices": step["remaskedIndices"],
                "commitReason": step["commitReason"],
                "nMaskedBefore": step["nMaskedBefore"],
                "nMaskedAfter": step["nMaskedAfter"],
                "xtNext": step["xtNext"],
                "isBlockComplete": step["isBlockComplete"],
                "earlyCompleted": step["earlyCompleted"],
            })
            xt = step["xtNext"]
            if step["isBlockComplete"]:
                break

        nfe = len(refinements_json)
        total_nfe += nfe
        early_comp = True if nfe == 0 else refinements_json[-1]["earlyCompleted"]
        ctx_traj = context_commit_ref(default_manifest, default_tensors, ctx_traj, xt)
        block_eos = eos_token_id in xt
        generated_tokens.extend(xt)

        blocks_out.append({
            "blockId": block_id,
            "blockSize": 4,
            "stepsPerBlock": 4,
            "nfe": nfe,
            "earlyCompleted": early_comp,
            "finalBlockTokens": list(xt),
            "stoppedByEos": block_eos,
            "refinements": refinements_json,
            "postCommitContext": serialize_ctx_checkpoint(ctx_traj, f"cat6.b{block_id}.postCtx"),
        })
        if block_eos:
            stopped_by_eos = True
            break

    cat6 = {
        "promptTokenIds": prompt_ids,
        "numBlocks": 2,
        "blockSize": 4,
        "stepsPerBlock": 4,
        "kvCapacity": kv_capacity,
        "temperature": 0.0,
        "confidenceThreshold": chosen_threshold,
        "eosTokenIds": [eos_token_id],
        "blocks": blocks_out,
        "generatedTokenIds": generated_tokens,
        "fullSequenceTokenIds": prompt_ids + generated_tokens,
        "totalNfe": total_nfe,
        "stoppedByEos": stopped_by_eos,
        "completedBlocks": len(blocks_out),
    }

    # ------------------------------------------------------------------
    # CAT 7: AR & Mock-AR Alignment Under Comparable Parameters & Policies
    # ------------------------------------------------------------------
    perturbed_align = run_two_step_ar_and_mock_ar_ref(
        default_manifest, default_tensors, prompt_ids, "cat7.perturbed"
    )
    identity_align = run_two_step_ar_and_mock_ar_ref(
        identity_manifest, identity_tensors, prompt_ids, "cat7.identity"
    )

    # Self-verify in PyTorch: under perturbationScale=0, Mock-AR === Context AR!
    for s_idx in range(2):
        ar_s = identity_align["arSteps"][s_idx]
        mk_s = identity_align["mockArSteps"][s_idx]
        assert ar_s["tokenId"] == mk_s["tokenId"], (
            f"Identity Mock-AR token mismatch at step {s_idx}: {ar_s['tokenId']} != {mk_s['tokenId']}"
        )
        max_logit_diff = max(
            abs(a - b) for a, b in zip(ar_s["logits"], mk_s["denoiserLogits"])
        )
        if max_logit_diff > 1e-6:
            raise RuntimeError(
                f"Identity Mock-AR vs Context AR logit mismatch at step {s_idx}: {max_logit_diff}"
            )

    cat7 = {
        "promptTokenIds": prompt_ids,
        "kvCapacity": kv_capacity,
        "numSteps": 2,
        "defaultPerturbed": perturbed_align,
        "identityZeroPerturbation": identity_align,
    }

    # ------------------------------------------------------------------
    # CAT 8: Sampler Edge Cases (Mask Exclusion, Strict > & Floor, Early Exit, EOS, Gumbel-Max)
    # ------------------------------------------------------------------
    # 8A: Mask exclusion & locked carry-over delta
    logits_mask = torch.randn((4, vocab_size), generator=gen, dtype=torch.float32) * 1.1
    logits_mask[:, mask_token_id] = 50.0  # Huge logit on [MASK] MUST be suppressed to prob=0!
    xt_mask = [5, mask_token_id, mask_token_id, 12]
    step_mask = confidence_unmask_step_ref(
        xt=xt_mask,
        logits=logits_mask,
        mask_token_id=mask_token_id,
        step_index=0,
        steps_per_block=4,
        confidence_threshold=0.95,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_mask["unscaledProbs"][:, mask_token_id].abs().max().item() == 0.0
    assert step_mask["predicted"][0] == 5 and step_mask["predicted"][3] == 12
    assert step_mask["predicted"][1] != mask_token_id and step_mask["predicted"][2] != mask_token_id

    # 8B: Strict '>' threshold (exact conf === 0.5 excluded!) + budget floor (K=2 < B=4 -> ceil(4/2)=2)
    logits_floor = torch.full((4, vocab_size), -100.0, dtype=torch.float32)
    # Slot 0: exact 2-way tie -> x_theta = 0.5000000 === threshold (strict '>' MUST exclude!)
    logits_floor[0, 0] = 0.0
    logits_floor[0, 1] = 0.0
    # Slot 1: exact 4-way tie -> x_theta = 0.25
    logits_floor[1, 2:6] = 0.0
    # Slot 2: confident winner -> sigmoid(1.2) ~= 0.7685 > 0.5
    logits_floor[2, 5] = 1.2
    logits_floor[2, 6] = 0.0
    # Slot 3: 3-way split -> exp(0.2)/(exp(0.2)+2) ~= 0.3795 < 0.5
    logits_floor[3, 8] = 0.2
    logits_floor[3, 9] = 0.0
    logits_floor[3, 10] = 0.0

    step_budget_floor = confidence_unmask_step_ref(
        xt=[mask_token_id] * 4,
        logits=logits_floor,
        mask_token_id=mask_token_id,
        step_index=0,
        steps_per_block=2,
        confidence_threshold=0.5,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_budget_floor["confidence"][0] == 0.5
    assert step_budget_floor["numAboveThreshold"] == 1
    assert step_budget_floor["minCommitFloor"] == 2
    assert step_budget_floor["tokensCommittedCount"] == 2
    assert step_budget_floor["commitReason"] == "budget_floor"
    assert step_budget_floor["committedIndices"] == [0, 2]
    assert step_budget_floor["thresholdCommittedIndices"] == [2]
    assert step_budget_floor["floorCommittedIndices"] == [0]
    assert step_budget_floor["remaskedIndices"] == [1, 3]

    # 8B-ii: Progress floor with exact equality (all masked slots <= 0.5, slot 0 === 0.5 -> commits 1 via progress_floor)
    logits_prog = logits_floor.clone()
    logits_prog[2, 5] = 0.0  # Now slot 2 is 2-way tie = 0.5 as well -> numAboveThreshold == 0!
    step_progress_floor = confidence_unmask_step_ref(
        xt=[mask_token_id] * 4,
        logits=logits_prog,
        mask_token_id=mask_token_id,
        step_index=0,
        steps_per_block=4,
        confidence_threshold=0.5,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_progress_floor["numAboveThreshold"] == 0
    assert step_progress_floor["tokensCommittedCount"] == 1
    assert step_progress_floor["commitReason"] == "progress_floor"
    # Slot 0 and slot 2 both have conf=0.5; ascending sort puts (0.5, 0) before (0.5, 2),
    # so the highest-ranked committed slot is index 2 and remasked are [0, 1, 3]!
    assert step_progress_floor["committedIndices"] == [2]
    assert step_progress_floor["remaskedIndices"] == [0, 1, 3]

    # 8C: Early completion (already unmasked no-op AND mid-budget early finish)
    logits_early_noop = torch.randn((4, vocab_size), generator=gen, dtype=torch.float32)
    step_early_noop = confidence_unmask_step_ref(
        xt=[4, 7, 11, 2],
        logits=logits_early_noop,
        mask_token_id=mask_token_id,
        step_index=1,
        steps_per_block=4,
        confidence_threshold=0.8,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_early_noop["earlyCompleted"] is True
    assert step_early_noop["isBlockComplete"] is True
    assert step_early_noop["commitReason"] == "early_completed"

    logits_early_finish = torch.full((4, vocab_size), -20.0, dtype=torch.float32)
    logits_early_finish[1, 14] = 4.0
    logits_early_finish[3, 21] = 4.5
    step_early_finish = confidence_unmask_step_ref(
        xt=[4, mask_token_id, 11, mask_token_id],
        logits=logits_early_finish,
        mask_token_id=mask_token_id,
        step_index=1,
        steps_per_block=4,
        confidence_threshold=0.8,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_early_finish["earlyCompleted"] is True
    assert step_early_finish["isBlockComplete"] is True
    assert step_early_finish["committedIndices"] == [1, 3]
    assert step_early_finish["xtNext"] == [4, 14, 11, 21]

    # 8D: Post-block EOS check (EOS emitted at slot 1 on step 0 while slot 3 stays masked -> finishes on step 1!)
    logits_eos_s0 = torch.full((4, vocab_size), -10.0, dtype=torch.float32)
    logits_eos_s0[0, 10] = 3.0
    logits_eos_s0[1, eos_token_id] = 3.5  # High-confidence EOS at slot 1
    logits_eos_s0[2, 4] = 2.8
    logits_eos_s0[3, 8] = 0.0
    logits_eos_s0[3, 9] = 0.0  # Low confidence (0.5 <= 0.8) -> stays masked after step 0!
    step_eos_0 = confidence_unmask_step_ref(
        xt=[mask_token_id] * 4,
        logits=logits_eos_s0,
        mask_token_id=mask_token_id,
        step_index=0,
        steps_per_block=4,
        confidence_threshold=0.8,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_eos_0["isBlockComplete"] is False
    assert step_eos_0["xtNext"] == [10, eos_token_id, 4, mask_token_id]
    logits_eos_s1 = torch.full((4, vocab_size), -10.0, dtype=torch.float32)
    logits_eos_s1[3, 8] = 4.0
    step_eos_1 = confidence_unmask_step_ref(
        xt=step_eos_0["xtNext"],
        logits=logits_eos_s1,
        mask_token_id=mask_token_id,
        step_index=1,
        steps_per_block=4,
        confidence_threshold=0.8,
        temperature=0.0,
        eos_token_ids=[eos_token_id],
    )
    assert step_eos_1["isBlockComplete"] is True
    assert step_eos_1["containsEos"] is True
    assert step_eos_1["xtNext"] == [10, eos_token_id, 4, 8]

    # 8E: Deterministic Gumbel-max sampling at temperature = 0.7 with explicit shared uniform u
    logits_gumbel = torch.randn((4, vocab_size), generator=gen, dtype=torch.float32) * 0.9
    logits_gumbel[:, mask_token_id] = 25.0  # Must STILL exclude [MASK] under Gumbel-max!
    xt_gumbel = [mask_token_id, 7, mask_token_id, mask_token_id]
    # Explicit Float32 uniform variates in [0, 1) (including 0.0 to exercise GUMBEL_UNIFORM_MIN clamp!)
    u_tensor = (torch.rand((4, vocab_size), generator=gen, dtype=torch.float32) * 0.98 + 0.01).to(torch.float32)
    u_tensor[0, 0] = 0.0  # Exercises max(u, 1e-10) clamp
    # Boost one non-greedy candidate's uniform variate close to 1.0 so Gumbel differs from greedy!
    greedy_pred, _, _ = gumbel_sample_ref(logits_gumbel, xt_gumbel, mask_token_id, 0.0)
    alt_tok = (greedy_pred[0] + 1) % (vocab_size - 2)
    u_tensor[0, alt_tok] = 0.9995
    u_list = to_f32_list(u_tensor, "cat8.gumbel.u")

    step_gumbel = confidence_unmask_step_ref(
        xt=xt_gumbel,
        logits=logits_gumbel,
        mask_token_id=mask_token_id,
        step_index=0,
        steps_per_block=4,
        confidence_threshold=0.25,
        temperature=0.7,
        eos_token_ids=[eos_token_id],
        uniform_variates=u_list,
    )
    assert step_gumbel["predicted"][1] == 7
    assert step_gumbel["predicted"][0] != greedy_pred[0], "Gumbel sample should diverge from greedy on slot 0"
    assert mask_token_id not in step_gumbel["predicted"]

    def pack_step(st: dict[str, Any], logits_t: torch.Tensor, tag: str) -> dict[str, Any]:
        return {
            "xtBefore": st["xtBefore"],
            "logits": to_f32_list(logits_t, f"{tag}.logits"),
            "stepIndex": st["stepIndex"],
            "stepsPerBlock": st["stepsPerBlock"],
            "tModel": st["tModel"],
            "predicted": st["predicted"],
            "confidence": to_nullable_confidence(st["confidence"], f"{tag}.conf"),
            "logXTheta": to_f32_list(st["logXTheta"], f"{tag}.logXTheta"),
            "unscaledProbs": to_f32_list(st["unscaledProbs"], f"{tag}.xTheta"),
            "numAboveThreshold": st["numAboveThreshold"],
            "minCommitFloor": st["minCommitFloor"],
            "tokensCommittedCount": st["tokensCommittedCount"],
            "commitReason": st["commitReason"],
            "committedIndices": st["committedIndices"],
            "thresholdCommittedIndices": st["thresholdCommittedIndices"],
            "floorCommittedIndices": st["floorCommittedIndices"],
            "remaskedIndices": st["remaskedIndices"],
            "xtNext": st["xtNext"],
            "nMaskedBefore": st["nMaskedBefore"],
            "nMaskedAfter": st["nMaskedAfter"],
            "earlyCompleted": st["earlyCompleted"],
            "isBlockComplete": st["isBlockComplete"],
            "containsEos": st["containsEos"],
        }

    cat8 = {
        "vocabSize": vocab_size,
        "maskTokenId": mask_token_id,
        "eosTokenId": eos_token_id,
        "maskExclusionAndCarryOver": {
            "confidenceThreshold": 0.95,
            "temperature": 0.0,
            **pack_step(step_mask, logits_mask, "cat8.mask"),
        },
        "strictThresholdAndBudgetFloor": {
            "confidenceThreshold": 0.5,
            "temperature": 0.0,
            **pack_step(step_budget_floor, logits_floor, "cat8.budget"),
        },
        "progressFloorExactEquality": {
            "confidenceThreshold": 0.5,
            "temperature": 0.0,
            **pack_step(step_progress_floor, logits_prog, "cat8.prog"),
        },
        "earlyCompletionNoOp": {
            "confidenceThreshold": 0.8,
            "temperature": 0.0,
            **pack_step(step_early_noop, logits_early_noop, "cat8.earlyNoop"),
        },
        "earlyCompletionMidBudget": {
            "confidenceThreshold": 0.8,
            "temperature": 0.0,
            **pack_step(step_early_finish, logits_early_finish, "cat8.earlyMid"),
        },
        "postBlockEosHalt": {
            "confidenceThreshold": 0.8,
            "temperature": 0.0,
            "step0PartialWithEos": pack_step(step_eos_0, logits_eos_s0, "cat8.eos0"),
            "step1CompletedWithEos": pack_step(step_eos_1, logits_eos_s1, "cat8.eos1"),
        },
        "deterministicGumbelMax": {
            "confidenceThreshold": 0.25,
            "temperature": 0.7,
            "uniformVariates": u_list,
            "greedyPredictedForComparison": greedy_pred,
            **pack_step(step_gumbel, logits_gumbel, "cat8.gumbel"),
        },
    }

    return {
        "metadata": {
            "generator": "scripts/reference/generate_twotower_pytorch_fixtures.py",
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
            "pinnedTorchVersion": PINNED_TORCH_VERSION,
            "referenceArchitecture": "Nemotron-Labs-TwoTower Diffusion Trajectory, AR/Mock-AR Alignment & Sampler (Cat 6-8)",
            "tolerances": {
                "cacheStateMaxAbs": 2e-5,
                "residualCheckpointMaxAbs": 2e-5,
                "lmHeadLogitsMaxAbs": 5e-5,
                "lmHeadProbsMaxAbs": 2e-5,
                "samplerLogProbMaxAbs": 1e-5,
                "samplerProbMaxAbs": 1e-5,
            },
        },
        "defaultModel": {
            "modelId": default_manifest["modelId"],
            "twoTowerFingerprint": default_manifest["twoTowerFingerprint"],
            "weightsSha256": default_manifest["weightsSha256"],
            "totalWeightFloats": default_manifest["totalFloats"],
        },
        "identityModel": {
            "modelId": identity_manifest["modelId"],
            "twoTowerFingerprint": identity_manifest["twoTowerFingerprint"],
            "weightsSha256": identity_manifest["weightsSha256"],
            "totalWeightFloats": identity_manifest["totalFloats"],
        },
        "cat6TwoBlockTrajectory": cat6,
        "cat7ArAndMockArAlignment": cat7,
        "cat8SamplerEdgeCases": cat8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate independent PyTorch 2.15.0 reference parity fixtures for Nemotron-Labs-TwoTower."
    )
    parser.add_argument(
        "--default-manifest",
        type=Path,
        default=DEFAULT_PARAMS_DIR / "twotower_default_manifest.json",
        help="Path to exported default TwoTower manifest JSON.",
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        default=DEFAULT_PARAMS_DIR / "twotower_identity_manifest.json",
        help="Path to exported identity-weight (perturbationScale=0) TwoTower manifest JSON.",
    )
    args = parser.parse_args()

    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Expected pinned PyTorch {PINNED_TORCH_VERSION}, got {torch.__version__}. "
            "Run via `uv run scripts/reference/generate_twotower_pytorch_fixtures.py`."
        )

    torch.manual_seed(20261125)
    gen = torch.Generator().manual_seed(20261125)

    default_manifest, default_tensors = load_exported_twotower_weights(args.default_manifest)
    identity_manifest, identity_tensors = load_exported_twotower_weights(args.identity_manifest)

    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    blocks_payload = build_blocks_parity_fixture(default_manifest, default_tensors, gen)
    blocks_out = fixtures_dir / "pytorch-twotower-blocks-parity.json"
    blocks_out.write_text(
        json.dumps(blocks_payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote TwoTower blocks parity fixture ({blocks_out.stat().st_size:,} bytes) -> {blocks_out}"
    )

    handoff_payload = build_handoff_parity_fixture(default_manifest, default_tensors)
    handoff_out = fixtures_dir / "pytorch-twotower-handoff-parity.json"
    handoff_out.write_text(
        json.dumps(handoff_payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote TwoTower handoff parity fixture ({handoff_out.stat().st_size:,} bytes) -> {handoff_out}"
    )

    traj_payload = build_trajectory_parity_fixture(
        default_manifest, default_tensors, identity_manifest, identity_tensors, gen
    )
    traj_out = fixtures_dir / "pytorch-twotower-trajectory-parity.json"
    traj_out.write_text(
        json.dumps(traj_payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote TwoTower trajectory parity fixture ({traj_out.stat().st_size:,} bytes) -> {traj_out}"
    )

    cat6 = traj_payload["cat6TwoBlockTrajectory"]
    print(
        f"Cat6 trajectory summary: threshold={cat6['confidenceThreshold']}, "
        f"totalNfe={cat6['totalNfe']}, blocks={[b['finalBlockTokens'] for b in cat6['blocks']]}, "
        f"reasons={[[r['commitReason'] for r in b['refinements']] for b in cat6['blocks']]}"
    )


if __name__ == "__main__":
    main()
