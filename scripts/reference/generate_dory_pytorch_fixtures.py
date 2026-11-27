#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator
for NVIDIA Nemotron-Labs-Dory1.0-4B Backbone (DORY1 Slice 1C).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12.x + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-dory-backbone.json` with 0 B diff:
  node scripts/reference/export_dory_params.mjs [paramsDir]
  uv run scripts/reference/generate_dory_pytorch_fixtures.py [--params-dir <paramsDir> | --manifest <manifestPath>] [--out <outPath>]

Clean-Room Mathematical Reference (authored directly from DORY0 §5–§7 equations):
  1. Weightless unit-sphere L2Norm N(x) = x / sqrt(sum(x^2) + eps), eps = 1e-6,
     across all 7 architectural axes plus zero / near-zero / literal N(2x) checks.
  2. All 5 Full-nGPT learned scaling families (active divisor sigma_0 = 0.0177):
     residual alpha (|raw * alpha_init / sigma_0|, unclamped > 1), Q-only sqk^2,
     output-gate s_gate^2 (honoring custom g_0, DR3), signed suv * sqrt(d) (NO .abs()),
     and FP32 lm_head.logit_scale / sigma_0.
  3. Channel-wise nGPT NLERP residual blend N(a + |alpha_hat| * (b - a)) (Anti-SLERP DR7)
     and Dual-Regime 100% NeoX rotate_half RoPE (theta_S = 16 vs theta_F = 1024, DR6).
  4. Physical 72-sublayer (16 Prelude + 40 Shared Recurrent Core + 16 Coda)
     Gated 4:1 GQA (SWA W = 8 vs Full) + attn_sub_norm and Normalized SwiGLU + ffn_sub_norm.
  5. Full 72L depth-looped forward for k in {1, 2, 3, 4, 6}, literal u_t = N(s_{t-1} + x)
     (incl. u_1 = N(2x)) and s_t = N(F_theta(u_t)) (DR8), no Layer-0 pre-norm (DR5).
  6. Dual-axis KV cache: per_loop chunked/incremental/eviction & long-chunk L = 24, 32
     transient workspace (DR24); last_loop Theorems 1–3 + constructive > 1e-2 partition
     divergence fixture + W_v = 0 degenerate counterexample (DR17).
  7. Layer A/C upstream differential audit (DR5/DR6/DR8/DR9): vLLM FP32 vs transformers
     BF16, RoPE<->L2Norm order, unit vs non-unit Layer-0 pre-norm, and literal N(2x).

Zero NVIDIA, Hugging Face, vLLM, llama.cpp, or community source code is copied.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PINNED_TORCH_VERSION = "2.15.0"
DEFAULT_PARAMS_DIR = Path("/tmp/nemotron-viz-dory-ref")
DEFAULT_OUT_PATH = Path("tests/fixtures/pytorch-dory-backbone.json")

EXPECTED_TENSOR_COUNT = 507
EXPECTED_FLOAT_COUNT = 93_248
EXPECTED_WEIGHT_BYTES = 372_992
CANONICAL_MAGIC = "DORY-MINI-F32-V1\n"

# Repeating 8-sublayer physical motif (x9 across 0..71):
# [SWA, MLP, SWA, MLP, SWA, MLP, Full, MLP]
MOTIF = (
    "sliding_attention",
    "mlp",
    "sliding_attention",
    "mlp",
    "sliding_attention",
    "mlp",
    "full_attention",
    "mlp",
)


# ============================================================================
# STRICT FINITE GUARDS & SERIALIZERS
# ============================================================================

def assert_all_finite(t: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(t).all().item()):
        raise RuntimeError(f"Non-finite value encountered in {label}")


def to_f32_list(t: torch.Tensor, label: str) -> list[float]:
    t32 = t.detach().to(torch.float32).contiguous()
    assert_all_finite(t32, label)
    return [float(x) for x in t32.reshape(-1).tolist()]


def to_f64_list(t: torch.Tensor, label: str) -> list[float]:
    t64 = t.detach().to(torch.float64).contiguous()
    assert_all_finite(t64, label)
    return [float(x) for x in t64.reshape(-1).tolist()]


def to_nullable_scores(scores: torch.Tensor, label: str) -> list[float | None]:
    flat = scores.detach().to(torch.float32).reshape(-1).tolist()
    out: list[float | None] = []
    for idx, val in enumerate(flat):
        if math.isnan(val) or (math.isinf(val) and val > 0):
            raise RuntimeError(f"Unexpected non-finite score {val} at {label}[{idx}]")
        out.append(None if math.isinf(val) else float(val))
    return out


# ============================================================================
# WEIGHT LOADING & CRYPTOGRAPHIC SHA-256 VERIFICATION
# ============================================================================

def load_exported_dory_weights(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing exported Dory manifest: {manifest_path}. "
            "Run `node scripts/reference/export_dory_params.mjs` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CANONICAL_MAGIC:
        raise RuntimeError(
            f"Unexpected manifest format {manifest.get('format')!r}, expected {CANONICAL_MAGIC!r}"
        )

    weights_path = Path(manifest["weightsFile"])
    if not weights_path.exists():
        weights_path = manifest_path.parent / weights_path.name
    raw_bytes = weights_path.read_bytes()
    if len(raw_bytes) != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_WEIGHT_BYTES} weight bytes, got {len(raw_bytes)}"
        )

    actual_raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_raw_sha256 != manifest["weightsSha256"]:
        raise RuntimeError(
            f"Raw weightsSha256 mismatch: {actual_raw_sha256} != {manifest['weightsSha256']}"
        )

    floats = array.array("f")
    floats.frombytes(raw_bytes)
    if sys.byteorder != "little":
        floats.byteswap()
    if floats.itemsize != 4 or len(floats) != EXPECTED_FLOAT_COUNT:
        raise RuntimeError(
            f"Unexpected float32 payload length: {len(floats)} (expected {EXPECTED_FLOAT_COUNT})"
        )

    flat_tensor = torch.tensor(floats, dtype=torch.float32)
    assert_all_finite(flat_tensor, "dory_exported_weights")

    framed_hasher = hashlib.sha256()
    framed_hasher.update(CANONICAL_MAGIC.encode("utf-8"))

    tensors: dict[str, torch.Tensor] = {}
    total_floats = 0
    for entry in manifest["tensors"]:
        name: str = entry["name"]
        off = int(entry["offsetFloats"])
        length = int(entry["length"])
        shape = [int(d) for d in entry["shape"]]
        byte_slice = raw_bytes[off * 4 : (off + length) * 4]

        actual_tensor_sha = hashlib.sha256(byte_slice).hexdigest()
        if actual_tensor_sha != entry["sha256"]:
            raise RuntimeError(f"Per-tensor SHA-256 mismatch on {name}")

        framed_hasher.update(name.encode("utf-8"))
        framed_hasher.update(b"\x00")
        framed_hasher.update(struct.pack("<I", len(shape)))
        for dim in shape:
            framed_hasher.update(struct.pack("<I", dim))
        framed_hasher.update(byte_slice)

        tensors[name] = flat_tensor[off : off + length].view(*shape).contiguous()
        total_floats += length

    actual_model_weight_id = framed_hasher.hexdigest()
    if actual_model_weight_id != manifest["modelWeightId"]:
        raise RuntimeError(
            f"Canonical framed modelWeightId mismatch: "
            f"{actual_model_weight_id} != {manifest['modelWeightId']}"
        )

    if len(tensors) != EXPECTED_TENSOR_COUNT or total_floats != EXPECTED_FLOAT_COUNT:
        raise RuntimeError(
            f"Unexpected tensor count/floats: {len(tensors)} / {total_floats}"
        )

    return manifest, tensors


# ============================================================================
# CLEAN-ROOM FULL-nGPT MATHEMATICAL OPERATORS (FROM DORY0 §5–§7)
# ============================================================================

def l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weightless unit-sphere L2 normalization along the last dimension (DORY0 §5.1)."""
    x32 = x.to(torch.float32)
    denom = torch.sqrt(torch.sum(x32 * x32, dim=-1, keepdim=True) + eps)
    return x32 / denom


def effective_residual_alpha(
    raw_alpha: torch.Tensor, alpha_init: float, sigma_0: float
) -> torch.Tensor:
    """Family 1: |alpha * (alpha_init / sigma_0)| — linear, unclamped >= 0."""
    return torch.abs(raw_alpha.to(torch.float32) * (alpha_init / sigma_0))


def effective_sqk_squared(
    raw_sqk: torch.Tensor, head_dim: int, sigma_0: float
) -> torch.Tensor:
    """Family 2: (sqk * sqrt(d_h) / sigma_0)^2 — squared, applied to Q only."""
    scaled = raw_sqk.to(torch.float32) * (math.sqrt(head_dim) / sigma_0)
    return scaled * scaled


def effective_s_gate_squared(
    raw_s_gate: torch.Tensor,
    hidden_size: int,
    sigma_0: float,
    g_0: float | None = None,
) -> torch.Tensor:
    """Family 3: (s_gate * sqrt(g_0) / sigma_0)^2 with g_0 = sqrt(d) when null (DR3)."""
    resolved_g0 = math.sqrt(hidden_size) if g_0 is None else float(g_0)
    scaled = raw_s_gate.to(torch.float32) * (math.sqrt(resolved_g0) / sigma_0)
    return scaled * scaled


def effective_suv_scale(raw_suv: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Family 4: suv * sqrt(d) — linear, SIGNED (NO .abs())."""
    return raw_suv.to(torch.float32) * math.sqrt(hidden_size)


def effective_logit_scale(raw_logit_scale: torch.Tensor, sigma_0: float) -> torch.Tensor:
    """Family 5: logit_scale / sigma_0 — linear, FP32."""
    return raw_logit_scale.to(torch.float32) / sigma_0


def nlerp_residual(
    residual: torch.Tensor,
    sublayer_out: torch.Tensor,
    raw_alpha: torch.Tensor,
    alpha_init: float,
    sigma_0: float,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Channel-wise nGPT NLERP residual blend (Anti-SLERP DR7, unclamped lambda >= 0)."""
    a = l2_norm(residual, eps=eps)
    b = l2_norm(sublayer_out, eps=eps)
    delta = b - a
    lam = effective_residual_alpha(raw_alpha, alpha_init, sigma_0)
    blended = a + lam * delta
    out = l2_norm(blended, eps=eps)
    return {
        "aNorm": a,
        "bNorm": b,
        "delta": delta,
        "effectiveAlpha": lam,
        "blendedPreNorm": blended,
        "output": out,
    }


def classical_trig_slerp(
    a_in: torch.Tensor, b_in: torch.Tensor, t_scalar: float, eps: float = 1e-6
) -> torch.Tensor:
    """Classical trigonometric SLERP (ONLY for Anti-SLERP contrast against NLERP)."""
    a = l2_norm(a_in, eps=eps)
    b = l2_norm(b_in, eps=eps)
    dot = torch.clamp(torch.sum(a * b, dim=-1, keepdim=True), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    w0 = torch.where(sin_omega.abs() < 1e-6, 1.0 - t_scalar, torch.sin((1.0 - t_scalar) * omega) / sin_omega)
    w1 = torch.where(sin_omega.abs() < 1e-6, t_scalar, torch.sin(t_scalar * omega) / sin_omega)
    return w0 * a + w1 * b


def compute_rope_tables(
    max_positions: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-regime 100% NeoX RoPE tables: inv_freq[j] = theta^(-2j / d_h)."""
    half = head_dim // 2
    j = torch.arange(half, dtype=torch.float64)
    inv_freq = (float(theta) ** (-2.0 * j / float(head_dim))).to(torch.float32)
    positions = torch.arange(max_positions, dtype=torch.float64)[:, None]
    angles = positions * inv_freq.to(torch.float64)[None, :]
    cos = torch.cos(angles).to(torch.float32)
    sin = torch.sin(angles).to(torch.float32)
    return inv_freq, cos, sin


def apply_neox_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> torch.Tensor:
    """
    100% NeoX rotate_half RoPE on x of shape [L, H, d_h]:
      y[..., :half] = x[..., :half] * cos - x[..., half:] * sin
      y[..., half:] = x[..., half:] * cos + x[..., :half] * sin
    """
    half = x.shape[-1] // 2
    cos = cos_table[position_ids][:, None, :]  # [L, 1, half]
    sin = sin_table[position_ids][:, None, :]  # [L, 1, half]
    x0 = x[..., :half]
    x1 = x[..., half:]
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    return torch.cat([y0, y1], dim=-1)


# ============================================================================
# CLEAN-ROOM DUAL-AXIS KV CACHE & 72-SUBLAYER FORWARD ENGINE
# ============================================================================

@dataclass
class PyTorchKvBank:
    physical_layer: int
    is_sliding: bool
    window_size: int | None
    k_entries: list[torch.Tensor]  # each [kv_dim]
    v_entries: list[torch.Tensor]  # each [kv_dim]
    abs_positions: list[int]
    writer_loops: list[int]

    @property
    def logical_len(self) -> int:
        return len(self.abs_positions)


class PyTorchDoryKvCache:
    """
    Clean-room reference dual-axis KV cache implementing DORY0 §7 equations:
      - `per_loop`: separate recurrent banks per loop t in 1..k.
      - `last_loop`: single shared recurrent bank per layer; loops t < k read
        committed history transiently without mutating the bank (DoryPastReader),
        and loop t == k commits the chunk.
      - SWA visibility: 0 <= q_pos - k_pos < W (past_visible <= W - 1 plus current chunk).
    """

    def __init__(self, cfg: dict[str, Any], k_loops: int, mode: str) -> None:
        self.cfg = cfg
        self.k_loops = k_loops
        self.mode = mode
        self.committed_len = 0
        self.input_banks: dict[int, PyTorchKvBank] = {}
        self.recurrent_banks: dict[tuple[int, int], PyTorchKvBank] = {}
        self.output_banks: dict[int, PyTorchKvBank] = {}

        w = int(cfg["slidingWindow"])
        n_in = int(cfg["nInputLayers"])
        n_rec = int(cfg["nRecurrentLayers"])
        n_total = int(cfg["numHiddenLayers"])

        for p in range(n_total):
            kind = MOTIF[p % 8]
            if kind == "mlp":
                continue
            is_swa = kind == "sliding_attention"
            win = w if is_swa else None
            if p < n_in:
                self.input_banks[p] = PyTorchKvBank(p, is_swa, win, [], [], [], [])
            elif p < n_in + n_rec:
                loops = range(1, k_loops + 1) if mode == "per_loop" else (1,)
                for slot in loops:
                    self.recurrent_banks[(p, slot)] = PyTorchKvBank(
                        p, is_swa, win, [], [], [], []
                    )
            else:
                self.output_banks[p] = PyTorchKvBank(p, is_swa, win, [], [], [], [])

    def get_bank(self, p: int, loop_iter: int) -> PyTorchKvBank:
        n_in = int(self.cfg["nInputLayers"])
        n_rec = int(self.cfg["nRecurrentLayers"])
        if p < n_in:
            return self.input_banks[p]
        if p < n_in + n_rec:
            slot = loop_iter if self.mode == "per_loop" else 1
            return self.recurrent_banks[(p, slot)]
        return self.output_banks[p]


def build_visible_kv(
    bank: PyTorchKvBank,
    start_pos: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    writer_loop: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Construct transient [past_visible || current_chunk] without mutating bank."""
    if start_pos != bank.logical_len:
        raise RuntimeError(
            f"Non-contiguous read at p={bank.physical_layer}: "
            f"start_pos={start_pos}, bank.logical_len={bank.logical_len}"
        )
    past_start = 0
    if bank.is_sliding and bank.window_size is not None:
        past_start = max(0, start_pos - (bank.window_size - 1))

    past_k = bank.k_entries[past_start:start_pos]
    past_v = bank.v_entries[past_start:start_pos]
    past_pos = list(bank.abs_positions[past_start:start_pos])
    past_writers = list(bank.writer_loops[past_start:start_pos])

    seq_len = k_new.shape[0]
    if past_k:
        k_vis = torch.cat([torch.stack(past_k, dim=0), k_new], dim=0)
        v_vis = torch.cat([torch.stack(past_v, dim=0), v_new], dim=0)
    else:
        k_vis = k_new
        v_vis = v_new

    abs_pos = past_pos + [start_pos + i for i in range(seq_len)]
    writers = past_writers + [writer_loop for _ in range(seq_len)]
    return k_vis, v_vis, abs_pos, writers


def commit_kv(
    bank: PyTorchKvBank,
    start_pos: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    writer_loop: int,
) -> None:
    if start_pos != bank.logical_len:
        raise RuntimeError(
            f"Non-contiguous commit at p={bank.physical_layer}: "
            f"start_pos={start_pos}, bank.logical_len={bank.logical_len}"
        )
    for i in range(k_new.shape[0]):
        bank.k_entries.append(k_new[i].detach().clone())
        bank.v_entries.append(v_new[i].detach().clone())
        bank.abs_positions.append(start_pos + i)
        bank.writer_loops.append(writer_loop)


def run_attention_sublayer(
    h: torch.Tensor,
    p: int,
    start_pos: int,
    loop_iter: int,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    bank: PyTorchKvBank,
    write_kv: bool,
    cfg: dict[str, Any],
    *,
    rope_after_norm: bool = False,
    bf16_scale_compute: bool = False,
    capture_trace: bool = False,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """
    Canonical Full-nGPT Gated GQA sublayer (DORY0 §5.6):
      - Q/K/V/G bias-free linear projections
      - Dual RoPE (sliding vs full) -> per-head L2Norm (or reversed for DR6 check)
      - Q-only squared sqk^2 scaling; cached K stays unit-norm
      - Causal 4:1 GQA with exact SWA mask 0 <= q_pos - k_pos < W
      - Input-dependent sigmoid output gate (s_gate^2) -> attn_sub_norm (over ALL q_dim!)
      - o_proj -> channel-wise NLERP residual
    """
    seq_len = h.shape[0]
    d = int(cfg["hiddenSize"])
    h_q = int(cfg["numAttentionHeads"])
    h_kv = int(cfg["numKeyValueHeads"])
    gqa_rep = h_q // h_kv
    d_h = int(cfg["headDim"])
    q_dim = h_q * d_h
    kv_dim = h_kv * d_h
    eps = float(cfg["l2NormEps"])
    sigma_0 = float(cfg["initializerRange"])
    alpha_init = float(cfg["ngptAlphaInit"])
    g_0 = cfg["attentionOutputGateLogitScaleInit"]
    window = int(cfg["slidingWindow"])

    is_sliding = MOTIF[p % 8] == "sliding_attention"
    regime = "sliding" if is_sliding else "full"
    _, cos_tab, sin_tab = rope_tables[regime]

    w_q = weights[f"backbone.layers.{p}.mixer.q_proj.weight"]
    w_k = weights[f"backbone.layers.{p}.mixer.k_proj.weight"]
    w_v = weights[f"backbone.layers.{p}.mixer.v_proj.weight"]
    w_g = weights[f"backbone.layers.{p}.mixer.g_proj.weight"]
    w_o = weights[f"backbone.layers.{p}.mixer.o_proj.weight"]
    raw_sqk = weights[f"backbone.layers.{p}.mixer.sqk"]
    raw_s_gate = weights[f"backbone.layers.{p}.mixer.s_gate"]
    raw_alpha = weights[f"backbone.layers.{p}.attn_alpha"]

    q_pre = h @ w_q.T  # [L, q_dim]
    k_pre = h @ w_k.T  # [L, kv_dim]
    v_new = h @ w_v.T  # [L, kv_dim]
    g_pre = h @ w_g.T  # [L, q_dim]

    pos_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long)

    q_heads = q_pre.view(seq_len, h_q, d_h)
    k_heads = k_pre.view(seq_len, h_kv, d_h)

    if not rope_after_norm:
        # Canonical vLLM / llama.cpp / Paper TikZ order: RoPE -> Q/K L2Norm (DR6)
        q_rot = apply_neox_rope(q_heads, pos_ids, cos_tab, sin_tab)
        k_rot = apply_neox_rope(k_heads, pos_ids, cos_tab, sin_tab)
        q_norm = l2_norm(q_rot, eps=eps)
        k_norm = l2_norm(k_rot, eps=eps)
    else:
        # transformers order: Q/K L2Norm -> RoPE (for DR6 differential audit)
        q_n0 = l2_norm(q_heads, eps=eps)
        k_n0 = l2_norm(k_heads, eps=eps)
        if bf16_scale_compute:
            q_n0 = q_n0.to(torch.bfloat16).to(torch.float32)
            k_n0 = k_n0.to(torch.bfloat16).to(torch.float32)
        q_norm = apply_neox_rope(q_n0, pos_ids, cos_tab, sin_tab)
        k_norm = apply_neox_rope(k_n0, pos_ids, cos_tab, sin_tab)
        q_rot = q_norm
        k_rot = k_norm

    if bf16_scale_compute:
        # transformers native BF16 scale compute (DR9 differential audit)
        sqk_mult_bf16 = torch.tensor(math.sqrt(d_h) / sigma_0, dtype=torch.bfloat16)
        sqk_lin = raw_sqk.to(torch.bfloat16) * sqk_mult_bf16
        eff_sqk_sq = (sqk_lin * sqk_lin).to(torch.bfloat16)
        q_scaled_flat = (q_norm.reshape(seq_len, q_dim).to(torch.bfloat16) * eff_sqk_sq).to(torch.float32)
        eff_sqk_sq_f32 = eff_sqk_sq.to(torch.float32)
    else:
        eff_sqk_sq_f32 = effective_sqk_squared(raw_sqk, d_h, sigma_0)
        q_scaled_flat = q_norm.reshape(seq_len, q_dim) * eff_sqk_sq_f32[None, :]

    q_scaled_heads = q_scaled_flat.view(seq_len, h_q, d_h)
    k_cache_flat = k_norm.reshape(seq_len, kv_dim)

    k_vis_flat, v_vis_flat, kv_abs_pos, kv_writers = build_visible_kv(
        bank, start_pos, k_cache_flat, v_new, loop_iter
    )
    kv_len = k_vis_flat.shape[0]
    past_len = kv_len - seq_len

    k_vis_heads = k_vis_flat.view(kv_len, h_kv, d_h).repeat_interleave(gqa_rep, dim=1)
    v_vis_heads = v_vis_flat.view(kv_len, h_kv, d_h).repeat_interleave(gqa_rep, dim=1)

    # [h_q, L, d_h] @ [h_q, d_h, kv_len] -> [h_q, L, kv_len]
    q_t = q_scaled_heads.permute(1, 0, 2)
    k_t = k_vis_heads.permute(1, 2, 0)
    v_t = v_vis_heads.permute(1, 0, 2)

    raw_scores = torch.matmul(q_t, k_t) / math.sqrt(d_h)

    q_pos_vec = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long)[:, None]
    k_pos_vec = torch.tensor(kv_abs_pos, dtype=torch.long)[None, :]
    diff = q_pos_vec - k_pos_vec
    if is_sliding:
        visible = (diff >= 0) & (diff < window)
    else:
        visible = diff >= 0

    scores = raw_scores.masked_fill(~visible[None, :, :], float("-inf"))
    probs = F.softmax(scores.to(torch.float64), dim=-1).to(torch.float32)
    attn_ctx_heads = torch.matmul(probs, v_t).permute(1, 0, 2).contiguous()  # [L, h_q, d_h]
    attn_context = attn_ctx_heads.view(seq_len, q_dim)

    if write_kv:
        commit_kv(bank, start_pos, k_cache_flat, v_new, loop_iter)

    if bf16_scale_compute:
        resolved_g0 = math.sqrt(d) if g_0 is None else float(g_0)
        sg_mult_bf16 = torch.tensor(math.sqrt(resolved_g0) / sigma_0, dtype=torch.bfloat16)
        sg_lin = raw_s_gate.to(torch.bfloat16) * sg_mult_bf16
        eff_sg_sq = (sg_lin * sg_lin).to(torch.bfloat16)
        gate_vals_bf16 = torch.sigmoid(g_pre.to(torch.bfloat16) * eff_sg_sq)
        gated_ctx_bf16 = attn_context.to(torch.bfloat16) * gate_vals_bf16
        gate_values = gate_vals_bf16.to(torch.float32)
        gated_context = gated_ctx_bf16.to(torch.float32)
        eff_s_gate_sq_f32 = eff_sg_sq.to(torch.float32)
        attn_sub_norm = l2_norm(gated_context, eps=eps)
        o_proj_out = (attn_sub_norm.to(torch.bfloat16) @ w_o.to(torch.bfloat16).T).to(torch.float32)
        # BF16 NLERP
        a_bf = l2_norm(h, eps=eps).to(torch.bfloat16)
        b_bf = l2_norm(o_proj_out, eps=eps).to(torch.bfloat16)
        lam_bf = torch.abs(
            raw_alpha.to(torch.bfloat16) * torch.tensor(alpha_init / sigma_0, dtype=torch.bfloat16)
        )
        blended_bf = a_bf + lam_bf * (b_bf - a_bf)
        out = l2_norm(blended_bf.to(torch.float32), eps=eps)
        nlerp_res = {
            "aNorm": a_bf.to(torch.float32),
            "bNorm": b_bf.to(torch.float32),
            "delta": (b_bf - a_bf).to(torch.float32),
            "effectiveAlpha": lam_bf.to(torch.float32),
            "blendedPreNorm": blended_bf.to(torch.float32),
            "output": out,
        }
    else:
        eff_s_gate_sq_f32 = effective_s_gate_squared(raw_s_gate, d, sigma_0, g_0)
        gate_values = torch.sigmoid(g_pre * eff_s_gate_sq_f32[None, :])
        gated_context = attn_context * gate_values
        # CRITICAL: attn_sub_norm is over ALL concatenated heads q_dim (NOT per-head!)
        attn_sub_norm = l2_norm(gated_context, eps=eps)
        o_proj_out = attn_sub_norm @ w_o.T
        nlerp_res = nlerp_residual(h, o_proj_out, raw_alpha, alpha_init, sigma_0, eps=eps)
        out = nlerp_res["output"]

    trace: dict[str, Any] | None = None
    if capture_trace:
        trace = {
            "physicalLayerIdx": p,
            "attentionKind": MOTIF[p % 8],
            "seqLen": seq_len,
            "startPos": start_pos,
            "pastLen": past_len,
            "kvLen": kv_len,
            "kvAbsPositions": kv_abs_pos,
            "kvWriterLoops": kv_writers,
            "qPre": to_f32_list(q_pre, f"p{p}.qPre"),
            "kPre": to_f32_list(k_pre, f"p{p}.kPre"),
            "vNew": to_f32_list(v_new, f"p{p}.vNew"),
            "gPre": to_f32_list(g_pre, f"p{p}.gPre"),
            "qRot": to_f32_list(q_rot.reshape(seq_len, q_dim), f"p{p}.qRot"),
            "kRot": to_f32_list(k_rot.reshape(seq_len, kv_dim), f"p{p}.kRot"),
            "qNorm": to_f32_list(q_norm.reshape(seq_len, q_dim), f"p{p}.qNorm"),
            "kCache": to_f32_list(k_cache_flat, f"p{p}.kCache"),
            "effectiveSqkSq": to_f32_list(eff_sqk_sq_f32, f"p{p}.effectiveSqkSq"),
            "qScaled": to_f32_list(q_scaled_flat, f"p{p}.qScaled"),
            "maskVisible": [int(x) for x in visible.to(torch.int64).reshape(-1).tolist()],
            "scores": to_nullable_scores(scores, f"p{p}.scores"),
            "probs": to_f32_list(probs, f"p{p}.probs"),
            "attnContext": to_f32_list(attn_context, f"p{p}.attnContext"),
            "effectiveSGateSq": to_f32_list(eff_s_gate_sq_f32, f"p{p}.effectiveSGateSq"),
            "gateValues": to_f32_list(gate_values, f"p{p}.gateValues"),
            "gatedContext": to_f32_list(gated_context, f"p{p}.gatedContext"),
            "attnSubNorm": to_f32_list(attn_sub_norm, f"p{p}.attnSubNorm"),
            "oProjOut": to_f32_list(o_proj_out, f"p{p}.oProjOut"),
            "nlerp": {
                "aNorm": to_f32_list(nlerp_res["aNorm"], f"p{p}.nlerp.aNorm"),
                "bNorm": to_f32_list(nlerp_res["bNorm"], f"p{p}.nlerp.bNorm"),
                "delta": to_f32_list(nlerp_res["delta"], f"p{p}.nlerp.delta"),
                "effectiveAlpha": to_f32_list(nlerp_res["effectiveAlpha"], f"p{p}.nlerp.effAlpha"),
                "blendedPreNorm": to_f32_list(nlerp_res["blendedPreNorm"], f"p{p}.nlerp.blended"),
                "output": to_f32_list(nlerp_res["output"], f"p{p}.nlerp.output"),
            },
            "output": to_f32_list(out, f"p{p}.output"),
        }

    return out, trace


def run_mlp_sublayer(
    h: torch.Tensor,
    p: int,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    bf16_scale_compute: bool = False,
    capture_trace: bool = False,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """
    Canonical Full-nGPT Normalized SwiGLU MLP sublayer (DORY0 §5.6):
      - gate_proj & up_proj (bias-free)
      - signed suv_gate * sqrt(d), suv_up * sqrt(d) (NO .abs())
      - ffn_sub_norm = N_{d_ff}(SiLU(U_g) * U_u)
      - down_proj -> channel-wise NLERP residual
    """
    d = int(cfg["hiddenSize"])
    eps = float(cfg["l2NormEps"])
    sigma_0 = float(cfg["initializerRange"])
    alpha_init = float(cfg["ngptAlphaInit"])

    w_gate = weights[f"backbone.layers.{p}.mixer.gate_proj.weight"]
    w_up = weights[f"backbone.layers.{p}.mixer.up_proj.weight"]
    w_down = weights[f"backbone.layers.{p}.mixer.down_proj.weight"]
    raw_suv_g = weights[f"backbone.layers.{p}.mixer.suv_gate"]
    raw_suv_u = weights[f"backbone.layers.{p}.mixer.suv_up"]
    raw_alpha = weights[f"backbone.layers.{p}.mlp_alpha"]

    gate_pre = h @ w_gate.T
    up_pre = h @ w_up.T

    if bf16_scale_compute:
        sqrt_d_bf = torch.tensor(math.sqrt(d), dtype=torch.bfloat16)
        eff_g_bf = raw_suv_g.to(torch.bfloat16) * sqrt_d_bf
        eff_u_bf = raw_suv_u.to(torch.bfloat16) * sqrt_d_bf
        gate_scaled_bf = gate_pre.to(torch.bfloat16) * eff_g_bf
        up_scaled_bf = up_pre.to(torch.bfloat16) * eff_u_bf
        gate_silu_bf = F.silu(gate_scaled_bf)
        swiglu_prod_bf = gate_silu_bf * up_scaled_bf
        eff_suv_g = eff_g_bf.to(torch.float32)
        eff_suv_u = eff_u_bf.to(torch.float32)
        gate_scaled = gate_scaled_bf.to(torch.float32)
        up_scaled = up_scaled_bf.to(torch.float32)
        gate_silu = gate_silu_bf.to(torch.float32)
        swiglu_prod = swiglu_prod_bf.to(torch.float32)
        ffn_sub_norm = l2_norm(swiglu_prod, eps=eps)
        down_proj_out = (ffn_sub_norm.to(torch.bfloat16) @ w_down.to(torch.bfloat16).T).to(torch.float32)
        a_bf = l2_norm(h, eps=eps).to(torch.bfloat16)
        b_bf = l2_norm(down_proj_out, eps=eps).to(torch.bfloat16)
        lam_bf = torch.abs(
            raw_alpha.to(torch.bfloat16) * torch.tensor(alpha_init / sigma_0, dtype=torch.bfloat16)
        )
        blended_bf = a_bf + lam_bf * (b_bf - a_bf)
        out = l2_norm(blended_bf.to(torch.float32), eps=eps)
        nlerp_res = {
            "aNorm": a_bf.to(torch.float32),
            "bNorm": b_bf.to(torch.float32),
            "delta": (b_bf - a_bf).to(torch.float32),
            "effectiveAlpha": lam_bf.to(torch.float32),
            "blendedPreNorm": blended_bf.to(torch.float32),
            "output": out,
        }
    else:
        eff_suv_g = effective_suv_scale(raw_suv_g, d)
        eff_suv_u = effective_suv_scale(raw_suv_u, d)
        gate_scaled = gate_pre * eff_suv_g[None, :]
        up_scaled = up_pre * eff_suv_u[None, :]
        gate_silu = F.silu(gate_scaled)
        swiglu_prod = gate_silu * up_scaled
        ffn_sub_norm = l2_norm(swiglu_prod, eps=eps)
        down_proj_out = ffn_sub_norm @ w_down.T
        nlerp_res = nlerp_residual(h, down_proj_out, raw_alpha, alpha_init, sigma_0, eps=eps)
        out = nlerp_res["output"]

    trace: dict[str, Any] | None = None
    if capture_trace:
        trace = {
            "physicalLayerIdx": p,
            "seqLen": h.shape[0],
            "gatePre": to_f32_list(gate_pre, f"p{p}.gatePre"),
            "upPre": to_f32_list(up_pre, f"p{p}.upPre"),
            "effectiveSuvGate": to_f32_list(eff_suv_g, f"p{p}.effSuvGate"),
            "effectiveSuvUp": to_f32_list(eff_suv_u, f"p{p}.effSuvUp"),
            "gateScaled": to_f32_list(gate_scaled, f"p{p}.gateScaled"),
            "upScaled": to_f32_list(up_scaled, f"p{p}.upScaled"),
            "gateSilu": to_f32_list(gate_silu, f"p{p}.gateSilu"),
            "swigluProduct": to_f32_list(swiglu_prod, f"p{p}.swigluProduct"),
            "ffnSubNorm": to_f32_list(ffn_sub_norm, f"p{p}.ffnSubNorm"),
            "downProjOut": to_f32_list(down_proj_out, f"p{p}.downProjOut"),
            "nlerp": {
                "aNorm": to_f32_list(nlerp_res["aNorm"], f"p{p}.nlerp.aNorm"),
                "bNorm": to_f32_list(nlerp_res["bNorm"], f"p{p}.nlerp.bNorm"),
                "delta": to_f32_list(nlerp_res["delta"], f"p{p}.nlerp.delta"),
                "effectiveAlpha": to_f32_list(nlerp_res["effectiveAlpha"], f"p{p}.nlerp.effAlpha"),
                "blendedPreNorm": to_f32_list(nlerp_res["blendedPreNorm"], f"p{p}.nlerp.blended"),
                "output": to_f32_list(nlerp_res["output"], f"p{p}.nlerp.output"),
            },
            "output": to_f32_list(out, f"p{p}.output"),
        }

    return out, trace


def run_forward_chunk(
    token_ids: list[int],
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    kv_cache: PyTorchDoryKvCache,
    *,
    layer0_prenorm: bool = False,
    rope_after_norm: bool = False,
    elide_n2x_and_post_norm: bool = False,
    bf16_scale_compute: bool = False,
    embeddings_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Execute one full physical-72 forward chunk (DORY0 §6.5)."""
    start_pos = kv_cache.committed_len
    k_loops = kv_cache.k_loops
    mode = kv_cache.mode
    eps = float(cfg["l2NormEps"])
    sigma_0 = float(cfg["initializerRange"])
    n_in = int(cfg["nInputLayers"])
    n_rec = int(cfg["nRecurrentLayers"])
    n_total = int(cfg["numHiddenLayers"])

    idx = torch.tensor(token_ids, dtype=torch.long)
    emb_table = embeddings_override if embeddings_override is not None else weights["backbone.embeddings.weight"]
    embeddings = emb_table[idx].clone()
    h = l2_norm(embeddings, eps=eps) if layer0_prenorm else embeddings

    # Stage 1: Input Prelude (0..15)
    for p in range(0, n_in):
        if MOTIF[p % 8] == "mlp":
            h, _ = run_mlp_sublayer(h, p, weights, cfg, bf16_scale_compute=bf16_scale_compute)
        else:
            bank = kv_cache.get_bank(p, 0)
            h, _ = run_attention_sublayer(
                h,
                p,
                start_pos,
                0,
                weights,
                rope_tables,
                bank,
                True,
                cfg,
                rope_after_norm=rope_after_norm,
                bf16_scale_compute=bf16_scale_compute,
            )

    x_anchor = h
    state = x_anchor
    loop_entries: list[torch.Tensor] = []
    loop_exits: list[torch.Tensor] = []

    # Stage 2: Shared Recurrent Core (16..55, repeated t = 1..k_loops)
    for t in range(1, k_loops + 1):
        if elide_n2x_and_post_norm and t == 1:
            u_t = x_anchor
        else:
            # Literal u_t = N(s_{t-1} + x), including literal u_1 = N(2x) (DR8)
            u_t = l2_norm(state + x_anchor, eps=eps)
        loop_entries.append(u_t)

        h = u_t
        write_kv = (mode == "per_loop") or (t == k_loops)
        for p in range(n_in, n_in + n_rec):
            if MOTIF[p % 8] == "mlp":
                h, _ = run_mlp_sublayer(h, p, weights, cfg, bf16_scale_compute=bf16_scale_compute)
            else:
                bank = kv_cache.get_bank(p, t)
                h, _ = run_attention_sublayer(
                    h,
                    p,
                    start_pos,
                    t,
                    weights,
                    rope_tables,
                    bank,
                    write_kv,
                    cfg,
                    rope_after_norm=rope_after_norm,
                    bf16_scale_compute=bf16_scale_compute,
                )

        if elide_n2x_and_post_norm:
            state = h
        else:
            # Literal post-loop normalization s_t = N(F_theta(u_t)) (DR8)
            state = l2_norm(h, eps=eps)
        loop_exits.append(state)

    # Stage 3: Output Coda (56..71, no outer skip)
    h = state
    for p in range(n_in + n_rec, n_total):
        if MOTIF[p % 8] == "mlp":
            h, _ = run_mlp_sublayer(h, p, weights, cfg, bf16_scale_compute=bf16_scale_compute)
        else:
            bank = kv_cache.get_bank(p, 0)
            h, _ = run_attention_sublayer(
                h,
                p,
                start_pos,
                0,
                weights,
                rope_tables,
                bank,
                True,
                cfg,
                rope_after_norm=rope_after_norm,
                bf16_scale_compute=bf16_scale_compute,
            )

    coda_hidden = h
    kv_cache.committed_len += len(token_ids)

    # Stage 4: Untied LM Head & FP32 Scaled Logits
    raw_logits = coda_hidden @ weights["lm_head.weight"].T
    eff_logit_scale = effective_logit_scale(weights["lm_head.logit_scale"], sigma_0)
    scaled_logits = raw_logits * eff_logit_scale[None, :]
    last_token_probs = F.softmax(scaled_logits[-1].to(torch.float64), dim=-1)

    return {
        "embeddings": embeddings,
        "xAnchor": x_anchor,
        "loopEntries": loop_entries,
        "loopExits": loop_exits,
        "codaHidden": coda_hidden,
        "rawLogits": raw_logits,
        "scaledLogits": scaled_logits,
        "lastTokenProbs": last_token_probs,
    }


def run_partitioned_sequence(
    token_ids: list[int],
    chunk_sizes: list[int],
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    k_loops: int,
    mode: str,
) -> dict[str, torch.Tensor]:
    if sum(chunk_sizes) != len(token_ids):
        raise ValueError(f"chunk_sizes {chunk_sizes} do not sum to {len(token_ids)}")
    kv_cache = PyTorchDoryKvCache(cfg, k_loops, mode)
    coda_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    scaled_chunks: list[torch.Tensor] = []
    final_s_chunks: list[torch.Tensor] = []
    offset = 0
    for sz in chunk_sizes:
        res = run_forward_chunk(
            token_ids[offset : offset + sz], weights, rope_tables, cfg, kv_cache
        )
        coda_chunks.append(res["codaHidden"])
        raw_chunks.append(res["rawLogits"])
        scaled_chunks.append(res["scaledLogits"])
        final_s_chunks.append(res["loopExits"][-1])
        offset += sz
    return {
        "codaHidden": torch.cat(coda_chunks, dim=0),
        "rawLogits": torch.cat(raw_chunks, dim=0),
        "scaledLogits": torch.cat(scaled_chunks, dim=0),
        "finalLoopExit": torch.cat(final_s_chunks, dim=0),
    }


# ============================================================================
# FIXTURE BUILDERS (CATEGORIES 1–6)
# ============================================================================

def build_single_op_fixtures(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    d = int(cfg["hiddenSize"])
    d_h = int(cfg["headDim"])
    h_q = int(cfg["numAttentionHeads"])
    h_kv = int(cfg["numKeyValueHeads"])
    sigma_0 = float(cfg["initializerRange"])
    alpha_init = float(cfg["ngptAlphaInit"])
    eps = float(cfg["l2NormEps"])

    # 1A. L2Norm across zero, near-zero, 8D hidden, 64D attn_sub_norm, 24D ffn_sub_norm, literal N(2x)
    zero_vec = torch.zeros(d, dtype=torch.float32)
    near_zero_vec = torch.tensor(
        [1e-8, -2e-8, 3e-8, -4e-8, 5e-8, -6e-8, 7e-8, -8e-8], dtype=torch.float32
    )
    std_hidden = torch.tensor(
        [1.25, -0.75, 0.5, 2.0, -1.5, 0.25, -0.125, 0.875], dtype=torch.float32
    )
    attn_sub_vec = torch.tensor(
        [math.sin(0.37 * (i + 1)) * 1.4 - 0.25 * math.cos(0.19 * i) for i in range(h_q * d_h)],
        dtype=torch.float32,
    )
    ffn_sub_vec = torch.tensor(
        [math.cos(0.41 * (i + 1)) * 2.1 + 0.3 * math.sin(0.23 * i) for i in range(int(cfg["intermediateSize"]))],
        dtype=torch.float32,
    )
    unit_x = weights["backbone.embeddings.weight"][1].clone()
    unit_s = weights["backbone.embeddings.weight"][12].clone()
    n_x = l2_norm(unit_x, eps=eps)
    n_2x = l2_norm(unit_x + unit_x, eps=eps)
    n_s_plus_x = l2_norm(unit_s + unit_x, eps=eps)
    n_antipodal = l2_norm(unit_x + (-unit_x), eps=eps)
    n2x_vs_nx_diff = float((n_2x - n_x).abs().max().item())
    if not (n2x_vs_nx_diff > 1e-8):
        raise RuntimeError(f"Expected non-zero N(2x) vs N(x) diff, got {n2x_vs_nx_diff}")

    # 1B. All 5 Scale Family Transforms (incl. planted > 1 alpha and custom g_0 = 4.0)
    p0_alpha = weights["backbone.layers.0.attn_alpha"]
    p16_alpha = weights["backbone.layers.16.attn_alpha"]
    p1_mlp_alpha = weights["backbone.layers.1.mlp_alpha"]
    p0_sqk = weights["backbone.layers.0.mixer.sqk"]
    p16_sqk = weights["backbone.layers.16.mixer.sqk"]
    p0_s_gate = weights["backbone.layers.0.mixer.s_gate"]
    p1_suv_g = weights["backbone.layers.1.mixer.suv_gate"]
    p1_suv_u = weights["backbone.layers.1.mixer.suv_up"]
    lm_logit_scale = weights["lm_head.logit_scale"]

    eff_p16_alpha = effective_residual_alpha(p16_alpha, alpha_init, sigma_0)
    if not (1.05 < float(eff_p16_alpha[6]) < 1.25 and 1.05 < float(eff_p16_alpha[7]) < 1.25):
        raise RuntimeError(
            f"Planted p=16 extrapolation channels out of (1.05, 1.25): "
            f"{eff_p16_alpha[6].item()}, {eff_p16_alpha[7].item()}"
        )
    if not bool((effective_suv_scale(p1_suv_g, d) < 0).any().item()):
        raise RuntimeError("Expected negative signed entries in effective suv_gate (no .abs())")

    # 1C. NLERP residual blend (p=0 standard + p=16 planted >1 extrapolation) & Anti-SLERP contrast
    res_rows = torch.stack(
        [weights["backbone.embeddings.weight"][tok] for tok in (1, 10, 12, 40)], dim=0
    )
    sub_rows = torch.stack(
        [
            weights["backbone.embeddings.weight"][tok] * (1.5 + 0.25 * idx)
            for idx, tok in enumerate((25, 29, 34, 43))
        ],
        dim=0,
    )
    nlerp_p0 = nlerp_residual(res_rows, sub_rows, p0_alpha, alpha_init, sigma_0, eps=eps)
    nlerp_p16 = nlerp_residual(res_rows, sub_rows, p16_alpha, alpha_init, sigma_0, eps=eps)
    # Contrast against classical trigonometric SLERP using mean(eff_p16_alpha)
    trig_slerp_p16 = classical_trig_slerp(
        res_rows, sub_rows, float(eff_p16_alpha.mean().item()), eps=eps
    )
    nlerp_vs_trig_slerp_diff = float((nlerp_p16["output"] - trig_slerp_p16).abs().max().item())
    if not (nlerp_vs_trig_slerp_diff > 1e-2):
        raise RuntimeError(
            f"Expected clear NLERP vs trig SLERP contrast > 1e-2, got {nlerp_vs_trig_slerp_diff}"
        )

    # 1D. Dual-RoPE tables and NeoX rotate_half sample rotations
    inv_s, cos_s, sin_s = rope_tables["sliding"]
    inv_f, cos_f, sin_f = rope_tables["full"]

    sample_q = torch.tensor(
        [math.sin(0.29 * (i + 1)) for i in range(5 * h_q * d_h)], dtype=torch.float32
    ).view(5, h_q, d_h)
    sample_k = torch.tensor(
        [math.cos(0.31 * (i + 1)) for i in range(5 * h_kv * d_h)], dtype=torch.float32
    ).view(5, h_kv, d_h)
    contig_pos = torch.arange(7, 12, dtype=torch.long)
    noncontig_pos = torch.tensor([0, 3, 8, 15, 31], dtype=torch.long)

    q_rot_s_contig = apply_neox_rope(sample_q, contig_pos, cos_s, sin_s)
    k_rot_s_contig = apply_neox_rope(sample_k, contig_pos, cos_s, sin_s)
    q_rot_f_contig = apply_neox_rope(sample_q, contig_pos, cos_f, sin_f)
    k_rot_f_contig = apply_neox_rope(sample_k, contig_pos, cos_f, sin_f)
    q_rot_f_noncontig = apply_neox_rope(sample_q, noncontig_pos, cos_f, sin_f)

    # 1E. Anchor-map Jacobian verification (Finding 9: 0, 60, 120, 179, 180 deg)
    jacobian_checks = []
    for deg in (0.0, 60.0, 120.0, 179.0, 180.0):
        rad = math.radians(deg)
        cos_half = max(0.0, math.cos(rad * 0.5))
        eps_jac = 1.0 / math.sqrt(4.0 * cos_half * cos_half + eps)
        # Finite-difference check on orthogonal perturbation along e_2 in R^8
        x_u = torch.zeros(d, dtype=torch.float64)
        x_u[0] = 1.0
        s_u = torch.zeros(d, dtype=torch.float64)
        s_u[0] = math.cos(rad)
        s_u[1] = math.sin(rad)
        delta_dir = torch.zeros(d, dtype=torch.float64)
        delta_dir[2] = 1.0
        h_fd = 1e-7
        f_plus = (s_u + h_fd * delta_dir + x_u) / torch.sqrt(
            torch.sum((s_u + h_fd * delta_dir + x_u) ** 2) + eps
        )
        f_minus = (s_u - h_fd * delta_dir + x_u) / torch.sqrt(
            torch.sum((s_u - h_fd * delta_dir + x_u) ** 2) + eps
        )
        fd_norm = float(torch.linalg.norm((f_plus - f_minus) / (2.0 * h_fd)).item())
        if abs(fd_norm - eps_jac) > 1e-3:
            raise RuntimeError(
                f"Anchor Jacobian finite-diff mismatch at {deg} deg: fd={fd_norm}, analytic={eps_jac}"
            )
        jacobian_checks.append(
            {
                "angleDegrees": deg,
                "angleRadians": rad,
                "epsStabilizedJacobianNorm": eps_jac,
                "finiteDiffOrthogonalNorm": fd_norm,
                "isLocalContractiveRegime": deg < 120.0,
            }
        )

    return {
        "l2Norm": {
            "zeroIn": to_f32_list(zero_vec, "l2.zeroIn"),
            "zeroOut": to_f32_list(l2_norm(zero_vec, eps=eps), "l2.zeroOut"),
            "nearZeroIn": to_f32_list(near_zero_vec, "l2.nearZeroIn"),
            "nearZeroOut": to_f32_list(l2_norm(near_zero_vec, eps=eps), "l2.nearZeroOut"),
            "stdHiddenIn": to_f32_list(std_hidden, "l2.stdHiddenIn"),
            "stdHiddenOut": to_f32_list(l2_norm(std_hidden, eps=eps), "l2.stdHiddenOut"),
            "attnSubNormIn": to_f32_list(attn_sub_vec, "l2.attnSubIn"),
            "attnSubNormOut": to_f32_list(l2_norm(attn_sub_vec, eps=eps), "l2.attnSubOut"),
            "ffnSubNormIn": to_f32_list(ffn_sub_vec, "l2.ffnSubIn"),
            "ffnSubNormOut": to_f32_list(l2_norm(ffn_sub_vec, eps=eps), "l2.ffnSubOut"),
            "unitX": to_f32_list(unit_x, "l2.unitX"),
            "unitS": to_f32_list(unit_s, "l2.unitS"),
            "nX": to_f32_list(n_x, "l2.nX"),
            "n2X": to_f32_list(n_2x, "l2.n2X"),
            "nSPlusX": to_f32_list(n_s_plus_x, "l2.nSPlusX"),
            "nAntipodal": to_f32_list(n_antipodal, "l2.nAntipodal"),
            "n2xVsNxMaxAbsDiff": n2x_vs_nx_diff,
        },
        "scaleTransforms": {
            "p0AttnAlphaEff": to_f32_list(effective_residual_alpha(p0_alpha, alpha_init, sigma_0), "scale.p0Alpha"),
            "p16AttnAlphaEff": to_f32_list(eff_p16_alpha, "scale.p16Alpha"),
            "p1MlpAlphaEff": to_f32_list(effective_residual_alpha(p1_mlp_alpha, alpha_init, sigma_0), "scale.p1Alpha"),
            "p0SqkSqEff": to_f32_list(effective_sqk_squared(p0_sqk, d_h, sigma_0), "scale.p0Sqk"),
            "p16SqkSqEff": to_f32_list(effective_sqk_squared(p16_sqk, d_h, sigma_0), "scale.p16Sqk"),
            "p0SGateSqDefaultEff": to_f32_list(effective_s_gate_squared(p0_s_gate, d, sigma_0, None), "scale.p0SGateDef"),
            "p0SGateSqCustomG0Value": 4.0,
            "p0SGateSqCustomG0Eff": to_f32_list(effective_s_gate_squared(p0_s_gate, d, sigma_0, 4.0), "scale.p0SGateCust"),
            "p1SuvGateEff": to_f32_list(effective_suv_scale(p1_suv_g, d), "scale.p1SuvG"),
            "p1SuvUpEff": to_f32_list(effective_suv_scale(p1_suv_u, d), "scale.p1SuvU"),
            "lmLogitScaleEff": to_f32_list(effective_logit_scale(lm_logit_scale, sigma_0), "scale.lmLogit"),
        },
        "nlerp": {
            "seqLen": 4,
            "residualRows": to_f32_list(res_rows, "nlerp.res"),
            "sublayerRows": to_f32_list(sub_rows, "nlerp.sub"),
            "p0": {k: to_f32_list(v, f"nlerp.p0.{k}") for k, v in nlerp_p0.items()},
            "p16PlantedExtrapolation": {k: to_f32_list(v, f"nlerp.p16.{k}") for k, v in nlerp_p16.items()},
            "p16NlerpVsTrigSlerpMaxAbsDiff": nlerp_vs_trig_slerp_diff,
        },
        "dualRope": {
            "slidingInvFreq": to_f32_list(inv_s, "rope.invS"),
            "slidingCos": to_f32_list(cos_s, "rope.cosS"),
            "slidingSin": to_f32_list(sin_s, "rope.sinS"),
            "fullInvFreq": to_f32_list(inv_f, "rope.invF"),
            "fullCos": to_f32_list(cos_f, "rope.cosF"),
            "fullSin": to_f32_list(sin_f, "rope.sinF"),
            "sampleQIn": to_f32_list(sample_q, "rope.sampleQ"),
            "sampleKIn": to_f32_list(sample_k, "rope.sampleK"),
            "contigStartPos": 7,
            "nonContigPosIds": [0, 3, 8, 15, 31],
            "qRotSlidingContig": to_f32_list(q_rot_s_contig, "rope.qRotSC"),
            "kRotSlidingContig": to_f32_list(k_rot_s_contig, "rope.kRotSC"),
            "qRotFullContig": to_f32_list(q_rot_f_contig, "rope.qRotFC"),
            "kRotFullContig": to_f32_list(k_rot_f_contig, "rope.kRotFC"),
            "qRotFullNonContig": to_f32_list(q_rot_f_noncontig, "rope.qRotFNC"),
        },
        "anchorJacobian": jacobian_checks,
    }


def build_sublayer_fixtures(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    emb = weights["backbone.embeddings.weight"]
    tokens_12 = [1, 10, 4, 18, 40, 25, 26, 12, 28, 29, 30, 31]
    h_12 = emb[torch.tensor(tokens_12, dtype=torch.long)].clone()
    h_6 = h_12[:6].clone()

    # p=0 SWA (W=8, S=12 crossing window)
    cache_p0 = PyTorchDoryKvCache(cfg, 1, "per_loop")
    _, trace_p0 = run_attention_sublayer(
        h_12, 0, 0, 0, weights, rope_tables, cache_p0.get_bank(0, 0), True, cfg, capture_trace=True
    )

    # p=6 Full Attention (theta_F=1024, S=12)
    cache_p6 = PyTorchDoryKvCache(cfg, 1, "per_loop")
    _, trace_p6 = run_attention_sublayer(
        h_12, 6, 0, 0, weights, rope_tables, cache_p6.get_bank(6, 0), True, cfg, capture_trace=True
    )

    # p=16 Recurrent SWA (planted >1 alpha channels 6,7): populate past=9 (evicting pos 0,1) then chunk=3
    cache_p16 = PyTorchDoryKvCache(cfg, 2, "per_loop")
    bank_p16 = cache_p16.get_bank(16, 1)
    _, _ = run_attention_sublayer(
        h_12[:9], 16, 0, 1, weights, rope_tables, bank_p16, True, cfg, capture_trace=False
    )
    _, trace_p16_chunk = run_attention_sublayer(
        h_12[9:12], 16, 9, 1, weights, rope_tables, bank_p16, True, cfg, capture_trace=True
    )

    # p=1 Prelude SwiGLU MLP & p=17 Recurrent SwiGLU MLP (S=6)
    _, trace_p1 = run_mlp_sublayer(h_6, 1, weights, cfg, capture_trace=True)
    _, trace_p17 = run_mlp_sublayer(h_6, 17, weights, cfg, capture_trace=True)

    return {
        "tokens12": tokens_12,
        "hidden12": to_f32_list(h_12, "sublayer.h12"),
        "hidden6": to_f32_list(h_6, "sublayer.h6"),
        "attnP0Swa": trace_p0,
        "attnP6Full": trace_p6,
        "attnP16RecurrentChunkAfterEviction": trace_p16_chunk,
        "mlpP1Prelude": trace_p1,
        "mlpP17Recurrent": trace_p17,
    }


def build_full_72l_fixtures(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    tokens = [1, 10, 4, 18, 40, 25, 26, 12, 28, 29]  # S=10 > W=8
    by_k: dict[str, Any] = {}
    for k in (1, 2, 3, 4, 6):
        kv = PyTorchDoryKvCache(cfg, k, "per_loop")
        res = run_forward_chunk(tokens, weights, rope_tables, cfg, kv)
        if k >= 4:
            d12 = float((res["loopExits"][1] - res["loopExits"][0]).abs().max().item())
            d24 = float((res["loopExits"][3] - res["loopExits"][1]).abs().max().item())
            if not (d12 > 1e-2 and d24 > 1e-2):
                raise RuntimeError(f"Recurrent states unexpectedly collapsed at k={k}: d12={d12}, d24={d24}")
        by_k[str(k)] = {
            "kLoops": k,
            "executedSublayerCount": 32 + 40 * k,
            "xAnchor": to_f32_list(res["xAnchor"], f"full72.k{k}.xAnchor"),
            "loopEntryStates": [
                to_f32_list(u, f"full72.k{k}.u{idx + 1}")
                for idx, u in enumerate(res["loopEntries"])
            ],
            "loopExitStates": [
                to_f32_list(s, f"full72.k{k}.s{idx + 1}")
                for idx, s in enumerate(res["loopExits"])
            ],
            "codaHidden": to_f32_list(res["codaHidden"], f"full72.k{k}.codaHidden"),
            "rawLogits": to_f32_list(res["rawLogits"], f"full72.k{k}.rawLogits"),
            "scaledLogits": to_f32_list(res["scaledLogits"], f"full72.k{k}.scaledLogits"),
            "lastTokenProbs": to_f64_list(res["lastTokenProbs"], f"full72.k{k}.lastTokenProbs"),
        }
    return {
        "tokenIds": tokens,
        "byK": by_k,
    }


def build_per_loop_and_long_chunk_fixtures(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # ["<think>", "loop", "sphere", "anchor", "cache", "window", "full", "gate", "scale", "draft", "verify", "accept"]
    tokens_12 = [12, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

    # S=12 per_loop invariance across full [12], chunked [4, 5, 3], 1-by-1 [1]*12 (k=2 and k=4)
    s12_results: dict[str, Any] = {}
    for k in (2, 4):
        full_res = run_partitioned_sequence(tokens_12, [12], weights, rope_tables, cfg, k, "per_loop")
        chunk_res = run_partitioned_sequence(tokens_12, [4, 5, 3], weights, rope_tables, cfg, k, "per_loop")
        inc_res = run_partitioned_sequence(tokens_12, [1] * 12, weights, rope_tables, cfg, k, "per_loop")
        diff_chunk = float((full_res["scaledLogits"] - chunk_res["scaledLogits"]).abs().max().item())
        diff_inc = float((full_res["scaledLogits"] - inc_res["scaledLogits"]).abs().max().item())
        if diff_chunk > 1e-5 or diff_inc > 1e-5:
            raise RuntimeError(
                f"per_loop partition drift at k={k}: chunk={diff_chunk}, inc={diff_inc}"
            )
        s12_results[str(k)] = {
            "kLoops": k,
            "codaHidden": to_f32_list(full_res["codaHidden"], f"pl12.k{k}.coda"),
            "scaledLogits": to_f32_list(full_res["scaledLogits"], f"pl12.k{k}.logits"),
            "pytorchChunkVsFullMaxAbsDiff": diff_chunk,
            "pytorchOneByOneVsFullMaxAbsDiff": diff_inc,
        }

    # Long-chunk workspace DR24: L=24 (> cap_swa=16) + 4-token continuation = 28 tokens, and L=32 (S_max)
    tokens_32 = [
        1, 10, 3, 18, 4, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50,
    ]
    tokens_28 = tokens_32[:28]

    l24_plus_4 = run_partitioned_sequence(tokens_28, [24, 4], weights, rope_tables, cfg, 2, "per_loop")
    l24_chunked_ref = run_partitioned_sequence(tokens_28, [8, 8, 8, 4], weights, rope_tables, cfg, 2, "per_loop")
    diff_l24 = float((l24_plus_4["scaledLogits"] - l24_chunked_ref["scaledLogits"]).abs().max().item())
    if diff_l24 > 1e-5:
        raise RuntimeError(f"Long-chunk L=24+4 drift vs [8,8,8,4]: {diff_l24}")

    l32_single = run_partitioned_sequence(tokens_32, [32], weights, rope_tables, cfg, 2, "per_loop")
    l32_split16 = run_partitioned_sequence(tokens_32, [16, 16], weights, rope_tables, cfg, 2, "per_loop")
    diff_l32 = float((l32_single["scaledLogits"] - l32_split16["scaledLogits"]).abs().max().item())
    if diff_l32 > 1e-5:
        raise RuntimeError(f"Long-chunk L=32 drift vs [16,16]: {diff_l32}")

    return {
        "tokens12": tokens_12,
        "chunkPartition": [4, 5, 3],
        "s12ByK": s12_results,
        "longChunkDr24": {
            "kLoops": 2,
            "tokens28": tokens_28,
            "l24Plus4": {
                "chunkSizes": [24, 4],
                "refChunkSizes": [8, 8, 8, 4],
                "codaHidden": to_f32_list(l24_plus_4["codaHidden"], "dr24.l24p4.coda"),
                "scaledLogits": to_f32_list(l24_plus_4["scaledLogits"], "dr24.l24p4.logits"),
                "pytorchVsRefMaxAbsDiff": diff_l24,
            },
            "tokens32": tokens_32,
            "l32SingleShot": {
                "chunkSizes": [32],
                "refChunkSizes": [16, 16],
                "codaHidden": to_f32_list(l32_single["codaHidden"], "dr24.l32.coda"),
                "scaledLogits": to_f32_list(l32_single["scaledLogits"], "dr24.l32.logits"),
                "pytorchVsRefMaxAbsDiff": diff_l32,
            },
        },
    }


def build_last_loop_theorems_fixtures(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # Constructive non-degenerate 12-token sequence (crosses W=8) exercising last_loop Theorems 1-3:
    # ["<think>", "loop", "sphere", "anchor", "cache", "window", "full", "gate", "scale", "draft", "verify", "accept"]
    tokens_12 = [12, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

    # Theorem 2: k=1 universal equivalence (last_loop == per_loop on all partitions)
    k1_pl_full = run_partitioned_sequence(tokens_12, [12], weights, rope_tables, cfg, 1, "per_loop")
    k1_ll_full = run_partitioned_sequence(tokens_12, [12], weights, rope_tables, cfg, 1, "last_loop")
    k1_ll_4_8 = run_partitioned_sequence(tokens_12, [4, 8], weights, rope_tables, cfg, 1, "last_loop")
    k1_ll_1by1 = run_partitioned_sequence(tokens_12, [1] * 12, weights, rope_tables, cfg, 1, "last_loop")

    th2_full_diff = float((k1_ll_full["scaledLogits"] - k1_pl_full["scaledLogits"]).abs().max().item())
    th2_4_8_diff = float((k1_ll_4_8["scaledLogits"] - k1_pl_full["scaledLogits"]).abs().max().item())
    th2_1by1_diff = float((k1_ll_1by1["scaledLogits"] - k1_pl_full["scaledLogits"]).abs().max().item())
    if th2_full_diff > 1e-6 or th2_4_8_diff > 1e-5 or th2_1by1_diff > 1e-5:
        raise RuntimeError(
            f"Theorem 2 (k=1) failed: full={th2_full_diff}, 4_8={th2_4_8_diff}, 1by1={th2_1by1_diff}"
        )

    # Theorems 1 & 3 for k=2 and k=4
    by_k: dict[str, Any] = {}
    for k in (2, 4):
        pl_full = run_partitioned_sequence(tokens_12, [12], weights, rope_tables, cfg, k, "per_loop")
        ll_full = run_partitioned_sequence(tokens_12, [12], weights, rope_tables, cfg, k, "last_loop")
        ll_4_8 = run_partitioned_sequence(tokens_12, [4, 8], weights, rope_tables, cfg, k, "last_loop")
        ll_1by1 = run_partitioned_sequence(tokens_12, [1] * 12, weights, rope_tables, cfg, k, "last_loop")

        th1_single_shot_diff = float((ll_full["scaledLogits"] - pl_full["scaledLogits"]).abs().max().item())
        prefix_0_4_diff = float((ll_4_8["scaledLogits"][:4] - pl_full["scaledLogits"][:4]).abs().max().item())
        suffix_4_12_vs_pl_diff = float((ll_4_8["scaledLogits"][4:] - pl_full["scaledLogits"][4:]).abs().max().item())
        chunk_vs_1by1_pos1_diff = float((ll_4_8["scaledLogits"][1:] - ll_1by1["scaledLogits"][1:]).abs().max().item())
        single_vs_1by1_pos1_diff = float((ll_full["scaledLogits"][1:] - ll_1by1["scaledLogits"][1:]).abs().max().item())

        if th1_single_shot_diff > 1e-6:
            raise RuntimeError(f"Theorem 1 failed at k={k}: {th1_single_shot_diff}")
        if prefix_0_4_diff > 1e-6:
            raise RuntimeError(f"Chunk [4,8] prefix [0..4] diverged at k={k}: {prefix_0_4_diff}")
        if not (suffix_4_12_vs_pl_diff > 1e-2):
            raise RuntimeError(
                f"Theorem 3 constructive divergence [4,8] vs per_loop too small at k={k}: {suffix_4_12_vs_pl_diff}"
            )
        if not (chunk_vs_1by1_pos1_diff > 1e-2):
            raise RuntimeError(
                f"Theorem 3 constructive divergence [4,8] vs 1-by-1 too small at k={k}: {chunk_vs_1by1_pos1_diff}"
            )

        by_k[str(k)] = {
            "kLoops": k,
            "theorem1SingleShotVsPerLoopMaxAbs": th1_single_shot_diff,
            "chunk4_8Prefix0_4VsPerLoopMaxAbs": prefix_0_4_diff,
            "chunk4_8Suffix4_12VsPerLoopMaxAbs": suffix_4_12_vs_pl_diff,
            "chunk4_8VsOneByOnePos1_12MaxAbs": chunk_vs_1by1_pos1_diff,
            "singleShotVsOneByOnePos1_12MaxAbs": single_vs_1by1_pos1_diff,
            "partitions": {
                "perLoopSingleShot": {
                    "codaHidden": to_f32_list(pl_full["codaHidden"], f"ll.k{k}.pl.coda"),
                    "scaledLogits": to_f32_list(pl_full["scaledLogits"], f"ll.k{k}.pl.logits"),
                },
                "lastLoopSingleShot": {
                    "codaHidden": to_f32_list(ll_full["codaHidden"], f"ll.k{k}.ll12.coda"),
                    "scaledLogits": to_f32_list(ll_full["scaledLogits"], f"ll.k{k}.ll12.logits"),
                },
                "lastLoopChunk4_8": {
                    "codaHidden": to_f32_list(ll_4_8["codaHidden"], f"ll.k{k}.ll48.coda"),
                    "scaledLogits": to_f32_list(ll_4_8["scaledLogits"], f"ll.k{k}.ll48.logits"),
                },
                "lastLoopOneByOne": {
                    "codaHidden": to_f32_list(ll_1by1["codaHidden"], f"ll.k{k}.ll1x12.coda"),
                    "scaledLogits": to_f32_list(ll_1by1["scaledLogits"], f"ll.k{k}.ll1x12.logits"),
                },
            },
        }

    # Finding 4 Degenerate W_v = 0 Counterexample (F_theta != Id, yet last_loop [12] == [4,8])
    zero_v_weights = {k_name: v.clone() for k_name, v in weights.items()}
    for p in range(16, 56):
        if MOTIF[p % 8] != "mlp":
            zero_v_weights[f"backbone.layers.{p}.mixer.v_proj.weight"].zero_()

    zv_full = run_partitioned_sequence(tokens_12, [12], zero_v_weights, rope_tables, cfg, 2, "last_loop")
    zv_4_8 = run_partitioned_sequence(tokens_12, [4, 8], zero_v_weights, rope_tables, cfg, 2, "last_loop")
    zv_pl_4_8 = run_partitioned_sequence(tokens_12, [4, 8], zero_v_weights, rope_tables, cfg, 2, "per_loop")
    zv_kv = PyTorchDoryKvCache(cfg, 2, "last_loop")
    zv_one = run_forward_chunk(tokens_12, zero_v_weights, rope_tables, cfg, zv_kv)
    core_non_identity_delta = float(
        (zv_one["loopExits"][0] - zv_one["loopEntries"][0]).abs().max().item()
    )
    zv_partition_diff = float((zv_full["scaledLogits"] - zv_4_8["scaledLogits"]).abs().max().item())
    zv_same_chunk_ll_vs_pl_diff = float(
        (zv_4_8["scaledLogits"] - zv_pl_4_8["scaledLogits"]).abs().max().item()
    )
    if not (
        core_non_identity_delta > 0.05
        and zv_partition_diff <= 1e-6
        and zv_same_chunk_ll_vs_pl_diff == 0.0
    ):
        raise RuntimeError(
            f"W_v=0 counterexample invariant failed: core_delta={core_non_identity_delta}, "
            f"full_vs_48={zv_partition_diff}, ll48_vs_pl48={zv_same_chunk_ll_vs_pl_diff}"
        )

    return {
        "tokens12": tokens_12,
        "theorem2K1": {
            "lastLoopSingleShotVsPerLoopMaxAbs": th2_full_diff,
            "lastLoopChunk4_8VsPerLoopMaxAbs": th2_4_8_diff,
            "lastLoopOneByOneVsPerLoopMaxAbs": th2_1by1_diff,
            "scaledLogits": to_f32_list(k1_ll_full["scaledLogits"], "ll.k1.logits"),
        },
        "byK": by_k,
        "degenerateZeroVProjCounterexample": {
            "kLoops": 2,
            "coreNonIdentityMaxAbsDelta": core_non_identity_delta,
            "lastLoopFullVsChunk4_8MaxAbs": zv_partition_diff,
            "lastLoopChunk4_8VsPerLoopChunk4_8MaxAbs": zv_same_chunk_ll_vs_pl_diff,
        },
    }


def build_upstream_differential_checks(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Layer A/C Upstream Differential Audit (DR5, DR6, DR8, DR9):
      - DR5: Layer-0 pre-norm (transformers) vs no Layer-0 pre-norm (vLLM/llama.cpp/TeX)
             under unit-norm embeddings (<= 1e-6 on L0) vs non-unit 3.0x embeddings (> 1e-3!).
      - DR6: RoPE -> Q/K L2Norm (vLLM/llama.cpp/TeX) vs Q/K L2Norm -> RoPE (transformers)
             in FP32 (<= 1e-6) vs BF16.
      - DR8: Literal u_1 = N(2x) & post-loop s_t = N(F_theta(u_t)) vs elided variant.
      - DR9: vLLM/llama.cpp FP32 scale/gate/SwiGLU/NLERP compute vs transformers native BF16.
    """
    tokens = [1, 10, 4, 18, 40, 25, 26, 12]
    emb = weights["backbone.embeddings.weight"]
    h_unit = emb[torch.tensor(tokens, dtype=torch.long)].clone()
    h_nonunit = h_unit * 3.0
    eps = float(cfg["l2NormEps"])

    # --- DR5: Layer-0 pre-norm on unit vs non-unit embeddings ---
    b_l0_canon_u = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_l0_canon_u, _ = run_attention_sublayer(h_unit, 0, 0, 0, weights, rope_tables, b_l0_canon_u, True, cfg)
    b_l0_pre_u = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_l0_pre_u, _ = run_attention_sublayer(
        l2_norm(h_unit, eps=eps), 0, 0, 0, weights, rope_tables, b_l0_pre_u, True, cfg
    )
    dr5_unit_l0_diff = float((out_l0_canon_u - out_l0_pre_u).abs().max().item())

    b_l0_canon_nu = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_l0_canon_nu, _ = run_attention_sublayer(h_nonunit, 0, 0, 0, weights, rope_tables, b_l0_canon_nu, True, cfg)
    b_l0_pre_nu = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_l0_pre_nu, _ = run_attention_sublayer(
        l2_norm(h_nonunit, eps=eps), 0, 0, 0, weights, rope_tables, b_l0_pre_nu, True, cfg
    )
    dr5_nonunit_l0_diff = float((out_l0_canon_nu - out_l0_pre_nu).abs().max().item())

    # Full 72L k=2 DR5 comparison (unit vs 3x non-unit)
    kv_u_canon = PyTorchDoryKvCache(cfg, 2, "per_loop")
    res_u_canon = run_forward_chunk(tokens, weights, rope_tables, cfg, kv_u_canon, layer0_prenorm=False)
    kv_u_pre = PyTorchDoryKvCache(cfg, 2, "per_loop")
    res_u_pre = run_forward_chunk(tokens, weights, rope_tables, cfg, kv_u_pre, layer0_prenorm=True)
    dr5_unit_72l_diff = float((res_u_canon["scaledLogits"] - res_u_pre["scaledLogits"]).abs().max().item())

    nonunit_emb_table = emb * 3.0
    kv_nu_canon = PyTorchDoryKvCache(cfg, 2, "per_loop")
    res_nu_canon = run_forward_chunk(
        tokens, weights, rope_tables, cfg, kv_nu_canon, layer0_prenorm=False, embeddings_override=nonunit_emb_table
    )
    kv_nu_pre = PyTorchDoryKvCache(cfg, 2, "per_loop")
    res_nu_pre = run_forward_chunk(
        tokens, weights, rope_tables, cfg, kv_nu_pre, layer0_prenorm=True, embeddings_override=nonunit_emb_table
    )
    dr5_nonunit_72l_diff = float((res_nu_canon["scaledLogits"] - res_nu_pre["scaledLogits"]).abs().max().item())

    if dr5_unit_l0_diff > 1e-6:
        raise RuntimeError(f"DR5 unit-embedding Layer-0 diff too large: {dr5_unit_l0_diff}")
    if not (dr5_nonunit_l0_diff > 1e-3 and dr5_nonunit_72l_diff > 1e-2):
        raise RuntimeError(
            f"DR5 non-unit embedding divergence too small: L0={dr5_nonunit_l0_diff}, 72L={dr5_nonunit_72l_diff}"
        )

    # --- DR6: RoPE -> L2Norm vs L2Norm -> RoPE (FP32 vs BF16) ---
    b_dr6_p0_a = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_dr6_p0_a, _ = run_attention_sublayer(
        h_unit, 0, 0, 0, weights, rope_tables, b_dr6_p0_a, True, cfg, rope_after_norm=False
    )
    b_dr6_p0_b = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_dr6_p0_b, _ = run_attention_sublayer(
        h_unit, 0, 0, 0, weights, rope_tables, b_dr6_p0_b, True, cfg, rope_after_norm=True
    )
    dr6_p0_fp32_diff = float((out_dr6_p0_a - out_dr6_p0_b).abs().max().item())

    b_dr6_p6_a = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(6, 0)
    out_dr6_p6_a, _ = run_attention_sublayer(
        h_unit, 6, 0, 0, weights, rope_tables, b_dr6_p6_a, True, cfg, rope_after_norm=False
    )
    b_dr6_p6_b = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(6, 0)
    out_dr6_p6_b, _ = run_attention_sublayer(
        h_unit, 6, 0, 0, weights, rope_tables, b_dr6_p6_b, True, cfg, rope_after_norm=True
    )
    dr6_p6_fp32_diff = float((out_dr6_p6_a - out_dr6_p6_b).abs().max().item())

    # BF16 RoPE <-> Norm order check on Q/K heads directly
    q_h0 = (h_unit @ weights["backbone.layers.0.mixer.q_proj.weight"].T).view(len(tokens), 8, 8)
    pos_t = torch.arange(len(tokens), dtype=torch.long)
    _, cos_s, sin_s = rope_tables["sliding"]
    q_rope_then_norm_bf16 = l2_norm(
        apply_neox_rope(q_h0.to(torch.bfloat16).to(torch.float32), pos_t, cos_s, sin_s), eps=eps
    ).to(torch.bfloat16)
    q_norm_then_rope_bf16 = apply_neox_rope(
        l2_norm(q_h0.to(torch.bfloat16).to(torch.float32), eps=eps).to(torch.bfloat16).to(torch.float32),
        pos_t,
        cos_s,
        sin_s,
    ).to(torch.bfloat16)
    dr6_q_bf16_diff = float(
        (q_rope_then_norm_bf16.to(torch.float32) - q_norm_then_rope_bf16.to(torch.float32)).abs().max().item()
    )

    if dr6_p0_fp32_diff > 1e-6 or dr6_p6_fp32_diff > 1e-6:
        raise RuntimeError(
            f"DR6 FP32 RoPE<->L2Norm order drift exceeded 1e-6: p0={dr6_p0_fp32_diff}, p6={dr6_p6_fp32_diff}"
        )

    # --- DR8: Literal u_1 = N(2x) & post-loop s_t = N(F_theta(u_t)) vs elided ---
    kv_dr8_elided = PyTorchDoryKvCache(cfg, 2, "per_loop")
    res_dr8_elided = run_forward_chunk(
        tokens, weights, rope_tables, cfg, kv_dr8_elided, elide_n2x_and_post_norm=True
    )
    dr8_u1_diff = float((res_u_canon["loopEntries"][0] - res_dr8_elided["loopEntries"][0]).abs().max().item())
    dr8_logits_diff = float((res_u_canon["scaledLogits"] - res_dr8_elided["scaledLogits"]).abs().max().item())
    if not (dr8_u1_diff > 1e-8 and dr8_u1_diff < 1e-5):
        raise RuntimeError(f"Unexpected DR8 literal N(2x) u_1 shift: {dr8_u1_diff}")

    # --- DR9: vLLM/llama.cpp FP32 vs transformers native BF16 scale/gate/SwiGLU/NLERP ---
    b_dr9_p0_bf = PyTorchDoryKvCache(cfg, 1, "per_loop").get_bank(0, 0)
    out_dr9_p0_bf, _ = run_attention_sublayer(
        h_unit, 0, 0, 0, weights, rope_tables, b_dr9_p0_bf, True, cfg, bf16_scale_compute=True
    )
    dr9_p0_attn_bf16_diff = float((out_l0_canon_u - out_dr9_p0_bf).abs().max().item())

    out_p1_fp32, _ = run_mlp_sublayer(h_unit, 1, weights, cfg, bf16_scale_compute=False)
    out_p1_bf16, _ = run_mlp_sublayer(h_unit, 1, weights, cfg, bf16_scale_compute=True)
    dr9_p1_mlp_bf16_diff = float((out_p1_fp32 - out_p1_bf16).abs().max().item())

    kv_dr9_bf16 = PyTorchDoryKvCache(cfg, 2, "per_loop")
    res_dr9_bf16 = run_forward_chunk(
        tokens,
        weights,
        rope_tables,
        cfg,
        kv_dr9_bf16,
        layer0_prenorm=True,
        rope_after_norm=True,
        bf16_scale_compute=True,
    )
    dr9_72l_bf16_max_diff = float((res_u_canon["scaledLogits"] - res_dr9_bf16["scaledLogits"]).abs().max().item())
    dr9_72l_bf16_mean_diff = float((res_u_canon["scaledLogits"] - res_dr9_bf16["scaledLogits"]).abs().mean().item())
    if not (dr9_72l_bf16_max_diff > 1e-3):
        raise RuntimeError(
            f"Expected measurable transformers BF16 vs vLLM FP32 drift > 1e-3, got {dr9_72l_bf16_max_diff}"
        )

    return {
        "tokenIds": tokens,
        "dr5Layer0PreNorm": {
            "unitEmbeddingsLayer0MaxAbsDiff": dr5_unit_l0_diff,
            "unitEmbeddingsFull72LMaxAbsDiff": dr5_unit_72l_diff,
            "nonUnit3xScaleFactor": 3.0,
            "nonUnitEmbeddingsLayer0MaxAbsDiff": dr5_nonunit_l0_diff,
            "nonUnitEmbeddingsFull72LMaxAbsDiff": dr5_nonunit_72l_diff,
            "canonicalUnitLayer0Out": to_f32_list(out_l0_canon_u, "dr5.canonU"),
            "canonicalNonUnitLayer0Out": to_f32_list(out_l0_canon_nu, "dr5.canonNU"),
            "preNormNonUnitLayer0Out": to_f32_list(out_l0_pre_nu, "dr5.preNU"),
        },
        "dr6RopeNormOrder": {
            "p0SwaFp32MaxAbsDiff": dr6_p0_fp32_diff,
            "p6FullFp32MaxAbsDiff": dr6_p6_fp32_diff,
            "p0QueryBf16MaxAbsDiff": dr6_q_bf16_diff,
        },
        "dr8LiteralLoopBoundaries": {
            "literalVsElidedU1MaxAbsDiff": dr8_u1_diff,
            "literalVsElidedFull72LLogitsMaxAbsDiff": dr8_logits_diff,
            "canonicalU1": to_f32_list(res_u_canon["loopEntries"][0], "dr8.canonU1"),
        },
        "dr9Fp32VsTransformersBf16": {
            "p0AttnBf16VsFp32MaxAbsDiff": dr9_p0_attn_bf16_diff,
            "p1MlpBf16VsFp32MaxAbsDiff": dr9_p1_mlp_bf16_diff,
            "full72LK2TransformersBf16VsVllmFp32MaxAbsDiff": dr9_72l_bf16_max_diff,
            "full72LK2TransformersBf16VsVllmFp32MeanAbsDiff": dr9_72l_bf16_mean_diff,
            "vllmFp32CanonicalScaledLogits": to_f32_list(res_u_canon["scaledLogits"], "dr9.fp32Logits"),
        },
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean-room PyTorch 2.15.0 backbone parity fixtures for Dory1.0-4B."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--params-dir",
        type=Path,
        default=None,
        help="Directory containing dory_mini_manifest.json and dory_mini_weights.f32 emitted by export_dory_params.mjs",
    )
    input_group.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to dory_mini_manifest.json emitted by export_dory_params.mjs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output compact single-line JSON fixture path",
    )
    args = parser.parse_args()

    if args.manifest is not None:
        manifest_path = args.manifest
    elif args.params_dir is not None:
        manifest_path = args.params_dir / "dory_mini_manifest.json"
    else:
        manifest_path = DEFAULT_PARAMS_DIR / "dory_mini_manifest.json"

    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Expected pinned torch=={PINNED_TORCH_VERSION}, got {torch.__version__}"
        )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    manifest, weights = load_exported_dory_weights(manifest_path)
    cfg = manifest["config"]

    rope_tables = {
        "sliding": compute_rope_tables(
            int(cfg["maxPositionEmbeddings"]), int(cfg["headDim"]), float(cfg["ropeThetaSliding"])
        ),
        "full": compute_rope_tables(
            int(cfg["maxPositionEmbeddings"]), int(cfg["headDim"]), float(cfg["ropeThetaFull"])
        ),
    }

    spot_names = [
        "backbone.embeddings.weight",
        "backbone.layers.0.attn_alpha",
        "backbone.layers.16.attn_alpha",
        "backbone.layers.16.mixer.sqk",
        "backbone.layers.17.mixer.suv_gate",
        "lm_head.weight",
        "lm_head.logit_scale",
    ]
    sha_lookup = {entry["name"]: entry["sha256"] for entry in manifest["tensors"]}
    spot_tensor_sha256 = {name: sha_lookup[name] for name in spot_names}

    cat1_single_ops = build_single_op_fixtures(weights, rope_tables, cfg)
    cat2_sublayers = build_sublayer_fixtures(weights, rope_tables, cfg)
    cat3_full72l = build_full_72l_fixtures(weights, rope_tables, cfg)
    cat4_per_loop = build_per_loop_and_long_chunk_fixtures(weights, rope_tables, cfg)
    cat5_last_loop = build_last_loop_theorems_fixtures(weights, rope_tables, cfg)
    cat6_upstream_diff = build_upstream_differential_checks(weights, rope_tables, cfg)

    fixture = {
        "schemaVersion": "dory1-backbone-pytorch-parity-v1",
        "format": manifest["format"],
        "modelWeightId": manifest["modelWeightId"],
        "weightsSha256": manifest["weightsSha256"],
        "backboneSeed": manifest["backboneSeed"],
        "tensorCount": manifest["tensorCount"],
        "totalFloats": manifest["totalFloats"],
        "totalBytes": manifest["totalBytes"],
        "torchVersion": torch.__version__,
        "pythonVersion": platform.python_version(),
        "spotTensorSha256": spot_tensor_sha256,
        "singleOps": cat1_single_ops,
        "sublayers": cat2_sublayers,
        "full72L": cat3_full72l,
        "perLoopAndLongChunk": cat4_per_loop,
        "lastLoopTheorems": cat5_last_loop,
        "upstreamDifferential": cat6_upstream_diff,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(fixture, allow_nan=False, separators=(",", ":")) + "\n"
    args.out.write_text(serialized, encoding="utf-8")

    k2_ll = cat5_last_loop["byK"]["2"]
    k4_ll = cat5_last_loop["byK"]["4"]
    print(
        f"Generated PyTorch {torch.__version__} Dory backbone fixture -> {args.out} "
        f"({len(serialized):,} B, modelWeightId={manifest['modelWeightId'][:16]}..., "
        f"last_loop_k2_div={k2_ll['chunk4_8Suffix4_12VsPerLoopMaxAbs']:.4e}, "
        f"last_loop_k4_div={k4_ll['chunk4_8Suffix4_12VsPerLoopMaxAbs']:.4e}, "
        f"dr9_bf16_drift={cat6_upstream_diff['dr9Fp32VsTransformersBf16']['full72LK2TransformersBf16VsVllmFp32MaxAbsDiff']:.4e})"
    )


if __name__ == "__main__":
    main()
