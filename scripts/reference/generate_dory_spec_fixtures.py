#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator
for NVIDIA Nemotron-Labs-Dory1.0-4B Contract A §6.2 Self-Speculative Decoding
(DORY2 Slice 2C).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12.x + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-dory-spec.json` with 0 B diff:
  node scripts/reference/export_dory_params.mjs [paramsDir]
  uv run scripts/reference/generate_dory_spec_fixtures.py [--params-dir <paramsDir> | --manifest <manifestPath>] [--out <outPath>]

Clean-Room Mathematical Reference (authored directly from DORY0 §5–§8 & DORY0-R):
  - Layer 1: Binary64 algebraic marginal identity <= 1e-15 across diverse (p, q) pairs.
  - Layer 2: Exact small-vocab (V = 4) 1-step & 2-step joint analytical enumeration <= 1e-15.
  - Layer 3: Explicit-variate deterministic boundary fixtures (r = 0, p/q +/- 1e-7, 1 - 1e-12,
             unconditional p >= q, and filtered (temperature, topK, topP) transform consistency).
  - Layer 4: Adversarial edge suite (p === q, disjoint support, partial support, sub-1e-38
             q = 1e-40 / p = 5e-41, near-identical Delta = 4e-13, concentrated & one-hot).
  - Layer 5: Full 72-sublayer Contract A (104 persistent banks: 8 Input + 80 Loops 1..4 +
             8 output_deep + 8 output_shallow) self-speculation across S > W = 8 eviction &
             rollback:
               * Multi-cycle greedy (T = 0) verifying reused k = 4 vs fresh uncached k = 4
                 (<= 2e-5), greedy === standard k = 4 AR, AND multi-cycle shallow draft vs
                 standalone k = 1 Dory (<= 2e-5).
               * Multi-cycle stochastic explicit-variate suite exercising partial_accept_reject,
                 full_accept_bonus (+ 1-token Shallow Coda catch-up), zero_accept_reject,
                 full_accept_eos (early draft EOS stop, no bonus), and final_slot_deep_decode
                 (L = S_max - 1 = 31, gamma_eff = 0).
  - Layer 6: Reference V = 64 target/proposal distributions (minExpectedCount >= 10 at N = 20,000)
             for Pearson chi^2 < 103.51 (df = 63, alpha = 0.001) PLUS analytical oracle metrics
             for 2 mandatory failing negative controls (resampling from p; buggy 1e-38 clamp)
             and buggy Z <= 1e-12 fallback.

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
DEFAULT_OUT_PATH = Path("tests/fixtures/pytorch-dory-spec.json")

EXPECTED_TENSOR_COUNT = 507
EXPECTED_FLOAT_COUNT = 93_248
EXPECTED_WEIGHT_BYTES = 372_992
CANONICAL_MAGIC = "DORY-MINI-F32-V1\n"

CONTRACT_A_BADGE = (
    "Reconstructed from Paper §6.2 (Contract A) · Not in Upstream Runtimes · per_loop Only"
)
CONTRACT_A_BANK_COUNT = 104
DRAFT_LOOPS = 1
VERIFY_LOOPS = 4
DEFAULT_GAMMA = 4
PEARSON_CHI2_DF63_ALPHA_001 = 103.51

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


def assert_finite_f64_list(vals: list[float], label: str) -> list[float]:
    out: list[float] = []
    for idx, v in enumerate(vals):
        fv = float(v)
        if not math.isfinite(fv):
            raise RuntimeError(f"Non-finite binary64 value {v} at {label}[{idx}]")
        out.append(fv)
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
    x32 = x.to(torch.float32)
    denom = torch.sqrt(torch.sum(x32 * x32, dim=-1, keepdim=True) + eps)
    return x32 / denom


def effective_residual_alpha(
    raw_alpha: torch.Tensor, alpha_init: float, sigma_0: float
) -> torch.Tensor:
    return torch.abs(raw_alpha.to(torch.float32) * (alpha_init / sigma_0))


def effective_sqk_squared(
    raw_sqk: torch.Tensor, head_dim: int, sigma_0: float
) -> torch.Tensor:
    scaled = raw_sqk.to(torch.float32) * (math.sqrt(head_dim) / sigma_0)
    return scaled * scaled


def effective_s_gate_squared(
    raw_s_gate: torch.Tensor,
    hidden_size: int,
    sigma_0: float,
    g_0: float | None = None,
) -> torch.Tensor:
    resolved_g0 = math.sqrt(hidden_size) if g_0 is None else float(g_0)
    scaled = raw_s_gate.to(torch.float32) * (math.sqrt(resolved_g0) / sigma_0)
    return scaled * scaled


def effective_suv_scale(raw_suv: torch.Tensor, hidden_size: int) -> torch.Tensor:
    return raw_suv.to(torch.float32) * math.sqrt(hidden_size)


def effective_logit_scale(raw_logit_scale: torch.Tensor, sigma_0: float) -> torch.Tensor:
    return raw_logit_scale.to(torch.float32) / sigma_0


def nlerp_residual(
    residual: torch.Tensor,
    sublayer_out: torch.Tensor,
    raw_alpha: torch.Tensor,
    alpha_init: float,
    sigma_0: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    a = l2_norm(residual, eps=eps)
    b = l2_norm(sublayer_out, eps=eps)
    lam = effective_residual_alpha(raw_alpha, alpha_init, sigma_0)
    return l2_norm(a + lam * (b - a), eps=eps)


def compute_rope_tables(
    max_positions: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    half = x.shape[-1] // 2
    cos = cos_table[position_ids][:, None, :]
    sin = sin_table[position_ids][:, None, :]
    x0 = x[..., :half]
    x1 = x[..., half:]
    return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)


# ============================================================================
# CLEAN-ROOM CONTRACT A 104-BANK KV CACHE & STAGED 72-SUBLAYER FORWARD ENGINE
# ============================================================================


@dataclass
class PyTorchKvBank:
    physical_layer: int
    bank_kind: str
    is_sliding: bool
    window_size: int | None
    k_entries: list[torch.Tensor]
    v_entries: list[torch.Tensor]
    abs_positions: list[int]
    writer_loops: list[int]

    @property
    def logical_len(self) -> int:
        return len(self.abs_positions)

    def rewind(self, target_len: int) -> None:
        if target_len < 0 or target_len > self.logical_len:
            raise RuntimeError(
                f"Invalid rewind on p={self.physical_layer} ({self.bank_kind}): "
                f"target_len={target_len}, logical_len={self.logical_len}"
            )
        del self.k_entries[target_len:]
        del self.v_entries[target_len:]
        del self.abs_positions[target_len:]
        del self.writer_loops[target_len:]


class PyTorchContractAKvCache:
    """
    Clean-room 104-bank Contract A KV cache (per_loop only, k_loops = 4):
      - 8 Input banks (p in {0,2,4,6,8,10,12,14})
      - 80 Recurrent banks (20 layers x 4 loops t in {1,2,3,4})
      - 8 output_deep Coda banks (p in {56,58,60,62,64,66,68,70})
      - 8 output_shallow Coda banks (p in {56,58,60,62,64,66,68,70}) when include_shallow=True
    """

    def __init__(self, cfg: dict[str, Any], k_loops: int, include_shallow: bool) -> None:
        self.cfg = cfg
        self.k_loops = k_loops
        self.include_shallow = include_shallow
        self.committed_len = 0

        self.input_banks: dict[int, PyTorchKvBank] = {}
        self.recurrent_banks: dict[tuple[int, int], PyTorchKvBank] = {}
        self.output_deep_banks: dict[int, PyTorchKvBank] = {}
        self.output_shallow_banks: dict[int, PyTorchKvBank] | None = {} if include_shallow else None
        self.all_banks: list[PyTorchKvBank] = []

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
                b = PyTorchKvBank(p, "input", is_swa, win, [], [], [], [])
                self.input_banks[p] = b
                self.all_banks.append(b)
            elif p < n_in + n_rec:
                for loop_slot in range(1, k_loops + 1):
                    b = PyTorchKvBank(p, "recurrent", is_swa, win, [], [], [], [])
                    self.recurrent_banks[(p, loop_slot)] = b
                    self.all_banks.append(b)
            else:
                bd = PyTorchKvBank(p, "output_deep", is_swa, win, [], [], [], [])
                self.output_deep_banks[p] = bd
                self.all_banks.append(bd)
                if self.output_shallow_banks is not None:
                    bs = PyTorchKvBank(p, "output_shallow", is_swa, win, [], [], [], [])
                    self.output_shallow_banks[p] = bs
                    self.all_banks.append(bs)

    def rewind_all(self, target_len: int) -> None:
        for bank in self.all_banks:
            bank.rewind(target_len)
        self.committed_len = target_len

    def assert_all_synced(self, expected_len: int) -> None:
        if self.committed_len != expected_len:
            raise RuntimeError(
                f"committed_len mismatch: {self.committed_len} != {expected_len}"
            )
        for bank in self.all_banks:
            if bank.logical_len != expected_len:
                raise RuntimeError(
                    f"Bank p={bank.physical_layer} ({bank.bank_kind}) desynced: "
                    f"{bank.logical_len} != {expected_len}"
                )


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
) -> torch.Tensor:
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

    q_pre = h @ w_q.T
    k_pre = h @ w_k.T
    v_new = h @ w_v.T
    g_pre = h @ w_g.T

    pos_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long)
    q_rot = apply_neox_rope(q_pre.view(seq_len, h_q, d_h), pos_ids, cos_tab, sin_tab)
    k_rot = apply_neox_rope(k_pre.view(seq_len, h_kv, d_h), pos_ids, cos_tab, sin_tab)

    q_norm = l2_norm(q_rot, eps=eps)
    k_norm = l2_norm(k_rot, eps=eps)

    eff_sqk_sq = effective_sqk_squared(raw_sqk, d_h, sigma_0)
    q_scaled = (q_norm.reshape(seq_len, q_dim) * eff_sqk_sq[None, :]).view(seq_len, h_q, d_h)
    k_cache_flat = k_norm.reshape(seq_len, kv_dim)

    if start_pos != bank.logical_len:
        raise RuntimeError(
            f"Non-contiguous KV access at p={p} ({bank.bank_kind}): "
            f"start_pos={start_pos}, logical_len={bank.logical_len}"
        )

    past_start = max(0, start_pos - (window - 1)) if is_sliding else 0
    past_k = bank.k_entries[past_start:start_pos]
    past_v = bank.v_entries[past_start:start_pos]
    past_pos = list(bank.abs_positions[past_start:start_pos])

    if past_k:
        k_vis_flat = torch.cat([torch.stack(past_k, dim=0), k_cache_flat], dim=0)
        v_vis_flat = torch.cat([torch.stack(past_v, dim=0), v_new], dim=0)
    else:
        k_vis_flat = k_cache_flat
        v_vis_flat = v_new
    kv_abs_pos = past_pos + [start_pos + i for i in range(seq_len)]
    kv_len = k_vis_flat.shape[0]

    k_vis_heads = k_vis_flat.view(kv_len, h_kv, d_h).repeat_interleave(gqa_rep, dim=1)
    v_vis_heads = v_vis_flat.view(kv_len, h_kv, d_h).repeat_interleave(gqa_rep, dim=1)

    raw_scores = torch.matmul(q_scaled.permute(1, 0, 2), k_vis_heads.permute(1, 2, 0)) / math.sqrt(d_h)
    q_pos_vec = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long)[:, None]
    k_pos_vec = torch.tensor(kv_abs_pos, dtype=torch.long)[None, :]
    diff = q_pos_vec - k_pos_vec
    visible = (diff >= 0) & (diff < window) if is_sliding else (diff >= 0)

    scores = raw_scores.masked_fill(~visible[None, :, :], float("-inf"))
    probs = F.softmax(scores.to(torch.float64), dim=-1).to(torch.float32)
    attn_ctx = torch.matmul(probs, v_vis_heads.permute(1, 0, 2)).permute(1, 0, 2).contiguous().view(seq_len, q_dim)

    if write_kv:
        for i in range(seq_len):
            bank.k_entries.append(k_cache_flat[i].detach().clone())
            bank.v_entries.append(v_new[i].detach().clone())
            bank.abs_positions.append(start_pos + i)
            bank.writer_loops.append(loop_iter)

    eff_s_gate_sq = effective_s_gate_squared(raw_s_gate, d, sigma_0, g_0)
    gate_vals = torch.sigmoid(g_pre * eff_s_gate_sq[None, :])
    attn_sub_norm = l2_norm(attn_ctx * gate_vals, eps=eps)
    o_proj_out = attn_sub_norm @ w_o.T
    return nlerp_residual(h, o_proj_out, raw_alpha, alpha_init, sigma_0, eps=eps)


def run_mlp_sublayer(
    h: torch.Tensor,
    p: int,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> torch.Tensor:
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

    gate_scaled = (h @ w_gate.T) * effective_suv_scale(raw_suv_g, d)[None, :]
    up_scaled = (h @ w_up.T) * effective_suv_scale(raw_suv_u, d)[None, :]
    ffn_sub_norm = l2_norm(F.silu(gate_scaled) * up_scaled, eps=eps)
    down_proj_out = ffn_sub_norm @ w_down.T
    return nlerp_residual(h, down_proj_out, raw_alpha, alpha_init, sigma_0, eps=eps)


def run_prelude_stage(
    token_ids: list[int],
    start_pos: int,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    kv_cache: PyTorchContractAKvCache,
    cfg: dict[str, Any],
) -> torch.Tensor:
    n_in = int(cfg["nInputLayers"])
    idx = torch.tensor(token_ids, dtype=torch.long)
    h = weights["backbone.embeddings.weight"][idx].clone()
    for p in range(0, n_in):
        if MOTIF[p % 8] == "mlp":
            h = run_mlp_sublayer(h, p, weights, cfg)
        else:
            h = run_attention_sublayer(
                h, p, start_pos, 0, weights, rope_tables, kv_cache.input_banks[p], True, cfg
            )
    return h


def run_recurrent_stage(
    prev_state: torch.Tensor,
    x_anchor: torch.Tensor,
    start_pos: int,
    loop_iter: int,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    kv_cache: PyTorchContractAKvCache,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    n_in = int(cfg["nInputLayers"])
    n_rec = int(cfg["nRecurrentLayers"])
    eps = float(cfg["l2NormEps"])

    u_t = l2_norm(prev_state + x_anchor, eps=eps)
    h = u_t
    for p in range(n_in, n_in + n_rec):
        if MOTIF[p % 8] == "mlp":
            h = run_mlp_sublayer(h, p, weights, cfg)
        else:
            bank = kv_cache.recurrent_banks[(p, loop_iter)]
            h = run_attention_sublayer(
                h, p, start_pos, loop_iter, weights, rope_tables, bank, True, cfg
            )
    s_t = l2_norm(h, eps=eps)
    return u_t, s_t


def run_coda_and_lm_head(
    state: torch.Tensor,
    start_pos: int,
    coda_kind: str,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    kv_cache: PyTorchContractAKvCache,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    n_in = int(cfg["nInputLayers"])
    n_rec = int(cfg["nRecurrentLayers"])
    n_total = int(cfg["numHiddenLayers"])
    sigma_0 = float(cfg["initializerRange"])

    coda_banks = (
        kv_cache.output_shallow_banks
        if coda_kind == "output_shallow"
        else kv_cache.output_deep_banks
    )
    if coda_banks is None:
        raise RuntimeError("output_shallow banks unavailable on this cache")

    h = state
    for p in range(n_in + n_rec, n_total):
        if MOTIF[p % 8] == "mlp":
            h = run_mlp_sublayer(h, p, weights, cfg)
        else:
            h = run_attention_sublayer(
                h, p, start_pos, 0, weights, rope_tables, coda_banks[p], True, cfg
            )
    raw_logits = h @ weights["lm_head.weight"].T
    scaled_logits = raw_logits * effective_logit_scale(weights["lm_head.logit_scale"], sigma_0)[None, :]
    return h, scaled_logits


def run_fresh_full_forward(
    token_ids: list[int],
    k_loops: int,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Completely fresh uncached single-shot k-loop forward returning [S, V] scaled logits."""
    kv = PyTorchContractAKvCache(cfg, k_loops=k_loops, include_shallow=False)
    x_anchor = run_prelude_stage(token_ids, 0, weights, rope_tables, kv, cfg)
    state = x_anchor
    for t in range(1, k_loops + 1):
        _, state = run_recurrent_stage(state, x_anchor, 0, t, weights, rope_tables, kv, cfg)
    _, scaled_logits = run_coda_and_lm_head(state, 0, "output_deep", weights, rope_tables, kv, cfg)
    return scaled_logits


# ============================================================================
# CLEAN-ROOM BINARY64 SAMPLING, REJECTION SAMPLING & ANALYTICAL ENUMERATION
# ============================================================================


def argmax_f64(vals: list[float]) -> int:
    best_i = 0
    best_v = vals[0]
    for i in range(1, len(vals)):
        if vals[i] > best_v:
            best_v = vals[i]
            best_i = i
    return best_i


def logits_to_sampling_probs_f64(
    scaled_logits: list[float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> list[float]:
    v_len = len(scaled_logits)
    if temperature == 0.0:
        out = [0.0] * v_len
        out[argmax_f64(scaled_logits)] = 1.0
        return out

    inv_t = 1.0 / float(temperature)
    max_z = max(scaled_logits)
    exps = [math.exp((z - max_z) * inv_t) for z in scaled_logits]
    s = math.fsum(exps)
    probs = [e / s for e in exps]

    apply_top_k = top_k > 0 and top_k < v_len
    apply_top_p = top_p < 1.0
    if not apply_top_k and not apply_top_p:
        return probs

    order = sorted(range(v_len), key=lambda i: (-probs[i], -scaled_logits[i], i))
    kept = v_len
    if apply_top_k:
        kept = top_k
        for r in range(kept, v_len):
            probs[order[r]] = 0.0
        sk = sum(probs)
        probs = [p / sk if p > 0.0 else 0.0 for p in probs]

    if apply_top_p:
        cum = probs[order[0]]
        nuc = 1
        while nuc < kept and cum < top_p:
            cum += probs[order[nuc]]
            nuc += 1
        for r in range(nuc, kept):
            probs[order[r]] = 0.0
        sp = sum(probs)
        probs = [p / sp if p > 0.0 else 0.0 for p in probs]

    return probs


def sample_categorical_f64(probs: list[float], u: float) -> int:
    cdf = 0.0
    last_pos = -1
    for i, p in enumerate(probs):
        if p > 0.0:
            last_pos = i
            cdf += p
            if u < cdf:
                return i
    if last_pos < 0:
        raise RuntimeError("Zero-mass categorical distribution")
    return last_pos


def cdf_bin_midpoint(probs: list[float], target_idx: int) -> tuple[float, float]:
    """Return (midpoint, half_width) of target_idx's ascending-CDF interval."""
    lo = 0.0
    for i in range(target_idx):
        lo += probs[i]
    width = probs[target_idx]
    if width <= 0.0:
        raise RuntimeError(f"Zero probability at target_idx={target_idx}")
    return lo + 0.5 * width, 0.5 * width


def evaluate_rejection_f64(
    d: int, p: list[float], q: list[float], u_accept: float | None, is_greedy: bool
) -> tuple[bool, str]:
    if is_greedy:
        best = argmax_f64(p)
        return (d == best, "greedy_match" if d == best else "greedy_mismatch")
    qd = q[d]
    pd = p[d]
    if not (qd > 0.0):
        raise ValueError(f"NON_POSITIVE_PROPOSAL_PROB: q[{d}]={qd}")
    if pd >= qd:
        return (True, "unconditional_p_ge_q")
    if u_accept is None:
        raise ValueError("Missing uniform variate when p[d] < q[d]")
    # Division-free binary64 acceptance: r * q[d] < p[d]
    accepted = (u_accept * qd) < pd
    return (accepted, "stochastic_accept" if accepted else "stochastic_reject")


def compute_residual_f64(p: list[float], q: list[float]) -> tuple[list[float], float]:
    raw = [max(0.0, pv - qv) for pv, qv in zip(p, q)]
    z = sum(raw)
    if not (z > 0.0):
        raise ValueError(f"ZERO_RESIDUAL_MASS: Z={z}")
    return ([rv / z if rv > 0.0 else 0.0 for rv in raw], z)


def enumerate_1step_exact_f64(p: list[float], q: list[float]) -> dict[str, Any]:
    v_len = len(p)
    marginal = [0.0] * v_len
    acc_rate = 0.0
    rej_rate = 0.0
    for d in range(v_len):
        qd = q[d]
        pd = p[d]
        if qd == 0.0:
            continue
        if pd >= qd:
            marginal[d] += qd
            acc_rate += qd
        else:
            marginal[d] += pd
            acc_rate += pd
            rej_rate += qd - pd

    residual_probs: list[float] | None = None
    residual_z: float = 0.0
    if rej_rate > 0.0:
        residual_probs, residual_z = compute_residual_f64(p, q)
        for v in range(v_len):
            marginal[v] += rej_rate * residual_probs[v]

    max_abs_err = max(abs(marginal[v] - p[v]) for v in range(v_len))
    return {
        "marginalProbs": assert_finite_f64_list(marginal, "enum1.marginal"),
        "acceptanceRate": float(acc_rate),
        "rejectionRate": float(rej_rate),
        "residualMassZ": float(residual_z),
        "residualProbs": residual_probs,
        "maxAbsError": float(max_abs_err),
    }


def enumerate_2step_exact_f64(
    p1: list[float],
    q1: list[float],
    p2_by_v1: list[list[float]],
    q2_by_v1: list[list[float]],
) -> dict[str, Any]:
    v_len = len(p1)
    step2 = [enumerate_1step_exact_f64(p2_by_v1[v1], q2_by_v1[v1]) for v1 in range(v_len)]
    rej1 = sum(max(0.0, q1[d] - p1[d]) for d in range(v_len))
    res1 = compute_residual_f64(p1, q1)[0] if rej1 > 0.0 else None

    joint = [0.0] * (v_len * v_len)
    for d1 in range(v_len):
        qd1 = q1[d1]
        pd1 = p1[d1]
        if qd1 == 0.0:
            continue
        if pd1 >= qd1:
            for v2 in range(v_len):
                joint[d1 * v_len + v2] += qd1 * step2[d1]["marginalProbs"][v2]
        else:
            for v2 in range(v_len):
                joint[d1 * v_len + v2] += pd1 * step2[d1]["marginalProbs"][v2]
            rej_mass = qd1 - pd1
            assert res1 is not None
            for v1 in range(v_len):
                w1 = rej_mass * res1[v1]
                if w1 == 0.0:
                    continue
                for v2 in range(v_len):
                    joint[v1 * v_len + v2] += w1 * step2[v1]["marginalProbs"][v2]

    target_joint = [p1[v1] * p2_by_v1[v1][v2] for v1 in range(v_len) for v2 in range(v_len)]
    max_abs_err = max(abs(joint[i] - target_joint[i]) for i in range(v_len * v_len))
    return {
        "vocabSize": v_len,
        "jointProbs": assert_finite_f64_list(joint, "enum2.joint"),
        "targetJointProbs": assert_finite_f64_list(target_joint, "enum2.target"),
        "maxAbsError": float(max_abs_err),
    }


def enumerate_buggy_resample_from_p_f64(p: list[float], q: list[float]) -> dict[str, Any]:
    """Negative Control 1: buggy sampler that resamples from p instead of norm(max(0, p - q))."""
    v_len = len(p)
    alpha = sum(min(p[v], q[v]) for v in range(v_len))
    rej = 1.0 - alpha
    marginal = [min(p[v], q[v]) + rej * p[v] for v in range(v_len)]
    max_abs_err = max(abs(marginal[v] - p[v]) for v in range(v_len))
    tvd = 0.5 * sum(abs(marginal[v] - p[v]) for v in range(v_len))
    return {
        "marginalProbs": assert_finite_f64_list(marginal, "neg1.marginal"),
        "acceptanceRate": float(alpha),
        "maxAbsError": float(max_abs_err),
        "tvdVsTarget": float(tvd),
    }


# ============================================================================
# CLEAN-ROOM CONTRACT A STATE MACHINE DRIVER (PYTORCH 2.15.0)
# ============================================================================


def prefill_contract_a(
    prompt_tokens: list[int],
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    initial_uniform: float | None = None,
) -> dict[str, Any]:
    kv = PyTorchContractAKvCache(cfg, k_loops=VERIFY_LOOPS, include_shallow=True)
    if len(kv.all_banks) != CONTRACT_A_BANK_COUNT:
        raise RuntimeError(f"Expected {CONTRACT_A_BANK_COUNT} Contract A banks, got {len(kv.all_banks)}")

    p_len = len(prompt_tokens)
    x_anchor = run_prelude_stage(prompt_tokens, 0, weights, rope_tables, kv, cfg)
    state = x_anchor
    s1_state: torch.Tensor | None = None
    for t in range(1, VERIFY_LOOPS + 1):
        _, state = run_recurrent_stage(state, x_anchor, 0, t, weights, rope_tables, kv, cfg)
        if t == 1:
            s1_state = state

    assert s1_state is not None
    _, deep_scaled_logits = run_coda_and_lm_head(
        state, 0, "output_deep", weights, rope_tables, kv, cfg
    )
    _, shallow_scaled_logits = run_coda_and_lm_head(
        s1_state, 0, "output_shallow", weights, rope_tables, kv, cfg
    )
    kv.committed_len = p_len
    kv.assert_all_synced(p_len)

    last_deep_logits = to_f32_list(deep_scaled_logits[-1], "prefill.lastDeepLogits")
    last_shallow_logits = to_f32_list(shallow_scaled_logits[-1], "prefill.lastShallowLogits")
    deep_probs = logits_to_sampling_probs_f64(last_deep_logits, temperature, top_k, top_p)
    shallow_probs = logits_to_sampling_probs_f64(last_shallow_logits, temperature, top_k, top_p)

    if temperature == 0.0:
        pending_tok = argmax_f64(last_deep_logits)
    else:
        assert initial_uniform is not None
        pending_tok = sample_categorical_f64(deep_probs, initial_uniform)

    return {
        "kv": kv,
        "committedLen": p_len,
        "pendingToken": pending_tok,
        "lastDeepScaledLogits": last_deep_logits,
        "lastDeepProbs": deep_probs,
        "lastShallowScaledLogits": last_shallow_logits,
        "lastShallowProbs": shallow_probs,
    }


def run_contract_a_cycle(
    kv: PyTorchContractAKvCache,
    committed_prefix: list[int],
    pending_token: int,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    *,
    gamma: int = DEFAULT_GAMMA,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    explicit_draft_uniforms: list[float] | None = None,
    explicit_accept_uniforms: list[float] | None = None,
    explicit_correction_or_bonus_uniform: float | None = None,
) -> dict[str, Any]:
    s_max = int(cfg["maxPositionEmbeddings"])
    eos_set = set(int(x) for x in cfg["eosTokenIds"])
    start_len = kv.committed_len
    kv.assert_all_synced(start_len)
    if len(committed_prefix) != start_len:
        raise RuntimeError("committed_prefix length mismatch")

    is_greedy = temperature == 0.0
    gamma_eff = min(gamma, s_max - 1 - start_len)

    # Special Transition: L = S_max - 1 (gamma_eff == 0)
    if gamma_eff == 0:
        x_anchor = run_prelude_stage([pending_token], start_len, weights, rope_tables, kv, cfg)
        state = x_anchor
        s1_state: torch.Tensor | None = None
        for t in range(1, VERIFY_LOOPS + 1):
            _, state = run_recurrent_stage(state, x_anchor, start_len, t, weights, rope_tables, kv, cfg)
            if t == 1:
                s1_state = state
        assert s1_state is not None
        _, deep_logits = run_coda_and_lm_head(
            state, start_len, "output_deep", weights, rope_tables, kv, cfg
        )
        # 1-token Shallow Coda catch-up
        run_coda_and_lm_head(
            s1_state, start_len, "output_shallow", weights, rope_tables, kv, cfg
        )
        kv.committed_len = start_len + 1
        kv.assert_all_synced(start_len + 1)

        deep_row = to_f32_list(deep_logits[0], "finalSlot.deepLogits")
        target_probs = logits_to_sampling_probs_f64(deep_row, temperature, top_k, top_p)
        if is_greedy:
            y_final = argmax_f64(deep_row)
        else:
            assert explicit_correction_or_bonus_uniform is not None
            y_final = sample_categorical_f64(target_probs, explicit_correction_or_bonus_uniform)

        halt_reason = "eos" if y_final in eos_set else "capacity_exhausted"

        # Independent fresh k=4 check
        fresh_k4 = run_fresh_full_forward(
            committed_prefix + [pending_token], VERIFY_LOOPS, weights, rope_tables, cfg
        )
        reused_vs_fresh_err = float((deep_logits[0] - fresh_k4[start_len]).abs().max().item())
        if reused_vs_fresh_err > 1e-5:
            raise RuntimeError(f"Final-slot deep logits mismatch vs fresh k=4: {reused_vs_fresh_err}")

        return {
            "startCommittedLen": start_len,
            "endCommittedLen": start_len + 1,
            "inputPendingToken": pending_token,
            "nextPendingToken": y_final,
            "gammaEffective": 0,
            "draftCount": 0,
            "draftEndedWithEos": False,
            "acceptedDraftCount": 0,
            "emittedTokens": [y_final],
            "outcome": "final_slot_deep_decode",
            "haltReason": halt_reason,
            "reusedActivationPositions": 0,
            "verifiedPositions": 1,
            "shallowCodaCatchupExecuted": True,
            "rolledBackPositions": 0,
            "correctionOrBonusToken": y_final,
            "draftSteps": [],
            "verifyPositions": [
                {
                    "verifyOffset": 0,
                    "absPos": start_len,
                    "inputTokenId": pending_token,
                    "reusedPreludeAndLoop1": False,
                    "deepScaledLogits": deep_row,
                    "accepted": None,
                }
            ],
            "reusedVsFreshK4MaxAbs": reused_vs_fresh_err,
            "shallowDraftVsStandaloneK1MaxAbs": 0.0,
            "committedDeltaTokens": [pending_token],
        }

    # Stage A: Shallow k = 1 Draft Phase (j = 0 .. gamma_eff - 1)
    tokens_seq = [pending_token]
    reused_x: list[torch.Tensor] = []
    reused_s1: list[torch.Tensor] = []
    draft_steps: list[dict[str, Any]] = []
    draft_ended_with_eos = False

    for j in range(gamma_eff):
        abs_pos = start_len + j
        inp_tok = tokens_seq[j]
        x_j = run_prelude_stage([inp_tok], abs_pos, weights, rope_tables, kv, cfg)
        _, s1_j = run_recurrent_stage(x_j, x_j, abs_pos, 1, weights, rope_tables, kv, cfg)
        _, sh_logits = run_coda_and_lm_head(
            s1_j, abs_pos, "output_shallow", weights, rope_tables, kv, cfg
        )
        sh_row = to_f32_list(sh_logits[0], f"draft[{j}].shallowLogits")
        q_probs = logits_to_sampling_probs_f64(sh_row, temperature, top_k, top_p)

        if is_greedy:
            prop_tok = argmax_f64(sh_row)
            u_draft = None
        else:
            assert explicit_draft_uniforms is not None
            u_draft = explicit_draft_uniforms[j]
            prop_tok = sample_categorical_f64(q_probs, u_draft)

        prop_is_eos = prop_tok in eos_set
        reused_x.append(x_j)
        reused_s1.append(s1_j)
        tokens_seq.append(prop_tok)
        draft_steps.append(
            {
                "draftOffset": j,
                "absPos": abs_pos,
                "inputTokenId": inp_tok,
                "proposedTokenId": prop_tok,
                "proposedIsEos": prop_is_eos,
                "shallowScaledLogits": sh_row,
                "proposalProbs": q_probs,
                "sampleUniform": u_draft,
            }
        )
        if prop_is_eos:
            draft_ended_with_eos = True
            break

    draft_count = len(draft_steps)
    need_bonus_slot = (not draft_ended_with_eos) and (start_len + draft_count < s_max)
    verified_positions = draft_count + 1 if need_bonus_slot else draft_count

    # Stage B: Batched Deep k = 4 Verify Phase reusing Prelude + Loop 1 for [L .. L + draftCount - 1]
    bonus_s1: torch.Tensor | None = None
    if need_bonus_slot:
        bonus_pos = start_len + draft_count
        bonus_tok = tokens_seq[draft_count]
        bx = run_prelude_stage([bonus_tok], bonus_pos, weights, rope_tables, kv, cfg)
        _, bs1 = run_recurrent_stage(bx, bx, bonus_pos, 1, weights, rope_tables, kv, cfg)
        reused_x.append(bx)
        reused_s1.append(bs1)
        bonus_s1 = bs1

    batch_x = torch.cat(reused_x, dim=0)
    batch_state = torch.cat(reused_s1, dim=0)
    for t in range(2, VERIFY_LOOPS + 1):
        _, batch_state = run_recurrent_stage(
            batch_state, batch_x, start_len, t, weights, rope_tables, kv, cfg
        )
    _, deep_logits = run_coda_and_lm_head(
        batch_state, start_len, "output_deep", weights, rope_tables, kv, cfg
    )

    # Independent PyTorch self-checks BEFORE commit/rewind:
    # 1. Reused k=4 verify logits vs completely fresh uncached k=4 forward
    verify_input_tokens = tokens_seq[:verified_positions]
    fresh_k4_all = run_fresh_full_forward(
        committed_prefix + verify_input_tokens, VERIFY_LOOPS, weights, rope_tables, cfg
    )
    reused_vs_fresh_k4 = float(
        (deep_logits - fresh_k4_all[start_len : start_len + verified_positions]).abs().max().item()
    )
    if reused_vs_fresh_k4 > 1e-5:
        raise RuntimeError(f"Reused k=4 verify diverged from fresh k=4: {reused_vs_fresh_k4}")

    # 2. Contract A shallow draft logits vs completely fresh standalone k=1 Dory
    draft_input_tokens = tokens_seq[:draft_count]
    fresh_k1_all = run_fresh_full_forward(
        committed_prefix + draft_input_tokens, DRAFT_LOOPS, weights, rope_tables, cfg
    )
    shallow_draft_tensor = torch.tensor(
        [ds["shallowScaledLogits"] for ds in draft_steps], dtype=torch.float32
    )
    shallow_vs_standalone_k1 = float(
        (shallow_draft_tensor - fresh_k1_all[start_len : start_len + draft_count]).abs().max().item()
    )
    if shallow_vs_standalone_k1 > 1e-5:
        raise RuntimeError(
            f"Contract A shallow draft diverged from standalone k=1: {shallow_vs_standalone_k1}"
        )

    # Stage C: Rejection sampling
    verify_positions_out: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_at = -1
    deep_target_probs_by_pos: list[list[float]] = []

    for j in range(draft_count):
        d_row = to_f32_list(deep_logits[j], f"verify[{j}].deepLogits")
        p_probs = logits_to_sampling_probs_f64(d_row, temperature, top_k, top_p)
        deep_target_probs_by_pos.append(p_probs)

        accepted_flag: bool | None = None
        reason_str: str | None = None
        if rejected_at == -1:
            ds = draft_steps[j]
            prop_tok = ds["proposedTokenId"]
            q_probs = ds["proposalProbs"]
            u_acc: float | None = None
            if not is_greedy and p_probs[prop_tok] < q_probs[prop_tok]:
                assert explicit_accept_uniforms is not None
                u_acc = explicit_accept_uniforms[j]
            accepted_flag, reason_str = evaluate_rejection_f64(
                prop_tok, p_probs, q_probs, u_acc, is_greedy
            )
            if accepted_flag:
                accepted_count += 1
            else:
                rejected_at = j

        verify_positions_out.append(
            {
                "verifyOffset": j,
                "absPos": start_len + j,
                "inputTokenId": tokens_seq[j],
                "reusedPreludeAndLoop1": True,
                "deepScaledLogits": d_row,
                "accepted": accepted_flag,
                "reason": reason_str,
            }
        )

    bonus_target_probs: list[float] | None = None
    if need_bonus_slot:
        b_row = to_f32_list(deep_logits[draft_count], "verify[bonus].deepLogits")
        bonus_target_probs = logits_to_sampling_probs_f64(b_row, temperature, top_k, top_p)
        verify_positions_out.append(
            {
                "verifyOffset": draft_count,
                "absPos": start_len + draft_count,
                "inputTokenId": tokens_seq[draft_count],
                "reusedPreludeAndLoop1": False,
                "deepScaledLogits": b_row,
                "accepted": None,
                "reason": None,
            }
        )

    # Stage D: Commit / Rewind across all 104 banks
    max_deep_written = start_len + verified_positions
    shallow_catchup = False
    halt_reason: str | None = None
    correction_or_bonus_tok: int | None = None
    residual_z: float | None = None

    if rejected_at != -1:
        a = rejected_at
        outcome = "zero_accept_reject" if a == 0 else "partial_accept_reject"
        end_len = start_len + a + 1
        if is_greedy:
            y_star = argmax_f64(verify_positions_out[a]["deepScaledLogits"])
        else:
            res_probs, residual_z = compute_residual_f64(
                deep_target_probs_by_pos[a], draft_steps[a]["proposalProbs"]
            )
            assert explicit_correction_or_bonus_uniform is not None
            y_star = sample_categorical_f64(res_probs, explicit_correction_or_bonus_uniform)

        correction_or_bonus_tok = y_star
        next_pending: int | None = y_star
        emitted = [draft_steps[i]["proposedTokenId"] for i in range(a)] + [y_star]
        if y_star in eos_set:
            halt_reason = "eos"
        elif end_len >= s_max:
            halt_reason = "capacity_exhausted"
        kv.rewind_all(end_len)
    elif draft_ended_with_eos:
        outcome = "full_accept_eos"
        end_len = start_len + draft_count
        next_pending = None
        halt_reason = "eos"
        emitted = [draft_steps[i]["proposedTokenId"] for i in range(draft_count)]
        kv.rewind_all(end_len)
    else:
        outcome = "full_accept_bonus"
        end_len = start_len + draft_count + 1
        assert bonus_target_probs is not None and bonus_s1 is not None
        if is_greedy:
            y_bonus = argmax_f64(verify_positions_out[draft_count]["deepScaledLogits"])
        else:
            assert explicit_correction_or_bonus_uniform is not None
            y_bonus = sample_categorical_f64(
                bonus_target_probs, explicit_correction_or_bonus_uniform
            )
        correction_or_bonus_tok = y_bonus
        next_pending = y_bonus
        emitted = [draft_steps[i]["proposedTokenId"] for i in range(draft_count)] + [y_bonus]

        # 1-token Shallow Coda catch-up on cached s_1^{(L + draftCount)}
        run_coda_and_lm_head(
            bonus_s1, start_len + draft_count, "output_shallow", weights, rope_tables, kv, cfg
        )
        shallow_catchup = True
        if y_bonus in eos_set:
            halt_reason = "eos"
        elif end_len >= s_max:
            halt_reason = "capacity_exhausted"
        kv.rewind_all(end_len)

    kv.assert_all_synced(end_len)
    rolled_back = max_deep_written - end_len
    committed_delta = tokens_seq[: end_len - start_len]

    # Strip large binary64 proposalProbs from JSON (only keep compact Float32 logits + scalars)
    draft_steps_compact = [
        {
            "draftOffset": ds["draftOffset"],
            "absPos": ds["absPos"],
            "inputTokenId": ds["inputTokenId"],
            "proposedTokenId": ds["proposedTokenId"],
            "proposedIsEos": ds["proposedIsEos"],
            "shallowScaledLogits": ds["shallowScaledLogits"],
        }
        for ds in draft_steps
    ]

    return {
        "startCommittedLen": start_len,
        "endCommittedLen": end_len,
        "inputPendingToken": pending_token,
        "nextPendingToken": next_pending,
        "gammaEffective": gamma_eff,
        "draftCount": draft_count,
        "draftEndedWithEos": draft_ended_with_eos,
        "acceptedDraftCount": accepted_count,
        "emittedTokens": emitted,
        "outcome": outcome,
        "haltReason": halt_reason,
        "reusedActivationPositions": draft_count,
        "verifiedPositions": verified_positions,
        "shallowCodaCatchupExecuted": shallow_catchup,
        "rolledBackPositions": rolled_back,
        "residualMassZ": residual_z,
        "correctionOrBonusToken": correction_or_bonus_tok,
        "draftSteps": draft_steps_compact,
        "verifyPositions": verify_positions_out,
        "reusedVsFreshK4MaxAbs": reused_vs_fresh_k4,
        "shallowDraftVsStandaloneK1MaxAbs": shallow_vs_standalone_k1,
        "committedDeltaTokens": committed_delta,
    }


def plan_deterministic_cycle_variates(
    kv_committed_prefix: list[int],
    pending_token: int,
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    *,
    target_outcome: str,
    target_reject_at: int | None = None,
    eos_at: int | None = None,
    gamma: int = DEFAULT_GAMMA,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """
    Constructs boundary-safe explicit uniform variates (choosing ascending-CDF bin midpoints
    with comfortable margin from bin edges) so independent PyTorch and TypeScript engines
    execute the exact same token path and exercise every Contract A transition cleanly.
    """
    eos_set = set(int(x) for x in cfg["eosTokenIds"])
    start_len = len(kv_committed_prefix)
    s_max = int(cfg["maxPositionEmbeddings"])
    gamma_eff = min(gamma, s_max - 1 - start_len)

    tokens_seq = [pending_token]
    draft_uniforms: list[float] = []
    accept_uniforms: list[float] = []
    q_list: list[list[float]] = []

    for j in range(gamma_eff):
        # Standalone k=1 matches Contract A shallow draft identically!
        k1_logits = run_fresh_full_forward(
            kv_committed_prefix + tokens_seq, DRAFT_LOOPS, weights, rope_tables, cfg
        )
        q_j = logits_to_sampling_probs_f64(
            to_f32_list(k1_logits[-1], "plan.q"), temperature=temperature
        )
        q_list.append(q_j)

        if eos_at is not None and j == eos_at:
            chosen_tok = 2  # '</s>' EOS token
        elif target_reject_at is not None and j == target_reject_at:
            # Pick non-EOS token maximizing (q[v] - p[v]) so p[d] < q[d] with huge margin!
            k4_logits = run_fresh_full_forward(
                kv_committed_prefix + tokens_seq, VERIFY_LOOPS, weights, rope_tables, cfg
            )
            p_j = logits_to_sampling_probs_f64(
                to_f32_list(k4_logits[-1], "plan.p"), temperature=temperature
            )
            best_tok = -1
            best_gap = -1.0
            for v in range(len(q_j)):
                if v in eos_set:
                    continue
                gap = q_j[v] - p_j[v]
                if gap > best_gap:
                    best_gap = gap
                    best_tok = v
            if best_tok < 0 or best_gap <= 1e-4:
                raise RuntimeError(f"Could not find strong reject token at j={j}: gap={best_gap}")
            chosen_tok = best_tok
        else:
            # Pick highest-probability non-EOS token so CDF bin width is wide & boundary-safe
            best_tok = max((v for v in range(len(q_j)) if v not in eos_set), key=lambda v: q_j[v])
            chosen_tok = best_tok

        u_mid, _ = cdf_bin_midpoint(q_j, chosen_tok)
        draft_uniforms.append(u_mid)
        tokens_seq.append(chosen_tok)

        if target_reject_at is not None and j == target_reject_at:
            accept_uniforms.append(0.999999)
        else:
            accept_uniforms.append(0.0)

        if eos_at is not None and j == eos_at:
            break

    # Compute boundary-safe correctionOrBonusUniform (choosing non-EOS token with highest prob!)
    corr_or_bonus_u: float | None = None
    if target_outcome in ("zero_accept_reject", "partial_accept_reject"):
        assert target_reject_at is not None
        k4_all = run_fresh_full_forward(
            kv_committed_prefix + tokens_seq, VERIFY_LOOPS, weights, rope_tables, cfg
        )
        p_rej = logits_to_sampling_probs_f64(
            to_f32_list(k4_all[start_len + target_reject_at], "plan.p_rej"),
            temperature=temperature,
        )
        res_probs, _ = compute_residual_f64(p_rej, q_list[target_reject_at])
        best_res_tok = max(
            (v for v in range(len(res_probs)) if v not in eos_set and res_probs[v] > 0.0),
            key=lambda v: res_probs[v],
        )
        corr_or_bonus_u, _ = cdf_bin_midpoint(res_probs, best_res_tok)
    elif target_outcome == "full_accept_bonus":
        k4_all = run_fresh_full_forward(
            kv_committed_prefix + tokens_seq, VERIFY_LOOPS, weights, rope_tables, cfg
        )
        p_bonus = logits_to_sampling_probs_f64(
            to_f32_list(k4_all[start_len + len(draft_uniforms)], "plan.p_bonus"),
            temperature=temperature,
        )
        best_bonus_tok = max(
            (v for v in range(len(p_bonus)) if v not in eos_set),
            key=lambda v: p_bonus[v],
        )
        corr_or_bonus_u, _ = cdf_bin_midpoint(p_bonus, best_bonus_tok)
    elif target_outcome == "full_accept_eos":
        corr_or_bonus_u = None

    return {
        "draftSampleUniforms": draft_uniforms,
        "acceptUniforms": accept_uniforms,
        "correctionOrBonusUniform": corr_or_bonus_u,
    }


# ============================================================================
# FIXTURE BUILDERS (LAYERS 1–6)
# ============================================================================


def build_layer1_and_layer2_fixtures() -> dict[str, Any]:
    """Layers 1 & 2: Binary64 algebraic identities & exact V=4 1-step / 2-step enumeration."""
    cases_1step = [
        {
            "name": "v4_general_overlap",
            "targetProbs": [0.45, 0.30, 0.15, 0.10],
            "proposalProbs": [0.20, 0.50, 0.25, 0.05],
        },
        {
            "name": "v4_partial_zero_support",
            "targetProbs": [0.35, 0.25, 0.25, 0.15],
            "proposalProbs": [0.60, 0.40, 0.0, 0.0],
        },
        {
            "name": "v4_identical_p_eq_q",
            "targetProbs": [0.10, 0.20, 0.30, 0.40],
            "proposalProbs": [0.10, 0.20, 0.30, 0.40],
        },
        {
            "name": "v4_disjoint_support",
            "targetProbs": [0.0, 0.0, 0.65, 0.35],
            "proposalProbs": [0.70, 0.30, 0.0, 0.0],
        },
        {
            "name": "v4_near_identical_delta_4e13",
            "targetProbs": [0.5 + 4e-13, 0.5 - 4e-13, 0.0, 0.0],
            "proposalProbs": [0.5, 0.5, 0.0, 0.0],
        },
        {
            "name": "v4_sub_1e38_tail",
            "targetProbs": [5e-41, 0.6, 0.4 - 5e-41, 0.0],
            "proposalProbs": [1e-40, 0.3, 0.7 - 1e-40, 0.0],
        },
    ]

    one_step_results = []
    for c in cases_1step:
        ev = enumerate_1step_exact_f64(c["targetProbs"], c["proposalProbs"])
        if ev["maxAbsError"] > 1e-15:
            raise RuntimeError(
                f"Layer 1/2 1-step exact enum error > 1e-15 on {c['name']}: {ev['maxAbsError']}"
            )
        one_step_results.append({**c, **ev})

    # Exact V = 4 2-step joint speculative marginal enumeration
    p1 = [0.40, 0.30, 0.20, 0.10]
    q1 = [0.15, 0.50, 0.25, 0.10]
    p2_by_v1 = [
        [0.50, 0.20, 0.20, 0.10],
        [0.10, 0.60, 0.20, 0.10],
        [0.25, 0.25, 0.35, 0.15],
        [0.05, 0.15, 0.30, 0.50],
    ]
    q2_by_v1 = [
        [0.20, 0.50, 0.20, 0.10],
        [0.40, 0.20, 0.30, 0.10],
        [0.10, 0.40, 0.10, 0.40],
        [0.35, 0.35, 0.15, 0.15],
    ]
    two_step = enumerate_2step_exact_f64(p1, q1, p2_by_v1, q2_by_v1)
    if two_step["maxAbsError"] > 1e-15:
        raise RuntimeError(f"Layer 2 V=4 2-step exact enum error > 1e-15: {two_step['maxAbsError']}")

    return {
        "oneStepCases": one_step_results,
        "twoStepV4": {
            "targetP1": p1,
            "proposalQ1": q1,
            "targetP2ByToken1": p2_by_v1,
            "proposalQ2ByToken1": q2_by_v1,
            **two_step,
        },
    }


def build_layer3_and_layer4_fixtures() -> dict[str, Any]:
    """Layers 3 & 4: Explicit-variate boundaries, adversarial edge cases, & filtered sampling."""
    p_b = [0.15, 0.45, 0.25, 0.15]
    q_b = [0.50, 0.20, 0.20, 0.10]
    # For d = 0: p[0] = 0.15, q[0] = 0.50 -> exact ratio p/q = 0.30
    # Residual = [0, 0.25, 0.05, 0.05] / 0.35 = [0, 5/7, 1/7, 1/7]
    res_b, z_b = compute_residual_f64(p_b, q_b)

    boundary_checks = []
    for label, u_draft, u_acc, u_res in [
        ("accept_r_zero", 0.25, 0.0, 0.5),
        ("accept_just_below_ratio", 0.25, 0.3 - 1e-7, 0.5),
        ("reject_exact_ratio_to_res_bin1", 0.25, 0.3, 0.1),
        ("reject_just_above_ratio_to_res_bin2", 0.25, 0.3 + 1e-7, 5.0 / 7.0 + 0.05),
        ("reject_near_one_to_res_bin3", 0.25, 1.0 - 1e-12, 6.0 / 7.0 + 0.05),
        ("unconditional_p_ge_q_d1", 0.60, 1.0 - 1e-12, 0.5),
    ]:
        d_tok = sample_categorical_f64(q_b, u_draft)
        acc, reason = evaluate_rejection_f64(
            d_tok, p_b, q_b, None if p_b[d_tok] >= q_b[d_tok] else u_acc, False
        )
        emitted = d_tok if acc else sample_categorical_f64(res_b, u_res)
        boundary_checks.append(
            {
                "label": label,
                "draftUniform": u_draft,
                "acceptUniform": u_acc,
                "residualUniform": u_res,
                "expectedDraftToken": d_tok,
                "expectedAccepted": acc,
                "expectedReason": reason,
                "expectedEmittedToken": emitted,
            }
        )

    # Filtered sampling consistency fixture (temperature + topK + topP)
    sample_logits = [2.4, 1.9, 1.85, 0.7, 0.1, -0.5, -1.2, -2.0]
    filtered_configs = [
        {"label": "greedy_t0", "temperature": 0.0, "topK": 0, "topP": 1.0},
        {"label": "temp_08_only", "temperature": 0.8, "topK": 0, "topP": 1.0},
        {"label": "temp_10_topK_4", "temperature": 1.0, "topK": 4, "topP": 1.0},
        {"label": "temp_09_topP_075", "temperature": 0.9, "topK": 0, "topP": 0.75},
        {"label": "temp_08_topK_5_topP_085", "temperature": 0.8, "topK": 5, "topP": 0.85},
    ]
    filtered_cases = [
        {
            **fc,
            "logits": sample_logits,
            "probs": logits_to_sampling_probs_f64(
                sample_logits, fc["temperature"], fc["topK"], fc["topP"]
            ),
        }
        for fc in filtered_configs
    ]

    # Adversarial sub-1e-38 & Delta=4e-13 edge suite
    sub_q = [1e-40, 1.0 - 1e-40]
    sub_p = [5e-41, 1.0 - 5e-41]
    buggy_clamp_accept_prob = sub_p[0] / max(sub_q[0], 1e-38)

    delta_p = [0.5 + 4e-13, 0.5 - 4e-13]
    delta_q = [0.5, 0.5]
    delta_res, delta_z = compute_residual_f64(delta_p, delta_q)

    return {
        "boundaryTargetProbs": p_b,
        "boundaryProposalProbs": q_b,
        "boundaryResidualProbs": res_b,
        "boundaryResidualMassZ": float(z_b),
        "boundaryChecks": boundary_checks,
        "filteredSamplingCases": filtered_cases,
        "edgeCases": {
            "sub1e38": {
                "targetProbs": sub_p,
                "proposalProbs": sub_q,
                "exactAcceptProbAtD0": 0.5,
                "buggyClamp1e38AcceptProbAtD0": float(buggy_clamp_accept_prob),
            },
            "delta4e13": {
                "targetProbs": delta_p,
                "proposalProbs": delta_q,
                "exactResidualProbs": delta_res,
                "exactResidualMassZ": float(delta_z),
                "buggyFallback1e12ResidualProbs": delta_p,
            },
            "concentrated": {
                "targetProbs": [1.0 - 1e-9, 5e-10, 3e-10, 2e-10],
                "proposalProbs": [0.7, 0.1, 0.1, 0.1],
            },
            "oneHotSame": {
                "targetProbs": [1.0, 0.0, 0.0, 0.0],
                "proposalProbs": [1.0, 0.0, 0.0, 0.0],
            },
            "oneHotDifferent": {
                "targetProbs": [0.0, 1.0, 0.0, 0.0],
                "proposalProbs": [1.0, 0.0, 0.0, 0.0],
            },
        },
    }


def build_layer5_contract_a_fixtures(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Layer 5: Full 72L Contract A (104 banks) greedy, stochastic, EOS & final-slot fixtures."""
    prompt8 = [1, 10, 4, 18, 40, 25, 26, 57]

    # --- 5A: Greedy (T = 0) 4 consecutive cycles starting at L = 8 (crossing W = 8 eviction) ---
    greedy_prefill = prefill_contract_a(prompt8, weights, rope_tables, cfg, temperature=0.0)
    kv_g: PyTorchContractAKvCache = greedy_prefill["kv"]
    prefix_g = list(prompt8)
    emitted_g = list(prompt8) + [greedy_prefill["pendingToken"]]
    pending_g: int | None = greedy_prefill["pendingToken"]
    greedy_cycles = []
    for _ in range(4):
        if pending_g is None or kv_g.committed_len >= int(cfg["maxPositionEmbeddings"]):
            break
        cyc = run_contract_a_cycle(
            kv_g, prefix_g, pending_g, weights, rope_tables, cfg, gamma=DEFAULT_GAMMA, temperature=0.0
        )
        prefix_g.extend(cyc["committedDeltaTokens"])
        emitted_g.extend(cyc["emittedTokens"])
        pending_g = cyc["nextPendingToken"]
        greedy_cycles.append(cyc)
        if cyc["haltReason"] is not None:
            break

    # Verify greedy === standard k = 4 autoregressive decode token-for-token
    ar_tokens = list(prompt8)
    target_total = len(emitted_g)
    while len(ar_tokens) < target_total:
        logits_ar = run_fresh_full_forward(ar_tokens, VERIFY_LOOPS, weights, rope_tables, cfg)
        next_tok = argmax_f64(to_f32_list(logits_ar[-1], "ar.logits"))
        ar_tokens.append(next_tok)
    if ar_tokens != emitted_g:
        raise RuntimeError(f"Greedy Contract A diverged from k=4 AR: {emitted_g} != {ar_tokens}")

    # --- 5B: Multi-cycle Stochastic Contract A exercising partial_accept_reject -> full_accept_bonus
    #         (+ 1-token Shallow Coda catch-up) -> zero_accept_reject -> partial_accept_reject ---
    init_prefill_probe = run_fresh_full_forward(prompt8, VERIFY_LOOPS, weights, rope_tables, cfg)
    init_p = logits_to_sampling_probs_f64(
        to_f32_list(init_prefill_probe[-1], "stoch.initP"), temperature=1.0
    )
    eos_set = set(int(x) for x in cfg["eosTokenIds"])
    init_tok = max((v for v in range(len(init_p)) if v not in eos_set), key=lambda v: init_p[v])
    init_u, _ = cdf_bin_midpoint(init_p, init_tok)

    stoch_prefill = prefill_contract_a(
        prompt8, weights, rope_tables, cfg, temperature=1.0, initial_uniform=init_u
    )
    kv_s: PyTorchContractAKvCache = stoch_prefill["kv"]
    prefix_s = list(prompt8)
    pending_s: int | None = stoch_prefill["pendingToken"]

    cycle_plans = [
        ("partial_accept_reject", 2),   # Cycle 0: a=2, rollback across S > 8, 0 extra Shallow Coda
        ("full_accept_bonus", None),    # Cycle 1: a=4, bonus + 1-token Shallow Coda catch-up
        ("zero_accept_reject", 0),      # Cycle 2: a=0, post-catchup verify Shallow Coda == k=1!
        ("partial_accept_reject", 1),   # Cycle 3: a=1, post-zero-reject verify SWA ring intact!
    ]
    stoch_cycles = []
    for expected_outcome, rej_at in cycle_plans:
        assert pending_s is not None
        variates = plan_deterministic_cycle_variates(
            prefix_s,
            pending_s,
            weights,
            rope_tables,
            cfg,
            target_outcome=expected_outcome,
            target_reject_at=rej_at,
            gamma=DEFAULT_GAMMA,
            temperature=1.0,
        )
        cyc = run_contract_a_cycle(
            kv_s,
            prefix_s,
            pending_s,
            weights,
            rope_tables,
            cfg,
            gamma=DEFAULT_GAMMA,
            temperature=1.0,
            explicit_draft_uniforms=variates["draftSampleUniforms"],
            explicit_accept_uniforms=variates["acceptUniforms"],
            explicit_correction_or_bonus_uniform=variates["correctionOrBonusUniform"],
        )
        if cyc["outcome"] != expected_outcome:
            raise RuntimeError(
                f"Expected stochastic outcome {expected_outcome}, got {cyc['outcome']}"
            )
        prefix_s.extend(cyc["committedDeltaTokens"])
        pending_s = cyc["nextPendingToken"]
        stoch_cycles.append({"explicitVariates": variates, "result": cyc})

    # --- 5C: Draft EOS Early Stop (full_accept_eos, no bonus token) ---
    eos_prefill = prefill_contract_a(
        prompt8, weights, rope_tables, cfg, temperature=1.0, initial_uniform=init_u
    )
    kv_e: PyTorchContractAKvCache = eos_prefill["kv"]
    eos_variates = plan_deterministic_cycle_variates(
        list(prompt8),
        eos_prefill["pendingToken"],
        weights,
        rope_tables,
        cfg,
        target_outcome="full_accept_eos",
        eos_at=1,
        gamma=DEFAULT_GAMMA,
        temperature=1.0,
    )
    eos_cycle = run_contract_a_cycle(
        kv_e,
        list(prompt8),
        eos_prefill["pendingToken"],
        weights,
        rope_tables,
        cfg,
        gamma=DEFAULT_GAMMA,
        temperature=1.0,
        explicit_draft_uniforms=eos_variates["draftSampleUniforms"],
        explicit_accept_uniforms=eos_variates["acceptUniforms"],
        explicit_correction_or_bonus_uniform=None,
    )
    if eos_cycle["outcome"] != "full_accept_eos" or not eos_cycle["draftEndedWithEos"]:
        raise RuntimeError(f"Expected full_accept_eos, got {eos_cycle['outcome']}")

    # --- 5D: Final Remaining Slot (L = S_max - 1 = 31, gammaEffective = 0) ---
    prompt31 = [1, 10, 4, 18] + [20 + (i % 35) for i in range(27)]
    final_prefill = prefill_contract_a(prompt31, weights, rope_tables, cfg, temperature=0.0)
    kv_f: PyTorchContractAKvCache = final_prefill["kv"]
    final_cycle = run_contract_a_cycle(
        kv_f,
        list(prompt31),
        final_prefill["pendingToken"],
        weights,
        rope_tables,
        cfg,
        gamma=DEFAULT_GAMMA,
        temperature=0.0,
    )
    if final_cycle["outcome"] != "final_slot_deep_decode" or final_cycle["gammaEffective"] != 0:
        raise RuntimeError(f"Expected final_slot_deep_decode, got {final_cycle['outcome']}")

    return {
        "prompt8": prompt8,
        "greedyMultiCycle": {
            "prefill": {
                "committedLen": greedy_prefill["committedLen"],
                "pendingToken": greedy_prefill["pendingToken"],
                "lastDeepScaledLogits": greedy_prefill["lastDeepScaledLogits"],
                "lastShallowScaledLogits": greedy_prefill["lastShallowScaledLogits"],
            },
            "cycles": greedy_cycles,
            "emittedSequence": emitted_g,
            "standaloneK4ArSequence": ar_tokens,
        },
        "stochasticMultiCycle": {
            "initialSampleUniform": init_u,
            "prefill": {
                "committedLen": stoch_prefill["committedLen"],
                "pendingToken": stoch_prefill["pendingToken"],
                "lastDeepScaledLogits": stoch_prefill["lastDeepScaledLogits"],
                "lastShallowScaledLogits": stoch_prefill["lastShallowScaledLogits"],
            },
            "cycles": stoch_cycles,
        },
        "draftEosEarlyStop": {
            "initialSampleUniform": init_u,
            "explicitVariates": eos_variates,
            "result": eos_cycle,
        },
        "finalSlotDeepDecode": {
            "prompt31": prompt31,
            "prefillPendingToken": final_prefill["pendingToken"],
            "result": final_cycle,
        },
    }


def build_layer6_chi2_and_negative_controls(
    weights: dict[str, torch.Tensor],
    rope_tables: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Layer 6: V = 64 target/proposal pairs for Pearson chi^2 (N = 20,000, df = 63, alpha = 0.001,
    crit = 103.51) PLUS analytical oracle metrics for the 2 mandatory failing negative controls.
    """
    prompt8 = [1, 10, 4, 18, 40, 25, 26, 57]
    k4_logits = to_f32_list(
        run_fresh_full_forward(prompt8, VERIFY_LOOPS, weights, rope_tables, cfg)[-1],
        "layer6.k4Logits",
    )
    k1_logits = to_f32_list(
        run_fresh_full_forward(prompt8, DRAFT_LOOPS, weights, rope_tables, cfg)[-1],
        "layer6.k1Logits",
    )

    n_draws = 20_000
    # Model-derived V = 64 pair at T = 1.0 (verify min expected count >= 10 across all 64 bins!)
    model_p = logits_to_sampling_probs_f64(k4_logits, temperature=1.0)
    model_q = logits_to_sampling_probs_f64(k1_logits, temperature=1.0)
    model_min_expected = n_draws * min(model_p)
    if model_min_expected < 10.0:
        raise RuntimeError(
            f"Model V=64 minExpectedCount={model_min_expected:.2f} < 10 at N={n_draws}"
        )
    model_enum = enumerate_1step_exact_f64(model_p, model_q)
    model_neg1 = enumerate_buggy_resample_from_p_f64(model_p, model_q)

    # High-divergence V = 64 pair (minExpectedCount >= 100 across all 64 bins, TVD(p, q) ~ 0.42)
    # guaranteeing massive Pearson chi^2 >> 103.51 under Negative Control 1 (resampling from p).
    v_size = int(cfg["vocabSize"])
    high_div_p = [0.0] * v_size
    high_div_q = [0.0] * v_size
    for v in range(v_size):
        # First half higher in p, second half higher in q; every bin >= 0.006 -> E[count] >= 120!
        high_div_p[v] = (3.0 + 0.5 * math.sin(v * 0.37)) if v < v_size // 2 else (0.7 + 0.1 * math.cos(v * 0.29))
        high_div_q[v] = (0.7 + 0.1 * math.sin(v * 0.41)) if v < v_size // 2 else (3.0 + 0.5 * math.cos(v * 0.23))
    sp = sum(high_div_p)
    sq = sum(high_div_q)
    high_div_p = [x / sp for x in high_div_p]
    high_div_q = [x / sq for x in high_div_q]

    high_div_min_expected = n_draws * min(high_div_p)
    if high_div_min_expected < 50.0:
        raise RuntimeError(f"High-divergence V=64 minExpectedCount={high_div_min_expected} < 50")

    high_div_enum = enumerate_1step_exact_f64(high_div_p, high_div_q)
    high_div_neg1 = enumerate_buggy_resample_from_p_f64(high_div_p, high_div_q)
    # Expected chi^2 non-centrality for Negative Control 1 at N = 20,000
    neg1_expected_noncentrality = n_draws * sum(
        ((high_div_neg1["marginalProbs"][v] - high_div_p[v]) ** 2) / high_div_p[v]
        for v in range(v_size)
    )
    if neg1_expected_noncentrality < 500.0:
        raise RuntimeError(
            f"Negative Control 1 signal too weak: noncentrality={neg1_expected_noncentrality}"
        )

    return {
        "sampleCount": n_draws,
        "alpha": 0.001,
        "degreesOfFreedom": v_size - 1,
        "criticalValueDf63Alpha001": PEARSON_CHI2_DF63_ALPHA_001,
        "modelPair": {
            "prompt8": prompt8,
            "k4ScaledLogits": k4_logits,
            "k1ScaledLogits": k1_logits,
            "targetProbs": model_p,
            "proposalProbs": model_q,
            "minExpectedCount": float(model_min_expected),
            "exactAcceptanceRate": model_enum["acceptanceRate"],
            "exactMaxAbsError": model_enum["maxAbsError"],
            "negControl1ResampleFromPTvd": model_neg1["tvdVsTarget"],
        },
        "highDivergencePair": {
            "targetProbs": high_div_p,
            "proposalProbs": high_div_q,
            "minExpectedCount": float(high_div_min_expected),
            "exactAcceptanceRate": high_div_enum["acceptanceRate"],
            "exactMaxAbsError": high_div_enum["maxAbsError"],
            "negControl1ResampleFromPMarginal": high_div_neg1["marginalProbs"],
            "negControl1ResampleFromPTvd": high_div_neg1["tvdVsTarget"],
            "negControl1ResampleFromPMaxAbsErr": high_div_neg1["maxAbsError"],
            "negControl1ExpectedChi2Noncentrality": float(neg1_expected_noncentrality),
        },
        "negControl2BuggyClamp1e38": {
            "qDraft": 1e-40,
            "pDraft": 5e-41,
            "exactAcceptRate": 0.5,
            "buggyClampAcceptRate": 0.005,
            "criticalValueDf1Alpha001": 10.828,
        },
    }


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean-room PyTorch 2.15.0 Contract A self-speculation parity fixtures for Dory1.0-4B."
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

    layer1_2 = build_layer1_and_layer2_fixtures()
    layer3_4 = build_layer3_and_layer4_fixtures()
    layer5 = build_layer5_contract_a_fixtures(weights, rope_tables, cfg)
    layer6 = build_layer6_chi2_and_negative_controls(weights, rope_tables, cfg)

    fixture = {
        "schemaVersion": "dory2-spec-pytorch-parity-v1",
        "format": manifest["format"],
        "contractBadge": CONTRACT_A_BADGE,
        "contractABankCount": CONTRACT_A_BANK_COUNT,
        "modelWeightId": manifest["modelWeightId"],
        "weightsSha256": manifest["weightsSha256"],
        "backboneSeed": manifest["backboneSeed"],
        "tensorCount": manifest["tensorCount"],
        "totalFloats": manifest["totalFloats"],
        "totalBytes": manifest["totalBytes"],
        "torchVersion": torch.__version__,
        "pythonVersion": platform.python_version(),
        "layers1And2ExactEnum": layer1_2,
        "layers3And4BoundariesAndEdges": layer3_4,
        "layer5ContractAExecution": layer5,
        "layer6Chi2AndNegativeControls": layer6,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(fixture, allow_nan=False, separators=(",", ":")) + "\n"
    args.out.write_text(serialized, encoding="utf-8")

    g_cyc = layer5["greedyMultiCycle"]["cycles"]
    s_cyc = layer5["stochasticMultiCycle"]["cycles"]
    print(
        f"Generated PyTorch {torch.__version__} Dory Contract A spec fixture -> {args.out} "
        f"({len(serialized):,} B, modelWeightId={manifest['modelWeightId'][:16]}..., "
        f"greedyCycles={len(g_cyc)}, stochCycles={len(s_cyc)}, "
        f"v4_2step_err={layer1_2['twoStepV4']['maxAbsError']:.2e}, "
        f"neg1_noncentrality={layer6['highDivergencePair']['negControl1ExpectedChi2Noncentrality']:.1f})"
    )


if __name__ == "__main__":
    main()
