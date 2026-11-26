#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.15.0",
# ]
# ///
"""
Independent Clean-Room Python 3.12 / PyTorch 2.15.0 Reference Fixture Generator
for Nemotron-Labs-Diffusion-Uni Progressive Self-Refining Diffusion, Dual-CFG
Image Editing & Read-Only OPD Parity (UNI3 Slice 3B / Gate G3).

Development/reference verification script ONLY — NOT a browser runtime dependency.
Pinned environment: Python 3.12 + PyTorch 2.15.0 (enforced via PEP 723 metadata).

Reproduce `tests/fixtures/pytorch-uni-diffusion.json`:
  node scripts/reference/export_uni_multimodal_params.mjs /tmp/nemotron-viz-uni-mm-ref
  uv run scripts/reference/generate_uni_diffusion_pytorch_fixtures.py \
    --params-dir /tmp/nemotron-viz-uni-mm-ref \
    --out tests/fixtures/pytorch-uni-diffusion.json

Clean-Room Mathematical Reference (zero NVIDIA/Adobe source code or checkpoints):
  Cryptographically verifies all 766 exported tensors (`weightsSha256` + every
  per-tensor SHA-256 digest) and emits compact JSON (`separators=(",", ":")`)
  covering ALL Gate G3 equations:
    1. BalancedSemanticTree (`recursive_centered_pca_rank_median_v1`, 0-ULP
       `leafToToken`/`tokenToLeaf`, 255 monotone `heapRadii`, `pca3d`), `U26`
       `D=17` vs `D=8` floor-rounding mismatch across `K in {1, 4, 8, 16}`,
       `semanticTreeLevelsForAges` (`N in {1, 2, 4, 8}` incl. `N=1` binary MDM),
       `sampleRepresentatives`, `buildSemanticTreeEmbeddings` (`U10`),
       `agesAtTime` (`Eq. 16`), `shift=5.0` quotas + `N-1` flush rounds, and
       `stratifiedRandom(4, 42)` via exact `xoshiro128**`.
    2. Full T2I trajectories (`steps=8, N=4` -> 11 rounds + 1 prefill = 12 passes,
       `B_cfg=2, w=3.0, T=0.8` with shared explicit variates `U23`, proving 0-ULP
       `elastic_pruned` vs `hf_faithful` Stage-2 equivalence in PyTorch; PLUS
       `N=1` binary MDM limit & `guidanceScale=0` `U30` `B_cfg=1` short-circuit).
    3. Full 3-branch Dual-CFG Image Editing trajectory on non-square
       `harbor_panorama_16x8 -> 16x16` (`B_cfg=3`, `lockSourcePaddingVq=True`,
       `cropToSource=True`, default `legacy` `w_txt=1.6, w_img=2.0`, PLUS 1-step
       Dual-CFG checks for `independent` & `ip2p`, proving 0-ULP `elastic_pruned`
       vs `hf_faithful` and 0-ULP locked padding cells across every round).
    4. Read-only OPD reference parity: `129`-bucket `tail_kl` (`reverseKl`,
       `forwardKl`, `arOpdLoss`, `selfConsistencyLoss`), `matched_sigmas(0.9, 1024, 40)`,
       and flow `x^r` + `exact_ce + 0.2 * grouped_ce`.
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
DEFAULT_OUT_PATH = Path("tests/fixtures/pytorch-uni-diffusion.json")

EXPECTED_FULL_TENSORS = 766
EXPECTED_FULL_FLOATS = 191_235
EXPECTED_FULL_BYTES = 764_940

UNI_BOS_ID = 1
UNI_IM_START_ID = 10
UNI_PAD_ID = 11
UNI_UND_PAD_ID = 23
UNI_NEWLINE_ID = 24

UNI_VQ_CODEBOOK_SIZE = 256
UNI_GEN_ROW_MARKER_ID = 256
UNI_GEN_IMAGE_MASK_ID = 257
UNI_GEN_IMAGE_END_ID = 258
UNI_GEN_FIXED_PREFIX_IDS = [259, 260, 261, 262]

MASK32 = 0xFFFFFFFF
OPEN_UNIT_MIN = 1e-15
OPEN_UNIT_MAX = 1.0 - 1e-15


# ============================================================================
# EXACT XOSHIRO128** & SPLITMIX32 (BIT-EXACT MATCH TO `src/uni/prng.ts`)
# ============================================================================

def rotl32(x: int, k: int) -> int:
    x &= MASK32
    return ((x << k) | (x >> (32 - k))) & MASK32


def splitmix32(seed: int) -> int:
    z = ((seed & MASK32) + 0x9E3779B9) & MASK32
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK32
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK32
    return (z ^ (z >> 16)) & MASK32


class Xoshiro128ss:
    def __init__(self, seed: int) -> None:
        s0 = splitmix32(seed & MASK32)
        s1 = splitmix32(s0)
        s2 = splitmix32(s1)
        s3 = splitmix32(s2)
        if (s0 | s1 | s2 | s3) == 0:
            s0 = 0x9E3779B9
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def next_uint32(self) -> int:
        res = (rotl32((self.s1 * 5) & MASK32, 7) * 9) & MASK32
        t = (self.s1 << 9) & MASK32
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = rotl32(self.s3, 11)
        return res

    def next_float64(self) -> float:
        hi = self.next_uint32() >> 5
        lo = self.next_uint32() >> 6
        return (hi * 67_108_864 + lo) / 9_007_199_254_740_992.0

    def next_uniform_open_unit(self) -> float:
        u = self.next_float64()
        if u < OPEN_UNIT_MIN:
            return OPEN_UNIT_MIN
        if u > OPEN_UNIT_MAX:
            return OPEN_UNIT_MAX
        return u


def next_gaussian_box_muller(rng: Xoshiro128ss) -> float:
    u1 = rng.next_uniform_open_unit()
    u2 = rng.next_uniform_open_unit()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


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
# SHARED ELASTIC-MOT 34L BACKBONE, VISION, VQ-VAE & CODEC OPERATORS
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


def layer_name(layer_idx: int, sub: str, branch: str) -> str:
    suffix = "_gen" if branch == "g" else ""
    if sub in ("input_layernorm", "post_attention_layernorm"):
        return f"encoder.layers.{layer_idx}.{sub}{suffix}.weight"
    if sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"encoder.layers.{layer_idx}.self_attn.{sub}{suffix}.weight"
    if sub in ("gate_proj", "up_proj", "down_proj"):
        return f"encoder.layers.{layer_idx}.mlp.{sub}{suffix}.weight"
    raise ValueError(f"Unknown layer sub-tensor: {sub}")


def run_elastic_forward_pass(
    inputs_embeds: torch.Tensor,
    routing: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    mask_mode: str,
    prefill_mode: str = "hf_faithful",
    past_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    past_routing: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor] | None]]:
    """Run stateless 34-layer Elastic-MoT forward supporting `hf_faithful` & `elastic_pruned`."""
    seq_len = inputs_embeds.shape[0]
    n_q = int(cfg["numAttentionHeads"])
    n_kv = int(cfg["numKeyValueHeads"])
    kv_group = int(cfg["kvGroupSize"])
    head_dim = int(cfg["headDim"])
    eps = float(cfg["rmsNormEps"])
    num_layers = int(cfg["numLayers"])
    dual_tower_layers = int(cfg["dualTowerLayers"])

    has_u = bool((routing == 0).any().item())
    has_g = bool((routing == 1).any().item())
    canonical = "mixed" if (has_u and has_g) else ("g_only" if has_g else "u_only")

    past_len = int(past_routing.shape[0]) if past_routing is not None else 0
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
        base_mask = k_idx <= q_idx
    elif mask_mode == "bidirectional":
        base_mask = torch.ones((seq_len, kv_len), dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    if past_len > 0:
        r_past = past_routing if past_routing is not None else torch.zeros(past_len, dtype=torch.int64)
        r_kv = torch.cat([r_past, routing], dim=0)
        cached_has_gen = bool((r_past == 1).any().item())
    else:
        r_kv = routing
        cached_has_gen = False

    skip_split = prefill_mode == "elastic_pruned" and canonical == "u_only"
    prune_u_split = prefill_mode == "elastic_pruned" and canonical == "mixed"
    scale = 1.0 / math.sqrt(float(head_dim))

    h = inputs_embeds.to(torch.float32).contiguous()
    new_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
    last_executed = -1

    for l in range(num_layers):
        split_tower = l >= dual_tower_layers
        if skip_split and split_tower:
            new_kvs[l] = None
            continue

        understanding_pruned = split_tower and prune_u_split

        input_norm = routed_rms_norm(
            h,
            routing,
            weights[layer_name(l, "input_layernorm", "u")],
            weights[layer_name(l, "input_layernorm", "g")],
            eps,
            skip_u=understanding_pruned,
        )
        q_pre = routed_linear(
            input_norm,
            routing,
            weights[layer_name(l, "q_proj", "u")],
            weights[layer_name(l, "q_proj", "g")],
            skip_u=understanding_pruned,
        )
        k_pre = routed_linear(
            input_norm,
            routing,
            weights[layer_name(l, "k_proj", "u")],
            weights[layer_name(l, "k_proj", "g")],
            skip_u=understanding_pruned,
        )
        v_new = routed_linear(
            input_norm,
            routing,
            weights[layer_name(l, "v_proj", "u")],
            weights[layer_name(l, "v_proj", "g")],
            skip_u=understanding_pruned,
        )

        q_rot_h = apply_rotary_pos_emb(q_pre.view(seq_len, n_q, head_dim), cos, sin)
        k_rot_h = apply_rotary_pos_emb(k_pre.view(seq_len, n_kv, head_dim), cos, sin)
        if understanding_pruned:
            u_mask = routing == 0
            q_rot_h[u_mask] = 0.0
            k_rot_h[u_mask] = 0.0

        q_scaled_h = q_rot_h * llama4_scales[:, None, None]
        new_kvs[l] = (k_rot_h.reshape(seq_len, n_kv * head_dim), v_new)

        drop_cached_u = split_tower and canonical == "g_only" and past_len > 0 and not cached_has_gen
        if split_tower:
            eff_mask = base_mask & (routing[:, None] == r_kv[None, :])
        else:
            eff_mask = base_mask

        if drop_cached_u or past_len == 0:
            k_active = k_rot_h
            v_active = v_new.view(seq_len, n_kv, head_dim)
            active_mask = eff_mask[:, past_len:]
        else:
            assert past_kvs is not None and past_kvs[l] is not None
            k_past = past_kvs[l][0].view(past_len, n_kv, head_dim)
            v_past = past_kvs[l][1].view(past_len, n_kv, head_dim)
            k_active = torch.cat([k_past, k_rot_h], dim=0)
            v_active = torch.cat([v_past, v_new.view(seq_len, n_kv, head_dim)], dim=0)
            active_mask = eff_mask

        attn_concat_h = torch.zeros((seq_len, n_q, head_dim), dtype=torch.float32)
        for hq in range(n_q):
            hkv = hq // kv_group
            q_h = q_scaled_h[:, hq, :]
            k_h = k_active[:, hkv, :]
            v_h = v_active[:, hkv, :]
            for qi in range(seq_len):
                if understanding_pruned and int(routing[qi].item()) == 0:
                    continue
                valid_cols = torch.nonzero(active_mask[qi], as_tuple=False).reshape(-1)
                if valid_cols.numel() == 0:
                    continue
                dots = torch.matmul(k_h[valid_cols], q_h[qi]) * scale
                p_valid = torch.softmax(dots, dim=-1)
                attn_concat_h[qi, hq, :] = torch.matmul(p_valid, v_h[valid_cols])

        attn_out = routed_linear(
            attn_concat_h.reshape(seq_len, n_q * head_dim),
            routing,
            weights[layer_name(l, "o_proj", "u")],
            weights[layer_name(l, "o_proj", "g")],
            skip_u=understanding_pruned,
        )
        if understanding_pruned:
            h_mid = h.clone()
            g_mask = routing == 1
            h_mid[g_mask] = h[g_mask] + attn_out[g_mask]
        else:
            h_mid = h + attn_out

        post_norm = routed_rms_norm(
            h_mid,
            routing,
            weights[layer_name(l, "post_attention_layernorm", "u")],
            weights[layer_name(l, "post_attention_layernorm", "g")],
            eps,
            skip_u=understanding_pruned,
        )
        gate_pre = routed_linear(
            post_norm,
            routing,
            weights[layer_name(l, "gate_proj", "u")],
            weights[layer_name(l, "gate_proj", "g")],
            skip_u=understanding_pruned,
        )
        gate_silu = F.silu(gate_pre)
        if understanding_pruned:
            gate_silu[routing == 0] = 0.0
        up_val = routed_linear(
            post_norm,
            routing,
            weights[layer_name(l, "up_proj", "u")],
            weights[layer_name(l, "up_proj", "g")],
            skip_u=understanding_pruned,
        )
        mlp_out = routed_linear(
            gate_silu * up_val,
            routing,
            weights[layer_name(l, "down_proj", "u")],
            weights[layer_name(l, "down_proj", "g")],
            skip_u=understanding_pruned,
        )
        if understanding_pruned:
            h_next = h_mid.clone()
            g_mask = routing == 1
            h_next[g_mask] = h_mid[g_mask] + mlp_out[g_mask]
            h = h_next
        else:
            h = h_mid + mlp_out
        last_executed = l

    final_norm = None
    if last_executed == num_layers - 1:
        final_norm = routed_rms_norm(
            h,
            routing,
            weights["encoder.norm.weight"],
            weights["encoder.norm_gen.weight"],
            eps,
            skip_u=prune_u_split,
        )
    return final_norm, new_kvs


def run_mini_radio_vision_pipeline(
    rgb_chw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    clip_mean: list[float],
    clip_std: list[float],
    radio_multiply: float,
) -> tuple[torch.Tensor, int, int]:
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
    raw_patches = torch.stack(patches_list, dim=0)
    patch_embed = F.linear(raw_patches, weights["encoder.vision_tower.patch_generator.embedder.weight"])

    base_pos = weights["encoder.vision_tower.pos_embed"]
    if pr == 4 and pc == 4:
        interp_pos = base_pos.clone()
    else:
        pos_4d = base_pos.view(4, 4, 16).permute(2, 0, 1).unsqueeze(0)
        interp_4d = F.interpolate(pos_4d, size=(pr, pc), mode="bilinear", align_corners=False)
        interp_pos = interp_4d.squeeze(0).permute(1, 2, 0).reshape(n_patches, 16).contiguous()

    cls_tokens = weights["encoder.vision_tower.cls_token"]
    x = torch.cat([cls_tokens, patch_embed + interp_pos], dim=0)
    seq_len = x.shape[0]
    scale = 0.5

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
        qh = q.view(seq_len, 4, 4).transpose(0, 1)
        kh = k.view(seq_len, 4, 4).transpose(0, 1)
        vh = v.view(seq_len, 4, 4).transpose(0, 1)
        probs = torch.softmax(torch.matmul(qh, kh.transpose(-2, -1)) * scale, dim=-1)
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
        fc2 = F.linear(
            F.gelu(fc1, approximate="none"),
            weights[f"encoder.vision_tower.blocks.{b}.mlp.fc2.weight"],
            weights[f"encoder.vision_tower.blocks.{b}.mlp.fc2.bias"],
        )
        x = post_attn + fc2

    patch_features = x[4:, :].contiguous()
    tr = pr // 2
    tc = pc // 2
    pf_grid = patch_features.view(pr, pc, 16)
    gathered_rows = [
        torch.cat(
            [
                pf_grid[2 * br, 2 * bc],
                pf_grid[2 * br, 2 * bc + 1],
                pf_grid[2 * br + 1, 2 * bc],
                pf_grid[2 * br + 1, 2 * bc + 1],
            ],
            dim=-1,
        )
        for br in range(tr)
        for bc in range(tc)
    ]
    gathered = torch.stack(gathered_rows, dim=0)
    normed = rms_norm(gathered, weights["encoder.mm_projector.norm.weight"], eps=1e-5)
    fc1_out = F.linear(normed, weights["encoder.mm_projector.fc1.weight"])
    fc2_out = F.linear(F.relu(fc1_out).pow(2), weights["encoder.mm_projector.fc2.weight"])
    return (fc2_out * float(radio_multiply)).to(torch.float32), tr, tc


def emu3_gn(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return F.group_norm(
        x,
        num_groups=4,
        weight=weights[f"{prefix}.weight"],
        bias=weights[f"{prefix}.bias"],
        eps=1e-6,
    )


def emu3_conv(
    x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str, stride: int = 1, padding: int = 1
) -> torch.Tensor:
    return F.conv2d(x, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"], stride=stride, padding=padding)


def emu3_resblock(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    h = F.silu(emu3_gn(x, weights, f"{prefix}.norm1"))
    h = emu3_conv(h, weights, f"{prefix}.conv1", stride=1, padding=1)
    h = F.silu(emu3_gn(h, weights, f"{prefix}.norm2"))
    h = emu3_conv(h, weights, f"{prefix}.conv2", stride=1, padding=1)
    skip = emu3_conv(x, weights, f"{prefix}.nin_shortcut", stride=1, padding=0) if f"{prefix}.nin_shortcut.weight" in weights else x
    return skip + h


def emu3_attnblock(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    _, c, h_px, w_px = x.shape
    hw = h_px * w_px
    h = emu3_gn(x, weights, f"{prefix}.norm")
    q = emu3_conv(h, weights, f"{prefix}.q", stride=1, padding=0).reshape(c, hw)
    k = emu3_conv(h, weights, f"{prefix}.k", stride=1, padding=0).reshape(c, hw)
    v = emu3_conv(h, weights, f"{prefix}.v", stride=1, padding=0).reshape(c, hw)
    probs = torch.softmax(torch.matmul(q.transpose(0, 1), k) / math.sqrt(float(c)), dim=-1)
    out = torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1).reshape(1, c, h_px, w_px)
    return x + emu3_conv(out, weights, f"{prefix}.proj_out", stride=1, padding=0)


def encode_mini_emu3_vq(rgb_chw: torch.Tensor, weights: dict[str, torch.Tensor]) -> list[int]:
    norm_in = (rgb_chw.to(torch.float32) * 2.0 - 1.0).unsqueeze(0)
    h = emu3_conv(norm_in, weights, "encoder.vqvae.encoder.conv_in", stride=1, padding=1)
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.down.0.block.0")
    h = emu3_conv(F.pad(h, (0, 1, 0, 1), value=0.0), weights, "encoder.vqvae.encoder.down.0.downsample.conv", stride=2, padding=0)
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.down.1.block.0")
    h = emu3_conv(F.pad(h, (0, 1, 0, 1), value=0.0), weights, "encoder.vqvae.encoder.down.1.downsample.conv", stride=2, padding=0)
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.mid.block_1")
    h = emu3_attnblock(h, weights, "encoder.vqvae.encoder.mid.attn_1")
    h = emu3_resblock(h, weights, "encoder.vqvae.encoder.mid.block_2")
    h = F.silu(emu3_gn(h, weights, "encoder.vqvae.encoder.norm_out"))
    h = emu3_conv(h, weights, "encoder.vqvae.encoder.conv_out", stride=1, padding=1)
    z_cont = emu3_conv(h, weights, "encoder.vqvae.quant_conv", stride=1, padding=0)
    _, z_ch, gr, gc = z_cont.shape
    z_flat = z_cont.squeeze(0).permute(1, 2, 0).reshape(gr * gc, z_ch).contiguous()
    codebook = weights["encoder.vqvae.quantize.embedding.weight"]
    return to_i32_list(torch.matmul(z_flat, codebook.transpose(0, 1)).argmax(dim=-1))


def decode_mini_emu3_vq(
    codes: list[int], gr: int, gc: int, weights: dict[str, torch.Tensor]
) -> torch.Tensor:
    codebook = weights["encoder.vqvae.quantize.embedding.weight"]
    idx = torch.tensor(codes, dtype=torch.int64)
    z_quant = codebook[idx].reshape(gr, gc, 8).permute(2, 0, 1).unsqueeze(0).contiguous()
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
    return torch.clamp((raw_out + 1.0) * 0.5, 0.0, 1.0)


def compress_packed_image_embeddings(
    child_embs: torch.Tensor,
    child_rows: int,
    child_cols: int,
    is_source_edit: bool,
    combined_scale: float,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    normed = rms_norm(child_embs, weights["encoder.packed_image_codec.down_norm.weight"], eps=1e-6)
    pr = child_rows // 2
    pc = child_cols // 2
    grid = normed.view(child_rows, child_cols, 8)
    packed_rows = [
        torch.cat(
            [grid[2 * r, 2 * c], grid[2 * r, 2 * c + 1], grid[2 * r + 1, 2 * c], grid[2 * r + 1, 2 * c + 1]],
            dim=-1,
        )
        for r in range(pr)
        for c in range(pc)
    ]
    packed = torch.stack(packed_rows, dim=0)
    proj = F.linear(packed, weights["encoder.packed_image_codec.down_projection.weight"])
    scaled = (proj * float(combined_scale)).to(torch.float32)
    sincos = compute_2d_sincos_pos_embed(pr, pc, 8, marker_column=(not is_source_edit))
    return (scaled + sincos).to(torch.float32)


def expand_packed_image_hidden_states(
    parent_hidden: torch.Tensor,
    parent_rows: int,
    parent_cols: int,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    normed = rms_norm(parent_hidden, weights["encoder.packed_image_codec.up_norm.weight"], eps=1e-6)
    up_proj = F.linear(normed, weights["encoder.packed_image_codec.up_projection.weight"])
    child_rows = parent_rows * 2
    child_cols = parent_cols * 2
    unpacked = torch.zeros((child_rows, child_cols, 8), dtype=torch.float32)
    for r in range(parent_rows):
        for c in range(parent_cols):
            vec = up_proj[r * parent_cols + c]
            unpacked[2 * r, 2 * c] = vec[0:8]
            unpacked[2 * r, 2 * c + 1] = vec[8:16]
            unpacked[2 * r + 1, 2 * c] = vec[16:24]
            unpacked[2 * r + 1, 2 * c + 1] = vec[24:32]
    flat = unpacked.reshape(child_rows * child_cols, 8).contiguous()
    return F.linear(flat, weights["encoder.gen_predictor.weight"])


# ============================================================================
# CATEGORY 1: BALANCED SEMANTIC TREE, PROGRESSIVE REFINEMENT & U26
# ============================================================================

def canonicalize_eigvec_fp64(vec: torch.Tensor) -> torch.Tensor:
    abs_v = torch.abs(vec)
    # Tie-break smallest index
    max_idx = int(torch.argmax(abs_v).item())
    if float(vec[max_idx].item()) < 0.0:
        vec = -vec
    return vec


def build_balanced_semantic_tree_pytorch(codebook: torch.Tensor) -> dict[str, Any]:
    vocab_size, latent_dim = codebook.shape
    depth = int(round(math.log2(vocab_size)))
    cb64 = codebook.to(torch.float64)
    norms = torch.clamp(torch.linalg.norm(cb64, dim=-1, keepdim=True), min=1e-12)
    normed = cb64 / norms

    leaf_to_token = [0] * vocab_size
    token_to_leaf = [0] * vocab_size
    heap_radii = [0.0] * (vocab_size - 1)
    leaf_cursor = 0

    def split_node(heap_idx: int, indices: list[int]) -> None:
        nonlocal leaf_cursor
        count = len(indices)
        if count == 1:
            tid = indices[0]
            leaf_to_token[leaf_cursor] = tid
            token_to_leaf[tid] = leaf_cursor
            leaf_cursor += 1
            return

        sub = normed[indices]  # [count, D]
        diffs = sub[:, None, :] - sub[None, :, :]
        dist_sq = (diffs * diffs).sum(dim=-1)
        max_dist_sq = float(dist_sq.max().item())
        heap_radii[heap_idx] = float(torch.tensor(max_dist_sq, dtype=torch.float32).item())

        centroid = sub.mean(dim=0)
        centered = sub - centroid
        cov = centered.transpose(0, 1) @ centered
        _, evecs = torch.linalg.eigh(cov)
        principal = canonicalize_eigvec_fp64(evecs[:, -1].clone())
        projs = (centered @ principal).tolist()

        paired = list(zip(indices, projs))
        paired.sort(key=lambda item: (item[1], item[0]))
        ordered = [item[0] for item in paired]
        half = count // 2
        split_node(2 * heap_idx + 1, ordered[:half])
        split_node(2 * heap_idx + 2, ordered[half:])

    split_node(0, list(range(vocab_size)))

    # Top-3 deterministic PCA projection
    centroid_all = normed.mean(dim=0)
    centered_all = normed - centroid_all
    cov_all = centered_all.transpose(0, 1) @ centered_all
    _, evecs_all = torch.linalg.eigh(cov_all)
    axes = [canonicalize_eigvec_fp64(evecs_all[:, -(k + 1)].clone()) for k in range(3)]
    basis = torch.stack(axes, dim=1)  # [D, 3]
    pca3d = (centered_all @ basis).to(torch.float32).contiguous()

    return {
        "vocabSize": vocab_size,
        "depth": depth,
        "latentDim": latent_dim,
        "leafToToken": leaf_to_token,
        "tokenToLeaf": token_to_leaf,
        "heapRadii": heap_radii,
        "pca3d": to_f32_list(pca3d, "tree.pca3d"),
    }


def semantic_tree_levels_for_ages(
    stages: list[int], depth: int, refinement_steps: int
) -> tuple[list[int], list[float], list[int]]:
    levels: list[int] = []
    noises: list[float] = []
    balls: list[int] = []
    full_ball = 1 << depth
    for s in stages:
        if s < 0:
            levels.append(0)
            noises.append(1.0)
            balls.append(full_ball)
        elif s >= refinement_steps:
            levels.append(depth)
            noises.append(0.0)
            balls.append(1)
        else:
            lvl = (s * depth) // refinement_steps
            levels.append(lvl)
            noises.append(float(torch.tensor(1.0 - s / refinement_steps, dtype=torch.float32).item()))
            balls.append(1 << (depth - lvl))
    return levels, noises, balls


def sample_representatives(
    tree: dict[str, Any], proposals: list[int], levels: list[int], rand_bits: list[int]
) -> list[int]:
    depth = tree["depth"]
    token_to_leaf = tree["tokenToLeaf"]
    leaf_to_token = tree["leafToToken"]
    out: list[int] = []
    for prop, lvl, bits in zip(proposals, levels, rand_bits):
        shift = depth - lvl
        if shift <= 0:
            out.append(prop)
        else:
            leaf = token_to_leaf[prop]
            start = (leaf >> shift) << shift
            mask = (1 << shift) - 1
            out.append(leaf_to_token[start + ((bits & MASK32) & mask)])
    return out


def build_semantic_tree_embeddings(
    representatives: list[int],
    noise_weights: list[float],
    gen_embedding: torch.Tensor,
    mask_row_id: int = UNI_GEN_IMAGE_MASK_ID,
) -> torch.Tensor:
    rep_t = torch.tensor(representatives, dtype=torch.int64)
    eta_t = torch.tensor(noise_weights, dtype=torch.float32)[:, None]
    base = gen_embedding[rep_t].to(torch.float32)
    mask_vec = gen_embedding[mask_row_id].to(torch.float32)[None, :]
    return torch.where(eta_t == 0.0, base, base + eta_t * mask_vec).to(torch.float32)


def semantic_tree_activation_counts(
    content_count: int, steps: int, refinement_steps: int, shift: float = 5.0, refinement_gap: int = 0
) -> dict[str, Any]:
    act_steps = min(steps, content_count)
    boundaries: list[int] = []
    for m in range(act_steps + 1):
        if m == act_steps:
            boundaries.append(content_count)
        else:
            t = m / act_steps
            sigma = (shift * t) / (1.0 + (shift - 1.0) * t)
            boundaries.append(int(math.floor(sigma * content_count)))

    raw = [max(1, boundaries[m + 1] - boundaries[m]) for m in range(act_steps)]
    total = sum(raw)
    idx = 0
    while total > content_count:
        if raw[idx] > 1:
            raw[idx] -= 1
            total -= 1
        idx = (idx + 1) % act_steps
    raw.reverse()

    flush_rounds = (refinement_steps - 1) + refinement_steps * refinement_gap
    total_rounds = act_steps + flush_rounds
    counts = raw + [0] * flush_rounds
    cum: list[int] = []
    running = 0
    for c in counts:
        running += c
        cum.append(running)
    return {
        "contentCount": content_count,
        "activationSteps": act_steps,
        "refinementSteps": refinement_steps,
        "flushRounds": flush_rounds,
        "totalRounds": total_rounds,
        "activationCounts": counts,
        "cumulativeActivations": cum,
    }


def stratified_random(n: int = 4, seed: int = 42) -> list[int]:
    rng = Xoshiro128ss(seed)

    def fisher_yates(arr: list[int]) -> None:
        for i in range(len(arr) - 1, 0, -1):
            j = rng.next_uint32() % (i + 1)
            arr[i], arr[j] = arr[j], arr[i]

    total = n * n
    levels = int(round(math.log2(n)))
    occupied = [0] * total
    order: list[int] = []

    for k in range(levels):
        blocks_per_side = 1 << k
        num_blocks = blocks_per_side * blocks_per_side
        child_size = n >> (k + 1)
        block_size = child_size * 2
        child_area = child_size * child_size

        block_order = list(range(num_blocks))
        fisher_yates(block_order)

        for b_idx in block_order:
            br = b_idx // blocks_per_side
            bc = b_idx % blocks_per_side
            quad_order = [0, 1, 2, 3]
            fisher_yates(quad_order)

            for q in quad_order:
                qr = q >> 1
                qc = q & 1
                r0 = br * block_size + qr * child_size
                c0 = bc * block_size + qc * child_size
                has_occ = False
                for dr in range(child_size):
                    for dc in range(child_size):
                        if occupied[(r0 + dr) * n + (c0 + dc)] == 1:
                            has_occ = True
                            break
                    if has_occ:
                        break
                if not has_occ:
                    pick = rng.next_uint32() % child_area
                    dr = pick // child_size
                    dc = pick % child_size
                    flat = (r0 + dr) * n + (c0 + dc)
                    occupied[flat] = 1
                    order.append(flat)

    if len(order) < total:
        rem = [flat for flat in range(total) if occupied[flat] == 0]
        fisher_yates(rem)
        order.extend(rem)
    return order


def ages_at_time(t: float, tau: list[float], epsilon: float, total_k: int) -> list[int]:
    out: list[int] = []
    for ti in tau:
        if t <= ti:
            out.append(0)
        elif t >= ti + epsilon:
            out.append(total_k)
        else:
            out.append(int(math.floor((total_k * (t - ti)) / epsilon)))
    return out


def analyze_u26_floor_mismatch(depth: int, refinement_steps: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mismatched = 0
    for s in range(1, refinement_steps):
        train_age = refinement_steps - s
        train_level = depth - ((train_age * depth) // refinement_steps)
        infer_level = (s * depth) // refinement_steps
        delta = train_level - infer_level
        if delta != 0:
            mismatched += 1
        rows.append(
            {
                "stageIndex": s,
                "trainAge": train_age,
                "trainLevel": train_level,
                "inferLevel": infer_level,
                "levelDelta": delta,
                "trainBallSize": 1 << (depth - train_level),
                "inferBallSize": 1 << (depth - infer_level),
            }
        )
    return {
        "treeDepth": depth,
        "refinementSteps": refinement_steps,
        "hasMismatch": mismatched > 0,
        "mismatchedStageCount": mismatched,
        "rows": rows,
    }


# ============================================================================
# CATEGORY 2 & 3: T2I & DUAL-CFG EDITING SAMPLERS IN PYTORCH 2.15.0
# ============================================================================

def apply_t2i_cfg(uncond: torch.Tensor, cond: torch.Tensor, w: float) -> torch.Tensor:
    wf = float(torch.tensor(w, dtype=torch.float32).item())
    return ((1.0 + wf) * cond - wf * uncond).to(torch.float32)


def apply_edit_dual_cfg(
    uncond: torch.Tensor,
    vq_drop: torch.Tensor,
    full: torch.Tensor,
    mode: str,
    w_txt: float,
    w_img: float,
) -> torch.Tensor:
    wt = float(torch.tensor(w_txt, dtype=torch.float32).item())
    wi = float(torch.tensor(w_img, dtype=torch.float32).item())
    if mode == "legacy":
        cond = (full + wi * vq_drop) / (1.0 + wi)
        return ((1.0 + wt) * cond - wt * uncond).to(torch.float32)
    if mode == "ip2p":
        return (uncond + wi * (vq_drop - uncond) + wt * (full - vq_drop)).to(torch.float32)
    if mode == "independent":
        return (uncond + wt * (full - uncond) + wi * (full - vq_drop)).to(torch.float32)
    raise ValueError(f"Unsupported mode: {mode}")


def sample_vq_proposals_pytorch(
    guided_logits: torch.Tensor,
    is_marker: list[int],
    temperature: float,
    round_uniforms: list[float] | None,
) -> list[int]:
    grid_len, vocab_size = guided_logits.shape
    proposals = [0] * grid_len
    cursor = 0
    for i in range(grid_len):
        if is_marker[i] == 1:
            proposals[i] = 0
            continue
        row = guided_logits[i]
        if temperature <= 0.0:
            proposals[i] = int(row.argmax(dim=-1).item())
            continue
        assert round_uniforms is not None
        max_val = float(row.max().item())
        best_idx = -1
        best_score = float("-inf")
        row_list = row.tolist()
        for v in range(vocab_size):
            u = min(max(float(round_uniforms[cursor]), OPEN_UNIT_MIN), OPEN_UNIT_MAX)
            cursor += 1
            lv = float(row_list[v])
            if not math.isfinite(lv):
                continue
            score = (lv - max_val) / float(temperature) - math.log(-math.log(u))
            if score > best_score:
                best_score = score
                best_idx = v
        proposals[i] = best_idx
    return proposals


def extract_top4(guided_logits: torch.Tensor) -> tuple[list[int], list[float]]:
    probs = torch.softmax(guided_logits, dim=-1)
    top_ids: list[int] = []
    top_probs: list[float] = []
    for i in range(guided_logits.shape[0]):
        row = probs[i].tolist()
        ranked = sorted(range(len(row)), key=lambda v: (-row[v], v))[:4]
        for v in ranked:
            top_ids.append(v)
            top_probs.append(float(row[v]))
    return top_ids, top_probs


def init_marked_tree_state(
    initial_codes: list[int],
    refinement_steps: int,
    depth: int = 8,
    locked_mask: list[int] | None = None,
    locked_source: list[int] | None = None,
) -> dict[str, Any]:
    stages: list[int] = []
    reps: list[int] = []
    levels: list[int] = []
    noises: list[float] = []
    is_marker: list[int] = []

    for r in range(4):
        for col in range(4):
            c = r * 4 + col
            is_marker.append(0)
            if locked_mask is not None and locked_mask[c] == 1:
                assert locked_source is not None
                stages.append(refinement_steps)
                reps.append(locked_source[c])
                levels.append(depth)
                noises.append(0.0)
            else:
                stages.append(-1)
                reps.append(initial_codes[c])
                levels.append(0)
                noises.append(1.0)
        marker = UNI_GEN_IMAGE_END_ID if r == 3 else UNI_GEN_ROW_MARKER_ID
        for _ in range(2):
            is_marker.append(1)
            stages.append(refinement_steps)
            reps.append(marker)
            levels.append(depth)
            noises.append(0.0)

    return {
        "stages": stages,
        "representatives": reps,
        "levels": levels,
        "noiseWeights": noises,
        "isMarker": is_marker,
        "refinementSteps": refinement_steps,
    }


def advance_tree_state(
    tree: dict[str, Any],
    state: dict[str, Any],
    proposals: list[int],
    activated_content: list[int],
    rand_bits_content: list[int],
    locked_mask: list[int] | None = None,
) -> dict[str, Any]:
    content_to_grid = [g for g, m in enumerate(state["isMarker"]) if m == 0]
    act_set = set(activated_content)
    n_ref = state["refinementSteps"]

    next_stages = list(state["stages"])
    next_reps = list(state["representatives"])
    next_levels = list(state["levels"])
    next_noises = list(state["noiseWeights"])

    adv_g: list[int] = []
    adv_next_stage: list[int] = []
    adv_props: list[int] = []
    adv_bits: list[int] = []

    for c, g in enumerate(content_to_grid):
        if locked_mask is not None and locked_mask[c] == 1:
            continue
        cur = state["stages"][g]
        if cur >= n_ref:
            continue
        ns = 1 if (cur < 0 and c in act_set) else (cur + 1 if cur >= 0 else None)
        if ns is not None:
            adv_g.append(g)
            adv_next_stage.append(ns)
            adv_props.append(proposals[g])
            adv_bits.append(rand_bits_content[c])

    if adv_g:
        lvls, nws, _ = semantic_tree_levels_for_ages(adv_next_stage, tree["depth"], n_ref)
        reps = sample_representatives(tree, adv_props, lvls, adv_bits)
        for g, ns, lvl, nw, rep in zip(adv_g, adv_next_stage, lvls, nws, reps):
            next_stages[g] = ns
            next_levels[g] = lvl
            next_noises[g] = nw
            next_reps[g] = rep

    return {
        "stages": next_stages,
        "representatives": next_reps,
        "levels": next_levels,
        "noiseWeights": next_noises,
        "isMarker": list(state["isMarker"]),
        "refinementSteps": n_ref,
    }


def strip_row_markers(marked: list[int]) -> list[int]:
    out: list[int] = []
    for r in range(4):
        out.extend(marked[r * 6 : r * 6 + 4])
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
        y0, y1 = r * vq_stride, (r + 1) * vq_stride
        for c in range(gc):
            x0, x1 = c * vq_stride, (c + 1) * vq_stride
            outside = y0 < pad_top or y1 > pad_top + res_h or x0 < pad_left or x1 > pad_left + res_w
            mask.append(1 if outside else 0)
    return mask


def run_image_trajectory_pytorch(
    *,
    task_kind: str,
    per_branch_embeds: list[torch.Tensor],
    per_branch_routing: list[torch.Tensor],
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inv_freq: torch.Tensor,
    attention_scaling: float,
    tree: dict[str, Any],
    combined_scale: float,
    activation_steps: int,
    refinement_steps: int,
    guidance_mode: str | None,
    guidance_scale_text: float,
    guidance_scale_image: float,
    temperature: float,
    initial_codes: list[int],
    round_uniforms_list: list[list[float]],
    round_rand_bits_list: list[list[int]],
    locked_mask: list[int] | None = None,
    locked_source: list[int] | None = None,
    letterbox_info: dict[str, Any] | None = None,
    crop_to_source: bool = False,
    verify_elastic_pruned_zero_ulp: bool = True,
    capture_dual_cfg_modes_round0: bool = False,
    capture_final_round_logits: bool = True,
) -> dict[str, Any]:
    b_cfg = len(per_branch_embeds)
    schedule = semantic_tree_activation_counts(16, activation_steps, refinement_steps, 5.0, 0)
    unmask_order = stratified_random(4, 42)
    gen_emb = weights["encoder.gen_embedding.weight"]
    fixed_prefix_emb = gen_emb[torch.tensor(UNI_GEN_FIXED_PREFIX_IDS, dtype=torch.int64)]
    all_g = torch.ones(10, dtype=torch.int64)

    # Stage 1 prefill in BOTH hf_faithful and (optionally) elastic_pruned
    past_kvs_hf: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
    past_kvs_pruned: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
    for b in range(b_cfg):
        _, kvs_hf = run_elastic_forward_pass(
            per_branch_embeds[b],
            per_branch_routing[b],
            weights,
            cfg,
            inv_freq,
            attention_scaling,
            mask_mode="causal",
            prefill_mode="hf_faithful",
        )
        past_kvs_hf.append(kvs_hf)
        if verify_elastic_pruned_zero_ulp:
            _, kvs_ep = run_elastic_forward_pass(
                per_branch_embeds[b],
                per_branch_routing[b],
                weights,
                cfg,
                inv_freq,
                attention_scaling,
                mask_mode="causal",
                prefill_mode="elastic_pruned",
            )
            past_kvs_pruned.append(kvs_ep)

    state = init_marked_tree_state(
        initial_codes,
        refinement_steps,
        depth=tree["depth"],
        locked_mask=locked_mask,
        locked_source=locked_source,
    )

    step_records: list[dict[str, Any]] = []
    round0_guided_logits: list[float] | None = None
    round0_alt_modes: dict[str, list[float]] | None = None
    final_round_guided_logits: list[float] | None = None

    for r in range(schedule["totalRounds"]):
        child_emb = build_semantic_tree_embeddings(state["representatives"], state["noiseWeights"], gen_emb)
        packed = compress_packed_image_embeddings(child_emb, 4, 6, False, combined_scale, weights)
        window_emb = torch.cat([fixed_prefix_emb, packed], dim=0)

        branch_final_norms: list[torch.Tensor] = []
        branch_logits: list[torch.Tensor] = []

        for b in range(b_cfg):
            fn_hf, _ = run_elastic_forward_pass(
                window_emb,
                all_g,
                weights,
                cfg,
                inv_freq,
                attention_scaling,
                mask_mode="bidirectional",
                prefill_mode="hf_faithful",
                past_kvs=past_kvs_hf[b],
                past_routing=per_branch_routing[b],
            )
            assert fn_hf is not None
            if verify_elastic_pruned_zero_ulp:
                fn_ep, _ = run_elastic_forward_pass(
                    window_emb,
                    all_g,
                    weights,
                    cfg,
                    inv_freq,
                    attention_scaling,
                    mask_mode="bidirectional",
                    prefill_mode="elastic_pruned",
                    past_kvs=past_kvs_pruned[b],
                    past_routing=per_branch_routing[b],
                )
                assert fn_ep is not None
                if not torch.equal(fn_hf, fn_ep):
                    max_diff = float((fn_hf - fn_ep).abs().max().item())
                    raise RuntimeError(
                        f"D5 0-ULP violation in PyTorch ({task_kind}, round={r}, branch={b}): maxDiff={max_diff}"
                    )

            branch_final_norms.append(fn_hf)
            logits_b = expand_packed_image_hidden_states(fn_hf[4:10], 2, 3, weights)
            branch_logits.append(logits_b)

        if b_cfg == 1:
            guided = branch_logits[0]
        elif b_cfg == 2:
            guided = apply_t2i_cfg(branch_logits[0], branch_logits[1], guidance_scale_text)
        else:
            assert guidance_mode is not None
            guided = apply_edit_dual_cfg(
                branch_logits[0],
                branch_logits[1],
                branch_logits[2],
                guidance_mode,
                guidance_scale_text,
                guidance_scale_image,
            )

        if r == 0:
            round0_guided_logits = to_f32_list(guided, f"{task_kind}.r0.guidedLogits")
            if capture_dual_cfg_modes_round0 and b_cfg == 3:
                round0_alt_modes = {
                    "independent": to_f32_list(
                        apply_edit_dual_cfg(
                            branch_logits[0],
                            branch_logits[1],
                            branch_logits[2],
                            "independent",
                            guidance_scale_text,
                            guidance_scale_image,
                        ),
                        "edit.r0.independent",
                    ),
                    "ip2p": to_f32_list(
                        apply_edit_dual_cfg(
                            branch_logits[0],
                            branch_logits[1],
                            branch_logits[2],
                            "ip2p",
                            guidance_scale_text,
                            guidance_scale_image,
                        ),
                        "edit.r0.ip2p",
                    ),
                }
        if capture_final_round_logits and r == schedule["totalRounds"] - 1:
            final_round_guided_logits = to_f32_list(guided, f"{task_kind}.rLast.guidedLogits")

        prev_cum = 0 if r == 0 else schedule["cumulativeActivations"][r - 1]
        curr_cum = schedule["cumulativeActivations"][r]
        activated = unmask_order[prev_cum:curr_cum]

        proposals = sample_vq_proposals_pytorch(
            guided,
            state["isMarker"],
            temperature,
            round_uniforms_list[r] if temperature > 0.0 else None,
        )
        next_state = advance_tree_state(
            tree,
            state,
            proposals,
            activated,
            round_rand_bits_list[r],
            locked_mask=locked_mask,
        )

        # Verify 0-ULP source-padding lock invariance at every round
        if locked_mask is not None and locked_source is not None:
            for c, is_locked in enumerate(locked_mask):
                if is_locked == 1:
                    g = (c // 4) * 6 + (c % 4)
                    if next_state["representatives"][g] != locked_source[c]:
                        raise RuntimeError(f"U18 locked cell {c} mutated at round {r}")

        top4_ids, top4_probs = extract_top4(guided)
        step_records.append(
            {
                "roundIndex": r,
                "isFlushRound": r >= schedule["activationSteps"],
                "activatedContentIndices": activated,
                "proposals": proposals,
                "stagesAfter": next_state["stages"],
                "representativesAfter": next_state["representatives"],
                "levelsAfter": next_state["levels"],
                "noiseWeightsAfter": next_state["noiseWeights"],
                "top4TokenIds": top4_ids,
                "top4Probs": top4_probs,
                "perBranchFinalNorm": [to_f32_list(fn, f"{task_kind}.r{r}.b{bi}.fn") for bi, fn in enumerate(branch_final_norms)],
            }
        )
        state = next_state

    final_marked = state["representatives"]
    final_content = strip_row_markers(final_marked)
    decoded_padded = decode_mini_emu3_vq(final_content, 4, 4, weights)
    if crop_to_source and letterbox_info is not None:
        pt = int(letterbox_info["padTop"])
        pl = int(letterbox_info["padLeft"])
        rh = int(letterbox_info["resizedHeight"])
        rw = int(letterbox_info["resizedWidth"])
        output_rgb = decoded_padded[:, pt : pt + rh, pl : pl + rw].contiguous()
    else:
        output_rgb = decoded_padded

    out: dict[str, Any] = {
        "taskKind": task_kind,
        "cfgBatchSize": b_cfg,
        "guidanceMode": guidance_mode,
        "guidanceScaleText": guidance_scale_text,
        "guidanceScaleImage": guidance_scale_image,
        "activationSteps": activation_steps,
        "refinementSteps": refinement_steps,
        "temperature": temperature,
        "elasticPrunedVerifiedZeroUlp": verify_elastic_pruned_zero_ulp,
        "passCount": 1 + schedule["totalRounds"],
        "branchForwardCount": b_cfg * (1 + schedule["totalRounds"]),
        "round0GuidedLogits": round0_guided_logits,
        "finalRoundGuidedLogits": final_round_guided_logits,
        "steps": step_records,
        "finalMarkedCodes": final_marked,
        "finalContentCodes": final_content,
        "decodedPaddedImage": to_f32_list(decoded_padded, f"{task_kind}.decodedPaddedImage"),
        "outputImage": to_f32_list(output_rgb, f"{task_kind}.outputImage"),
        "outputWidth": int(output_rgb.shape[2]),
        "outputHeight": int(output_rgb.shape[1]),
    }
    if round0_alt_modes is not None:
        out["round0DualCfgAltModes"] = round0_alt_modes
    return out


# ============================================================================
# CATEGORY 4: READ-ONLY OPD REFERENCE PARITY (U24, §5.3)
# ============================================================================

def compute_tail_kl_pytorch(
    target_probs: torch.Tensor,
    student_probs: torch.Tensor,
    top_k_target: int = 64,
    top_k_student: int = 128,
    tail_floor: float = 1e-5,
) -> dict[str, Any]:
    vocab = target_probs.shape[0]
    t_list = target_probs.tolist()
    s_list = student_probs.tolist()
    t_top = sorted(range(vocab), key=lambda i: (-t_list[i], i))[: min(vocab, top_k_target)]
    s_top = sorted(range(vocab), key=lambda i: (-s_list[i], i))[: min(vocab, top_k_student)]

    seen: set[int] = set()
    union: list[int] = []
    limit = min(vocab, top_k_student)
    for idx in t_top:
        if idx not in seen:
            seen.add(idx)
            union.append(idx)
    for idx in s_top:
        if len(union) >= limit:
            break
        if idx not in seen:
            seen.add(idx)
            union.append(idx)
    union.sort()

    u_idx = torch.tensor(union, dtype=torch.int64)
    p_t_u = target_probs.to(torch.float64)[u_idx]
    p_s_u = student_probs.to(torch.float64)[u_idx]

    raw_t = torch.cat(
        [
            torch.clamp(p_t_u, min=tail_floor),
            torch.clamp(1.0 - p_t_u.sum(dim=0, keepdim=True), min=tail_floor),
        ]
    )
    raw_s = torch.cat(
        [
            torch.clamp(p_s_u, min=tail_floor),
            torch.clamp(1.0 - p_s_u.sum(dim=0, keepdim=True), min=tail_floor),
        ]
    )
    p_t = raw_t / raw_t.sum()
    p_s = raw_s / raw_s.sum()

    reverse_kl = max(0.0, float((p_s * torch.log(p_s / p_t)).sum().item()))
    forward_kl = max(0.0, float((p_t * torch.log(p_t / p_s)).sum().item()))
    return {
        "unionIndices": union,
        "bucketCount": len(union) + 1,
        "pTargetBuckets": to_f32_list(p_t.to(torch.float32), "opd.pTargetBuckets"),
        "pStudentBuckets": to_f32_list(p_s.to(torch.float32), "opd.pStudentBuckets"),
        "pTargetTailMass": float(p_t[-1].to(torch.float32).item()),
        "pStudentTailMass": float(p_s[-1].to(torch.float32).item()),
        "reverseKl": reverse_kl,
        "forwardKl": forward_kl,
    }


def compute_matched_sigmas_pytorch(
    r: float = 0.9, seq_len: int = 1024, steps: int = 40, sigma_end: float = 0.02
) -> dict[str, Any]:
    frac = min(1.0, max(0.0, (seq_len - 256.0) / (4096.0 - 256.0)))
    mu = 0.5 + frac * (1.15 - 0.5)
    exp_mu = math.exp(mu)
    t_start = r / (r + exp_mu * (1.0 - r))
    t_end = sigma_end / (sigma_end + exp_mu * (1.0 - sigma_end))
    t_grid = torch.linspace(t_start, t_end, steps, dtype=torch.float64)
    sigmas = (exp_mu / (exp_mu + (1.0 / t_grid - 1.0))).to(torch.float32)
    return {
        "r": r,
        "seqLen": seq_len,
        "steps": steps,
        "mu": mu,
        "sigmaStart": r,
        "sigmaEnd": sigma_end,
        "sigmas": to_f32_list(sigmas, "opd.sigmas"),
    }


def build_opd_reference_fixture() -> dict[str, Any]:
    seed = 0x4F504431
    rng = Xoshiro128ss(seed)
    vocab = 256
    group_vocab = 32
    latent_size = 16
    response_len = 32

    t_logits = torch.zeros(vocab, dtype=torch.float32)
    sa_logits = torch.zeros(vocab, dtype=torch.float32)
    sd_logits = torch.zeros(vocab, dtype=torch.float32)
    for i in range(vocab):
        base = next_gaussian_box_muller(rng) * 1.4
        ar_d = next_gaussian_box_muller(rng) * 0.45
        df_d = next_gaussian_box_muller(rng) * 0.35
        t_logits[i] = float(base)
        sa_logits[i] = float(base * 0.9 + ar_d)
        sd_logits[i] = float(float(sa_logits[i].item()) * 0.95 + df_d)

    teacher_ar_probs = torch.softmax(t_logits.to(torch.float64), dim=-1).to(torch.float32)
    student_ar_probs = torch.softmax(sa_logits.to(torch.float64), dim=-1).to(torch.float32)
    student_diff_probs = torch.softmax(sd_logits.to(torch.float64), dim=-1).to(torch.float32)

    x0 = torch.zeros(latent_size, dtype=torch.float32)
    noise_delta = torch.zeros(latent_size, dtype=torch.float32)
    for i in range(latent_size):
        x0[i] = float(next_gaussian_box_muller(rng) * 0.5)
        noise_delta[i] = float(next_gaussian_box_muller(rng))

    exact_logits = torch.tensor([next_gaussian_box_muller(rng) * 1.2 for _ in range(vocab)], dtype=torch.float32)
    target_code = int(exact_logits.argmax().item())
    group_logits = torch.tensor([next_gaussian_box_muller(rng) * 1.2 for _ in range(group_vocab)], dtype=torch.float32)
    target_group = int(group_logits.argmax().item())

    ar_tail = compute_tail_kl_pytorch(teacher_ar_probs, student_ar_probs, 64, 128)
    cons_tail = compute_tail_kl_pytorch(student_ar_probs, student_diff_probs, 64, 128)
    expected_mask_denom = ((1.0 + 0.001) / 2.0) * response_len
    ar_opd_loss = 4.0 * ar_tail["reverseKl"]
    self_cons_loss = cons_tail["forwardKl"] / expected_mask_denom

    matched = compute_matched_sigmas_pytorch(0.9, 1024, 40, 0.02)
    x_noisy_r = ((1.0 - 0.9) * x0 + 0.9 * noise_delta).to(torch.float32)
    exact_ce = float(F.cross_entropy(exact_logits.to(torch.float64).unsqueeze(0), torch.tensor([target_code])).item())
    grouped_ce = float(F.cross_entropy(group_logits.to(torch.float64).unsqueeze(0), torch.tensor([target_group])).item())
    total_edit_loss = exact_ce + 0.2 * grouped_ce

    return {
        "seed": seed,
        "inputs": {
            "responseLen": response_len,
            "teacherArProbs": to_f32_list(teacher_ar_probs, "opd.teacherArProbs"),
            "studentArProbs": to_f32_list(student_ar_probs, "opd.studentArProbs"),
            "studentDiffProbs": to_f32_list(student_diff_probs, "opd.studentDiffProbs"),
            "x0": to_f32_list(x0, "opd.x0"),
            "noiseDelta": to_f32_list(noise_delta, "opd.noiseDelta"),
            "exactLogits": to_f32_list(exact_logits, "opd.exactLogits"),
            "targetCode": target_code,
            "groupLogits": to_f32_list(group_logits, "opd.groupLogits"),
            "targetGroup": target_group,
        },
        "understanding": {
            "expectedMaskDenominator": expected_mask_denom,
            "arTailKl": ar_tail,
            "consistencyTailKl": cons_tail,
            "arOpdLoss": ar_opd_loss,
            "selfConsistencyLoss": self_cons_loss,
            "totalUnderstandingOpdLoss": ar_opd_loss + self_cons_loss,
        },
        "flowEditing": {
            "r": 0.9,
            "matchedSigmas": matched,
            "xNoisyR": to_f32_list(x_noisy_r, "opd.xNoisyR"),
            "exactCe": exact_ce,
            "groupedCe": grouped_ce,
            "groupedCeWeight": 0.2,
            "totalEditingOpdLoss": total_edit_loss,
        },
    }


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def scene_to_tensor(scene: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(scene["data"], dtype=torch.float32).view(3, int(scene["height"]), int(scene["width"])).contiguous()


def build_equal_uncond_tokens(prefix_len: int) -> list[int]:
    if prefix_len == 1:
        return [UNI_BOS_ID]
    return [UNI_BOS_ID, UNI_IM_START_ID] + [UNI_PAD_ID] * (prefix_len - 2)


def main() -> None:
    if not torch.__version__.startswith(PINNED_TORCH_VERSION):
        raise RuntimeError(
            f"Pinned PyTorch {PINNED_TORCH_VERSION} required, found {torch.__version__}"
        )

    parser = argparse.ArgumentParser(description="Generate compact PyTorch 2.15.0 UNI3 diffusion parity fixtures")
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    manifest_path = args.params_dir / "uni_mini_full_manifest.json"
    manifest, weights = load_exported_full_weights(manifest_path)

    cfg = manifest["config"]
    clip_mean: list[float] = manifest["clipMean"]
    clip_std: list[float] = manifest["clipStd"]
    gain_presets = manifest["gainPresets"]
    mini_gen_scale = float(gain_presets["mini_default"]["combinedGenerationScale"])
    mini_radio_mult = float(gain_presets["mini_default"]["radioProjectorMultiply"])

    inv_freq, attention_scaling = compute_yarn_inv_freq(cfg["rope"])
    torch.set_grad_enabled(False)

    # 1. Semantic Tree, Progressive Refinement & U26 Floor Mismatch
    codebook = weights["encoder.vqvae.quantize.embedding.weight"]
    tree = build_balanced_semantic_tree_pytorch(codebook)

    u26_d17 = {str(k): analyze_u26_floor_mismatch(17, k) for k in (1, 4, 8, 16)}
    u26_d8 = {str(k): analyze_u26_floor_mismatch(8, k) for k in (1, 2, 4, 8)}

    levels_cases: dict[str, Any] = {}
    for n_ref in (1, 2, 4, 8):
        stages = [-1] + list(range(n_ref + 1))
        lvls, nws, balls = semantic_tree_levels_for_ages(stages, 8, n_ref)
        levels_cases[str(n_ref)] = {
            "stages": stages,
            "levels": lvls,
            "noiseWeights": nws,
            "ballSizes": balls,
        }

    rep_rng = Xoshiro128ss(0x52455031)
    rep_proposals = [rep_rng.next_uint32() & 0xFF for _ in range(16)]
    rep_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 0, 4, 8]
    rep_bits = [rep_rng.next_uint32() for _ in range(16)]
    rep_outputs = sample_representatives(tree, rep_proposals, rep_levels, rep_bits)

    emb_reps = [rep_outputs[c % 16] if (c % 6) < 4 else (UNI_GEN_IMAGE_END_ID if c >= 18 else UNI_GEN_ROW_MARKER_ID) for c in range(24)]
    emb_noises = [1.0, 0.75, 0.5, 0.25, 0.0, 0.0] * 4
    tree_embs = build_semantic_tree_embeddings(emb_reps, emb_noises, weights["encoder.gen_embedding.weight"])

    eq16_tau = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
    eq16_cases = [
        {"t": t_val, "epsilon": 0.25, "K": 17, "ages": ages_at_time(t_val, eq16_tau, 0.25, 17)}
        for t_val in (0.0, 0.2, 0.4, 0.65, 1.0)
    ]

    quotas = {
        "steps8_N4": semantic_tree_activation_counts(16, 8, 4, 5.0, 0),
        "steps8_N1": semantic_tree_activation_counts(16, 8, 1, 5.0, 0),
        "steps4_N2": semantic_tree_activation_counts(16, 4, 2, 5.0, 0),
    }
    stratified_4_42 = stratified_random(4, 42)

    # Shared compact explicit variates (U23) for 11 rounds (quantized to 6 decimals -> ~400 KB compact JSON)
    var_rng = Xoshiro128ss(0x56415233)
    shared_initial_codes = [var_rng.next_uint32() & 0xFF for _ in range(16)]
    shared_round_rand_bits = [[var_rng.next_uint32() for _ in range(16)] for _ in range(11)]
    shared_round_uniforms = [
        [round(min(max(var_rng.next_uniform_open_unit(), 1e-6), 1.0 - 1e-6), 6) for _ in range(16 * 256)]
        for _ in range(11)
    ]

    # 2. T2I Trajectories (Default steps=8 N=4 w=3.0 T=0.8; N=1 binary MDM; U30 guidanceScale=0 B=1)
    embed_table = weights["encoder.embed_tokens.weight"]
    t2i_prompt = [1, 10, 29, 3, 4, 6, 15, 5, 9]  # <s> <|im_start|> generate a red cat in blue sky
    t2i_uncond = build_equal_uncond_tokens(len(t2i_prompt))
    t2i_emb_uncond = embed_table[torch.tensor(t2i_uncond, dtype=torch.int64)]
    t2i_emb_cond = embed_table[torch.tensor(t2i_prompt, dtype=torch.int64)]
    t2i_route_u = torch.zeros(len(t2i_prompt), dtype=torch.int64)

    t2i_default = run_image_trajectory_pytorch(
        task_kind="text_to_image",
        per_branch_embeds=[t2i_emb_uncond, t2i_emb_cond],
        per_branch_routing=[t2i_route_u, t2i_route_u],
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        tree=tree,
        combined_scale=mini_gen_scale,
        activation_steps=8,
        refinement_steps=4,
        guidance_mode=None,
        guidance_scale_text=3.0,
        guidance_scale_image=0.0,
        temperature=0.8,
        initial_codes=shared_initial_codes,
        round_uniforms_list=shared_round_uniforms,
        round_rand_bits_list=shared_round_rand_bits,
        verify_elastic_pruned_zero_ulp=True,
    )

    t2i_binary_mdm_n1 = run_image_trajectory_pytorch(
        task_kind="text_to_image",
        per_branch_embeds=[t2i_emb_uncond, t2i_emb_cond],
        per_branch_routing=[t2i_route_u, t2i_route_u],
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        tree=tree,
        combined_scale=mini_gen_scale,
        activation_steps=8,
        refinement_steps=1,
        guidance_mode=None,
        guidance_scale_text=3.0,
        guidance_scale_image=0.0,
        temperature=0.8,
        initial_codes=shared_initial_codes,
        round_uniforms_list=shared_round_uniforms[:8],
        round_rand_bits_list=shared_round_rand_bits[:8],
        verify_elastic_pruned_zero_ulp=True,
        capture_final_round_logits=False,
    )

    t2i_guidance_zero_u30 = run_image_trajectory_pytorch(
        task_kind="text_to_image",
        per_branch_embeds=[t2i_emb_cond],
        per_branch_routing=[t2i_route_u],
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        tree=tree,
        combined_scale=mini_gen_scale,
        activation_steps=8,
        refinement_steps=4,
        guidance_mode=None,
        guidance_scale_text=0.0,
        guidance_scale_image=0.0,
        temperature=0.8,
        initial_codes=shared_initial_codes,
        round_uniforms_list=shared_round_uniforms,
        round_rand_bits_list=shared_round_rand_bits,
        verify_elastic_pruned_zero_ulp=True,
        capture_final_round_logits=False,
    )

    # 3. 3-Branch Dual-CFG Image Editing on non-square harbor_panorama_16x8 -> 16x16
    harbor_rgb = scene_to_tensor(manifest["scenes"]["harbor_panorama_16x8"])
    harbor_lb_entry = manifest["scenes"]["harbor_panorama_letterboxed_16x16"]
    harbor_lb_rgb = scene_to_tensor(harbor_lb_entry["image"])
    harbor_lb_info = harbor_lb_entry["info"]

    harbor_vis_embs, vis_rows, vis_cols = run_mini_radio_vision_pipeline(
        harbor_rgb, weights, clip_mean, clip_std, mini_radio_mult
    )
    assert (vis_rows, vis_cols) == (1, 2)
    harbor_src_codes = encode_mini_emu3_vq(harbor_lb_rgb, weights)
    harbor_pad_mask = compute_source_padding_vq_mask(harbor_lb_info, 4)

    gen_emb = weights["encoder.gen_embedding.weight"]
    src_child_emb = gen_emb[torch.tensor(harbor_src_codes, dtype=torch.int64)]
    src_packed_emb = compress_packed_image_embeddings(src_child_emb, 4, 4, True, mini_gen_scale, weights)

    edit_instr_tokens = [1, 10, 30, 36, 9]  # <s> <|im_start|> edit bright sky
    instr_emb = embed_table[torch.tensor(edit_instr_tokens, dtype=torch.int64)]
    pad4_emb = embed_table[torch.tensor([UNI_PAD_ID] * 4, dtype=torch.int64)]
    edit_prefix_len = 2 + 4 + len(edit_instr_tokens)  # 11
    edit_uncond_tokens = build_equal_uncond_tokens(edit_prefix_len)

    edit_emb0 = embed_table[torch.tensor(edit_uncond_tokens, dtype=torch.int64)]
    edit_emb1 = torch.cat([harbor_vis_embs, pad4_emb, instr_emb], dim=0)
    edit_emb2 = torch.cat([harbor_vis_embs, src_packed_emb, instr_emb], dim=0)

    edit_route0 = torch.zeros(edit_prefix_len, dtype=torch.int64)
    edit_route1 = torch.zeros(edit_prefix_len, dtype=torch.int64)
    edit_route2 = torch.zeros(edit_prefix_len, dtype=torch.int64)
    edit_route2[2:6] = 1  # 4 packed source-VQ G slots

    edit_harbor_legacy = run_image_trajectory_pytorch(
        task_kind="image_edit",
        per_branch_embeds=[edit_emb0, edit_emb1, edit_emb2],
        per_branch_routing=[edit_route0, edit_route1, edit_route2],
        weights=weights,
        cfg=cfg,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        tree=tree,
        combined_scale=mini_gen_scale,
        activation_steps=8,
        refinement_steps=4,
        guidance_mode="legacy",
        guidance_scale_text=1.6,
        guidance_scale_image=2.0,
        temperature=0.8,
        initial_codes=shared_initial_codes,
        round_uniforms_list=shared_round_uniforms,
        round_rand_bits_list=shared_round_rand_bits,
        locked_mask=harbor_pad_mask,
        locked_source=harbor_src_codes,
        letterbox_info=harbor_lb_info,
        crop_to_source=True,
        verify_elastic_pruned_zero_ulp=True,
        capture_dual_cfg_modes_round0=True,
    )

    # 4. Read-Only OPD Reference Parity
    opd_ref = build_opd_reference_fixture()

    fixture = {
        "schemaVersion": "uni-diffusion-parity-v1",
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
        "semanticTree": {
            "tree": tree,
            "u26FloorMismatch": {"d17": u26_d17, "d8": u26_d8},
            "levelsByRefinementSteps": levels_cases,
            "sampleRepresentatives": {
                "proposals": rep_proposals,
                "levels": rep_levels,
                "randBits": rep_bits,
                "representatives": rep_outputs,
            },
            "buildEmbeddings": {
                "representatives": emb_reps,
                "noiseWeights": emb_noises,
                "embeddings": to_f32_list(tree_embs, "tree.buildEmbeddings"),
            },
            "agesAtTimeEq16": {"tau": eq16_tau, "cases": eq16_cases},
            "quotas": quotas,
            "stratifiedOrder4Seed42": stratified_4_42,
        },
        "sharedExplicitVariates": {
            "initialRootCodes": shared_initial_codes,
            "roundRepRandBits": shared_round_rand_bits,
            "roundProposalUniforms": shared_round_uniforms,
        },
        "t2i": {
            "promptTokenIds": t2i_prompt,
            "defaultSteps8N4": t2i_default,
            "binaryMdmN1": t2i_binary_mdm_n1,
            "guidanceZeroU30": t2i_guidance_zero_u30,
        },
        "imageEdit": {
            "scenePresetId": "harbor_panorama_16x8",
            "promptTokenIds": edit_instr_tokens,
            "sourceVqCodes": harbor_src_codes,
            "sourcePaddingMask": harbor_pad_mask,
            "harborLegacy": edit_harbor_legacy,
        },
        "opdRef": opd_ref,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(fixture, separators=(",", ":"), allow_nan=False)
    args.out.write_text(payload, encoding="utf-8")
    print(
        f"Generated compact {args.out} ({len(payload):,} B, torch=={torch.__version__}, "
        f"766 tensors / 191,235 floats, sha256={manifest['weightsSha256'][:16]}...)"
    )


if __name__ == "__main__":
    main()
