"""Tests for the sequence-parallel detection helper (``_sp.is_sp_active``).

The full SP signal (``forward_context.sp_active``) is only meaningful inside a
vllm-omni diffusion forward, so the GPU/SP behaviour is exercised by the
multi-GPU runs (TP engages, Ulysses falls back). On CPU we verify the SAFE
DEFAULT: without vllm-omni / a forward context, ``is_sp_active()`` returns False
so single-GPU, warmup, and CPU paths are never gated off.
"""
from __future__ import annotations

import torch

from lvsa_vllm_omni._sp import is_sp_active


def test_is_sp_active_false_without_vllm_omni():
    # vllm-omni absent in the CPU test env → import guard returns False, never raises.
    assert is_sp_active() is False


def test_is_sp_active_is_robust_to_repeated_calls():
    # Pure function over the (absent) forward context — stable, no side effects.
    assert is_sp_active() is False
    assert is_sp_active() is False


# NOTE: there is intentionally NO "forward_cuda falls back under SP" test.
# The backend MUST engage under Ulysses-SP (the framework gathers the full grid
# before forward_cuda) — GPU-verified. SP-safety is the geometry check, not an
# is_sp_active() gate; see attention_impl.forward_cuda.
