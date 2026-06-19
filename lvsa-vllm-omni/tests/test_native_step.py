"""Tests for native denoise-step integration (parity fix #2, Wan path).

The plugin reads the true denoise step from vllm-omni's diffusion
``ForwardContext.denoise_step_idx`` when available, falling back to the
call-counting heuristic when it is None (older vllm-omni, no active context,
or a pipeline that does not publish it — HunyuanVideo 1.5 / Cosmos in 0.22.0).

These tests are mock-based and env-independent: ``native_denoise_step`` is
monkeypatched so the suite runs identically under the ``.venv`` test env
(where ``vllm_omni`` is absent and the guarded import returns None).
"""
from __future__ import annotations

import os

import pytest

import lvsa_vllm_omni.attention_impl as ai
from lvsa_vllm_omni import step_tracker
from lvsa_vllm_omni.config import LVSAConfig
from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    step_tracker.reset()
    yield
    step_tracker.reset()


# ── native_denoise_step helper ───────────────────────────────────────────────


def test_native_denoise_step_returns_none_without_vllm_omni():
    """In the .venv test env vllm_omni is absent → guarded import → None."""
    assert step_tracker.native_denoise_step() is None


# ── _StepCounter (attention_impl) ────────────────────────────────────────────


class TestStepCounterNative:
    def test_native_step_used_when_available(self, monkeypatch):
        """When the native step is populated it wins over call-counting."""
        monkeypatch.setenv("LVSA_N_BLOCKS", "3")
        c = ai._StepCounter()
        # Past calibration: call-counting alone would yield step 0 here.
        monkeypatch.setattr(ai, "native_denoise_step", lambda: 7)
        for i in range(5):
            step = c.tick(layer_id=i % 3, seq_len=100)
        assert step == 7
        assert c.step == 7
        # Side effect mirrored: thread-local step set to native value.
        assert step_tracker.get_step() == 7

    def test_falls_back_to_call_counting_when_native_none(self, monkeypatch):
        """native None → existing call-counting behavior is unchanged."""
        monkeypatch.setenv("LVSA_N_BLOCKS", "3")
        monkeypatch.setattr(ai, "native_denoise_step", lambda: None)
        c = ai._StepCounter()
        # 3 blocks x 2 passes = 6 calls per step (default cfg_passes=2).
        steps = [c.tick(layer_id=i % 3, seq_len=100) for i in range(12)]
        assert steps[5] == 0 and steps[6] == 1 and steps[11] == 1


# ── HunyuanLVSAState (hunyuan_hook) ──────────────────────────────────────────


class TestHunyuanStateNative:
    def test_native_step_used_when_available(self, monkeypatch):
        """Hook tick: native step wins over call-counting."""
        import lvsa_vllm_omni.hunyuan_hook as hh
        monkeypatch.setattr(hh, "native_denoise_step", lambda: 7)
        s = HunyuanLVSAState(LVSAConfig())
        for lid in (1, 2, 3, 1, 2):
            step = s.tick(layer_id=lid, seq_len=100)
        assert step == 7
        assert s._step == 7

    def test_falls_back_to_call_counting_when_native_none(self, monkeypatch):
        """native None → existing hook call-counting behavior is unchanged."""
        import lvsa_vllm_omni.hunyuan_hook as hh
        monkeypatch.setattr(hh, "native_denoise_step", lambda: None)
        monkeypatch.setenv("LVSA_CFG_PASSES", "1")
        s = HunyuanLVSAState(LVSAConfig())
        for lid in (1, 2, 3, 1):
            s.tick(layer_id=lid, seq_len=100)
        assert s._n_blocks == 3
        assert s._step == 1
