"""Tests for LVSAConfig."""
import os
import pytest
from lvsa_vllm_omni.config import LVSAConfig


class TestConfigDefaults:
    def test_default_values(self):
        c = LVSAConfig()
        assert c.window_size == 12
        assert c.n_first_frames == 4
        assert c.key_frame_interval == 16
        assert c.auto_keyframes is True
        assert c.rotate_keyframes is False
        assert c.expand_window is True
        assert c.backend == "sdpa"
        assert c.vae_temporal_factor == 4
        assert c.total_latent_frames is None
        assert c.sparsity_scale == 1.0

    def test_latent_properties(self):
        c = LVSAConfig(window_size=12, n_first_frames=4, key_frame_interval=16,
                        vae_temporal_factor=4)
        assert c.latent_window_size == 3
        assert c.latent_n_first_frames == 1
        assert c.latent_key_frame_interval == 4


class TestConfigFromEnv:
    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("LVSA_WINDOW_SIZE", "8")
        monkeypatch.setenv("LVSA_BACKEND", "flashinfer")
        monkeypatch.setenv("LVSA_ROTATE_KEYFRAMES", "true")
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", "61")
        c = LVSAConfig.from_env()
        assert c.window_size == 8
        assert c.backend == "flashinfer"
        assert c.rotate_keyframes is True
        assert c.total_latent_frames == 61

    def test_defaults_without_env(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("LVSA_"):
                monkeypatch.delenv(key, raising=False)
        c = LVSAConfig.from_env()
        assert c.window_size == 12

    def test_bool_false_values(self, monkeypatch):
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "0")
        monkeypatch.setenv("LVSA_EXPAND_WINDOW", "false")
        c = LVSAConfig.from_env()
        assert c.auto_keyframes is False
        assert c.expand_window is False

    def test_sparsity_scale_from_env(self, monkeypatch):
        monkeypatch.setenv("LVSA_SPARSITY_SCALE", "2.5")
        c = LVSAConfig.from_env()
        assert c.sparsity_scale == 2.5

    def test_expand_window_zero_disables(self, monkeypatch):
        """Parity fix #6: LVSA_EXPAND_WINDOW=0 must yield expand_window=False.

        The vllm-omni hooks feed ``config.expand_window`` into the authoritative
        ``LVSAMetadata.build`` call, so this flag must round-trip through from_env
        for the documented adaptive-window mode to actually take effect.
        """
        monkeypatch.setenv("LVSA_EXPAND_WINDOW", "0")
        assert LVSAConfig.from_env().expand_window is False

    def test_expand_window_defaults_true(self, monkeypatch):
        monkeypatch.delenv("LVSA_EXPAND_WINDOW", raising=False)
        monkeypatch.delenv("LVSA_CONFIG", raising=False)
        assert LVSAConfig.from_env().expand_window is True


class TestConfigFromJson:
    def test_parses_json(self):
        c = LVSAConfig.from_json('{"window_size": 16, "backend": "flashinfer", "sparsity_scale": 0.75}')
        assert c.window_size == 16
        assert c.backend == "flashinfer"
        assert c.n_first_frames == 4  # default preserved
        assert c.sparsity_scale == 0.75

    def test_ignores_unknown_keys(self):
        c = LVSAConfig.from_json('{"window_size": 8, "unknown_key": 42}')
        assert c.window_size == 8

    def test_json_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LVSA_CONFIG", '{"window_size": 24}')
        c = LVSAConfig.from_env()
        assert c.window_size == 24
