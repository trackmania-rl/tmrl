"""Tests for RT-MDP configuration defaults.

Verifies that:
- RTGYM_CONFIG has reset_act_buf=True by default
- act_buf_len is set and positive
- UPDATE_MODEL_INTERVAL is in a reasonable range
"""

import tmrl.config as cfg


class TestRTGYMConfig:
    def test_reset_act_buf_set_true(self):
        rtgym_cfg = cfg.ENV_CONFIG.get("RTGYM_CONFIG", {})
        assert rtgym_cfg.get("reset_act_buf") is True, (
            "reset_act_buf must default to True to clear stale pre-reset actions"
        )

    def test_act_buf_len_positive(self):
        rtgym_cfg = cfg.ENV_CONFIG.get("RTGYM_CONFIG", {})
        abl = rtgym_cfg.get("act_buf_len", 0)
        assert abl > 0, f"act_buf_len must be > 0 for RT-MDP, got {abl}"

    def test_act_buf_len_is_integer(self):
        rtgym_cfg = cfg.ENV_CONFIG.get("RTGYM_CONFIG", {})
        abl = rtgym_cfg.get("act_buf_len")
        assert isinstance(abl, int), f"act_buf_len should be int, got {type(abl)}"


class TestUpdateModelInterval:
    def test_update_interval_exists(self):
        model_cfg = cfg.TMRL_CONFIG.get("MODEL", {})
        assert "UPDATE_MODEL_INTERVAL" in model_cfg

    def test_update_interval_positive(self):
        model_cfg = cfg.TMRL_CONFIG.get("MODEL", {})
        val = model_cfg["UPDATE_MODEL_INTERVAL"]
        assert val > 0, f"UPDATE_MODEL_INTERVAL must be > 0, got {val}"
