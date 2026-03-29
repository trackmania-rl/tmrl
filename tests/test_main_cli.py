import json
from unittest.mock import MagicMock, patch

import pytest
import tmrl.config.config_objects as cfg_obj
from tmrl.__main__ import TmrlCLI, main


def test_main_rejects_invalid_config_json():
    with pytest.raises(SystemExit, match="Invalid JSON"):
        main(TmrlCLI(config="not json"))


def test_main_rejects_non_object_config():
    with pytest.raises(SystemExit, match="JSON object"):
        main(TmrlCLI(config=json.dumps([1, 2, 3])))


def test_worker_path_uses_config_copy_not_singleton():
    sentinel = "__tmrl_cli_config_isolation__"
    assert sentinel not in cfg_obj.CONFIG_DICT
    with patch("tmrl.__main__.RolloutWorker") as RW:
        RW.return_value = MagicMock()
        main(TmrlCLI(worker=True, config=json.dumps({sentinel: 42})))
    env_cls = RW.call_args.kwargs["env_cls"]
    passed_cfg = env_cls.keywords["gym_kwargs"]["config"]
    assert passed_cfg is not cfg_obj.CONFIG_DICT
    assert passed_cfg[sentinel] == 42
    assert sentinel not in cfg_obj.CONFIG_DICT
