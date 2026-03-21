"""Save TrackMania replays by running a standalone rollout worker with save_replays enabled."""

from dataclasses import dataclass

import numpy as np
import tyro

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial


@dataclass
class SaveReplaysCli:
    """CLI for saving a fixed number of replays."""

    nb_replays: int = 0
    """Number of replays to record (0 = unlimited)."""


def save_replays(nb_replays: float = np.inf) -> None:
    """Run a standalone worker that saves TrackMania replays.

    Args:
        nb_replays: Maximum number of replays to save (default: no limit).
    """
    env_config = cfg_obj.CONFIG_DICT.copy()
    env_config["interface_kwargs"] = {"save_replays": True}
    rollout_worker = RolloutWorker(
        env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": env_config}),
        actor_module_cls=partial(cfg_obj.POLICY),
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        device="cuda" if cfg.CUDA_INFERENCE else "cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        crc_debug=cfg.CRC_DEBUG,
        standalone=True,
    )
    limit = int(nb_replays) if nb_replays else np.inf
    rollout_worker.run_episodes(10000, nb_episodes=limit)


if __name__ == "__main__":
    cli = tyro.cli(SaveReplaysCli)
    save_replays(np.inf if cli.nb_replays == 0 else float(cli.nb_replays))
