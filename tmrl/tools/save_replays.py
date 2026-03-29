from dataclasses import dataclass

import numpy as np
import tyro

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial


def save_replays(nb_replays=np.inf):
    config = cfg_obj.CONFIG_DICT
    config["interface_kwargs"] = {"save_replays": True}
    rw = RolloutWorker(
        env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
        actor_module_cls=partial(cfg_obj.POLICY),
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        device="cuda" if cfg.CUDA_INFERENCE else "cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        crc_debug=cfg.CRC_DEBUG,
        standalone=True,
    )

    rw.run_episodes(10000, nb_episodes=nb_replays)


@dataclass
class SaveReplaysCLI:
    """Record TrackMania replays."""

    nb_replays: int = -1
    """Number of replays to record; use -1 for unlimited."""


def main() -> None:
    args = tyro.cli(SaveReplaysCLI)
    n = np.inf if args.nb_replays < 0 else args.nb_replays
    save_replays(n)


if __name__ == "__main__":
    main()
