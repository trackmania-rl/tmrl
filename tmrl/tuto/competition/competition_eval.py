"""
Evaluate your submission to the TMRL competition.
Assumes the ActorModule script is in the same folder and named "custom_actor_module.py",
and that your ActorModule class is named "MyActorModule".
Set "SLEEP_TIME_AT_RESET" to 0.0 in config.json when using this script.
"""

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.tuto.competition.custom_actor_module import (
    MyActorModule,
)  # change this to match your ActorModule name
from tmrl.util import partial

# rtgym environment class (full TrackMania2020 Gymnasium environment with replays enabled):
config = cfg_obj.CONFIG_DICT
config["interface_kwargs"] = {"save_replays": True}
env_cls = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config})

# Device for inference (competition evaluation is on CPU)
device_worker = "cpu"

try:
    from tmrl.tuto.competition.custom_actor_module import obs_preprocessor
except Exception:
    obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


if __name__ == "__main__":
    rw = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=MyActorModule,
        device=device_worker,
        obs_preprocessor=obs_preprocessor,
        standalone=True,
    )
    rw.run_episodes()
