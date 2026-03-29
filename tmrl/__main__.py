import copy
import json
import time
from dataclasses import dataclass
from typing import Annotated

import tyro
from loguru import logger

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker, Server, Trainer
from tmrl.tools.check_environment import check_env_tm20full, check_env_tm20lidar
from tmrl.tools.record import record_reward_dist
from tmrl.tools.save_replays import save_replays
from tmrl.util import partial


@dataclass
class TmrlCLI:
    """TMRL command-line interface."""

    install: bool = False
    """Checks TMRL installation."""
    server: bool = False
    """Launches the server."""
    trainer: bool = False
    """Launches the trainer."""
    worker: bool = False
    """Launches a rollout worker."""
    expert: bool = False
    """Launches an expert rollout worker (no model update)."""
    test: bool = False
    """Runs inference without training."""
    benchmark: bool = False
    """Runs a benchmark of the environment."""
    record_reward: bool = False
    """Record a reward function in TM20."""
    record_episode: bool = False
    """Record TrackMania replays (standalone worker)."""
    use_keyboard: bool = False
    """Modifier for --record-reward."""
    check_environment: bool = False
    """Check the environment."""
    wandb: bool = False
    """With --trainer: log to Weights & Biases."""
    config: Annotated[str, tyro.conf.arg(aliases=("-d",))] = "{}"
    """JSON object: rtgym environment configuration modifiers (worker/test/benchmark/expert)."""


def main(args: TmrlCLI) -> None:
    try:
        config_modifiers = json.loads(args.config) if args.config.strip() else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in --config: {e}") from e
    if not isinstance(config_modifiers, dict):
        raise SystemExit("--config must be a JSON object (e.g. '{{}}')")

    if args.server:
        _server = Server()
        while True:
            time.sleep(1.0)
    elif args.worker or args.test or args.benchmark or args.expert:
        config = copy.deepcopy(cfg_obj.CONFIG_DICT)
        for k, v in config_modifiers.items():
            config[k] = v
        rw = RolloutWorker(
            env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
            actor_module_cls=cfg_obj.POLICY,
            sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
            device="cuda" if cfg.CUDA_INFERENCE else "cpu",
            server_ip=cfg.SERVER_IP_FOR_WORKER,
            max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
            crc_debug=cfg.CRC_DEBUG,
            standalone=args.test,
        )
        if args.worker:
            rw.run()
        elif args.expert:
            rw.run(expert=True)
        elif args.benchmark:
            rw.run_env_benchmark(nb_steps=1000, test=False)
        else:
            rw.run_episodes(10000)
    elif args.trainer:
        trainer = Trainer(
            training_cls=cfg_obj.TRAINER,
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            model_path=cfg.MODEL_PATH_TRAINER,
            checkpoint_path=cfg.CHECKPOINT_PATH,
            dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
            load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,
            updater_fn=cfg_obj.UPDATER_FN,
        )
        logger.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
        if args.wandb:
            trainer.run_with_wandb(
                entity=cfg.WANDB_ENTITY,
                project=cfg.WANDB_PROJECT,
                run_id=cfg.WANDB_RUN_ID,
            )
        else:
            trainer.run()
    elif args.record_reward:
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=args.use_keyboard)
    elif args.record_episode:
        save_replays(nb_replays=args.record_episode_count)
    elif args.check_environment:
        if cfg.PRAGMA_LIDAR:
            check_env_tm20lidar()
        else:
            check_env_tm20full()
    elif args.install:
        logger.info(f"TMRL folder: {cfg.TMRL_FOLDER}")
    else:
        raise SystemExit("Enter a valid mode flag (e.g. --trainer). Use --help for options.")


if __name__ == "__main__":
    main(tyro.cli(TmrlCLI))
