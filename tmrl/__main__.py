"""TMRL entrypoint: server, trainer, rollout worker, and utilities.

Use tyro CLI: e.g. python -m tmrl --server, python -m tmrl --trainer
(wandb on by default; use --no-wandb to disable).
"""

import json
import signal
import sys
import time
from dataclasses import dataclass

import tyro
from loguru import logger

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker, Server, Trainer
from tmrl.tools.check_environment import (
    check_env_tm20_trackmap,
    check_env_tm20full,
    check_env_tm20lidar,
)
from tmrl.tools.import_player_runs import import_player_runs
from tmrl.tools.record_reward import record_reward_dist
from tmrl.util import partial


@dataclass
class TmrlCli:
    """Command-line interface for TMRL.

    Exactly one of the mode flags (install, server, trainer, worker, etc.)
    should be used per run. Optional modifiers (e.g. --wandb, --config)
    apply to the chosen mode.
    """

    install: bool = False
    """Check TMRL installation (prints TmrlData path)."""

    server: bool = False
    """Run the central server (buffer + model distribution)."""

    trainer: bool = False
    """Run the trainer (learn from replay, push model to server)."""

    worker: bool = False
    """Run a rollout worker (collect experience, pull model)."""

    expert: bool = False
    """Run an expert rollout worker (no model updates)."""

    test: bool = False
    """Run inference only (no training, standalone episodes)."""

    benchmark: bool = False
    """Benchmark the environment (no training)."""

    record_reward: bool = False
    """Record a reward function in TrackMania 2020."""

    use_keyboard: bool = False
    """Use keyboard (instead of gamepad) when recording reward."""

    record_episode: bool = False
    """Record an episode into the replay buffer."""

    record_episode_count: int = 1
    """Number of episodes to record when using --record-episode."""

    record_episode_output_dir: str = ""
    """Output folder for player-run files (default: ~/TmrlData/player_runs)."""

    record_episode_max_samples: int = 0
    """Max samples per recorded episode (0 uses config default)."""

    import_player_runs: bool = False
    """Import recorded player-run files into dataset data.pkl."""

    player_runs_paths: str = ""
    """Comma-separated list of player-run .pkl files to import."""

    player_runs_overwrite: bool = False
    """Overwrite dataset instead of appending when importing player runs."""

    player_runs_max_samples: int = 0
    """Max raw samples to keep after import (0 keeps all)."""

    player_runs_dry_run: bool = False
    """Validate import only, without writing dataset."""

    check_env: bool = False
    """Verify environment (Lidar/Full/TrackMap) works."""

    wandb: bool = True
    """Enable Weights & Biases logging on trainer (default True; use --no-wandb to disable)."""

    config: str = "{}"
    """JSON dict of rtgym config overrides, e.g. -d '{\"time_step_duration\": 0.1}'."""

    wsl_ip: bool = False
    """Print this machine's IP (for PUBLIC_IP_SERVER when worker runs on Windows)."""


def main(cli: TmrlCli) -> None:
    """Dispatch to the selected mode (server, trainer, worker, or utility)."""
    if cli.server:
        server = Server()
        shutdown = [False]  # use list so closure can mutate

        def _handle_int(_signum, _frame):
            shutdown[0] = True

        try:
            signal.signal(signal.SIGINT, _handle_int)
            signal.signal(signal.SIGTERM, _handle_int)
        except ValueError:
            pass  # signal only works in main thread
        try:
            while not shutdown[0]:
                time.sleep(0.5)
        except KeyboardInterrupt:
            shutdown[0] = True  # Ctrl+C can raise this instead of calling the handler
        if shutdown[0]:
            logger.info("Shutting down server (Ctrl+C or SIGTERM).")
            server.stop()
            sys.exit(0)
    elif cli.worker or cli.test or cli.benchmark or cli.expert:
        env_config = cfg_obj.CONFIG_DICT.copy()
        try:
            config_overrides = json.loads(cli.config) if cli.config else {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid --config JSON: {e}")
            raise
        for key, value in config_overrides.items():
            env_config[key] = value
        # Worker episode cap: use env's ep_max_length when set so increasing it stops early resets
        _max_samp = env_config.get("ep_max_length")
        if _max_samp is None:
            _max_samp = cfg.RW_MAX_SAMPLES_PER_EPISODE
        logger.info(" Worker: interface={}", cfg_obj.INTERFACE_DISPLAY_NAME)
        rollout_worker = RolloutWorker(
            env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": env_config}),
            actor_module_cls=cfg_obj.POLICY,
            sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
            device="cuda" if cfg.CUDA_INFERENCE else "cpu",
            server_ip=cfg.SERVER_IP_FOR_WORKER,
            max_samples_per_episode=_max_samp,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
            crc_debug=cfg.CRC_DEBUG,
            standalone=cli.test,
        )
        if cli.worker:
            rollout_worker.run(
                test_episode_interval=cfg.RW_TEST_EPISODE_INTERVAL,
                verbose=True,
            )
        elif cli.expert:
            rollout_worker.run(expert=True)
        elif cli.benchmark:
            rollout_worker.run_env_benchmark(nb_steps=1000, test=False)
        else:
            rollout_worker.run_episodes(10000)
    elif cli.trainer:
        logger.info(
            " Trainer: interface={} (env_cls from same config)",
            cfg_obj.INTERFACE_DISPLAY_NAME,
        )
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
        if cli.wandb:
            trainer.run_with_wandb(
                entity=cfg.WANDB_ENTITY,
                project=cfg.WANDB_PROJECT,
                run_id=cfg.WANDB_RUN_ID,
            )
        else:
            trainer.run()
    elif cli.record_reward:
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=cli.use_keyboard)
    elif cli.check_env:
        if cfg.PRAGMA_LIDAR:
            if cfg.PRAGMA_TRACKMAP:
                check_env_tm20_trackmap()
            else:
                check_env_tm20lidar()
        else:
            check_env_tm20full()
    elif cli.record_episode:
        from tmrl.tools.record_episode import record_episode

        record_episode(
            nb_episodes=cli.record_episode_count,
            output_dir=cli.record_episode_output_dir or None,
            max_samples_per_episode=(
                None if cli.record_episode_max_samples <= 0 else cli.record_episode_max_samples
            ),
        )
    elif cli.import_player_runs:
        if not cli.player_runs_paths:
            raise ValueError("--player-runs-paths is required with --import-player-runs")
        import_player_runs(
            paths=[p.strip() for p in cli.player_runs_paths.split(",") if p.strip()],
            overwrite=cli.player_runs_overwrite,
            max_samples=None if cli.player_runs_max_samples <= 0 else cli.player_runs_max_samples,
            dry_run=cli.player_runs_dry_run,
        )
    elif cli.install:
        logger.info(f"TMRL folder: {cfg.TMRL_FOLDER}")
    elif cli.wsl_ip:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        print(ip)
        logger.info(f"Set PUBLIC_IP_SERVER to this in Windows TmrlData\\config\\config.json: {ip}")
    else:
        raise ValueError(
            "No mode selected."
            " Use one of: "
            " --install, "
            " --server, "
            " --trainer, "
            " --worker, "
            " --test, "
            " --benchmark, "
            " --expert, "
            " --record-reward, "
            " --record-episode, "
            " --import-player-runs, "
            " --check-environment, "
            " --wsl-ip."
        )


if __name__ == "__main__":
    cli_args = tyro.cli(TmrlCli)
    main(cli_args)
