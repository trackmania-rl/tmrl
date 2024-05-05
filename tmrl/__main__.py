import json
import logging
import time
from argparse import ArgumentParser, ArgumentTypeError

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import Server, Trainer, RolloutWorker
from tmrl.tools.check_environment import check_env_tm20lidar, check_env_tm20full
from tmrl.tools.record import record_reward_dist
from tmrl.util import partial


def main(args):
    if args.server:
        serv = Server()
        while True:
            time.sleep(1.0)
    elif args.worker or args.test or args.benchmark or args.expert:
        config = cfg_obj.CONFIG_DICT
        config_modifiers = args.config
        for k, v in config_modifiers.items():
            config[k] = v
        rw = RolloutWorker(env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
                           actor_module_cls=cfg_obj.POLICY,
                           sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=args.test)
        if args.worker:
            rw.run()
        elif args.expert:
            rw.run(expert=True)
        elif args.benchmark:
            rw.run_env_benchmark(nb_steps=1000, test=False)
        else:
            rw.run_episodes(10000)
    elif args.trainer:
        trainer = Trainer(training_cls=cfg_obj.TRAINER,
                          server_ip=cfg.SERVER_IP_FOR_TRAINER,
                          model_path=cfg.MODEL_PATH_TRAINER,
                          checkpoint_path=cfg.CHECKPOINT_PATH,
                          dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
                          load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,
                          updater_fn=cfg_obj.UPDATER_FN)
        logging.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
        if args.wandb:
            trainer.run_with_wandb(entity=cfg.WANDB_ENTITY,
                                   project=cfg.WANDB_PROJECT,
                                   run_id=cfg.WANDB_RUN_ID)
        else:
            trainer.run()
    elif args.record_reward:
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=args.use_keyboard)
    elif args.check_env:
        if cfg.PRAGMA_LIDAR:
            check_env_tm20lidar()
        else:
            check_env_tm20full()
    elif args.install:
        logging.info(f"TMRL folder: {cfg.TMRL_FOLDER}")
    else:
        raise ArgumentTypeError('Enter a valid argument')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--install', action='store_true', help='checks TMRL installation')
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--expert', action='store_true', help='launches an expert rollout worker (no model update)')
    parser.add_argument('--test', action='store_true', help='runs inference without training')
    parser.add_argument('--benchmark', action='store_true', help='runs a benchmark of the environment')
    parser.add_argument('--record-reward', dest='record_reward', action='store_true', help='utility to record a reward function in TM20')
    parser.add_argument('--use-keyboard', dest='use_keyboard', action='store_true', help='modifier for --record-reward')
    parser.add_argument('--check-environment', dest='check_env', action='store_true', help='utility to check the environment')
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='(use with --trainer) if you want to log results on Weights and Biases, use this option')
    parser.add_argument('-d', '--config', type=json.loads, default={}, help='dictionary containing configuration options (modifiers) for the rtgym environment')
    arguments = parser.parse_args()

    main(arguments)
