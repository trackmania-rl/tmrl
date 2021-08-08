# standard library imports
import time
from argparse import ArgumentParser, ArgumentTypeError
import logging

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl import run_tm, run_wandb_tm
from tmrl.envs import UntouchedGymEnv
from tmrl.networking import RolloutWorker, Server, TrainerInterface
from tmrl.util import partial


def main(args):
    if args.server:
        Server(samples_per_server_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES)
    elif args.worker or args.test or args.benchmark:
        rw = RolloutWorker(env_cls=partial(UntouchedGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": cfg_obj.CONFIG_DICT}),
                           actor_module_cls=partial(cfg_obj.POLICY, act_buf_len=cfg.ACT_BUF_LEN),
                           get_local_buffer_sample=cfg_obj.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=args.test)
        if args.worker:
            rw.run()
        elif args.benchmark:
            rw.run_env_benchmark(nb_steps=1000, deterministic=False)
        else:
            rw.run_episodes(10000)

    elif args.trainer:
        main_train(args)
    else:
        raise ArgumentTypeError('Enter a valid argument')
    while True:
        time.sleep(1.0)


def main_train(args):

    train_cls = cfg_obj.TRAINER

    logging.info(f"--- NOW RUNNING: SAC/DCAC trackmania ---")
    interface = TrainerInterface(server_ip=cfg.SERVER_IP_FOR_TRAINER, model_path=cfg.MODEL_PATH_TRAINER)
    if not args.no_wandb:
        run_wandb_tm(entity=cfg.WANDB_ENTITY,
                     project=cfg.WANDB_PROJECT,
                     run_id=cfg.WANDB_RUN_ID,
                     interface=interface,
                     run_cls=train_cls,
                     checkpoint_path=cfg.CHECKPOINT_PATH,
                     dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
                     load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN)
    else:
        run_tm(interface=interface, run_cls=train_cls, checkpoint_path=cfg.CHECKPOINT_PATH, dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN, load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--trainer', action='store_true')
    parser.add_argument('--worker', action='store_true')  # not used
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--no-wandb', dest='no_wandb', action='store_true', help='if you do not want to log results on Weights and Biases, use this option')
    args = parser.parse_args()
    logging.info(args)

    main(args)
