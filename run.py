from argparse import ArgumentParser, ArgumentTypeError
from agents import TrainingOffline, run_wandb_tm, run_tm
from agents.util import partial
from agents.sac import Agent
from agents.envs import UntouchedGymEnv
from agents.networking import RedisServer, RolloutWorker, TrainerInterface
import agents.custom.config as cfg
import time


def main(args):
    if args.server:
        RedisServer(samples_per_redis_batch=1000, localhost=cfg.LOCALHOST)
    elif args.worker or args.test:
        rw = RolloutWorker(env_id="gym_real_time:gym-rt-v0",
                           env_config=cfg.CONFIG_DICT,
                           actor_module_cls=partial(cfg.POLICY, act_in_obs=cfg.ACT_IN_OBS),
                           get_local_buffer_sample=cfg.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.PRAGMA_CUDA else 'cpu',
                           redis_ip=cfg.REDIS_IP,
                           samples_per_worker_batch=1000,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg.OBS_PREPROCESSOR)
        if args.worker:
            rw.run()
        else:
            rw.run_test_episode(1000)
    elif args.trainer:
        main_train(args)
    else:
        raise ArgumentTypeError('Enter a valid argument')
    while True:
        time.sleep(1.0)


def main_train(args):
    sac_tm = partial(
        TrainingOffline,
        Env=partial(UntouchedGymEnv, id="gym_real_time:gym-rt-v0", gym_kwargs={"config": cfg.CONFIG_DICT}),
        epochs=400,  # 10
        rounds=10,  # 50
        steps=1000,  # 2000
        update_model_interval=1000,
        update_buffer_interval=1000,
        max_training_steps_per_env_step=1.0,
        Agent=partial(Agent,
                      OutputNorm=partial(beta=0., zero_debias=False),
                      Memory=cfg.MEMORY,
                      device='cuda' if cfg.PRAGMA_CUDA else 'cpu',
                      Model=partial(cfg.TRAIN_MODEL, act_in_obs=cfg.ACT_IN_OBS),
                      memory_size=1000000,
                      batchsize=128,  # default: 256
                      lr=0.0003,  # default 0.0003
                      discount=0.995,  # default and best tmnf so far: 0.99
                      target_update=0.005,
                      reward_scale=1.0,  # default: 5.0, best tmnf so far: 0.1
                      entropy_scale=1.0),  # default: 1.0
    )

    print("--- NOW RUNNING: SAC trackmania ---")
    interface = TrainerInterface(redis_ip=cfg.REDIS_IP, model_path=cfg.MODEL_PATH_TRAINER)
    if not args.no_wandb:
        run_wandb_tm(entity=cfg.WANDB_ENTITY,
                     project=cfg.WANDB_PROJECT,
                     run_id=cfg.WANDB_RUN_ID,
                     interface=interface,
                     run_cls=sac_tm,
                     checkpoint_path=cfg.CHECKPOINT_PATH)
    else:
        run_tm(interface=interface,
               run_cls=sac_tm,
               checkpoint_path=cfg.CHECKPOINT_PATH)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--trainer', action='store_true')
    parser.add_argument('--worker', action='store_true')  # not used
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no-wandb', dest='no_wandb', action='store_true', help='if you do not want to log results on Weights and Biases, use this option')
    args = parser.parse_args()
    print(args)
    main(args)
