from argparse import ArgumentParser, ArgumentTypeError
from agents.networking import *
from agents import TrainingOffline, run_wandb_tm, run_tm
from agents.util import partial
from agents.sac import Agent
from agents.envs import UntouchedGymEnv

def main(args):

    if args.server:
        RedisServer(samples_per_redis_batch=1000, localhost=LOCALHOST)
    elif args.worker:
        RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                      actor_module_cls=partial(POLICY, act_in_obs=ACT_IN_OBS),
                      device='cuda' if PRAGMA_CUDA else 'cpu',
                      redis_ip=REDIS_IP,
                      samples_per_worker_batch=1000,
                      model_path=MODEL_PATH_WORKER,
                      obs_preprocessor=OBS_PREPROCESSOR).run()
    elif args.trainer:
        main_train(args)
    elif args.test:
        RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                      actor_module_cls=partial(POLICY, act_in_obs=ACT_IN_OBS),
                      device='cuda' if PRAGMA_CUDA else 'cpu',
                      redis_ip=REDIS_IP,
                      samples_per_worker_batch=1000,
                      model_path=MODEL_PATH_WORKER,
                      obs_preprocessor=OBS_PREPROCESSOR).run_test_episode(1000)
    else:
        raise ArgumentTypeError('Enter a valid argument')
    while True:
        time.sleep(3.0)


def main_train(args):
    Sac_tm = partial(
        TrainingOffline,
        Env=partial(UntouchedGymEnv, id="gym_tmrl:gym-tmrl-v0", gym_kwargs={"config": CONFIG_DICT}),
        epochs=400,  # 10
        rounds=10,  # 50
        steps=1000,  # 2000
        update_model_interval=1000,
        update_buffer_interval=1000,
        max_training_steps_per_env_step=1.0,
        Agent=partial(Agent,
                      OutputNorm=partial(beta=0., zero_debias=False),
                      Memory=MEMORY,
                      device='cuda' if PRAGMA_CUDA else 'cpu',
                      Model=partial(TRAIN_MODEL, act_in_obs=ACT_IN_OBS),
                      memory_size=1000000,
                      batchsize=128,  # 64
                      lr=0.0003,  # default 0.0003
                      discount=0.99,
                      target_update=0.005,
                      reward_scale=0.5,  # default: 5.0
                      entropy_scale=1.0),  # default: 1.0
    )

    print("--- NOW RUNNING: SAC trackmania ---")
    interface = TrainerInterface(redis_ip=REDIS_IP, model_path=MODEL_PATH_TRAINER)
    if args.wandb:
        run_wandb_tm(entity=WANDB_ENTITY,
                     project=WANDB_PROJECT,
                     run_id=WANDB_RUN_ID,
                     interface=interface,
                     run_cls=Sac_tm,
                     checkpoint_path=CHECKPOINT_PATH)
    else:
        run_tm(interface=interface,
               run_cls=Sac_tm,
               checkpoint_path=CHECKPOINT_PATH)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--trainer', action='store_true')
    parser.add_argument('--worker', action='store_true')  # not used
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb', action='store_false', help='if you do not want to log results on Weights and Biases, set this to False')
    args = parser.parse_args()
    print(args)
    main(args)
