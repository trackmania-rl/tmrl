from tmrl.config import *
from agents.tm import *

def main(args):
    worker = args.worker
    redis = args.redis
    trainer = args.trainer

    if redis:
        rs = RedisServer(samples_per_redis_batch=1000, localhost=LOCALHOST)
    elif worker:
        rw = RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                           actor_module_cls=partial(POLICY, act_in_obs=ACT_IN_OBS),
                           device='cuda' if PRAGMA_CUDA else 'cpu',
                           redis_ip=REDIS_IP,
                           samples_per_worker_batch=1000,
                           model_path=MODEL_PATH_WORKER,
                           obs_preprocessor=OBS_PREPROCESSOR)
        rw.run()
    else:
        main_train()
    while True:
        time.sleep(1.0)


def main_train():
    from agents import TrainingOffline, run_wandb_tm
    from agents.util import partial
    from agents.sac import Agent
    from agents.envs import UntouchedGymEnv

    Sac_tm = partial(
        TrainingOffline,
        Env=partial(UntouchedGymEnv,
                    id="gym_tmrl:gym-tmrl-v0",
                    gym_kwargs={"config": CONFIG_DICT}),
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
                      Model=partial(TRAIN_MODEL,
                                    act_in_obs=ACT_IN_OBS),
                      memory_size=1000000,
                      batchsize=2,  # 64
                      lr=0.0003,  # default 0.0003
                      discount=0.99,
                      target_update=0.005,
                      reward_scale=0.5,  # default: 5.0
                      entropy_scale=1.0),  # default: 1.0
    )

    print("--- NOW RUNNING: SAC trackmania ---")
    interface = TrainerInterface(redis_ip=REDIS_IP, model_path=MODEL_PATH_TRAINER)
    run_wandb_tm(entity=WANDB_ENTITY,
                 project=WANDB_PROJECT,
                 run_id=WANDB_RUN_ID,
                 interface=interface,
                 run_cls=Sac_tm,
                 checkpoint_path=CHECKPOINT_PATH)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--redis', dest='redis', action='store_true')
    parser.set_defaults(redis=False)
    parser.add_argument('--trainer', dest='trainer', action='store_true')
    parser.set_defaults(trainer=False)
    parser.add_argument('--worker', dest='worker', action='store_true')  # not used
    parser.set_defaults(worker=False)
    args = parser.parse_args()
    main(args)
