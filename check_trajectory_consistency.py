from agents.networking import *
from agents import TrainingOffline, run_wandb_tm
from agents.util import partial
from agents.sac import Agent
from agents.envs import UntouchedGymEnv

rs = RedisServer(samples_per_redis_batch=5, localhost=LOCALHOST)

time.sleep(3.0)
print("STOPPED SLEEPING")

interface = TrainerInterface(redis_ip=REDIS_IP, model_path=MODEL_PATH_TRAINER)

time.sleep(3.0)
print("STOPPED SLEEPING")

rw = RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                   actor_module_cls=partial(POLICY, act_in_obs=ACT_IN_OBS),
                   device='cuda' if PRAGMA_CUDA else 'cpu',
                   redis_ip=REDIS_IP,
                   samples_per_worker_batch=5,
                   model_path=MODEL_PATH_WORKER,
                   obs_preprocessor=OBS_PREPROCESSOR)

time.sleep(3.0)
print("STOPPED SLEEPING")

print("INFO: collecting samples")
traj = rw.collect_n_steps_and_debug_trajectory(5, train=True)
print("INFO: copying buffer for sending")
rw.send_and_clear_buffer()

print("trajectory:")
for i, t in enumerate(traj):
    print(f"--- {i} ---:")
    for j in t:
        print(j)

# print("INFO: checking for new weights")
# rw.update_actor_weights()

time.sleep(3.0)
print("STOPPED SLEEPING")

Sac_tm = partial(
    TrainingOffline,
    Env=partial(UntouchedGymEnv,
                id="gym_tmrl:gym-tmrl-v0",
                gym_kwargs={"config": CONFIG_DICT}),
    epochs=1,
    rounds=1,
    steps=1,
    update_model_interval=1,
    update_buffer_interval=1,
    max_training_steps_per_env_step=1.0,
    Agent=partial(Agent,
                  OutputNorm=partial(beta=0., zero_debias=False),
                  Memory=MEMORY,
                  device='cuda' if PRAGMA_CUDA else 'cpu',
                  Model=partial(TRAIN_MODEL,
                                act_in_obs=ACT_IN_OBS),
                  memory_size=500000,
                  batchsize=1,
                  lr=0.0003,  # default 0.0003
                  discount=0.99,
                  target_update=0.005,
                  reward_scale=0.5,  # default: 5.0
                  entropy_scale=1.0),  # default: 1.0
)

print("--- NOW RUNNING: SAC trackmania ---")

run_instance = Sac_tm()
# run_instance.run_epoch(interface=interface)
run_instance.check_ratio(interface)
# run_instance.agent.train()

for i in range(3):
    print(f"--- sample {i} ---")
    obs, act, rew, next_obs, terminals = run_instance.agent.memory.sample()
    print(f"obs: {obs}")
    print(f"act: {act}")
    print(f"rew: {rew}")
    print(f"next_obs: {next_obs}")
    print(f"terminals: {terminals}")

# run_wandb_tm(None,
#              None,
#              None,
#              interface,
#              run_cls=Sac_tm,
#              checkpoint_path=CHECKPOINT_PATH)
