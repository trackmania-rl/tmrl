from agents.envs import UntouchedGymEnv
from agents.util import load
from agents.sac_models import *
from threading import Lock
from copy import deepcopy
from agents.util import collate, partition

from collections import deque


# BUFFER: ===========================================

class Buffer:
    def __init__(self, maxlen=100000):
        self.memory = deque(maxlen=maxlen)
        self.lock = Lock()

    def flush(self):
        """
        empties the buffer
        """
        self.lock.acquire()
        self.memory.clear()
        self.lock.release()

    def append(self, obs, rew, done, info):
        self.lock.acquire()
        self.memory.append((obs, rew, done, info, ))
        self.lock.release()


# TRAINER: ==========================================

class TrainerInterface:
    """
    This is the trainer's network interface
    Any RolloutWorker can connect to this interface and add itself to the list of workers
    There is one listening thread per connected RolloutWorker
    When a RolloutWorker sends a new batch of experiences, it gets buffered in self.buffer
    """
    def __init__(self):
        self.buffer_lock = Lock()
        self.buffer = Buffer()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule (sac_models.py)
        broadcasts the model's weights to all connected RolloutWorkers
        """
        torch.save(model.state_dict(), r'C:/Users/Yann/Desktop/git/tmrl/checkpoint/weights/exp.pth')

    def retrieve_buffer(self):
        """
        updates the replay buffer with the local buffer
        empties the local buffer
        """
        pass


# ROLLOUT WORKER: ===================================

class RolloutWorker:
    def __init__(self,
                 env_id,
                 actor_module_cls,
                 obs_space,
                 act_space,
                 device="cpu"):
        self.env = UntouchedGymEnv(id=env_id)
        self.actor = actor_module_cls(obs_space, act_space)
        self.device = device
        self.buffer = Buffer()

    def act(self, obs, train=False):
        """
        converts inputs to torch tensors and converts outputs to numpy arrays
        """
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action_distribution = self.actor(obs)
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        action, = partition(action)
        return action

    def collect_n_steps(self, n, train=True):
        """
        empties the local buffer and collects n transitions
        set train to False for test samples, True for train samples
        """
        self.buffer.flush()
        obs = self.env.reset()
        print(f"DEBUG: init obs[0]:{obs[0]}")
        print(f"DEBUG: init obs[1][-1].shape:{obs[1][-1].shape}")
        obs_mod = (obs[0], obs[1][-1], )  # speed and most recent image
        self.buffer.append(obs_mod, 0.0, False, {})
        for _ in range(n):
            act = self.act(obs, train)
            obs, rew, done, info = self.env.step(act)
            obs_mod = (obs[0], obs[1][-1],)  # speed and most recent image
            self.buffer.append(obs_mod, rew, done, info)

    def update_actor_weights(self):
        """
        updates the model with new weights from the trainer when available
        """
        wpath = r"C:/Users/Yann/Desktop/git/tmrl/checkpoint/weights/exp.pth"
        self.actor.load_state_dict(torch.load(wpath))


if __name__ == "__main__":
    speed = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
    img = gym.spaces.Box(low=0.0, high=1.0, shape=(4, 3, 50, 190))
    observation_space = gym.spaces.Tuple((speed, img))
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))
    rw = RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                       actor_module_cls=TMPolicy,
                       obs_space=observation_space,
                       act_space=action_space,
                       device="cpu")
    rw.update_actor_weights()
    rw.collect_n_steps(100, train=True)
