import numpy as np
import pickle
import multiprocessing as mp
import time
import torch
from tmrl.util import partial

USE_MP = False


def get_env_state(env):
    dtypes = (int, np.ndarray, np.random.RandomState)
    state = dict()
    state.update({k: v for k, v in env.__dict__.items() if isinstance(v, dtypes)})
    if hasattr(env, 'env'):
        state.update(env=get_env_state(env.env))
    elif hasattr(env, 'sim'):
        state.update(sim_state=env.sim.get_state())
    return state


def set_env_state(env, state):
    if 'env' in state:
        set_env_state(env.env, state.pop('env'))
    if 'sim_state' in state:
        env.sim.set_state(state.pop('sim_state'))
    env.__dict__.update(state)


def _step(args):
    env, action = args
    return env.step(action)


def _step_nd(args):
    envs, actions = args
    return list(map(_step, zip(envs, actions)))


def _set_envs_from_pickle(envs):
    _envs = [pickle.loads(e) for e in envs]


class BatchEnv:
    def __init__(self, Env, batch_size=128, num_avg=32):
        self.num_avg = num_avg
        self.envs = [[Env() for _ in range(batch_size)] for _ in range(num_avg)]

    def init_from_pickle(self, states):
        for envs in self.envs:
            self._init_from_state_dict(envs, map(pickle.loads, states))

    def _init_from_state_dict(self, envs, states):
        all(map(lambda args: set_env_state(*args), zip(envs, states)))

    def step(self, actions):
        if len(actions) < len(self.envs):
            # first action
            actions = actions.repeat(self.num_avg, 0)
        if USE_MP:
            if not hasattr(self, 'pool'):
                self.pool = mp.Pool()
            res = self.pool.map(_step_nd, zip(self.envs, actions))
        else:
            res = list(map(_step_nd, zip(self.envs, actions)))

        obss, rewards, dones, info = [], [], [], []
        for res_i in res:
            obss_i, rewards_i, dones_i, info_i = [], [], [], []
            for r in res_i:
                obss_i.append(r[0])
                rewards_i.append(r[1])
                dones_i.append(r[2])
                info_i.append(r[3])
            obss.append(obss_i)
            rewards.append(rewards_i)
            dones.append(dones_i)
            info_i.append(info_i)
        return obss, rewards, dones, info


if __name__ == '__main__':

    bs = 32
    num_avg = 8
    num_steps = 20

    from tmrl.envs import GymEnv

    e = GymEnv(id='Ant-v2')
    e.reset()

    s = get_env_state(e)
    e_pck = [pickle.dumps(s)] * bs  # serialized envs


    # dummy policy
    def policy():
        return np.array([[e.action_space.sample() for _ in range(bs)] for _ in range(num_avg)])


    be = BatchEnv(partial(GymEnv, id='Ant-v2'), batch_size=bs, num_avg=num_avg)

    t = time.time()

    be.init_from_pickle(e_pck)

    actions = policy()
    # action dim is num_avg x bs x action
    print('actions', actions.shape)
    obss, rewards, dones, _ = be.step(actions)
    # obs dim is num_avg x bs x obs

    for _ in range(num_steps):
        actions = policy()
        print('actions', actions.shape)
        be.step(actions)

    print(time.time() - t)
