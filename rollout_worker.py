from agents.envs import UntouchedGymEnv
from agents import TrainingOffline
from agents.util import load
import pickle as pkl


class RolloutWorker():
    def __init__(self, env_id, chk_path):
        self.env = UntouchedGymEnv(id=env_id)
        self.obs = self.env.reset()
        self.done = False
        with open(chk_path, 'rb') as f:
            self.cls = load(chk_path)
            self.model = self.cls.agent.model

    def step(self):
        if self.done:
            self.obs = self.env.reset()
            self.done = False
        act = self.model(self.obs)
        o, r, d, i = self.env.step(act)
        self.obs = o
        self.done = d
        return o, r, d, i

    def collect_n_steps(self, n):
        for i in range(n):
            o_s, r_s, d_s, i_s = self.step()


if __name__ == "__main__":
    rw = RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0", chk_path=r"C:/Users/Yann/Desktop/git/tmrl/cp.pkl")
    rw.collect_n_steps(100)
