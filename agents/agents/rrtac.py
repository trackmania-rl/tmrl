from copy import deepcopy
from dataclasses import dataclass
from functools import reduce

import torch
from agents.envs import AvenueEnv
from torch.nn.functional import mse_loss
import numpy as np

import agents.sac
from agents.memory import Memory, TrajMemory
from agents.nn import no_grad, exponential_moving_average, PopArt, detach
from agents.util import partial
from agents.rrtac_models import LstmModel, LstmDouble


@dataclass(eq=0)
class Agent(agents.sac.Agent):
    Model: type = LstmModel
    loss_alpha: float = 0.05
    history_length: int = 8
    training_steps: float = 1 / 8  # training steps per environment interaction step

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = TrajMemory(self.memory_size, self.batchsize, device, self.history_length)

        self.is_training = False

    def act(self, state, obs, r, done, info, train=False):
        stats = []
        state = self.model.reset() if state is None else state  # initialize state if necessary
        action, next_state, _ = self.model.act(state, obs, r, done, info, train)

        if train:
            self.memory.append(np.float32(r), np.float32(done), info, obs, state, action)
            self.environment_steps += 1

            total_updates_target = (self.environment_steps - self.start_training) * self.training_steps
            while self.total_updates < int(total_updates_target):
                if self.total_updates == 0:
                    print("starting training")
                stats += self.train(),
                self.total_updates += 1
        return action, next_state, stats

    def train(self):
        obs, state, a, r, terminals = self.memory.sample()
        terminals = terminals[:, None]  # expand for correct broadcasting below

        # torch.autograd.set_detect_anomaly(True)

        state_i = state[0]  # we recompute all the following memory states
        loss_actor = 0
        loss_critic = 0
        value_targets = []
        for i in range(self.history_length):
            new_action_distribution, state_i, _, critic_logits = self.model(state_i, obs[i])
            new_actions = new_action_distribution.rsample()
            new_actions_log_prob = new_action_distribution.log_prob(new_actions)[:, None]

            # critic loss
            _, _, next_value_target, _ = self.model_target(detach(state_i), (obs[i + 1][0], new_actions.detach()))
            next_value_target = reduce(torch.min, next_value_target)

            value_target = self.discount * self.outputnorm_target.unnormalize(next_value_target)
            value_target += self.reward_scale * r[i][:, None]
            value_target -= self.entropy_scale * new_actions_log_prob.detach()
            value_targets += value_target,
            value_target = self.outputnorm.normalize(value_target)
            values = tuple(c(h) for c, h in zip(self.model.critic_output_layers, critic_logits))  # recompute values (weights changed)

            assert values[0].shape == value_target.shape and not value_target.requires_grad
            loss_critic += sum(mse_loss(v, value_target) for v in values) / self.history_length

            # actor loss
            _, _, next_value, _ = self.model_nograd(state_i, (obs[i + 1][0], new_actions))
            next_value = reduce(torch.min, next_value)
            loss_actor_i = - (1. - terminals) * self.discount * self.outputnorm.unnormalize(next_value)
            loss_actor_i += self.entropy_scale * new_actions_log_prob
            assert loss_actor_i.shape == (self.batchsize, 1)
            loss_actor += self.outputnorm.normalize(loss_actor_i).mean() / self.history_length

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model and normalizers
        self.outputnorm.update(torch.cat(value_targets))
        exponential_moving_average(self.model_target.parameters(), self.model.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            outputnorm_mean=float(self.outputnorm.mean),
            outputnorm_std=float(self.outputnorm.std),
            memory_size=len(self.memory),
            # entropy_scale=self.entropy_scale
        )


# AvenueAgent = partial(
#   Agent,
#   entropy_scale=0.05,
#   lr=0.0002,
#   memory_size=500000,
#   batchsize=100,
#   training_interval=4,
#   start_training=10000,
#   Model=partial(ConvDouble)
# )


if __name__ == "__main__":
    from agents import Training, run

    RrtacTest = partial(
        Training,
        epochs=5,
        rounds=10,
        steps=200,
        Agent=partial(Agent, Model=partial(LstmDouble, hidden_units=32), device='cpu', memory_size=1000000, start_training=200, batchsize=8),
        # Agent=partial(Agent, Model=partial(LstmModel, hidden_units=32), device='cuda', memory_size=1000000, start_training=200, batchsize=8),
        Env=partial(id="Pendulum-v0", real_time=True),
        # Env=partial(id="HalfCheetah-v2", real_time=True),
    )

    # Rtac_Avenue_Test = partial(
    #   Training,
    #   epochs=3,
    #   rounds=5,
    #   steps=300,
    #   Agent=partial(AvenueAgent, device='cpu', start_training=256, batchsize=4, Model=rtac_models.ConvSeparate),
    #   Env=partial(AvenueEnv, real_time=True),
    #   Test=partial(number=0),  # laptop can't handle more than that
    # )

    run(RrtacTest)
# run(Rtac_Avenue_Test)
